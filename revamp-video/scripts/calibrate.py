"""Stage D: joint camera calibration against the tracked tool point.

Solves, by robust nonlinear least squares over all accepted track samples:

    focal f (fx=fy), principal point (cx,cy), distortion (k1,k2),
    extrinsics (rvec,tvec)  [base frame -> camera frame, cv2 convention],
    clock offset delta      [t_video = t_log - video_start_epoch + delta],
    tool offset r           [3D, in the tracked frame's body frame]

Residuals are tracked-pixel minus projected FK point; FK poses of the tracked
frame are precomputed over the whole log once and interpolated in time, so
delta can move freely inside the optimizer. Tag corners enter as a weak prior
(the paper tag is wrinkled — never trusted hard). The physical arm deviates
from FK by up to a few mm (loose Panda tolerances), so a few-px residual
floor is expected; soft_l1 keeps stragglers from steering the solution.

Writes calib/camera.json and scratch/calib_residuals.png.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TAG_SIGMA_PX = 25.0   # weak prior: wrinkled paper
TRACK_SIGMA_PX = 4.0
OUTLIER_PX = 25.0     # drop-and-refit threshold on round-1 residuals


class FrameTrajectory:
    """Time-interpolated world pose of one URDF frame over the log."""

    def __init__(self, frame_name):
        js = common.load_joint_states()
        qs = common.joint_matrix(js)
        self.t = js["t"]
        arm = common.ArmKinematics()
        Xs = [arm.frame_pose(frame_name, q) for q in qs]
        self.p = np.array([X[:3, 3] for X in Xs])
        self.rot = Rotation.from_matrix(np.array([X[:3, :3] for X in Xs]))
        self.slerp = Slerp(self.t, self.rot)

    def at(self, t_epoch):
        t = np.clip(t_epoch, self.t[0], self.t[-1])
        p = np.stack([np.interp(t, self.t, self.p[:, i]) for i in range(3)],
                     axis=-1)
        return p, self.slerp(t)


def project(params, pts3):
    f, cx, cy, k1, k2 = params[:5]
    rvec, tvec = params[5:8], params[8:11]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    dist = np.array([k1, k2, 0, 0])
    uv, _ = cv2.projectPoints(pts3.reshape(-1, 3), rvec, tvec, K, dist)
    return uv.reshape(-1, 2)


def main():
    calib = common.CALIB_DIR
    tag = common.read_json(os.path.join(calib, "tag_init.json"))
    track = common.read_json(os.path.join(calib, "track.json"))
    # Initialize from the best silhouette-based camera (never from our own
    # previous output).
    for name in ("camera_deep.json", "camera_sil.json"):
        p = os.path.join(calib, name)
        if os.path.exists(p):
            cam0 = common.read_json(p)
            print(f"initializing from {name}")
            break
    else:
        sys.exit("run silhouette_calib.py / deep_calib.py first")
    v0 = cam0["video_start_epoch"]
    w, h = common.VIDEO_WH

    pts = track["points"]
    t_video = np.array([p["t_video"] for p in pts])
    uv_obs = np.array([[p["u"], p["v"]] for p in pts])
    print(f"{len(pts)} track points")

    traj = FrameTrajectory(track["frame"])

    # Tag prior geometry (same corner assignment tag_init selected).
    s = tag["tag_size_m"] / 2
    square = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
    tag_obj = np.roll(square, tag["rot"], axis=0) + np.array(
        tag["tag_center_base"])
    tag_uv = np.array(tag["corners_px"])

    # Parameters: f cx cy k1 k2 rvec(3) tvec(3) delta r(3)
    x0 = np.concatenate([
        [cam0["f"], cam0["cx"], cam0["cy"], cam0["k1"], cam0["k2"]],
        cam0["rvec"], cam0["tvec"], [cam0["delta_s"]], np.zeros(3)])

    def residuals(x):
        delta, r = x[11], x[12:15]
        p, rot = traj.at(t_video + v0 - delta)
        pts3 = p + rot.apply(r)
        res_track = (project(x, pts3) - uv_obs) / TRACK_SIGMA_PX
        res_tag = (project(x, tag_obj) - tag_uv) / TAG_SIGMA_PX
        return np.concatenate([res_track.ravel(), res_tag.ravel()])

    def solve(x0, mask=None):
        def fun(x):
            r = residuals(x)
            return r if mask is None else r[mask]
        return least_squares(fun, x0, loss="soft_l1", f_scale=2.0,
                             x_scale="jac", max_nfev=200)

    sol = solve(x0)
    res = residuals(sol.x).reshape(-1, 2)
    track_err = np.linalg.norm(res[:len(pts)], axis=1) * TRACK_SIGMA_PX
    print(f"round 1: rms {np.sqrt((track_err**2).mean()):.2f} px, "
          f"median {np.median(track_err):.2f} px")

    keep = track_err < OUTLIER_PX
    mask = np.concatenate([np.repeat(keep, 2),
                           np.ones(8, bool)])  # keep tag rows
    sol = solve(sol.x, mask)
    res = residuals(sol.x).reshape(-1, 2)
    track_err = np.linalg.norm(res[:len(pts)], axis=1) * TRACK_SIGMA_PX
    inl = track_err[keep]
    x = sol.x
    print(f"round 2 ({keep.sum()}/{len(pts)} inliers): "
          f"rms {np.sqrt((inl**2).mean()):.2f} px, "
          f"median {np.median(inl):.2f} px")
    print(f"f={x[0]:.1f} c=({x[1]:.1f},{x[2]:.1f}) k=({x[3]:.4f},{x[4]:.4f}) "
          f"delta={x[11]:+.3f}s r=[{x[12]:+.4f} {x[13]:+.4f} {x[14]:+.4f}]")
    R = cv2.Rodrigues(x[5:8])[0]
    cam = (-R.T @ x[8:11])
    print(f"camera at {np.round(cam, 3)} m in base frame")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6))
    a1.plot(t_video, track_err, ".", ms=2)
    a1.axhline(OUTLIER_PX, color="r", lw=0.6)
    a1.set_xlabel("video time (s)")
    a1.set_ylabel("reproj err (px)")
    a1.set_ylim(0, 50)
    a2.hist(inl, bins=60)
    a2.set_xlabel("inlier reproj err (px)")
    os.makedirs(common.SCRATCH, exist_ok=True)
    fig.savefig(os.path.join(common.SCRATCH, "calib_residuals.png"), dpi=110)

    # camera.json outranks camera_deep.json in camera_path(), so only claim
    # that rank when the point-based solve is actually trustworthy.
    if keep.sum() < 200 or np.sqrt((inl**2).mean()) > 6.0:
        print("NOT writing camera.json (too few inliers or rms too high); "
              "silhouette calibration remains authoritative")
        return
    common.write_json(os.path.join(calib, "camera.json"), {
        "image_wh": [w, h],
        "f": x[0], "cx": x[1], "cy": x[2], "k1": x[3], "k2": x[4],
        "rvec": x[5:8].tolist(), "tvec": x[8:11].tolist(),
        "delta_s": x[11], "tool_offset": x[12:15].tolist(),
        "tool_frame": track["frame"],
        "camera_in_base": cam.tolist(),
        "video_start_epoch": v0,
        "inlier_rms_px": float(np.sqrt((inl**2).mean())),
        "inlier_median_px": float(np.median(inl)),
        "n_inliers": int(keep.sum()), "n_track": len(pts),
        "note": "cv2 convention; base->camera. "
                "video_time = t_log - video_start_epoch + delta_s.",
    })


if __name__ == "__main__":
    main()
