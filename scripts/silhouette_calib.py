"""Stage C/D alternative: refine the camera by silhouette contrast.

The URDF is spherized for collision (53 spheres over arm + tool), so an
approximate robot silhouette renders as projected filled circles — no GL.
The white robot sits on black curtains, so a well-aligned silhouette has
maximal brightness contrast between its interior and a surrounding ring.
We maximize that contrast over a set of sample frames, starting from the
tag-init camera, over [f, rvec, tvec, delta]; the wrinkled tag's corners
remain a weak anchor. Result seeds the final NCC+NLLS polish and directly
drives the Phase-2 matched sim render.

Writes calib/camera_sil.json and scratch/sil_*.jpg diagnostics.
"""
from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SCALE = 4            # evaluate at 1/SCALE resolution (960x540)
N_FRAMES = 16
T_RANGE = (10.0, 382.0)
RING_PX = 10         # ring width at eval resolution
TAG_SIGMA_PX = 25.0
TAG_WEIGHT = 0.02    # contrast units per (px/sigma)^2 — weak anchor


def collision_spheres():
    """[(body_frame_name, p_local(3), radius)] from the URDF's proximity geoms."""
    from pydrake.geometry import Sphere
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    parser.package_map().Add("fr3_marker", os.path.dirname(common.URDF))
    (model,) = parser.AddModels(common.URDF)
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("fr3_link0", model))
    plant.Finalize()
    insp = scene_graph.model_inspector()
    out = []
    for gid in insp.GetAllGeometryIds():
        shape = insp.GetShape(gid)
        if isinstance(shape, Sphere):
            body = plant.GetBodyFromFrameId(insp.GetFrameId(gid))
            out.append((body.body_frame().name(),
                        insp.GetPoseInFrame(gid).translation().copy(),
                        shape.radius()))
    return out


class SphereTrajectories:
    """World positions of every collision sphere, interpolable in time."""

    def __init__(self):
        js = common.load_joint_states()
        qs = common.joint_matrix(js)
        self.t = js["t"]
        idx = np.arange(len(qs))
        spheres = collision_spheres()
        arm = common.ArmKinematics()
        frames = sorted({f for f, _, _ in spheres})
        pose_p, pose_R = {}, {}
        for f in frames:
            Xs = [arm.frame_pose(f, q) for q in qs]
            pose_p[f] = np.array([X[:3, 3] for X in Xs])
            pose_R[f] = Rotation.from_matrix(np.array([X[:3, :3] for X in Xs]))
        # (T, S, 3) world sphere centers over the whole log
        self.centers = np.stack(
            [pose_p[f] + pose_R[f].apply(np.tile(p, (len(idx), 1)))
             for f, p, _ in spheres], axis=1)
        self.radii = np.array([r for _, _, r in spheres])

    def at(self, t_epoch):
        t = np.clip(t_epoch, self.t[0], self.t[-1])
        i = np.searchsorted(self.t, t) - 1
        i = np.clip(i, 0, len(self.t) - 2)
        w = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        return (1 - w) * self.centers[i] + w * self.centers[i + 1]


def grab_gray_small(t):
    w, h = common.VIDEO_WH[0] // SCALE, common.VIDEO_WH[1] // SCALE
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", common.VIDEO,
           "-frames:v", "1", "-vf", f"scale={w}:{h}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)


def silhouette_mask(params, centers, radii, wh):
    f, rvec, tvec = params[0], params[1:4], params[4:7]
    R = cv2.Rodrigues(rvec)[0]
    pc = centers @ R.T + tvec
    z = pc[:, 2]
    ok = z > 0.2
    K = f / SCALE
    u = pc[:, 0] / z * K + wh[0] / 2 / 1
    v = pc[:, 1] / z * K + wh[1] / 2
    r_px = radii / z * K
    mask = np.zeros((wh[1], wh[0]), np.uint8)
    for ui, vi, ri, o in zip(u, v, r_px, ok):
        if o:
            cv2.circle(mask, (int(ui), int(vi)), max(int(ri), 1), 255, -1)
    return mask


def main():
    calib = common.CALIB_DIR
    tag = common.read_json(os.path.join(calib, "tag_init.json"))
    sync = common.read_json(os.path.join(calib, "sync.json"))
    motion = common.read_json(os.path.join(calib, "motion_delta.json"))
    v0 = sync["video_start_epoch"]
    w, h = common.VIDEO_WH
    ws, hs = w // SCALE, h // SCALE

    print("precomputing sphere trajectories...")
    traj = SphereTrajectories()

    times = np.linspace(*T_RANGE, N_FRAMES)
    frames = [grab_gray_small(t) for t in times]
    print(f"{len(frames)} frames loaded")

    s = tag["tag_size_m"] / 2
    square = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
    tag_obj = np.roll(square, tag["rot"], axis=0) + np.array(
        tag["tag_center_base"])
    tag_uv = np.array(tag["corners_px"])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * RING_PX + 1, 2 * RING_PX + 1))

    def contrast(x):
        f, rvec, tvec, delta = x[0], x[1:4], x[4:7], x[7]
        score = 0.0
        for t_vid, img in zip(times, frames):
            centers = traj.at(t_vid + v0 - delta)
            m = silhouette_mask(np.concatenate([[f], rvec, tvec]),
                                centers, traj.radii, (ws, hs))
            ring = cv2.dilate(m, kernel) & ~m
            mi, ri = m > 0, ring > 0
            if mi.sum() < 100 or ri.sum() < 100:
                return 1e6
            score += float(img[mi].mean() - img[ri].mean())
        # weak tag anchor
        K = np.array([[x[0], 0, w / 2], [0, x[0], h / 2], [0, 0, 1]])
        uvt, _ = cv2.projectPoints(tag_obj, x[1:4], x[4:7], K, None)
        tag_pen = ((np.linalg.norm(uvt.reshape(-1, 2) - tag_uv, axis=1)
                    / TAG_SIGMA_PX) ** 2).sum()
        return -(score / len(frames)) + TAG_WEIGHT * tag_pen

    x0 = np.concatenate([[tag["f_guess_px"]], tag["rvec"], tag["tvec"],
                         [motion["delta_s"]]])
    print(f"initial objective {contrast(x0):.3f}")
    res = minimize(contrast, x0, method="Powell",
                   options={"maxfev": 4000, "xtol": 1e-4, "ftol": 1e-5})
    x = res.x
    print(f"final objective {res.fun:.3f} after {res.nfev} evals")
    R = cv2.Rodrigues(x[1:4])[0]
    cam = -R.T @ x[4:7]
    print(f"f={x[0]:.1f} delta={x[7]:+.3f}s camera at {np.round(cam, 3)}")

    os.makedirs(common.SCRATCH, exist_ok=True)
    for t_vid, img in zip(times[::5], frames[::5]):
        centers = traj.at(t_vid + v0 - x[7])
        m = silhouette_mask(np.concatenate([x[:1], x[1:4], x[4:7]]),
                            centers, traj.radii, (ws, hs))
        vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(common.SCRATCH, f"sil_{t_vid:.0f}.jpg"), vis)

    common.write_json(os.path.join(calib, "camera_sil.json"), {
        "image_wh": [w, h],
        "f": float(x[0]), "cx": w / 2, "cy": h / 2, "k1": 0.0, "k2": 0.0,
        "rvec": x[1:4].tolist(), "tvec": x[4:7].tolist(),
        "delta_s": float(x[7]), "camera_in_base": cam.tolist(),
        "video_start_epoch": v0, "objective": float(res.fun),
        "note": "silhouette-contrast fit; cv2 convention, base->camera."})


if __name__ == "__main__":
    main()
