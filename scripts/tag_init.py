"""Stage C1: coarse camera pose from the AprilTag (init only — paper is
wrinkled, so this is never trusted beyond seeding search windows).

The 36h11 id-1 tag lies flat on TOP of the 10 cm block in front of the robot
base (apriltag_desc.txt: tag center 23 cm in front of, 19 cm above the base;
"center of the apriltag, i.e. top of the block"). The observed ~2:1 vertical
foreshortening at ~30 deg camera elevation confirms the tag faces UP.

Detects the tag over the first seconds of video, averages corners across
frames, and solves PnP with a guessed focal and tag size; the in-plane yaw of
the tag (unknown, block placed by hand) is resolved by trying all four corner
rotations and keeping the physically sensible camera (in front of the robot,
above the table).

Writes calib/tag_init.json: averaged corners px, K guess, X_cam_base.
"""
from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
from pupil_apriltags import Detector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TAG_CENTER = np.array([0.23, 0.0, 0.19])
TAG_SIZE = 0.083   # m, estimated from pixels vs the 10 cm block; init only
F_GUESS = 2700.0   # px, iPhone main lens ballpark for 4K; refined later
DETECT_TIMES = [0.5, 1.0, 2.0, 3.0, 4.0]


def grab_gray(t):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", common.VIDEO,
           "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    w, h = common.VIDEO_WH
    return np.frombuffer(buf, np.uint8).reshape(h, w)


def main():
    det = Detector(families="tag36h11", nthreads=4)
    corner_sets = []
    for t in DETECT_TIMES:
        img = grab_gray(t)
        for r in det.detect(img):
            if r.tag_id == 1 and r.decision_margin > 30:
                corner_sets.append(r.corners)
    if not corner_sets:
        sys.exit("tag 36h11 id 1 not found in early frames")
    corners_px = np.mean(corner_sets, axis=0)  # (4,2)
    spread = np.std(corner_sets, axis=0).max()
    print(f"{len(corner_sets)} detections, max corner std {spread:.2f} px")

    # Tag on the top face, normal +z, edges nominally along base x/y.
    # pupil_apriltags corner order is a fixed cycle around the tag; the
    # unknown in-plane yaw = trying the 4 cyclic rotations of this square.
    h = TAG_SIZE / 2
    square = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    obj_base = square + TAG_CENTER

    w, hgt = common.VIDEO_WH
    K = np.array([[F_GUESS, 0, w / 2], [0, F_GUESS, hgt / 2], [0, 0, 1]])

    best = None
    for rot in range(4):
        obj = np.roll(obj_base, rot, axis=0).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj, corners_px.astype(np.float64), K, None,
            flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        cam_in_base = (-R.T @ tvec).ravel()
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - corners_px, axis=1).mean())
        # Physically sensible: camera in front of the robot (+x beyond the
        # maze edge at 0.85 m), above the maze plane, not underground.
        plausible = cam_in_base[0] > 0.7 and cam_in_base[2] > 0.2
        print(f"rot {rot}: err {err:6.2f} px  cam_in_base "
              f"[{cam_in_base[0]:+.2f} {cam_in_base[1]:+.2f} "
              f"{cam_in_base[2]:+.2f}]  {'ok' if plausible else 'reject'}")
        if plausible and (best is None or err < best["err"]):
            best = {"rot": rot, "err": err, "rvec": rvec.ravel().tolist(),
                    "tvec": tvec.ravel().tolist(),
                    "cam_in_base": cam_in_base.tolist()}
    if best is None:
        sys.exit("no plausible PnP solution — check TAG_CENTER/TAG_SIZE")

    common.write_json(
        os.path.join(common.CALIB_DIR, "tag_init.json"),
        {"corners_px": corners_px.tolist(), "corner_std_px": float(spread),
         "tag_center_base": TAG_CENTER.tolist(), "tag_size_m": TAG_SIZE,
         "f_guess_px": F_GUESS, **best,
         "note": "rvec/tvec map base-frame points to camera frame (cv2)."})
    print(f"camera at {np.round(best['cam_in_base'], 3)} m in base frame, "
          f"tag reproj err {best['err']:.2f} px")


if __name__ == "__main__":
    main()
