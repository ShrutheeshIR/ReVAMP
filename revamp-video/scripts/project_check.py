"""Diagnostic: project the FK skeleton + logged eef through the current
camera guess onto sampled video frames. Writes scratch/proj_<t>.jpg.

Usage: project_check.py [t_video ...]
"""
from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def load_camera():
    tag = common.read_json(os.path.join(common.CALIB_DIR, "tag_init.json"))
    w, h = common.VIDEO_WH
    K = np.array([[tag["f_guess_px"], 0, w / 2],
                  [0, tag["f_guess_px"], h / 2], [0, 0, 1]])
    return K, np.array(tag["rvec"]), np.array(tag["tvec"])


def grab_bgr(t):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", common.VIDEO,
           "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    w, h = common.VIDEO_WH
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()


def main():
    ts = [float(a) for a in sys.argv[1:]] or [30.0, 60.0, 120.0]
    K, rvec, tvec = load_camera()
    sync = common.read_json(os.path.join(common.CALIB_DIR, "sync.json"))
    delta = common.read_json(
        os.path.join(common.CALIB_DIR, "motion_delta.json"))["delta_s"]
    v0 = sync["video_start_epoch"]

    js = common.load_joint_states()
    qs = common.joint_matrix(js)
    t_log = js["t"]
    t_video_log = t_log - v0 + delta

    arm = common.ArmKinematics()
    os.makedirs(common.SCRATCH, exist_ok=True)
    for t in ts:
        i = int(np.argmin(np.abs(t_video_log - t)))
        pts = arm.skeleton(qs[i])
        proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
        proj = proj.reshape(-1, 2)
        img = grab_bgr(t)
        for a, b in zip(proj[:-1], proj[1:]):
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                     (0, 0, 255), 3)
        for p, name in zip(proj, arm.SKELETON_FRAMES):
            cv2.circle(img, tuple(p.astype(int)), 10, (0, 255, 0), 2)
        # logged eef (hand_tcp) in blue
        eef = common.eef_matrix(js)[i:i + 1]
        pe, _ = cv2.projectPoints(eef, rvec, tvec, K, None)
        cv2.circle(img, tuple(pe.reshape(2).astype(int)), 14, (255, 100, 0), 4)
        out = os.path.join(common.SCRATCH, f"proj_{t:.0f}.jpg")
        cv2.imwrite(out, cv2.resize(img, (1280, 720)))
        print(out, "log idx", i, "q7=%.2f" % qs[i][6])


if __name__ == "__main__":
    main()
