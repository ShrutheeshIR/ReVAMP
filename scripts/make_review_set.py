"""Render the current-state review set into scratch/current/.

Fixed filenames, overwritten on every run — ONE current set, no A/B file
clutter. Each still is a labeled two-panel montage: left = baseline
(identity correction), right = current (the correction under review), so
a single image answers "did it get better here?".

    flange_<t>.jpg   crop centered on projected fr3_link8 (the flange
                     collar — the most salient alignment feature)
    tip_<t>.jpg      crop centered on projected marker_holder
    full_<t>.jpg     full-frame overlay (downscaled 2x)
    clip_start.mp4 / clip_285.mp4   (--clips; 10 s ghost segments)

Ghost silhouette EDGES are drawn over the real frame (orange = baseline,
green = current) — edges, not fills, so the misalignment is measurable
by eye in pixels.

Usage: make_review_set.py [--corr calib/model_correction.json]
                          [--camera <path>] [--clips]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

os.environ["DEEP_SCALE"] = "1"          # full 4K masks
import deep_calib  # noqa: E402

FLANGE_TS = [9.5, 60.0, 150.0, 235.0, 285.0, 350.0]
TIP_TS = [9.5, 100.0, 285.0]
FULL_TS = [9.5, 235.0]
CROP_HALF = 500                          # px at 4K
OUT_DIR = os.path.join(common.SCRATCH, "current")

IDENTITY = {"dq": [0.0] * 7, "frames": {}}


def grab_4k(t):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i",
           common.VIDEO, "-frames:v", "1", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    w, h = common.VIDEO_WH
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()


def label(img, text):
    cv2.putText(img, text, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(img, text, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 3, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", default=common.MODEL_CORRECTION)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--clips", action="store_true")
    args = ap.parse_args()

    cam_path = args.camera or common.camera_path()
    cam = common.read_json(cam_path)
    corr = common.load_model_correction(args.corr)
    os.makedirs(OUT_DIR, exist_ok=True)
    dq = common.correction_dq(corr)
    print(f"camera {cam_path}", flush=True)
    print(f"correction {args.corr} "
          f"(|dq|max {1000*np.abs(dq).max():.1f} mrad)", flush=True)

    v0, delta = cam["video_start_epoch"], cam["delta_s"]
    K = np.array([[cam["f"], 0, cam["cx"]],
                  [0, cam["f"], cam["cy"]], [0, 0, 1]])
    rvec = np.array(cam["rvec"], float)
    tvec = np.array(cam["tvec"], float)
    margs = (cam["f"], cam["cx"], cam["cy"], rvec, tvec)

    ms = deep_calib.MeshSilhouette()
    arm = common.ArmKinematics(correction=corr)

    def mask_edges(t, c):
        ms.set_correction(c)
        m = ms.mask(t + v0 - delta, *margs).astype(np.uint8)
        return m & ~cv2.erode(m, None)

    def project(frame_name, t):
        q = ms.q_at(t + v0 - delta)
        p = arm.frame_pose(frame_name, q)[:3, 3]
        uv = cv2.projectPoints(p.reshape(1, 3), rvec, tvec, K,
                               np.zeros(4))[0].ravel()
        return uv

    def panel(real, edges, color, text):
        img = real.copy()
        img[edges > 0] = color
        return label(img, text)

    def crop(img, uv):
        h, w = img.shape[:2]
        x = int(np.clip(uv[0], CROP_HALF, w - CROP_HALF))
        y = int(np.clip(uv[1], CROP_HALF, h - CROP_HALF))
        return img[y - CROP_HALF:y + CROP_HALF,
                   x - CROP_HALF:x + CROP_HALF]

    for kind, ts, frame_name in (("flange", FLANGE_TS, "fr3_link8"),
                                 ("tip", TIP_TS, "fr3_marker_holder")):
        for t in ts:
            real = grab_4k(t)
            e_base = mask_edges(t, IDENTITY)
            e_cur = mask_edges(t, corr)
            arm.set_correction(corr)
            uv = project(frame_name, t)
            left = label(crop(panel(real, e_base, (0, 140, 255), ""), uv)
                         .copy(), f"baseline  t={t:g}")
            right = label(crop(panel(real, e_cur, (0, 230, 0), ""), uv)
                          .copy(), f"current  t={t:g}")
            outp = os.path.join(OUT_DIR, f"{kind}_{t:g}.jpg")
            cv2.imwrite(outp, np.hstack([left, right]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(outp, flush=True)

    for t in FULL_TS:
        real = grab_4k(t)
        e_cur = mask_edges(t, corr)
        img = panel(real, e_cur, (0, 230, 0), f"current  t={t}")
        img = cv2.resize(img, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
        outp = os.path.join(OUT_DIR, f"full_{t:g}.jpg")
        cv2.imwrite(outp, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(outp, flush=True)

    if args.clips:
        env = dict(os.environ, MODEL_CORRECTION_PATH=args.corr)
        for name, t0, t1 in (("clip_start", 9, 19), ("clip_285", 280, 290)):
            outp = os.path.join(OUT_DIR, f"{name}.mp4")
            subprocess.run(
                [sys.executable, os.path.join(HERE, "sim_ghost.py"),
                 "--segment", str(t0), str(t1), outp], check=True, env=env)
            print(outp, flush=True)
    print("REVIEW SET DONE", flush=True)


if __name__ == "__main__":
    main()
