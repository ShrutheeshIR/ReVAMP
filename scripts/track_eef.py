"""Stage C2: track the marker-holder assembly through the video by NCC.

One streaming full-res decode at TRACK_FPS. For each frame we project the
FK-predicted 3D holder point through the coarse (tag) camera, crop a search
window around prediction + running bias, and template-match. Templates are
kept per q7 bin (the holder's appearance rotates with joint 7) and are seeded
from the crop at the predicted location the first time a bin is visited —
mis-centering of that seed is a *consistent* offset to a rigid tool point,
which calibrate.py absorbs by refining the tracked point's 3D offset r.

Writes calib/track.json: samples of (t_video, u, v, score, q index).
"""
from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TRACK_FPS = 3.0
TRACK_FRAME = "fr3_marker_holder"
TEMPLATE = 110          # px, template side
SEARCH = 90             # px, search half-window; window (2*SEARCH) must
                        # exceed TEMPLATE or matching/seeding can't happen —
                        # ±35 px slack fits the refined camera's prediction
                        # error with room to spare
MIN_SCORE = 0.55        # accept threshold for NCC peak
SEED_SCORE = 0.75       # min score to allow template EMA update
T_START = 9.0           # skip the tag block / setup at the very start
Q7_BIN = 0.6            # rad per template bin


def main():
    cam_path = common.camera_path()
    cam = common.read_json(cam_path)
    print(f"predicting with {os.path.basename(cam_path)}")
    v0 = cam["video_start_epoch"]
    delta = cam["delta_s"]
    w, h = common.VIDEO_WH
    K = np.array([[cam["f"], 0, cam["cx"]],
                  [0, cam["f"], cam["cy"]], [0, 0, 1]])
    rvec, tvec = np.array(cam["rvec"]), np.array(cam["tvec"])

    js = common.load_joint_states()
    qs = common.joint_matrix(js)
    t_video_log = js["t"] - v0 + delta

    print("precomputing FK track-point positions...")
    arm = common.ArmKinematics()
    n_frames = int(common.VIDEO_DURATION_S * TRACK_FPS)
    samples = []  # (frame_idx, t_video, log_idx, pred_uv)
    for fi in range(n_frames):
        t = (fi + 0.5) / TRACK_FPS
        if t < T_START or t > t_video_log[-1]:
            continue
        li = int(np.argmin(np.abs(t_video_log - t)))
        if abs(t_video_log[li] - t) > 0.1:
            continue
        p3 = arm.frame_pose(TRACK_FRAME, qs[li])[:3, 3]
        uv, _ = cv2.projectPoints(p3[None], rvec, tvec, K, None)
        samples.append((fi, t, li, uv.reshape(2)))
    by_frame = {s[0]: s for s in samples}
    print(f"{len(samples)} track samples")

    cmd = ["ffmpeg", "-loglevel", "error", "-i", common.VIDEO,
           "-vf", f"fps={TRACK_FPS}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 4)

    templates = {}   # q7 bin -> float32 template
    bias = np.zeros(2)
    have_bias = False
    out = []
    fi = 0
    nbytes = w * h
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        s = by_frame.get(fi)
        fi += 1
        if s is None:
            continue
        _, t, li, pred = s
        frame = np.frombuffer(buf, np.uint8).reshape(h, w)
        center = pred + (bias if have_bias else 0)
        x0 = int(np.clip(center[0] - SEARCH, 0, w - 2 * SEARCH))
        y0 = int(np.clip(center[1] - SEARCH, 0, h - 2 * SEARCH))
        win = frame[y0:y0 + 2 * SEARCH, x0:x0 + 2 * SEARCH].astype(np.float32)

        key = int(np.floor(qs[li][6] / Q7_BIN))
        tmpl = templates.get(key)
        if tmpl is None:
            # Seed this q7 bin from the predicted location.
            cx, cy = int(center[0] - x0), int(center[1] - y0)
            t0x, t0y = cx - TEMPLATE // 2, cy - TEMPLATE // 2
            if (t0x < 0 or t0y < 0 or t0x + TEMPLATE > win.shape[1]
                    or t0y + TEMPLATE > win.shape[0]):
                continue
            templates[key] = win[t0y:t0y + TEMPLATE, t0x:t0x + TEMPLATE].copy()
            continue

        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        uv = np.array([x0 + loc[0] + TEMPLATE / 2,
                       y0 + loc[1] + TEMPLATE / 2])
        if score >= MIN_SCORE:
            out.append({"t_video": round(t, 4), "log_idx": int(li),
                        "u": float(uv[0]), "v": float(uv[1]),
                        "score": round(float(score), 3)})
            bias = 0.8 * bias + 0.2 * (uv - pred)
            have_bias = True
            if score >= SEED_SCORE:
                m = win[loc[1]:loc[1] + TEMPLATE, loc[0]:loc[0] + TEMPLATE]
                templates[key] = 0.9 * templates[key] + 0.1 * m
    proc.wait()

    scores = np.array([o["score"] for o in out])
    print(f"accepted {len(out)}/{len(samples)} "
          f"(median score {np.median(scores):.3f})")
    common.write_json(
        os.path.join(common.CALIB_DIR, "track.json"),
        {"frame": TRACK_FRAME, "fps": TRACK_FPS, "template_px": TEMPLATE,
         "n_accepted": len(out), "n_samples": len(samples),
         "points": out})


if __name__ == "__main__":
    main()
