"""Stage C0: estimate the clock offset delta by motion correlation.

Cross-correlates the video's frame-difference motion energy against the
logged joint-speed profile. This refines the wall-clock sync (sync.json) well
enough that template tracking's FK-projected search windows land on target;
the final delta is still re-estimated by calibrate.py.

Writes calib/motion_delta.json and scratch/motion_profile.png.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FPS = 6.0
W, H = 480, 270


def video_motion_energy():
    """Per-sample mean |frame difference| over the arm region of the video."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", common.VIDEO,
        "-vf", f"fps={FPS},scale={W}:{H}", "-pix_fmt", "gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev, energy = None, []
    nbytes = W * H
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.int16)
        # Arm region: upper 60% of the frame (maze floor reflections and the
        # operator's legs live lower); crop borders against camera-edge noise.
        roi = frame[: int(H * 0.6), int(W * 0.05): int(W * 0.95)]
        if prev is not None:
            energy.append(np.abs(roi - prev).mean())
        prev = roi
    proc.wait()
    return np.array(energy)


def joint_speed_profile(t_grid):
    js = common.load_joint_states()
    qs = common.joint_matrix(js)
    t = js["t"]
    speed = np.linalg.norm(np.diff(qs, axis=0), axis=1) / np.diff(t)
    mid = 0.5 * (t[1:] + t[:-1])
    return np.interp(t_grid, mid, speed, left=0.0, right=0.0)


def main():
    sync = common.read_json(os.path.join(common.CALIB_DIR, "sync.json"))
    v0 = sync["video_start_epoch"]

    energy = video_motion_energy()
    t_video = (np.arange(len(energy)) + 1.0) / FPS  # diff sample i ~ frame i+1

    # Candidate deltas: logged motion mapped to video time t - v0 + delta.
    deltas = np.arange(-4.0, 4.0, 1.0 / FPS)
    e = (energy - energy.mean()) / energy.std()
    best, scores = None, []
    for d in deltas:
        s = joint_speed_profile(t_video + v0 - d)
        if s.std() < 1e-9:
            scores.append(0.0)
            continue
        s = (s - s.mean()) / s.std()
        scores.append(float(np.dot(e, s) / len(e)))
    scores = np.array(scores)
    best = float(deltas[np.argmax(scores)])
    print(f"best delta = {best:+.3f} s  (corr {scores.max():.3f}; "
          f"runner-up gap {scores.max() - np.sort(scores)[-2]:.4f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6))
    a1.plot(deltas, scores)
    a1.axvline(best, color="r", lw=0.8)
    a1.set_xlabel("delta (s)")
    a1.set_ylabel("corr")
    s = joint_speed_profile(t_video + v0 - best)
    a2.plot(t_video, e, lw=0.5, label="video motion (z)")
    a2.plot(t_video, (s - s.mean()) / (s.std() + 1e-9), lw=0.5,
            label=f"joint speed shifted {best:+.2f}s (z)")
    a2.legend()
    a2.set_xlabel("video time (s)")
    os.makedirs(common.SCRATCH, exist_ok=True)
    fig.savefig(os.path.join(common.SCRATCH, "motion_profile.png"), dpi=110)

    common.write_json(
        os.path.join(common.CALIB_DIR, "motion_delta.json"),
        {"delta_s": best, "corr": float(scores.max()),
         "fps": FPS, "note": "video_time = t_log - video_start_epoch + delta"})


if __name__ == "__main__":
    main()
