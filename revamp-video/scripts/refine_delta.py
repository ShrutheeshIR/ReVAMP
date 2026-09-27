"""1-D clock-offset refinement over fast-motion frames.

Sweeps delta with all other camera parameters frozen (camera_sil.json),
scoring silhouette contrast on frames sampled from the highlight window,
where the arm moves fast and the objective is sharpest in time. Updates
delta_s in camera_sil.json if the sweep improves on the stored value.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from silhouette_calib import (  # noqa: E402
    SphereTrajectories, grab_gray_small, silhouette_mask, SCALE, RING_PX)

TIMES = np.arange(228.0, 274.0, 3.0)
DELTAS = np.arange(-1.6, 0.4, 0.04)


def main():
    calib = common.CALIB_DIR
    cam = common.read_json(os.path.join(calib, "camera_sil.json"))
    v0 = cam["video_start_epoch"]
    w, h = common.VIDEO_WH
    ws, hs = w // SCALE, h // SCALE
    params = np.concatenate([[cam["f"]], cam["rvec"], cam["tvec"]])

    print("precomputing sphere trajectories...")
    traj = SphereTrajectories()
    frames = [grab_gray_small(t) for t in TIMES]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))

    def score(delta):
        s = 0.0
        for t_vid, img in zip(TIMES, frames):
            m = silhouette_mask(params, traj.at(t_vid + v0 - delta),
                                traj.radii, (ws, hs))
            ring = cv2.dilate(m, kernel) & ~m
            mi, ri = m > 0, ring > 0
            if mi.sum() < 100 or ri.sum() < 100:
                return -1e9
            s += float(img[mi].mean() - img[ri].mean())
        return s / len(frames)

    scores = [score(d) for d in DELTAS]
    best = DELTAS[int(np.argmax(scores))]
    cur = score(cam["delta_s"])
    print(f"stored delta {cam['delta_s']:+.3f} score {cur:.2f}")
    print(f"best   delta {best:+.3f} score {max(scores):.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(DELTAS, scores)
    plt.axvline(cam["delta_s"], color="gray", lw=0.8, label="stored")
    plt.axvline(best, color="r", lw=0.8, label="best")
    plt.xlabel("delta (s)")
    plt.ylabel("contrast")
    plt.legend()
    plt.savefig(os.path.join(common.SCRATCH, "delta_scan.png"), dpi=110)

    if max(scores) > cur + 0.5:
        cam["delta_s"] = float(best)
        cam["note"] = cam.get("note", "") + " delta refined by refine_delta.py."
        common.write_json(os.path.join(calib, "camera_sil.json"), cam)
        print("updated camera_sil.json")
    else:
        print("stored delta kept (no significant improvement)")


if __name__ == "__main__":
    main()
