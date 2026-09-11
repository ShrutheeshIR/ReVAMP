"""
Visual check for sample_pose_line: draws the start/end poses (larger,
labeled) plus n interpolated poses between them, all as draw_pose_frame
glyphs, to confirm position + heading interpolation looks right before
using this as the sampled-line input to the methodology figure's panels.

Usage:
    python pose_line_test.py
Produces pose_line_test.png / .svg in this directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from robot_arm_fig import draw_pose_frame, sample_pose_line, PALETTE

START_POSE = (0.0, 0.0, -0.4)
END_POSE = (3.2, 1.4, 2.3)
N_SAMPLES = 4


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    poses = sample_pose_line(START_POSE, END_POSE, N_SAMPLES)
    for i, (x, y, heading) in enumerate(poses):
        color = PALETTE[i % len(PALETTE)]
        draw_pose_frame(ax, (x, y), heading, color, axis_len=0.4, lw=3.0)

    # start/end labeled explicitly, drawn last so they sit on top
    draw_pose_frame(ax, START_POSE[:2], START_POSE[2], "black", axis_len=0.5, lw=3.5)
    draw_pose_frame(ax, END_POSE[:2], END_POSE[2], "black", axis_len=0.5, lw=3.5)
    ax.annotate("start", START_POSE[:2], xytext=(0, -22),
                textcoords="offset points", ha="center", fontsize=11)
    ax.annotate("end", END_POSE[:2], xytext=(0, -22),
                textcoords="offset points", ha="center", fontsize=11)

    ax.plot([START_POSE[0], END_POSE[0]], [START_POSE[1], END_POSE[1]],
            linestyle="--", color="#cccccc", zorder=1, lw=1.5)

    ax.set_xlim(-0.8, 4.0)
    ax.set_ylim(-1.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig("pose_line_test.png", dpi=300, facecolor="white")
    fig.savefig("pose_line_test.svg", facecolor="white")
    print("Saved pose_line_test.png / pose_line_test.svg")


if __name__ == "__main__":
    main()
