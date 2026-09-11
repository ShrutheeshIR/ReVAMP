"""
Side-by-side comparison of the two oriented-pose glyph styles added to
robot_arm_fig.py -- draw_pose_arrow (short arrow/wedge) and
draw_pose_frame (tiny 2-axis frame). Purely to eyeball which style reads
better before committing to one for the methodology figure's task-space
pre-check panel.

Usage:
    python pose_marker_test.py
Produces pose_marker_test.png / .svg in this directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from robot_arm_fig import draw_pose_arrow, draw_pose_frame, PALETTE

# A handful of poses (x, y, heading) spread out so overlapping glyphs at
# small spacing (as in a real 4-sample line) can also be judged.
POSES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.3, np.pi / 6),
    (2.0, -0.2, -np.pi / 4),
    (3.0, 0.4, np.pi / 2),
]


def render_row(ax, draw_fn, title):
    for i, (x, y, heading) in enumerate(POSES):
        color = PALETTE[i % len(PALETTE)]
        draw_fn(ax, (x, y), heading, color)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.8, 3.8)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5))
    render_row(axes[0], draw_pose_arrow, "draw_pose_arrow (arrow/wedge)")
    render_row(axes[1], draw_pose_frame, "draw_pose_frame (2-axis frame)")

    fig.tight_layout()
    fig.savefig("pose_marker_test.png", dpi=300, facecolor="white")
    fig.savefig("pose_marker_test.svg", facecolor="white")
    print("Saved pose_marker_test.png / pose_marker_test.svg")


if __name__ == "__main__":
    main()
