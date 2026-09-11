"""
Task-space pre-check panel: the 4 sampled SE(3) poses (as draw_pose_frame
glyphs, PALETTE-colored like ik_line_demo.py) against a couple of dummy
obstacle spheres. A sample whose position falls inside an obstacle gets a
red highlight ring -- this is the cheap "check the pose itself before
spending an IK solve on it" step, before the IK-resolve panel.

Usage:
    python pose_precheck_test.py
Produces pose_precheck_test.png / .svg in this directory.
"""

import matplotlib.pyplot as plt

from robot_arm_fig import (
    draw_pose_frame, draw_obstacles, draw_highlight_circle,
    point_in_collision, sample_pose_line, PALETTE,
)

# Same start/end poses used across all pieces (precheck, IK-resolve,
# merge) -- small scale so the toy arm (base at origin, LINK_LENGTHS
# (1.0, 1.1), max reach 2.1) can actually reach every sample.
START_POSE = (0.9, 1.5, -0.3)
END_POSE = (-0.3, 2.0, 0.6)
N_SAMPLES = 4

# (center, radius) -- a scattered clutter field rather than a tidy row,
# so the scene reads as a real environment rather than a test pattern.
# Sample index 1 lands inside the first obstacle; the rest are misses
# (some near-misses/grazes once the arm links are checked in
# pose_collision_check_test.py, not just this panel's single wrist point).
OBSTACLES = [
    ((0.5, 1.6), 0.17),
    ((1.3, 0.95), 0.2),
    ((-0.55, 0.95), 0.15),
    ((0.05, 2.15), 0.13),
]

POSE_MARGIN = 0.05  # pose glyph's own footprint, added to obstacle radius


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    draw_obstacles(ax, OBSTACLES)

    poses = sample_pose_line(START_POSE, END_POSE, N_SAMPLES)
    ax.plot([p[0] for p in poses], [p[1] for p in poses],
            linestyle="--", color="#cccccc", zorder=1, lw=1.5)

    for i, (x, y, heading) in enumerate(poses):
        color = PALETTE[i % len(PALETTE)]
        draw_pose_frame(ax, (x, y), heading, color, axis_len=0.25, lw=2.4)
        if point_in_collision((x, y), OBSTACLES, margin=POSE_MARGIN):
            draw_highlight_circle(ax, (x, y), radius=0.28)

    ax.set_xlim(-0.8, 1.8)
    ax.set_ylim(0.55, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig("pose_precheck_test.png", dpi=300, facecolor="white")
    fig.savefig("pose_precheck_test.svg", facecolor="white")
    print("Saved pose_precheck_test.png / pose_precheck_test.svg")


if __name__ == "__main__":
    main()
