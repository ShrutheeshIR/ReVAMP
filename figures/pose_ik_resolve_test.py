"""
IK-resolve panel: the same 4 sampled SE(3) poses as pose_precheck_test.py
(same START_POSE/END_POSE/N_SAMPLES), resolved to arm configurations via
ik_line_demo.py's analytic 2-link IK -- one subplot per sample, matching
ik_line_demo.py's existing side-by-side layout (overlaying 4 resolved
arms in one shared axes gets visually cluttered, since the links
themselves would overlap).

This runs independently of pose_precheck_test.py's task-space check --
"in parallel", not gated on it -- it just happens to share the same input
samples so the two panels are directly comparable.

Usage:
    python pose_ik_resolve_test.py
Produces pose_ik_resolve_test.png / .svg in this directory.
"""

import matplotlib.pyplot as plt

from robot_arm_fig import (
    draw_arm, draw_pose_frame, gripper_center, PALETTE, LINK_LENGTHS,
    sample_pose_line,
)
from ik_line_demo import analytic_2link_ik
from pose_precheck_test import START_POSE, END_POSE, N_SAMPLES

BASE = (0.0, 0.0)


def main():
    poses = sample_pose_line(START_POSE, END_POSE, N_SAMPLES)

    fig, axes = plt.subplots(1, N_SAMPLES, figsize=(3 * N_SAMPLES, 4.2))
    for i, (ax, (x, y, heading)) in enumerate(zip(axes, poses)):
        color = PALETTE[i % len(PALETTE)]
        solution = analytic_2link_ik(BASE, (x, y), LINK_LENGTHS, elbow_up=True)
        if solution is None:
            ax.set_title("unreachable")
            ax.axis("off")
            continue

        pts, final_heading = draw_arm(ax, BASE, solution, LINK_LENGTHS, color)
        # Draw the pose glyph at the fork's middle, not the wrist joint the
        # IK solve actually targets -- avoids it reading as a collision
        # object sitting on top of the gripper.
        draw_pose_frame(ax, gripper_center(pts[-1], final_heading),
                         final_heading, color, axis_len=0.14, lw=1.6)

        ax.set_xlim(-2.0, 2.2)
        ax.set_ylim(-0.5, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("pose_ik_resolve_test.png", dpi=300, facecolor="white")
    fig.savefig("pose_ik_resolve_test.svg", facecolor="white")
    print("Saved pose_ik_resolve_test.png / pose_ik_resolve_test.svg")


if __name__ == "__main__":
    main()
