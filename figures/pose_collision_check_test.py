"""
Full collision-check panel: takes the IK-resolved arm configurations from
pose_ik_resolve_test.py and checks the actual arm links (not just the
un-resolved wrist point, unlike pose_precheck_test.py) against the same
OBSTACLES -- this is the "resolved config handed to the real collision
checker" step downstream of the IK-resolve panel. Reuses
draw_highlight_circle exactly like robot_arm_fig.py's own elbow-highlight
example, now driven by an actual link/obstacle collision instead of being
hand-placed.

Usage:
    python pose_collision_check_test.py
Produces pose_collision_check_test.png / .svg in this directory.
"""

import matplotlib.pyplot as plt

from robot_arm_fig import (
    draw_arm, draw_pose_frame, gripper_center, draw_obstacles,
    draw_highlight_circle, arm_obstacle_collision, sample_pose_line,
    PALETTE, LINK_LENGTHS,
)
from ik_line_demo import analytic_2link_ik
from pose_precheck_test import START_POSE, END_POSE, N_SAMPLES, OBSTACLES

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

        draw_obstacles(ax, OBSTACLES, zorder=0)
        joint_points, final_heading = draw_arm(ax, BASE, solution, LINK_LENGTHS, color)
        draw_pose_frame(ax, gripper_center(joint_points[-1], final_heading),
                         final_heading, color, axis_len=0.14, lw=1.6)

        hit = arm_obstacle_collision(joint_points, OBSTACLES)
        if hit is not None:
            closest, _radius = hit
            # Fixed small ring right at the contact point -- marks a graze
            # at the link's edge as a graze, not a circle engulfing the
            # whole obstacle regardless of how much of it actually overlaps.
            draw_highlight_circle(ax, closest, radius=0.2)

        ax.set_xlim(-2.0, 2.2)
        ax.set_ylim(-0.5, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("pose_collision_check_test.png", dpi=300, facecolor="white")
    fig.savefig("pose_collision_check_test.svg", facecolor="white")
    print("Saved pose_collision_check_test.png / pose_collision_check_test.svg")


if __name__ == "__main__":
    main()
