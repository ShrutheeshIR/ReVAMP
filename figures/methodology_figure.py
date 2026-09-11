"""
Full methodology figure: composes the individually-verified pieces
(pose_line_test.py, pose_precheck_test.py, pose_ik_resolve_test.py) into
one image, laid out as a row-aligned 3-stage pipeline:

    col 0: sampled SE(3) line   col 1: task-space pre-check   col 2: IK-resolve
    row 0 .. row 3, one row per sample, same color per row across all
    three columns so the pipeline reads left-to-right *and* top-to-bottom
    consistently.

Column 1 (pre-check) and column 2 (IK-resolve) run in parallel off the
same 4 samples from column 0 -- pre-check is a cheap goal-vs-obstacle
check on the un-resolved pose itself; IK-resolve is ik_line_demo.py's
analytic 2-link IK, independent of whether pre-check passed.

Usage:
    python methodology_figure.py
Produces methodology_figure.png / .svg in this directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from robot_arm_fig import (
    draw_arm, draw_goal_marker, gripper_center, point_in_collision,
    arm_obstacle_collision, sample_pose_line, PALETTE, LINK_LENGTHS,
)
from ik_line_demo import analytic_2link_ik

BASE = (0.0, 0.0)

# Vertical line: fixed x, y sweeping from START_POSE to END_POSE. Small
# scale so the toy arm (base at origin, LINK_LENGTHS (1.0, 1.1), max
# reach 2.1) can reach every sample.
START_POSE = (0.3, 1.3, -0.3)
END_POSE = (0.3, 2.0, 0.6)
N_SAMPLES = 4

# (center, radius) -- same 4 obstacles, same locations, used by *both*
# the early task-space pre-check and the later full link-collision check
# -- one environment, checked two different ways. Placed so exactly one
# sample fails each stage instead of everything piling up on one check:
# sample 1's *goal* sits right on an obstacle (fails early), and sample
# 3's goal is clear but its forearm sweeps through a different obstacle
# on the way there (only caught by the later, link-aware check).
OBSTACLES = [
    ((0.75, 1.85), 0.16),
    ((0.6, 1.15), 0.12),
    ((0.3, 1.5333), 0.035),
    ((0.263, 1.915), 0.035),
]
POSE_MARGIN = 0.05

LINE_XLIM = (-0.7, 1.3)
LINE_YLIM = (0.85, 2.35)

LINE_COL_XLIM = (-0.5, 0.5)
LINE_COL_YLIM = (-0.6, 3.6)

OBSTACLE_COLOR = "#b0b0b0"
OBSTACLE_EDGE = "#8f8f8f"

TITLES = [
    "interpolated SE(3) poses",
    "early collision checking",
    "IK resolution",
    "link collision check",
]


def bare(ax):
    """Hide ticks/labels/spines but keep the axes' own facecolor patch
    visible -- ax.axis('off') would drop the tint along with the ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_obstacle(ax, center, radius):
    """Obstacle 'sphere', drawn as a flat gray disc -- same color for
    every obstacle in every row."""
    ax.add_patch(Circle(center, radius, facecolor=OBSTACLE_COLOR,
                         edgecolor=OBSTACLE_EDGE, lw=1.1, zorder=1))


def draw_line_column(ax, poses):
    """Single panel spanning all 4 rows: one straight line actually
    connecting all 4 sampled poses, not 4 separate cells stitched
    together after the fact."""
    ax.plot([0, 0], [0, N_SAMPLES - 1], linestyle=(0, (2, 2)),
            color="#bfbfbf", lw=1.6, zorder=1)
    ax.text(-0.36, (N_SAMPLES - 1) / 2, "t : 0 → 1", rotation=90,
            ha="center", va="center", fontsize=10.5, color="#555555")
    for i, (_x, _y, heading) in enumerate(poses):
        draw_goal_marker(ax, (0.0, float(i)), heading,
                          PALETTE[i % len(PALETTE)], ring_radius=0.16,
                          tick_len=0.28, lw=3.0)
    ax.set_xlim(*LINE_COL_XLIM)
    ax.set_ylim(*LINE_COL_YLIM)
    ax.set_aspect("equal")
    bare(ax)


def draw_precheck_panel(ax, pose, color):
    x, y, heading = pose
    for center, radius in OBSTACLES:
        draw_obstacle(ax, center, radius)

    draw_goal_marker(ax, (x, y), heading, color, ring_radius=0.11,
                      tick_len=0.24, lw=2.6)

    if point_in_collision((x, y), OBSTACLES, margin=POSE_MARGIN):
        ax.plot(x, y, marker="x", ms=13, mew=2.6, color="#c0392b", zorder=7)

    ax.set_xlim(*LINE_XLIM)
    ax.set_ylim(*LINE_YLIM)
    ax.set_aspect("equal")
    bare(ax)


IK_XLIM = (-1.85, 2.35)
IK_YLIM = (-0.3, 3.0)


def draw_ik_panel(ax, pose, color):
    x, y, _heading = pose
    solution = analytic_2link_ik(BASE, (x, y), LINK_LENGTHS, elbow_up=True)
    if solution is None:
        ax.text(0.5, 0.5, "unreachable", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#888888")
        ax.set_xlim(*IK_XLIM)
        ax.set_ylim(*IK_YLIM)
        bare(ax)
        return

    pts, final_heading = draw_arm(ax, BASE, solution, LINK_LENGTHS, color)
    draw_goal_marker(ax, gripper_center(pts[-1], final_heading),
                      final_heading, color, ring_radius=0.075,
                      tick_len=0.16, lw=1.8)
    ax.set_xlim(*IK_XLIM)
    ax.set_ylim(*IK_YLIM)
    ax.set_aspect("equal")
    bare(ax)


def draw_full_check_panel(ax, pose, color):
    """Same resolved arm as the IK panel, but now checked link-by-link
    against the environment instead of just the goal point -- catches a
    collision the cheap task-space pre-check can't see, since the
    pre-check only ever looks at the un-resolved goal, never the swept
    links."""
    x, y, _heading = pose
    for center, radius in OBSTACLES:
        draw_obstacle(ax, center, radius)

    solution = analytic_2link_ik(BASE, (x, y), LINK_LENGTHS, elbow_up=True)
    if solution is None:
        ax.text(0.5, 0.5, "unreachable", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#888888")
        ax.set_xlim(*IK_XLIM)
        ax.set_ylim(*IK_YLIM)
        bare(ax)
        return

    pts, final_heading = draw_arm(ax, BASE, solution, LINK_LENGTHS, color)
    draw_goal_marker(ax, gripper_center(pts[-1], final_heading),
                      final_heading, color, ring_radius=0.075,
                      tick_len=0.16, lw=1.8)

    hit = arm_obstacle_collision(pts, OBSTACLES)
    if hit is not None:
        hit_center, _hit_radius = hit
        ax.plot(*hit_center, marker="x", ms=15, mew=3.0, color="#c0392b",
                zorder=7)

    ax.set_xlim(*IK_XLIM)
    ax.set_ylim(*IK_YLIM)
    ax.set_aspect("equal")
    bare(ax)


def draw_column_titles(fig, axes_per_column):
    for ax, title in zip(axes_per_column, TITLES):
        bbox = ax.get_position()
        fig.text(bbox.x0, bbox.y1 + 0.012, title,
                  ha="left", va="bottom", fontsize=12.5, fontweight="bold")


def _row_y_frac(ylim, y):
    return (y - ylim[0]) / (ylim[1] - ylim[0])


def draw_fan_out(fig, line_ax, precheck_axes):
    """One colored line per sample, fanning from its dot on the shared
    t: 0->1 timeline straight to its own lane -- the timeline branches
    into N independent lanes here, instead of everything funnelling
    through one shared step."""
    line_bbox = line_ax.get_position()
    for i, ax in enumerate(precheck_axes):
        src_y = line_bbox.y0 + _row_y_frac(LINE_COL_YLIM, i) * line_bbox.height
        src = (line_bbox.x1, src_y)
        tgt_bbox = ax.get_position()
        tgt = (tgt_bbox.x0, tgt_bbox.y0 + tgt_bbox.height / 2)
        line = Line2D([src[0], tgt[0]], [src[1], tgt[1]],
                      transform=fig.transFigure, color=PALETTE[i % len(PALETTE)],
                      lw=2.0, solid_capstyle="round", zorder=0.5)
        fig.add_artist(line)


def main():
    poses = sample_pose_line(START_POSE, END_POSE, N_SAMPLES)

    fig = plt.figure(figsize=(12.6, 9.2))
    # Column 1 is a deliberate empty spacer -- it's what gives the
    # fan-out lines from the timeline enough room to actually read as
    # diagonal branches instead of collapsing into a stub in the wspace
    # gutter.
    gs = gridspec.GridSpec(N_SAMPLES, 5,
                            width_ratios=[0.62, 0.42, 1.0, 0.95, 0.95],
                            wspace=0.05, hspace=0.05, figure=fig,
                            left=0.02, right=0.98, top=0.90, bottom=0.02)

    line_ax = fig.add_subplot(gs[:, 0])
    draw_line_column(line_ax, poses)

    # Row 0 (top) gets the *last* sample (index N_SAMPLES - 1) so the
    # top-to-bottom order matches the line's start(bottom)-to-end(top)
    # direction instead of running opposite it.
    top_axes = [line_ax, None, None, None]
    precheck_axes = [None] * N_SAMPLES
    for i, pose in enumerate(poses):
        color = PALETTE[i % len(PALETTE)]
        row = N_SAMPLES - 1 - i
        precheck_ax = fig.add_subplot(gs[row, 2])
        ik_ax = fig.add_subplot(gs[row, 3])
        full_ax = fig.add_subplot(gs[row, 4])
        draw_precheck_panel(precheck_ax, pose, color)
        draw_ik_panel(ik_ax, pose, color)
        draw_full_check_panel(full_ax, pose, color)
        precheck_axes[i] = precheck_ax
        if row == 0:
            top_axes[1], top_axes[2], top_axes[3] = precheck_ax, ik_ax, full_ax

    fig.canvas.draw()
    draw_column_titles(fig, top_axes)
    draw_fan_out(fig, line_ax, precheck_axes)

    fig.savefig("methodology_figure.png", dpi=300, facecolor="white")
    fig.savefig("methodology_figure.svg", facecolor="white")
    print("Saved methodology_figure.png / methodology_figure.svg")


if __name__ == "__main__":
    main()
