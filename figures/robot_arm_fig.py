"""
Illustrative 2-link planar robot arm with a rectangular parallel-jaw
gripper, flat mechanical style (sharp rectangular links + pivot discs).

Usage:
    python robot_arm_fig.py

Produces robot_arm_fig.png and robot_arm_fig.svg in this directory.
Tweak ARM_POSES / PALETTE / sizes below to reuse for other figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon
import math
# ----------------------------------------------------------------------
# style constants
# ----------------------------------------------------------------------
LINK_HALF_WIDTH = 0.05       # half-width of arm link rectangles
BASE_HALF_WIDTH = 0.075       # half-width (thickness) of the base rail
JOINT_RADIUS = 0.05         # pivot disc radius at each rotational joint

STUB_LEN = 0.0028              # stub length before the gripper crossbar
CROSSBAR_HALF_WIDTH = 0.05   # half-thickness of the gripper crossbar
FINGER_LEN = 0.35            # length of each parallel finger
JAW_HALF_GAP = 0.18          # half-distance between the two fingers

BASE_RAIL_HALF = 0.55        # half-length of the base rail segment


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def forward_kinematics(base, angles, lengths):
    """
    base: (x, y) position of the first joint.
    angles: sequence of relative bend angles (radians), first one measured
        from "straight up", each subsequent one relative to the previous
        link's direction.
    lengths: sequence of link lengths, same length as angles.

    Returns (joint_points, final_direction_angle).
    joint_points includes the base as the first point.
    """
    x, y = base
    heading = np.pi / 2  # start pointing straight up
    pts = [(x, y)]
    for theta, L in zip(angles, lengths):
        heading += theta
        x = x + L * np.cos(heading)
        y = y + L * np.sin(heading)
        pts.append((x, y))
    return pts, heading


def rect_corners(p0, p1, half_width):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return None
    dirv = d / length
    normal = np.array([-dirv[1], dirv[0]])
    return np.array([
        p0 + normal * half_width,
        p1 + normal * half_width,
        p1 - normal * half_width,
        p0 - normal * half_width,
    ])


# ----------------------------------------------------------------------
# drawing primitives
# ----------------------------------------------------------------------
def draw_rect_segment(ax, p0, p1, color, half_width, zorder=2, alpha=1.0):
    corners = rect_corners(p0, p1, half_width)
    if corners is None:
        return
    ax.add_patch(Polygon(corners, closed=True, facecolor=color,
                          edgecolor="none", zorder=zorder, alpha=alpha))



def draw_joint(ax, center, color, radius=JOINT_RADIUS, zorder=3, alpha=1.0):
    ax.add_patch(Circle(center, radius, facecolor=color, edgecolor="none",
                         zorder=zorder, alpha=alpha))


def draw_base_rail(ax, base, color, half_len=BASE_RAIL_HALF,
                    half_width=BASE_HALF_WIDTH, zorder=2, alpha=1.0):
    x, y = base
    draw_rect_segment(ax, (x - half_len, y), (x + half_len, y), color,
                       half_width, zorder=zorder, alpha=alpha)


def draw_gripper(ax, wrist, heading, color, zorder=2):
    """Parallel-jaw gripper fork: stub -> crossbar -> two fingers, all
    drawn as a single uniform-width stroke (no thick/thin mismatch)."""
    wrist = np.asarray(wrist, dtype=float)
    d = np.array([np.cos(heading), np.sin(heading)])   # along-arm direction
    n = np.array([-np.sin(heading), np.cos(heading)])  # left-hand normal

    stub_tip = wrist + d * STUB_LEN
    # draw_rect_segment(ax, wrist, stub_tip, color, CROSSBAR_HALF_WIDTH, zorder)

    left_root = stub_tip + n * JAW_HALF_GAP
    right_root = stub_tip - n * JAW_HALF_GAP
    draw_rect_segment(ax, left_root, right_root, color,
                       CROSSBAR_HALF_WIDTH, zorder)

    left_tip = left_root + d * FINGER_LEN
    right_tip = right_root + d * FINGER_LEN
    draw_rect_segment(ax, left_root, left_tip, color, CROSSBAR_HALF_WIDTH,
                       zorder)
    draw_rect_segment(ax, right_root, right_tip, color, CROSSBAR_HALF_WIDTH,
                       zorder)

    # small pivot discs to keep the stub/crossbar/finger seams clean
    draw_joint(ax, stub_tip, color, radius=CROSSBAR_HALF_WIDTH, zorder=zorder + 1)
    draw_joint(ax, left_root, color, radius=CROSSBAR_HALF_WIDTH,
               zorder=zorder + 1)
    draw_joint(ax, right_root, color, radius=CROSSBAR_HALF_WIDTH,
               zorder=zorder + 1)


def gripper_center(wrist, heading):
    """Point midway along the two fingers' length, centered between them
    -- i.e. the middle of the fork, not the wrist pivot the arm's IK
    actually solves for. Use this (not the wrist) as where a pose glyph
    or grasp target should be drawn, since "the target" is conceptually
    what sits between the jaws, not the joint the stub hangs off of."""
    wrist = np.asarray(wrist, dtype=float)
    d = np.array([np.cos(heading), np.sin(heading)])
    return wrist + d * (STUB_LEN + FINGER_LEN / 2)


def gripper_tip(wrist, heading):
    """The fingertip point (fork's far end, along the central axis) --
    the actual "eef" point that should sit on the constraint plane."""
    wrist = np.asarray(wrist, dtype=float)
    d = np.array([np.cos(heading), np.sin(heading)])
    return wrist + d * (STUB_LEN + FINGER_LEN)


def draw_pose_arrow(ax, pos, heading, color, length=0.32, head_width=0.14,
                     head_length=0.16, lw=0, zorder=6):
    """Oriented pose glyph, style 1: a short filled arrow/wedge pointing
    along `heading` from `pos`. Compact -- reads as a single mark even at
    small panel size, at the cost of only showing heading, not a full
    frame."""
    x, y = pos
    dx = length * np.cos(heading)
    dy = length * np.sin(heading)
    ax.arrow(x - dx / 2, y - dy / 2, dx, dy, color=color,
              width=head_width * 0.35, head_width=head_width,
              head_length=head_length, linewidth=lw,
              length_includes_head=True, zorder=zorder)


def draw_pose_frame(ax, pos, heading, color, axis_len=0.32, lw=2.6,
                     secondary_color="#9a9a9a", zorder=6):
    """Oriented pose glyph, style 2: a tiny 2-axis frame at `pos` -- the
    "forward" axis (along `heading`) in the pose's own color, the
    perpendicular axis in a shared neutral gray so it reads as "the other
    axis" rather than a second distinct lane. More explicitly "this is a
    full pose", more visual clutter than draw_pose_arrow at 4-per-panel."""
    x, y = pos
    fwd = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-np.sin(heading), np.cos(heading)])

    ax.annotate("", xy=(x + fwd[0] * axis_len, y + fwd[1] * axis_len),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=14),
                zorder=zorder)
    ax.annotate("", xy=(x + side[0] * axis_len, y + side[1] * axis_len),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=secondary_color,
                                 lw=lw, mutation_scale=14),
                zorder=zorder)
    draw_joint(ax, pos, color, radius=0.045, zorder=zorder + 1)


def draw_goal_marker(ax, pos, heading, color, ring_radius=0.11, tick_len=0.24,
                      lw=2.4, zorder=6, dim=False):
    """Oriented *goal* glyph: a solid dot at the goal position with a
    plain arrow pointing along `heading` -- the standard, immediately
    readable way to show "a point, facing this way". `dim` renders a
    desaturated gray version for "other samples" context in a panel
    that's highlighting one particular goal."""
    x, y = pos
    c = "#c7c7c7" if dim else color
    fwd = np.array([np.cos(heading), np.sin(heading)])

    base_radius = ring_radius * 0.55
    ax.add_patch(Circle((x, y), base_radius, facecolor=c, edgecolor="none",
                         zorder=zorder + 1))

    start = np.array([x, y]) + fwd * base_radius
    tip = np.array([x, y]) + fwd * (base_radius + tick_len)
    ax.annotate("", xy=tuple(tip), xytext=tuple(start),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                 mutation_scale=lw * 6),
                zorder=zorder)


def sample_pose_line(start_pose, end_pose, n):
    """Linearly interpolate n poses (x, y, heading) between start_pose and
    end_pose, inclusive. Position lerps directly; heading lerps along the
    shortest angular path (wrapped to (-pi, pi]) so a start/end pair like
    (heading=3.0, heading=-3.0) turns the short way around instead of
    sweeping almost a full turn. Stands in for SE(3) pose interpolation in
    this toy 2D figure the same way sample_line (position-only) does in
    ik_line_demo.py."""
    x0, y0, h0 = start_pose
    x1, y1, h1 = end_pose
    dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi  # shortest signed delta

    t = np.linspace(0.0, 1.0, n)
    xs = x0 * (1 - t) + x1 * t
    ys = y0 * (1 - t) + y1 * t
    hs = h0 + dh * t
    return list(zip(xs, ys, hs))


def draw_obstacles(ax, obstacles, color="#c9c9c9", edgecolor="#a8a8a8",
                    zorder=1):
    """Dummy obstacle 'spheres', drawn as filled circles. `obstacles` is a
    list of (center, radius) pairs. Shared between the task-space
    pre-check panel and the full collision-check panel so the same
    obstacles can be reused across both."""
    for center, radius in obstacles:
        ax.add_patch(Circle(center, radius, facecolor=color,
                             edgecolor=edgecolor, lw=1.2, zorder=zorder))


def point_in_collision(pos, obstacles, margin=0.0):
    """True if `pos` (x, y) falls within any (center, radius) obstacle in
    `obstacles`, inflated by `margin` (e.g. the pose glyph's own visual
    footprint). Purely a cheap point-vs-sphere check on the pose itself --
    stands in for checking a sampled SE(3) pose against the environment
    before spending an IK solve on it."""
    pos = np.asarray(pos, dtype=float)
    for center, radius in obstacles:
        if np.linalg.norm(pos - np.asarray(center, dtype=float)) <= radius + margin:
            return True
    return False


def closest_point_on_segment(p0, p1, q):
    """Closest point on segment p0-p1 to point q, plus the distance to it."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q = np.asarray(q, dtype=float)
    d = p1 - p0
    length2 = float(np.dot(d, d))
    if length2 < 1e-12:
        t = 0.0
    else:
        t = float(np.clip(np.dot(q - p0, d) / length2, 0.0, 1.0))
    closest = p0 + t * d
    return closest, float(np.linalg.norm(q - closest))


def arm_obstacle_collision(joint_points, obstacles, link_half_width=LINK_HALF_WIDTH):
    """Checks every link segment of an arm (consecutive pairs in
    `joint_points`, e.g. from draw_arm's returned pts) against every
    (center, radius) obstacle, inflated by the link's half-width -- a
    stand-in for the "resolved ambient configuration against the real
    collision checker" step, one level more faithful than
    point_in_collision's single-point check on the un-resolved pose.
    Returns the (center, radius) of the closest point along the arm that
    collides, or None if the arm is collision-free."""
    for p0, p1 in zip(joint_points[:-1], joint_points[1:]):
        for center, radius in obstacles:
            closest, distance = closest_point_on_segment(p0, p1, center)
            if distance <= radius + link_half_width:
                return closest, radius
    return None



def draw_target(ax, pos, color="#d7263d", w=0.22, h=0.55, zorder=4, rotation=0):
    """Small rectangular 'object' marker."""
    x, y = pos
    w, h = float(w), float(h)
    # apply rotation, assume it is 90 degrees for now
    if rotation != 0:
        (w, h) = (h, w)

    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.0,rounding_size=0.02",
                          linewidth=0, facecolor=color, zorder=zorder)
    ax.add_patch(box)


def draw_sphere(ax, pos, radius=0.18, color="#d7263d", zorder=4,
                 highlight_color="#ffffff", highlight_alpha=0.35):
    """Small filled-circle 'sphere' obstacle marker, with an offset
    lighter highlight ellipse to read as a shaded ball rather than a flat
    disc."""
    x, y = pos
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="none",
                         zorder=zorder))
    hl_dx, hl_dy = -0.35 * radius, 0.35 * radius
    ax.add_patch(Ellipse((x + hl_dx, y + hl_dy), width=0.6 * radius,
                          height=0.4 * radius, facecolor=highlight_color,
                          edgecolor="none", alpha=highlight_alpha,
                          zorder=zorder + 0.01))


def draw_highlight_circle(ax, center, radius=0.55, color="#d7263d",
                           lw=2.5, zorder=5):
    circ = Circle(center, radius, fill=False, edgecolor=color, lw=lw,
                  zorder=zorder)
    ax.add_patch(circ)


def draw_plane(ax, y, half_width, skew=0.25, depth=0.55, color="#b5b0a8",
               alpha=0.18, edgecolor="#6e6a63", edge_alpha=0.8, lw=1.6,
               zorder=1, shadow_color="#000000", shadow_alpha=0.05,
               shadow_offset=(0.05, -0.07), wave_amp=0.0, wave_cycles=1.4,
               n_grid_u=2, n_grid_v=9, grid_lw=0.8, grid_alpha=0.45,
               n_samples=48):
    """Constraint-manifold surface: a gently undulating (not flat) sheet
    with a wireframe grid over it, drawn as a skewed quad for pseudo-3D
    depth (as in draw_plane before) -- the standard "manifold", not
    "plane", look in constrained-planning figures. The corrugation runs
    along the width (u) and is constant in depth (v), like ridges running
    front-to-back, so it still reads as one continuous surface rather than
    a flat rectangle."""
    corners = np.array([
        [-half_width - skew, y - depth / 2],
        [half_width - skew, y - depth / 2],
        [half_width + skew, y + depth / 2],
        [-half_width + skew, y + depth / 2],
    ])
    bl, br, tr, tl = corners

    def surf(u, v):
        bottom = bl + u * (br - bl)
        top = tl + u * (tr - tl)
        p = bottom + v * (top - bottom)
        p[1] += wave_amp * np.sin(u * wave_cycles * 2 * np.pi)
        return p

    def edge_path(u_range, v_range, n=n_samples):
        us = np.atleast_1d(u_range)
        vs = np.atleast_1d(v_range)
        if len(us) == 1:
            us = np.full(n, us[0])
            vs = np.linspace(vs[0], vs[-1], n)
        else:
            us = np.linspace(us[0], us[-1], n)
            vs = np.full(n, vs[0])
        return np.array([surf(u, v) for u, v in zip(us, vs)])

    boundary = np.concatenate([
        edge_path([0, 1], [0, 0]),
        edge_path([1, 1], [0, 1]),
        edge_path([1, 0], [1, 1]),
        edge_path([0, 0], [1, 0]),
    ])

    shadow = boundary + np.array(shadow_offset)
    ax.add_patch(Polygon(shadow, closed=True, facecolor=shadow_color,
                          alpha=shadow_alpha, edgecolor="none",
                          zorder=zorder - 0.1))
    ax.add_patch(Polygon(boundary, closed=True, facecolor=color, alpha=alpha,
                          edgecolor="none", zorder=zorder))

    # for u in np.linspace(0, 1, n_grid_u + 2)[1:-1]:
    #     line = edge_path([u, u], [0, 1])
    #     ax.plot(line[:, 0], line[:, 1], color=edgecolor, alpha=grid_alpha,
    #             lw=grid_lw, zorder=zorder + 0.05)
    # for v in np.linspace(0, 1, n_grid_v + 2)[1:-1]:
    #     line = edge_path([0, 1], [v, v])
    #     ax.plot(line[:, 0], line[:, 1], color=edgecolor, alpha=grid_alpha,
    #             lw=grid_lw, zorder=zorder + 0.05)

    ax.add_patch(Polygon(boundary, closed=True, facecolor="none",
                          edgecolor=edgecolor, alpha=edge_alpha, lw=lw,
                          zorder=zorder + 0.1))


def draw_arm(ax, base, angles, lengths, color, draw_rail=True,
             draw_gripper_only=False, shaft_zorder=0.5, gripper_zorder=1.5,
             alpha_arm_body=1.0, in_collision=False, collision_point=None,
             collision_color="#e63946", collision_ring_radius=0.4,
             recolor_on_collision=True):
    """Draw a full arm: base rail, rectangular links + pivot discs,
    rectangular gripper. The shaft is drawn *below* the constraint plane's
    zorder and the gripper *above* it (see draw_plane), so the arm reads
    as reaching up from underneath and the gripper as resting on top of
    the plane's surface, rather than skewering straight through it.

    `in_collision=True` flags this pose as invalid, drawn two ways at
    once (both are standard conventions; set `recolor_on_collision=False`
    to keep just the ring):
      - the whole arm recolors to `collision_color` (a clear at-a-glance
        "this config is bad" cue), and
      - a red ring (draw_highlight_circle) is drawn at `collision_point`
        -- pass the point from arm_obstacle_collision(pts, obstacles) so
        the ring lands exactly on the offending contact, e.g.:
            pts, heading = forward_kinematics(base, angles, lengths)
            hit = arm_obstacle_collision(pts, obstacles)
            draw_arm(..., in_collision=hit is not None,
                     collision_point=hit[0] if hit else None)
        If omitted, the ring falls back to the arm's middle joint.

    Returns joint points and final heading."""
    pts, heading = forward_kinematics(base, angles, lengths)
    draw_color = collision_color if (in_collision and recolor_on_collision) else color

    draw_gripper(ax, pts[-1], heading, draw_color, zorder=gripper_zorder)
    if draw_gripper_only:
        if in_collision:
            ring_point = collision_point if collision_point is not None else pts[-1]
            draw_highlight_circle(ax, ring_point, radius=collision_ring_radius,
                                   color=collision_color, zorder=gripper_zorder + 1)
        return pts, heading
    if draw_rail:
        draw_base_rail(ax, base, draw_color, zorder=shaft_zorder)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw_rect_segment(ax, p0, p1, draw_color, LINK_HALF_WIDTH, zorder=shaft_zorder, alpha=alpha_arm_body)
    for joint_pt in pts[1:-1]:
        draw_joint(ax, joint_pt, draw_color, zorder=shaft_zorder + 0.1, alpha=alpha_arm_body)
    if draw_rail:
        draw_joint(ax, pts[0], draw_color, radius=BASE_HALF_WIDTH, zorder=shaft_zorder + 0.1, alpha=alpha_arm_body * 0.5)
    if in_collision:
        ring_point = collision_point if collision_point is not None else pts[len(pts) // 2]
        draw_highlight_circle(ax, ring_point, radius=collision_ring_radius,
                               color=collision_color, zorder=shaft_zorder + 2)
    return pts, heading


def draw_contact_shadow(ax, pos, color="#000000", rx=0.16, ry=0.055,
                         alpha=0.18, zorder=1.05):
    """Soft flattened ellipse directly under a gripper's contact point on
    the plane, reinforcing "resting on the surface" (a standard cheap
    grounding cue, borrowed from flat-shadow illustration style)."""
    ax.add_patch(Ellipse(pos, width=2 * rx, height=2 * ry, facecolor=color,
                          edgecolor="none", alpha=alpha, zorder=zorder))


def link_point(pts, link_idx, t):
    """Point a fraction `t` in [0, 1] along link `link_idx` (0 = base
    link, 1 = forearm, ...) of a `forward_kinematics` joint-point list."""
    p0 = np.asarray(pts[link_idx], dtype=float)
    p1 = np.asarray(pts[link_idx + 1], dtype=float)
    return p0 + t * (p1 - p0)


def make_example_obstacles(arm_idx=2, link_idx=1, t=0.4, side_x=1.9,
                            base=(0.0, 0.0)):
    """Two example obstacles (styled like the red boxes in the reference
    figure), taking inspiration from its "obstacle sitting near the arm"
    look:

    - `link_obstacle`: centered `t` of the way along link `link_idx` of
      arm `arm_idx` (0-indexed, so arm_idx in {1, 2, 3} = "arms 2, 3, 4").
      `t` well below 1 keeps it clear of the wrist/gripper, so it fouls
      the link but not the end-effector.
    - `plane_obstacle`: resting on top of the constraint plane, off to the
      side of the whole fan so it doesn't touch any arm.

    Returns (link_obstacle_pos, plane_obstacle_pos, arm_idx) for use with
    draw_target() and, if wanted, arm_obstacle_collision() for a sanity
    check against the actual link geometry.
    """
    angles = ARM_POSES[arm_idx]
    lengths = ARM_LENGTHS[arm_idx]
    pts, _ = forward_kinematics(base, angles, lengths)
    link_obstacle = tuple(link_point(pts, link_idx, t))
    link_obstacle = (link_obstacle[0] + 0.3, link_obstacle[1] - 0.25)
    plane_obstacle = (-0.45, TARGET_HEIGHT - 0.3)
    return link_obstacle, plane_obstacle, arm_idx


def draw_example_obstacles(ax, arm_idx=2, link_idx=1, t=0.4, side_x=1.9,
                            base=(0.0, 0.0), color="#EE0000", radius=0.18):
    """Draws the two obstacles from make_example_obstacles() as spheres
    (draw_sphere()) rather than boxes."""
    link_obstacle, plane_obstacle, _ = make_example_obstacles(
        arm_idx=arm_idx, link_idx=link_idx, t=t, side_x=side_x, base=base)
    draw_sphere(ax, link_obstacle, radius=radius, color=color, zorder=1.2)
    draw_sphere(ax, plane_obstacle, radius=radius, color=color, zorder=1.2)
    return link_obstacle, plane_obstacle


# ----------------------------------------------------------------------
# example figure: four arm poses side by side.
# Palette: ColorBrewer "Set2" -- the pastel/muted counterpart of "Dark2"
# (same 4 hues, softer saturation), for a lighter look on a white
# background.
# ----------------------------------------------------------------------
PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

LINK_LENGTHS = (1.0, 1.1)

# Full closed-form 2-link IK for four wrist targets, equally spaced in x,
# all at the same height -- so both constraints hold exactly (unlike the
# earlier attempts, which fixed theta1 or theta2 first and only satisfied
# one constraint at a time). Elbow sign is chosen per-target (mirrored
# about center) rather than forced the same for all four: sharing one
# branch made the inner poses bend awkwardly just to stay on-branch with
# the outer ones, and the discontinuity at the switch is fine to accept.
TARGET_XS = np.linspace(-1.175, 1.175, 4)
TARGET_HEIGHT = 1.8
ELBOW_SIGN = -1
# ELBOW_SIGNS = [-1 if x < 0 else 1 for x in TARGET_XS]
ELBOW_SIGNS = [-1 if x < 0 else -1 for x in TARGET_XS]


def solve_ik_xy(x, y, lengths, elbow_sign=ELBOW_SIGN):
    L1, L2 = lengths
    r2 = x * x + y * y
    c2 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        raise ValueError(
            f"target ({x}, {y}) unreachable: needs |{c2:.3f}| <= 1 "
            f"(reach is [{abs(L1 - L2)}, {L1 + L2}])")
    theta2 = elbow_sign * np.arccos(np.clip(c2, -1.0, 1.0))
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    phi1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    theta1 = phi1 - np.pi / 2
    return theta1, theta2


# The IK target is the fingertip ("eef"), not the wrist: solve as if link 2
# were longer by the stub+finger offset, then draw with the true (scaled)
# link lengths below -- same heading, so the wrist ends up short of the
# target by exactly that offset and the fingertip lands on TARGET_HEIGHT.
# The two middle poses are shrunk slightly: at equal x-spacing the middle
# targets sit closer to the base, so their un-shrunk elbows crowd together
# more than the outer pair's -- scaling them down evens out the spacing.
LENGTH_SCALES = [0.95, 0.8, 0.72, 0.88]
EEF_OFFSET = STUB_LEN + FINGER_LEN
ARM_LENGTHS = [
    (LINK_LENGTHS[0] * scale, LINK_LENGTHS[1] * scale) for scale in LENGTH_SCALES
]
ARM_POSES = [
    solve_ik_xy(x, TARGET_HEIGHT, (lengths[0], lengths[1] + EEF_OFFSET),
                elbow_sign=sign)
    for x, sign, lengths in zip(TARGET_XS, ELBOW_SIGNS, ARM_LENGTHS)
]


def draw_all():
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    base = (0.0, 0.0)

    # single shared base rail, drawn once in a neutral color underneath
    # all the overlaid poses
    draw_base_rail(ax, base, "#4d4d4d", zorder=0.5)
    draw_joint(ax, base, "#4d4d4d", radius=BASE_HALF_WIDTH, zorder=0.6)

    # Plane sits centered on the wrist height: the shaft (zorder < plane)
    # dips into the plane's depth band and gets occluded there, as if
    # reaching up through the surface from underneath, while the gripper
    # (zorder > plane) draws on top of it -- so it reads as resting on the
    # surface, fingers rising off the top, rather than skewering through.
    draw_plane(ax, TARGET_HEIGHT, half_width=1.8)

    for color, angles, lengths in zip(PALETTE, ARM_POSES, ARM_LENGTHS):
        pts, heading = draw_arm(ax, base, angles, lengths, color,
                                 draw_rail=False, shaft_zorder=0.5,
                                 gripper_zorder=1.5)
        draw_contact_shadow(ax, gripper_tip(pts[-1], heading))

    draw_example_obstacles(ax)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig("robot_arm_fig.png", dpi=300, facecolor="white")
    fig.savefig("robot_arm_fig.svg", facecolor="white")
    print("Saved robot_arm_fig.png / robot_arm_fig.svg")


def draw_start_config_end_only():
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    base = (0.0, 0.0)

    # single shared base rail, drawn once in a neutral color underneath
    # all the overlaid poses
    draw_base_rail(ax, base, "#4d4d4d", zorder=0.5)
    draw_joint(ax, base, "#4d4d4d", radius=BASE_HALF_WIDTH, zorder=0.6)

    # Plane sits centered on the wrist height: the shaft (zorder < plane)
    # dips into the plane's depth band and gets occluded there, as if
    # reaching up through the surface from underneath, while the gripper
    # (zorder > plane) draws on top of it -- so it reads as resting on the
    # surface, fingers rising off the top, rather than skewering through.
    draw_plane(ax, TARGET_HEIGHT, half_width=1.8)

    for arm_idx, (color, angles, lengths) in enumerate(zip(PALETTE, ARM_POSES, ARM_LENGTHS)):
        if arm_idx == 0:
            draw_gripper_only = False
            alpha_arm_body = 0.3

        elif arm_idx == len(PALETTE) - 1:
            draw_gripper_only = True
        else:
            continue        

        pts, heading = draw_arm(ax, base, angles, lengths, color,
                                 draw_rail=False, shaft_zorder=0.5,
                                 gripper_zorder=1.5, draw_gripper_only = draw_gripper_only, alpha_arm_body = alpha_arm_body)
        draw_contact_shadow(ax, gripper_tip(pts[-1], heading))

    draw_example_obstacles(ax)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig("robot_arm_fig.png", dpi=300, facecolor="white")
    fig.savefig("robot_arm_fig.svg", facecolor="white")
    print("Saved robot_arm_fig.png / robot_arm_fig.svg")



def draw_eefs_only():
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    base = (0.0, 0.0)

    # single shared base rail, drawn once in a neutral color underneath
    # all the overlaid poses
    draw_base_rail(ax, base, "#4d4d4d", zorder=0.5)
    draw_joint(ax, base, "#4d4d4d", radius=BASE_HALF_WIDTH, zorder=0.6)

    # Plane sits centered on the wrist height: the shaft (zorder < plane)
    # dips into the plane's depth band and gets occluded there, as if
    # reaching up through the surface from underneath, while the gripper
    # (zorder > plane) draws on top of it -- so it reads as resting on the
    # surface, fingers rising off the top, rather than skewering through.
    draw_plane(ax, TARGET_HEIGHT, half_width=1.8)

    for arm_idx, (color, angles, lengths) in enumerate(zip(PALETTE, ARM_POSES, ARM_LENGTHS)):
        if arm_idx == 0:
            draw_gripper_only = False
            alpha_arm_body = 0.3

        else:
            draw_gripper_only = True

        pts, heading = draw_arm(ax, base, angles, lengths, color,
                                 draw_rail=False, shaft_zorder=0.5,
                                 gripper_zorder=1.5, draw_gripper_only = draw_gripper_only, alpha_arm_body = alpha_arm_body)
        draw_contact_shadow(ax, gripper_tip(pts[-1], heading))

    draw_example_obstacles(ax)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig("robot_arm_fig.png", dpi=300, facecolor="white")
    fig.savefig("robot_arm_fig.svg", facecolor="white")
    print("Saved robot_arm_fig.png / robot_arm_fig.svg")


if __name__ == "__main__":
    draw_all()
    # draw_start_config_end_only()
    # draw_eefs_only()