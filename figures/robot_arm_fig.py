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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse, Polygon
import math
# ----------------------------------------------------------------------
# style constants
# ----------------------------------------------------------------------
LINK_HALF_WIDTH = 0.05       # half-width of arm link rectangles
BASE_HALF_WIDTH = 0.015       # half-width (thickness) of the base rail
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



def gripper_obstacle_collision(wrist, heading, obstacles,
                                half_width=CROSSBAR_HALF_WIDTH):
    """Like arm_obstacle_collision(), but checks only the gripper's central
    axis (wrist -> fingertip), not the arm's links -- for figures (e.g.
    draw_eefs_only()) that only render the gripper, so a collision ring
    never appears over link geometry that isn't actually drawn."""
    wrist = np.asarray(wrist, dtype=float)
    tip = gripper_tip(wrist, heading)
    for center, radius in obstacles:
        closest, distance = closest_point_on_segment(wrist, tip, center)
        if distance <= radius + half_width:
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


def draw_sphere(ax, pos, radius=0.18, color="#8a5a58", zorder=4, hatch=None,
                 edgecolor="#7a1420"):
    """Small filled-circle obstacle marker -- flat fill, no glossy
    highlight, matching the flat style used everywhere else in the
    figure (rather than reading as a shiny 3D ball). Pass `hatch` (e.g.
    "////") to distinguish an obstacle by texture rather than a second
    color."""
    ax.add_patch(Circle(pos, radius, facecolor=color,
                         edgecolor=edgecolor if hatch else "none",
                         linewidth=0.7 if hatch else 0, hatch=hatch,
                         zorder=zorder))


def draw_highlight_circle(ax, center, radius=0.55, color="#d7263d",
                           lw=2.5, zorder=5):
    circ = Circle(center, radius, fill=False, edgecolor=color, lw=lw,
                  zorder=zorder)
    ax.add_patch(circ)


def draw_success_mark(ax, center, radius=0.16, color="#2ecc71",
                       mark_color="white", lw=2.4, zorder=10):
    """Filled green circle with a white checkmark -- the "no collision
    found here" counterpart to draw_highlight_circle()'s red ring."""
    ax.add_patch(Circle(center, radius, facecolor=color, edgecolor="none",
                         zorder=zorder))
    cx, cy = center
    check = np.array([
        [cx - 0.5 * radius, cy - 0.02 * radius],
        [cx - 0.15 * radius, cy - 0.4 * radius],
        [cx + 0.55 * radius, cy + 0.35 * radius],
    ])
    ax.plot(check[:, 0], check[:, 1], color=mark_color, lw=lw,
            solid_capstyle="round", solid_joinstyle="round", zorder=zorder + 1)


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
             collision_color="#e63946", collision_ring_radius=0.15,
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


def draw_interpolation_line(ax, points, color="#555555", lw=1.6,
                             dash=(6, 3), zorder=1.06, marker_radius=0.05):
    """Dashed line through a sequence of contact points on the plane, plus
    a small dot at each -- shows the four eef poses as samples along one
    interpolated path across the manifold, rather than four unrelated
    points."""
    points = np.asarray(points, dtype=float)
    ax.plot(points[:, 0], points[:, 1], color=color, lw=lw,
            dashes=dash, zorder=zorder, solid_capstyle="round")
    for p in points:
        ax.add_patch(Circle(p, marker_radius, facecolor=color,
                             edgecolor="none", zorder=zorder + 0.01))


def link_point(pts, link_idx, t):
    """Point a fraction `t` in [0, 1] along link `link_idx` (0 = base
    link, 1 = forearm, ...) of a `forward_kinematics` joint-point list."""
    p0 = np.asarray(pts[link_idx], dtype=float)
    p1 = np.asarray(pts[link_idx + 1], dtype=float)
    return p0 + t * (p1 - p0)


def make_example_obstacles(arm_idx=2, link_idx=1, t=0.4, side_x=1.9,
                            base=(0.0, 0.0), plane_obstacle_y_offset=0.0):
    """Two example obstacles (styled like the red boxes in the reference
    figure), taking inspiration from its "obstacle sitting near the arm"
    look:

    - `link_obstacle`: centered `t` of the way along link `link_idx` of
      arm `arm_idx` (0-indexed, so arm_idx in {1, 2, 3} = "arms 2, 3, 4").
      `t` well below 1 keeps it clear of the wrist/gripper, so it fouls
      the link but not the end-effector.
    - `plane_obstacle`: resting on top of the constraint plane, off to the
      side of the whole fan so it doesn't touch any arm. Pass
      `plane_obstacle_y_offset` (e.g. a couple units) to lift it well
      clear of the plane instead, so it's visibly present but guaranteed
      collision-free.

    Returns (link_obstacle_pos, plane_obstacle_pos, arm_idx) for use with
    draw_target() and, if wanted, arm_obstacle_collision() for a sanity
    check against the actual link geometry.
    """
    angles = ARM_POSES[arm_idx]
    lengths = ARM_LENGTHS[arm_idx]
    pts, _ = forward_kinematics(base, angles, lengths)
    link_obstacle = tuple(link_point(pts, link_idx, t))
    link_obstacle = (link_obstacle[0] + -0.2, link_obstacle[1] + 0.025)
    plane_obstacle = (-0.45, TARGET_HEIGHT - 0.3 + plane_obstacle_y_offset)
    return link_obstacle, plane_obstacle, arm_idx


def draw_example_obstacles(ax, arm_idx=2, link_idx=1, t=0.4, side_x=1.9,
                            base=(0.0, 0.0), colors=None, hatches=None,
                            radius=0.18, include_plane_obstacle=True,
                            plane_obstacle_y_offset=0.0):
    """Draws the two obstacles from make_example_obstacles() as spheres
    (draw_sphere()) rather than boxes -- link and plane obstacles share
    one color (OBSTACLE_COLORS by default) and are told apart by texture
    (OBSTACLE_HATCHES) instead, so they read as distinct objects without
    needing two different colors. Set include_plane_obstacle=False to
    draw (and return) only the link obstacle."""
    colors = colors or OBSTACLE_COLORS
    hatches = hatches or OBSTACLE_HATCHES
    link_obstacle, plane_obstacle, _ = make_example_obstacles(
        arm_idx=arm_idx, link_idx=link_idx, t=t, side_x=side_x, base=base,
        plane_obstacle_y_offset=plane_obstacle_y_offset)
    draw_sphere(ax, link_obstacle, radius=radius, color=colors[0],
                hatch=hatches[0], zorder=1.2)
    if include_plane_obstacle:
        draw_sphere(ax, plane_obstacle, radius=radius, color=colors[1],
                    hatch=hatches[1], zorder=1.2)
    return link_obstacle, (plane_obstacle if include_plane_obstacle else None)


# ----------------------------------------------------------------------
# example figure: four arm poses side by side.
# Palette: ColorBrewer "Dark2" -- the same 4 hues as before but back to
# full saturation (Set2's pastel version read as too washed-out/muted).
# ----------------------------------------------------------------------
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

# Both obstacles share one pale, dusty red -- deliberately much lighter
# than the collision-highlight ring's bright "#e63946" (draw_arm's
# collision_color / draw_highlight_circle), so an obstacle and a "this
# collided" ring drawn in the same spot never read as the same red.
OBSTACLE_COLORS = ("#ec6670", "#e9505b")
OBSTACLE_HATCHES = (None, None)

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


def draw_ground_hatch(ax, base, half_len=BASE_RAIL_HALF, n=11,
                       tick_len=0.11, angle_deg=45, gap=0.0,
                       color="#4d4d4d", lw=1.0, zorder=0.4):
    """Diagonal "////" ground-hatch ticks just below the base rail -- the
    standard architectural-drawing symbol for "this is fixed to the
    ground", instead of the rail floating on bare white space."""
    x0, y0 = base
    y0 = y0 - BASE_HALF_WIDTH - gap
    dx = tick_len * np.cos(np.radians(angle_deg))
    dy = tick_len * np.sin(np.radians(angle_deg))
    for x in np.linspace(x0 - half_len, x0 + half_len, n):
        ax.plot([x, x - dx], [y0, y0 - dy], color=color, lw=lw,
                 solid_capstyle="butt", zorder=zorder)


def _draw_shared_base_and_plane(ax):
    """Base rail + pivot + constraint-manifold plane, common to every
    panel below."""
    base = (0.0, 0.0)
    draw_base_rail(ax, base, "#4d4d4d", zorder=0.5)
    draw_ground_hatch(ax, base)
    draw_joint(ax, base, "#4d4d4d", radius=BASE_HALF_WIDTH, zorder=0.6)
    # Plane sits centered on the wrist height: the shaft (zorder < plane)
    # dips into the plane's depth band and gets occluded there, as if
    # reaching up through the surface from underneath, while the gripper
    # (zorder > plane) draws on top of it -- so it reads as resting on the
    # surface, fingers rising off the top, rather than skewering through.
    draw_plane(ax, TARGET_HEIGHT, half_width=1.8)
    return base


def _finish_panel(ax, ylim=(-0.5, 2.6)):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_all_content(ax, include_plane_obstacle=True,
                      plane_obstacle_y_offset=0.0):
    """All four poses, each recolored red with a ring at its exact
    collision point if it fouls an obstacle. Set include_plane_obstacle=
    False to drop the obstacle resting on the plane (and skip checking
    against it), leaving just the link obstacle. Or, to keep it visible
    but guaranteed collision-free, leave include_plane_obstacle=True and
    pass a large plane_obstacle_y_offset to lift it well clear of the
    plane and every arm."""
    base = _draw_shared_base_and_plane(ax)

    link_obs, plane_obs, _ = make_example_obstacles(
        plane_obstacle_y_offset=plane_obstacle_y_offset)
    obstacles = [(link_obs, 0.18)]
    if include_plane_obstacle:
        obstacles.append((plane_obs, 0.18))

    eef_points = []
    for color, angles, lengths in zip(PALETTE, ARM_POSES, ARM_LENGTHS):
        pts, heading = forward_kinematics(base, angles, lengths)
        hit = arm_obstacle_collision(pts, obstacles)
        pts, heading = draw_arm(ax, base, angles, lengths, color,
                                 draw_rail=False, shaft_zorder=0.5,
                                 gripper_zorder=1.5,
                                 in_collision=hit is not None,
                                 collision_point=hit[0] if hit else None,
                                 recolor_on_collision=False)
        eef_point = gripper_tip(pts[-1], heading)
        draw_contact_shadow(ax, eef_point)
        eef_points.append(eef_point)

    # Connect the four eef contact points on the plane to show they're
    # samples along one interpolated path, not four unrelated poses.
    draw_interpolation_line(ax, eef_points)

    draw_example_obstacles(ax, include_plane_obstacle=include_plane_obstacle,
                            plane_obstacle_y_offset=plane_obstacle_y_offset)
    _finish_panel(ax)


def draw_start_config_end_only_content(ax, plane_obstacle_y_offset=0.0):
    """Arm 0 (faded, full body) and arm 3 (gripper only) -- start/end
    configuration pair."""
    base = _draw_shared_base_and_plane(ax)

    eef_points = []
    for arm_idx, (color, angles, lengths) in enumerate(zip(PALETTE, ARM_POSES, ARM_LENGTHS)):
        # Every pose's eef point feeds the interpolation line, even the
        # two not actually drawn in this view.
        all_pts, all_heading = forward_kinematics(base, angles, lengths)
        eef_points.append(gripper_tip(all_pts[-1], all_heading))

        if arm_idx == 0:
            draw_gripper_only = False
            alpha_arm_body = 0.3
        elif arm_idx == len(PALETTE) - 1:
            draw_gripper_only = True
        else:
            continue

        pts, heading = draw_arm(ax, base, angles, lengths, color,
                                 draw_rail=False, shaft_zorder=0.5,
                                 gripper_zorder=1.5, draw_gripper_only=draw_gripper_only,
                                 alpha_arm_body=alpha_arm_body)
        draw_contact_shadow(ax, gripper_tip(pts[-1], heading))

    draw_interpolation_line(ax, eef_points)
    draw_example_obstacles(ax, plane_obstacle_y_offset=plane_obstacle_y_offset)
    _finish_panel(ax)


def draw_eefs_only_content(ax, plane_obstacle_y_offset=0.0):
    """Arm 0 (faded, full body) plus every other arm's gripper only, with
    collision checked against the gripper geometry alone for the
    gripper-only poses (their links aren't drawn, so a link collision
    there would show a ring over nothing)."""
    base = _draw_shared_base_and_plane(ax)

    link_obs, plane_obs, _ = make_example_obstacles(
        plane_obstacle_y_offset=plane_obstacle_y_offset)
    obstacles = [(link_obs, 0.18), (plane_obs, 0.18)]

    eef_points = []
    for arm_idx, (color, angles, lengths) in enumerate(zip(PALETTE, ARM_POSES, ARM_LENGTHS)):
        if arm_idx == 0:
            draw_gripper_only = False
            alpha_arm_body = 0.3
        else:
            draw_gripper_only = True

        pts, heading = forward_kinematics(base, angles, lengths)
        if draw_gripper_only:
            hit = gripper_obstacle_collision(pts[-1], heading, obstacles)
        else:
            hit = arm_obstacle_collision(pts, obstacles)
        pts, heading = draw_arm(ax, base, angles, lengths, color,
                                 draw_rail=False, shaft_zorder=0.5,
                                 gripper_zorder=1.5, draw_gripper_only=draw_gripper_only,
                                 alpha_arm_body=alpha_arm_body,
                                 in_collision=hit is not None,
                                 collision_point=hit[0] if hit else None,
                                 recolor_on_collision=False)
        eef_point = gripper_tip(pts[-1], heading)
        draw_contact_shadow(ax, eef_point)
        eef_points.append(eef_point)

    draw_interpolation_line(ax, eef_points)
    draw_example_obstacles(ax, plane_obstacle_y_offset=plane_obstacle_y_offset)
    _finish_panel(ax)


def _save_fig(fig, name, skip_tight_layout=False, crop=True, **tight_layout_kwargs):
    if not skip_tight_layout:
        fig.tight_layout(**tight_layout_kwargs)
    # bbox_inches="tight" crops any leftover blank canvas around the
    # already-laid-out content -- it only trims the outer margin, it
    # doesn't touch the internal spacing set up above.
    save_kwargs = dict(facecolor="white")
    if crop:
        save_kwargs.update(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{name}.png", dpi=300, **save_kwargs)
    fig.savefig(f"{name}.svg", **save_kwargs)
    print(f"Saved {name}.png / {name}.svg")


def draw_all():
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_all_content(ax)
    _save_fig(fig, "robot_arm_fig")


def draw_start_config_end_only():
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_start_config_end_only_content(ax)
    _save_fig(fig, "robot_arm_fig")


def draw_eefs_only():
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_eefs_only_content(ax)
    _save_fig(fig, "robot_arm_fig")


def compute_eef_points(base=(0.0, 0.0)):
    """The fingertip ("eef") point of every one of the 4 arm poses, in the
    same (x, y) frame the robot renders use -- the raw task-space
    coordinate each full linkage figure ultimately reduces to."""
    points = []
    for angles, lengths in zip(ARM_POSES, ARM_LENGTHS):
        pts, heading = forward_kinematics(base, angles, lengths)
        points.append(gripper_tip(pts[-1], heading))
    return points


def rotate_points(points, angle_deg, pivot):
    """Rotate a list of (x, y) points by angle_deg (CCW) about `pivot`,
    preserving their relative arrangement -- used to tilt the abstract
    SE(3) plot's start->goal line to a chosen angle without touching the
    actual arm/plane geometry."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    pivot = np.asarray(pivot, dtype=float)
    rotated = []
    for p in points:
        v = np.asarray(p, dtype=float) - pivot
        rv = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        rotated.append(tuple(rv + pivot))
    return rotated


LABEL_FONT = "serif"  # matches a paper's body text far better than the
                       # default sans-serif for the SE(3)/Q/(a)(b)(c)/IK
                       # annotation text sprinkled through these figures.


def draw_axis_indicator(ax, origin=(-2.3, +0.9), length=2.0,
                         color="#333333", lw=2., label="SE(3)",
                         fontsize=15):
    """Small corner "L" axis glyph (two arrows from a fixed origin),
    labeled right at the origin corner -- stands in for coordinate axes
    on the abstract SE(3) plot without the clutter of full spanning
    tick-labeled axes."""
    ox, oy = origin
    ax.annotate("", xy=(ox + length, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=12), zorder=5)
    ax.annotate("", xy=(ox, oy + length), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=12), zorder=5)
    if label:
        ax.text(ox - 0.12, oy - 0.12, label, fontsize=fontsize, color=color,
                 va="top", ha="right", zorder=5, family=LABEL_FONT)


def draw_se3_content(ax, eef_points, obstacles, show_all_samples=True,
                      obstacle_colors=None, obstacle_hatches=None,
                      collision_index=None):
    """Bare x-y plot of the same task-space points/obstacles shown in the
    robot render alongside it, labeled generically as "SE(3)" rather than
    with literal x/y units -- collapsing the full linkage down to just its
    eef coordinate and the obstacles it must avoid.

    show_all_samples=False: only start (arm 0) and goal (arm 3), joined by
    a single straight line -- the naive start->goal path.
    show_all_samples=True: all 4 sampled poses along that path, joined by
    a dashed line, one dot per sample (matching PALETTE).
    """
    ax.set_facecolor("white")
    obstacle_colors = obstacle_colors or OBSTACLE_COLORS
    obstacle_hatches = obstacle_hatches or OBSTACLE_HATCHES
    for (center, radius), color, hatch in zip(obstacles, obstacle_colors, obstacle_hatches):
        ax.add_patch(Circle(center, radius, facecolor=color,
                             edgecolor="#7a1420" if hatch else "none",
                             linewidth=0.7 if hatch else 0, hatch=hatch,
                             alpha=0.9, zorder=2))

    if show_all_samples:
        xs = [p[0] for p in eef_points]
        ys = [p[1] for p in eef_points]
        ax.plot(xs, ys, color="#555555", lw=1.6, dashes=(6, 3), zorder=1)
        for color, p in zip(PALETTE, eef_points):
            ax.scatter([p[0]], [p[1]], s=120, color=color, zorder=3,
                       edgecolor="white", linewidth=1.3)
        if collision_index is not None:
            draw_highlight_circle(ax, eef_points[collision_index], radius=0.32,
                                   color="#e63946", lw=2.2, zorder=4)
    else:
        start, goal = eef_points[0], eef_points[-1]
        ax.plot([start[0], goal[0]], [start[1], goal[1]], color="#555555",
                lw=1.8, zorder=1)
        ax.scatter([start[0]], [start[1]], s=140, color=PALETTE[0], zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.scatter([goal[0]], [goal[1]], s=140, color=PALETTE[-1], zorder=3,
                   edgecolor="white", linewidth=1.4)

    # Autoscale to the (possibly rotated) data rather than reusing the
    # physical figure's fixed limits, which no longer bound a tilted line.
    pad = 0.5
    xs_all = [p[0] for p in eef_points] + [c[0] for c, _ in obstacles]
    ys_all = [p[1] for p in eef_points] + [c[1] for c, _ in obstacles]
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
    ax.set_ylim(min(ys_all) - pad - 0.9, max(ys_all) + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_axis_indicator(ax, origin=(min(xs_all) - pad + 0.25, min(ys_all) - pad + 0.1))


def catmull_rom(points, n_per_seg=30):
    """Smooth curve passing through every point in `points` (a Catmull-Rom
    spline) -- for a "curved interpolated path" that still hits each
    sample exactly, instead of the straight/dashed polyline used for the
    task-space SE(3) panels."""
    pts = np.asarray(points, dtype=float)
    pts_ext = np.vstack([pts[0], pts, pts[-1]])
    curve = []
    for i in range(1, len(pts_ext) - 2):
        p0, p1, p2, p3 = pts_ext[i - 1], pts_ext[i], pts_ext[i + 1], pts_ext[i + 2]
        for t in np.linspace(0.0, 1.0, n_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            point = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                            (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            curve.append(point)
    curve.append(pts[-1])
    return np.array(curve)


def blob_points(center, base_radius, n=80, harmonics=(2, 3, 4),
                 amp=(0.3, 0.18, 0.1), seed=0):
    """Points tracing a smooth, irregular closed blob around `center` --
    radius(theta) as a sum of a few random-phase harmonics, so no spline
    library is needed to keep it organic-looking rather than a polygon
    with visible corners."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=len(harmonics))
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(thetas)
    for k, a, phi in zip(harmonics, amp, phases):
        r = r + a * np.cos(k * thetas + phi)
    r = base_radius * r
    xs = center[0] + r * np.cos(thetas)
    ys = center[1] + r * np.sin(thetas)
    return np.column_stack([xs, ys])


def draw_c_obstacle(ax, center, base_radius=0.18, inflate=1.6,
                     color="#EE0000", alpha=0.4, edge_alpha=0.9, lw=1.8,
                     seed=0, zorder=2, label=r"$\mathcal{C}_{obs}$",
                     fontsize=12, label_color="#333333", hatch=None):
    """A configuration-space obstacle (C-obstacle): an irregular blob,
    inflated relative to the task-space sphere it corresponds to (C-space
    obstacles are the Minkowski sum of the workspace obstacle with the
    robot, so they're always at least as large and rarely still round),
    labeled C_obs rather than drawn as a plain sphere. Pass `hatch` to
    distinguish one obstacle from another by texture instead of color."""
    pts = blob_points(center, base_radius * inflate, seed=seed)
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=alpha,
                          edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Polygon(pts, closed=True, facecolor="none",
                          edgecolor=color, alpha=edge_alpha, lw=lw,
                          hatch=hatch, zorder=zorder + 0.05))
    if label:
        ax.text(center[0], center[1], label, fontsize=fontsize,
                color=label_color, ha="center", va="center", zorder=zorder + 1)


def bend_path(points, amount=0.6, cycles=1.0):
    """Offset each point perpendicular to the straight start->goal line by
    amount*sin(2*pi*cycles*t) (t = 0 at start, 1 at goal) -- a genuine
    bend, since q-space points sampled from an SE(3)-straight-line
    interpolation are themselves collinear and a spline through them is
    just that same straight line.

    cycles=1.0 (the default) swings to one side and back past center to
    the other side -- non-monotonic, like a mirrored "Z" -- rather than
    cycles=0.5's single one-sided hump."""
    points = np.asarray(points, dtype=float)
    start, goal = points[0], points[-1]
    direction = goal - start
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return [tuple(p) for p in points]
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    n = len(points)
    bent = [p + normal * (amount * np.sin(2 * np.pi * cycles * i / (n - 1)))
            for i, p in enumerate(points)]
    return [tuple(p) for p in bent]


def draw_qspace_content(ax, eef_points, obstacles, path_color="#555555",
                         bend_amount=0.7, collision_index=None,
                         colliding_obstacle_index=None, obstacle_colors=None,
                         obstacle_hatches=None):
    """The q-space counterpart of draw_se3_content()'s "all samples" panel:
    a smooth curved interpolated path (catmull_rom, bent off the straight
    line via bend_path) through the same 4 sampled configurations,
    threading between C-obstacles (blobs) instead of a straight/dashed
    line past spherical ones.

    Since this is an illustrative 2D stand-in (a link collision has no
    literal position in a "1 point per sample" plot), the actual collision
    is made unambiguous rather than left to whatever the generic bend
    happens to produce: pass `collision_index` (which sample) and
    `colliding_obstacle_index` (which entry in `obstacles` it hit) and the
    blob for that obstacle is snapped onto that sample's point, while any
    *other* obstacle found sitting too close to a (non-colliding) sample
    is pushed clear -- so only the true collision reads as touching."""
    ax.set_facecolor("white")
    obstacle_colors = obstacle_colors or OBSTACLE_COLORS
    obstacle_hatches = obstacle_hatches or OBSTACLE_HATCHES

    bent_points = bend_path(eef_points, amount=bend_amount)
    bent_points_arr = np.asarray(bent_points)

    centers = [np.asarray(c, dtype=float) for c, _ in obstacles]
    radii = [r for _, r in obstacles]
    if collision_index is not None and colliding_obstacle_index is not None:
        centers[colliding_obstacle_index] = bent_points_arr[collision_index].copy()
    for j, center in enumerate(centers):
        if j == colliding_obstacle_index:
            continue
        for k, p in enumerate(bent_points_arr):
            if k == collision_index:
                continue
            min_clear = radii[j] * 1.6 + 0.35
            offset = center - p
            dist = np.linalg.norm(offset)
            if dist < min_clear:
                direction = offset / dist if dist > 1e-6 else np.array([0.0, 1.0])
                centers[j] = p + direction * min_clear
    
    # print(centers, radii)
    centers[1] = [-1.3075138205,  3.016659015]
    for i, (center, radius) in enumerate(zip(centers, radii)):
        draw_c_obstacle(ax, center, base_radius=radius, seed=i,
                         color=obstacle_colors[i], hatch=obstacle_hatches[i])

    curve = catmull_rom(bent_points, n_per_seg=40)
    ax.plot(curve[:, 0], curve[:, 1], color=path_color, lw=2.0, zorder=1)
    for color, p in zip(PALETTE, bent_points):
        ax.scatter([p[0]], [p[1]], s=120, color=color, zorder=3,
                   edgecolor="white", linewidth=1.3)
    if collision_index is not None:
        draw_highlight_circle(ax, bent_points[collision_index], radius=0.18,
                               color="#e63946", lw=2.2, zorder=4)

    pad = 0.7
    xs_all = ([p[0] for p in bent_points] + [c[0] for c in centers] +
              list(curve[:, 0]))
    ys_all = ([p[1] for p in bent_points] + [c[1] for c in centers] +
              list(curve[:, 1]))
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
    ax.set_ylim(min(ys_all) - pad - 0.9, max(ys_all) + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_axis_indicator(ax, origin=(min(xs_all) - pad + 0.25, min(ys_all) - pad + 0.1),
                         label="Q")


def draw_composite_start_and_eefs():
    """Image 1, as a 2x2 grid matching the layout/sizing used for image 2:
    top row is the two robot renders (draw_start_config_end_only(),
    draw_eefs_only()); bottom row is each one's SE(3) plot directly below
    it (straight start->goal line, then all 4 samples). Top-row panels
    are sized to match the original 3x2 grid's reference panel size
    exactly; the bottom row is only slightly smaller and sits close
    beneath it; (a)/(b) labels sit under each column."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2),
                              gridspec_kw={"height_ratios": [1.0, 0.75],
                                           "wspace": -0.15, "hspace": -0.1})
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    for ax in axes[0, :]:
        ax.set_anchor("S")
    for ax in axes[1, :]:
        ax.set_anchor("N")

    # Plane obstacle nudged up (per earlier request), consistently in
    # both the physical panels and the abstract SE(3) ones.
    plane_obstacle_y_offset = 0.45
    draw_start_config_end_only_content(axes[0, 0], plane_obstacle_y_offset=plane_obstacle_y_offset)
    draw_eefs_only_content(axes[0, 1], plane_obstacle_y_offset=plane_obstacle_y_offset)

    eef_points = compute_eef_points()
    link_obs, plane_obs, _ = make_example_obstacles(plane_obstacle_y_offset=plane_obstacle_y_offset)
    obstacle_radius = 0.18

    # Tilt the abstract SE(3) start->goal line to ~40 degrees (rather than
    # the near-horizontal angle it inherits from the arms' shared height),
    # rotating the obstacles along with it about the start point so their
    # placement relative to the path is unchanged.
    start, goal = eef_points[0], eef_points[-1]
    current_angle = np.degrees(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    delta = 40.0 - current_angle
    eef_points = rotate_points(eef_points, delta, pivot=start)
    link_obs, plane_obs = rotate_points([link_obs, plane_obs], delta, pivot=start)
    obstacles = [(link_obs, obstacle_radius), (plane_obs, obstacle_radius)]

    draw_se3_content(axes[1, 0], eef_points, obstacles, show_all_samples=False)
    draw_se3_content(axes[1, 1], eef_points, obstacles, show_all_samples=True)
    for ax in axes[1, :]:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top - 0.35)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.09,
                         wspace=0.204, hspace=0.03)
    fig.canvas.draw()

    col_letters = "ab"
    bottoms = [axes[1, c].get_position().y0 for c in range(2)]
    label_y = min(bottoms) + 0.05
    for c, letter in enumerate(col_letters):
        top_box = axes[0, c].get_position()
        x_center = (top_box.x0 + top_box.x1) / 2.0
        fig.text(x_center, label_y, f"({letter})", fontsize=16,
                  color="#333333", ha="center", va="top", zorder=10,
                  family=LABEL_FONT)

    _save_fig(fig, "robot_arm_fig_composite_start_eefs", skip_tight_layout=True)


def draw_composite_start_eefs_all_no_plane_obstacle():
    """Image 2, as a 2x3 grid: top row is each abstract plot (SE(3)
    straight line, SE(3) all 4 samples, Q-space curved path + C-obstacle
    blobs); bottom row is the two robot renders directly below it
    (draw_eefs_only(), draw_all() -- with the plane obstacle lifted well
    clear of the plane and every arm, still visible but guaranteed
    collision-free). An "IK" arrow sits between the eefs/all columns,
    marking that transition as where the SE(3) samples actually get
    resolved into q via inverse kinematics."""
    # Top row (abstract SE(3)/Q plots) only slightly smaller than the
    # bottom row, and pulled in tight above it -- a supporting note,
    # not a second equally-weighted row, but still clearly legible.
    # figsize/wspace tuned so each bottom-row panel comes out at exactly
    # the same physical size (~5.06in x 3.14in) as the original 3x2
    # grid's robot-render column -- bottom row must NOT shrink from that
    # reference, only the top row is allowed to be slightly smaller.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2),
                              gridspec_kw={"height_ratios": [0.75, 1.0],
                                           "wspace": -0.15, "hspace": -0.25})
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    # Anchor top row to the bottom and bottom row to the top of their
    # respective cells: equal-aspect axes otherwise center their (much
    # smaller than the cell) actual box, which is what was leaving a big
    # visible gap between the two rows no matter how small hspace got.
    for ax in axes[0, :]:
        ax.set_anchor("S")
    for ax in axes[1, :]:
        ax.set_anchor("N")

    # draw_start_config_end_only_content(axes[1, 0], plane_obstacle_y_offset=0.75)
    draw_eefs_only_content(axes[1, 0], plane_obstacle_y_offset=0.65)
    # Column (b) checks each gripper against the obstacles and finds none
    # colliding -- a green checkmark flags that success, the same way
    # column (c)'s red ring flags the failure it does find.
    # draw_success_mark(axes[1, 1], (0.8, 0.65))
    draw_all_content(axes[1, 1], plane_obstacle_y_offset=0.65)
    # (Deliberately NOT trimming row 1's ylim here: with equal aspect,
    # shrinking the data y-range shrinks the rendered box too -- that's
    # what was quietly shrinking the bottom row below the reference size.
    # The small blank margin below the base rail is left as-is instead.)

    eef_points = compute_eef_points()
    # Same obstacles the top-row draw_all_content() panel actually uses
    # (plane obstacle lifted clear via the same offset), not the unlifted
    # default -- otherwise the abstract plots silently disagree with what
    # the robot render shows.
    link_obs, plane_obs, _ = make_example_obstacles(plane_obstacle_y_offset=0.75)
    obstacle_radius = 0.18

    # Which sample actually collides (against these same obstacles), so
    # the abstract plots can flag it the same way the robot render does.
    base = (0.0, 0.0)
    real_obstacles = [(link_obs, obstacle_radius), (plane_obs, obstacle_radius)]
    collision_index = None
    colliding_obstacle_index = None
    for i, (angles, lengths) in enumerate(zip(ARM_POSES, ARM_LENGTHS)):
        pts, heading = forward_kinematics(base, angles, lengths)
        for j, obs in enumerate(real_obstacles):
            if arm_obstacle_collision(pts, [obs]) is not None:
                collision_index, colliding_obstacle_index = i, j
                break
        if collision_index is not None:
            break

    start, goal = eef_points[0], eef_points[-1]
    current_angle = np.degrees(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    delta = 40.0 - current_angle
    eef_points = rotate_points(eef_points, delta, pivot=start)
    link_obs, plane_obs = rotate_points([link_obs, plane_obs], delta, pivot=start)
    obstacles = [(link_obs, obstacle_radius), (plane_obs, obstacle_radius)]

    # draw_se3_content(axes[0, 0], eef_points, obstacles, show_all_samples=False)
    # No collision_index here: this SE(3) panel is just the raw samples --
    # at this stage nothing has checked them against obstacles yet, so it
    # shouldn't presuppose which one (if any) turns out to collide.
    draw_se3_content(axes[0, 0], eef_points, obstacles, show_all_samples=True)
    draw_qspace_content(axes[0, 1], eef_points, obstacles,
                         collision_index=collision_index,
                         colliding_obstacle_index=colliding_obstacle_index)
    # Trim the blank margin above the topmost point in each top-row
    # panel -- with anchor="S" this dead space was sitting right at the
    # seam with row 1, same issue as the row-1 trim above.
    for ax in axes[0, :]:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top - 0.35)

    # Exact margins (not tight_layout, which would recompute its own and
    # drift away from the size match above) -- solved so column width
    # comes out to the reference ~5.06in and rows sit close together.
    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.09,
                         wspace=0.08, hspace=0.03)
    # "IK" arrow between the eefs (col 1) and all (col 2) columns, at the
    # bottom row's vertical level -- placed in figure fraction coordinates
    # after a draw pass so it lines up with the actual (post-aspect-
    # shrink) axes positions rather than their nominal cells.
    fig.canvas.draw()
    box_eefs = axes[1, 0].get_position()
    box_all = axes[1, 1].get_position()
    # x from the plot row (row 0): the robot-render row's equal-aspect
    # boxes overlap in x by design (negative wspace, narrower actual
    # content than their nominal cells), so box_eefs.x1/box_all.x0 can
    # cross and flip the arrow's direction -- the plot row's boxes don't
    # overlap and its columns line up with the same columns below.
    box_se3 = axes[0, 0].get_position()
    box_qspace = axes[0, 1].get_position()
    y_mid = (box_eefs.y0 + box_eefs.y1) / 2.0 + 0.25
    x_start = box_se3.x1 + 0.012
    x_end = box_qspace.x0 - 0.012
    # Bold *outlined* chevron (hollow, not a solid-filled blob) with "IK"
    # sitting above it in the same serif used for the other labels.
    arrow = FancyArrowPatch((x_start, y_mid), (x_end, y_mid),
                             transform=fig.transFigure,
                             arrowstyle="-|>", mutation_scale=32,
                             lw=3.2, color="#333333", zorder=10)
    fig.add_artist(arrow)
    fig.text((x_start + x_end) / 2.0, y_mid - 0.05, "IK", fontsize=17,
             color="#333333", ha="center", va="bottom", zorder=11,
             family=LABEL_FONT, fontweight="bold")

    # (a)/(b)/(c) under each column.
    col_letters = "ab"
    bottoms = [axes[1, c].get_position().y0 for c in range(2)]
    label_y = min(bottoms) + 0.05
    for c, letter in enumerate(col_letters):
        top_box = axes[0, c].get_position()
        x_center = (top_box.x0 + top_box.x1) / 2.0
        fig.text(x_center, label_y, f"({letter})", fontsize=16,
                  color="#333333", ha="center", va="top", zorder=10,
                  family=LABEL_FONT)

    _save_fig(fig, "robot_arm_fig_composite_start_eefs_all", skip_tight_layout=True)


def build_side_by_side_svg(
        path_a="robot_arm_fig_composite_start_eefs.png",
        path_b="robot_arm_fig_composite_start_eefs_all.png",
        out_path="robot_arm_fig_composite_side_by_side.svg",
        label_a="Case 1: SE(3) samples, collision in P-space",
        label_b="Case 2: SE(3) -> IK -> Q-space, collision found",
        color_a="#2a6f97", color_b="#bb4d00",
        pad=5, gap=10, title_h=40, target_h=520):
    """Places the two already-rendered composite PNGs side by side inside
    their own bordered, labeled box (one per "case"), as a single SVG --
    a lightweight way to visually distinguish the two without redoing
    either composite as vector subplots."""
    import base64
    from PIL import Image

    im_a = Image.open(path_a)
    im_b = Image.open(path_b)
    wa, ha = im_a.size
    wb, hb = im_b.size
    scale_a = target_h / ha
    scale_b = target_h / hb
    wa_s, ha_s = wa * scale_a, ha * scale_a
    wb_s, hb_s = wb * scale_b, hb * scale_b

    box_a_w = wa_s + 2 * pad
    box_b_w = wb_s + 2 * pad
    box_h = max(ha_s, hb_s) + 2 * pad + title_h

    total_w = box_a_w + gap + box_b_w + 2 * pad
    total_h = box_h + 2 * pad

    def _b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    b64_a, b64_b = _b64(path_a), _b64(path_b)

    x_a = pad
    x_b = x_a + box_a_w + gap
    y_box = pad

    img_a_x = x_a + pad
    img_a_y = y_box + title_h + pad + (max(ha_s, hb_s) - ha_s) / 2
    img_b_x = x_b + pad
    img_b_y = y_box + title_h + pad + (max(ha_s, hb_s) - hb_s) / 2

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{total_w:.1f}" height="{total_h:.1f}" viewBox="0 0 {total_w:.1f} {total_h:.1f}">
  <rect width="100%" height="100%" fill="white"/>
  <rect x="{x_a:.1f}" y="{y_box:.1f}" width="{box_a_w:.1f}" height="{box_h:.1f}" rx="14" ry="14" fill="none" stroke="{color_a}" stroke-width="3"/>
  <text x="{x_a + box_a_w / 2:.1f}" y="{y_box + title_h / 2 + 6:.1f}" font-size="20" font-family="sans-serif" fill="{color_a}" text-anchor="middle" font-weight="600">{label_a}</text>
  <image x="{img_a_x:.1f}" y="{img_a_y:.1f}" width="{wa_s:.1f}" height="{ha_s:.1f}" xlink:href="data:image/png;base64,{b64_a}" href="data:image/png;base64,{b64_a}"/>

  <rect x="{x_b:.1f}" y="{y_box:.1f}" width="{box_b_w:.1f}" height="{box_h:.1f}" rx="14" ry="14" fill="none" stroke="{color_b}" stroke-width="3"/>
  <text x="{x_b + box_b_w / 2:.1f}" y="{y_box + title_h / 2 + 6:.1f}" font-size="20" font-family="sans-serif" fill="{color_b}" text-anchor="middle" font-weight="600">{label_b}</text>
  <image x="{img_b_x:.1f}" y="{img_b_y:.1f}" width="{wb_s:.1f}" height="{hb_s:.1f}" xlink:href="data:image/png;base64,{b64_b}" href="data:image/png;base64,{b64_b}"/>
</svg>'''
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    draw_composite_start_and_eefs()
    draw_composite_start_eefs_all_no_plane_obstacle()
    build_side_by_side_svg()