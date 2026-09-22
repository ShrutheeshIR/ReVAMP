"""Stage E/F: draw calibrated overlays on the real footage.

--montage            12 spread frames with skeleton + trace -> scratch/
--frames T [T ...]   like --montage but at explicit video times
--segment T0 T1      render the annotated highlight cut -> out/highlight_4k.mp4

Overlay content (all 3D in the robot base frame, projected via camera.json):
  * executed marker-tip trace (fr3_tip FK), recent tail bright, history faded
  * current goal marker (from planning_queries.jsonl)
  * replan side panel: solve time, iterations; "goal blocked" state styled
    distinctly; obstacle rings flashed at each replan ("what the planner saw")
  * live constraint z-error readout from the log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from calibrate import FrameTrajectory  # noqa: E402

# Paper palette (preamble.tex), as BGR.
REVAMP_BLUE = (203, 160, 141)    # #8da0cb
HIGHLIGHT = (225, 105, 65)       # #4169E1 rblue
WARN_ORANGE = (0, 143, 209)      # #D18F00
WHITE = (245, 245, 245)

TRACE_FRAME = "fr3_tip"
TAIL_S = 10.0         # bright tail duration
MID_S = 40.0          # mid-fade band
HIST_S = 75.0         # older than this is dropped entirely
OBSTACLE_FLASH_S = 1.8
FONT = cv2.FONT_HERSHEY_DUPLEX

# Experimental --tree overlay: the whole RRT-connect explored tree for the
# active query row (planned_trajectory_info/trees/NNN.npz). A brighter
# magenta ((255,60,220) @ alpha=0.55/width=3) was tried first and was too
# loud -- competed with the actual trace/plan instead of sitting behind
# them. Back to dark/muted (darker than the very first (226,43,138)
# @0.25/2, not brighter) -- the fix for "hard to see lines" goes to the
# trace/plan widths below instead, not to making the tree loud.
TREE_COLOR = (110, 15, 75)    # dark plum, BGR
TREE_ALPHA = 0.30
TREE_WIDTH = 2

# Fixed narrative callouts: NOT tied to replan detection at all -- specific,
# hand-picked moments (see CALLOUTS below), each frozen with its own
# on-screen caption, regardless of what slow_windows()/--speedup would
# otherwise do. The first three were found by taking the nearest ACTUAL
# replan (goal-repeat, i.e. the current plan was invalidated -- see
# slow_windows()'s docstring) to each of Tommy's approximate times
# (30s/130s/250s), then eyeballing the annotated frame at each to confirm
# it matches the described moment before committing to the exact time.
#
# 4th entry (ghost_row set): a genuine "goal blocked" query -- draw_panel()
# already shows "goal blocked - replanning..." for every unsolved row during
# ordinary playback, but this one additionally freezes and ghost-renders the
# planner's own closest-approach-to-goal configuration (from
# planned_trajectory_info/trees/<row>.npz's closest_to_goal_* keys -- see
# that directory's README), so the "still couldn't connect" is something you
# SEE, not just a caption. Picked row 67 out of the 11 unsolved rows that
# fall inside this 10-300s segment (rows 41-43, 58-65, 67): it has by far the
# smallest closest_to_goal_distance (0.26 m vs. 0.36-0.74 m for the others)
# -- the most dramatic "so close, yet still no valid connection" moment, and
# it doesn't sit within CALLOUT_LEAD_S+CALLOUT_HOLD_S of any other callout.
GOAL_BLOCKED_ROW = 67
_gb_tree = np.load(os.path.join(common.TREE_DIR, f"{GOAL_BLOCKED_ROW:03d}.npz"))
GOAL_BLOCKED_Q = _gb_tree["closest_to_goal_ambient_q"].astype(float)
GOAL_BLOCKED_DIST_CM = float(_gb_tree["closest_to_goal_distance"]) * 100.0

# Each entry: (t, text, ghost_q, arrow) -- arrow is ((tx, ty), (dx, dy)):
# a big on-screen arrow whose TIP is at 4K pixel (tx, ty) and whose tail
# sits at (tx+dx, ty+dy), drawn only during the freeze (same fade as the
# caption). Per Tommy: a frozen frame alone doesn't tell the viewer WHERE
# to look -- point at the part of interest outright. Coordinates are
# hand-picked off each callout's own frozen frame (stable, since the
# freeze pins an exact source frame).
#
# Times: each callout time is chosen so the FREEZE (at t - CALLOUT_LEAD_S)
# lands just AFTER its replan has fired -- panel showing the replan's
# stats, the NEW plan drawn, the obstacle in place -- matched frame-by-
# frame against Tommy's two reference screenshots (2026-09-21):
#   1st: freeze 34.8, right after the 409 ms / 85,917-iteration replan at
#        33.07 (row 5) -- hand releasing the just-placed piece. (A freeze
#        BEFORE the replan, mid-placement, showed a moment where nothing
#        was blocked yet -- rejected twice.)
#   2nd: freeze 248.2, right after the 145 ms / 37,128-iteration replan
#        at 246.80 (row 49) -- stick pressed into the arm, big
#        reconfiguration visible. (An "obstacle at the elbow" callout at
#        213.82 sat between these two and was dropped per Tommy.)
#
# Arrow targets point at the CAUSE -- the obstacle (the placed piece, the
# marker spheres on the stick) -- not at the robot part it affects
# (per Tommy).
CALLOUTS = [
    (36.8, "Path blocked by obstacle - replanning", None,
     ((1710, 1235), (-520, 420))),
    (250.2, "Elbow obstacle forces significant reconfiguration and a new maze path",
     None, ((1186, 518), (420, 460))),
    (289.05, f"Goal blocked: closest config (ghost) still "
             f"{GOAL_BLOCKED_DIST_CM:.0f} cm short of goal",
     GOAL_BLOCKED_Q, ((1150, 950), (560, 440))),
]
# Was a 1x SLOWDOWN window (playing on through at 1x). Per Tommy: freeze
# the frame outright for CALLOUT_HOLD_S seconds instead -- the moment
# holds still so the caption is actually readable rather than competing
# with continued motion -- then resume normal (sped-up) playback from
# the SAME source instant, not further ahead. See render_segment()'s
# freeze injection.
CALLOUT_HOLD_S = 3.0
# Freeze starts this many seconds BEFORE the exact replan timestamp, not
# ON it -- per Tommy, all three callouts get a 2s lead-in so the frozen
# frame shows the moment building up to the event rather than already
# past it.
CALLOUT_LEAD_S = 2.0
CALLOUT_FADE_FRAC = 0.4   # last 40% of the hold fades the caption out
# Bigger than the first pass (2.2) since it's now a still frame, not
# competing with motion -- checked the longest of the three CALLOUTS
# strings still clears the top-left replan panel at this scale.
CALLOUT_FONT_SCALE = 2.8
CALLOUT_THICKNESS = 6
CALLOUT_OUTLINE_THICKNESS = 13
CALLOUT_COLOR = (245, 245, 245)          # BGR white
CALLOUT_BG = (20, 20, 20)


def draw_callout(img, text, alpha=1.0):
    """Top-right caption bar, solid background, big black-outline-then-
    white-fill text (same convention as the "replanned in X ms" panel),
    alpha-blended onto `img` in place so the hold's last CALLOUT_FADE_FRAC
    can fade it out smoothly instead of popping off."""
    h_img, w_img = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, CALLOUT_FONT_SCALE,
                                  CALLOUT_THICKNESS)
    pad = 34
    margin = 40
    x1 = w_img - margin
    x0 = x1 - tw - 2 * pad
    y0 = margin
    y1 = y0 + th + 2 * pad
    layer = img.copy()
    cv2.rectangle(layer, (x0, y0), (x1, y1), CALLOUT_BG, -1)
    cv2.rectangle(layer, (x0, y0), (x1, y1), CALLOUT_COLOR, 3)
    tx, ty = (x0 + x1 - tw) // 2, y1 - pad
    cv2.putText(layer, text, (tx, ty), FONT, CALLOUT_FONT_SCALE, (0, 0, 0),
               CALLOUT_OUTLINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(layer, text, (tx, ty), FONT, CALLOUT_FONT_SCALE, CALLOUT_COLOR,
               CALLOUT_THICKNESS, cv2.LINE_AA)
    if alpha >= 1.0:
        img[:] = layer
    else:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


# The wand is physically IN FRONT of the goal-blocked ghost, so the real
# wand pixels are painted back OVER the composited ghost (per Tommy:
# "composite the wand out, then redraw it over the ghost"). 3D occlusion
# from the tracked spheres was tried first and failed: the belief
# snapshot is a step function, so at the freeze instant the spheres
# describe where the wand WAS seconds earlier, not where it visibly is.
# A 2D mask is exact and stable because the freeze pins one fixed source
# frame (t = CALLOUTS[-1] - CALLOUT_LEAD_S). Traced by hand off that
# frame: a thick polyline down the shaft, plus circles for the three
# OptiTrack marker balls (part of the stick -- per Tommy they must
# occlude too) and the blue tape wrap.
WAND_REPAINT_4K = {
    "polyline": [(1740, 20), (1550, 180), (1320, 530), (1200, 800),
                 (940, 1160), (700, 1410), (300, 1880), (120, 2090)],
    "width": 120,
    "circles": [(868, 1106, 52), (1320, 488, 42), (1614, 124, 42),
                (945, 1220, 130)],
}


def wand_repaint_mask(shape):
    """uint8 mask of the wand (shaft + marker balls + tape) in the
    goal-blocked freeze frame, from WAND_REPAINT_4K."""
    mask = np.zeros(shape[:2], np.uint8)
    pts = np.array(WAND_REPAINT_4K["polyline"], np.int32)
    cv2.polylines(mask, [pts], False, 255, WAND_REPAINT_4K["width"],
                  cv2.LINE_AA)
    for x, y, r in WAND_REPAINT_4K["circles"]:
        cv2.circle(mask, (x, y), r, 255, -1, cv2.LINE_AA)
    return mask


def draw_callout_arrow(img, arrow, alpha=1.0):
    """Big black-outline-then-white-fill arrow pointing AT the part of
    interest during a callout freeze (same fade as the caption). `arrow`
    is ((tx, ty), (dx, dy)): tip at 4K pixel (tx, ty), tail at
    (tx+dx, ty+dy). Frozen frames don't tell the viewer where to look on
    their own -- the arrow does (per Tommy)."""
    if arrow is None:
        return
    (tx, ty), (dx, dy) = arrow
    tail, tip = (tx + dx, ty + dy), (tx, ty)
    layer = img.copy()
    cv2.arrowedLine(layer, tail, tip, (0, 0, 0), 44, cv2.LINE_AA,
                    tipLength=0.32)
    cv2.arrowedLine(layer, tail, tip, CALLOUT_COLOR, 24, cv2.LINE_AA,
                    tipLength=0.32)
    if alpha >= 1.0:
        img[:] = layer
    else:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def callout_hold_alphas(fps):
    """Per-frame alpha for one CALLOUT_HOLD_S-second freeze: full opacity
    for the first (1 - CALLOUT_FADE_FRAC) of it, then a linear fade to 0
    over the rest, so the caption eases out instead of cutting."""
    n = max(1, round(CALLOUT_HOLD_S * fps))
    alphas = []
    for i in range(n):
        frac = i / max(1, n - 1)
        hold_frac = 1.0 - CALLOUT_FADE_FRAC
        if frac <= hold_frac:
            alphas.append(1.0)
        else:
            alphas.append(max(0.0, 1.0 - (frac - hold_frac) / CALLOUT_FADE_FRAC))
    return alphas


# Playback-speed badge, top-center (top-left is the replan panel, top-right
# is the callout caption -- center is the one place free in every layout).
# Shown on every frame so the viewer always knows whether they're looking
# at real-time or sped-up footage, not just during --callouts.
SPEED_FONT_SCALE = 1.6
SPEED_THICKNESS = 4
SPEED_OUTLINE_THICKNESS = 9


def draw_speed_badge(img, speedup):
    text = f"{speedup}x" if speedup and speedup > 1 else "1x"
    h_img, w_img = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, SPEED_FONT_SCALE, SPEED_THICKNESS)
    tx, ty = (w_img - tw) // 2, 30 + th
    cv2.putText(img, text, (tx, ty), FONT, SPEED_FONT_SCALE, (0, 0, 0),
               SPEED_OUTLINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), FONT, SPEED_FONT_SCALE, (245, 245, 245),
               SPEED_THICKNESS, cv2.LINE_AA)

# Live-plan overlay (maze_expt_logs trajectories): the remaining portion of
# the active plan, ahead of the robot, dashed so it reads as intent rather
# than executed path. Green = "path ahead is clear"; orange already means
# obstacle/blocked and both blues belong to the executed trace.
PLAN_COLOR = (90, 205, 60)
PLAN_REFRESH_S = 0.1       # ~10 Hz update tick, per Tommy
PLAN_ALPHA = 0.75
PLAN_FLASH_S = 0.6         # full-bright right after a replan lands
PLAN_WIDTH = 9             # was 6 -- thickened along with the trace bands
PLAN_SEARCH_AHEAD = 250    # waypoints scanned per tick to advance progress
PLAN_FULL_ALPHA = 0.35     # faint, so the bright remaining-ahead dash reads
PLAN_FULL_WIDTH = 3        # was 2


def dashed_polyline(img, uv, color, width, dash=34, gap=22):
    """polylines() but dashed, following the (dense) point chain in uv."""
    if len(uv) < 2:
        return
    d = np.linalg.norm(np.diff(uv.astype(float), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    on = (s % (dash + gap)) < dash
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 1:
                cv2.polylines(img, [uv[start:i + 1]], False, color, width,
                              cv2.LINE_AA)
            start = None
    if start is not None and len(uv) - start >= 2:
        cv2.polylines(img, [uv[start:]], False, color, width, cv2.LINE_AA)


class PlanPaths:
    """Marker-tip positions of every replayed planned trajectory.

    Plans are (n,7) joint-space waypoint arrays; the overlay wants the same
    physical point the executed trace uses (fr3_tip, the pen tip on the
    maze), so each waypoint goes through FK once. ~25k FK calls total,
    cached in scratch keyed by the trajectory directory's content.
    """

    def __init__(self, arm):
        import glob
        files = sorted(glob.glob(os.path.join(common.TRAJ_DIR, "*.npy")))
        sig = np.array([os.path.getsize(f) for f in files])
        cache = os.path.join(common.SCRATCH, "plan_tip_cache.npz")
        self.tip = {}
        if os.path.exists(cache):
            z = np.load(cache)
            if "sig" in z and np.array_equal(z["sig"], sig):
                self.tip = {int(k[1:]): z[k] for k in z.files if k != "sig"}
        if not self.tip:
            print(f"FK over {len(files)} planned trajectories...")
            for f in files:
                i = int(os.path.basename(f)[:3])
                q = np.load(f)
                self.tip[i] = arm.frame_positions(TRACE_FRAME, q)
            os.makedirs(common.SCRATCH, exist_ok=True)
            np.savez_compressed(cache, sig=sig,
                                **{f"r{i}": p for i, p in self.tip.items()})
            print(f"cached -> {cache}")


class TreeOverlay:
    """The whole RRT-connect explored tree for one query row, lazily
    loaded from planned_trajectory_info/trees/NNN.npz (not every row has
    one). Nodes are task-space poses; the first 3 columns are already an
    (x,y,z) point in the base frame -- no FK needed, unlike PlanPaths."""

    def __init__(self, overlay):
        self.ov = overlay
        self.row = None
        self.segments_uv = None   # cached projected edges for self.row
        self.pts3 = None          # (n,3) z-corrected node positions, this row
        self.owner = None         # (n,) which RRT-connect tree (0=start,1=goal)
        self.parents = None

    def _ensure(self, row):
        if row == self.row:
            return
        self.row = row
        path = os.path.join(common.TREE_DIR, f"{row:03d}.npz")
        if not os.path.exists(path):
            self.segments_uv = self.pts3 = self.owner = self.parents = None
            return
        z = np.load(path)
        nodes, parents, owner = z["nodes"], z["parents"], z["owner"]
        pts3 = nodes[:, :3].copy()
        # Task-space nodes carry the planner's own fixed maze-plane z
        # (0.14) rather than the URDF/camera-calibration frame's z for the
        # same physical height -- draw_goal already applies this identical
        # correction to start_state/goal_state (which ARE this tree's own
        # root nodes, verified byte-identical), so apply it here too for
        # every node, not just the two roots.
        pts3[:, 2] = self.ov.tip_plane_z
        self.pts3, self.owner, self.parents = pts3, owner, parents
        child = np.arange(len(parents))
        has_parent = child != parents   # root has parents[0] == 0 == itself
        if not has_parent.any():
            # A "direct connection" solve: both RRT-connect trees are a
            # single root, zero edges grown. cv2.projectPoints returns
            # None (not an empty array) for zero points, so short-circuit
            # rather than let that crash .reshape() downstream.
            self.segments_uv = np.zeros((0, 2, 2), dtype=np.int32)
            return
        a = self.ov.px(pts3[parents[has_parent]])
        b = self.ov.px(pts3[child[has_parent]])
        self.segments_uv = np.stack([a, b], axis=1).astype(np.int32)

    def draw(self, img, row):
        self._ensure(row)
        if self.segments_uv is None or len(self.segments_uv) == 0:
            return
        canvas = img.copy()
        cv2.polylines(canvas, list(self.segments_uv), False, TREE_COLOR,
                      TREE_WIDTH, cv2.LINE_AA)
        cv2.addWeighted(canvas, TREE_ALPHA, img, 1 - TREE_ALPHA, 0, dst=img)

    def closest_start_node_to_goal(self, row):
        """On a failed solve: the start-tree (owner==0) node nearest the
        goal -- how close that side's search actually got. z is already
        the same constant for every node (tip_plane_z), so this is
        effectively the planar (x,y) distance on the maze."""
        self._ensure(row)
        if self.pts3 is None:
            return None
        is_root = np.arange(len(self.parents)) == self.parents
        goal_roots = np.where(is_root & (self.owner == 1))[0]
        start_mask = self.owner == 0
        if len(goal_roots) == 0 or not start_mask.any():
            return None
        goal_pt = self.pts3[goal_roots[0]]
        start_pts = self.pts3[start_mask]
        idx = np.argmin(np.linalg.norm(start_pts - goal_pt, axis=1))
        return start_pts[idx]


class Overlay:
    def __init__(self, plan=False, tree=False):
        cam = common.read_json(common.camera_path())
        self.cam = cam
        self.K = np.array([[cam["f"], 0, cam["cx"]],
                           [0, cam["f"], cam["cy"]], [0, 0, 1]])
        self.dist = np.array([cam["k1"], cam["k2"], 0, 0])
        self.rvec = np.array(cam["rvec"])
        self.tvec = np.array(cam["tvec"])
        self.v0 = cam["video_start_epoch"]
        self.delta = cam["delta_s"]

        js = common.load_joint_states()
        self.t_log = js["t"]
        self.z_err_mm = js["z_error"] * 1000.0
        self.tip = FrameTrajectory(TRACE_FRAME)
        self.tip_plane_z = float(np.median(self.tip.p[:, 2]))
        self.queries = common.load_queries()
        self.q_t = np.array([q["t"] for q in self.queries])
        # The task alternates between exactly two fixed points (confirmed:
        # only 2 distinct goal_eef_pos values across all 101 queries) --
        # each is "goal" while a query targets it and "start" the moment
        # the next query targets the other one. Both are drawn permanently.
        self.end_points = sorted({tuple(np.round(q["goal_eef_pos"], 6))
                                  for q in self.queries})
        self.arm = common.ArmKinematics()
        self.qs = common.joint_matrix(js)
        self.plans = PlanPaths(self.arm) if plan else None
        self.tree = TreeOverlay(self) if tree else None
        # Sequential-render state for the live-plan overlay: the active plan
        # row, how far along it the robot has progressed, and the projected
        # remaining path drawn since the last 10 Hz tick.
        self._plan_state = {"row": None, "prog": 0, "tick": None, "uv": None,
                            "full_uv": None}

    # -- time mapping ------------------------------------------------------
    def log_time(self, t_video):
        return t_video + self.v0 - self.delta

    # -- projection --------------------------------------------------------
    def px(self, pts3):
        uv, _ = cv2.projectPoints(np.asarray(pts3, float).reshape(-1, 3),
                                  self.rvec, self.tvec, self.K, self.dist)
        return uv.reshape(-1, 2)

    # -- drawing pieces ----------------------------------------------------
    def draw_trace(self, img, t_video):
        te = self.log_time(t_video)
        t0, t1 = self.t_log[0], te
        if t1 <= t0:
            return
        ts = np.arange(max(t0, t1 - HIST_S), t1, 1.0 / 30)
        if len(ts) < 2:
            return
        p, _ = self.tip.at(ts)
        uv = self.px(p).astype(np.int32)
        age = te - ts
        # Age bands, oldest first so the fresh tail draws on top; each band
        # gets its own alpha so history recedes instead of accumulating.
        # Widths bumped (5/6/9 -> 7/9/13) per Tommy: the fix for "lines are
        # hard to see" is making the actual trace/plan thicker, not the
        # RRT tree brighter (that was tried and reverted -- too loud).
        bands = [
            (age <= HIST_S) & (age > MID_S), REVAMP_BLUE, 7, 0.30,
            (age <= MID_S) & (age > TAIL_S), REVAMP_BLUE, 9, 0.55,
            (age <= TAIL_S), HIGHLIGHT, 13, 0.85,
        ]
        new = uv[age <= TAIL_S]
        for i in range(0, len(bands), 4):
            sel, color, width, alpha = bands[i:i + 4]
            seg = uv[sel]
            if len(seg) > 1:
                canvas = img.copy()
                cv2.polylines(canvas, [seg], False, color, width, cv2.LINE_AA)
                cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, dst=img)
        if len(new):
            cv2.circle(img, tuple(new[-1]), 14, HIGHLIGHT, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(new[-1]), 14, WHITE, 2, cv2.LINE_AA)

    def active_query(self, t_video):
        te = self.log_time(t_video)
        i = np.searchsorted(self.q_t, te) - 1
        return (self.queries[i], te - self.q_t[i]) if i >= 0 else (None, 1e9)

    def draw_goal(self, img, t_video):
        """Both fixed end points, permanently: whichever the active query
        targets is "goal" (bright, filled); the other is "start" (hollow,
        dimmer) -- they swap on every query, since the task just bounces
        between these same two points the whole time."""
        q, _ = self.active_query(t_video)
        goal_pos = np.array((q or self.queries[0])["goal_eef_pos"], float)
        goal_key = tuple(np.round(goal_pos, 6))
        for p in self.end_points:
            g = np.array(p, float)
            g[2] = self.tip_plane_z
            uv = self.px(g)[0].astype(int)
            if p == goal_key:
                cv2.circle(img, tuple(uv), 22, WHITE, 3, cv2.LINE_AA)
                cv2.circle(img, tuple(uv), 8, WHITE, -1, cv2.LINE_AA)
                cv2.putText(img, "goal", (uv[0] + 30, uv[1] + 8), FONT, 1.4,
                           WHITE, 2, cv2.LINE_AA)
            else:
                cv2.circle(img, tuple(uv), 22, WHITE, 3, cv2.LINE_AA)
                cv2.circle(img, tuple(uv), 8, WHITE, 3, cv2.LINE_AA)
                cv2.putText(img, "start", (uv[0] + 30, uv[1] + 8), FONT, 1.4,
                           WHITE, 2, cv2.LINE_AA)

    def draw_obstacles(self, img, t_video):
        q, age = self.active_query(t_video)
        if q is None or age > OBSTACLE_FLASH_S:
            return
        alpha = 1.0 - age / OBSTACLE_FLASH_S
        canvas = img.copy()
        for s in q.get("spheres", []):
            c = np.array(s["position"], float)
            uv = self.px(c)[0].astype(int)
            edge = self.px(c + [0, s["radius"], 0])[0]
            r_px = int(max(np.linalg.norm(edge - uv), 8) * 1.6)
            cv2.circle(canvas, tuple(uv), r_px, WARN_ORANGE, 4, cv2.LINE_AA)
        for b in q.get("cuboids", []):
            c = np.array(b.get("center", b.get("position")), float)
            uv = self.px(c)[0].astype(int)
            cv2.circle(canvas, tuple(uv), 60, WARN_ORANGE, 4, cv2.LINE_AA)
        cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, dst=img)

    def draw_panel(self, img, t_video):
        q, age = self.active_query(t_video)
        if q is None:
            return
        lines = []
        if q["solved"]:
            ms = q["rrtc_nanoseconds"] / 1e6
            lines.append((f"replanned in {ms:.1f} ms", WHITE))
            if q["rrtc_iterations"] > 0:
                lines.append((f"{q['rrtc_iterations']:,} iterations · "
                              f"{q['num_waypoints']} waypoints",
                              (200, 200, 200)))
            else:
                lines.append((f"direct connection · "
                              f"{q['num_waypoints']} waypoints",
                              (200, 200, 200)))
        else:
            lines.append(("goal blocked - replanning...", WARN_ORANGE))
        te = self.log_time(t_video)
        i = np.searchsorted(self.t_log, te)
        if 0 < i < len(self.z_err_mm):
            lines.append((f"constraint error {abs(self.z_err_mm[i]):.2f} mm",
                          (200, 200, 200)))
        n_replans = int(np.searchsorted(self.q_t, te))
        lines.append((f"replan #{n_replans}", (160, 160, 160)))

        x, y = 90, 130
        # flash the panel briefly on a fresh replan
        if age < 0.4 and q["solved"]:
            cv2.rectangle(img, (x - 30, y - 70), (x + 1150, y + len(lines) * 78),
                          HIGHLIGHT, 6)
        for text, color in lines:
            cv2.putText(img, text, (x, y), FONT, 2.2, (0, 0, 0), 9, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), FONT, 2.2, color, 4, cv2.LINE_AA)
            y += 78

    def active_plan_row(self, te):
        """Latest query row at te whose plan the robot is executing: the
        most recent SOLVED row (during a goal-blocked cluster the robot
        keeps following the previous plan, which is what we show)."""
        i = int(np.searchsorted(self.q_t, te)) - 1
        while i >= 0 and i not in self.plans.tip:
            i -= 1
        return i if i >= 0 else None

    def draw_plan(self, img, t_video):
        te = self.log_time(t_video)
        row = self.active_plan_row(te)
        if row is None:
            return
        st = self._plan_state
        tick = int(t_video / PLAN_REFRESH_S)
        # Global relocalization on the first eval and on non-sequential time
        # (stills/montage); sequential renders advance a windowed search.
        jumped = st["tick"] is None or abs(tick - st["tick"]) > 3
        if row != st["row"] or tick != st["tick"] or st["uv"] is None:
            pts = self.plans.tip[row]
            if row != st["row"]:
                st["row"], st["prog"] = row, 0
                st["full_uv"] = (self.px(pts).astype(np.int32)
                                  if len(pts) >= 2 else None)
            cur, _ = self.tip.at(te)
            d = np.linalg.norm(pts[:, :2] - cur[:2], axis=1)
            if jumped:
                # Non-sequential eval (montage/stills): relocalize globally.
                st["prog"] = int(np.argmin(d))
            else:
                lo = st["prog"]
                hi = min(len(pts), lo + PLAN_SEARCH_AHEAD)
                st["prog"] = lo + int(np.argmin(d[lo:hi]))
            st["tick"] = tick
            remaining = pts[st["prog"]:]
            st["uv"] = (self.px(remaining).astype(np.int32)
                        if len(remaining) >= 2 else None)
        if st["full_uv"] is not None:
            canvas = img.copy()
            cv2.polylines(canvas, [st["full_uv"]], False, PLAN_COLOR,
                          PLAN_FULL_WIDTH, cv2.LINE_AA)
            cv2.addWeighted(canvas, PLAN_FULL_ALPHA, img,
                            1 - PLAN_FULL_ALPHA, 0, dst=img)
        if st["uv"] is None:
            return
        age = te - self.q_t[row]     # age of THIS plan, not the newest query
        alpha = 1.0 if age < PLAN_FLASH_S else PLAN_ALPHA
        canvas = img.copy()
        dashed_polyline(canvas, st["uv"], PLAN_COLOR, PLAN_WIDTH)
        cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, dst=img)

    def draw_skeleton(self, img, t_video):
        te = self.log_time(t_video)
        i = int(np.argmin(np.abs(self.t_log - te)))
        uv = self.px(self.arm.skeleton(self.qs[i])).astype(int)
        cv2.polylines(img, [uv], False, (0, 0, 255), 3, cv2.LINE_AA)
        for p in uv:
            cv2.circle(img, tuple(p), 9, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_tree(self, img, t_video):
        te = self.log_time(t_video)
        row = int(np.searchsorted(self.q_t, te)) - 1
        if row >= 0:
            self.tree.draw(img, row)

    def draw_closest_approach(self, img, t_video):
        """On a failed ("goal blocked") query: mark the start-tree node
        that got nearest the goal -- how close that attempt actually
        came, not just that it failed."""
        te = self.log_time(t_video)
        row = int(np.searchsorted(self.q_t, te)) - 1
        if row < 0 or self.queries[row].get("solved", True):
            return
        pt = self.tree.closest_start_node_to_goal(row)
        if pt is None:
            return
        uv = self.px(pt)[0].astype(int)
        cv2.drawMarker(img, tuple(uv), WARN_ORANGE, cv2.MARKER_TILTED_CROSS,
                       28, 3, cv2.LINE_AA)
        cv2.putText(img, "closest attempt", (uv[0] + 20, uv[1] + 6), FONT,
                   1.1, WARN_ORANGE, 2, cv2.LINE_AA)

    def annotate(self, img, t_video, skeleton=False, obstacle_rings=True,
                 plan=True):
        """plan=False suppresses the green dashed plan line for this one
        frame -- used by the goal-blocked callout freeze, where the whole
        point is that NO path is available (a plan line there would
        contradict the caption)."""
        self.draw_trace(img, t_video)
        if self.tree is not None:
            self.draw_tree(img, t_video)
            self.draw_closest_approach(img, t_video)
        if self.plans is not None and plan:
            self.draw_plan(img, t_video)
        self.draw_goal(img, t_video)
        if obstacle_rings:
            self.draw_obstacles(img, t_video)
        self.draw_panel(img, t_video)
        if skeleton:
            self.draw_skeleton(img, t_video)
        return img


# -- frame IO ---------------------------------------------------------------

def grab_bgr(t):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", common.VIDEO,
           "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    w, h = common.VIDEO_WH
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()


def montage(ov, times, out_path, skeleton=True):
    tiles = []
    for t in times:
        img = ov.annotate(grab_bgr(t), t, skeleton=skeleton)
        cv2.putText(img, f"t={t:.1f}s", (60, 2100), FONT, 3, (0, 255, 255),
                    6, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (960, 540)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    grid = np.vstack([r for r in rows if r.shape[1] == rows[0].shape[1]])
    cv2.imwrite(out_path, grid)
    print(out_path)


def slow_windows(ov, margin_pre=1.0, margin_post=1.0):
    """[(t0,t1), ...] video-time windows to keep at 1x around a REPLAN --
    a query issued because the current plan became invalid (an obstacle
    blocked it), not because the robot reached its goal and is being
    handed the next one. There's no explicit reason/goal_reached field in
    planning_queries.jsonl, but goal_eef_pos is: a query whose goal matches
    the previous query's goal is a retry against the same still-unreached
    goal (a real replan); a query with a NEW goal is just the next leg of
    the task and was never sped up because anything failed, so it gets no
    slowdown window. Per Tommy: only slow down for actual replanning, and
    only 1s on either side (was 2s)."""
    q_video_t = ov.q_t - ov.v0 + ov.delta
    windows = []
    prev_goal = None
    for t, q in zip(q_video_t, ov.queries):
        goal = q.get("goal_eef_pos")
        if prev_goal is not None and goal == prev_goal:
            windows.append((t - margin_pre, t + margin_post))
        prev_goal = goal
    return windows


def ask_slow_windows(ov, t0, t1, margin_pre=1.0, margin_post=1.0):
    """Interactive, optional alternative to slow_windows(): pops up a
    window per replan inside [t0,t1] with that frame annotated, and asks
    y/n whether to keep it at 1x. Requires a real display (cv2.imshow) --
    not for headless/SSH-only sessions. [q] stops asking and keeps
    whatever was already confirmed; anything not explicitly confirmed
    stays at full speedup."""
    q_video_t = ov.q_t - ov.v0 + ov.delta
    in_range = [(i, t) for i, t in enumerate(q_video_t) if t0 <= t <= t1]
    windows = []
    for i, t in in_range:
        img = ov.annotate(grab_bgr(t), t, skeleton=False)
        disp = cv2.resize(img, (960, 540))
        cv2.putText(disp, f"replan #{i + 1}  t={t:.1f}s  --  keep at 1x?  "
                          "[y]es  [n]o  [q]uit asking",
                   (20, 40), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("confirm slow-down (any other key = no)", disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('y'), ord(' ')):
            windows.append((t - margin_pre, t + margin_post))
            print(f"  replan #{i + 1} (t={t:.1f}s): kept at 1x")
        else:
            print(f"  replan #{i + 1} (t={t:.1f}s): left sped up")
    cv2.destroyAllWindows()
    return windows


def render_segment(ov, t0, t1, out_path, fps=None, ghost=None,
                   ghost_alpha=0.45, speedup=1, slow_margin=(1.0, 1.0),
                   windows=None, callouts=None):
    """ghost: optional sim_ghost.GhostRenderer composited (with interpolated
    obstacle bubbles, ReVAMP-blue tint) under the annotations; ring flashes
    are suppressed since the bubbles already show the obstacles.

    speedup > 1: keep every frame within `slow_margin` seconds of any
    replan (1x there), otherwise keep only 1 in `speedup` frames. Dropped
    frames are never decoded into an annotated image at all (skip the
    expensive per-frame overlay work, not just the encode), so this cuts
    render time roughly in proportion to how much of the segment ends up
    fast. Output stays at the source fps -- dropping frames rather than
    re-timestamping is what makes the fast stretches play sped-up.

    windows: explicit [(t0,t1), ...] 1x windows (e.g. from
    ask_slow_windows()), overriding the default of every replan. None
    means "every replan, via slow_windows()".

    callouts: [(t, text), ...] -- at each one, FREEZE that source frame
    for CALLOUT_HOLD_S seconds of output (repeating the same annotated
    image, caption fading out over the back CALLOUT_FADE_FRAC of the
    hold -- see callout_hold_alphas()), then resume normal playback from
    that same source instant. This replaced an earlier "slow window"
    version of callouts (play on at 1x) -- per Tommy, a real freeze reads
    the caption without competing with continued motion. Independent of
    `windows`/`slow_margin`, which still drive the ordinary replan
    slowdown when callouts is None.
    """
    fps = fps or common.VIDEO_FPS
    w, h = common.VIDEO_WH
    if speedup <= 1:
        windows = []
    elif windows is None:
        windows = slow_windows(ov, *slow_margin)
    callouts = callouts or []
    # A callout whose time already passed before this segment even starts
    # (e.g. rendering a short sub-segment for a preview) must never fire --
    # otherwise it "catches up" and freezes on frame 1, since t >= ct is
    # trivially true for the whole segment.
    callout_done = [co[0] - CALLOUT_LEAD_S < t0 for co in callouts]
    hold_alphas = callout_hold_alphas(fps)
    goal_blocked_ghost = None  # lazily built iff a callout actually needs it

    def is_slow(t):
        return any(a <= t <= b for a, b in windows)

    dec = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t0), "-to", str(t1),
         "-i", common.VIDEO, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    n = 0            # decoded (source) frame count
    n_out = 0         # encoded (output) frame count
    fast_i = 0        # decimation phase, advanced only on fast frames
    nbytes = w * h * 3
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        t = t0 + n / fps
        n += 1

        due = next((i for i, (co, done) in
                    enumerate(zip(callouts, callout_done))
                    if not done and t >= co[0] - CALLOUT_LEAD_S), None)
        if due is not None:
            callout_done[due] = True
            _, text, ghost_q, arrow = callouts[due]
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            if ghost is not None:
                img = ghost.composite(img, t, ghost_alpha, obstacles=True,
                                      tint=(203, 160, 141))
            if ghost_q is not None:
                # A second, distinctly-tinted ghost: not the moving
                # time-indexed one above (that shows the ACTUAL executed
                # trajectory, if --ghost is even on), but a static render of
                # a config the robot never reached -- the planner's best
                # candidate, still short of the goal. Orange tint (vs. the
                # moving ghost's ReVAMP blue) so the two never read as the
                # same thing if both happen to be on screen at once.
                if goal_blocked_ghost is None:
                    from sim_ghost import GhostRenderer
                    goal_blocked_ghost = GhostRenderer()
                real = img.copy()
                img = goal_blocked_ghost.composite_q(
                    img, ghost_q, alpha=0.55, tint=WARN_ORANGE)
                # Real wand pixels back on top -- see WAND_REPAINT_4K.
                m = wand_repaint_mask(img.shape) > 0
                img[m] = real[m]
            # No plan line on the goal-blocked freeze (ghost_q set): the
            # caption's point is that no path exists.
            ov.annotate(img, t, obstacle_rings=ghost is None,
                        plan=ghost_q is None)
            draw_speed_badge(img, 1)   # frozen -- not playing at `speedup` at all
            for a in hold_alphas:
                frame = img.copy()
                draw_callout(frame, text, alpha=a)
                draw_callout_arrow(frame, arrow, alpha=a)
                enc.stdin.write(frame.tobytes())
                n_out += 1
            print(f"{n} src frames, {n_out} kept ({t:.1f}s) -- "
                  f"froze {CALLOUT_HOLD_S:.1f}s for callout: {text!r}")
            continue

        slow = is_slow(t)
        if not slow:
            keep = (fast_i % speedup == 0)
            fast_i += 1
            if not keep:
                continue
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        if ghost is not None:
            img = ghost.composite(img, t, ghost_alpha, obstacles=True,
                                  tint=(203, 160, 141))
        ov.annotate(img, t, obstacle_rings=ghost is None)
        draw_speed_badge(img, 1 if slow else speedup)
        enc.stdin.write(img.tobytes())
        n_out += 1
        if n % 300 == 0:
            print(f"{n} src frames, {n_out} kept ({t:.1f}s)")
    dec.wait()
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        sys.exit("encoder failed")
    print(f"{n} src frames -> {n_out} kept -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--frames", type=float, nargs="+")
    ap.add_argument("--segment", type=float, nargs=2)
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--ghost", action="store_true",
                    help="composite the sim ghost + obstacle bubbles "
                         "under the annotations")
    ap.add_argument("--ghost-alpha", type=float, default=0.45)
    ap.add_argument("--plan", action="store_true",
                    help="overlay the live current plan (remaining path "
                         "ahead of the robot, ~10 Hz refresh)")
    ap.add_argument("--tree", action="store_true",
                    help="experimental: draw the whole RRT-connect "
                         "explored tree for the active query, very faint")
    ap.add_argument("--speedup", type=int, default=1,
                    help="play this many x faster everywhere except "
                         "within --slow-margin seconds of a replan, kept "
                         "at 1x; dropped frames also skip the (expensive) "
                         "overlay work entirely, so render time drops too")
    ap.add_argument("--slow-margin", type=float, nargs=2, default=(1.0, 1.0),
                    metavar=("PRE", "POST"),
                    help="seconds before/after each replan kept at 1x "
                         "(only relevant with --speedup > 1)")
    ap.add_argument("--ask-slow", action="store_true",
                    help="optional: interactively confirm per-replan "
                         "(pops up a window, y/n) which ones to keep at "
                         "1x, instead of defaulting to all of them -- "
                         "needs a real display, not headless/SSH-only")
    ap.add_argument("--no-slow", action="store_true",
                    help="disable the replan slowdown entirely -- play "
                         "uniformly at --speedup with no 1x windows "
                         "anywhere (overrides --slow-margin/--ask-slow)")
    ap.add_argument("--callouts", action="store_true",
                    help="slow to 1x ONLY at the fixed CALLOUTS moments "
                         "(each with its own on-screen caption -- see "
                         "draw_callout()), instead of every replan; "
                         "overrides --no-slow/--slow-margin/--ask-slow")
    args = ap.parse_args()

    ov = Overlay(plan=args.plan, tree=args.tree)
    os.makedirs(common.SCRATCH, exist_ok=True)
    if args.montage:
        times = list(np.linspace(12, 380, 12))
        montage(ov, times,
                os.path.join(common.SCRATCH, "verify_montage.jpg"),
                skeleton=not args.no_skeleton)
    if args.frames:
        for t in args.frames:
            img = ov.annotate(grab_bgr(t), t,
                              skeleton=not args.no_skeleton)
            p = os.path.join(common.SCRATCH, f"overlay_{t:.1f}.jpg")
            cv2.imwrite(p, cv2.resize(img, (1920, 1080)))
            print(p)
    if args.segment:
        ghost = None
        if args.ghost:
            from sim_ghost import GhostRenderer
            ghost = GhostRenderer(with_obstacle_slots=9)
        windows = None
        callouts = None
        if args.callouts:
            windows = []           # freeze-hold replaces the 1x window now
            callouts = CALLOUTS
        elif args.no_slow:
            windows = []
        elif args.speedup > 1 and args.ask_slow:
            windows = ask_slow_windows(ov, args.segment[0], args.segment[1],
                                       *args.slow_margin)
        name = ("highlight" + ("_plan" if args.plan else "")
                + ("_ghost" if args.ghost else "")
                + ("_tree" if args.tree else "")
                + (f"_{args.speedup}x" if args.speedup > 1 else "")
                + "_4k.mp4")
        render_segment(ov, args.segment[0], args.segment[1],
                       os.path.join(common.REPO, "out", name),
                       ghost=ghost, ghost_alpha=args.ghost_alpha,
                       speedup=args.speedup, slow_margin=args.slow_margin,
                       windows=windows, callouts=callouts)


if __name__ == "__main__":
    main()
