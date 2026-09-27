"""Overlay revamp/maze-style captions directly on the REAL RBY1 hardware
footage -- no sim render, no side-by-side. Matches
revamp-video/scripts/render_overlay.py's convention: title, planning time,
and a single "constraint error" readout, plus a support-polygon stability
inset here (CoM margin to the base's support polygon, not a raw CoM
readout -- see rby1-constrained-planning/src/rby1_opt_ik.py's own
check_com_stability/inset_support_polygon_xy, which this reuses).

    python3 render_hw_overlay.py point_00
    python3 render_hw_overlay.py point_01
    python3 render_hw_overlay.py --all

Numbers shown are computed from the REAL logged joint encoder positions via
Drake FK -- this is NOT the IKFast-vs-Drake comparison
rby1-constrained-planning/notes/ikfast_drake_model_mismatch.md warns about
(that note is about Drake FK on an IKFast-*produced* config, which carries a
~3e-7 m modelling floor). Here Drake FK is applied to real hardware encoder
readings, which is exactly the "actually worth showing" measured tracking
error the note points to instead (0.5-2 mm, 1.5-5.5 mrad).

Constraint error = drift of the left-ee-to-right-ee transform (bimanual,
both arms) from its value at the run's first logged sample -- same
single-number convention as bimanual-iiwa's caption, applied to real data.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import robot_camera_fit as M  # noqa: E402

# The support-polygon geometry and safety inset below are copied verbatim
# from rby1-constrained-planning/src/rby1_opt_ik.py (support_polygon_xyzs,
# support_polygon_edge_lengths, _com_support_polygon_residuals,
# DEFAULT_SUPPORT_POLYGON_INSET) rather than imported: that module does
# `from common import RepoDir`, and since THIS script already imported its
# own scripts/common.py under the name "common", Python's sys.modules cache
# would hand rby1_opt_ik our common.py instead of rby1-constrained-planning's
# -- inserting that repo's src/ earlier on sys.path does not fix it, because
# the "common" name is already resolved and cached by the time rby1_opt_ik
# imports it. Keeping these as pure-numpy copies sidesteps the clash and
# keeps this script's only dependency on that repo's model files, not its
# module-naming.
support_polygon_xyzs = np.array([
    [0.228000, -0.265000, 0.0],   # right wheel
    [0.228000, 0.265000, 0.0],    # left wheel
    [-0.248686, 0.066310, 0.0],   # left caster
    [-0.248686, -0.066310, 0.0],  # right caster
])
support_polygon_edge_lengths = np.linalg.norm(
    np.roll(support_polygon_xyzs[:, :2], -1, axis=0) - support_polygon_xyzs[:, :2],
    axis=1)
DEFAULT_SUPPORT_POLYGON_INSET = 0.0823


def _com_support_polygon_residuals(com_xy, base_xyt):
    tx, ty, theta = base_xyt[0], base_xyt[1], base_xyt[2]
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])
    n = len(support_polygon_xyzs)
    pts_hom = np.hstack([support_polygon_xyzs[:, :2], np.ones((n, 1))])
    verts = (T @ pts_hom.T).T[:, :2]
    residuals = []
    for i in range(n):
        d = verts[(i + 1) % n] - verts[i]
        inward_normal = np.array([-d[1], d[0]])
        residuals.append(-inward_normal @ (com_xy - verts[i]))
    return np.array(residuals)

# Fitted to these two 2026-08-28 clips (see side_by_side.py's docstring for
# the envelope check) -- the original narrow 900x980 half-panel crop, tuned
# so two runs could sit side by side. A wider 16:9 crop (either the full
# native frame, or a zoomed-in-but-still-16:9 crop) put too much lab clutter
# (desks, chairs, cabinets) in frame -- the robot's own vertical motion
# envelope forces a crop wide enough that peripheral furniture can't be
# excluded while staying 16:9 landscape. So instead: keep this tight
# original crop's aspect ratio as-is (no distortion), scale it to fill the
# OUTPUT height, and pad the leftover width as a solid black banner on
# either side (video the other side) -- which is also where the caption
# text and running plots now live, off the video entirely instead of
# overlaid on it. Side is chosen per render_one() call (--banner-side);
# only these sizes are fixed.
CROP_W, CROP_H = 900, 980
CROP_X, CROP_Y = 560, 80
OUT_W, OUT_H = 1920, 1080
SCALED_W = int(round(CROP_W * OUT_H / CROP_H / 2)) * 2  # even, ~992
PAD_X = OUT_W - SCALED_W  # banner width, ~928

HW_VIDEO = {
    "point_00": "20260828_142748.mp4",
    "point_01": "20260828_143004.mp4",
}
# Bold per-run label, not the raw box coordinates -- "box at x=..., y=..."
# read as a parameter dump, not "these are two different runs" (per Tommy).
INSTANCE_LABEL = {
    "point_00": "Point 00",
    "point_01": "Point 01",
}
PLAN_CACHE = {
    "point_00": os.path.join(common.RBY1_REPO, "plans/grid_cache/point_00.pkl"),
    "point_01": os.path.join(common.RBY1_REPO, "plans/grid_cache/point_01.pkl"),
}

FPS_OUT = 30
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

WHITE = (245, 245, 245)
GRAY = (200, 200, 200)
WARN_ORANGE = (209, 143, 0)
HIGHLIGHT = (65, 105, 225)
ERROR_WARN_MM = 3.0   # real controller tracking error; see module docstring

MARGIN_SAFE_COLOR = (120, 220, 120)
MARGIN_INSIDE_SAFETY_COLOR = WARN_ORANGE
MARGIN_OUTSIDE_COLOR = (235, 80, 80)


def margin_color_fn(margin_mm):
    """Same red/orange/green thresholds the CoM-margin readout always used
    for its single current-value number, now also driving the whole running
    plot's line color -- per Tommy, the trend of "how close to the safety
    boundary has this run been" should read at a glance, not just the
    instantaneous value."""
    if margin_mm < 0:
        return MARGIN_OUTSIDE_COLOR
    elif margin_mm < DEFAULT_SUPPORT_POLYGON_INSET * 1000.0:
        return MARGIN_INSIDE_SAFETY_COLOR
    else:
        return MARGIN_SAFE_COLOR

# Fitted real camera (position+focal from
# rby1-constrained-planning/notes/camera_fit_from_logs.md, orientation
# refined here from a look-at seed via fit_camera_from_log.py's own
# Nelder-Mead polish -- see the in-session camera-fit writeup). Lets the
# bimanual-iiwa-style EE trace overlay project onto this REAL footage too.
CAMERA_FIT = {
    "point_00": os.path.join(common.REPO, "scratch/camera_fit/fit_20260828_142748.json"),
    "point_01": os.path.join(common.REPO, "scratch/camera_fit/fit_20260828_143004.json"),
}

# Same convention as bimanual-iiwa's render_segment.py TRACE_SPEC: the
# midpoint (parameterized -- the quantity the bimanual constraint actually
# holds fixed) is blue; both arms (resolved from it) are orange. Executed
# fades with age (see _draw_fading) rather than bimanual-iiwa's uniform
# solid, per Tommy -- this is real hardware footage running many seconds
# per step, where a same-brightness trail back to the start of the step
# reads as clutter; light/translucent (constant alpha) is still the rest of
# the (already fully logged) run, same width, unchanged from bimanual-iiwa.
TRACE_WIDTH = 8
TRACE_PLAN_ALPHA = 90
TRACE_FADE_TAIL_S = 2.0    # full brightness within this many seconds of "now"
TRACE_FADE_HIST_S = 6.0    # faded out entirely by this many seconds behind
TRACE_SPEC = [("left", WARN_ORANGE), ("right", WARN_ORANGE), ("mid", HIGHLIGHT)]


def _runs(pts2d):
    runs, run = [], []
    for p in pts2d:
        if p is None:
            if len(run) >= 2:
                runs.append(run)
            run = []
        else:
            run.append(p)
    if len(run) >= 2:
        runs.append(run)
    return runs


def _draw_fading(overlay_draw, pts2d, t_log, t_now, color):
    """The executed portion: full brightness within TRACE_FADE_TAIL_S of
    `t_now`, linearly fading out to nothing by TRACE_FADE_HIST_S behind it
    -- drawn one short segment at a time (each pair of consecutive samples
    ages independently) since PIL has no built-in variable-alpha polyline."""
    for i in range(len(pts2d) - 1):
        p0, p1 = pts2d[i], pts2d[i + 1]
        if p0 is None or p1 is None:
            continue
        age = t_now - t_log[i + 1]
        if age <= TRACE_FADE_TAIL_S:
            alpha = 255
        elif age >= TRACE_FADE_HIST_S:
            continue
        else:
            frac = (age - TRACE_FADE_TAIL_S) / (TRACE_FADE_HIST_S - TRACE_FADE_TAIL_S)
            alpha = round(255 * (1.0 - frac))
        overlay_draw.line([p0, p1], fill=(*color, alpha), width=TRACE_WIDTH, joint="curve")


def draw_traces(img, spec, pts2d_by_key, t_log, t_now, progress_idx, plan_end_idx):
    """spec: [(key, color), ...] into pts2d_by_key (key -> the WHOLE run's
    projected 2D points, precomputed once against the static real camera).
    progress_idx: index reached so far -- fading-with-age up to it (see
    _draw_fading). The translucent "plan" runs from it out to plan_end_idx
    only -- the end of the CURRENT (or next, if none has started) step, not
    the whole rest of the run, per Tommy: "draw the plan only for the next
    segment" (bimanual-iiwa's plan was naturally bounded this way already,
    since each one of its renders IS a single segment; this hardware log
    strings several steps together, so the same bound has to be applied
    explicitly here). Returns the composited image (`img` itself is
    unchanged) -- both parts draw onto one overlay so the fading trail and
    the translucent plan blend the same way against the video underneath.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for key, color in spec:
        pts2d = pts2d_by_key[key]
        _draw_fading(overlay_draw, pts2d[:progress_idx + 1],
                    t_log[:progress_idx + 1], t_now, color)
        for run in _runs(pts2d[progress_idx:plan_end_idx]):
            overlay_draw.line(run, fill=(*color, TRACE_PLAN_ALPHA),
                              width=TRACE_WIDTH, joint="curve")
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

def draw_boxed_lines(img, lines, anchor_xy, align="top-left", pad=12,
                     line_gap=8, box_fill=(0, 0, 0, 165), box_outline=(120, 120, 120, 255)):
    """Alpha-composite a translucent background box sized to fit `lines`
    (each (text, font, color)) onto `img` (RGB), then the text inside it.
    `anchor_xy` is the frame corner the box hugs, sized from the actual text
    extents so long lines never bleed past the box. Returns the new image.

    Real hardware footage has a light wall behind the robot, so plain text
    with no backing (the sim-render convention this was copied from) was
    unreadable. Bottom-left/-right corners were tried next, but at the
    crouched grasp pose (~t=21-22s in point_00) the robot's hands and the box
    it's picking up drop into BOTH bottom corners, and an opaque panel there
    hid the actual pick-up action -- the one moment this footage most needs
    to stay visible. Top-left stays clear through that crouch in every case
    checked; it does cross the raised arms during premove/reach, which is
    genuinely unavoidable (arms sweep the whole top there), so the panel is
    also translucent, as a second line of defence, rather than opaque.
    """
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    sizes = [draw.textbbox((0, 0), text, font=font) for text, font, _ in lines]
    widths = [b[2] - b[0] for b in sizes]
    heights = [b[3] - b[1] for b in sizes]
    box_w = max(widths) + 2 * pad
    box_h = sum(heights) + line_gap * (len(lines) - 1) + 2 * pad

    ax, ay = anchor_xy
    x0 = ax - box_w if "right" in align else ax
    y0 = ay - box_h if "bottom" in align else ay
    x1, y1 = x0 + box_w, y0 + box_h
    draw.rectangle([x0, y0, x1, y1], fill=box_fill, outline=box_outline, width=1)

    y = y0 + pad
    for (text, font, color), h, b in zip(lines, heights, sizes):
        draw.text((x0 + pad - b[0], y - b[1]), text, fill=(*color, 255), font=font)
        y += h + line_gap

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_running_plot(img, rect, title, xs, ys, y_lo, y_hi, cur_t, color,
                      font_title, font_val, val_fmt="{:+.1f}",
                      color_fn=None, legend=None):
    """Draw a small time-series line plot (axis box + zero-line + line +
    a cursor dot/readout at the current time) into `rect` = (x0, y0, x1, y1)
    on `img` (RGB). `xs`/`ys` are the (already downsampled) series; NaN `ys`
    breaks the line, matching how the constraint-error plot is undefined
    before grasp / after release. Replaces the plain numeric readouts this
    panel used to show -- per Tommy, a running plot reads the *trend*
    (approaching a limit, holding steady) at a glance where a bare number
    doesn't. `rect` is the full allotted area; the plot box itself is inset
    from it to leave room for y-axis value ticks (left) and x-axis time
    ticks (bottom).

    `color_fn(v) -> (r,g,b)`, if given, colors the line/cursor/value-text
    per sample instead of the flat `color` -- used for the CoM margin plot,
    where per Tommy the whole trace should read green/orange/red against
    the same safe / inside-safety-margin / outside-polygon thresholds the
    single current-value number used to show. `legend`: optional
    [(label, (r,g,b)), ...] swatch list drawn inside the box when
    `color_fn` is used, since the color now carries meaning that needs a
    key."""
    x0, y0, x1, y1 = rect
    LEFT_PAD, TOP_PAD, BOTTOM_PAD = 100, 12, 28
    bx0, by0, bx1, by1 = x0 + LEFT_PAD, y0 + TOP_PAD, x1, y1 - BOTTOM_PAD
    w, h = bx1 - bx0, by1 - by0
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def to_px(t, v):
        x = bx0 + (t - xs[0]) / (xs[-1] - xs[0]) * w
        y = by1 - (v - y_lo) / (y_hi - y_lo) * h
        return (x, min(max(y, by0), by1))

    draw.rectangle([bx0, by0, bx1, by1], outline=(90, 90, 90, 255), width=1)
    N_GRID = 4
    for k in range(1, N_GRID):
        gy = by0 + h * k / N_GRID
        draw.line([(bx0, gy), (bx1, gy)], fill=(55, 55, 55, 255), width=1)
        gx = bx0 + w * k / N_GRID
        draw.line([(gx, by0), (gx, by1)], fill=(55, 55, 55, 255), width=1)
    if y_lo < 0 < y_hi:
        zy = to_px(xs[0], 0.0)[1]
        draw.line([(bx0, zy), (bx1, zy)], fill=(120, 120, 120, 220), width=1)
        draw.text((bx0 - 8, zy), "0", fill=(*GRAY, 255), font=font_val, anchor="rm")
    draw.text((bx0 - 8, by0), val_fmt.format(y_hi), fill=(*GRAY, 255),
              font=font_val, anchor="rm")
    draw.text((bx0 - 8, by1), val_fmt.format(y_lo), fill=(*GRAY, 255),
              font=font_val, anchor="rm")
    draw.text((bx0, by1 + 6), "0s", fill=(*GRAY, 255), font=font_val, anchor="la")
    draw.text((bx1, by1 + 6), f"{xs[-1] - xs[0]:.0f}s", fill=(*GRAY, 255),
              font=font_val, anchor="ra")

    # Segment-by-segment (not one big polyline) so color_fn can vary the
    # color along the line's own length when given -- a flat `color`
    # degenerates to the same per-segment call every time, so one code path
    # covers both.
    prev_px = None
    cur_px = None
    for t, v in zip(xs, ys):
        if t > cur_t:
            break
        if v != v:  # NaN
            prev_px = None
            continue
        px = to_px(t, v)
        seg_color = color_fn(v) if color_fn else color
        if prev_px is not None:
            draw.line([prev_px, px], fill=(*seg_color, 255), width=3)
        prev_px = px
        cur_px = px
    if cur_px is not None:
        idx0 = min(np.searchsorted(xs, cur_t), len(ys) - 1)
        cur_color = color_fn(ys[idx0]) if color_fn else color
        r = 6
        draw.ellipse([cur_px[0] - r, cur_px[1] - r, cur_px[0] + r, cur_px[1] + r],
                     fill=(*cur_color, 255))

    if legend:
        sw = 22
        lx = bx0 + 10
        ly = by1 - 8 - len(legend) * (sw + 8) + 8  # stack up from the
        # bottom-left corner, not the top -- the data in practice sits
        # elevated near the top of this particular plot (CoM margin stays
        # "safe" most of the run), so a top-left legend collided with the
        # line itself; near-zero at the bottom is comparatively empty.
        for label, lcolor in legend:
            draw.rectangle([lx, ly, lx + sw, ly + sw], fill=(*lcolor, 255))
            draw.text((lx + sw + 8, ly + sw // 2), label, fill=(*GRAY, 255),
                      font=font_val, anchor="lm")
            ly += sw + 8

    out = Image.alpha_composite(base, overlay).convert("RGB")
    d2 = ImageDraw.Draw(out)
    idx = min(np.searchsorted(xs, cur_t), len(ys) - 1)
    cur_v = ys[idx]
    val_text = val_fmt.format(cur_v) if cur_v == cur_v else "--"
    val_color = color_fn(cur_v) if (color_fn and cur_v == cur_v) else color
    d2.text((x0, y0 - 40), title, fill=WHITE, font=font_title)
    d2.text((x1, y0 - 40), val_text, fill=val_color, font=font_val, anchor="ra")
    return out


STEP_LABELS = {
    "premove": "premove",
    "reach_approach": "reach (approach)",
    "reach_descend": "reach (descend)",
    "grasp": "grasp",
    "lift": "lift",
    "place": "place",
    "release": "release",
    "home_retreat": "retreat",
    "home": "home",
}


class _StubUnpickler(pickle.Unpickler):
    """Unpickle grid_cache plan pickles without the planner's own modules
    installed (rby1_interface etc. live in rby1-constrained-planning's
    venv, not necessarily this one). Only meta["wall_s"] -- a plain float
    in a plain dict -- is read from these, so any class that fails to
    import is stubbed rather than reconstructed faithfully."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (), {"__setstate__": lambda self, state: None})


def render_one(name, out_path, pad_s=0.6, duration_s=None, banner_side="right"):
    if banner_side == "right":
        VIDEO_X0, BANNER_X0, BANNER_X1 = 0, SCALED_W, OUT_W
    else:
        VIDEO_X0, BANNER_X0, BANNER_X1 = PAD_X, 0, PAD_X
    rec = common.load_record(name)
    t_log, q_log = common.load_states(rec)
    # Drop the premove block entirely, per Shrutheesh: the run opens with
    # the arms spreading wide apart before ever approaching the box, which
    # doesn't demonstrate anything about the planner -- cut straight to
    # the first real step (reach_approach). This trims video, captions,
    # and every running plot together, since they all key off t_log/q_log.
    if rec.get("steps"):
        premove_end = rec["steps"][0]["t_start"]
        keep = t_log >= premove_end
        t_log, q_log = t_log[keep], q_log[keep]
    with open(PLAN_CACHE[name], "rb") as f:
        plan_wall_s = _StubUnpickler(f).load()["meta"]["wall_s"]

    plant, sg, diagram = common.build_scene()
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)

    left_frame = plant.GetFrameByName("ee_body", plant.GetModelInstanceByName("left_gripper"))
    right_frame = plant.GetFrameByName("ee_body", plant.GetModelInstanceByName("right_gripper"))

    def left_right_transform(q):
        common.set_plant_from_q26(plant, plant_ctx, q)
        return plant.CalcRelativeTransform(plant_ctx, left_frame, right_frame)

    com_instances = [plant.GetModelInstanceByName(n) for n in (
        "base", "torso", "right_arm", "left_arm",
        "right_gripper", "left_gripper", "head")]

    def stability_margin_mm(q):
        """Signed distance (mm) from the CoM's xy to the nearest edge of the
        base support polygon -- positive = inside, negative = outside (tip
        risk). The base never drives in this record (see
        rby1-constrained-planning/notes/camera_fit_from_logs.md), so the
        base frame IS the world frame here (base_xyt = 0,0,0) and the CoM's
        world xy can be used directly against the base-frame polygon."""
        common.set_plant_from_q26(plant, plant_ctx, q)
        com_xy = plant.CalcCenterOfMassPositionInWorld(plant_ctx, com_instances)[:2]
        residuals = _com_support_polygon_residuals(com_xy, np.zeros(3))
        margins = -residuals / support_polygon_edge_lengths
        return float(np.min(margins) * 1000.0)

    # Reference is the left-right transform at the START of "lift", not the
    # run's first sample: the arms start "premove" spread wide apart and
    # only come together once they grasp the box, so a t=0 reference would
    # report a meaningless multi-metre "error" for the whole reach phase.
    # The constraint (holding the box between both hands) is only active
    # from grasp through release -- see the CONSTRAINED_STEPS gate below.
    lift_t_start = next(s["t_start"] for s in rec["steps"] if s["name"] == "lift")
    X_ref = left_right_transform(common.q_at(t_log, q_log, lift_t_start))
    CONSTRAINED_STEPS = {"grasp", "lift", "place", "release"}

    # Full-resolution series behind the three running plots (below), then
    # downsampled -- redrawing thousands of raw log samples with PIL every
    # output frame is too slow and adds nothing visually over ~240 points.
    margin_full = np.array([stability_margin_mm(qs) for qs in q_log])
    constrained_mask = np.array([common.step_at(rec, ts) in CONSTRAINED_STEPS for ts in t_log])
    R_ref = X_ref.rotation().matrix()
    pos_err_full = np.full(len(t_log), np.nan)
    ori_err_full = np.full(len(t_log), np.nan)
    for i in np.nonzero(constrained_mask)[0]:
        X = left_right_transform(q_log[i])
        pos_err_full[i] = float(np.linalg.norm(X.translation() - X_ref.translation()) * 1000.0)
        # Angle (deg) of the relative rotation between the current and
        # reference left-to-right transforms -- same convention as the
        # position error, just for orientation instead of translation.
        R_rel = R_ref.T @ X.rotation().matrix()
        cos_ang = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        ori_err_full[i] = float(np.degrees(np.arccos(cos_ang)))

    def _downsample_windowed(full):
        """Interpolate `full` (NaN outside CONSTRAINED_STEPS) onto plot_t,
        only inside the constrained window, so the plotted line doesn't
        bridge the gap either side of grasp/release."""
        out = np.full(PLOT_N, np.nan)
        if constrained_mask.any():
            lo_t, hi_t = t_log[constrained_mask][0], t_log[constrained_mask][-1]
            in_window = (plot_t >= lo_t) & (plot_t <= hi_t)
            out[in_window] = np.interp(
                plot_t[in_window], t_log[constrained_mask], full[constrained_mask])
        return out

    PLOT_N = 240
    plot_t = np.linspace(t_log[0], t_log[-1], PLOT_N)
    plot_margin = np.interp(plot_t, t_log, margin_full)
    plot_pos_err = _downsample_windowed(pos_err_full)
    plot_ori_err = _downsample_windowed(ori_err_full)

    MARGIN_Y_HI = max(float(np.nanmax(margin_full)) * 1.15, 50.0)
    MARGIN_Y_LO = min(float(np.nanmin(margin_full)) * 1.15, 0.0)
    # No large fixed floor on these two (there used to be one, 5.0/2.0) --
    # per Tommy, a floor that big made the actual trace read as a flat tiny
    # sliver whenever the real error stayed small (which is the common
    # case, since these are both errors we want to be small). Scale to
    # this run's own min/max instead, with just a small margin so the
    # cursor dot isn't flush against the top edge.
    POS_ERR_Y_HI = (max(float(np.nanmax(pos_err_full[constrained_mask])) * 1.15, 1e-3)
                   if constrained_mask.any() else 1.0)
    ORI_ERR_Y_HI = (max(float(np.nanmax(ori_err_full[constrained_mask])) * 1.15, 1e-3)
                   if constrained_mask.any() else 1.0)

    # EE trace (see CAMERA_FIT/TRACE_SPEC docstrings): left/right arms
    # (orange, resolved) plus the midpoint (blue, parameterized) -- same
    # convention as bimanual-iiwa's DualFollower. Precompute once, over the
    # raw encoder log (not resampled), and project with a STATIC camera
    # (the phone never moved -- unlike bimanual-iiwa's orbiting synthetic
    # one, this is one fixed X_CW for the whole clip).
    have_trace = name in CAMERA_FIT and os.path.exists(CAMERA_FIT[name])
    if have_trace:
        with open(CAMERA_FIT[name]) as f:
            fit = json.load(f)
        X_cam = M.pose_from_vec(fit["X_WC_vec"])
        R_cam = X_cam.rotation().matrix()
        eye_cam = X_cam.translation()
        f_cam = fit["focal_px_full"]
        cx_cam, cy_cam = fit["principal_point"]
        dt_cam = fit["dt_s"]

        scale_x, scale_y = SCALED_W / CROP_W, OUT_H / CROP_H

        def project(p_W):
            p_C = R_cam.T @ (p_W - eye_cam)
            if p_C[2] <= 0.05:
                return None
            u = cx_cam + f_cam * p_C[0] / p_C[2]
            v = cy_cam + f_cam * p_C[1] / p_C[2]
            # Into the crop, scaled up the same as the crop->output ffmpeg
            # filter scales the pixels, then shifted by VIDEO_X0 -- 0 if the
            # video sits flush left (banner on the right), or the banner's
            # own width if the video is pushed right instead.
            return ((u - CROP_X) * scale_x + VIDEO_X0, (v - CROP_Y) * scale_y)

        left_pts_world, right_pts_world, mid_pts_world = [], [], []
        for q_sample in q_log:
            common.set_plant_from_q26(plant, plant_ctx, q_sample)
            pl = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(), left_frame).translation()
            pr = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(), right_frame).translation()
            left_pts_world.append(pl)
            right_pts_world.append(pr)
            mid_pts_world.append((pl + pr) / 2.0)
        pts2d_by_key = {
            "left": [project(p) for p in left_pts_world],
            "right": [project(p) for p in right_pts_world],
            "mid": [project(p) for p in mid_pts_world],
        }

        # Blue (mid) is the PARAMETERIZED quantity, meaningful only while
        # the bimanual constraint is actually active -- per Tommy: "the
        # blue should only be for the constrained segment." Outside
        # CONSTRAINED_STEPS the two arms move independently, so there's no
        # fixed relative transform for a midpoint to parameterize. Blank
        # those samples out to None rather than gating in the draw call, so
        # _runs() just breaks the line there like a point behind the camera.
        constrained_idxs = [i for i, s in enumerate(t_log)
                            if common.step_at(rec, s) in CONSTRAINED_STEPS]
        if constrained_idxs:
            lo, hi = constrained_idxs[0], constrained_idxs[-1]
            pts2d_by_key["mid"] = [
                p if lo <= j <= hi else None
                for j, p in enumerate(pts2d_by_key["mid"])]

    video_path = os.path.join(common.REPO, HW_VIDEO[name])
    t_clip0 = M.clip_start_wall(video_path)
    start_s = t_log[0] - t_clip0 - pad_s
    end_s = t_log[-1] - t_clip0 + pad_s
    if duration_s is not None:
        end_s = min(end_s, start_s + duration_s)

    # Single sequential decode pass (one seek, then straight through) --
    # cv2.VideoCapture + a CAP_PROP_POS_MSEC seek on every frame was
    # reseeking to the nearest keyframe and redecoding forward each time,
    # which made a ~50s clip take many minutes. ffmpeg's own -ss + one long
    # read, with the crop and fps conversion done in its filter graph, is a
    # single forward decode.
    dur = end_s - start_s
    decode_cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.3f}", "-i", video_path,
        "-t", f"{dur:.3f}",
        "-vf", f"crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y},"
              f"scale={SCALED_W}:{OUT_H},"
              f"pad={OUT_W}:{OUT_H}:{VIDEO_X0}:0:black,fps={FPS_OUT}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    src = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    frame_bytes = OUT_W * OUT_H * 3

    try:
        font = ImageFont.truetype(FONT_PATH, 30)
        font_small = ImageFont.truetype(FONT_PATH_REG, 26)
        font_plot = ImageFont.truetype(FONT_PATH_REG, 24)
    except OSError:
        font = font_small = font_plot = ImageFont.load_default()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS_OUT), "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an", out_path,
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    i = 0
    print(f"{name}: clip t = {start_s:.2f} .. {end_s:.2f} s")
    while True:
        t = start_s + i / FPS_OUT
        if t > end_s:
            break
        raw = src.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame_rgb = np.frombuffer(raw, np.uint8).reshape(OUT_H, OUT_W, 3)

        t_log_query = min(max(t + t_clip0, t_log[0]), t_log[-1])
        q = common.q_at(t_log, q_log, t_log_query)
        raw_step = common.step_at(rec, t_log_query)
        margin_mm = stability_margin_mm(q)
        step = STEP_LABELS.get(raw_step, "")
        elapsed = t_log_query - t_log[0]

        img = Image.fromarray(frame_rgb)

        if have_trace:
            t_proj = min(max(t + t_clip0 + dt_cam, t_log[0]), t_log[-1])
            progress_idx = bisect.bisect_right(t_log, t_proj)
            # Current step if inside one, else the upcoming one (t <= its
            # t_end is trivially true for every step before it starts too,
            # so the first such step is exactly "the next segment").
            active_step = next((s for s in rec["steps"] if t_proj <= s["t_end"]),
                               rec["steps"][-1])
            plan_end_idx = bisect.bisect_right(t_log, active_step["t_end"])
            img = draw_traces(img, TRACE_SPEC, pts2d_by_key, t_log, t_proj,
                              progress_idx, plan_end_idx)

        margin_color = margin_color_fn(margin_mm)
        pos_err_color = WARN_ORANGE
        ori_err_color = (90, 190, 230)

        # Captions now live in the solid black banner to the video's right
        # (PAD_X wide), not overlaid on the footage -- no translucent box
        # needed since the banner is already opaque black. No per-instance
        # "Point 00"/"Point 01" title line here -- the final assembly's own
        # title card already names this section, so it was redundant.
        img = draw_boxed_lines(img, [
            (f"step: {step}   t = {elapsed:.2f} s", font, WHITE),
            (f"planning time {plan_wall_s * 1000:.0f} ms", font_small, GRAY),
        ], (BANNER_X0 + 20, 20), align="top-left", box_fill=(0, 0, 0, 0), box_outline=None)

        # Running plots (in place of the old bare numeric readouts) -- a
        # trend (approaching a limit, holding steady) reads faster than a
        # single changing number, and shows the whole run's shape, not just
        # the instant. Position and orientation error are two genuinely
        # different failure modes (a translational drift vs. a tilted box),
        # so they get their own plots rather than being folded into one.
        img = draw_running_plot(
            img, (BANNER_X0 + 20, 225, BANNER_X1 - 20, 405),
            "bimanual position error, mm", plot_t, plot_pos_err,
            0.0, POS_ERR_Y_HI, t_log_query, pos_err_color, font_small, font_plot)
        img = draw_running_plot(
            img, (BANNER_X0 + 20, 475, BANNER_X1 - 20, 655),
            "bimanual orientation error, deg", plot_t, plot_ori_err,
            0.0, ORI_ERR_Y_HI, t_log_query, ori_err_color, font_small, font_plot)
        img = draw_running_plot(
            img, (BANNER_X0 + 20, 725, BANNER_X1 - 20, 905),
            "CoM margin to support polygon, mm", plot_t, plot_margin,
            MARGIN_Y_LO, MARGIN_Y_HI, t_log_query, margin_color, font_small, font_plot,
            color_fn=margin_color_fn, legend=[
                ("safe", MARGIN_SAFE_COLOR),
                ("safe, close to limit", MARGIN_INSIDE_SAFETY_COLOR),
                ("unsafe (outside polygon)", MARGIN_OUTSIDE_COLOR),
            ])

        # Static "1x" -- this is real hardware footage played back at its
        # own logged rate, never sped up, but per Tommy every video in the
        # final assembly should say its playback speed explicitly. Placed at
        # the banner's own bottom-left, clear of the video entirely.
        img = draw_boxed_lines(img, [("1x", font_small, WHITE)],
                               (BANNER_X0 + 20, OUT_H - 20), align="bottom-left",
                               box_fill=(0, 0, 0, 0), box_outline=None)

        pipe.stdin.write(np.array(img).tobytes())
        if i % 90 == 0:
            print(f"  frame {i}  t={t:.2f}s  step={step}")
        i += 1

    src.stdout.close()
    src.wait()
    pipe.stdin.close()
    pipe.wait()
    if pipe.returncode != 0:
        print(pipe.stderr.read().decode()[-2000:])
        sys.exit(1)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", choices=list(common.RECORDS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--duration", type=float, default=None,
                    help="render only N seconds (for a quick test)")
    ap.add_argument("--banner-side", choices=["left", "right"], default="right")
    args = ap.parse_args()

    names = list(common.RECORDS) if args.all else ([args.name] if args.name else None)
    if not names:
        sys.exit("give NAME (point_00 / point_01), or --all")
    for n in names:
        suffix = "" if args.banner_side == "right" else "_bannerleft"
        out_path = os.path.join(common.OUT_DIR, f"hw_overlay_{n}{suffix}.mp4")
        render_one(n, out_path, duration_s=args.duration, banner_side=args.banner_side)


if __name__ == "__main__":
    main()
