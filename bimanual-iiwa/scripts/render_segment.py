"""Render one bimanual-IIWA trajectory (Drake VTK) with a revamp-style
caption panel.

    python3 render_segment.py DualFollower T->B
    python3 render_segment.py --all              # all 12, + one concat per method

Caption panel mirrors revamp-video/scripts/render_overlay.py's own
draw_panel: white title, gray detail lines, and the same single-number
"constraint error X.XX mm" convention (there: z-height; here:
gripper-to-gripper drift from one global reference transform (see
REFERENCE_METHOD/REFERENCE_SEGMENT below), the same across all 12
renders regardless of which one is run -- so a real drift at a handoff
between two segments shows up as error instead of each segment
re-zeroing against its own start).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (
    ClippingRange, ColorRenderCamera, CameraInfo, MakeRenderEngineVtk,
    RenderCameraCore, RenderEngineVtkParams, RigidTransform, RotationMatrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

RENDERER = "vtk"
WIDTH, HEIGHT = 1920, 1080
FPS = 30
# side_by_side.py center-crops each panel to 960 wide (PANEL_W there),
# i.e. keeps x in [(WIDTH-960)/2, (WIDTH+960)/2) = [480, 1440) of this
# frame -- text anchored at the old x=30 (true left edge) fell inside the
# cropped-away region and vanished entirely in the side-by-side. Anchoring
# at the crop window's own left edge + margin keeps it visible in both the
# standalone video and the side-by-side.
TEXT_X = (WIDTH - 960) // 2 + 30
# side_by_side.py's legend banner (make_legend_banner.py, 100px tall) is
# now top-anchored, not bottom -- it used to sit clear of these captions;
# push them down below it (+90, banner height + a small margin) so the
# method-name title doesn't render underneath and become illegible.
TEXT_Y_OFFSET = 90
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
HEAD_HOLD_S = 0.5
TAIL_HOLD_S = 1.0

# revamp-video/scripts/render_overlay.py's palette, as RGB (that file is
# BGR/cv2; PIL wants RGB) -- kept identical so the two projects visually
# match.
WHITE = (245, 245, 245)
GRAY = (200, 200, 200)
WARN_ORANGE = (209, 143, 0)
HIGHLIGHT = (65, 105, 225)

# Orbiting camera: a fixed-angle view of a bimanual carry mostly shows one
# gripper occluding the other, so the eye sweeps in azimuth around
# ORBIT_CENTER over the course of the whole clip (holds included -- kept
# simple rather than pausing/resuming the sweep at hold boundaries).
ORBIT_CENTER = np.array([0.5, 0.4, 0.45])
ORBIT_RADIUS = 2.4
ORBIT_ELEV_DEG = 22.0
# "end" is the mirror image of "start" about the image's vertical axis --
# azimuth reflected about the scene's own left/right symmetry axis (the
# two arms sit symmetric about this ORBIT_CENTER's y), which is the same
# axis at both 0 and 180 deg. -160/-240 (20 deg out from -180 on each
# side) is that mirror pair. Still routed through -180, not through 0:
# azimuth near -10/+20 swings the shelf's solid back panel between camera
# and the arms, blocking everything.
ORBIT_AZ_START_DEG = -160.0
ORBIT_AZ_END_DEG = -240.0   # == +120 mod 360, mirror of -160 about -180


def camera_pose(frac):
    """frac in [0,1] -> X_WC, sweeping azimuth around ORBIT_CENTER."""
    az = np.radians(ORBIT_AZ_START_DEG
                    + frac * (ORBIT_AZ_END_DEG - ORBIT_AZ_START_DEG))
    el = np.radians(ORBIT_ELEV_DEG)
    eye = ORBIT_CENTER + ORBIT_RADIUS * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return _look_at(eye, ORBIT_CENTER)

# The single global reference (see module docstring) has to be the SAME
# transform regardless of which process renders it -- build_video.py runs
# every segment as its own subprocess, so "first trajectory rendered in
# this process" would silently give each one its own reference instead of
# one shared across all 12. Anchoring on one fixed, named trajectory
# instead is deterministic across processes: computing it is one cheap FK
# call, so there's no reason to cache/share it externally.
REFERENCE_METHOD, REFERENCE_SEGMENT = "DualFollower", "T->B"

# Above this, the "constraint error" line goes WARN_ORANGE instead of GRAY
# -- purely a caption-color threshold, not a pass/fail judgment.
ERROR_WARN_MM = 0.5

# End-effector trace overlay, matching revamp-video/scripts/render_overlay.py's
# convention (executed trace vs. remaining plan ahead) but with the color
# carrying which ARM instead of executed-vs-planned -- per Tommy: left arm
# blue, right arm orange for LeaderFollower (visualizes the asymmetry
# directly); for DualFollower both arms are symmetric so they share the
# follower/orange color, with the shared midpoint (the quantity the
# constraint actually holds fixed) drawn blue on top. No fade -- full
# brightness throughout, unlike render_overlay.py's tail/mid/history bands.
# The remaining-ahead ("plan") portion is the SAME width as the solid
# executed trace, filled (not hollow, not tapered), just light/translucent
# -- per Tommy: "it can be light, and transparent of the same size as the
# trajectory." Drawn on its own RGBA overlay at TRACE_PLAN_ALPHA and
# composited in, since plain ImageDraw on an RGB image has no translucency.
TRACE_WIDTH = 8
TRACE_PLAN_ALPHA = 90
TRACE_SPEC = {
    "LeaderFollower": [("left", HIGHLIGHT), ("right", WARN_ORANGE)],
    "DualFollower": [("left", WARN_ORANGE), ("right", WARN_ORANGE),
                     ("mid", HIGHLIGHT)],
}


def _project(X_CW, fx, fy, cx, cy, p_W):
    """World point -> (u, v) pixel, or None if behind the camera."""
    p_C = X_CW.rotation().matrix() @ p_W + X_CW.translation()
    if p_C[2] <= 0.05:
        return None
    return (cx + fx * p_C[0] / p_C[2], cy + fy * p_C[1] / p_C[2])


def _runs(pts2d):
    """[pt-or-None, ...] -> list of contiguous runs (each >= 2 points) of
    real pixel coordinates, so a point behind the camera breaks the line
    instead of drawing a nonsense chord through it."""
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


def _draw_solid(draw, pts2d, color):
    for run in _runs(pts2d):
        draw.line(run, fill=color, width=TRACE_WIDTH, joint="curve")


def _draw_translucent(overlay_draw, pts2d, color):
    """The remaining-ahead "plan" portion: filled, same TRACE_WIDTH as the
    solid executed trace, at TRACE_PLAN_ALPHA -- drawn onto an RGBA overlay
    (see draw_traces), not the base image, so it reads as light/faint."""
    for run in _runs(pts2d):
        overlay_draw.line(run, fill=(*color, TRACE_PLAN_ALPHA),
                          width=TRACE_WIDTH, joint="curve")


def draw_traces(img, spec, full_pts, progress_idx, X_WC, fx, fy, cx, cy):
    """spec: [(key, color), ...] into full_pts (key -> list of world
    points across the WHOLE segment). progress_idx is the index reached so
    far (inclusive) -- points up to it are the solid, opaque "executed"
    trace; from it onward the light/translucent "remaining plan ahead"
    (see _draw_translucent), same width. Returns the composited image (a
    new object -- `img` itself is unchanged)."""
    X_CW = X_WC.inverse()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for key, color in spec:
        pts_w = full_pts[key]
        pts2d = [_project(X_CW, fx, fy, cx, cy, p) for p in pts_w]
        _draw_translucent(overlay_draw, pts2d[progress_idx:], color)
    composited = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(composited)
    for key, color in spec:
        pts_w = full_pts[key]
        pts2d = [_project(X_CW, fx, fy, cx, cy, p) for p in pts_w]
        _draw_solid(draw, pts2d[:progress_idx + 1], color)
    return composited


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])
    return RigidTransform(RotationMatrix(R), eye)


def reference_transform(plant, plant_ctx):
    """The single global reference transform -- see the REFERENCE_METHOD/
    REFERENCE_SEGMENT comment above."""
    d = common.load_traj_json(REFERENCE_METHOD, REFERENCE_SEGMENT)
    return common.gripper_transform(plant, plant_ctx, d["configs"][0])


def render_one(method, segment, out_path, speed=1.0):
    plant, sg, diagram = common.build_scene()
    if not sg.HasRenderer(RENDERER):
        # True-black background. `exposure` brightens the robot/table/
        # shelf without lifting the background off black -- 15.0 turned
        # out too bright, 7.0 is a middle ground to check. Deliberately
        # NOT overriding `lights`: VTK's own default lighting is correct;
        # two explicit directional lights tried here rim-lit everything
        # into near-silhouettes instead of actually illuminating the arms.
        sg.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams(
            default_clear_color=np.array([0.02, 0.02, 0.022]),
            exposure=7.0)))
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)
    sg_ctx = sg.GetMyMutableContextFromRoot(context)

    X_ref = reference_transform(plant, plant_ctx)

    d = common.load_traj_json(method, segment)
    traj = common.retime_configs(d["configs"], plant, speed=speed)
    t0, t1 = traj.start_time(), traj.end_time()
    n_motion = max(2, int(round((t1 - t0) * FPS)))
    ts = np.linspace(t0, t1, n_motion)
    configs = [traj.value(t).flatten() for t in ts]
    n_head, n_tail = int(HEAD_HOLD_S * FPS), int(TAIL_HOLD_S * FPS)
    frames = [configs[0]] * n_head + configs + [configs[-1]] * n_tail
    print(f"{method} {segment}: {len(d['configs'])} waypoints -> "
          f"{t1 - t0:.2f}s retimed, {len(frames)} frames")

    # Full end-effector path for the whole segment, computed once up front
    # (it's known ahead of time -- this is a pre-planned trajectory, not an
    # online replan) -- see draw_traces()/TRACE_SPEC.
    full_pts = {"left": [], "right": [], "mid": []}
    for q in configs:
        pl, pr = common.gripper_world_positions(plant, plant_ctx, q)
        full_pts["left"].append(pl)
        full_pts["right"].append(pr)
        full_pts["mid"].append((pl + pr) / 2.0)
    trace_spec = TRACE_SPEC[method]

    core = RenderCameraCore(RENDERER, CameraInfo(WIDTH, HEIGHT, 0.75),
                            ClippingRange(0.05, 20.0), RigidTransform())
    camera = ColorRenderCamera(core, show_window=False)
    intr = core.intrinsics()
    fx, fy, cx, cy = intr.focal_x(), intr.focal_y(), intr.center_x(), intr.center_y()

    try:
        # Bigger than before -- these renders get scaled down to half-width
        # for the DualFollower/LeaderFollower side-by-side, so the method
        # name needs to stay legible (and the difference between the two
        # panels obvious) at half size.
        font = ImageFont.truetype(FONT_PATH, 52)
        font_small = ImageFont.truetype(FONT_PATH_REG, 24)
    except OSError:
        font = font_small = ImageFont.load_default()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an", out_path,
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    n_frames = len(frames)
    for i, q in enumerate(frames):
        plant.SetPositions(plant_ctx, q)
        X_WC = camera_pose(i / max(1, n_frames - 1))
        qobj = sg.get_query_output_port().Eval(sg_ctx)
        color_image = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
        arr = np.asarray(color_image.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()

        X = common.gripper_transform(plant, plant_ctx, q)
        err_mm = common.constraint_error_mm(X_ref, X)
        err_color = WARN_ORANGE if err_mm > ERROR_WARN_MM else GRAY
        # Which sample of `configs` this frame has reached -- clamped flat
        # during the head/tail holds (see frames' construction above).
        progress_idx = min(max(i - n_head, 0), n_motion - 1)

        img = Image.fromarray(arr)
        img = draw_traces(img, trace_spec, full_pts, progress_idx, X_WC, fx, fy, cx, cy)
        draw = ImageDraw.Draw(img)
        draw.text((TEXT_X, 30 + TEXT_Y_OFFSET), method, fill=HIGHLIGHT, font=font)
        draw.text((TEXT_X, 96 + TEXT_Y_OFFSET), f"planning time {d['planning_time_s'] * 1000:.2f} ms",
                 fill=GRAY, font=font_small)
        draw.text((TEXT_X, 126 + TEXT_Y_OFFSET), f"constraint error {err_mm:.3f} mm",
                 fill=err_color, font=font_small)
        pipe.stdin.write(np.array(img).tobytes())
        if i % 60 == 0:
            print(f"  frame {i}/{len(frames)}")

    pipe.stdin.close()
    pipe.wait()
    if pipe.returncode != 0:
        print(pipe.stderr.read().decode()[-2000:])
        sys.exit(1)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", nargs="?", choices=common.METHODS)
    ap.add_argument("segment", nargs="?", choices=common.SEGMENTS)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for method in common.METHODS:
            for segment in common.SEGMENTS:
                name = f"{method.lower()}_{segment.replace('->', '_to_')}.mp4"
                render_one(method, segment,
                          os.path.join(common.OUT_DIR, name), speed=args.speed)
        return

    if not (args.method and args.segment):
        sys.exit("give METHOD SEGMENT, or --all")
    name = f"{args.method.lower()}_{args.segment.replace('->', '_to_')}.mp4"
    render_one(args.method, args.segment,
              os.path.join(common.OUT_DIR, name), speed=args.speed)


if __name__ == "__main__":
    main()
