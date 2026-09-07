"""Render the FR3 maze animation to PNG frames. Runs INSIDE Blender.

Not launched directly -- render_video.py builds the command line and encodes the
frames. To run it by hand:

    ~/opt/blender-5.0.1-linux-x64/blender --background \
        --python scripts/figures/blender_maze_video.py -- \
        --html out/figures/scene/problem_108.html \
        --manifest out/figures/scene/planks.json \
        --frames-dir out/figures/frames/problem_108

Shares everything with blender_maze_still.py except what it does with time: the
still bakes several poses onto one frame, this one renders every frame. Camera,
lighting, materials and the lid variants are the same code.

Prints MAZE_VIDEO_DONE on success. The driver requires that line rather than
trusting the exit status.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import blender_common as bc  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    bc.add_common_arguments(ap)
    ap.add_argument("--frames-dir", required=True, help="directory for frame_%%04d.png")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="render frame rate; the importer interpolates DOWN from the "
                         "recording rate and never up")
    ap.add_argument("--preview-frames", type=int, default=0,
                    help="render only the first N frames, after fitting the camera to "
                         "the WHOLE animation, for checking framing cheaply")
    ap.add_argument("--camera", choices=("fixed", "orbit"), default="fixed")
    ap.add_argument("--orbit-degrees", type=float, default=360.0,
                    help="--camera orbit: degrees swept across the animation")
    ap.add_argument("--trace-mode", choices=("none", "full", "growing"), default="growing",
                    help="none omits the trace; full draws it complete from frame 1; "
                         "growing draws it as the marker lays it down (default)")
    ap.add_argument("--frame-subject", default="scene",
                    choices=("scene", "board", "robot", "robot+board"))
    return ap.parse_args(bc.script_args())


def orbit_camera(cam, target, center, distance, elevation, start_azimuth, degrees,
                 frame_start, frame_end):
    """Keyframe the camera's location around the subject across the animation."""
    for frame in range(frame_start, frame_end + 1):
        t = (frame - frame_start) / max(1, frame_end - frame_start)
        az = math.radians(start_azimuth + degrees * t)
        el = math.radians(elevation)
        cam.location = center + Vector((
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        )) * distance
        cam.keyframe_insert("location", frame=frame)
    if cam.animation_data and cam.animation_data.action:
        for fcurve in cam.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR'
    print(f"[camera] orbiting {degrees:.0f} degrees over frames {frame_start}-{frame_end}")


def main():
    args = parse_args()

    objs = bc.import_scene(args.html, target_fps=args.fps)
    parts = bc.classify(objs, args.manifest, expect_collision=args.expect_collision)

    scene = bpy.context.scene
    frame_start, frame_end = scene.frame_start, scene.frame_end
    if frame_end <= frame_start:
        raise SystemExit(
            f"[video] the scene has a single frame ({frame_start}..{frame_end}), so there "
            "is nothing to animate. Export with a --trajectory, and check the export "
            "reported more than one recorded frame.")
    print(f"[video] {frame_end - frame_start + 1} frames at {args.fps} fps "
          f"({(frame_end - frame_start + 1) / args.fps:.2f} s)")

    bc.improve_scene_quality()
    bc.restyle_maze(parts, args)
    bc.apply_lid_variant(parts, args)

    trace = None
    if args.trace_mode != "none":
        # One control point per frame, so the growth keyframes below advance the
        # drawn end in step with the marker rather than along arc length. See
        # blender_common.animate_trace_growth.
        n_frames = frame_end - frame_start + 1
        trace = bc.add_tip_trace(
            args, resample_frames=n_frames if args.trace_mode == "growing" else None)
        if trace is not None and args.trace_mode == "growing":
            bc.animate_trace_growth(trace, frame_start, frame_end)

    # Resolution before the camera fit, which reads the aspect ratio.
    bc.configure_output(args)
    bc.configure_render(args)

    # Fit on the union of every frame's geometry, so nothing swims out of shot
    # partway through. Sampling ~40 frames is enough for a 7-DOF arm.
    subject = animation_subject(parts, args.frame_subject, frame_start, frame_end)
    light_subject = parts["robot"] or parts["planks"]
    center, radius, azimuth, target, cam = bc.setup_camera(args, subject, light_subject)
    bc.setup_studio_lighting(args, center, radius, azimuth, target)
    bc.setup_world(args)

    if args.camera == "orbit":
        distance = (cam.location - Vector(target.location)).length
        for con in list(cam.constraints):
            cam.constraints.remove(con)
        con = cam.constraints.new(type='TRACK_TO')
        con.target = target
        con.track_axis = 'TRACK_NEGATIVE_Z'
        con.up_axis = 'UP_Y'
        elevation = args.camera_elevation if args.camera_elevation is not None \
            else bc.DEFAULT_ELEVATION
        orbit_camera(cam, target, Vector(target.location), distance, elevation,
                     azimuth, args.orbit_degrees, frame_start, frame_end)

    if args.camera_json:
        bc.write_camera_json(cam, args.camera_json)

    if args.preview_frames > 0:
        scene.frame_end = min(frame_end, frame_start + args.preview_frames - 1)
        print(f"[video] preview: rendering frames {frame_start}-{scene.frame_end} only "
              "(camera was still fitted to the whole animation)")

    os.makedirs(args.frames_dir, exist_ok=True)
    # Blender appends the frame number to this prefix.
    scene.render.filepath = os.path.join(os.path.abspath(args.frames_dir), "frame_")
    scene.render.image_settings.file_format = 'PNG'
    scene.frame_step = 1
    bpy.ops.render.render(animation=True)

    written = len([f for f in os.listdir(args.frames_dir) if f.endswith(".png")])
    print(f"[video] wrote {written} frames to {args.frames_dir}")
    print("MAZE_VIDEO_DONE")


def animation_subject(parts, choice, frame_start, frame_end, samples=40):
    """Objects to frame, plus a pseudo-object standing in for their swept extent.

    The camera has to contain the arm at EVERY frame, not just the first. Rather
    than fitting per frame (which would make the camera drift), this samples the
    animation and returns the moving objects' corners across all of it, so
    fit_frustum solves once for a box that holds the whole motion.
    """
    board = parts["planks"]
    robot = parts["robot"]
    if choice == "board":
        return board
    subject = robot if choice == "robot" else (board + robot)

    scene = bpy.context.scene
    step = max(1, (frame_end - frame_start) // samples)
    swept = _SweptCorners()
    for frame in range(frame_start, frame_end + 1, step):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj in robot:
            if obj.type != 'MESH':
                continue
            swept.add(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    print(f"[camera] swept the arm over {len(range(frame_start, frame_end + 1, step))} "
          "sampled frames for framing")
    return subject + [swept]


class _SweptCorners:
    """Quacks like a mesh object for corners_of(), carrying accumulated points.

    corners_of() asks each object for `.type` and its bound_box under
    matrix_world; this presents the swept point cloud through the same interface
    so the camera fit needs no special case.
    """

    type = 'MESH'

    def __init__(self):
        self._pts = []

    def add(self, points):
        self._pts.extend(points)

    @property
    def bound_box(self):
        if not self._pts:
            return [(0, 0, 0)] * 8
        lo = [min(p[i] for p in self._pts) for i in range(3)]
        hi = [max(p[i] for p in self._pts) for i in range(3)]
        return [(lo[0], lo[1], lo[2]), (lo[0], lo[1], hi[2]),
                (lo[0], hi[1], hi[2]), (lo[0], hi[1], lo[2]),
                (hi[0], lo[1], lo[2]), (hi[0], lo[1], hi[2]),
                (hi[0], hi[1], hi[2]), (hi[0], hi[1], lo[2])]

    @property
    def matrix_world(self):
        from mathutils import Matrix
        return Matrix.Identity(4)


if __name__ == "__main__":
    main()
