#!/usr/bin/env python3
"""Render the FR3 maze animation to an MP4, and check that it actually worked.

    python3 scripts/figures/render_video.py --html out/figures/scene/problem_108.html

    # cheap framing check: 24 frames, small, few samples
    python3 scripts/figures/render_video.py --preview-frames 24

    # slow motion, orbiting camera
    python3 scripts/figures/render_video.py --speed 0.5 --camera orbit

Stdlib only. Drives blender_maze_video.py for the frames, then ffmpeg for the
encode.

Two things this does that the sibling repo's equivalent does not, both because
that repo documents being bitten by them:

* It streams Blender's output and prints a heartbeat with the frame count, so a
  hung render is distinguishable from a slow one. `capture_output=True` makes
  those two look identical for hours.
* It keys the frame cache on the whole render request, so changing the lid
  variant or the trace re-renders rather than silently reusing frames.

Renders are serial and there is deliberately no --jobs flag: two Blender
processes sharing one GPU have deadlocked on this hardware.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_meshcat_importer  # noqa: E402
import maze_manifest  # noqa: E402
from blender_paths import BLENDER  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENE_DIR = os.path.join(_REPO_ROOT, "out", "figures", "scene")
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "out", "figures")

RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "blender_maze_video.py")
SENTINEL = "MAZE_VIDEO_DONE"

# Drake records meshcat animations at 64 fps by default. The importer
# interpolates DOWN to a requested rate but cannot invent frames, so a render
# rate above the recording rate would only duplicate them.
DEFAULT_RECORDING_FPS = 64.0


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", help="animated meshcat HTML (default: newest in the scene dir)")
    ap.add_argument("--manifest", help="planks.json (default: beside the HTML)")
    ap.add_argument("--trace", help="tip_path.json (default: beside the HTML)")
    ap.add_argument("--out", help="output MP4 (default: derived from the HTML name)")
    ap.add_argument("--frames-dir", help="frame cache directory (default: beside the MP4)")

    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed; 0.5 is half speed. Implemented by rendering "
                         "more frames, so it is limited by the recording rate")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower is better)")
    ap.add_argument("--hold-start", type=float, default=1.0,
                    help="seconds to hold the first frame before the motion starts "
                         "(default: 1.0). Applied at encode time by duplicating the "
                         "frame, so changing it does not re-render anything")
    ap.add_argument("--hold-end", type=float, default=1.5,
                    help="seconds to hold the last frame after the motion finishes "
                         "(default: 1.5), so the completed trace can be read")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--samples", type=int, default=64)

    ap.add_argument("--lid-variant", choices=("keep", "cull", "glass", "cutaway"),
                    default="keep")
    ap.add_argument("--cutaway-margin", type=float, default=0.02)
    ap.add_argument("--trace-mode", choices=("none", "full", "growing"), default="growing")

    ap.add_argument("--camera", choices=("fixed", "orbit"), default="fixed")
    ap.add_argument("--orbit-degrees", type=float, default=360.0)
    ap.add_argument("--preview-frames", type=int, default=0,
                    help="render only the first N frames, with the camera still fitted "
                         "to the whole animation")

    ap.add_argument("--force", action="store_true", help="ignore the frame cache")
    ap.add_argument("--keep-frames", action="store_true",
                    help="do not delete the frame directory after encoding")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--blender", default=BLENDER)

    passthrough = ap.add_argument_group("forwarded to Blender")
    passthrough.add_argument("--camera-azimuth", type=float, default=None)
    passthrough.add_argument("--camera-elevation", type=float, default=None)
    passthrough.add_argument("--camera-distance", type=float, default=None)
    passthrough.add_argument("--camera-target", default=None)
    passthrough.add_argument("--camera-json", default=None)
    passthrough.add_argument("--lens", type=float, default=None)
    passthrough.add_argument("--frame-margin", type=float, default=None)
    passthrough.add_argument("--frame-subject", default=None)
    passthrough.add_argument("--device", default="auto")
    passthrough.add_argument("--engine", default="CYCLES")
    passthrough.add_argument("--lid-alpha", type=float, default=None)
    passthrough.add_argument("--z-fight-nudge", type=float, default=None)
    passthrough.add_argument("--base-color", default=None)
    passthrough.add_argument("--wall-color", default=None)
    passthrough.add_argument("--floor-color", default=None)
    passthrough.add_argument("--lid-color", default=None)
    passthrough.add_argument("--robot-color", default=None)
    passthrough.add_argument("--trace-radius", type=float, default=None)
    passthrough.add_argument("--trace-lift", type=float, default=None)
    passthrough.add_argument("--trace-color", default=None)
    passthrough.add_argument("--trace-emission", type=float, default=None)
    passthrough.add_argument("--world-strength", type=float, default=None)
    passthrough.add_argument("--ambient", type=float, default=None)
    passthrough.add_argument("--light-strength", type=float, default=None)
    passthrough.add_argument("--key-elevation", type=float, default=None)
    # Video renders on the dark backdrop by default -- see the plan; stills stay
    # transparent for the paper page.
    passthrough.add_argument("--transparent-bg", action="store_true",
                             help="render a transparent background instead of the dark "
                                  "one (unusual for video; x264 has no alpha)")
    passthrough.add_argument("--expect-collision", action="store_true")
    return ap


PASSTHROUGH_FLAGS = (
    "camera_azimuth", "camera_elevation", "camera_distance", "camera_target",
    "camera_json", "lens", "frame_margin", "frame_subject", "device", "engine",
    "lid_alpha", "z_fight_nudge",
    "base_color", "wall_color", "floor_color", "lid_color", "robot_color",
    "trace_radius", "trace_lift", "trace_color", "trace_emission",
    "world_strength", "ambient", "light_strength", "key_elevation",
)


def render_fps_for(args):
    """Frames per second Blender should sample the recording at.

    `--speed 0.5` means the motion takes twice as long, which needs twice as many
    frames from the same recording. The importer can only interpolate downward,
    so this refuses rather than silently emitting duplicates.
    """
    needed = args.fps / max(args.speed, 1e-6)
    if needed > DEFAULT_RECORDING_FPS + 1e-6:
        raise SystemExit(
            f"--speed {args.speed} at --fps {args.fps} needs {needed:.1f} fps out of a "
            f"recording made at {DEFAULT_RECORDING_FPS:.0f} fps. The extra frames would "
            "be duplicates. Either raise the export's --fps and re-export, or slow down "
            "less.")
    return needed


def newest_animated_html(scene_dir):
    import glob
    candidates = sorted(glob.glob(os.path.join(scene_dir, "*.html")),
                        key=os.path.getmtime, reverse=True)
    if not candidates:
        raise SystemExit(
            f"No meshcat HTML in {scene_dir}. Generate one from a plan first:\n"
            "    python3 scripts/figures/export_maze_scene.py --trajectory <problem_N.txt>")
    return candidates[0]


def resolve_inputs(args):
    html = os.path.abspath(args.html or newest_animated_html(DEFAULT_SCENE_DIR))
    if not os.path.isfile(html):
        raise SystemExit(f"No such HTML: {html}")
    scene_dir = os.path.dirname(html)

    manifest = os.path.abspath(args.manifest or os.path.join(scene_dir, "planks.json"))
    if not os.path.isfile(manifest):
        raise SystemExit(f"No manifest at {manifest}; re-run export_maze_scene.py.")

    trace = None
    if args.trace_mode != "none":
        trace = os.path.abspath(args.trace or os.path.join(scene_dir, "tip_path.json"))
        if not os.path.isfile(trace):
            if args.trace:
                raise SystemExit(f"No such tip path: {trace}")
            print(f"[input] no tip_path.json beside {os.path.basename(html)}; "
                  "rendering without a trace")
            trace = None

    meta_path = os.path.join(scene_dir, "scene_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    if meta.get("animated") is False:
        raise SystemExit(
            f"{os.path.basename(html)} was exported as a STATIC scene, so there is no "
            "animation to render. Re-export with --trajectory <problem_N.txt>.")
    return html, manifest, trace, meta


def cutaway_names(trace_path, margin):
    if not trace_path:
        print("[cutaway] no tip path, so no lids can be selected")
        return []
    with open(trace_path, "r") as f:
        xyz = json.load(f)["xyz"]
    planks = maze_manifest.load_planks()
    names = maze_manifest.select_cutaway_lids(planks, [(p[0], p[1]) for p in xyz], margin)
    total = maze_manifest.role_counts(planks)[maze_manifest.ROLE_LID]
    print(f"[cutaway] {len(names)}/{total} lids occlude the path: "
          f"{' '.join(names) if names else '(none)'}")
    return names


def child_args(args, html, manifest, trace, frames_dir, cull, render_fps):
    argv = ["--html", html, "--manifest", manifest, "--frames-dir", frames_dir,
            "--fps", str(render_fps), "--lid-variant", args.lid_variant,
            "--trace-mode", args.trace_mode, "--camera", args.camera,
            "--orbit-degrees", str(args.orbit_degrees),
            "--resolution", str(args.width), str(args.height),
            "--samples", str(args.samples)]
    if trace:
        argv += ["--trace-json", trace]
    if args.lid_variant == "cutaway" and cull:
        argv += ["--cull-planks"] + list(cull)
    if args.preview_frames > 0:
        argv += ["--preview-frames", str(args.preview_frames)]
    if not args.transparent_bg:
        argv.append("--opaque-bg")
    if args.expect_collision:
        argv.append("--expect-collision")

    for name in PASSTHROUGH_FLAGS:
        value = getattr(args, name, None)
        if value is None:
            continue
        argv += ["--" + name.replace("_", "-"), str(value)]
    return argv


def request_key(args, html, manifest, trace, argv):
    def stat(path):
        if not path or not os.path.exists(path):
            return None
        st = os.stat(path)
        return {"path": path, "mtime": st.st_mtime, "size": st.st_size}

    return {"html": stat(html), "manifest": stat(manifest), "trace": stat(trace),
            "argv": [a for a in argv if a != "--frames-dir"],
            "preview_frames": args.preview_frames}


def cached(frames_dir, key, force):
    side = os.path.join(frames_dir, "render_config.json")
    if force or not os.path.isfile(side):
        return False
    try:
        with open(side, "r") as f:
            if json.load(f) != key:
                return False
    except Exception:
        return False
    return bool(frame_files(frames_dir))


def frame_files(frames_dir):
    if not os.path.isdir(frames_dir):
        return []
    return sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))


def run_blender(args, argv, frames_dir):
    """Run Blender, streaming its output with a frame-count heartbeat.

    A Cycles render is bursty between frames, so a single idle moment means
    nothing; a frame count that stops climbing is the signal that matters.
    """
    cmd = [args.blender, "--background", "--python", RENDER_SCRIPT, "--"] + argv
    if args.verbose:
        print("  " + " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    last_beat = time.time()
    last_count = -1
    started = time.time()

    try:
        for line in proc.stdout:
            lines.append(line)
            stripped = line.rstrip()
            if stripped.startswith(("[import]", "[classify]", "[lids]", "[material]",
                                    "[trace]", "[camera]", "[light]", "[output]",
                                    "[render]", "[video]")):
                print("  " + stripped)
            if args.verbose:
                print("  | " + stripped)

            now = time.time()
            if now - last_beat >= 30.0:
                count = len(frame_files(frames_dir))
                state = "stalled" if count == last_count else "rendering"
                print(f"  [heartbeat] {count} frames, {now - started:.0f}s elapsed, {state}")
                last_beat, last_count = now, count
            if now - started > args.timeout:
                proc.kill()
                raise SystemExit(f"[video] timed out after {args.timeout}s")
    finally:
        proc.wait()

    stdout = "".join(lines)
    if proc.returncode != 0 or SENTINEL not in stdout:
        print(f"[video] FAILED (exit {proc.returncode}, "
              f"sentinel {'present' if SENTINEL in stdout else 'ABSENT'})")
        print(stdout[-6000:])
        raise SystemExit(1)
    return stdout


def verify(stdout, frames_dir, args):
    problems = []
    frames = frame_files(frames_dir)
    if not frames:
        problems.append(f"no frames written to {frames_dir}")
    if "[classify] robot=" not in stdout:
        problems.append("no [classify] line")
    else:
        line = next(l for l in stdout.splitlines() if l.startswith("[classify] robot="))
        if "robot=0" in line:
            problems.append("classified 0 robot objects")
        if "planks=124/124" not in line:
            problems.append(f"not all 124 planks present: {line}")
    if args.engine == "CYCLES" and args.device in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        if f"CYCLES on GPU via {args.device}" not in stdout:
            problems.append(f"--device {args.device} requested but not reported in use")
    if problems:
        for p in problems:
            print(f"[verify] {p}")
        raise SystemExit(1)
    return frames


def encode(frames_dir, out, fps, crf, hold_start=0.0, hold_end=0.0):
    """PNG sequence -> H.264. libx264, never NVENC: measured slower here and
    needing far more bitrate for the same quality.

    The holds are done here with tpad rather than by rendering duplicate frames,
    so their length is free to change and costs no GPU time.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH; cannot encode. The frames are in "
                         f"{frames_dir}.")

    # Even dimensions and yuv420p, or the file will not play in most browsers.
    filters = ["pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    if hold_start > 0 or hold_end > 0:
        pad = "tpad="
        if hold_start > 0:
            pad += f"start_mode=clone:start_duration={hold_start:g}:"
        if hold_end > 0:
            pad += f"stop_mode=clone:stop_duration={hold_end:g}:"
        filters.append(pad.rstrip(":"))

    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(frames_dir, "frame_%04d.png"),
           "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-vf", ",".join(filters), "-pix_fmt", "yuv420p",
           "-an", out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(out):
        print(result.stderr[-4000:])
        raise SystemExit("[video] ffmpeg failed")
    print(f"[video] encoded {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    args = build_parser().parse_args()

    if not check_meshcat_importer.check(verbose=True):
        raise SystemExit(
            "The meshcat_html_importer add-on is missing or wrong. Without it Blender "
            "renders an EMPTY scene and exits 0. Verify with:\n"
            "    python3 scripts/figures/check_meshcat_importer.py --probe")

    html, manifest, trace, meta = resolve_inputs(args)
    render_fps = render_fps_for(args)
    stem = os.path.splitext(os.path.basename(html))[0]

    out = os.path.abspath(args.out or os.path.join(
        DEFAULT_OUT_DIR, f"{stem}_{args.lid_variant}.mp4"))
    frames_dir = os.path.abspath(args.frames_dir or os.path.join(
        DEFAULT_OUT_DIR, "frames", f"{stem}_{args.lid_variant}"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"[input] html={os.path.relpath(html, _REPO_ROOT)}")
    print(f"[input] trace={os.path.relpath(trace, _REPO_ROOT) if trace else '(none)'}")
    if meta.get("recording"):
        rec = meta["recording"]
        print(f"[input] recording {rec['frames']} frames over {rec['duration']:.2f}s "
              f"at {rec['fps']}fps")
    print(f"[video] sampling the recording at {render_fps:.1f} fps for "
          f"{args.fps} fps output at {args.speed}x speed")

    cull = cutaway_names(trace, args.cutaway_margin) if args.lid_variant == "cutaway" else []
    argv = child_args(args, html, manifest, trace, frames_dir, cull, render_fps)
    key = request_key(args, html, manifest, trace, argv)

    if cached(frames_dir, key, args.force):
        frames = frame_files(frames_dir)
        print(f"[video] reusing {len(frames)} cached frames in "
              f"{os.path.relpath(frames_dir, _REPO_ROOT)}")
    else:
        for stale in frame_files(frames_dir):
            os.remove(os.path.join(frames_dir, stale))
        stdout = run_blender(args, argv, frames_dir)
        frames = verify(stdout, frames_dir, args)
        with open(os.path.join(frames_dir, "render_config.json"), "w") as f:
            json.dump(key, f, indent=2)

    print(f"[video] {len(frames)} frames"
          + (f", holding {args.hold_start:g}s at the start and {args.hold_end:g}s at "
             "the end" if (args.hold_start or args.hold_end) else ""))
    encode(frames_dir, out, args.fps, args.crf, args.hold_start, args.hold_end)

    if not args.keep_frames and args.preview_frames == 0:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[video] removed the frame cache (pass --keep-frames to keep it)")

    print(f"[video] done: {os.path.relpath(out, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
