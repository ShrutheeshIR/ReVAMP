#!/usr/bin/env python3
"""Drive Blender to render maze stills, and check that it actually worked.

    # one static render at the default framing
    python3 scripts/figures/render_still.py

    # cheap camera search -- 15 previews, a couple of minutes
    python3 scripts/figures/render_still.py --preview \
        --azimuth-sweep 0 20 35 50 65 --elevation-sweep 25 40 55 --contact-sheet

    # compare all four lid treatments
    python3 scripts/figures/render_still.py --preview --lid-variant all --contact-sheet

    # the paper's chronophotography figure
    python3 scripts/figures/render_still.py --mode ghosts --n-poses 6

Stdlib only, and deliberately so: this runs in the ordinary interpreter, and
importing pydrake here would tie the whole render path to a Drake install it
does not need.

Everything this script does beyond assembling a command line is checking. A
Blender render fails in ways that exit 0 and leave a plausible-looking PNG: the
add-on can be missing (empty scene), the GPU can silently fall back to the CPU,
the HTML can be stale. So each stage of blender_maze_still.py prints a countable
line and this driver requires them.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_meshcat_importer  # noqa: E402
import maze_manifest  # noqa: E402
from blender_paths import BLENDER  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENE_DIR = os.path.join(_REPO_ROOT, "out", "figures", "scene")
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "out", "figures")

RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "blender_maze_still.py")
SENTINEL = "MAZE_STILL_DONE"

LID_VARIANTS = ("keep", "cull", "glass", "cutaway")

# Cheap enough to iterate on: seconds rather than minutes, which is the whole
# point of bracketing camera and material values at one frame instead of paying
# for a full render to find out.
PREVIEW_RESOLUTION = (1000, 563)
PREVIEW_SAMPLES = 24


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--html", help=f"meshcat HTML (default: newest in "
                                   f"{os.path.relpath(DEFAULT_SCENE_DIR, _REPO_ROOT)})")
    ap.add_argument("--manifest", help="planks.json (default: beside the HTML)")
    ap.add_argument("--trace", help="tip_path.json (default: beside the HTML if present)")
    ap.add_argument("--no-trace", action="store_true", help="never draw the tip trace")
    ap.add_argument("--out", help="output PNG (default: derived from the settings)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    ap.add_argument("--mode", choices=("static", "ghosts"), default="static")
    ap.add_argument("--lid-variant", choices=LID_VARIANTS + ("all",), default="keep",
                    help="'all' renders each treatment in turn for comparison")
    ap.add_argument("--cutaway-margin", type=float, default=0.02,
                    help="metres to inflate each lid's footprint when deciding whether "
                         "it occludes the traced path (default: 0.02)")

    ap.add_argument("--preview", action="store_true",
                    help=f"{PREVIEW_RESOLUTION[0]}x{PREVIEW_RESOLUTION[1]} at "
                         f"{PREVIEW_SAMPLES} samples, for iterating")
    ap.add_argument("--azimuth-sweep", type=float, nargs="*", default=None)
    ap.add_argument("--elevation-sweep", type=float, nargs="*", default=None)
    ap.add_argument("--contact-sheet", nargs="?", const="", default=None,
                    metavar="PATH", help="montage the rendered set for comparison")

    ap.add_argument("--force", action="store_true", help="ignore the render cache")
    ap.add_argument("--verbose", action="store_true", help="stream Blender's output")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--blender", default=BLENDER)

    # Forwarded verbatim to blender_maze_still.py. Kept as a passthrough list
    # rather than redeclared so the two cannot drift apart.
    passthrough = ap.add_argument_group("forwarded to Blender")
    passthrough.add_argument("--resolution", type=int, nargs=2, default=None)
    passthrough.add_argument("--samples", type=int, default=None)
    passthrough.add_argument("--device", default="auto")
    passthrough.add_argument("--engine", default="CYCLES")
    passthrough.add_argument("--camera-azimuth", type=float, default=None)
    passthrough.add_argument("--camera-elevation", type=float, default=None)
    passthrough.add_argument("--camera-distance", type=float, default=None)
    passthrough.add_argument("--camera-target", default=None)
    passthrough.add_argument("--camera-json", default=None)
    passthrough.add_argument("--lens", type=float, default=None)
    passthrough.add_argument("--frame-margin", type=float, default=None)
    passthrough.add_argument("--frame-subject", default=None)
    passthrough.add_argument("--frame", type=int, default=None)
    passthrough.add_argument("--n-poses", type=int, default=None)
    passthrough.add_argument("--t-start", type=float, default=None)
    passthrough.add_argument("--t-end", type=float, default=None)
    passthrough.add_argument("--spacing", default=None)
    passthrough.add_argument("--mid-bias", type=float, default=None)
    passthrough.add_argument("--pose-nudge", type=float, nargs="*", default=None)
    passthrough.add_argument("--drop-poses", type=int, nargs="*", default=None)
    passthrough.add_argument("--alpha", type=float, default=None)
    passthrough.add_argument("--endpoint-alpha", type=float, default=None)
    passthrough.add_argument("--static-tol", type=float, default=None)
    passthrough.add_argument("--ghost-shadows", default=None)
    passthrough.add_argument("--lid-alpha", type=float, default=None)
    passthrough.add_argument("--z-fight-nudge", type=float, default=None)
    passthrough.add_argument("--maze-material", choices=("flat", "wood"), default=None)
    passthrough.add_argument("--base-color", default=None)
    passthrough.add_argument("--wall-color", default=None)
    passthrough.add_argument("--floor-color", default=None)
    passthrough.add_argument("--lid-color", default=None)
    passthrough.add_argument("--robot-color", default=None)
    passthrough.add_argument("--trace-radius", type=float, default=None)
    passthrough.add_argument("--trace-lift", type=float, default=None)
    passthrough.add_argument("--trace-color", default=None)
    passthrough.add_argument("--trace-emission", type=float, default=None)
    passthrough.add_argument("--trace-progress", type=float, default=None)
    passthrough.add_argument("--opaque-bg", action="store_true")
    passthrough.add_argument("--world-strength", type=float, default=None)
    passthrough.add_argument("--ambient", type=float, default=None)
    passthrough.add_argument("--light-strength", type=float, default=None)
    passthrough.add_argument("--key-elevation", type=float, default=None)
    passthrough.add_argument("--expect-collision", action="store_true")
    return ap


PASSTHROUGH_FLAGS = (
    "resolution", "samples", "device", "engine",
    "camera_azimuth", "camera_elevation", "camera_distance", "camera_target",
    "camera_json", "lens", "frame_margin", "frame_subject", "frame",
    "n_poses", "t_start", "t_end", "spacing", "mid_bias", "pose_nudge", "drop_poses",
    "alpha", "endpoint_alpha", "static_tol", "ghost_shadows",
    "lid_alpha", "z_fight_nudge", "maze_material",
    "base_color", "wall_color", "floor_color", "lid_color", "robot_color",
    "trace_radius", "trace_lift", "trace_color", "trace_emission", "trace_progress",
    "world_strength", "ambient", "light_strength", "key_elevation",
)
STORE_TRUE_FLAGS = ("opaque_bg", "expect_collision")


def newest_html(scene_dir):
    candidates = sorted(glob.glob(os.path.join(scene_dir, "*.html")),
                        key=os.path.getmtime, reverse=True)
    if not candidates:
        raise SystemExit(
            f"No meshcat HTML found in {scene_dir}.\n"
            "Generate one first:\n"
            "    python3 scripts/figures/export_maze_scene.py")
    return candidates[0]


def resolve_inputs(args):
    html = os.path.abspath(args.html or newest_html(DEFAULT_SCENE_DIR))
    if not os.path.isfile(html):
        raise SystemExit(f"No such HTML: {html}")
    scene_dir = os.path.dirname(html)

    manifest = os.path.abspath(args.manifest or os.path.join(scene_dir, "planks.json"))
    if not os.path.isfile(manifest):
        raise SystemExit(
            f"No manifest at {manifest}. It is written alongside the HTML by "
            "export_maze_scene.py; re-run that, or pass --manifest.")

    trace = None
    if not args.no_trace:
        trace = os.path.abspath(args.trace or os.path.join(scene_dir, "tip_path.json"))
        if not os.path.isfile(trace):
            if args.trace:
                raise SystemExit(f"No such tip path: {trace}")
            trace = None  # a static scene simply has none
    return html, manifest, trace


def cutaway_names(manifest_path, trace_path, margin):
    """Lids that occlude the traced path, computed here so the choice is visible.

    Done driver-side rather than inside Blender so the decision is a printable
    list of names that can be checked, reproduced, and pinned.
    """
    if not trace_path:
        print("[cutaway] no tip path available, so no lids can be selected. The cutaway "
              "variant will render identically to 'keep'.")
        return []
    with open(trace_path, "r") as f:
        xyz = json.load(f)["xyz"]
    planks = maze_manifest.load_planks()
    names = maze_manifest.select_cutaway_lids(planks, [(p[0], p[1]) for p in xyz], margin)
    total = maze_manifest.role_counts(planks)[maze_manifest.ROLE_LID]
    print(f"[cutaway] {len(names)} of {total} lids occlude the path (margin "
          f"{margin * 1000:.0f} mm): {' '.join(names) if names else '(none)'}")
    return names


def output_name(args, lid_variant, azimuth, elevation):
    if args.out:
        return os.path.abspath(args.out)
    az = azimuth if azimuth is not None else "def"
    el = elevation if elevation is not None else "def"
    az = f"{az:03.0f}" if isinstance(az, float) else az
    el = f"{el:02.0f}" if isinstance(el, float) else el
    stem = f"{args.mode}_{lid_variant}_az{az}_el{el}"
    if args.preview:
        stem = "preview_" + stem
    return os.path.abspath(os.path.join(args.out_dir, stem + ".png"))


def child_args(args, html, manifest, trace, out, lid_variant, cull, azimuth, elevation):
    """Assemble the argument list passed after Blender's `--`."""
    argv = ["--html", html, "--manifest", manifest, "--out", out,
            "--mode", args.mode, "--lid-variant", lid_variant]

    if trace:
        argv += ["--trace-json", trace]
    if lid_variant == "cutaway" and cull:
        argv += ["--cull-planks"] + list(cull)

    if args.preview:
        argv += ["--resolution", str(PREVIEW_RESOLUTION[0]), str(PREVIEW_RESOLUTION[1]),
                 "--samples", str(PREVIEW_SAMPLES)]

    for name in PASSTHROUGH_FLAGS:
        value = getattr(args, name, None)
        if value is None:
            continue
        # --preview has already set these; an explicit flag still wins.
        if args.preview and name in ("resolution", "samples") and value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, (list, tuple)):
            argv += [flag] + [str(v) for v in value]
        else:
            argv += [flag, str(value)]

    for name in STORE_TRUE_FLAGS:
        if getattr(args, name, False):
            argv.append("--" + name.replace("_", "-"))

    if azimuth is not None and args.camera_azimuth is None:
        argv += ["--camera-azimuth", str(azimuth)]
    if elevation is not None and args.camera_elevation is None:
        argv += ["--camera-elevation", str(elevation)]
    return argv


def request_key(args, html, manifest, trace, argv):
    """The full render request, for the frame cache.

    Keyed on everything that can change the picture, and compared by exact
    equality -- every value in here is one this driver wrote, so there is no
    float round-trip through Blender to make an exact comparison fail.
    """
    def stat(path):
        if not path or not os.path.exists(path):
            return None
        st = os.stat(path)
        return {"path": path, "mtime": st.st_mtime, "size": st.st_size}

    return {"html": stat(html), "manifest": stat(manifest), "trace": stat(trace),
            "argv": [a for a in argv if a not in ("--out",)],
            "preview": bool(args.preview), "mode": args.mode}


def cached(out, key, force):
    side = out + ".request.json"
    if force or not (os.path.exists(out) and os.path.exists(side)):
        return False
    try:
        with open(side, "r") as f:
            return json.load(f) == key
    except Exception:
        return False


def run_blender(args, argv, out, key):
    cmd = [args.blender, "--background", "--python", RENDER_SCRIPT, "--"] + argv
    print(f"[render] {os.path.basename(out)}")
    if args.verbose:
        print("  " + " ".join(cmd))

    result = subprocess.run(cmd, capture_output=not args.verbose, text=True,
                            timeout=args.timeout)
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0 or SENTINEL not in stdout:
        # stderr first: Blender writes a wall of startup and glTF chatter to
        # stdout before anything diagnostic, so the useful line is usually here.
        print(f"[render] FAILED (exit {result.returncode}, "
              f"sentinel {'present' if SENTINEL in stdout else 'ABSENT'})")
        if stderr.strip():
            print(stderr.strip()[-6000:])
        if stdout.strip():
            print(stdout.strip()[-4000:])
        raise SystemExit(1)

    echo(stdout, args)
    verify(stdout, out, args)

    with open(out + ".request.json", "w") as f:
        json.dump(key, f, indent=2)
    return stdout


def echo(stdout, args):
    """Surface the stages' own accounting, which is the only proof of what ran."""
    for line in stdout.splitlines():
        if line.startswith(("[import]", "[classify]", "[lids]", "[material]", "[trace]",
                            "[poses]", "[ghost]", "[static]", "[camera]", "[light]",
                            "[output]", "[render]")):
            print("  " + line)


def verify(stdout, out, args):
    """Assert the render did what it claims. Each check has a silent failure mode."""
    problems = []

    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        problems.append(f"no output written to {out}")

    if "[classify] robot=" not in stdout:
        problems.append("no [classify] line -- the scene was never classified")
    else:
        line = next(l for l in stdout.splitlines() if l.startswith("[classify] robot="))
        if "robot=0" in line:
            problems.append("classified 0 robot objects -- the arm is missing")
        if "planks=124/124" not in line:
            problems.append(f"not all 124 planks present: {line}")

    if "[lids] variant=" not in stdout:
        problems.append("no [lids] line -- the lid variant was never applied")

    # A silent CPU fallback is ~20x slower and otherwise identical to success.
    if args.engine == "CYCLES" and args.device in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        if f"CYCLES on GPU via {args.device}" not in stdout:
            problems.append(f"--device {args.device} was requested but the render did not "
                            f"report using it")

    if problems:
        for problem in problems:
            print(f"[verify] {problem}")
        raise SystemExit(1)


def main():
    args = build_parser().parse_args()

    if not check_meshcat_importer.check(verbose=True):
        raise SystemExit(
            "The meshcat_html_importer add-on is missing or wrong. Without it Blender "
            "renders an EMPTY scene and exits 0. Verify with:\n"
            "    python3 scripts/figures/check_meshcat_importer.py --probe")

    html, manifest, trace = resolve_inputs(args)
    print(f"[input] html={os.path.relpath(html, _REPO_ROOT)}")
    print(f"[input] manifest={os.path.relpath(manifest, _REPO_ROOT)}")
    print(f"[input] trace={os.path.relpath(trace, _REPO_ROOT) if trace else '(none)'}")

    variants = LID_VARIANTS if args.lid_variant == "all" else (args.lid_variant,)
    cull = (cutaway_names(manifest, trace, args.cutaway_margin)
            if "cutaway" in variants else [])

    azimuths = args.azimuth_sweep or [args.camera_azimuth]
    elevations = args.elevation_sweep or [args.camera_elevation]

    outputs, labels = [], []
    for variant in variants:
        for azimuth in azimuths:
            for elevation in elevations:
                out = output_name(args, variant, azimuth, elevation)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                argv = child_args(args, html, manifest, trace, out, variant, cull,
                                  azimuth, elevation)
                key = request_key(args, html, manifest, trace, argv)

                if cached(out, key, args.force):
                    print(f"[render] {os.path.basename(out)} (cached)")
                else:
                    # Serial, always. Two Blender processes sharing this laptop's
                    # GPU have deadlocked; there is deliberately no --jobs flag.
                    run_blender(args, argv, out, key)

                outputs.append(out)
                labels.append(sheet_label(args, variant, azimuth, elevation))

    print(f"[render] {len(outputs)} image(s):")
    for out in outputs:
        print(f"  {os.path.relpath(out, _REPO_ROOT)}")

    if args.contact_sheet is not None and len(outputs) > 1:
        make_contact_sheet(args, outputs, labels)


def sheet_label(args, variant, azimuth, elevation):
    bits = []
    if len(LID_VARIANTS) > 1 and args.lid_variant == "all":
        bits.append(variant)
    if azimuth is not None:
        bits.append(f"az {azimuth:.0f}")
    if elevation is not None:
        bits.append(f"el {elevation:.0f}")
    return ", ".join(bits) or variant


def make_contact_sheet(args, outputs, labels):
    sheet = args.contact_sheet or os.path.join(args.out_dir, "contact_sheet.png")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_sheet.py")
    # Images first, then --labels last: --labels takes nargs="*", so anything
    # after it is swallowed as a label rather than parsed as a positional.
    cmd = [sys.executable, script] + outputs + ["--out", sheet, "--bg", "checker"]
    cmd += ["--labels"] + labels
    print(f"[sheet] {os.path.relpath(sheet, _REPO_ROOT)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
