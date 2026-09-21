"""A resumable, multi-stage build driver -- "one command rebuilds this deliverable."

Copy this file into your project, replace the ``STAGES`` list at the bottom
with your own pipeline, and adjust ``check_prereqs``'s configuration to match
what your stages actually need. It has no third-party dependencies.

A build is a list of **stages**, each producing one output file by running a
subprocess (typically another script in your project). Re-running the driver
skips any stage whose output already exists, is large enough to be plausible,
and is not stale -- so the common case ("I changed one thing, rebuild") only
re-does the work that actually needs it, while a from-scratch run (delete
everything, or ``--force``) reproduces the whole deliverable from nothing.

## What "stale" means

Staleness is **output older than any of its declared sources**, never mere
absence of the output. A stage's sources are the union of:

1. Any argv entry that names an existing file with a source-shaped extension
   (see ``SOURCE_EXTENSIONS`` below) -- typically the stage's own script, and
   any input file it takes as a CLI argument.
2. The **transitive closure of in-project modules that script imports**,
   found by ``ast.parse``-ing it and walking its ``Import`` / ``ImportFrom``
   nodes -- never by executing the script (importing it for real could have
   arbitrary side effects, be slow, or simply fail outside its subprocess
   environment). This is what catches "I edited a shared helper module that
   three stages import but none of them name on their command line."
3. An explicit ``deps`` list in the stage's ``opts``, for inputs a script
   hardcodes internally rather than accepting as an argument (a default
   input path, a constants file it reads by convention) -- and, typically,
   for an "assemble everything" final stage whose real dependency is every
   other stage's output.

## Two blind spots -- both by design, both worth knowing before you rely on this

**(a) A constant edited inside this driver file itself marks nothing stale.**
If a stage's command-line arguments are built from a literal in *this* file
(a `--title "..."` string, a resolution constant) rather than read from a
file on disk, editing that literal does not change any file's mtime that the
staleness scan can see -- because this driver script is not, itself, named in
any stage's own argv. Re-run with ``--force-from <stage>`` after that kind of
edit; do not expect the automatic staleness check to notice it.

**(b) Output-shaped paths must be excluded from the dependency scan.** If a
stage's own output path is fed back into its argv (or into a later stage that
happens to sit upstream in some other sense) and gets picked up as a
"source," the stage will see its own output as an input that is always at
least as new as itself, and will look perpetually fresh or perpetually stale
depending on how the comparison lands -- either way, wrong. Keep
``SOURCE_EXTENSIONS`` matching your actual *input* file types (scripts,
configs, scene files) and do not let it accidentally match your output
extension too.

## One more sharp edge

An mtime-based scheme is **correct but conservative**: editing a comment or
reformatting whitespace in a stage's script bumps its mtime and marks the
stage stale, even though its actual output would come out byte-identical.
That is the intended trade-off (missing a real staleness is worse than one
extra unnecessary re-run) -- but if a re-run is expensive and you are certain
the edit could not change the output, ``touch``ing the stage's *output* file
after the edit is the deliberate way to skip it.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time

PY = sys.executable

# --------------------------------------------------------------------------
# Configuration -- edit these for your project.
# --------------------------------------------------------------------------

# The directory stage scripts live in, used to resolve an imported module
# name (e.g. "import camera_utils") to a file on disk (e.g.
# "<SCRIPTS_DIR>/camera_utils.py") when walking the import graph. Point this
# at wherever your pipeline's own helper modules live.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Only argv entries whose extension is in this set, AND which name a file
# that currently exists on disk, are treated as declared sources (see blind
# spot (b) above -- this must not overlap your stages' *output* extensions).
SOURCE_EXTENSIONS = (".py", ".html", ".json", ".yaml", ".yml")

# External binaries every stage collectively needs on PATH. Checked once, up
# front, by check_prereqs() -- a missing one should cost this script a
# fraction of a second to report, not surface as a subprocess failure after
# an hour of upstream stages have already run.
REQUIRED_BINARIES = [
    # "ffmpeg", "ffprobe", "blender",
]

# Absolute font paths your stages load directly (e.g. for a text-drawing
# library with no fallback font search). Fill in real paths if you have any;
# leave empty otherwise.
REQUIRED_FONTS = [
    # "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Python packages that must be importable by the interpreter that will
# actually run your stages. Stages run as subprocesses (see run_stage), so
# "importable in the process running this driver" is not good enough --
# each is checked with `sys.executable -c "import <module>"` specifically.
REQUIRED_MODULES = [
    # "numpy", "matplotlib",
]


# --------------------------------------------------------------------------
# Import-graph staleness.
# --------------------------------------------------------------------------

def local_imports(path, seen=None):
    """Absolute paths of the in-project modules ``path`` imports, transitively.

    Only resolves imports to files that actually sit in SCRIPTS_DIR --
    standard-library and third-party imports are silently skipped, since
    editing numpy is not this project's staleness concern. Uses ``ast.parse``
    rather than importing the module for real: importing has side effects
    (and a module written to run only as ``__main__`` under a subprocess
    environment may not even import cleanly here).

    Known limitation: only absolute imports are followed. A relative import
    (``from . import helper``) is skipped, so if you organise your stage
    scripts as a package rather than as a flat directory, editing a shared
    helper reached only that way will NOT mark dependent stages stale --
    the exact failure this mechanism exists to prevent. Either keep the
    stage scripts flat, or list such helpers explicitly in a stage's
    ``deps``.
    """
    seen = seen if seen is not None else set()
    path = os.path.abspath(path)
    if path in seen or not os.path.isfile(path):
        return seen
    seen.add(path)
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return seen
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    for n in names:
        mod = os.path.join(SCRIPTS_DIR, n + ".py")
        if os.path.isfile(mod):
            local_imports(mod, seen)
    return seen


def stage_sources(argv, opts):
    """Everything a stage's output depends on, as absolute paths.

    See the module docstring's "What 'stale' means" section for the three
    contributing sets this combines.
    """
    named = [os.path.abspath(a) for a in argv if isinstance(a, str)
             and a.endswith(SOURCE_EXTENSIONS) and os.path.isfile(a)]
    imported = set()
    for p in named:
        if p.endswith(".py"):
            local_imports(p, imported)
    return (named + sorted(imported - set(named))
            + [os.path.abspath(d) for d in opts.get("deps", ())])


# --------------------------------------------------------------------------
# Running one stage.
# --------------------------------------------------------------------------

def run_stage(stage, args):
    """Run one ``(name, output, argv, min_size_bytes, opts)`` stage, or skip it.

    ``opts`` is an optional dict (stages may omit it) that may carry:
      - ``deps``: extra source paths, for inputs the script hardcodes rather
        than takes on argv (see the module docstring).
      - ``env``: a dict, or a zero-arg callable returning one, of extra
        environment variables to set for the subprocess. A callable is
        useful when the right value is only known after ``check_prereqs``
        has run (e.g. a resolved install path).
      - ``cwd``: working directory for the subprocess, if not the current one.
    """
    name, out, argv, min_size = stage[:4]
    opts = stage[4] if len(stage) > 4 else {}

    stale = [p for p in stage_sources(argv, opts)
             if os.path.exists(out) and os.path.exists(p)
             and os.path.getmtime(p) > os.path.getmtime(out)]

    if (os.path.exists(out) and os.path.getsize(out) >= min_size
            and not stale and not args._forcing(name)):
        print(f"[skip] {name}: {out} already built")
        return
    if stale and not args._forcing(name):
        # Report the MOST recently changed source, not just the first one in
        # an arbitrary scan order -- when several inputs are stale, the newest
        # is the one that most likely explains why you are looking at this.
        newest = max(stale, key=os.path.getmtime)
        extra = f" (+{len(stale) - 1} more)" if len(stale) > 1 else ""
        print(f"[stale] {name}: {newest} is newer than {out}{extra}")

    if args.dry_run:
        print(f"[would run] {name}: {' '.join(str(a) for a in argv[:6])} ...")
        return

    print(f"[run ] {name}")
    t0 = time.time()

    env = dict(os.environ)
    env_opt = opts.get("env", {})
    env.update(env_opt() if callable(env_opt) else env_opt)
    r = subprocess.run(argv, cwd=opts.get("cwd", None), env=env)
    if r.returncode != 0:
        print(f"\n[FAIL] {name} exited {r.returncode}")
        sys.exit(r.returncode)

    # A zero exit is NOT proof the stage produced anything. This is not a
    # theoretical worry: a crashed renderer that still prints its usual
    # closing status lines, or a stage that silently no-ops because its own
    # input looked already-built to it, both exit 0 while shipping either
    # nothing or the *previous* run's output untouched. Re-check for real.
    if not os.path.exists(out):
        print(f"\n[FAIL] {name} exited 0 but did not write {out}")
        sys.exit(1)
    size = os.path.getsize(out)
    if size < min_size:
        print(f"\n[FAIL] {name} wrote only {size} bytes to {out} "
              f"(expected >= {min_size}) -- treating as a failed run")
        sys.exit(1)
    print(f"[ok  ] {name}  {time.time() - t0:.0f}s  -> {out} ({size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------
# Prerequisites, checked up front.
# --------------------------------------------------------------------------

def check_prereqs():
    """Fail before spending real time on stage 1, not while stage 5 crashes.

    Checks, in order: external binaries on PATH, fonts at their absolute
    paths, and Python packages importable by the *specific interpreter*
    (``sys.executable``) that will run the stages -- since stages run as
    subprocesses, a package installed in whatever environment launched this
    driver does not help if the stages themselves run under a different one.
    """
    problems = []

    for tool in REQUIRED_BINARIES:
        if shutil.which(tool) is None:
            problems.append(f"{tool} not on PATH")

    for font in REQUIRED_FONTS:
        if not os.path.isfile(font):
            problems.append(f"missing font: {font}")

    for module in REQUIRED_MODULES:
        r = subprocess.run([PY, "-c", f"import {module}"],
                            capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(f"{module!r} is not importable by {PY}")

    if problems:
        print("Cannot build:\n")
        for p in problems:
            print("  * " + p)
        sys.exit(1)


# --------------------------------------------------------------------------
# CLI driver -- generic, works over whatever STAGES list is passed in.
# --------------------------------------------------------------------------

def main(stages):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild every stage")
    ap.add_argument("--force-from", metavar="STAGE", default=None,
                     help="rebuild this stage and every stage after it")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the plan without running or fetching anything")
    ap.add_argument("--list", action="store_true",
                     help="print the stage names and exit")
    args = ap.parse_args()

    names = [s[0] for s in stages]
    if args.list:
        for n in names:
            print(n)
        return

    if args.force_from and args.force_from not in names:
        ap.error(f"unknown stage {args.force_from!r}; --list shows them")
    start = names.index(args.force_from) if args.force_from else None

    def _forcing(name):
        if args.force:
            return True
        return start is not None and names.index(name) >= start

    args._forcing = _forcing

    if not args.dry_run:
        check_prereqs()

    for s in stages:
        run_stage(s, args)

    print("\nDone.")
    for s in stages:
        out = s[1]
        if os.path.exists(out):
            print(f"  {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")


# --------------------------------------------------------------------------
# Example stage list -- replace this with your own pipeline. Each stage is
# (name, output_path, argv, min_size_bytes[, opts]). This example is runnable
# as-is (it only uses `python -c ...` so it has no external dependencies),
# and demonstrates: a plain stage, a stage with an explicit `deps` entry for
# a hardcoded input, and a final "assemble" stage that depends on every
# earlier stage's output.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # The demo writes under the system temp directory rather than beside this
    # file: this script is meant to be copied into a project, and running the
    # example should not litter wherever the copy happens to live. The path is
    # stable across runs (not mkdtemp) so a second run demonstrates the skip.
    example_dir = os.path.join(tempfile.gettempdir(), "build_pipeline_example")
    os.makedirs(example_dir, exist_ok=True)
    scene_out = os.path.join(example_dir, "scene.txt")
    frames_out = os.path.join(example_dir, "frames.txt")
    final_out = os.path.join(example_dir, "deliverable.txt")

    EXAMPLE_STAGES = [
        # A stage with no extra dependencies beyond its own argv.
        ("scene", scene_out,
         [PY, "-c", f"open({scene_out!r}, 'w').write('scene data\\n')"],
         5),
        # A stage whose script hardcodes reading `scene_out` rather than
        # taking it as a CLI argument -- so it must be declared via `deps`,
        # or an edit to scene.txt would never mark this stage stale.
        ("frames", frames_out,
         [PY, "-c",
          f"open({frames_out!r}, 'w').write(open({scene_out!r}).read() * 3)"],
         5, {"deps": [scene_out]}),
        # The final assembly: depends on every prior stage's output, so it
        # re-runs if *anything* upstream changed, not just its own argv.
        ("assemble", final_out,
         [PY, "-c",
          f"open({final_out!r}, 'w').write(open({frames_out!r}).read())"],
         5, {"deps": [scene_out, frames_out]}),
    ]

    main(EXAMPLE_STAGES)
