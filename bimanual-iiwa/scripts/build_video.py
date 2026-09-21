"""One-command, resumable build of all 12 bimanual-IIWA segment renders +
one concatenated video per method.

    venv/bin/python scripts/build_video.py            # everything
    venv/bin/python scripts/build_video.py --list     # show stages
    venv/bin/python scripts/build_video.py --force-from dualfollower_T_to_B

Stage mechanics (staleness, prereq checks) live in build_pipeline.py, a
copy of revamp-video's own driver -- this file only declares the pipeline,
matching that project's convention.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pipeline
import common
from build_pipeline import main, check_prereqs  # noqa: F401

build_pipeline.REQUIRED_BINARIES[:] = ["ffmpeg", "ffprobe"]
build_pipeline.REQUIRED_MODULES[:] = ["numpy", "pydrake", "PIL"]

PY = sys.executable
S = os.path.dirname(os.path.abspath(__file__))
OUT = common.OUT_DIR


def sp(name):
    return os.path.join(S, name)


def seg_name(method, segment):
    return f"{method.lower()}_{segment.replace('->', '_to_')}"


STAGES = []

for method in common.METHODS:
    for segment in common.SEGMENTS:
        name = seg_name(method, segment)
        STAGES.append((
            name, os.path.join(OUT, f"{name}.mp4"),
            [PY, sp("render_segment.py"), method, segment], 20_000,
            {"deps": [common.traj_path(method, segment)]},
        ))


def concat_stage(method):
    name = f"{method.lower()}_all"
    out = os.path.join(OUT, f"{name}.mp4")
    seg_outs = [os.path.join(OUT, f"{seg_name(method, s)}.mp4")
               for s in common.SEGMENTS]
    return (name, out, [PY, sp("concat_method.py"), method], 20_000,
           {"deps": seg_outs})


for method in common.METHODS:
    STAGES.append(concat_stage(method))


if __name__ == "__main__":
    main(STAGES)
