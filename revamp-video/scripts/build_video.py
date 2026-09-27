"""One-command, resumable build of the annotated revamp hardware video.

    venv/bin/python scripts/build_video.py            # build everything
    venv/bin/python scripts/build_video.py --list     # show stages
    venv/bin/python scripts/build_video.py --force-from calibrate

Stage mechanics (staleness from the import graph, size floors, prereq
checks) live in build_pipeline.py; this file only declares the pipeline.
The 2.3 GB source video is NOT in the repo — every user provides
shru_revamp_vid_hanlanphone-001.MOV at the repo root themselves.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pipeline
import common
from build_pipeline import main, check_prereqs  # noqa: F401

build_pipeline.REQUIRED_BINARIES[:] = ["ffmpeg", "ffprobe"]
build_pipeline.REQUIRED_MODULES[:] = [
    "numpy", "scipy", "cv2", "pydrake", "pupil_apriltags", "matplotlib"]

PY = sys.executable
S = os.path.join(common.REPO, "scripts")
C = common.CALIB_DIR


def sp(name):
    return os.path.join(S, name)


def cj(name):
    return os.path.join(C, name)


STAGES = [
    ("tool_frame", cj("tool_offset.json"),
     [PY, sp("identify_tool_frame.py")], 200,
     {"deps": [common.JOINT_CSV]}),
    ("sync", cj("sync.json"),
     [PY, sp("sync.py")], 200,
     {"deps": [common.JOINT_CSV, common.QUERIES_JSONL]}),
    ("motion_delta", cj("motion_delta.json"),
     [PY, sp("motion_profile.py")], 100,
     {"deps": [cj("sync.json")]}),
    ("tag_init", cj("tag_init.json"),
     [PY, sp("tag_init.py")], 300, {}),
    ("silhouette", cj("camera_sil.json"),
     [PY, sp("silhouette_calib.py")], 300,
     {"deps": [cj("sync.json"), cj("motion_delta.json"),
               cj("tag_init.json")]}),
    ("gl_meshes",
     common.URDF_GL_PATH,
     [PY, sp("make_gl_meshes.py")], 1000, {}),
    ("deep_calib", cj("camera_deep.json"),
     [PY, sp("deep_calib.py")], 300,
     {"deps": [cj("camera_sil.json"), common.URDF_GL_PATH]}),
    # NOTE: the NCC-track NLLS (track_eef.py + calibrate.py) is deliberately
    # NOT a stage: the near-planar track collapses into the focal-distance
    # degeneracy; calibrate.py guards against promoting such a solve.
    ("montage", os.path.join(common.SCRATCH, "verify_montage.jpg"),
     [PY, sp("render_overlay.py"), "--montage"], 100_000,
     {"deps": [cj("camera_deep.json")]}),
    ("highlight", os.path.join(common.REPO, "out", "highlight_4k.mp4"),
     [PY, sp("render_overlay.py"), "--segment", "227.7", "276.15"], 5_000_000,
     {"deps": [cj("camera_deep.json"), common.QUERIES_JSONL]}),
    ("highlight_ghost",
     os.path.join(common.REPO, "out", "highlight_ghost_4k.mp4"),
     [PY, sp("render_overlay.py"), "--segment", "227.7", "276.15", "--ghost"],
     5_000_000,
     {"deps": [cj("camera_deep.json"), common.QUERIES_JSONL,
               common.URDF_GL_PATH]}),
    ("highlight_plan",
     os.path.join(common.REPO, "out", "highlight_plan_4k.mp4"),
     [PY, sp("render_overlay.py"), "--segment", "227.7", "276.15", "--plan"],
     5_000_000,
     {"deps": [cj("camera_deep.json"), common.QUERIES_JSONL,
               common.TRAJ_DIR]}),
]

if __name__ == "__main__":
    if not os.path.isfile(common.VIDEO):
        sys.exit(f"missing source video: {common.VIDEO}\n"
                 "It is not in the repo (2.3 GB); obtain it and place it "
                 "at the repo root.")
    main(STAGES)
