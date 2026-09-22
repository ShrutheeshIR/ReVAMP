"""Assemble the final video: title cards (fade in/out) interleaved with the
three main clips.

    python3 build_final_video.py

Normalizes every input to 1920x1080/30fps/yuv420p before concatenation
(the maze clip is 4K/~60fps; the title cards and the other two clips are
already 1080p/30fps) via per-input scale+fps+setsar, then a filter_complex
concat -- not the concat demuxer, since inputs don't already match.
"""
from __future__ import annotations

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLE_DIR = os.path.join(REPO, "scratch", "title_clips")
OUT_PATH = os.path.join(REPO, "out", "revamp_final.mp4")

SEQUENCE = [
    os.path.join(TITLE_DIR, "title1.mp4"),
    os.path.join(TITLE_DIR, "title2.mp4"),
    os.path.join(REPO, "out", "highlight_plan_tree_4x_4k.mp4"),
    os.path.join(TITLE_DIR, "title3.mp4"),
    # ONE segment (B->T) at 1x, not all six at 2x -- per Tommy: a single
    # good exemplar with the method difference readable beats the full
    # sped-up matrix.
    os.path.join(REPO, "bimanual-iiwa", "out", "sidebyside_B_to_T.mp4"),
    os.path.join(TITLE_DIR, "title4.mp4"),
    os.path.join(REPO, "rby1_humanoid", "out", "hw_overlay_point_00.mp4"),
]


def main():
    for p in SEQUENCE:
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    inputs = []
    norm_chain = []
    for i, p in enumerate(SEQUENCE):
        inputs += ["-i", p]
        norm_chain.append(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]")
    concat_inputs = "".join(f"[v{i}]" for i in range(len(SEQUENCE)))
    filter_complex = ";".join(norm_chain) + f";{concat_inputs}concat=n={len(SEQUENCE)}:v=1:a=0[out]"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", *inputs,
        "-filter_complex", filter_complex, "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p", OUT_PATH,
    ]
    subprocess.run(cmd, check=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
