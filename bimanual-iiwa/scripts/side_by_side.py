"""Composite one segment's DualFollower and LeaderFollower renders side by
side into one 16:9 (1920x1080, matching revamp-video's own maze footage)
frame -- each panel scaled to half-width, full height.

    python3 side_by_side.py T->B
    python3 side_by_side.py --all

Deliberately NOT time-synced: the two methods take their own planning/
motion time and are not stretched or trimmed to match -- per Tommy, "let's
not time the two to start and finish exactly, they can take their own
time." The shorter render just holds its own last frame (tpad) until the
longer one finishes, rather than looping or cutting early.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

PANEL_W, PANEL_H = 960, 1080
OUT_W, OUT_H = 1920, 1080


def seg_name(method, segment):
    return f"{method.lower()}_{segment.replace('->', '_to_')}"


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


LEGEND_PNG = os.path.join(common.REPO, "models", "legend_banner.png")


def side_by_side(segment, out_path):
    left = os.path.join(common.OUT_DIR, f"{seg_name('LeaderFollower', segment)}.mp4")
    right = os.path.join(common.OUT_DIR, f"{seg_name('DualFollower', segment)}.mp4")
    for p in (left, right, LEGEND_PNG):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    dur_l, dur_r = probe_duration(left), probe_duration(right)
    longest = max(dur_l, dur_r)

    # tpad holds each input's last frame (clone) out to `longest` so
    # neither side is stretched/retimed to match the other -- just held,
    # same convention as render_segment.py's own HEAD/TAIL_HOLD_S.
    # Center-CROP to half-width, not scale -- the source is already
    # 1920x1080 (16:9, same as the maze footage), so scaling each panel
    # down to 960x1080 squished it horizontally (8:9). Cropping instead
    # keeps every pixel at its native aspect ratio; it just loses the
    # outer edges of the frame, which is fine since ORBIT_CENTER keeps
    # both arms near the middle of the shot anyway.
    # The legend/caption banner (make_legend_banner.py) is a single static
    # image, overlaid for the whole clip -- it never changes frame to
    # frame, so there's no reason to bake it in per-frame. Top-anchored per
    # Tommy (was bottom).
    filter_complex = (
        f"[0:v]crop={PANEL_W}:{PANEL_H}:(iw-{PANEL_W})/2:0,"
        f"tpad=stop_mode=clone:stop_duration={longest - dur_l:.3f}[left];"
        f"[1:v]crop={PANEL_W}:{PANEL_H}:(iw-{PANEL_W})/2:0,"
        f"tpad=stop_mode=clone:stop_duration={longest - dur_r:.3f}[right];"
        f"[left][right]hstack=inputs=2[stacked];"
        f"[stacked][2:v]overlay=0:0[out]"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", left, "-i", right,
         "-loop", "1", "-i", LEGEND_PNG,
         "-filter_complex", filter_complex, "-map", "[out]", "-t", f"{longest:.3f}",
         "-c:v", "libx264", "-crf", "20", "-preset", "fast",
         "-pix_fmt", "yuv420p", out_path],
        check=True)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segment", nargs="?", choices=common.SEGMENTS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for segment in common.SEGMENTS:
            name = seg_name("sidebyside", segment)
            side_by_side(segment, os.path.join(common.OUT_DIR, f"{name}.mp4"))
        return

    if not args.segment:
        sys.exit("give SEGMENT, or --all")
    name = seg_name("sidebyside", args.segment)
    side_by_side(args.segment, os.path.join(common.OUT_DIR, f"{name}.mp4"))


if __name__ == "__main__":
    main()
