"""Bottom banner for hw_overlay_side_by_side.mp4 -- a static PNG, overlaid
for the whole clip (see the ffmpeg hstack+overlay command), signalling
"these are two different runs being compared" before the viewer even reads
either panel's own caption. Each panel's own caption already carries a bold
per-run label ("Point 00" / "Point 01", not raw box coordinates -- see
render_hw_overlay.py's INSTANCE_LABEL) for the finer-grained identification;
this banner is the coarse, at-a-glance one.
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

WIDTH, HEIGHT = 1920, 70
BG = (0, 0, 0, 190)
WHITE = (245, 245, 245, 255)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def main(out_path=None):
    out_path = out_path or os.path.join(common.REPO, "models", "comparison_banner.png")
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=BG)

    font = ImageFont.truetype(FONT_PATH, 28)
    text = "RBY1 Hardware — Same Pick-Lift-Place Plan, Two Grid Points"
    tw = draw.textlength(text, font=font)
    draw.text(((WIDTH - tw) / 2, 20), text, fill=WHITE, font=font)

    img.save(out_path)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
