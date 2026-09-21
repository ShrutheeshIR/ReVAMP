"""Generate one title-card PNG (1920x1080, black bg, centered white bold
text, word-wrapped across explicit lines) for build_final_video.py's title
slides.

    python3 make_title_card.py OUT.png "Line one" "Line two" ...
"""
from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MAX_TEXT_WIDTH = 1600


def wrap_line(draw, text, font, max_width):
    """Greedy word-wrap one logical line across as many rendered lines as
    it takes to stay under max_width."""
    words = text.split()
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def make_card(out_path, lines, font_size=64, line_gap=20):
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, font_size)

    rendered = []
    for line in lines:
        rendered.extend(wrap_line(draw, line, font, MAX_TEXT_WIDTH))

    heights = [draw.textbbox((0, 0), l, font=font)[3] for l in rendered]
    total_h = sum(heights) + line_gap * (len(rendered) - 1)
    y = (HEIGHT - total_h) / 2
    for line, h in zip(rendered, heights):
        w = draw.textlength(line, font=font)
        draw.text(((WIDTH - w) / 2, y), line, fill=(245, 245, 245), font=font)
        y += h + line_gap

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path")
    ap.add_argument("lines", nargs="+")
    ap.add_argument("--font-size", type=int, default=64)
    args = ap.parse_args()
    make_card(args.out_path, args.lines, font_size=args.font_size)


if __name__ == "__main__":
    main()
