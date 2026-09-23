"""Generate one title-card PNG (1920x1080, black bg, centered white bold
text, word-wrapped across explicit lines) for build_final_video.py's title
slides.

    python3 make_title_card.py OUT.png "Line one" "Line two" ...
    python3 make_title_card.py OUT.png "Title line" --font-size 56 \\
        --author-line "A. Author, B. Author" \\
        --affil-line "1Purdue University   2MIT"
"""
from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont, ImageChops

WIDTH, HEIGHT = 1920, 1080
LOGO_SIZE = 140
LOGO_MARGIN = 60
FONT_DIR = "/home/olorin/.fonts"
FONT_WEIGHTS = {
    "bold": f"{FONT_DIR}/SourceSans3-Bold.ttf",
    "semibold": f"{FONT_DIR}/SourceSans3-SemiBold.ttf",
}
FONT_PATH_REGULAR = f"{FONT_DIR}/SourceSans3-Regular.ttf"
FONT_PATH_LIGHT = f"{FONT_DIR}/SourceSans3-Light.ttf"
MAX_TEXT_WIDTH = 1750


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


def paste_logo(img, logo_path, negate=True, size=LOGO_SIZE, margin=LOGO_MARGIN):
    """Top-right corner logo. negate=True inverts RGB (keeping alpha) --
    for a black-on-transparent logo against this card's black background,
    so it reads as white instead of vanishing."""
    logo = Image.open(logo_path).convert("RGBA")
    logo.thumbnail((size, size), Image.LANCZOS)
    if negate:
        r, g, b, a = logo.split()
        rgb = ImageChops.invert(Image.merge("RGB", (r, g, b)))
        logo = Image.merge("RGBA", (*rgb.split(), a))
    x = WIDTH - margin - logo.width
    y = margin
    img.paste(logo, (x, y), logo)


def make_card(out_path, blocks, line_gap=20, block_gaps=70, logo=None,
              logo_negate=True):
    """blocks: list of (logical_lines, font_path, font_size, color) groups,
    each wrapped/rendered at its own font, stacked top to bottom, all
    centered as one unit vertically in the frame. block_gaps: either a
    single number (same gap between every pair of blocks) or a list of
    len(blocks)-1 numbers, one per boundary, for independent spacing."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if not isinstance(block_gaps, (list, tuple)):
        block_gaps = [block_gaps] * (len(blocks) - 1)

    block_renders = []  # list of (rendered_lines, font, color, heights)
    for logical_lines, font_path, font_size, color in blocks:
        font = ImageFont.truetype(font_path, font_size)
        rendered = []
        for line in logical_lines:
            rendered.extend(wrap_line(draw, line, font, MAX_TEXT_WIDTH))
        heights = [draw.textbbox((0, 0), l, font=font)[3] for l in rendered]
        block_renders.append((rendered, font, color, heights))

    total_h = 0
    for rendered, font, color, heights in block_renders:
        total_h += sum(heights) + line_gap * (len(rendered) - 1)
    total_h += sum(block_gaps)

    y = (HEIGHT - total_h) / 2
    for bi, (rendered, font, color, heights) in enumerate(block_renders):
        for line, h in zip(rendered, heights):
            w = draw.textlength(line, font=font)
            draw.text(((WIDTH - w) / 2, y), line, fill=color, font=font)
            y += h + line_gap
        if bi < len(block_renders) - 1:
            y += block_gaps[bi] - line_gap

    if logo:
        paste_logo(img, logo, negate=logo_negate)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path")
    ap.add_argument("lines", nargs="+")
    ap.add_argument("--font-size", type=int, default=64)
    ap.add_argument("--weight", choices=list(FONT_WEIGHTS), default="semibold",
                    help="title font weight (author/affiliation lines are "
                         "always Regular/Light)")
    ap.add_argument("--author-line", help="second block: author names")
    ap.add_argument("--author-font-size", type=int, default=36)
    ap.add_argument("--affil-line", help="third block: affiliations")
    ap.add_argument("--affil-font-size", type=int, default=28)
    ap.add_argument("--title-author-gap", type=int, default=90,
                    help="gap between the title block and the author line")
    ap.add_argument("--author-affil-gap", type=int, default=40,
                    help="gap between the author line and the affiliations")
    ap.add_argument("--logo", help="path to a top-right corner logo")
    ap.add_argument("--logo-no-negate", action="store_true",
                    help="paste the logo as-is instead of inverting RGB")
    args = ap.parse_args()

    blocks = [(args.lines, FONT_WEIGHTS[args.weight], args.font_size,
              (245, 245, 245))]
    gaps = []
    if args.author_line:
        blocks.append(([args.author_line], FONT_PATH_REGULAR,
                       args.author_font_size, (245, 245, 245)))
        gaps.append(args.title_author_gap)
    if args.affil_line:
        blocks.append(([args.affil_line], FONT_PATH_LIGHT,
                       args.affil_font_size, (190, 190, 190)))
        gaps.append(args.author_affil_gap)
    make_card(args.out_path, blocks, block_gaps=gaps or 70, logo=args.logo,
              logo_negate=not args.logo_no_negate)


if __name__ == "__main__":
    main()
