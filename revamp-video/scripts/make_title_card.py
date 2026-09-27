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
LOGO_SIZE = 100
LOGO_MARGIN = 150
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


def load_logo(logo_path, negate, height):
    """negate=True inverts RGB (keeping alpha) -- for a black-on-transparent
    logo against this card's black background, so it reads as white instead
    of vanishing. A full-color logo (e.g. CSAIL's) should NOT be negated.

    Crops to the alpha channel's bounding box first: logo source files carry
    very different amounts of internal transparent padding (CSAIL's mark is
    ~514x300 inside a 716x500 canvas; commie's is ~472x654 inside 654x654),
    so a plain thumbnail() to the same box makes them look mismatched in
    size even though neither logo actually asked for that. Then scaled so
    every logo shares the same rendered HEIGHT, which is what reads as
    "same size" when they sit side by side at different aspect ratios."""
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    scale = height / logo.height
    logo = logo.resize((max(1, round(logo.width * scale)), height),
                       Image.LANCZOS)
    if negate:
        r, g, b, a = logo.split()
        rgb = ImageChops.invert(Image.merge("RGB", (r, g, b)))
        logo = Image.merge("RGBA", (*rgb.split(), a))
    return logo


def render_logo_row(logos, height=LOGO_SIZE, gap=44):
    """logos: list of (path, negate, scale) -> a bottom-aligned RGBA strip,
    same layout math as paste_logos but returned as its own image instead
    of pasted onto the card directly, so it can be nested inside a column
    (institute name on top, its logos underneath)."""
    if not isinstance(gap, (list, tuple)):
        gap = [gap] * (len(logos) - 1)
    rendered = [load_logo(p, negate, round(height * scale))
               for p, negate, scale in logos]
    total_w = sum(l.width for l in rendered) + sum(gap)
    strip = Image.new("RGBA", (max(1, round(total_w)), height), (0, 0, 0, 0))
    x = 0.0
    for i, logo in enumerate(rendered):
        strip.paste(logo, (round(x), height - logo.height), logo)
        x += logo.width + (gap[i] if i < len(gap) else 0)
    return strip


def render_affil_column(text, font_path, font_size, text_color, logos,
                        logo_height, text_logo_gap=24, logo_gap=26):
    """One 'institute name over its logos' column: text centered on top,
    its logo row centered underneath. Returns an RGBA image sized to its
    own content (both text and logos independently centered on the
    column's own width, which is whichever of the two is wider)."""
    tmp = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(tmp)
    font = ImageFont.truetype(font_path, font_size)
    tw = draw.textlength(text, font=font)
    th = draw.textbbox((0, 0), text, font=font)[3]

    logos_img = render_logo_row(logos, height=logo_height, gap=logo_gap)

    col_w = max(tw, logos_img.width)
    col_h = th + text_logo_gap + logos_img.height
    col = Image.new("RGBA", (round(col_w), round(col_h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(col)
    d.text(((col_w - tw) / 2, 0), text, fill=text_color, font=font)
    col.paste(logos_img, (round((col_w - logos_img.width) / 2), th + text_logo_gap),
              logos_img)
    return col


def render_affil_columns(columns, col_gap=110):
    """columns: list of render_affil_column(...) images -> one combined
    row image, top-aligned (all columns share the same text font/size, so
    their text tops already line up; logos rows may differ slightly in
    height across columns, a minor cosmetic difference)."""
    total_w = sum(c.width for c in columns) + col_gap * (len(columns) - 1)
    total_h = max(c.height for c in columns)
    row = Image.new("RGBA", (round(total_w), total_h), (0, 0, 0, 0))
    x = 0
    for c in columns:
        row.paste(c, (round(x), 0), c)
        x += c.width + col_gap
    return row


def paste_logos(img, logos, height=LOGO_SIZE, margin=LOGO_MARGIN, gap=44):
    """logos: list of (path, negate, scale) pasted in a row bottom-center,
    left to right in list order. `scale` adjusts an individual logo's
    height relative to the shared baseline (1.0 = full `height`), for
    logos whose mark reads visually heavier/lighter than others at the
    same nominal height. All logos still bottom-align.

    gap: either one number (same gap everywhere) or a list of
    len(logos)-1 numbers, one per boundary -- e.g. a tight gap within an
    institute/lab pair and a wider gap between pairs."""
    if not isinstance(gap, (list, tuple)):
        gap = [gap] * (len(logos) - 1)
    rendered = [load_logo(p, negate, round(height * scale))
               for p, negate, scale in logos]
    total_w = sum(l.width for l in rendered) + sum(gap)
    x = (WIDTH - total_w) / 2
    y_base = HEIGHT - margin - height
    for i, logo in enumerate(rendered):
        y = y_base + (height - logo.height)
        img.paste(logo, (round(x), y), logo)
        x += logo.width + (gap[i] if i < len(gap) else 0)


def make_card(out_path, blocks, line_gap=20, block_gaps=70, logos=None,
              logo_gaps=None):
    """blocks: list of entries, each either
      - ("TEXT", logical_lines, font_path, font_size, color): wrapped/
        rendered at its own font, or
      - ("IMAGE", rgba_image): pasted as-is, horizontally centered
        (e.g. an affiliation-name-over-its-logos column row).
    All stacked top to bottom and centered as one unit vertically in the
    frame. block_gaps: either a single number (same gap between every
    pair of blocks) or a list of len(blocks)-1 numbers, one per boundary.

    `logos`/`logo_gaps`: a separate, independent bottom-center logo row
    (see paste_logos) -- use this OR an "IMAGE" block's own logos, not
    both, for the same set of logos."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if not isinstance(block_gaps, (list, tuple)):
        block_gaps = [block_gaps] * (len(blocks) - 1)

    block_renders = []
    for entry in blocks:
        if entry[0] == "IMAGE":
            _, image = entry
            block_renders.append(("IMAGE", image, image.height))
        else:
            _, logical_lines, font_path, font_size, color = entry
            font = ImageFont.truetype(font_path, font_size)
            rendered = []
            for line in logical_lines:
                rendered.extend(wrap_line(draw, line, font, MAX_TEXT_WIDTH))
            heights = [draw.textbbox((0, 0), l, font=font)[3] for l in rendered]
            h = sum(heights) + line_gap * (len(rendered) - 1)
            block_renders.append(("TEXT", (rendered, font, color, heights), h))

    total_h = sum(h for _, _, h in block_renders) + sum(block_gaps)

    y = (HEIGHT - total_h) / 2
    for bi, (kind, payload, h) in enumerate(block_renders):
        if kind == "IMAGE":
            img.paste(payload, (round((WIDTH - payload.width) / 2), round(y)),
                      payload)
            y += h
        else:
            rendered, font, color, heights = payload
            for line, lh in zip(rendered, heights):
                w = draw.textlength(line, font=font)
                draw.text(((WIDTH - w) / 2, y), line, fill=color, font=font)
                y += lh + line_gap
            y -= line_gap
        if bi < len(block_renders) - 1:
            y += block_gaps[bi]

    if logos:
        paste_logos(img, logos, gap=logo_gaps if logo_gaps is not None else 44)

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
    ap.add_argument("--logo-spec", action="append", default=[],
                    help="one bottom-center logo (flat row, no per-"
                         "affiliation grouping), in left-to-right order: "
                         "'path[,negate][,SCALE]'. 'negate' inverts a "
                         "black-on-transparent mark to white; a bare "
                         "number sets its height scale (default 1.0). "
                         "Repeatable. Mutually exclusive with --affil-col.")
    ap.add_argument("--logo-gaps", type=float, nargs="*", default=None,
                    help="gaps (px) between consecutive --logo-spec "
                         "entries, len(logos)-1 numbers; omit for a "
                         "uniform default gap")
    ap.add_argument("--affil-col", action="append", default=[],
                    help="one 'institute name over its own logos' column, "
                         "in left-to-right order: 'TEXT|logo-spec1;"
                         "logo-spec2;...' (logo-spec as in --logo-spec). "
                         "Repeatable -- replaces --affil-line + "
                         "--logo-spec with grouped columns instead of a "
                         "single affiliation line and a separate flat "
                         "logo row.")
    ap.add_argument("--affil-col-gap", type=float, default=110,
                    help="gap (px) between --affil-col columns")
    ap.add_argument("--affil-col-logo-gap", type=float, default=26,
                    help="gap (px) between logos within one --affil-col")
    ap.add_argument("--affil-col-logo-height", type=int, default=LOGO_SIZE,
                    help="baseline logo height (px) within --affil-col columns")
    ap.add_argument("--affil-col-text-logo-gap", type=int, default=24,
                    help="gap (px) between a column's text and its logos")
    args = ap.parse_args()

    def parse_logo_spec(spec):
        parts = spec.split(",")
        path, negate, scale = parts[0], False, 1.0
        for p in parts[1:]:
            if p == "negate":
                negate = True
            else:
                scale = float(p)
        return (path, negate, scale)

    blocks = [("TEXT", args.lines, FONT_WEIGHTS[args.weight], args.font_size,
              (245, 245, 245))]
    gaps = []
    if args.author_line:
        blocks.append(("TEXT", [args.author_line], FONT_PATH_REGULAR,
                       args.author_font_size, (245, 245, 245)))
        gaps.append(args.title_author_gap)

    if args.affil_col:
        columns = []
        for col in args.affil_col:
            text, _, logo_specs = col.partition("|")
            logos = [parse_logo_spec(s) for s in logo_specs.split(";") if s]
            columns.append(render_affil_column(
                text, FONT_PATH_LIGHT, args.affil_font_size, (190, 190, 190),
                logos, args.affil_col_logo_height,
                text_logo_gap=args.affil_col_text_logo_gap,
                logo_gap=args.affil_col_logo_gap))
        row = render_affil_columns(columns, col_gap=args.affil_col_gap)
        blocks.append(("IMAGE", row))
        gaps.append(args.author_affil_gap)
        logos_flat, logo_gaps = [], None
    else:
        if args.affil_line:
            blocks.append(("TEXT", [args.affil_line], FONT_PATH_LIGHT,
                           args.affil_font_size, (190, 190, 190)))
            gaps.append(args.author_affil_gap)
        logos_flat = [parse_logo_spec(s) for s in args.logo_spec]
        logo_gaps = args.logo_gaps

    make_card(args.out_path, blocks, block_gaps=gaps or 70, logos=logos_flat,
              logo_gaps=logo_gaps)


if __name__ == "__main__":
    main()
