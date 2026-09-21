"""Generate the bottom caption/legend banner overlaid on the side-by-side
comparison video (see side_by_side.py) -- a static PNG, ffmpeg-overlaid for
the whole clip rather than redrawn per frame, since it never changes.

Left: what the constraint IS ("Bimanual Constraint: Fixed relative
transform between two arms"). Right: what the two trace COLORS mean -- this
is about which arm is directly planned vs. derived, not about how much of
the trace has executed (that's opacity, drawn per-frame in render_segment.py,
and is a separate axis from this legend):

  LeaderFollower: the LEFT arm's 7 joints are the parameterized/planning
    variable (per vamp/scripts/bimanual_iiwa_leader_follower_shelf.py:
    "the left/leader arm's 7 joint angles"); the right/follower arm is
    resolved from it via the fixed relative transform.
  DualFollower: the shared MIDPOINT pose is the parameterized/planning
    variable (per bimanual_iiwa_task_space_example.py's compute_mid_pose());
    BOTH arms are resolved from it via a fixed offset each way.

So blue (render_segment.py's HIGHLIGHT, used for "left"/"mid" in
TRACE_SPEC) is always the parameterized quantity, and orange (WARN_ORANGE,
"right", and both arms under DualFollower) is always resolved from it --
matching TRACE_SPEC there exactly, this legend just labels what the colors
already mean rather than changing them.
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

WIDTH = 1920
HEIGHT = 100
BG = (0, 0, 0, 190)
WHITE = (245, 245, 245, 255)
GRAY = (190, 190, 190, 255)
# render_segment.py's HIGHLIGHT/WARN_ORANGE exactly -- the legend colors
# have to match TRACE_SPEC there, not stand in for it with a placeholder.
PARAM_COLOR = (65, 105, 225)
RESOLVED_COLOR = (209, 143, 0)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main(out_path=None):
    out_path = out_path or os.path.join(common.REPO, "models", "legend_banner.png")
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=BG)

    title_font = ImageFont.truetype(FONT_PATH, 32)
    label_font = ImageFont.truetype(FONT_PATH_REG, 24)

    draw.text((30, 20), "Bimanual Constraint: Fixed relative transform "
                        "between two arms", fill=WHITE, font=title_font)

    # Legend, right-aligned: which ARM COLOR is directly planned
    # (parameterized) vs. derived (resolved) -- see module docstring. Not
    # about opacity/how-much-executed, which is a separate, per-frame axis.
    entries = [("resolved (follower arm(s))", RESOLVED_COLOR),
              ("parameterized (leader / midpoint)", PARAM_COLOR)]
    x = WIDTH - 30
    for label, color in reversed(entries):
        tw = draw.textlength(label, font=label_font)
        x -= tw
        draw.text((x, HEIGHT / 2 + 2), label, fill=GRAY, font=label_font)
        x -= 10
        swatch_w = 46
        x -= swatch_w
        draw.line([(x, HEIGHT / 2 + 14), (x + swatch_w, HEIGHT / 2 + 14)],
                 fill=(*color, 255), width=8)
        x -= 28

    img.save(out_path)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
