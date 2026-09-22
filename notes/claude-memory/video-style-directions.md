---
name: video-style-directions
description: "Tommy's standing style directions for the ReVAMP demo video (callouts, camera motion, what to show)"
metadata:
  node_type: memory
  pinned: false
---

Standing directions from Tommy (2026-09-21) on how the demo video should
read. These apply to any future re-render or new clip in this effort:

1. **Callouts pause outright and point.** Don't rely on 1x slow-down
   windows (they read as choppy) and don't rely on the viewer knowing
   where in the frame to look: freeze the frame and draw a big arrow at
   the part of interest, with the caption. The `--callouts` freeze
   mechanism in `scripts/render_overlay.py` is the accepted approach;
   per-replan slow-down gating is not.
2. **Minimal synthetic camera motion.** A little orbit to show depth is
   fine; the original bimanual-iiwa 80° sweep was "too much".
3. **Show one good exemplar, not the full matrix.** For bimanual-iiwa:
   one segment (Bottom -> Top) at 1x, not all six segments sped up.
   Use the freed screen time to highlight the METHOD difference
   (DualFollower = more balanced, direct motion vs LeaderFollower).
4. **One hardware clip for RB-Y1, not two side-by-side.** Two panels
   would echo the bimanual side-by-side (which compares methods) and
   dilute focus. Tommy prefers clip 1 (point_00, 20260828_142748): it
   bends over farthest, best showing whole-body motion.
