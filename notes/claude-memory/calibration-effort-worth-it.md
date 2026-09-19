---
name: calibration-effort-worth-it
description: Spend generous compute/effort refining camera calibration; use the whole video
metadata: 
  node_type: memory
  pinned: false
  originSessionId: 78709ea8-7edc-470d-a064-b941ac51821a
  modified: 2026-09-18T02:37:31.436Z
---

Tommy explicitly said that iterating on the pose estimate / camera calibration is time, tokens, and compute well spent — keep refining it (even overnight-length runs) rather than settling for "close enough," and use ALL segments of the video for the estimate, not just the highlight window. When alignment artifacts appear in overlays (ghost lag during fast motion, tool offsets), the right response is another refinement round (finer Δt, full intrinsics, true mesh silhouettes, more frames), not acceptance.
