# RBY1 real-hardware overlay pipeline

Overlays revamp/maze-style captions **directly on the real RBY1 hardware
footage** (two clips, `point_00`/`point_01` — two different box positions on
the same 20-grid-point pick-lift-place task) — no sim render, no ghost
overlay. This is deliberately a *different* approach from `bimanual-iiwa`
(which is a pure Drake sim render): here the numbers and the end-effector
trace come from the REAL logged joint encoder positions via Drake forward
kinematics, projected onto the real video with a camera pose fitted from the
footage itself.

Python: `/home/olorin/projects/PVAMP/rby1-constrained-planning/.venv/bin/python`
(borrowed venv, same as bimanual-iiwa — this project has none of its own).

## Directory map

```
rby1_humanoid/
  20260828_142748.mp4, 20260828_143004.mp4   the two real hardware clips (point_00, point_01)
  records/                                    execution-record pickles (real logged joint states)
  scripts/
    common.py                model loading, execution-record parsing, FK-by-joint-name helpers
    robot_camera_fit.py       ported from rby1-constrained-planning history (commit b9fcdbb,
                              since removed there) -- record -> q(t), video clock helpers,
                              look_at()/pose_from_vec()/vec_from_pose()
    fit_camera_from_log.py    the actual camera-pose optimizer (ported the same way)
    render_hw_overlay.py      the main script -- captions + EE trace on the real footage
    make_comparison_banner.py generates the side-by-side's shared banner PNG
  models/
    comparison_banner.png     generated banner (regenerate with make_comparison_banner.py)
  scratch/camera_fit/          fitted-camera JSONs + seed JSONs (see "Camera fit" below)
  out/                          hw_overlay_point_00.mp4, hw_overlay_point_01.mp4, hw_overlay_side_by_side.mp4
```

## Reproducing everything from scratch

### 1. Camera fit (only needed once per clip — `scratch/camera_fit/fit_*.json`
   are the durable output; skip this whole section if they already exist)

There is no calibrated camera for this footage anywhere upstream —
`rby1-constrained-planning/scripts/video/` has camera code, but it's for the
*simulated* Blender/VTK explainer renders (matching a synthetic camera
between two render engines), not this real phone footage, and
`hardware_framing.py` there is only a fixed 2D crop rectangle, no 3D pose.
The one real lead is `rby1-constrained-planning/notes/camera_fit_from_logs.md`,
which reports a bundled fit's **position + focal length** (not full
orientation) for exactly these two clips from earlier work:

```
f = 945.8 px
20260828_142748 (point_00): camera (2.4858, -0.0568, 1.3662) m,  dt -0.118 s
20260828_143004 (point_01): camera (2.4906, -0.0510, 1.3793) m,  dt -0.858 s
```

Bootstrap a full 6-DOF seed from that (position from the note, orientation
from a look-at toward the robot's rough chest height — phones are usually
level and pointed at the subject), then let `fit_camera_from_log.py`'s own
Nelder-Mead refine polish the orientation (skip its expensive coarse grid
search entirely by seeding it):

```bash
PY=/home/olorin/projects/PVAMP/rby1-constrained-planning/.venv/bin/python
cd /home/olorin/projects/PVAMP/revamp-video/rby1_humanoid
mkdir -p scratch/camera_fit

# per clip (point_00 shown; point_01 uses its own eye/dt from the table above
# and 20260828_143004.mp4 / point_01's record):
$PY -c "
import sys, json; sys.path.insert(0,'scripts')
import robot_camera_fit as M
import numpy as np
eye = np.array([2.4858, -0.0568, 1.3662])
X0 = M.look_at(eye, np.array([0.0, 0.0, 1.1]))
fit = dict(X_WC_vec=list(map(float, M.vec_from_pose(X0))),
          focal_px_full=945.8, dt_s=-0.118, score=0.0)
json.dump(fit, open('scratch/camera_fit/seed_142748.json','w'), indent=2)
"
$PY scripts/fit_camera_from_log.py \
  --video 20260828_142748.mp4 \
  --record records/point_00_exec_20260828_142919.pkl \
  --seed-json scratch/camera_fit/seed_142748.json \
  --out scratch/camera_fit --n-fine 12
# writes scratch/camera_fit/fit_20260828_142748.json
```

This takes a few minutes (three Nelder-Mead refine rounds at increasing
resolution, plus an edge-chamfer polish) — **run it in the background and
poll**, don't block on it synchronously.

**Accuracy, honestly**: reprojecting known gripper positions back onto real
frames and eyeballing the overlay shows it's excellent for upright poses
(premove/reach — spot-on) and good-but-imperfect during the crouched
grasp/lift pose (tens of pixels off at some specific timestamps). This is a
look-at-seeded single-clip refine, not the full two-stage
correspondence+PnP pipeline `camera_fit_from_logs.md` describes for its
sub-millimeter numbers — good enough for a demo overlay, not for metrology.

### 2. Render both hardware overlays

```bash
cd /home/olorin/projects/PVAMP/revamp-video/rby1_humanoid
$PY scripts/render_hw_overlay.py --all
# or one at a time: $PY scripts/render_hw_overlay.py point_00 [--duration 25]
```

Each output is the CROP window only (`900x980`, fitted to where these two
clips' robot motion actually is — NOT `rby1-constrained-planning`'s own
`HW_CROP`, which is tuned to a different, earlier camera setup), 30fps, with:

- A translucent caption box, top-left (`draw_boxed_lines`): bold per-run
  label ("Point 00"/"Point 01" — NOT raw box coordinates, which read as a
  parameter dump rather than "these are two different runs"), step name +
  elapsed time, planning wall-clock time (from `plans/grid_cache/point_NN.pkl`'s
  `meta["wall_s"]`), a bimanual constraint-error readout, and a CoM-to-support-
  polygon stability margin (reusing `rby1-constrained-planning/src/rby1_opt_ik.py`'s
  own `support_polygon_xyzs`/`_com_support_polygon_residuals` — copied as
  plain numpy, not imported, because that module does `from common import
  RepoDir` and this script's own `scripts/common.py` already claimed the name
  "common" in `sys.modules` by the time it's imported; inserting the other
  repo's `src/` earlier on `sys.path` does NOT fix this, since Python resolves
  a module name to whatever already got cached under it — copy the few
  needed symbols verbatim instead of fighting the cache).
  - Positioned top-left, translucent, not opaque and not bottom-anchored:
    an opaque box in either bottom corner gets covered by the robot's own
    hands/box during the crouched grasp pose (~t=21-22s in point_00), hiding
    the one moment most worth seeing.
- **Constraint error** only shown once actually grasped (`CONSTRAINED_STEPS =
  {"grasp","lift","place","release"}`) — before that the arms are still
  independently reaching and a t=0 reference would report a meaningless
  multi-metre "error" for the whole approach.
- **End-effector trace**, same color convention as bimanual-iiwa's
  `DualFollower` (there's only ever a midpoint here, no leader/follower
  split): both arms orange/resolved, midpoint blue/parameterized. Three
  things differ from bimanual-iiwa because this is real, continuous
  hardware footage rather than one short pre-planned segment:
  1. **Blue (midpoint) only during `CONSTRAINED_STEPS`** — outside that
     window the arms move independently and there's no fixed relative
     transform for a midpoint to parameterize, so those samples are set to
     `None` rather than gated in the draw call (breaks the line like a point
     behind the camera).
  2. **The translucent "plan" is bounded to the current/next step**, not the
     whole rest of the run — bimanual-iiwa's plan was naturally bounded this
     way already (one render = one segment); this hardware log strings
     several steps together (premove → reach → grasp → lift → place →
     release → retreat → home), so without an explicit bound the plan
     overlay for, say, "premove" would tangle through the entire rest of the
     run and look like clutter.
  3. **Executed-so-far fades with age** (`TRACE_FADE_TAIL_S=2.0` full
     brightness, fading out entirely by `TRACE_FADE_HIST_S=6.0` behind
     "now") instead of bimanual-iiwa's uniform solid — a real hardware step
     can run many seconds, and a same-brightness trail back to its start
     read as clutter.
  - Camera is STATIC (the phone never moved) — one fixed `X_CW` for the
    whole clip, unlike bimanual-iiwa's per-frame orbiting synthetic camera.
    Uses the fit's own `dt_s` time offset when picking which logged sample
    to project (`t_proj`), kept separate from the caption's own time basis
    (`t_log_query`, no `dt_s`) so a small ~0.1-0.8s camera-clock correction
    doesn't shift when the "not yet grasped" caption line flips.

### 3. Side-by-side + banner

```bash
$PY scripts/make_comparison_banner.py     # regenerate models/comparison_banner.png if its text changes
ffmpeg -y -ss 8 -i out/hw_overlay_point_00.mp4 -ss 8 -i out/hw_overlay_point_01.mp4 \
  -loop 1 -i models/comparison_banner.png \
  -filter_complex "\
[0:v]tpad=stop_mode=clone:stop_duration=0.2[left];\
[1:v]tpad=stop_mode=clone:stop_duration=0.4[right];\
[left][right]hstack=inputs=2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x000000[stacked];\
[stacked][2:v]overlay=0:0[out]" \
  -map "[out]" -t 43.1 -c:v libx264 -crf 20 -preset fast -pix_fmt yuv420p \
  out/hw_overlay_side_by_side.mp4
```

Notes on the magic numbers above:
- `-ss 8` on both inputs trims the first 8s (mostly the static "premove"
  standing pose) off BOTH clips uniformly before compositing.
- `-t 43.1` matches the (post-trim) shorter+padding duration; recompute if
  you change the trim or the source clips.
- The banner is overlaid **top-anchored** (`overlay=0:0`), and it lands
  clear of the per-panel caption boxes (which start at `y=20` *within* each
  panel) because `pad=...:(oh-ih)/2` centers the two 900x980 panels
  vertically inside the 1920x1080 canvas, leaving a real gap above them —
  verify this still holds if you change `CROP_H`, the pad size, or the
  banner height (`make_comparison_banner.py`'s `HEIGHT=70`).
- Per-panel titles are the bold instance label ONLY ("Point 00"/"Point 01")
  — the "RBY1 Hardware" framing lives once, in the shared banner, not
  repeated in both panels.

## Gotchas worth not repeating

- **The "common" module-name clash** (see above) will bite you again if you
  add a new script here that needs anything from
  `rby1-constrained-planning/src/`. Either copy the needed symbols verbatim
  (as `render_hw_overlay.py` does), or `importlib`-load that repo's module
  under a different name — don't just reorder `sys.path` insertions and
  assume it'll resolve differently; it won't, once "common" is cached.
- **`cv2.VideoCapture` + a `CAP_PROP_POS_MSEC` seek per frame is a trap** —
  reseeking to the nearest keyframe and redecoding forward on every frame
  turned a ~50s clip into a many-minutes render. `render_hw_overlay.py`
  instead does one `ffmpeg -ss <start> -t <dur>` seek and reads the whole
  raw stream forward in one pass (crop + fps conversion done in ffmpeg's own
  filter graph, not per-frame in Python).
- **`plans/grid_cache/point_NN.pkl`'s `meta["timings"]` is NOT what
  `wall_s` sums to** — see `rby1-constrained-planning/CLAUDE.md`'s "two
  things that repeatedly surprise people": the sweep nests multiple levels
  of fork-based parallelism, so per-stage seconds don't sum to the real
  wall-clock; `meta["wall_s"]` (used here for the "planning time" caption)
  is the one number that's actually correct.
