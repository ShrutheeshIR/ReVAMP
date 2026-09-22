# bimanual-iiwa sim video pipeline

Renders the two constrained bimanual-planning methods (`DualFollower`,
`LeaderFollower`) as a Drake VTK sim, over 6 pre-planned shelf-to-table
trajectory segments each, with an end-effector trace overlay, then composites
the two methods side by side into one video with a shared title/legend
banner.

This is a **sibling of `rby1-constrained-planning`** (the actual planner/
model repo) — it only reads trajectory JSONs and model files from there, and
never modifies it. Everything textured/colored here is a **local copy**, not
an edit of the shared repo, because Drake's SDFormat parser and its runtime
material overrides don't do what you'd expect for that repo's own shelf/table
assets (see "Textures" below — read this before touching material colors).

Python: `/home/olorin/projects/PVAMP/rby1-constrained-planning/.venv/bin/python`
(this project has no venv of its own; it borrows the sibling repo's, which has
Drake + PIL + numpy).

## Directory map

```
bimanual-iiwa/
  scripts/
    common.py              shared scene loading, TOPPRA retiming, FK helpers
    render_segment.py       renders one (method, segment) -> out/<method>_<segment>.mp4
    side_by_side.py          composites LeaderFollower|DualFollower -> out/sidebyside_<segment>.mp4
    make_textured_box.py      generates a UV-mapped box .obj+.mtl for a given photo texture
    make_legend_banner.py     generates the static title/legend banner PNG
  models/
    scene.dmd.yaml            LOCAL directives file (table+shelf point at local textured models)
    table_metal/              brushed-metal table (table.sdf, table_top.obj/.mtl, metal.png)
    shelf_wood/                plywood shelf (shelves.sdf, shelf_1/shelf_2/shelf_back .obj/.mtl, plywood.png)
    legend_banner.png          generated banner (regenerate with make_legend_banner.py)
  trajectories/                *_<segment>.json waypoint files (method, segment, planning_time_s, configs)
  out/                          all rendered/composited mp4s land here
```

## Reproducing everything from scratch

### 0. Textures (only needed if `models/table_metal/` or `models/shelf_wood/`
   are missing their `.obj`/`.mtl` files — the source photos are already
   committed)

```bash
PY=/home/olorin/projects/PVAMP/rby1-constrained-planning/.venv/bin/python
cd /home/olorin/projects/PVAMP/revamp-video/bimanual-iiwa

$PY scripts/make_textured_box.py models/shelf_wood shelf_1 0.4 1.0 0.014 models/shelf_wood/plywood.png --tile-m 0.3
$PY scripts/make_textured_box.py models/shelf_wood shelf_2 0.4 1.0 0.014 models/shelf_wood/plywood.png --tile-m 0.3
$PY scripts/make_textured_box.py models/shelf_wood shelf_back 0.03 1.0 0.9 models/shelf_wood/plywood.png --tile-m 0.3
$PY scripts/make_textured_box.py models/table_metal table_top 2.2 1.6 0.04 models/table_metal/metal.png --tile-m 0.5
```

**Why textured meshes exist at all instead of just using
`rby1-constrained-planning`'s own `old_shelves.sdf`/`old_table/table_wide.sdf`:**
Drake's SDFormat parser has **no support for a texture map on a primitive
`<box>` geometry** — a `<material><pbr>...</pbr></material>` block on a box is
parsed and then silently dropped ("Ignoring unsupported SDFormat element in
material: pbr"). And the shared repo's real table mesh (`old_table/table_wide.obj`)
has a baked AO-atlas texture whose UVs don't tile a plain photo cleanly, **and**
that mesh's own `map_Kd` texture ignores any runtime `PerceptionProperties`
diffuse override (confirmed with an isolated black-diffuse render test — the
pixel came out bright regardless). So both the shelf and table are small,
purpose-built, UV-mapped box meshes (`make_textured_box.py`) with a real photo
in a fresh local `.mtl`, tiled at a real-world scale (`--tile-m`) rather than
stretched edge-to-edge.

If you write your own box mesh by hand instead of using the generator: **face
winding matters**. An earlier version of `make_textured_box.py` used
hand-picked corner-index orderings and got 3 of 6 faces backwards (confirmed
by `CalcSpatialInertia()` reporting a *negative* mesh volume when the mesh was
loaded standalone, and by those faces rendering solid black in Drake/VTK —
which does not shade the far side of a backward-facing triangle). The current
generator builds each face from its own outward normal `N` plus tangents
`U, V` with `U × V = N` exactly, which is winding-correct by construction —
don't go back to picking corner indices by hand.

### 1. Render all 12 segments (both methods × 6 segments)

```bash
cd /home/olorin/projects/PVAMP/revamp-video/bimanual-iiwa
$PY scripts/render_segment.py --all
# or one at a time, e.g.:
$PY scripts/render_segment.py DualFollower "T->B"      # quote the arrow, or the shell eats it as a redirect
```

Each render is 1920x1080/30fps, true-black background, an orbiting camera
(`ORBIT_CENTER=[0.5,0.4,0.45]`, `ORBIT_RADIUS=2.4`, `ORBIT_ELEV_DEG=22`,
azimuth sweeping `-180°→-220°` (was `-160°→-240°`; halved per Tommy —
"a bit is okay to show depth", 80° was too much), a mirror pair around the scene's own -180°
symmetry axis — routed through -180 rather than 0 because azimuth near
-10°/+20° swings the shelf's solid back panel between camera and arms), with:

- A caption panel (title = method name, then planning time and a single
  "constraint error" mm readout, drift of the left-right gripper transform
  from one fixed global reference — `REFERENCE_METHOD/REFERENCE_SEGMENT =
  ("DualFollower", "T->B")`'s first waypoint, the SAME reference regardless of
  which of the 12 renders is running, so a real drift at a segment handoff
  shows up as error instead of each segment re-zeroing against its own start).
  Anchored at `TEXT_X = (1920-960)//2 + 30` so it survives the side-by-side's
  center-crop, and `TEXT_Y_OFFSET = 90` so it clears the top banner.
- An end-effector trace, precomputed once per segment (the whole path is
  known ahead of time — this is a pre-planned trajectory, not an online
  replan) and projected every frame with that frame's own camera. **Color
  encodes which quantity is directly planned (parameterized) vs. derived
  (resolved) from it, NOT which arm bluntly, and NOT executed-vs-remaining**:
  - `LeaderFollower`: left arm (blue/`HIGHLIGHT`) is the parameterized leader
    (its 7 joints are the actual planning variable); right arm (orange/
    `WARN_ORANGE`) is resolved from it via the fixed relative transform.
  - `DualFollower`: the shared midpoint (blue) is parameterized; BOTH arms
    (orange) are resolved from it via a fixed offset each way.
  - Executed-so-far is solid/opaque; the rest of the (already fully known)
    segment is the SAME width but translucent (`TRACE_PLAN_ALPHA=90/255`),
    not a dash and not a hollow outline — both were tried and rejected (a
    hollow tube's inner "hole" got crushed to invisible by h264 chroma
    subsampling until the tube was widened to ~16px; a taper read fine but
    was superseded by "light and transparent, same size as the trajectory").
    Drawn onto its own RGBA overlay and alpha-composited in, since plain
    `ImageDraw` on an RGB image has no translucency.

### 2. Side-by-side composite per segment + banner

```bash
$PY scripts/make_legend_banner.py     # regenerate models/legend_banner.png if you change its text/colors
$PY scripts/side_by_side.py --all
# or one at a time: $PY scripts/side_by_side.py "T->B"
```

Center-crops each 1920x1080 render to 960 wide (not a `scale` — scaling
squished 16:9 into 8:9) and `hstack`s LeaderFollower (left) | DualFollower
(right). The two methods are **deliberately not time-synced** — each plays at
its own planning/motion duration; the shorter one holds its last frame
(`tpad`) until the longer one finishes, rather than being stretched or cut.
The banner (`models/legend_banner.png`, made by `make_legend_banner.py`) is
overlaid top-anchored (`overlay=0:0`) for the whole clip, not baked in
per-frame, since it's static: title "Bimanual Constraint: Fixed relative
transform between two arms" plus a color legend for parameterized (blue) vs.
resolved (orange) — this has to be regenerated (and `render_segment.py`
re-run, since the caption's `TEXT_Y_OFFSET` assumes a banner height) if you
move the banner or change its height.

### 3. Concatenate all 6 + speed up 2x

```bash
cd out
cat > concat_list.txt << 'EOF'
file 'sidebyside_T_to_B.mp4'
file 'sidebyside_B_to_M.mp4'
file 'sidebyside_M_to_B.mp4'
file 'sidebyside_M_to_T.mp4'
file 'sidebyside_T_to_M.mp4'
file 'sidebyside_B_to_T.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat_list.txt -c copy sidebyside_all.mp4
ffmpeg -y -i sidebyside_all.mp4 -filter:v "setpts=0.5*PTS" -r 30 \
  -c:v libx264 -crf 20 -preset fast -pix_fmt yuv420p sidebyside_all_2x.mp4
```

`sidebyside_all_2x.mp4` is the full six-segment tour. The top-level
`revamp-video/scripts/build_final_video.py` assembly instead uses
**`sidebyside_B_to_T.mp4` — one segment, 1x** (per Tommy: a single good
exemplar with the method difference readable beats the sped-up matrix;
the per-panel caption's METHOD_TRAIT line spells out the
LeaderFollower/DualFollower difference).

## Gotchas or previously-wrong-decisions worth not repeating

- **`common.py`'s "common" module name clashes with `rby1-constrained-planning`'s
  own `src/common.py`.** Not currently an issue in this project (nothing here
  imports `rby1_opt_ik` or similar), but if you ever add code that does:
  Python caches modules by name, so whichever `common.py` gets imported first
  in a process wins for the whole process, regardless of `sys.path` order
  changes made afterward. See `rby1_humanoid/scripts/render_hw_overlay.py`'s
  module docstring for how that project worked around it (copying the few
  needed constants/functions verbatim instead of importing across the clash).
- **Table/shelf sizing**: the table is `2.2 x 1.6 x 0.04` m (`table_top.obj`),
  bumped up once already after looking too small next to the shelf/arms —
  regenerate with `make_textured_box.py` at a different size if it still
  looks off, and remember the table's visual pose in `table_metal/table.sdf`
  is offset by `-thickness/2` so its TOP surface stays at local z=0 (matching
  `table_origin`'s weld position in `scene.dmd.yaml` — don't move that weld
  without also fixing this offset).
- **VTK exposure**: `exposure=7.0` in `render_segment.py`'s `RenderEngineVtkParams`
  — 15.0 was tried and was too bright. Don't add explicit directional lights;
  VTK's own default lighting was tried against two explicit lights and the
  explicit ones rim-lit everything into near-silhouettes instead of actually
  illuminating the arms.
