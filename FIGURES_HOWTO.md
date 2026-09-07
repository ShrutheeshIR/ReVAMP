# Reproducing the CRAMP figures and video

Everything the paper's maze figures and supplementary video are made of, from a
bare clone. The pipeline is **Drake → Meshcat `StaticHtml()` → Blender → Cycles →
ffmpeg**, driven entirely by scripts under `scripts/figures/`; there is no manual
Blender work anywhere in it, so every figure is reproducible from a command line.

`scripts/figures/README.md` is the reference for the pipeline's design and its
settled defaults. This file is the setup-and-run path.

---

## 0. What you need

| | version | notes |
|---|---|---|
| Python | 3.12 | with `pydrake`, `numpy`, `matplotlib` |
| Blender | **5.0.x** | 5.0.1 is what these figures were made with |
| `meshcat_html_importer` | **v0.1.3** | Blender add-on, not bundled here — see §2 |
| ffmpeg | any recent | system package, for the video encode |
| GPU | NVIDIA (OptiX/CUDA) | strongly preferred; Cycles on CPU is the same picture, ~20× slower |

`trimesh` + `pycollada` are needed only if you re-run `convert_visual_meshes.py`,
which you should not need to — its output is committed.

---

## 1. Blender 5.0.1

Blender 5.0 is **required**, not merely preferred: the add-on declares
`blender_version_min = "5.0.0"`, and 5.0 changed two APIs the pipeline touches
(actions became slotted, and the COLLADA importer was removed).

Download the portable tarball rather than using a distro package, which is
usually older:

```bash
mkdir -p ~/opt && cd ~/opt
wget https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz
tar -xf blender-5.0.1-linux-x64.tar.xz
~/opt/blender-5.0.1-linux-x64/blender --version    # expect: Blender 5.0.1
```

`~/opt/blender-5.0.1-linux-x64/blender` is exactly where
`scripts/figures/blender_paths.py` looks by default. Anywhere else, set
`$BLENDER` to the executable and everything will find it.

## 2. The `meshcat_html_importer` add-on

Not bundled with this repository. It is Nicholas Pfaff's, BSD-2-Clause:

> <https://github.com/nepfaff/drake-blender-recorder>, path
> `blender_addons/meshcat_html_importer`, pinned at tag **`v0.1.3`**

Blender loads user extensions from a fixed directory, so the add-on tree has to
be copied there:

```bash
git clone --depth 1 --branch v0.1.3 \
    https://github.com/nepfaff/drake-blender-recorder.git /tmp/dbr
mkdir -p ~/.config/blender/5.0/extensions/user_default
cp -r /tmp/dbr/blender_addons/meshcat_html_importer \
      ~/.config/blender/5.0/extensions/user_default/
```

**Then verify it, and do not skip this.** A missing or broken add-on does not
raise: Blender renders a clean picture of an empty scene and exits 0.

```bash
python3 scripts/figures/check_meshcat_importer.py --probe
# expect: [addon] ok: meshcat_html_importer 0.1.3 at ...
#         [addon] probe ok: imported inside /home/.../blender
```

`--probe` launches Blender and imports the add-on inside it, which is the only
check that catches version skew. The plain `--check` only looks at files on disk.

## 3. Python and Drake

```bash
pip install -r fr3_trajopt/requirements.txt \
    --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly
```

That pins the Drake nightly the project was built against. Nightly wheels are
retained about 56 days, so an old pin may 404 — if so, take a recent nightly and
expect to sanity-check the scene, since the pipeline uses Meshcat recording APIs
that do occasionally move.

## 4. Trajectories

**Not in the repository** — `out/` is gitignored, so planned paths are not
committed and you have to supply them.

The format is a VAMP *ambient path*: one comma-separated 7-DOF joint
configuration per line, as written by `write_ambient_path`. Put them anywhere;
`out/plans/` is the convention:

```bash
mkdir -p out/plans
# copy your problem_*.txt files here
```

Sanity-check that a file really belongs to this scene before rendering it — the
maze in `fr3_trajopt/resources/environments/maze_cuboids.json` is a specific
124-plank board, and a path planned against a different one will look plausible
and be wrong:

```bash
python3 scripts/figures/maze_manifest.py --selftest
# expect: base 1, border 2, wall 68, floor 28, lid 25   (124 total)
```

A correct FR3 maze plan, checked against this scene, has zero joint-limit
violations, is collision-free at every waypoint, and holds the marker tip at a
**constant z within each file** (the plane constraint the task is defined by).
`export_maze_scene.py` prints the tip z range; if it is not flat to a fraction of
a millimetre, the trajectory is not for this task.

---

## 5. Render

### The scene export (Drake side)

```bash
python3 scripts/figures/export_maze_scene.py \
    --trajectory out/plans/problem_108.txt
```

Writes into `out/figures/scene/`: the meshcat HTML, `planks.json` (plank roles),
`tip_path.json` (the marker's path), and `scene_meta.json` (provenance — source
file hash, timing, Drake version, and the retiming verification report).

Check the line `[export] recorded N frames`. **N must be greater than 1.** One
frame means the animation collapsed, which then renders as a chronophotography
figure with N identical ghosts.

Without `--trajectory` you get a static scene, which is enough for the maze
figures but not for anything showing motion.

### The paper hero figure (chronophotography)

```bash
python3 scripts/figures/render_still.py --force \
    --html out/figures/scene/problem_108.html \
    --mode ghosts --n-poses 5 --pose-nudge 0 0.05 0 0 0 \
    --lid-variant keep --frame-subject scene \
    --camera-azimuth 0 --camera-elevation 45 --trace-radius 0.0035 \
    --resolution 2800 1500 --samples 160 \
    --out out/figures/hero_ghosts.png
```

Five poses spaced by arc length along the motion, ghosted at alpha 0.45, with the
marker's traced path in the accent red. Output is RGBA with a transparent
background, so it drops onto the paper's page. ~3 minutes on an RTX 3080 Ti.

`--pose-nudge` shifts individual poses along the path; the value above moves the
second one slightly later, which was chosen by eye to keep it clear of the first
pose's forearm in projection.

### The video

```bash
python3 scripts/figures/render_video.py \
    --html out/figures/scene/problem_108.html \
    --lid-variant keep --camera-azimuth 0 --camera-elevation 45 \
    --trace-radius 0.0035 --width 1920 --height 1080 --samples 64 \
    --hold-start 1.0 --hold-end 1.5 \
    --out out/figures/problem_108.mp4
```

156 frames plus the holds, ~25 minutes. Check framing first with
`--preview-frames 24 --width 960 --height 540 --samples 16`, which takes about a
minute.

### The 2D schematic

```bash
cd figures && python3 maze_schematic.py --show-lids
```

No Drake and no Blender; add `--tip-path ../out/figures/scene/tip_path.json` to
overlay the marker's path.

---

## 6. Iterating

The pipeline is built so a bad guess is cheap to find. `--preview` drops stills
to 1000×563 at 24 samples, and sweeps render a comparison montage:

```bash
# camera search: 15 previews, a couple of minutes
python3 scripts/figures/render_still.py --preview \
    --azimuth-sweep 0 20 35 50 65 --elevation-sweep 25 40 55 --contact-sheet

# the four maze-lid treatments side by side
python3 scripts/figures/render_still.py --preview --lid-variant all --contact-sheet
```

Renders are cached on the full render request, so changing one flag re-renders
only what it affects. **The cache does not key on the source code**, so pass
`--force` after editing anything under `scripts/figures/`.

Sweeps run serially by design — two Blender processes sharing one GPU have
deadlocked on this hardware, and there is deliberately no `--jobs` flag.

---

## 7. Verifying a render actually worked

This pipeline's characteristic failures all look exactly like success, so each
stage prints a line worth reading. In order:

| line | what it must say |
|---|---|
| `[import]` | ≥136 objects. Near zero means the add-on failed. |
| `[classify]` | `robot=12 planks=124/124 ... deleted=0`. `deleted > 0` means the HTML is stale. |
| `[lids]` | the variant you asked for |
| `[trace]` | point count, and a source z range that is flat |
| `[camera]` | `fill=(fx, fy)` — well under 1.0 on an axis means change the *resolution aspect*, not the camera |
| `[render]` | `CYCLES on GPU via OPTIX`. Anything else and you are on the CPU. |
| sentinel | `MAZE_STILL_DONE` / `MAZE_VIDEO_DONE`, which the drivers require in addition to a zero exit |

For video, watch the frame count climb rather than the log. Cycles is bursty
between frames, so one idle moment means nothing, but a flat count together with
sustained 0% GPU is the hang signature; `render_video.py` prints a heartbeat for
this.

**The growing trace is the one thing to check by measurement, not by eye.** It
has to track the marker in *time*; a TOPPRA-retimed trajectory varies its speed,
so anything that advances the trace by arc length drifts against the robot. An
earlier version was "verified" by looking at two frames and passed while running
5% of the path ahead of the marker mid-clip, which is obvious in motion. The
correct setting is `bevel_factor_mapping_end = 'RESOLUTION'` with the spline
resampled to one control point per frame; see `blender_common.animate_trace_growth`
for the measured mapping table.

---

## 8. Things that will bite you

- **COLLADA meshes vanish silently.** The add-on handles only `gltf`, `glb` and
  `obj`. The FR3's visual meshes were therefore converted to OBJ
  (`fr3_trajopt/models/fr3_marker/README.md`); the `.dae` originals are still
  there but are not what the URDF points at. `python3
  scripts/figures/convert_visual_meshes.py --check` verifies this.
- **Text glTF is not a substitute for OBJ here.** It is a Y-up format and these
  meshes are Z-up, so they import with y and z swapped. Binary `.glb` is worse:
  Drake refuses to publish it from a URDF at all.
- **`context.SetTime(t)` is required while recording**, or every frame lands at
  t=0. None of the `fr3_trajopt/notebooks/` playback loops do this, because they
  do not record — so copying one into a recorder is the trap.
- **60 of the 124 planks share the plane z = 0.1025 exactly.** Exactly coplanar
  faces render as black patches on Cycles' CPU backend; `--z-fight-nudge` lifts
  the lids by 0.1 mm to break the tie.
- **Stacked transparency needs bounce budget.** `transparent_max_bounces = 32` is
  set for the ghosts; without it overlapping poses go black.
