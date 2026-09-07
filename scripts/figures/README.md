# Figure and animation pipeline

Renders the FR3 tracing the maze, for the CRAMP paper and its supplementary
video. The route is **Drake → Meshcat `StaticHtml()` → Blender (`meshcat_html_importer`)
→ Cycles → PNG → ffmpeg**, the house style shared with the sibling repos under
`../ift/`. Deliberately not Drake's `RenderEngineVtk` or `RenderEngineGltfClient`:
the point of the HTML route is that the add-on maps meshcat materials, opacity
and the recorded animation onto Blender objects.

## Quick start

```bash
# 0. verify the add-on actually imports inside Blender (see "Why so much checking")
python3 scripts/figures/check_meshcat_importer.py --probe
python3 scripts/figures/maze_manifest.py --selftest

# 1. Drake side: meshcat HTML + plank manifest + tip path
python3 scripts/figures/export_maze_scene.py --trajectory out/plans/.../problem_108.txt

# 2. stills
python3 scripts/figures/render_still.py --mode static     # one pose
python3 scripts/figures/render_still.py --mode ghosts --n-poses 4 --camera-elevation 55

# 3. video
python3 scripts/figures/render_video.py --preview-frames 24   # check framing first
python3 scripts/figures/render_video.py
```

Everything writes under `out/figures/`, which is gitignored. Trajectories live
in `out/plans/` and are not committed either.

## Iterating

The whole pipeline is built around making a bad guess cheap to discover:

```bash
# 15 previews at 1000x563 / 24 samples, then a montage to compare
python3 scripts/figures/render_still.py --preview \
    --azimuth-sweep 0 20 35 50 65 --elevation-sweep 25 40 55 --contact-sheet

# all four lid treatments side by side
python3 scripts/figures/render_still.py --preview --lid-variant all --contact-sheet
```

`--preview` drops to 1000×563 at 24 samples; a sweep of fifteen takes a couple of
minutes rather than an hour. Renders are cached on the *full* render request, so
re-running with one flag changed re-renders only what that flag affects; pass
`--force` to ignore the cache.

Sweeps run **serially on purpose**. Two Blender processes sharing one GPU
deadlocked on this hardware in the sibling repo, so there is no `--jobs` flag.

## Settled defaults, and why

| setting | value | reason |
|---|---|---|
| camera | az 0, el 40–55 | The board is 1.415 × 0.575 × 0.115 m — twelve times longer than tall. At el 25 it foreshortens into a strip showing only its lids; az ≥ 35 turns it diagonal and the arm starts occluding it. Settled by contact sheet, not guessed. |
| `--light-strength` | 0.5 | At 1.0 the pale board clips to white and the channels stop reading as recessed; at 0.25 the arm goes muddy. |
| `--key-elevation` | 30 | Low, so the 9.5 cm walls throw short shadows into the channels and the maze reads as relief. The sibling repo's 45 gives a flat wash on a horizontal subject. |
| `--lid-variant` | keep | The arm passes through the gaps between lids, so they are part of the task. `cull` reads the maze best; `glass` and `cutaway` exist for when the path needs to be seen through a cap. |
| `--trace-lift` | 0.0015 m | The trace is otherwise coincident with geometry it sits on. |
| `--z-fight-nudge` | 1e-4 m | 60 of the 124 planks share the plane z = 0.1025 exactly. Exactly coplanar faces render as black patches under Cycles' CPU/Embree backend. |
| ghost poses | 4–5 | The FR3's upper links barely move while tracing, so six or more stack into an unreadable mass while only the wrist separates. |
| stills background | transparent | `film_transparent` + the `Standard` view transform, so figures drop onto the paper's white page. Video uses the opaque `#1a1a2e`. |
| `--hold-start` / `--hold-end` | 1.0 / 1.5 s | The clip is only ~5 s, so it needs a beat at each end to read. Applied with ffmpeg's `tpad` at encode time, so changing them costs no GPU time. |

## Files

| file | runs in | purpose |
|---|---|---|
| `blender_paths.py` | python | Find Blender. Verbatim port from the sibling repos. |
| `check_meshcat_importer.py` | python | Verify the add-on; `--probe` imports it inside Blender. |
| `maze_manifest.py` | python | The one definition of the 124 planks' roles and meshcat paths. |
| `convert_visual_meshes.py` | python | COLLADA → OBJ for the FR3's visual meshes. One-time; `--check` to verify. |
| `export_maze_scene.py` | python | **The only file that imports pydrake.** Scene → HTML + manifest + tip path + provenance. |
| `blender_common.py` | Blender | Import, classify, lids, materials, trace, ghosts, lights, camera, render config. |
| `blender_maze_still.py` | Blender | `--mode static` / `--mode ghosts`. |
| `blender_maze_video.py` | Blender | Animation → PNG frames. |
| `render_still.py` | python | Drives stills; sweeps, cache, post-checks. |
| `render_video.py` | python | Drives frames, then ffmpeg. |
| `contact_sheet.py` | python | Montage for comparing variants. |

`figures/maze_schematic.py` is the 2D top-down companion; it reads
`maze_manifest.py` so the 2D and 3D figures cannot disagree about the maze.

## Why so much checking

Every stage prints a countable line and the drivers assert on it, because the
characteristic failure of this pipeline is one that **looks exactly like
success**:

- A missing add-on makes Blender render an empty scene and exit 0.
- Cycles silently falling back from GPU to CPU changes only the wall time.
- A `.dae` mesh is dropped by the add-on with no warning at all — this is why
  the FR3's visual meshes are now OBJ; see `fr3_trajopt/models/fr3_marker/README.md`.
- Forgetting `context.SetTime(t)` while recording collapses the animation to a
  single pose, which then renders as a chronophotography figure with N identical
  ghosts.

So: `[import]` object count, `[classify] robot=12 planks=124/124`, `[lids]`
counts, `[camera] fill=(fx, fy)`, `[render] CYCLES on GPU via OPTIX`, and a
`MAZE_STILL_DONE` / `MAZE_VIDEO_DONE` sentinel that is required in addition to a
zero exit status. A `fill` well under 1.0 means the *resolution aspect* wants
changing, not the camera.

**The growing trace must track the marker in time, not arc length.** The
animation advances in equal steps of time; a TOPPRA-retimed trajectory varies its
speed, so arc length and time are not proportional. Measured on this scene,
`bevel_factor_mapping_end` behaves as:

| mapping | maps by |
|---|---|
| `SEGMENTS` | arc length |
| `SPLINE` | arc length (identical to SEGMENTS) |
| `RESOLUTION` | control-point index |

So `RESOLUTION` is the only correct choice, together with a spline resampled to
one control point per animation frame. **Do not check this by eye on a couple of
frames** -- an earlier version was "verified" that way and passed while still
running up to 5% of the path ahead of the marker mid-clip, which is plainly
visible in motion. Measure instead: at several frames, find the path index of the
robot's tip and the path index the drawn trace reaches, and compare. With
`RESOLUTION` the two agree to a few tenths of a percent.

Note the frame cache keys on the render *request*, not on this source code, so
pass `--force` after changing anything in `blender_common.py`.

For video, watch the frame count climb rather than the log: Cycles is bursty
between frames, so one idle moment means nothing, but a flat count together with
sustained 0% GPU is the hang signature. `render_video.py` prints a heartbeat for
exactly this.

## Requirements

Blender 5.0.x, the `meshcat_html_importer` add-on (upstream
<https://github.com/nepfaff/drake-blender-recorder>, tag `v0.1.3`, installed into
`~/.config/blender/5.0/extensions/user_default/`), ffmpeg, and a Python with
pydrake, numpy and matplotlib. Blender and ffmpeg are system packages, not pip
ones. A CUDA/OptiX GPU is strongly preferred; Cycles on CPU produces the same
image far more slowly, and the drivers will tell you which one ran.

**Setting all of that up from a bare clone, and the trajectories the renders
need, is covered in `FIGURES_HOWTO.md` at the repo root.**
