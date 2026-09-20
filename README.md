# ReVAMP Hardware Video

Annotated highlight video of the FR3 tracing the wooden maze with real-time
ReVAMP replanning around moving obstacles. The pipeline calibrates the
camera (intrinsics + extrinsics + clock offset) against ground-truth robot
pose from the logs, then renders overlay variants (executed-path trace,
replan panel, sim ghost, live plan) on the raw phone footage.

## Prerequisites (things NOT in this repo)

1. **The source video** — `shru_revamp_vid_hanlanphone-001.MOV` (~2.3 GB,
   3840×2160@59.97, the single static-camera take). Place it at the repo
   root. It is gitignored; every user provides their own copy. The build
   driver checks for it up front and refuses to start without it.
2. **The CRAMP codebase** — cloned at `codebase/` (also gitignored):

   ```
   git clone --recursive git@github.com:ShrutheeshIR/CRAMP.git codebase
   ```

   Only the robot model is used:
   `codebase/fr3_trajopt/models/fr3_marker/fr3_expo_spherized.urdf`.
   A pristine checkout is fine — the `gl_meshes` build stage converts the
   .dae visual meshes to Drake-renderable OBJs itself (into
   `meshes/visual_gl/` + a generated `fr3_expo_spherized_gl.urdf`).
3. **ffmpeg / ffprobe** on PATH.
4. **Python 3.12 venv** at `venv/`:

   ```
   uv venv venv
   uv pip install --python venv/bin/python -r requirements.txt
   ```

   (Plain `python3 -m venv venv && venv/bin/pip install -r requirements.txt`
   works too.)

## Building

One command, resumable:

```
venv/bin/python scripts/build_video.py           # everything
venv/bin/python scripts/build_video.py --list    # show stages
venv/bin/python scripts/build_video.py --dry-run # what would run
venv/bin/python scripts/build_video.py --force-from <stage>
```

Outputs land in `out/` (gitignored): `highlight_4k.mp4` (annotated cut,
225–275 s) and `highlight_ghost_4k.mp4` (same cut with the matched-camera
sim ghost + interpolated obstacle bubbles composited underneath).

### Skipping the calibration recompute

The calibration artifacts in `calib/` are **committed**, so you do not need
to re-run the expensive calibration stages (`deep_calib` alone is hours of
Powell iterations over rendered silhouettes). Staleness is mtime-based and
a fresh clone gives every file the same checkout time, so before the first
build mark the committed calibration as current:

```
touch calib/*.json
```

Then `build_video.py` runs only the mesh-conversion and render stages.
Delete files in `calib/` (or use `--force-from`) if you actually want to
re-derive the calibration from the video.

### Ad-hoc renders (bypassing build_video.py)

`render_overlay.py` can be run directly for a custom segment/flag
combination instead of going through a pipeline stage, e.g. to render
almost the whole video with the live plan and the RRT-connect explored
tree overlaid:

```
python3 scripts/render_overlay.py --segment 5.0 450.0 --plan --tree --no-skeleton
```

## Data

- `final_vid_0917/` — the original hardware run's logs:
  `joint_states.csv` (11187 rows @ ~28.7 Hz: t, q1–q7, eef xyz, z-error,
  orientation error) and `planning_queries.jsonl` (101 replan events:
  start/goal states, smm, solve times/iterations, per-query obstacle
  spheres in the robot base frame — but no waypoint arrays).
- `maze_expt_logs/` — offline deterministic replay of the same log that
  reconstructs the actual planned trajectories: `trajectories/NNN.npy` is
  an (n, 7) float32 joint-space waypoint array for original-log row NNN
  (only solved rows have files). All 80 solved rows reproduce the original
  `num_waypoints` exactly. See `maze_expt_logs/planned_traj_readme` for the
  full schema and the replay methodology.
- `calib/` — committed calibration artifacts. `camera_deep.json` is the
  authoritative camera (mesh-silhouette fit over the whole video):
  f (px), principal point, extrinsics (cv2 convention, base→camera), and
  `delta_s` (video time = log time − `video_start_epoch` + `delta_s`).
- `apriltag_desc.txt` — where the AprilTag sits relative to the robot base.
  The tag is printed on slightly wrinkled paper, so it is only used to
  *initialize* the calibration; robot FK is the ground truth.

## How the calibration works (short version)

Point-feature tracking on the near-planar eef path is degenerate (focal
length trades off against camera distance), so the authoritative fit is
silhouette-based: render the URDF through Drake GL at candidate camera
parameters and maximize interior-vs-ring brightness contrast of the
projected robot silhouette against the real frames (white robot, black
curtains). Stages: tool-frame identification from FK vs logged eef →
wall-clock sync → motion-correlation clock offset → AprilTag PnP init →
sphere-silhouette Powell fit → full mesh-silhouette refinement
(`deep_calib.py`, trimmed-mean contrast over 120 frames spanning the whole
video). Expect a few-mm floor between FK and the physical arm (known FR3
manufacturing-tolerance issue) — that is not a calibration bug.

## Repo layout

```
scripts/         pipeline code (build_video.py is the entry point)
calib/           committed calibration artifacts (small JSON)
final_vid_0917/  hardware run logs
maze_expt_logs/  replayed planned trajectories (+ schema readme)
notes/           tracked project notes, incl. Claude's working memory
scratch/         gitignored diagnostics / preview renders
out/             gitignored rendered deliverables
paper/           the ReVAMP paper source (terminology + colors)
```
