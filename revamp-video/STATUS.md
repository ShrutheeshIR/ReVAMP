# ReVAMP Hardware Video — Status

This is a companion to `README.md` (which covers prerequisites, the build
command, data schemas, and the calibration methodology in detail — read
that first). This file is a snapshot of *where things stand in this
checkout* and *what's left to do*, so it's expected to go stale faster
than `README.md` and shouldn't be treated as a permanent doc.

## What this repo does

Builds one annotated highlight video: the FR3 tracing the wooden maze
with real-time ReVAMP replanning around moving obstacles. It calibrates a
static phone camera (intrinsics + extrinsics + clock offset) against
ground-truth robot pose reconstructed from the hardware run's logs, then
renders overlay variants (executed-path trace, replan panel, sim ghost,
live plan) on top of the raw 4K phone footage. `scripts/build_video.py` is
a resumable, stage-based build driver — re-running it only redoes stages
whose inputs actually changed (see `scripts/build_pipeline.py`'s
docstring for the staleness mechanics).

## Status in this checkout

- **Calibration: done and committed.** `calib/camera_deep.json` is a
  converged mesh-silhouette fit (commit history shows the "deep calib"
  objective converged to 94.78). Nothing needs to be re-derived unless the
  source video or robot model changes.
- **Data: present and committed.** `final_vid_0917/` (the original
  hardware run's log) and `maze_expt_logs/` (offline-replayed planned
  trajectories, 80/101 logged replan rows solved and have `.npy` files)
  are both in the repo.
- **Not yet built here.** `out/` doesn't exist in this checkout — nobody
  has run `build_video.py` yet, so the render stages (`montage`,
  `highlight`, `highlight_ghost`, `highlight_plan`) haven't been
  smoke-tested in this environment.
- **Missing local prerequisites**, all gitignored by design: `codebase/`
  (the CRAMP clone, for the robot URDF), `venv/` (Python env), and the
  2.3 GB source `.MOV` are all absent right now.
- **No combined "ghost + live plan" render stage exists.**
  `scripts/render_overlay.py` supports `--ghost` and `--plan` together
  (its own filename logic already handles it —
  `highlight_plan_ghost_4k.mp4`), but `scripts/build_video.py`'s `STAGES`
  list only wires up `highlight`, `highlight_ghost`, and `highlight_plan`
  individually. If the final deliverable needs sim-ghost + live-plan +
  obstacle-bubbles all at once, that stage doesn't exist yet.
- Working tree is clean; nothing here is mid-edit.

## What needs to be done

1. **Get the source video** onto the repo root —
   `shru_revamp_vid_hanlanphone-001.MOV`. Not tracked; every user supplies
   their own copy. `build_video.py` refuses to start without it.
2. **Clone the CRAMP codebase** at `codebase/` for the robot URDF (see
   README's Prerequisites section for the exact command).
3. **Set up the Python venv** and install `requirements.txt`.
4. **`touch calib/*.json`** before the first build, so a fresh checkout
   doesn't look older than the committed calibration and trigger an
   unnecessary multi-hour recompute.
5. **Run `venv/bin/python scripts/build_video.py`** to produce
   `out/highlight_4k.mp4` and `out/highlight_ghost_4k.mp4`.
6. **Check the montage** (`scratch/verify_montage.jpg`, 12 frames spanning
   the video) before trusting the full highlight cut — it's the cheapest
   way to catch a calibration or overlay regression before spending render
   time on the full segment.
7. **If a combined ghost+plan cut is wanted**, add a stage to `STAGES` in
   `scripts/build_video.py` modeled on the existing `highlight_ghost` /
   `highlight_plan` entries, calling `render_overlay.py --segment 225 275
   --ghost --plan` with `deps` covering `common.QUERIES_JSONL`,
   `common.TRAJ_DIR`, and `common.URDF_GL_PATH`.
8. **Check whether the 21/101 unsolved replan rows** in `maze_expt_logs/`
   (rows with no `trajectories/NNN.npy`) fall inside the highlighted
   225–275 s window — only worth fixing if the live-plan overlay needs
   continuous coverage there.

None of this requires re-deriving calibration — that part is already
finished and pinned by the committed `calib/*.json` files.
