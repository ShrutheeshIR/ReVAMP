---
name: kinematic-model-correction
description: "calib/model_correction.json is a viz-only kinematic fit; how it was made and why its parameterization looks the way it does"
metadata:
  node_type: memory
  pinned: false
  originSessionId: 4b97f428-3624-48bf-a751-68cab434e863
---

The ghost/overlay renderers apply a fitted kinematic correction from
`calib/model_correction.json` (adopted 2026-09-20, commit ce6c8f5):
per-joint angle biases dq plus SE(3) deltas on two joint parent frames,
loaded via `common.load_model_correction()` and applied by every
renderer automatically (`ArmKinematics`, `MeshSilhouette`,
`GhostRenderer`). Facts to remember when touching it:

1. **Visualization only** (per Tommy): it exists so the video overlay
   *looks* right. It must never feed planning, IK, or any downstream
   robot application, and non-physical parameter values are acceptable.
2. **Why it exists**: the FR3's loose tolerances leave a pose-dependent
   FK error (wrist/flange ~40–60 px off at the video start, up to
   ~90 px at t≈350 s) that no camera-only calibration can absorb.
3. **Parameterization choices** (degeneracy, not physics): the frame
   delta lives on `fr3_joint8` and NOT `fr3_hand_joint` (link8 has no
   mesh, so the two are exactly redundant; joint8's origin also centers
   the flange crop box); flange yaw is excluded (redundant with dq7);
   the NCC track template offset `track_r` stays frozen at its night3
   value (redundant with the hand→marker_holder translation). The
   hand→marker_holder translation was enabled only after gate review
   showed a flange-vs-tip tradeoff; its fitted −3.6 mm z pulls the
   rendered tip back onto the real one.
4. **Mechanism**: Drake wraps every URDF joint's parent attachment in a
   `FixedOffsetFrame` whose pose is a context parameter —
   `SetPoseInParentFrame` after `Finalize()`, no URDF edits. The applier
   resets all joint frames to URDF defaults before composing deltas
   (stale-offset bug caught by test during development). The
   `MODEL_CORRECTION_PATH` env var overrides the file so uncommitted
   candidates can be reviewed end to end.
5. **The fitter is `scripts/kin_calib.py`** (stages K/J via env knobs,
   warm-startable, checkpoint-on-improvement, held-out frames logged as
   an overfitting alarm) and the review set is
   `scripts/make_review_set.py` → `scratch/current/` (fixed filenames,
   baseline|current panels; Tommy wants exactly ONE current review set,
   overwritten in place). Independent critic agents judged each stage's
   panels before acceptance; final residual is ~3–10 px at 4K
   (~1–4 px at 1080p playback).
6. `calib/camera_deep.json` was co-polished in the final stage (f
   2718.4 → 2724.5, delta −0.7347 → −0.7247); camera and correction are
   a matched pair — refitting one without the other will degrade the
   overlay.
