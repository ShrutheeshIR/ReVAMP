"""Stage A: identify which URDF frame + constant offset the logged eef is.

The planner logged eef_x/y/z from its own (VAMP/cricket) kinematic chain. To
project logged data through the camera we need that same 3D point from the
Drake URDF chain. For each candidate frame F we solve, in closed form, the
constant body-frame offset r minimizing || p_F(q_t) + R_F(q_t) r - eef_t ||
over a subsample of the log, then keep the best frame.

Writes calib/tool_offset.json: {frame, offset, rms_m, per_frame_rms}.

The two chains agree to sub-mm (they model the same robot); the PHYSICAL arm
may still deviate from FK by a few mm (loose Panda manufacturing tolerances),
which downstream stages absorb with robust losses.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def fit_offset(arm, frame, qs, targets):
    """Least-squares body-frame offset r for one candidate frame.

    Stacking R_t r = (target_t - p_t) over N samples is a linear LS problem.
    """
    Rs, ps = [], []
    for q in qs:
        X = arm.frame_pose(frame, q)
        Rs.append(X[:3, :3])
        ps.append(X[:3, 3])
    A = np.concatenate(Rs, axis=0)              # (3N,3)
    b = np.concatenate(targets - np.array(ps))  # (3N,)
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = (A @ r - b).reshape(-1, 3)
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return r, rms


def main():
    js = common.load_joint_states()
    qs = common.joint_matrix(js)
    eef = common.eef_matrix(js)

    # Subsample for speed; spread over the whole log for conditioning.
    idx = np.linspace(0, len(qs) - 1, 400).astype(int)
    arm = common.ArmKinematics()

    results = {}
    for frame in arm.CANDIDATE_FRAMES:
        r, rms = fit_offset(arm, frame, qs[idx], eef[idx])
        results[frame] = {"offset": r.tolist(), "rms_m": rms}
        print(f"{frame:20s} rms {rms * 1000:8.3f} mm  offset "
              f"[{r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}]")

    # All frames on the last link tie at ~0 rms (rigidly connected); the
    # meaningful pick is the frame whose fitted offset is smallest — i.e. the
    # frame the planner's chain actually reported.
    best = min(results, key=lambda f: (round(results[f]["rms_m"], 6),
                                       np.linalg.norm(results[f]["offset"])))
    print(f"\nbest: {best}  ({results[best]['rms_m'] * 1000:.3f} mm rms, "
          f"|r| = {np.linalg.norm(results[best]['offset']) * 100:.2f} cm)")

    common.write_json(
        os.path.join(common.CALIB_DIR, "tool_offset.json"),
        {"frame": best, "offset": results[best]["offset"],
         "rms_m": results[best]["rms_m"], "all_candidates": results,
         "n_samples": len(idx)})


if __name__ == "__main__":
    main()
