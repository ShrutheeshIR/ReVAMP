"""Kinematic model-error fit against the video (viz-only, full video).

The night3 camera fit left a pose-dependent error (~2 cm at the wrist in
the outstretched start pose, tip fine) that no camera fit can absorb: it
is robot-model error, expected from the FR3's loose tolerances. This fits
a viz-only correction — per-joint angle biases dq(7) plus an SE(3) tweak
of the link7->flange frame (translation xyz + roll/pitch; yaw excluded as
degenerate with dq7) — applied through the hooks in common.py, so every
renderer picks the result up from calib/model_correction.json.

Objective (extends night_calib's; frames span the whole video):

    score =   contrast_trim
            - 0.5 * chamfer_all          (whole-silhouette edge fit)
            - 1.0 * chamfer_tool         (edge fit near marker_holder)
            - 2.0 * chamfer_flange       (edge fit near link8 — the flange
                                          collar, the most salient feature)
            - 0.2 * track_med_px         (NCC marker track reprojection,
                                          dq-aware FK, r frozen at night3)

Stages (env KIN_STAGE):
  K  — kinematic params only (12, or 15 with KIN_TIER_B=1 adding a
       hand->marker_holder translation), camera frozen at KIN_CAM.
  J  — joint polish: camera (f cx cy rvec tvec delta, 10) + kinematic,
       camera tightly bounded around KIN_CAM.

Powell in pre-scaled units (all params O(1)) with bounds, jittered
restarts, checkpoint-on-improvement to KIN_OUT (model_correction.json
schema), and a held-out frame set scored at every checkpoint as an
overfitting alarm. Env knobs: KIN_STAGE KIN_SCALE KIN_SCHED KIN_DT_CAP
KIN_HOURS KIN_INIT KIN_CAM KIN_OUT KIN_TIER_B KIN_W_*.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

STAGE = os.environ.get("KIN_STAGE", "K")
SCALE = int(os.environ.get("KIN_SCALE", 2))
SCHED = os.environ.get("KIN_SCHED", "coarse")
DT_CAP = float(os.environ.get("KIN_DT_CAP", 30.0))
HOURS = float(os.environ.get("KIN_HOURS", 2.5))
TIER_B = os.environ.get("KIN_TIER_B", "0") == "1"
RING_PX = max(4, 32 // SCALE)
TRIM = 0.15               # arm/tool terms: drop this fraction of frames
TRIM_FLANGE = 0.25        # flange is maze-occluded in some chunks
TOOL_BOX = 440 // SCALE
FLANGE_BOX = 260 // SCALE
W_CHAMFER = float(os.environ.get("KIN_W_CHAMFER", 0.5))
W_TOOL = float(os.environ.get("KIN_W_TOOL", 1.0))
W_FLANGE = float(os.environ.get("KIN_W_FLANGE", 2.0))
W_TRACK = float(os.environ.get("KIN_W_TRACK", 0.2))
INIT = os.environ.get("KIN_INIT", "")            # prior correction json
CAM = os.environ.get("KIN_CAM",
                     os.path.join(common.CALIB_DIR, "camera_deep.json"))
OUT = os.environ.get("KIN_OUT",
                     os.path.join(common.SCRATCH, "kin_fit.json"))
TOOL_JOINT = "fr3_marker_holder_joint"           # tier-B translation site

os.environ["DEEP_SCALE"] = str(SCALE)
import deep_calib  # noqa: E402

assert deep_calib.SCALE == SCALE

# ---------------------------------------------------------------- schedule --
T_RANGE = (9.0, 384.0)
CLUSTERS = [(9, 13), (232, 238), (283, 289), (347, 352)]


def frame_times():
    if SCHED == "coarse":
        n_uni, n_cl = 48, 6
    elif SCHED == "dense":
        n_uni, n_cl = 80, 10
    else:
        raise SystemExit(f"unknown KIN_SCHED {SCHED}")
    ts = [np.linspace(*T_RANGE, n_uni)]
    ts += [np.linspace(a, b, n_cl) for a, b in CLUSTERS]
    return np.unique(np.round(np.concatenate(ts), 3))


def holdout_times():
    # Offset half a coarse step from the uniform grid; never optimized.
    step = (T_RANGE[1] - T_RANGE[0]) / 47
    return np.round(np.linspace(*T_RANGE, 24) + step / 2, 3)[:-1]


def load_frames(times, tag):
    cache = os.path.join(common.SCRATCH, f"kin_frames_s{SCALE}_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z["times"]) == len(times) and np.allclose(z["times"], times):
            return z["frames"]
    import subprocess
    w, h = common.VIDEO_WH[0] // SCALE, common.VIDEO_WH[1] // SCALE
    frames = []
    for t in times:
        cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i",
               common.VIDEO, "-frames:v", "1", "-vf", f"scale={w}:{h}",
               "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        buf = subprocess.run(cmd, capture_output=True, check=True).stdout
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w))
        if len(frames) % 20 == 0:
            print(f"[{tag}] loaded {len(frames)}/{len(times)}", flush=True)
    frames = np.array(frames)
    os.makedirs(common.SCRATCH, exist_ok=True)
    np.savez_compressed(cache, times=times, frames=frames)
    return frames


def edge_dts(frames):
    out = []
    for img in frames:
        edges = cv2.Canny(img, 40, 120)
        dt = cv2.distanceTransform((edges == 0).astype(np.uint8),
                                   cv2.DIST_L2, 3)
        out.append(np.minimum(dt, DT_CAP).astype(np.float32))
    return np.array(out)


# ------------------------------------------------------------- FK (dq-aware) --
class FK:
    """On-demand corrected FK at arbitrary epoch times (replaces
    calibrate.FrameTrajectory, which pre-bakes poses and can't see dq)."""

    def __init__(self):
        js = common.load_joint_states()
        self.t_log = js["t"]
        self.qs = common.joint_matrix(js)
        self.arm = common.ArmKinematics(
            correction={"dq": [0.0] * 7, "frames": {}})

    def set_correction(self, corr):
        self.arm.set_correction(corr)

    def q_at(self, te):
        te = np.atleast_1d(te)
        return np.stack([np.interp(te, self.t_log, self.qs[:, j])
                         for j in range(7)], axis=1)

    def poses(self, frame_name, te):
        """positions (N,3) and rotations (N,3,3), correction applied."""
        Xs = [self.arm.frame_pose(frame_name, q) for q in self.q_at(te)]
        Xs = np.array(Xs)
        return Xs[:, :3, 3], Xs[:, :3, :3]


# ------------------------------------------------------------------ params --
# Layout: [dq(7), flange_xyz(3), flange_rp(2) (, tool_xyz(3))] (+ camera
# [f cx cy rvec(3) tvec(3) delta] appended in stage J).
KIN_SCALES = ([0.005] * 7) + ([0.005] * 3) + ([0.01] * 2)
KIN_BOUNDS = ([0.015] * 7) + ([0.008] * 3) + ([0.026] * 2)
if TIER_B:
    KIN_SCALES += [0.005] * 3
    KIN_BOUNDS += [0.006] * 3
N_KIN = len(KIN_SCALES)
CAM_SCALES = [2.0, 2.0, 2.0] + [0.002] * 3 + [0.005] * 3 + [0.02]
CAM_BOUNDS = [8.0, 6.0, 6.0] + [0.003] * 3 + [0.006] * 3 + [0.010]


def build_corr(k):
    """Correction dict from the physical kinematic param block."""
    frames = {"fr3_joint8": {"xyz": k[7:10].tolist(),
                             "rpy": [k[10], k[11], 0.0]}}
    if TIER_B:
        frames[TOOL_JOINT] = {"xyz": k[12:15].tolist(), "rpy": [0, 0, 0]}
    return {"dq": k[:7].tolist(), "frames": frames}


def corr_to_k(corr):
    k = np.zeros(N_KIN)
    k[:7] = corr.get("dq", [0.0] * 7)
    fr = corr.get("frames") or {}
    j8 = fr.get("fr3_joint8", {})
    k[7:10] = j8.get("xyz", [0, 0, 0])
    k[10:12] = j8.get("rpy", [0, 0, 0])[:2]
    if TIER_B and TOOL_JOINT in fr:
        k[12:15] = fr[TOOL_JOINT].get("xyz", [0, 0, 0])
    return k


def main():
    cam0 = common.read_json(CAM)
    v0 = cam0["video_start_epoch"]
    track_r = np.array(cam0["track_r"])   # frozen at night3
    w4k, h4k = common.VIDEO_WH
    print(f"stage {STAGE} scale {SCALE} sched {SCHED} dtcap {DT_CAP} "
          f"tierB {TIER_B} cam {os.path.basename(CAM)} -> {OUT}", flush=True)

    times = frame_times()
    frames = load_frames(times, SCHED)
    dts = edge_dts(frames)
    ho_times = holdout_times()
    ho_frames = load_frames(ho_times, "holdout")
    ho_dts = edge_dts(ho_frames)
    n_frames = len(times)
    n_keep = int(round(n_frames * (1 - TRIM)))
    n_keep_fl = int(round(n_frames * (1 - TRIM_FLANGE)))
    print(f"{n_frames} train + {len(ho_times)} holdout frames", flush=True)

    ms = deep_calib.MeshSilhouette()
    fk = FK()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))

    track = common.read_json(os.path.join(common.CALIB_DIR, "track.json"))
    pts = track["points"]
    t_track = np.array([p["t_video"] for p in pts])
    uv_obs = np.array([[p["u"], p["v"]] for p in pts])
    track_frame = track["frame"]

    # ---- param vector assembly -------------------------------------------
    corr0 = (common.read_json(INIT) if INIT and os.path.exists(INIT)
             else {"dq": [0.0] * 7, "frames": {}})
    k0 = corr_to_k(corr0)
    cam_vec0 = np.concatenate([[cam0["f"], cam0["cx"], cam0["cy"]],
                               cam0["rvec"], cam0["tvec"],
                               [cam0["delta_s"]]])
    if STAGE == "K":
        x0_phys = k0
        scales = np.array(KIN_SCALES)
        halfw = np.array(KIN_BOUNDS)
        centers = np.zeros(N_KIN)
    elif STAGE == "J":
        x0_phys = np.concatenate([k0, cam_vec0])
        scales = np.array(KIN_SCALES + CAM_SCALES)
        halfw = np.array(KIN_BOUNDS + CAM_BOUNDS)
        centers = np.concatenate([np.zeros(N_KIN), cam_vec0])
    else:
        raise SystemExit(f"unknown KIN_STAGE {STAGE}")

    def unscale(xs):
        return centers + xs * scales

    def to_scaled(xp):
        return (xp - centers) / scales

    # Scaled space is centered on `centers`, so bounds are +/- halfw/scale.
    bounds = [(-h / s, h / s) for h, s in zip(halfw, scales)]

    def split(xp):
        k = xp[:N_KIN]
        camv = xp[N_KIN:] if STAGE == "J" else cam_vec0
        return k, camv

    def sil_terms(camv, corr, tt, fr, dt_arr, keep, keep_fl):
        """(contrast, cham_all, cham_tool, cham_flange) on a frame set."""
        f, cx, cy = camv[0], camv[1], camv[2]
        rvec, tvec, delta = camv[3:6], camv[6:9], camv[9]
        te = tt + v0 - delta
        p_tool, _ = fk.poses(track_frame, te)
        p_fl, _ = fk.poses("fr3_link8", te)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        uv_tool = cv2.projectPoints(p_tool.reshape(-1, 3), rvec, tvec, K,
                                    np.zeros(4))[0].reshape(-1, 2) / SCALE
        uv_fl = cv2.projectPoints(p_fl.reshape(-1, 3), rvec, tvec, K,
                                  np.zeros(4))[0].reshape(-1, 2) / SCALE
        contrast, c_all, c_tool, c_fl = [], [], [], []
        for i, (t_vid, img, dt) in enumerate(zip(tt, fr, dt_arr)):
            m = ms.mask(t_vid + v0 - delta, f, cx, cy, rvec, tvec)
            mu8 = m.astype(np.uint8)
            ring = (cv2.dilate(mu8, kernel) & ~mu8) > 0
            if m.sum() < 100 or ring.sum() < 100:
                contrast.append(-50.0)
                c_all.append(DT_CAP)
                c_tool.append(DT_CAP)
                c_fl.append(DT_CAP)
                continue
            gimg = img.astype(np.float32)
            contrast.append(float(gimg[m].mean() - gimg[ring].mean()))
            edges = mu8 & ~cv2.erode(mu8, None)
            ey, ex = np.nonzero(edges)
            c_all.append(float(dt[ey, ex].mean()) if len(ey) else DT_CAP)
            u, v = uv_tool[i]
            sel = (np.abs(ex - u) < TOOL_BOX) & (np.abs(ey - v) < TOOL_BOX)
            c_tool.append(float(dt[ey[sel], ex[sel]].mean())
                          if sel.sum() > 50 else DT_CAP)
            u, v = uv_fl[i]
            sel = ((np.abs(ex - u) < FLANGE_BOX)
                   & (np.abs(ey - v) < FLANGE_BOX))
            c_fl.append(float(dt[ey[sel], ex[sel]].mean())
                        if sel.sum() > 50 else DT_CAP)
        c = float(np.mean(sorted(contrast)[-keep:]))
        ca = float(np.mean(sorted(c_all)[:keep]))
        ct = float(np.mean(sorted(c_tool)[:keep]))
        cf = float(np.mean(sorted(c_fl)[:keep_fl]))
        return c, ca, ct, cf

    def track_med(camv):
        # fk correction already set by caller
        f, cx, cy = camv[0], camv[1], camv[2]
        rvec, tvec, delta = camv[3:6], camv[6:9], camv[9]
        p, R = fk.poses(track_frame, t_track + v0 - delta)
        pts3 = p + np.einsum("nij,j->ni", R, track_r)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        uv = cv2.projectPoints(pts3.reshape(-1, 3), rvec, tvec, K,
                               np.zeros(4))[0].reshape(-1, 2)
        return float(np.median(np.linalg.norm(uv - uv_obs, axis=1)))

    state = {"best": -1e9, "evals": 0, "t0": time.time(),
             "deadline": time.time() + HOURS * 3600, "x_best": None}

    class OutOfTime(Exception):
        pass

    def score(camv, corr, tt, fr, dt_arr, keep, keep_fl, with_track=True):
        ms.set_correction(corr)
        fk.set_correction(corr)
        c, ca, ct, cf = sil_terms(camv, corr, tt, fr, dt_arr, keep, keep_fl)
        tm = track_med(camv) if with_track else 0.0
        val = (c - W_CHAMFER * ca - W_TOOL * ct - W_FLANGE * cf
               - W_TRACK * tm)
        return val, {"contrast": c, "chamfer_all": ca, "chamfer_tool": ct,
                     "chamfer_flange": cf, "track_median_px": tm}

    def objective(xs):
        if state["evals"] > 0 and time.time() > state["deadline"]:
            raise OutOfTime
        xp = unscale(np.asarray(xs))
        k, camv = split(xp)
        corr = build_corr(k)
        val, comp = score(camv, corr, times, frames, dts, n_keep, n_keep_fl)
        state["evals"] += 1
        if val > state["best"]:
            state["best"] = val
            state["x_best"] = np.array(xs)
            ho_val, ho_comp = score(camv, corr, ho_times, ho_frames, ho_dts,
                                    int(round(len(ho_times) * (1 - TRIM))),
                                    int(round(len(ho_times)
                                              * (1 - TRIM_FLANGE))),
                                    with_track=False)
            out = dict(corr)
            out["meta"] = {
                "stage": STAGE, "sched": SCHED, "scale": SCALE,
                "dt_cap": DT_CAP, "objective": val, "components": comp,
                "holdout_score": ho_val, "holdout_components": ho_comp,
                "camera": os.path.basename(CAM),
                "weights": {"chamfer": W_CHAMFER, "tool": W_TOOL,
                            "flange": W_FLANGE, "track": W_TRACK},
                "note": "viz-only kinematic correction (kin_calib.py); "
                        "dq added to logged q, frames composed onto URDF "
                        "defaults; never for planning/IK."}
            if STAGE == "J":
                R = cv2.Rodrigues(camv[3:6])[0]
                out["meta"]["camera_fit"] = {
                    "image_wh": [w4k, h4k], "f": float(camv[0]),
                    "cx": float(camv[1]), "cy": float(camv[2]),
                    "k1": 0.0, "k2": 0.0, "rvec": camv[3:6].tolist(),
                    "tvec": camv[6:9].tolist(), "delta_s": float(camv[9]),
                    "camera_in_base": (-R.T @ camv[6:9]).tolist(),
                    "video_start_epoch": v0, "track_r": track_r.tolist()}
            common.write_json(OUT, out)
            dq_mr = np.abs(k[:7]).max() * 1000
            print(f"[{state['evals']:5d} evals "
                  f"{time.time()-state['t0']:6.0f}s] best {val:.3f} "
                  f"(c {comp['contrast']:.2f} ca {comp['chamfer_all']:.2f} "
                  f"ct {comp['chamfer_tool']:.2f} "
                  f"cf {comp['chamfer_flange']:.2f} "
                  f"tm {comp['track_median_px']:.1f}px "
                  f"ho {ho_val:.3f}) |dq|max {dq_mr:.1f}mrad "
                  f"fl {1000*np.linalg.norm(k[7:10]):.1f}mm", flush=True)
        return -val

    xs0 = to_scaled(x0_phys)
    xs0 = np.clip(xs0, [b[0] for b in bounds], [b[1] for b in bounds])
    print(f"initial objective {-objective(xs0):.3f} "
          f"({state['evals']} evals so far)", flush=True)

    rng = np.random.default_rng(0)
    x_start = xs0
    for round_i in range(40):
        if time.time() > state["deadline"]:
            print("time budget reached", flush=True)
            break
        try:
            res = minimize(objective, x_start, method="Powell",
                           bounds=bounds,
                           options={"maxfev": 4000, "xtol": 1e-4,
                                    "ftol": 1e-5})
        except OutOfTime:
            print("time budget reached mid-round; checkpoint stands",
                  flush=True)
            break
        print(f"round {round_i}: {-res.fun:.3f} after {res.nfev} evals",
              flush=True)
        base = (state["x_best"] if state["x_best"] is not None else res.x)
        if np.allclose(res.x, x_start, atol=1e-6):
            # Converged: restart with jitter around the incumbent best.
            jit = rng.normal(0, 0.3, size=len(base))
            x_start = np.clip(base + jit, [b[0] for b in bounds],
                              [b[1] for b in bounds])
            print("jitter restart", flush=True)
        else:
            x_start = res.x

    if state["x_best"] is not None:
        xp = unscale(state["x_best"])
        k, camv = split(xp)
        print(f"FINAL best {state['best']:.3f}", flush=True)
        print(f"dq (mrad): {np.round(k[:7]*1000, 2).tolist()}", flush=True)
        print(f"flange xyz (mm): {np.round(k[7:10]*1000, 2).tolist()} "
              f"rp (mrad): {np.round(k[10:12]*1000, 2).tolist()}",
              flush=True)
        if TIER_B:
            print(f"tool xyz (mm): {np.round(k[12:15]*1000, 2).tolist()}",
                  flush=True)
        if STAGE == "J":
            print(f"camera: f={camv[0]:.1f} c=({camv[1]:.1f},{camv[2]:.1f})"
                  f" delta={camv[9]:+.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
