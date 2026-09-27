"""Fit the camera <-> robot-base transform for a hardware clip, using the joint
configuration logged in the matching execution record.

    venv/bin/python scripts/fit_camera_from_log.py \
        --video scratch/hardware_media/raw/20260828_142748.mp4 \
        --record results/point_00_exec_20260828_142919.pkl

See ``scripts/robot_camera_fit.py`` for why this is well posed where the earlier
gripper-only fits were not.

Unknowns: the six-DOF pose ``X_WC``, one focal length (the phone's intrinsics are
not in the file), and one global time offset between the video clock and the
record's wall clock.  The principal point is held at the image centre and the
pixels are assumed square; both are re-checked at the end by how well the fit
predicts frames it never saw.

Objective: motion, not appearance.  For a pair of times, the normalised
correlation between the image difference and the exclusive-or of the two
rendered silhouettes.  Static structure -- the lab and the robot's own base and
torso -- cancels on both sides, so nothing has to segment a white robot from a
white wall.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robot_camera_fit as M


def blur(x, k):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), k)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float(a.dot(a)) * float(b.dot(b)))
    return 0.0 if d < 1e-9 else float(a.dot(b)) / d


class MotionObjective:
    """Score a camera against the observed motion in a clip.

    ``times`` are clip-relative seconds; ``pairs`` index into them.  The image
    side is fixed once (the difference images); the model side is re-rendered
    for every candidate camera.

    Two terms are summed.  A normalised cross-correlation says *where* the
    motion is, and is what pulls a wrong camera back from far away.  A soft
    intersection-over-union says *how big* it is, which is what separates a
    close short-focal camera from a distant long-focal one -- NCC alone is
    nearly blind to that trade.

    Pairs whose two frames are almost identical carry no motion and are dropped:
    the record opens with a premove that turned out to be a no-op, and scoring
    noise against noise only dilutes the pairs that do say something.
    """

    MIN_MOTION = 20.0 / 255.0        # p99.5 of |dI| below this is noise, not motion

    def __init__(self, video, t_log, q_log, t_clip0, times, pairs, width, height,
                 sigma=2.0, renderer=None, log=print):
        self.t_log, self.q_log, self.t_clip0 = t_log, q_log, t_clip0
        self.times = list(times)
        self.W, self.H = width, height
        self.sigma = sigma
        vw, vh, _, _ = M.probe(video)
        self.scale = width / float(vw)
        frames = M.grab(video, self.times, width=width, height=height, gray=True)
        self.frames = [f.astype(np.float32) / 255.0 for f in frames]
        self.diff = {}
        self.pairs = []
        for i, j in pairs:
            d = blur(np.abs(self.frames[i] - self.frames[j]), sigma)
            p = float(np.percentile(d, 99.5))
            if p < self.MIN_MOTION:
                continue
            self.pairs.append((i, j))
            self.diff[(i, j)] = np.clip(d / p, 0.0, 1.0).ravel()
        log(f"  {len(self.pairs)}/{len(pairs)} frame pairs carry motion "
            f"at {width}x{height}")
        self.r = renderer if renderer is not None else M.Renderer(width, height)
        self.evals = 0

    def masks(self, X_WC, fx_full, dt):
        """Silhouettes at every sampled time, for one camera and time offset."""
        cam = self.r.camera(fx_full * self.scale)
        out = []
        for t in self.times:
            q = M.q_at(self.t_log, self.q_log, self.t_clip0 + t + dt)
            self.r.set_config(q)
            out.append(self.r.mask(X_WC, cam))
        return out

    def score(self, X_WC, fx_full, dt):
        self.evals += 1
        ms = self.masks(X_WC, fx_full, dt)
        cover = sum(float(m.mean()) for m in ms) / len(ms)
        if cover < 0.004 or cover > 0.45:
            return -1.0                      # off frame, or filling it
        tot = 0.0
        for i, j in self.pairs:
            x = blur(np.logical_xor(ms[i], ms[j]).astype(np.float32), self.sigma)
            mx = float(x.max())
            if mx < 1e-6:
                continue
            x = np.clip(x / mx, 0.0, 1.0).ravel()
            d = self.diff[(i, j)]
            tot += ncc(d, x)
            tot += 2.0 * float(np.minimum(d, x).sum()) / float(np.maximum(d, x).sum() + 1e-9)
        return tot / len(self.pairs)


class EdgeTerm:
    """A truncated edge-chamfer polish, on top of a motion-based seed.

    A global edge chamfer is what walked the camera to 7 m in the earlier
    gripper-only attempts, and it is not used that way here.  Two things make it
    safe as a *local* term: the distance is truncated (an edge more than
    ``trunc`` pixels away contributes a constant, so the score cannot be improved
    by fleeing to where the silhouette is small), and it is only ever evaluated
    near a pose the motion objective already agrees with.

    What it adds is the static structure -- the base, the torso column, the head
    -- which cancels out of every frame difference and so is invisible to the
    motion term.  Those are the parts whose pose is known most rigidly, so they
    are worth the extra term.
    """

    def __init__(self, video, t_log, q_log, t_clip0, times, width, height,
                 renderer, trunc=14.0, log=print):
        self.t_log, self.q_log, self.t_clip0 = t_log, q_log, t_clip0
        self.times = list(times)
        self.W, self.H = width, height
        self.trunc = trunc
        vw, _, _, _ = M.probe(video)
        self.scale = width / float(vw)
        self.r = renderer
        self.dt_maps = []
        for f in M.grab(video, self.times, width=width, height=height, gray=True):
            e = cv2.Canny(cv2.GaussianBlur(f, (0, 0), 1.2), 40, 110)
            d = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)
            self.dt_maps.append(np.minimum(d, trunc))
        log(f"  edge maps for {len(self.dt_maps)} frames at {width}x{height}")

    def score(self, X_WC, fx_full, dt):
        """In [0, 1]: 1 when every silhouette boundary pixel sits on an image edge."""
        cam = self.r.camera(fx_full * self.scale)
        tot, n = 0.0, 0
        for t, dtm in zip(self.times, self.dt_maps):
            self.r.set_config(M.q_at(self.t_log, self.q_log, self.t_clip0 + t + dt))
            m = self.r.mask(X_WC, cam).astype(np.uint8)
            b = m - cv2.erode(m, np.ones((3, 3), np.uint8))
            k = int(b.sum())
            if k < 50:
                continue
            tot += float(dtm[b.astype(bool)].mean())
            n += 1
        if n == 0:
            return -1.0
        return 1.0 - (tot / n) / self.trunc


class Combined:
    def __init__(self, motion, edge, weight):
        self.motion, self.edge, self.weight = motion, edge, weight

    def score(self, X_WC, fx_full, dt):
        s = self.motion.score(X_WC, fx_full, dt)
        if s <= -1.0:
            return s
        return s + self.weight * self.edge.score(X_WC, fx_full, dt)


def coarse_search(obj, grid, log=print):
    best = (-math.inf, None)
    n = 0
    t0 = time.time()
    for phi in grid["phi"]:
        for d in grid["dist"]:
            for h in grid["height"]:
                for zt in grid["target_z"]:
                    eye = [d * math.cos(phi), d * math.sin(phi), h]
                    X = M.look_at(eye, [0.0, 0.0, zt])
                    for f in grid["focal"]:
                        s = obj.score(X, f, 0.0)
                        n += 1
                        if s > best[0]:
                            best = (s, dict(phi=phi, dist=d, height=h, target_z=zt,
                                            focal=f, X=X))
                            log(f"  [{n:5d}] score {s:+.4f}  phi={math.degrees(phi):+6.1f} "
                                f"d={d:.2f} h={h:.2f} zt={zt:.2f} f={f:.0f}")
    log(f"  coarse: {n} evaluations in {time.time()-t0:.1f} s")
    return best


def refine(obj, X0, f0, dt0, fit_dt=True, log=print, maxiter=900, restarts=3):
    """Polish the seed with Nelder-Mead over (pose, focal, time offset).

    The step scale matters and cannot be left to scipy: started from an all-zero
    vector it builds its initial simplex from 2.5e-4 *absolute*, which through
    the scaling below is a 1e-8 m camera move -- every vertex scores identically
    and the search returns the seed untouched.  So the simplex is given
    explicitly, one unit step per coordinate, and the search is restarted from
    wherever it lands.
    """
    v0 = M.vec_from_pose(X0)
    #        translation (m)      rotation (rad)      focal (px)  dt (s)
    scale = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 15.0, 0.05])
    if not fit_dt:
        scale[7] = 0.0

    def unpack(z):
        return (M.pose_from_vec(v0 + z[:6] * scale[:6]),
                f0 + z[6] * scale[6],
                dt0 + z[7] * scale[7])

    def neg(z):
        X, f, dt = unpack(z)
        if not (600.0 < f < 2600.0):
            return 1.0
        return -obj.score(X, f, dt)

    z = np.zeros(8)
    best = neg(z)
    for k in range(restarts):
        step = 1.0 / (2.0 ** k)
        simplex = np.vstack([z] + [z + step * np.eye(8)[i] for i in range(8)])
        res = minimize(neg, z, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, initial_simplex=simplex,
                                    xatol=1e-3, fatol=1e-6, adaptive=True))
        log(f"    restart {k}: score {-res.fun:+.4f} ({res.nfev} evals)")
        if res.fun < best:
            best, z = res.fun, res.x
    X, f, dt = unpack(z)
    log(f"  refine: score {-best:+.4f}  camera {np.round(X.translation(),3)}  "
        f"f={f:.1f} dt={dt:+.3f} s")
    return -best, X, f, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", default="scratch/camera_fit")
    ap.add_argument("--coarse-size", type=int, default=240)
    ap.add_argument("--fine-size", type=int, default=480)
    ap.add_argument("--final-size", type=int, default=960)
    ap.add_argument("--n-coarse", type=int, default=8)
    ap.add_argument("--n-fine", type=int, default=16)
    ap.add_argument("--no-edge-polish", action="store_true")
    ap.add_argument("--edge-weight", type=float, default=1.5)
    ap.add_argument("--seed-json", default=None,
                    help="skip the coarse grid and start from this fit file")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rec = pickle.load(open(args.record, "rb"))
    t_log, q_log = M.load_states(rec)
    t_clip0 = M.clip_start_wall(args.video)
    vw, vh, dur, _ = M.probe(args.video)
    lo, hi = t_log[0] - t_clip0, t_log[-1] - t_clip0
    print(f"video {os.path.basename(args.video)}  {vw}x{vh}  {dur:.2f} s")
    print(f"record {os.path.basename(args.record)}  {len(t_log)} states  "
          f"covering clip t = {lo:.2f} .. {hi:.2f} s")
    if lo < 0 or hi > dur:
        print("  WARNING: the logged window is not contained in the clip")

    pad = 0.6
    def sched(n):
        ts = np.linspace(lo + pad, hi - pad, n)
        prs = ([(i, i + 1) for i in range(n - 1)]
               + [(i, i + 3) for i in range(n - 3)]
               + [(i, i + 7) for i in range(n - 7)])
        return ts, prs

    diagram = None
    if args.seed_json:
        fit = json.load(open(args.seed_json))
        X0 = M.pose_from_vec(fit["X_WC_vec"])
        f0, dt0, s0 = fit["focal_px_full"], fit["dt_s"], fit.get("score", 0.0)
        print(f"seed from {args.seed_json}: score {s0:+.4f} f={f0:.1f} dt={dt0:+.3f}")
    else:
        ts, prs = sched(args.n_coarse)
        h = int(round(args.coarse_size * vh / vw))
        obj_c = MotionObjective(args.video, t_log, q_log, t_clip0, ts, prs,
                                args.coarse_size, h, sigma=1.5)
        diagram = obj_c.r.diagram
        grid = dict(
            phi=np.radians(np.linspace(-30, 30, 5)),
            dist=np.linspace(2.2, 5.0, 8),
            height=np.linspace(0.7, 2.0, 6),
            target_z=np.linspace(0.5, 1.4, 4),
            focal=np.linspace(850, 1750, 7),
        )
        print("coarse grid over camera azimuth/distance/height/target/focal ...")
        s0, best = coarse_search(obj_c, grid)
        X0, f0, dt0 = best["X"], best["focal"], 0.0

    for size, n, sigma, fit_dt, iters in (
        (args.fine_size, args.n_fine, 2.0, True, 900),
        (args.final_size, args.n_fine, 2.5, True, 700),
    ):
        ts, prs = sched(n)
        h = int(round(size * vh / vw))
        print(f"refine at {size}x{h} with {n} frames ...")
        obj = MotionObjective(args.video, t_log, q_log, t_clip0, ts, prs, size, h,
                              sigma=sigma, renderer=M.Renderer(size, h, diagram))
        diagram = obj.r.diagram
        s0, X0, f0, dt0 = refine(obj, X0, f0, dt0, fit_dt=fit_dt, maxiter=iters)
        last_obj, last_ts = obj, ts

    if not args.no_edge_polish:
        print("edge polish (truncated chamfer, added to the motion term) ...")
        edge = EdgeTerm(args.video, t_log, q_log, t_clip0, last_ts,
                        last_obj.W, last_obj.H, last_obj.r)
        comb = Combined(last_obj, edge, args.edge_weight)
        print(f"  before: motion {last_obj.score(X0, f0, dt0):+.4f}  "
              f"edge {edge.score(X0, f0, dt0):+.4f}")
        s0, X0, f0, dt0 = refine(comb, X0, f0, dt0, fit_dt=True, maxiter=700, restarts=2)
        print(f"  after:  motion {last_obj.score(X0, f0, dt0):+.4f}  "
              f"edge {edge.score(X0, f0, dt0):+.4f}")

    fit = dict(
        video=os.path.abspath(args.video),
        record=os.path.abspath(args.record),
        video_width=vw, video_height=vh,
        clip_start_wall=t_clip0,
        X_WC_vec=list(map(float, M.vec_from_pose(X0))),
        X_WC_translation=list(map(float, X0.translation())),
        X_WC_rotation=[list(map(float, r)) for r in X0.rotation().matrix()],
        focal_px_full=float(f0),
        principal_point=[vw / 2.0, vh / 2.0],
        dt_s=float(dt0),
        score=float(s0),
    )
    name = os.path.splitext(os.path.basename(args.video))[0]
    path = os.path.join(args.out, f"fit_{name}.json")
    json.dump(fit, open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    print(f"  camera at {np.round(X0.translation(), 4)} m in the base frame")
    print(f"  focal {f0:.1f} px on a {vw}-wide frame -> hfov {math.degrees(2*math.atan(vw/2/f0)):.1f} deg")
    print(f"  time offset {dt0:+.3f} s, final score {s0:+.4f}")


if __name__ == "__main__":
    main()
