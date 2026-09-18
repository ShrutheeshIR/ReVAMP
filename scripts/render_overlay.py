"""Stage E/F: draw calibrated overlays on the real footage.

--montage            12 spread frames with skeleton + trace -> scratch/
--frames T [T ...]   like --montage but at explicit video times
--segment T0 T1      render the annotated highlight cut -> out/highlight_4k.mp4

Overlay content (all 3D in the robot base frame, projected via camera.json):
  * executed marker-tip trace (fr3_tip FK), recent tail bright, history faded
  * current goal marker (from planning_queries.jsonl)
  * replan side panel: solve time, iterations; "goal blocked" state styled
    distinctly; obstacle rings flashed at each replan ("what the planner saw")
  * live constraint z-error readout from the log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from calibrate import FrameTrajectory  # noqa: E402

# Paper palette (preamble.tex), as BGR.
REVAMP_BLUE = (203, 160, 141)    # #8da0cb
HIGHLIGHT = (225, 105, 65)       # #4169E1 rblue
WARN_ORANGE = (0, 143, 209)      # #D18F00
WHITE = (245, 245, 245)

TRACE_FRAME = "fr3_tip"
TAIL_S = 10.0         # bright tail duration
MID_S = 40.0          # mid-fade band
HIST_S = 75.0         # older than this is dropped entirely
OBSTACLE_FLASH_S = 1.8
FONT = cv2.FONT_HERSHEY_DUPLEX

# Live-plan overlay (maze_expt_logs trajectories): the remaining portion of
# the active plan, ahead of the robot, dashed so it reads as intent rather
# than executed path. Green = "path ahead is clear"; orange already means
# obstacle/blocked and both blues belong to the executed trace.
PLAN_COLOR = (90, 205, 60)
PLAN_REFRESH_S = 0.1       # ~10 Hz update tick, per Tommy
PLAN_ALPHA = 0.75
PLAN_FLASH_S = 0.6         # full-bright right after a replan lands
PLAN_WIDTH = 6
PLAN_SEARCH_AHEAD = 250    # waypoints scanned per tick to advance progress
PLAN_FULL_ALPHA = 0.35     # faint, so the bright remaining-ahead dash reads
PLAN_FULL_WIDTH = 2


def dashed_polyline(img, uv, color, width, dash=34, gap=22):
    """polylines() but dashed, following the (dense) point chain in uv."""
    if len(uv) < 2:
        return
    d = np.linalg.norm(np.diff(uv.astype(float), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    on = (s % (dash + gap)) < dash
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 1:
                cv2.polylines(img, [uv[start:i + 1]], False, color, width,
                              cv2.LINE_AA)
            start = None
    if start is not None and len(uv) - start >= 2:
        cv2.polylines(img, [uv[start:]], False, color, width, cv2.LINE_AA)


class PlanPaths:
    """Marker-tip positions of every replayed planned trajectory.

    Plans are (n,7) joint-space waypoint arrays; the overlay wants the same
    physical point the executed trace uses (fr3_tip, the pen tip on the
    maze), so each waypoint goes through FK once. ~25k FK calls total,
    cached in scratch keyed by the trajectory directory's content.
    """

    def __init__(self, arm):
        import glob
        files = sorted(glob.glob(os.path.join(common.TRAJ_DIR, "*.npy")))
        sig = np.array([os.path.getsize(f) for f in files])
        cache = os.path.join(common.SCRATCH, "plan_tip_cache.npz")
        self.tip = {}
        if os.path.exists(cache):
            z = np.load(cache)
            if "sig" in z and np.array_equal(z["sig"], sig):
                self.tip = {int(k[1:]): z[k] for k in z.files if k != "sig"}
        if not self.tip:
            print(f"FK over {len(files)} planned trajectories...")
            for f in files:
                i = int(os.path.basename(f)[:3])
                q = np.load(f)
                self.tip[i] = arm.frame_positions(TRACE_FRAME, q)
            os.makedirs(common.SCRATCH, exist_ok=True)
            np.savez_compressed(cache, sig=sig,
                                **{f"r{i}": p for i, p in self.tip.items()})
            print(f"cached -> {cache}")


class Overlay:
    def __init__(self, plan=False):
        cam = common.read_json(common.camera_path())
        self.cam = cam
        self.K = np.array([[cam["f"], 0, cam["cx"]],
                           [0, cam["f"], cam["cy"]], [0, 0, 1]])
        self.dist = np.array([cam["k1"], cam["k2"], 0, 0])
        self.rvec = np.array(cam["rvec"])
        self.tvec = np.array(cam["tvec"])
        self.v0 = cam["video_start_epoch"]
        self.delta = cam["delta_s"]

        js = common.load_joint_states()
        self.t_log = js["t"]
        self.z_err_mm = js["z_error"] * 1000.0
        self.tip = FrameTrajectory(TRACE_FRAME)
        self.tip_plane_z = float(np.median(self.tip.p[:, 2]))
        self.queries = common.load_queries()
        self.q_t = np.array([q["t"] for q in self.queries])
        self.arm = common.ArmKinematics()
        self.qs = common.joint_matrix(js)
        self.plans = PlanPaths(self.arm) if plan else None
        # Sequential-render state for the live-plan overlay: the active plan
        # row, how far along it the robot has progressed, and the projected
        # remaining path drawn since the last 10 Hz tick.
        self._plan_state = {"row": None, "prog": 0, "tick": None, "uv": None,
                            "full_uv": None}

    # -- time mapping ------------------------------------------------------
    def log_time(self, t_video):
        return t_video + self.v0 - self.delta

    # -- projection --------------------------------------------------------
    def px(self, pts3):
        uv, _ = cv2.projectPoints(np.asarray(pts3, float).reshape(-1, 3),
                                  self.rvec, self.tvec, self.K, self.dist)
        return uv.reshape(-1, 2)

    # -- drawing pieces ----------------------------------------------------
    def draw_trace(self, img, t_video):
        te = self.log_time(t_video)
        t0, t1 = self.t_log[0], te
        if t1 <= t0:
            return
        ts = np.arange(max(t0, t1 - HIST_S), t1, 1.0 / 30)
        if len(ts) < 2:
            return
        p, _ = self.tip.at(ts)
        uv = self.px(p).astype(np.int32)
        age = te - ts
        # Age bands, oldest first so the fresh tail draws on top; each band
        # gets its own alpha so history recedes instead of accumulating.
        bands = [
            (age <= HIST_S) & (age > MID_S), REVAMP_BLUE, 5, 0.30,
            (age <= MID_S) & (age > TAIL_S), REVAMP_BLUE, 6, 0.55,
            (age <= TAIL_S), HIGHLIGHT, 9, 0.85,
        ]
        new = uv[age <= TAIL_S]
        for i in range(0, len(bands), 4):
            sel, color, width, alpha = bands[i:i + 4]
            seg = uv[sel]
            if len(seg) > 1:
                canvas = img.copy()
                cv2.polylines(canvas, [seg], False, color, width, cv2.LINE_AA)
                cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, dst=img)
        if len(new):
            cv2.circle(img, tuple(new[-1]), 14, HIGHLIGHT, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(new[-1]), 14, WHITE, 2, cv2.LINE_AA)

    def active_query(self, t_video):
        te = self.log_time(t_video)
        i = np.searchsorted(self.q_t, te) - 1
        return (self.queries[i], te - self.q_t[i]) if i >= 0 else (None, 1e9)

    def draw_goal(self, img, t_video):
        q, _ = self.active_query(t_video)
        if q is None:
            return
        g = np.array(q["goal_eef_pos"], float)
        g[2] = self.tip_plane_z
        uv = self.px(g)[0].astype(int)
        cv2.circle(img, tuple(uv), 22, WHITE, 3, cv2.LINE_AA)
        cv2.circle(img, tuple(uv), 8, WHITE, -1, cv2.LINE_AA)
        cv2.putText(img, "goal", (uv[0] + 30, uv[1] + 8), FONT, 1.4,
                    WHITE, 2, cv2.LINE_AA)

    def draw_obstacles(self, img, t_video):
        q, age = self.active_query(t_video)
        if q is None or age > OBSTACLE_FLASH_S:
            return
        alpha = 1.0 - age / OBSTACLE_FLASH_S
        canvas = img.copy()
        for s in q.get("spheres", []):
            c = np.array(s["position"], float)
            uv = self.px(c)[0].astype(int)
            edge = self.px(c + [0, s["radius"], 0])[0]
            r_px = int(max(np.linalg.norm(edge - uv), 8) * 1.6)
            cv2.circle(canvas, tuple(uv), r_px, WARN_ORANGE, 4, cv2.LINE_AA)
        for b in q.get("cuboids", []):
            c = np.array(b.get("center", b.get("position")), float)
            uv = self.px(c)[0].astype(int)
            cv2.circle(canvas, tuple(uv), 60, WARN_ORANGE, 4, cv2.LINE_AA)
        cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, dst=img)

    def draw_panel(self, img, t_video):
        q, age = self.active_query(t_video)
        if q is None:
            return
        lines = []
        if q["solved"]:
            ms = q["rrtc_nanoseconds"] / 1e6
            lines.append((f"replanned in {ms:.1f} ms", WHITE))
            if q["rrtc_iterations"] > 0:
                lines.append((f"{q['rrtc_iterations']:,} iterations · "
                              f"{q['num_waypoints']} waypoints",
                              (200, 200, 200)))
            else:
                lines.append((f"direct connection · "
                              f"{q['num_waypoints']} waypoints",
                              (200, 200, 200)))
        else:
            lines.append(("goal blocked - replanning...", WARN_ORANGE))
        te = self.log_time(t_video)
        i = np.searchsorted(self.t_log, te)
        if 0 < i < len(self.z_err_mm):
            lines.append((f"constraint error {abs(self.z_err_mm[i]):.2f} mm",
                          (200, 200, 200)))
        n_replans = int(np.searchsorted(self.q_t, te))
        lines.append((f"replan #{n_replans}", (160, 160, 160)))

        x, y = 90, 130
        # flash the panel briefly on a fresh replan
        if age < 0.4 and q["solved"]:
            cv2.rectangle(img, (x - 30, y - 70), (x + 1150, y + len(lines) * 78),
                          HIGHLIGHT, 6)
        for text, color in lines:
            cv2.putText(img, text, (x, y), FONT, 2.2, (0, 0, 0), 9, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), FONT, 2.2, color, 4, cv2.LINE_AA)
            y += 78

    def active_plan_row(self, te):
        """Latest query row at te whose plan the robot is executing: the
        most recent SOLVED row (during a goal-blocked cluster the robot
        keeps following the previous plan, which is what we show)."""
        i = int(np.searchsorted(self.q_t, te)) - 1
        while i >= 0 and i not in self.plans.tip:
            i -= 1
        return i if i >= 0 else None

    def draw_plan(self, img, t_video):
        te = self.log_time(t_video)
        row = self.active_plan_row(te)
        if row is None:
            return
        st = self._plan_state
        tick = int(t_video / PLAN_REFRESH_S)
        # Global relocalization on the first eval and on non-sequential time
        # (stills/montage); sequential renders advance a windowed search.
        jumped = st["tick"] is None or abs(tick - st["tick"]) > 3
        if row != st["row"] or tick != st["tick"] or st["uv"] is None:
            pts = self.plans.tip[row]
            if row != st["row"]:
                st["row"], st["prog"] = row, 0
                st["full_uv"] = (self.px(pts).astype(np.int32)
                                  if len(pts) >= 2 else None)
            cur, _ = self.tip.at(te)
            d = np.linalg.norm(pts[:, :2] - cur[:2], axis=1)
            if jumped:
                # Non-sequential eval (montage/stills): relocalize globally.
                st["prog"] = int(np.argmin(d))
            else:
                lo = st["prog"]
                hi = min(len(pts), lo + PLAN_SEARCH_AHEAD)
                st["prog"] = lo + int(np.argmin(d[lo:hi]))
            st["tick"] = tick
            remaining = pts[st["prog"]:]
            st["uv"] = (self.px(remaining).astype(np.int32)
                        if len(remaining) >= 2 else None)
        if st["full_uv"] is not None:
            canvas = img.copy()
            cv2.polylines(canvas, [st["full_uv"]], False, PLAN_COLOR,
                          PLAN_FULL_WIDTH, cv2.LINE_AA)
            cv2.addWeighted(canvas, PLAN_FULL_ALPHA, img,
                            1 - PLAN_FULL_ALPHA, 0, dst=img)
        if st["uv"] is None:
            return
        age = te - self.q_t[row]     # age of THIS plan, not the newest query
        alpha = 1.0 if age < PLAN_FLASH_S else PLAN_ALPHA
        canvas = img.copy()
        dashed_polyline(canvas, st["uv"], PLAN_COLOR, PLAN_WIDTH)
        cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, dst=img)

    def draw_skeleton(self, img, t_video):
        te = self.log_time(t_video)
        i = int(np.argmin(np.abs(self.t_log - te)))
        uv = self.px(self.arm.skeleton(self.qs[i])).astype(int)
        cv2.polylines(img, [uv], False, (0, 0, 255), 3, cv2.LINE_AA)
        for p in uv:
            cv2.circle(img, tuple(p), 9, (0, 255, 0), 2, cv2.LINE_AA)

    def annotate(self, img, t_video, skeleton=False, obstacle_rings=True):
        self.draw_trace(img, t_video)
        if self.plans is not None:
            self.draw_plan(img, t_video)
        self.draw_goal(img, t_video)
        if obstacle_rings:
            self.draw_obstacles(img, t_video)
        self.draw_panel(img, t_video)
        if skeleton:
            self.draw_skeleton(img, t_video)
        return img


# -- frame IO ---------------------------------------------------------------

def grab_bgr(t):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", common.VIDEO,
           "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    w, h = common.VIDEO_WH
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()


def montage(ov, times, out_path, skeleton=True):
    tiles = []
    for t in times:
        img = ov.annotate(grab_bgr(t), t, skeleton=skeleton)
        cv2.putText(img, f"t={t:.1f}s", (60, 2100), FONT, 3, (0, 255, 255),
                    6, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (960, 540)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    grid = np.vstack([r for r in rows if r.shape[1] == rows[0].shape[1]])
    cv2.imwrite(out_path, grid)
    print(out_path)


def render_segment(ov, t0, t1, out_path, fps=None, ghost=None,
                   ghost_alpha=0.45):
    """ghost: optional sim_ghost.GhostRenderer composited (with interpolated
    obstacle bubbles, ReVAMP-blue tint) under the annotations; ring flashes
    are suppressed since the bubbles already show the obstacles."""
    fps = fps or common.VIDEO_FPS
    w, h = common.VIDEO_WH
    dec = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t0), "-to", str(t1),
         "-i", common.VIDEO, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    n = 0
    nbytes = w * h * 3
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        t = t0 + n / fps
        if ghost is not None:
            img = ghost.composite(img, t, ghost_alpha, obstacles=True,
                                  tint=(203, 160, 141))
        ov.annotate(img, t, obstacle_rings=ghost is None)
        enc.stdin.write(img.tobytes())
        n += 1
        if n % 300 == 0:
            print(f"{n} frames ({t0 + n / fps:.1f}s)")
    dec.wait()
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        sys.exit("encoder failed")
    print(f"{n} frames -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--frames", type=float, nargs="+")
    ap.add_argument("--segment", type=float, nargs=2)
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--ghost", action="store_true",
                    help="composite the sim ghost + obstacle bubbles "
                         "under the annotations")
    ap.add_argument("--ghost-alpha", type=float, default=0.45)
    ap.add_argument("--plan", action="store_true",
                    help="overlay the live current plan (remaining path "
                         "ahead of the robot, ~10 Hz refresh)")
    args = ap.parse_args()

    ov = Overlay(plan=args.plan)
    os.makedirs(common.SCRATCH, exist_ok=True)
    if args.montage:
        times = list(np.linspace(12, 380, 12))
        montage(ov, times,
                os.path.join(common.SCRATCH, "verify_montage.jpg"),
                skeleton=not args.no_skeleton)
    if args.frames:
        for t in args.frames:
            img = ov.annotate(grab_bgr(t), t,
                              skeleton=not args.no_skeleton)
            p = os.path.join(common.SCRATCH, f"overlay_{t:.1f}.jpg")
            cv2.imwrite(p, cv2.resize(img, (1920, 1080)))
            print(p)
    if args.segment:
        ghost = None
        if args.ghost:
            from sim_ghost import GhostRenderer
            ghost = GhostRenderer(with_obstacle_slots=9)
        name = ("highlight" + ("_plan" if args.plan else "")
                + ("_ghost" if args.ghost else "") + "_4k.mp4")
        render_segment(ov, args.segment[0], args.segment[1],
                       os.path.join(common.REPO, "out", name),
                       ghost=ghost, ghost_alpha=args.ghost_alpha)


if __name__ == "__main__":
    main()
