"""Recover the camera <-> robot-base transform from a hardware clip plus the
execution record that was logged while it was filmed.

Why this exists
---------------
The earlier attempts to superimpose the model on the hardware media fitted a
camera against the *gripper alone*, with the gripper geometry itself unknown.
That is under-determined, and all three variants failed (a global edge chamfer
walked the camera to 7 m, silhouette IoU needs a segmentation that a white robot
in a white lab does not admit, and PnP from a handful of automatically-derived
points put the camera below the floor).

With an execution record in hand the problem is a different one.  The record logs
the full joint vector at 10 Hz, so the robot's shape in space is *known* at every
instant; the only unknowns are where the camera was, what its focal length was,
and how the video clock lines up with the record's wall clock.  That is ordinary
camera <-> robot extrinsic estimation, and it is well posed.

The three facts that make it work on this footage:

* The camera is **static**.  Measured by forward-backward Lucas-Kanade on
  background features: over the whole 125 s of ``20260828_142748.mp4`` the
  background moves a median of 0.9-1.0 px (p95 1.4 px).  So one global
  ``X_BC`` describes the entire clip.  (``20260828_140707.mp4`` is handheld --
  130 px -- and is not usable.)
* The **base never drives and the head never pans** in these records (wheel
  travel 0.005-0.02 rad, head 0.001-0.08 rad), so the base pose folds into the
  extrinsic and the fit happens in the base frame.  Nothing needs to know where
  the robot stood in the room.
* Segmentation is avoided entirely.  The objective compares *motion*: the image
  difference between two frames against the exclusive-or of the two rendered
  silhouettes.  Everything static -- wall, benches, and the robot's own base and
  torso -- cancels on both sides, so the score is driven by the arms sweeping
  through the frame, which is the part that is both unambiguous in the image and
  strongly sensitive to the camera pose.

Frames of reference
-------------------
``W`` is the plant world frame, which is the robot base frame here (the base
joints stay at zero).  ``C`` is the camera frame in Drake's convention: +x right,
+y down, +z forward along the optical axis.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/olorin/projects/PVAMP/rby1-constrained-planning/src")

from pydrake.geometry import (  # noqa: E402
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
)
from pydrake.math import RigidTransform, RotationMatrix  # noqa: E402
from pydrake.systems.sensors import CameraInfo  # noqa: E402

# ---------------------------------------------------------------------------
# The execution record -> q(t)
# ---------------------------------------------------------------------------

# Layout of the 24-vector the robot server logs, which is *not* the planner's
# q23 and *not* the plant's position vector.  Setting joints by name from these
# slices sidesteps both orderings.
Q24 = {
    "wheel": slice(0, 2),      # never moves in these records
    "torso": slice(2, 8),
    "right": slice(8, 15),
    "left": slice(15, 22),
    "head": slice(22, 24),
}

# The 24-vector carries no gripper DOF -- the jaws are commanded on their own
# channel and logged as ``left_gripper`` / ``right_gripper``, a single aperture
# in metres per hand (0.1 open, 0.001 closed).  ``load_states`` appends the two
# so the state is self-contained, and ``set_plant_from_q24`` drives the prismatic
# finger joints from them.  Leaving them out is not harmless: the plant's default
# is a closed jaw, so an open gripper renders ~50 mm off on each finger, which is
# most of the finger.
GRIPPER = {"left_gripper": 24, "right_gripper": 25}


def load_states(rec):
    """Concatenate every logged state in an execution record into (t, q24).

    The premove and the eight steps each carry their own log; together they are
    a single 10 Hz stream over the whole run.  Returns wall-clock seconds and an
    (N, 24) array, sorted and de-duplicated.
    """
    ts, qs = [], []
    blocks = [rec.get("premove")] + list(rec.get("steps", []))
    for blk in blocks:
        if not blk:
            continue
        log = blk.get("logs") if "logs" in blk else blk
        if not log:
            continue
        t = log.get("timestamp")
        q = log.get("joint_position")
        if t is None or q is None or len(t) == 0:
            continue
        t = np.asarray(t, float)
        q = np.asarray(q, float)
        n = min(len(t), len(q))
        grip = np.zeros((n, 2))
        for name, col in GRIPPER.items():
            g = log.get(name)
            if g is None:
                continue
            g = np.asarray(g, float).ravel()
            if len(g) >= n:
                grip[:, col - 24] = g[:n]
            elif len(g):
                grip[:, col - 24] = g[-1]
        ts.append(t[:n])
        qs.append(np.hstack([q[:n], grip]))
    t = np.concatenate(ts)
    q = np.concatenate(qs, axis=0)
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    return t[keep], q[keep]


def q_at(t_log, q_log, when):
    """Linear interpolation of the logged configuration, clamped at the ends."""
    if when <= t_log[0]:
        return q_log[0].copy()
    if when >= t_log[-1]:
        return q_log[-1].copy()
    i = bisect.bisect_right(t_log, when) - 1
    a = (when - t_log[i]) / (t_log[i + 1] - t_log[i])
    return (1.0 - a) * q_log[i] + a * q_log[i + 1]


def set_plant_from_q24(plant, pctx, q):
    """Drive the plant from a logged state, by joint name.

    Accepts the raw 24-vector or the 26-vector ``load_states`` returns.  Setting
    joints by name sidesteps both the planner's q23 ordering and the plant's own
    position ordering, which disagree about whether the left or the right arm
    comes first.
    """
    def setj(name, val):
        plant.GetJointByName(name).set_angle(pctx, float(val))

    for i in range(6):
        setj(f"torso_{i}", q[2 + i])
    for i in range(7):
        setj(f"right_arm_{i}", q[8 + i])
        setj(f"left_arm_{i}", q[15 + i])
    for i in range(2):
        setj(f"head_{i}", q[22 + i])
    if len(q) >= 26:
        # One aperture per hand; the two prismatic joints open symmetrically
        # about the closing plane, so each finger travels half of it.  The
        # joints are signed opposite one another (finger_1 runs on -x, finger_2
        # on +x through its yaw of pi).
        for side, col in (("left_gripper", 24), ("right_gripper", 25)):
            inst = plant.GetModelInstanceByName(side)
            half = 0.5 * float(q[col])
            plant.GetJointByName("gripper_finger_1", inst).set_translation(pctx, -half)
            plant.GetJointByName("gripper_finger_2", inst).set_translation(pctx, half)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class Renderer:
    """Silhouettes of the robot at a logged configuration, from an arbitrary camera.

    Depth is used rather than colour or label: the model in
    ``MakeRby1Diagram`` contains nothing but the robot, so "the depth return is
    finite" *is* the silhouette, with no segmentation and no material setup.
    """

    def __init__(self, width, height, diagram=None):
        from rby1_opt_ik import MakeRby1Diagram

        self.diagram = diagram if diagram is not None else MakeRby1Diagram()
        self.plant = self.diagram.plant()
        self.sg = self.diagram.scene_graph()
        if not self.sg.HasRenderer("fit"):
            self.sg.AddRenderer("fit", MakeRenderEngineVtk(RenderEngineVtkParams()))
        self.ctx = self.diagram.CreateDefaultContext()
        self.pctx = self.plant.GetMyContextFromRoot(self.ctx)
        self.sgctx = self.sg.GetMyContextFromRoot(self.ctx)
        self.width = width
        self.height = height
        self.z_far = 20.0

    def set_config(self, q24):
        set_plant_from_q24(self.plant, self.pctx, q24)

    def camera(self, fx, fy=None, cx=None, cy=None):
        fy = fx if fy is None else fy
        cx = self.width / 2.0 if cx is None else cx
        cy = self.height / 2.0 if cy is None else cy
        core = RenderCameraCore(
            "fit",
            CameraInfo(self.width, self.height, fx, fy, cx, cy),
            ClippingRange(0.05, self.z_far),
            RigidTransform(),
        )
        return DepthRenderCamera(core, DepthRange(0.05, self.z_far - 0.1))

    def mask(self, X_WC, cam):
        qo = self.sg.get_query_output_port().Eval(self.sgctx)
        d = np.asarray(qo.RenderDepthImage(cam, self.sg.world_frame_id(), X_WC).data)
        d = d.reshape(self.height, self.width)
        return np.isfinite(d) & (d > 0.0) & (d < self.z_far - 0.2)

    def depth(self, X_WC, cam):
        qo = self.sg.get_query_output_port().Eval(self.sgctx)
        d = np.asarray(qo.RenderDepthImage(cam, self.sg.world_frame_id(), X_WC).data)
        return d.reshape(self.height, self.width)


# ---------------------------------------------------------------------------
# Camera pose parameterisation
# ---------------------------------------------------------------------------


def look_at(eye, target, roll=0.0):
    """Drake-convention camera pose (+x right, +y down, +z forward) looking at a point."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world)
    n = np.linalg.norm(right)
    if n < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= n
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd])
    if abs(roll) > 0:
        c, s = math.cos(roll), math.sin(roll)
        R = R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return RigidTransform(RotationMatrix(R), eye)


def pose_from_vec(v):
    """(tx, ty, tz, rx, ry, rz) -> X_WC, with the rotation as a rotation vector."""
    t = np.asarray(v[:3], float)
    r = np.asarray(v[3:6], float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        R = np.eye(3)
    else:
        k = r / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return RigidTransform(RotationMatrix(R), t)


def vec_from_pose(X):
    R = X.rotation().matrix()
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if ang < 1e-9:
        r = np.zeros(3)
    else:
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        r = ax / (2.0 * math.sin(ang)) * ang
    return np.concatenate([X.translation(), r])


# ---------------------------------------------------------------------------
# Video access
# ---------------------------------------------------------------------------


def probe(path):
    """(width, height, duration_s, creation_time_epoch or None)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:format_tags=creation_time:stream=width,height",
         "-select_streams", "v:0", "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True, check=True).stdout
    d = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    ct = d.get("TAG:creation_time")
    epoch = None
    if ct:
        epoch = dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp()
    return int(d["width"]), int(d["height"]), float(d["duration"]), epoch


def clip_start_wall(path):
    """Wall-clock epoch of the clip's first frame.

    Samsung writes ``creation_time`` at the *end* of the recording -- the tag on
    ``20260828_142748.mp4`` reads 14:29:54 local while the filename, which is the
    start, reads 14:27:48.  So the start is the tag minus the duration.  This is
    only the initial guess; the fit owns the residual offset.
    """
    _, _, dur, epoch = probe(path)
    if epoch is None:
        raise RuntimeError(f"{path} carries no creation_time")
    return epoch - dur


def grab(path, times, width=None, height=None, gray=True):
    """Decode frames at the given clip times (seconds), in one pass per frame.

    Uses an input seek followed by a short output seek so the decode lands on
    the intended frame rather than the preceding keyframe.
    """
    import cv2

    out = []
    for t in times:
        pre = max(0.0, t - 2.0)
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{pre:.6f}", "-i", path,
               "-ss", f"{t - pre:.6f}", "-frames:v", "1"]
        if width:
            cmd += ["-vf", f"scale={width}:{height}"]
        cmd += ["-f", "image2pipe", "-vcodec", "png", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        img = cv2.imdecode(np.frombuffer(raw, np.uint8),
                           cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to decode {path} at t={t}")
        out.append(img)
    return out
