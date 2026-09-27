"""Shared model loading + execution-record playback helpers for the RBY1
humanoid sim render.

Unlike bimanual-iiwa (two pre-planned trajectories per segment, TOPPRA-retimed
from scratch), this is a single real hardware execution log played back at its
own logged rate -- there is no planning step to visualize and no obstacle
scene, just the one pick-lift-place run. See notes/ikfast_drake_model_mismatch.md
in rby1-constrained-planning before computing ANY error/constraint number from
these configs: Drake FK on an IKFast-produced config carries a spurious
~3e-7 m floor that is not real error, so this pipeline deliberately does not
caption a constraint-error number at all (there is also no second method to
compare against here, unlike bimanual-iiwa's DualFollower/LeaderFollower).
"""
from __future__ import annotations

import bisect
import os
import pickle

import numpy as np

# Sibling planner/model repo. Resolution order: $RBY1_REPO env var, then
# the first known checkout location that actually exists on this machine.
_RBY1_CANDIDATES = [
    "/home/olorin/projects/PVAMP/rby1-constrained-planning",
    os.path.expanduser("~/Documents/programming/work/rlg/"
                       "minimal-coordinates/ift/rby1-constrained-planning"),
]
RBY1_REPO = os.environ.get("RBY1_REPO") or next(
    (p for p in _RBY1_CANDIDATES if os.path.isdir(p)), _RBY1_CANDIDATES[0])
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS_DIR = os.path.join(REPO, "records")
OUT_DIR = os.path.join(REPO, "out")

RECORDS = {
    "point_00": "point_00_exec_20260828_142919.pkl",
    "point_01": "point_01_exec_20260828_143100.pkl",
}

# Layout of the 24-vector the robot server logs (src/robot_camera_fit.py in
# rby1-constrained-planning history, commit b9fcdbb -- since removed there,
# ported here verbatim). NOT the planner's q23 and NOT the plant's own
# position ordering; setting joints by NAME below sidesteps both.
Q24 = {
    "wheel": slice(0, 2),      # never moves in these records
    "torso": slice(2, 8),
    "right": slice(8, 15),
    "left": slice(15, 22),
    "head": slice(22, 24),
}
# No gripper DOF in the 24-vector -- the jaws are commanded on their own
# channel, one aperture in metres per hand (0.1 open, 0.001 closed).
GRIPPER = {"left_gripper": 24, "right_gripper": 25}


def load_record(name):
    with open(os.path.join(RECORDS_DIR, RECORDS[name]), "rb") as f:
        return pickle.load(f)


def load_states(rec):
    """Concatenate every logged state in an execution record into (t, q26).

    The premove and the eight steps each carry their own log; together they
    are a single ~10 Hz stream over the whole run.  Returns wall-clock
    seconds and an (N, 26) array (24-vector + 2 gripper apertures), sorted
    and de-duplicated.
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
        for gname, col in GRIPPER.items():
            g = log.get(gname)
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


def step_at(rec, when):
    """Name of the step ('premove', 'reach_approach', 'grasp', ...) active
    at wall-clock time `when`, for the caption."""
    steps = rec.get("steps", [])
    if steps and when < steps[0]["t_start"]:
        return "premove"
    for step in steps:
        if step["t_start"] <= when <= step["t_end"]:
            return step["name"]
    return steps[-1]["name"] if steps else "premove"


def q_at(t_log, q_log, when):
    """Linear interpolation of the logged configuration, clamped at ends."""
    if when <= t_log[0]:
        return q_log[0].copy()
    if when >= t_log[-1]:
        return q_log[-1].copy()
    i = bisect.bisect_right(t_log, when) - 1
    a = (when - t_log[i]) / (t_log[i + 1] - t_log[i])
    return (1.0 - a) * q_log[i] + a * q_log[i + 1]


def set_plant_from_q26(plant, pctx, q):
    """Drive the plant from one logged state (26-vector), by joint name."""
    def setj(name, val):
        plant.GetJointByName(name).set_angle(pctx, float(val))

    for i in range(6):
        setj(f"torso_{i}", q[2 + i])
    for i in range(7):
        setj(f"right_arm_{i}", q[8 + i])
        setj(f"left_arm_{i}", q[15 + i])
    for i in range(2):
        setj(f"head_{i}", q[22 + i])
    # One aperture per hand; the two prismatic joints open symmetrically
    # about the closing plane, so each finger travels half of it. Signed
    # opposite one another (finger_1 on -x, finger_2 on +x through its
    # yaw of pi).
    for side, col in (("left_gripper", 24), ("right_gripper", 25)):
        inst = plant.GetModelInstanceByName(side)
        half = 0.5 * float(q[col])
        plant.GetJointByName("gripper_finger_1", inst).set_translation(pctx, -half)
        plant.GetJointByName("gripper_finger_2", inst).set_translation(pctx, half)


def build_scene():
    """(plant, scene_graph, diagram) with the full RBY1 model plus a plain
    dark floor plane. No shelf/table/obstacles -- deliberately a bare scene,
    unlike bimanual-iiwa, per Tommy: this run has no obstacles and is a
    single plan, and the render needs to look visually distinct from
    bimanual-iiwa's shelf/table scene while keeping the same dark,
    maze-style aesthetic."""
    from pydrake.geometry import Box
    from pydrake.math import RigidTransform
    from pydrake.multibody.parsing import (
        Parser, LoadModelDirectives, ProcessModelDirectives)
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(RBY1_REPO, "package.xml"))
    directives = LoadModelDirectives(os.path.join(
        RBY1_REPO,
        "models/ruby/rby1_description_drake/"
        "add_rby1_sim_with_holonomic_base_actuators.dmd.yaml"))
    ProcessModelDirectives(directives, parser)

    # Plain dark floor, big enough that the orbiting camera never sees its
    # edge. diffuse_color on this convenience overload sets BOTH the
    # illustration and perception role directly (no baked texture, no
    # separate role dance needed -- unlike bimanual-iiwa's shelf/table,
    # which are pre-existing SDF geometries with their own roles already
    # assigned).
    plant.RegisterVisualGeometry(
        plant.world_body(), RigidTransform([0, 0, -0.005]),
        Box(20.0, 20.0, 0.01), "floor",
        np.array([0.05, 0.05, 0.055, 1.0]))

    plant.Finalize()

    diagram = builder.Build()
    return plant, scene_graph, diagram
