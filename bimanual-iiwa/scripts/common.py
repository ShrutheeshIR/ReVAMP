"""Shared scene loading, TOPPRA retiming, and constraint-error helpers for
the bimanual-IIWA sim renders.

Uses rby1-constrained-planning's own model files (old_shelves.dmd.yaml)
via its package.xml -- this is a sibling-repo dependency, not a copy, so
a change there (e.g. table/shelf position) is picked up automatically.
"""
from __future__ import annotations

import json
import os

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
# Local copy of old_shelves.dmd.yaml (not iiwa_bimanual_table_only.dmd.yaml
# -- T/B/M are the shelf's Top/Bottom/Middle levels, so the shelf has to be
# in frame), modified to point the table and shelf at LOCAL textured meshes
# (models/table_metal/, models/shelf_wood/) instead of rby1-constrained-
# planning's own old_table/old_shelves -- see models/scene.dmd.yaml and
# bimanual-iiwa/README.md's "Textures" section for why.
DIRECTIVES = os.path.join(REPO, "models", "scene.dmd.yaml")
TRAJ_DIR = os.path.join(REPO, "trajectories")
SCRATCH = os.path.join(REPO, "scratch")
OUT_DIR = os.path.join(REPO, "out")

METHODS = ["DualFollower", "LeaderFollower"]
# Was all 6 shelf-to-shelf legs; per Tommy, the final cut only uses these 2
# (T->B, B->M) -- fewer, slower clips read better than six quick ones. The
# other 4 segment JSONs/trajectories are untouched in case they're wanted
# again later, just not rendered by --all anymore.
SEGMENTS = ["T->B", "B->M"]


def traj_path(method, segment):
    fname = f"{method.lower()}_{segment.replace('->', '_to_')}.json"
    return os.path.join(TRAJ_DIR, fname)


def load_traj_json(method, segment):
    with open(traj_path(method, segment)) as f:
        d = json.load(f)
    assert d["method"] == method and d["segment"] == segment, (
        f"{traj_path(method, segment)}: file says "
        f"{d['method']!r}/{d['segment']!r}, expected {method!r}/{segment!r}")
    return d


def build_scene():
    """(plant, scene_graph, diagram) for the table-only bimanual scene. 14
    positions: iiwa_left's 7 joints then iiwa_right's 7 (verified against
    the .dmd.yaml directive order -- do not assume, re-check if the model
    file changes).

    Deliberately does NOT create a Context: a renderer added to
    scene_graph after a Context already exists is invisible to that
    Context ("No renderer exists with name: ..." even though HasRenderer
    is true on the system) -- register any renderer first, then call
    diagram.CreateDefaultContext()."""
    from pydrake.multibody.parsing import (
        Parser, LoadModelDirectives, ProcessModelDirectives)
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(RBY1_REPO, "package.xml"))
    parser.package_map().AddPackageXml(os.path.join(REPO, "package.xml"))
    directives = LoadModelDirectives(DIRECTIVES)
    ProcessModelDirectives(directives, parser)
    plant.Finalize()

    # Shelf and table both now carry their own real photo textures (plywood
    # / brushed metal) baked into their local .mtl files, loaded via
    # DIRECTIVES -- no runtime color override needed any more.
    diagram = builder.Build()
    assert plant.num_positions() == 14, (
        f"expected 14 positions (7+7), got {plant.num_positions()} -- "
        "the .dmd.yaml changed shape, update this assumption")
    return plant, scene_graph, diagram


def retime_configs(configs, plant, speed=1.0):
    """A (n,14) waypoint path -> a Drake Trajectory, TOPPRA-retimed against
    the plant's joint velocity/acceleration limits.

    configs is treated purely as a PATH (arbitrary uniform parameter 0..n-1
    over CubicWithContinuousSecondDerivatives) -- the actual per-waypoint
    dt/times in the source JSON is discarded here on purpose, per Tommy:
    the planner never timed these, so 0.01 was a placeholder, not a real
    velocity profile. Mirrors generate_iiwa_meshcat.lift_and_retime's
    relaxation-escalation loop, minus the analytic-IK lifting step (our
    configs are already 14-D, nothing to lift)."""
    from pydrake.trajectories import PiecewisePolynomial, PathParameterizedTrajectory
    from pydrake.multibody.optimization import Toppra, CalcGridPointsOptions

    configs = np.asarray(configs, dtype=float)
    n, dof = configs.shape
    assert dof == 14, f"expected 14-DOF configs, got {dof}"
    ts = np.arange(n, dtype=float)
    path = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        ts, configs.T, np.zeros(dof), np.zeros(dof))

    gridpoints = Toppra.CalcGridPoints(
        path, CalcGridPointsOptions(max_iter=2, min_points=200))
    time_traj = None
    for relaxation in (0.0, 1e-4, 1e-3, 1e-2, 5e-2):
        toppra = Toppra(path, plant, gridpoints)
        toppra.AddJointVelocityLimit(plant.GetVelocityLowerLimits(),
                                     plant.GetVelocityUpperLimits())
        toppra.AddJointAccelerationLimit(plant.GetAccelerationLowerLimits(),
                                         plant.GetAccelerationUpperLimits())
        if relaxation:
            toppra.set_constraint_relaxation(relaxation)
        time_traj = toppra.SolvePathParameterization()
        if time_traj is not None:
            if relaxation:
                print(f"    TOPPRA needed constraint_relaxation={relaxation:g}")
            break
    if time_traj is None:
        raise RuntimeError("TOPPRA failed at every relaxation level")
    if speed != 1.0:
        pts = np.linspace(time_traj.start_time(), time_traj.end_time(), 801)
        ss = np.array([[float(np.asarray(time_traj.value(t)).flatten()[0])
                        for t in pts]])
        time_traj = PiecewisePolynomial.CubicShapePreserving(pts / speed, ss)
    return PathParameterizedTrajectory(path, time_traj)


def gripper_frames(plant):
    Fl = plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg_left"))
    Fr = plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg_right"))
    return Fl, Fr


def gripper_transform(plant, plant_context, q):
    """X_LeftGripper_RightGripper at configuration q (14,)."""
    Fl, Fr = gripper_frames(plant)
    plant.SetPositions(plant_context, q)
    return plant.CalcRelativeTransform(plant_context, Fl, Fr)


def gripper_world_positions(plant, plant_context, q):
    """(p_left, p_right): world-frame translation (3,) of each gripper's
    "body" frame origin at configuration q."""
    Fl, Fr = gripper_frames(plant)
    plant.SetPositions(plant_context, q)
    W = plant.world_frame()
    pl = plant.CalcRelativeTransform(plant_context, W, Fl).translation()
    pr = plant.CalcRelativeTransform(plant_context, W, Fr).translation()
    return pl, pr


def constraint_error_mm(X_ref, X):
    """Scalar error (mm) between two gripper-to-gripper transforms: pure
    translation drift of the relative transform (rotation drift would need
    a second, differently-scaled number -- kept out of the single mm
    readout to match revamp's single-number z_error convention)."""
    return float(np.linalg.norm(X.translation() - X_ref.translation()) * 1000.0)
