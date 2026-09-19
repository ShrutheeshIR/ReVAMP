"""Shared paths, log loading, and Drake FK helpers for the revamp video pipeline.

All 3D quantities are in the robot base frame (fr3_link0 == world in the URDF
scene); logged planner states live in the same frame. Times are Unix epoch
seconds unless suffixed _rel (relative to VIDEO start, see sync.json).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(REPO, "shru_revamp_vid_hanlanphone-001.MOV")
LOG_DIR = os.path.join(REPO, "final_vid_0917")
JOINT_CSV = os.path.join(LOG_DIR, "joint_states.csv")
QUERIES_JSONL = os.path.join(LOG_DIR, "planning_queries.jsonl")
CALIB_DIR = os.path.join(REPO, "calib")
SCRATCH = os.path.join(REPO, "scratch")
# Offline replay of the planner (see maze_expt_logs/planned_traj_readme):
# trajectories/NNN.npy is the (n,7) joint-space plan for query row NNN.
TRAJ_DIR = os.path.join(REPO, "maze_expt_logs", "trajectories")
# RRT-connect explored tree per query row: trees/NNN.npz has "nodes"
# ((n,8) task-space pose [x,y,z,qx,qy,qz,qw,psi], first 3 cols already a
# usable 3D point -- no FK needed), "parents" ((n,) int64, edge to i's
# parent), "owner" ((n,) uint8, which of the two RRT-connect trees).
TREE_DIR = os.path.join(REPO, "planned_trajectory_info", "trees")
FR3_MARKER_URDF = os.path.join(
    REPO, "models", "fr3_marker", "fr3_expo_spherized.urdf")
URDF = FR3_MARKER_URDF
# GL-renderable variant (visual meshes with normals; see make_gl_meshes.py).
# Same kinematics; use for Drake render engines, fall back to URDF for FK.
URDF_GL_PATH = URDF.replace(".urdf", "_gl.urdf")


def urdf_for_rendering():
    return URDF_GL_PATH if os.path.exists(URDF_GL_PATH) else URDF

VIDEO_WH = (3840, 2160)
VIDEO_FPS = 4641200 / 77389  # avg_frame_rate from ffprobe (~59.97)
VIDEO_DURATION_S = 386.943333
# Container creation_time is when the file was CLOSED (end of recording).
VIDEO_CREATION_UTC = "2026-09-17T22:42:43.000000Z"


def load_joint_states():
    """joint_states.csv -> dict of numpy arrays keyed by column name."""
    with open(JOINT_CSV) as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def joint_matrix(js):
    """(N,7) joint angles from a load_joint_states() dict."""
    return np.stack([js[f"q{i}"] for i in range(1, 8)], axis=1)


def eef_matrix(js):
    """(N,3) logged (VAMP-chain) end-effector positions."""
    return np.stack([js["eef_x"], js["eef_y"], js["eef_z"]], axis=1)


def load_queries():
    with open(QUERIES_JSONL) as f:
        return [json.loads(line) for line in f]


def trajectory_path(i):
    return os.path.join(TRAJ_DIR, f"{i:03d}.npy")


def load_trajectory(i):
    """(n,7) planned joint-space waypoints for query row i, or None if the
    row didn't solve (only solved rows have a replayed trajectory file)."""
    p = trajectory_path(i)
    return np.load(p) if os.path.exists(p) else None


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    """Atomic write: checkpointing optimizers write while renderers read."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def camera_path():
    """Best available camera calibration, most refined first."""
    for name in ("camera.json", "camera_deep.json", "camera_sil.json"):
        p = os.path.join(CALIB_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no camera calibration in calib/")


# ---------------------------------------------------------------- Drake FK --

class ArmKinematics:
    """FK over the fr3_marker URDF. Frame poses in the base (world) frame."""

    # Chain of interest for skeleton overlays, base to tip.
    SKELETON_FRAMES = [
        "fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4",
        "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8", "fr3_hand",
        "fr3_marker_holder", "fr3_marker", "fr3_tip",
    ]
    # Candidate tool frames when identifying the VAMP eef point.
    CANDIDATE_FRAMES = [
        "fr3_link8", "fr3_hand", "fr3_hand_tcp", "fr3_marker_holder",
        "fr3_marker", "fr3_tip",
    ]

    def __init__(self):
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant

        self.plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self.plant)
        parser.package_map().Add(
            "fr3_marker", os.path.dirname(URDF))
        (model,) = parser.AddModels(URDF)
        base = self.plant.GetFrameByName("fr3_link0", model)
        self.plant.WeldFrames(self.plant.world_frame(), base)
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()
        self.model = model

    def frame_pose(self, frame_name, q):
        """4x4 pose of `frame_name` in world at joint config q (7,)."""
        self.plant.SetPositions(self.context, self.model, np.asarray(q))
        frame = self.plant.GetFrameByName(frame_name, self.model)
        X = self.plant.CalcRelativeTransform(
            self.context, self.plant.world_frame(), frame)
        return X.GetAsMatrix4()

    def frame_positions(self, frame_name, qs):
        """(N,3) world positions of a frame over an (N,7) config array."""
        return np.array([self.frame_pose(frame_name, q)[:3, 3] for q in qs])

    def skeleton(self, q):
        """(len(SKELETON_FRAMES),3) world positions of the skeleton chain."""
        return np.array(
            [self.frame_pose(f, q)[:3, 3] for f in self.SKELETON_FRAMES])
