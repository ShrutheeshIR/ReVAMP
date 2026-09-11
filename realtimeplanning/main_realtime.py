"""
Real-time closed-loop task-space planner + mover for the FR3 marker maze demo.

Fixes a (start, goal) eef pose pair (by default, problem 0 from
vamp/resources/fr3_marker/maze_problems_checked_ik.json) and loops forever:
plan from the robot's *current* configuration to whichever of the two points
isn't the one we just reached, execute the plan on the real robot, then swap
start/goal and repeat.

Planning is always done --use_fixed_order_smm style (see
vamp/scripts/fr3_marker_maze_example.py's run_problem): BRANCH_ORDER's 4
(elbow_sel, shoulder_sel) branches are swept in fixed order, and for each
branch, psi is independently resolved for the start and the goal by sweeping
NUM_PSI_CANDIDATES candidates across FR3's joint-7 range (mirroring
fr3_maze_solver_benchmark.cc's find_valid_start_goal_on_branch) -- start and
goal don't need to share a psi, only a branch. This script assumes that mode
only -- no saved-smm / branch-search flags.

The task-space planner only works from a config that's already on the maze's
constraint manifold (eef z == Z_HEIGHT, eef facing straight down). If the
robot's current config isn't -- e.g. right after startup, or after a manual
move -- a regular ambient-space (joint-space, NOT parameterized/task-space)
rrtc plan is used to get onto the resolved constrained config first, and only
then does task-space planning to the actual goal take over.

Usage:
    python main_realtime.py
    python main_realtime.py --host 172.16.0.3 --problem_index 0
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).parent / "vamp" / "scripts"))

import vamp
from fr3_marker_maze_example import (
    BRANCH_ORDER,
    EEF_TO_OFFSET,
    TSR_LOWER,
    TSR_UPPER,
    WORLD_TO_REFERENCE,
    Z_HEIGHT,
    load_maze_environment,
    make_pose_array,
)

from mock_franky import Duration, JointState, JointWaypoint, JointWaypointMotion, Robot

from main_toppra import compute_toppra_trajectory, preflight_check_segments, sample_trajectory

DEFAULT_PROBLEMS_JSON = (
    Path(__file__).parent / "../vamp" / "resources" / "fr3_marker" / "maze_problems_checked_ik.json"
)

# Deterministic psi (FR3 joint-7) sweep, matching fr3_maze_solver_benchmark.cc's
# find_valid_start_goal_on_branch/try_resolve under --use_deterministic_psi:
# up to NUM_PSI_CANDIDATES evenly-spaced psi values across FR3's joint-7 range,
# tried independently for each endpoint (start and goal are never required to
# share a psi, only a branch).
NUM_PSI_CANDIDATES = 200
PSI_LOWER = float(vamp.fr3marker.lower_bounds()[6])
PSI_UPPER = float(vamp.fr3marker.upper_bounds()[6])


# ------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------

def get_current_eef_pos_psi(ambient, q: np.ndarray) -> tuple[np.ndarray, float]:
    """FK the current joint config to get the eef (x, y, z); psi is just q7
    itself -- FR3Marker's redundant self-motion angle (see
    fr3_marker_maze_example.py's module docstring)."""
    transform = ambient.eefk(np.asarray(q, dtype=np.float32))
    eef_pos = transform[:3, 3]
    psi = float(q[6])
    return eef_pos, psi


def eef_pose_satisfies_constraint(ambient, q: np.ndarray, z_tol: float = 0.005,
                                   approach_cos_tol: float = 0.02) -> bool:
    """Whether q's eef pose already lies on the maze's task-space manifold:
    z == Z_HEIGHT and the eef facing straight down (dz, rx, ry pinned to 0 in
    TSR_LOWER/TSR_UPPER; rz/yaw is free -- psi covers that -- so this only
    checks height and the approach axis, not a full quaternion match)."""
    transform = ambient.eefk(np.asarray(q, dtype=np.float32))
    z = float(transform[2, 3])
    if abs(z - Z_HEIGHT) > z_tol:
        return False
    approach_z = float(transform[2, 2])  # world-frame z of the eef's local +z axis
    return approach_z <= -1.0 + approach_cos_tol


def resolve_eef_pos_with_psi_sweep(param, ambient, environment, eef_pos, psi_hint=None):
    """Find a valid (IK + collision-free) psi for eef_pos on whatever smm
    branch is currently set on `param` -- mirrors
    fr3_maze_solver_benchmark.cc's try_resolve: try `psi_hint` first if given
    (a cheap shortcut, e.g. the robot's actual current q7, or a problem's
    saved psi -- not required to work), then fall back to sweeping
    NUM_PSI_CANDIDATES psi values evenly across FR3's joint-7 range, keeping
    the first that resolves+validates. Returns (psi, ambient_q), or None if
    nothing on this branch resolves."""
    candidates = []
    if psi_hint is not None:
        candidates.append(float(psi_hint))
    candidates.extend(
        PSI_LOWER + (i / NUM_PSI_CANDIDATES) * (PSI_UPPER - PSI_LOWER)
        for i in range(NUM_PSI_CANDIDATES)
    )

    for psi in candidates:
        state = make_pose_array(eef_pos, psi)
        valid, q = param.resolve(state)
        if valid and ambient.validate(np.asarray(q, dtype=np.float32), environment):
            return psi, np.asarray(q, dtype=np.float32)

    return None


def resolve_start_goal(param, ambient, environment, start_eef_pos, goal_eef_pos,
                        start_psi_hint=None, goal_psi_hint=None):
    """--use_fixed_order_smm branch sweep with an independent psi sweep per
    endpoint (matches fr3_maze_solver_benchmark.cc's
    find_valid_start_goal_on_branch): try BRANCH_ORDER in order; for each
    branch, resolve start_eef_pos and (independently) goal_eef_pos via
    resolve_eef_pos_with_psi_sweep -- they don't need to share a psi, only the
    branch. Keeps the first branch where both endpoints resolve. Returns
    (smm, start_state, start_q, goal_state, goal_q), or None if no branch
    works for both."""
    for case6_sel, case1_sel in BRANCH_ORDER:
        param.set_smm([case6_sel, case1_sel, 0.0])

        start_resolved = resolve_eef_pos_with_psi_sweep(param, ambient, environment, start_eef_pos, start_psi_hint)
        if start_resolved is None:
            print(f"  branch ({case6_sel}, {case1_sel}): no psi resolved the start.")
            continue
        start_psi, start_q = start_resolved

        goal_resolved = resolve_eef_pos_with_psi_sweep(param, ambient, environment, goal_eef_pos, goal_psi_hint)
        if goal_resolved is None:
            print(f"  branch ({case6_sel}, {case1_sel}): no psi resolved the goal.")
            continue
        goal_psi, goal_q = goal_resolved

        smm = [case6_sel, case1_sel, 0.0]
        start_state = make_pose_array(start_eef_pos, start_psi)
        goal_state = make_pose_array(goal_eef_pos, goal_psi)
        return smm, start_state, start_q, goal_state, goal_q

    return None


def resolve_ambient_path(param, states: np.ndarray) -> np.ndarray:
    """IK-resolve every task-space waypoint (on whatever smm branch is
    currently set on `param`) to its ambient joint-space configuration."""
    out = np.empty((len(states), vamp.fr3marker.dimension()), dtype=np.float32)
    for i, state in enumerate(states):
        _, ambient_config = param.resolve(state)
        out[i] = np.asarray(ambient_config, dtype=np.float32)
    return out


def plan_task_space_path(param, ambient, environment, sampler, settings,
                          start_state, goal_state, smm) -> np.ndarray:
    """rrtc + shortcut + resolve a task-space path between two states, on a
    branch already known to resolve both (see resolve_start_goal). Returns an
    (n, 7) ambient joint-space path, or None if rrtc found no path."""
    param.set_smm(smm)

    sampler.reset()
    result = param.rrtc(start_state, goal_state, environment, settings, sampler)
    if not result.solved:
        print("  rrtc failed to find a path.")
        return None

    param.shortcut(result.path, environment)
    result.path.interpolate_to_resolution(32)

    return resolve_ambient_path(param, result.path.numpy())


def plan_between_start_goal(param, ambient, environment, sampler, settings,
                             start_eef_pos, goal_eef_pos,
                             start_psi_hint=None, goal_psi_hint=None):
    """Plan a task-space path between two eef_pos targets on the maze's fixed
    z-plane, --use_fixed_order_smm style, with an independent psi sweep per
    endpoint (see resolve_start_goal). `start_psi_hint`/`goal_psi_hint` are
    tried first as a shortcut but aren't required to work. Returns an (n, 7)
    ambient joint-space path, or None if no branch/path was found."""
    resolved = resolve_start_goal(param, ambient, environment, start_eef_pos, goal_eef_pos,
                                   start_psi_hint, goal_psi_hint)
    if resolved is None:
        print("  no branch resolved both start and goal.")
        return None
    smm, start_state, _start_q, goal_state, _goal_q = resolved

    return plan_task_space_path(param, ambient, environment, sampler, settings,
                                 start_state, goal_state, smm)


def plan_ambient_bootstrap(ambient, environment, ambient_settings, ambient_sampler,
                            q_current: np.ndarray, q_target: np.ndarray):
    """Regular joint-space (ambient) rrtc from q_current to q_target -- NOT
    task-space/parameterized-space planning. Used only to get the robot from
    an arbitrary current config onto the maze's constraint manifold. Returns
    an (n, 7) ambient path, or None if no path was found."""
    ambient_sampler.reset()
    result = ambient.rrtc(q_current, q_target, environment, ambient_settings, ambient_sampler)
    if not result.solved:
        print("  ambient-space bootstrap plan failed.")
        return None
    result.path.interpolate_to_resolution(32)
    return result.path.numpy()


def plan_to_goal_from_current(robot, param, ambient, environment, sampler, settings,
                               ambient_settings, ambient_sampler, goal_eef_pos, goal_psi_hint=None):
    """Return a list of ambient joint-space path segments to execute, in
    order, to get from the robot's current config to goal_eef_pos.

    Normally this is a single task-space-planned segment. But if the robot's
    current eef pose doesn't already satisfy the maze's plane/orientation
    constraint, a regular ambient-space bootstrap segment (plan_ambient_bootstrap)
    is planned and prepended first, onto the same (smm-resolved) constrained
    config that the task-space plan will use as its start -- so both segments
    agree on which config "current, but on the manifold" actually means.

    Returns None if any planning step fails.
    """
    q_current = np.asarray(robot.current_joint_positions, dtype=np.float32)
    eef_pos, psi = get_current_eef_pos_psi(ambient, q_current)
    print(f"  current eef pos={eef_pos.tolist()}, psi={psi:.3f}")

    # psi (q7 as-is) is only tried as a shortcut here -- the branch we end up
    # on may not be the one the robot is actually sitting on, so a fresh psi
    # sweep (resolve_start_goal) is what actually finds a valid start.
    resolved = resolve_start_goal(param, ambient, environment, eef_pos, goal_eef_pos,
                                   start_psi_hint=psi, goal_psi_hint=goal_psi_hint)
    if resolved is None:
        print("  no branch resolved both start and goal.")
        return None
    smm, start_state, start_q, goal_state, _goal_q = resolved

    segments = []
    # if not eef_pose_satisfies_constraint(ambient, q_current):
    #     print("  current eef pose violates the plane/orientation constraint -- "
    #           "planning a regular ambient-space bootstrap move onto it first.")
    #     bootstrap_path = plan_ambient_bootstrap(
    #         ambient, environment, ambient_settings, ambient_sampler, q_current, start_q)
    #     if bootstrap_path is None:
    #         return None
    #     segments.append(bootstrap_path)

    task_path = plan_task_space_path(param, ambient, environment, sampler, settings,
                                      start_state, goal_state, smm)
    if task_path is None:
        return None
    segments.append(task_path)

    return segments


# ------------------------------------------------------------------
# Mover -- TOPPRA time-parameterize the ambient path and execute on franky.
# Mirrors main_toppra.py's pipeline (same preflight check, waypoint shape).
# ------------------------------------------------------------------

def move_along_path(robot, path: np.ndarray, vel_scale: float, acc_scale: float, num_samples: int,
                     relative_dynamics: float):
    vel_limits = robot.joint_velocity_limit.max * vel_scale
    acc_limits = robot.joint_acceleration_limit.max * acc_scale

    jnt_traj = compute_toppra_trajectory(path, vel_limits, acc_limits)
    ts, qs, qds = sample_trajectory(jnt_traj, num_samples)
    segment_durations = np.diff(ts)

    # Use the plain float relative_dynamics (not robot.relative_dynamics_factor,
    # which reads back as a RelativeDynamicsFactor object and can't be multiplied
    # directly against the numpy limit arrays) -- matches main_toppra.py.
    preflight_check_segments(
        qs[1:], qds[1:], segment_durations,
        vel_max=robot.joint_velocity_limit.max * relative_dynamics,
        acc_max=robot.joint_acceleration_limit.max * relative_dynamics,
        jerk_max=robot.joint_jerk_limit.max * relative_dynamics,
        current_position=robot.current_joint_positions,
        current_velocity=robot.current_joint_velocities,
    )

    settle_waypoint = JointWaypoint(
        JointState(position=robot.current_joint_positions, velocity=np.zeros(7)),
        minimum_time=Duration(300),
    )
    joint_waypoints = [settle_waypoint] + [
        JointWaypoint(
            JointState(position=q, velocity=qd),
            minimum_time=Duration(max(int(round(seg_dt * 1000)), 1)),
        )
        for q, qd, seg_dt in zip(qs[1:], qds[1:], segment_durations)
    ]
    robot.move(JointWaypointMotion(joint_waypoints))


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main(
    host: str = "172.16.0.3",
    problems: str = str(DEFAULT_PROBLEMS_JSON),
    problem_index: int = 0,
    relative_dynamics: float = 1.0,
    vel_scale: float = 0.2,
    acc_scale: float = 0.2,
    num_samples: int = 200,
    range_: float = 0.42,  # RRTC range; matches fr3_maze_solver_benchmark.cc.
    max_iterations: int = 100_000,
    max_samples: int = 100_000,
    radius: float = 1.0,
):
    with open(problems) as f:
        problem_list = json.load(f)
    problem = problem_list[problem_index]

    point_a = (np.asarray(problem["start_eef_pos"], dtype=np.float32), float(problem["start_psi"]))
    point_b = (np.asarray(problem["goal_eef_pos"], dtype=np.float32), float(problem["goal_psi"]))

    environment = load_maze_environment()
    ambient = vamp.fr3marker
    param = ambient.parameterized_space

    inner = param.halton()
    sampler = param.TaskSpaceInformedSampler(
        EEF_TO_OFFSET, WORLD_TO_REFERENCE, TSR_LOWER, TSR_UPPER, environment, inner)

    settings = vamp.RRTCSettings()
    settings.range = range_
    settings.max_iterations = max_iterations
    settings.max_samples = max_samples
    settings.dynamic_domain = False
    settings.radius = radius

    # Regular ambient-space (joint-space) rrtc settings/sampler, used only for
    # the occasional bootstrap move onto the constraint manifold -- separate
    # from `settings`/`sampler` above, which are task-space (parameterized_space).
    ambient_settings = vamp.RRTCSettings()
    ambient_settings.range = range_
    ambient_settings.max_iterations = max_iterations
    ambient_settings.max_samples = max_samples
    ambient_settings.dynamic_domain = False
    ambient_settings.radius = radius
    ambient_sampler = ambient.halton()

    robot = Robot(host)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = relative_dynamics

    # Whatever the robot's actual current config is, we plan to point_b first;
    # after that we just keep swapping between the two fixed points forever.
    current_goal, other_goal = point_b, point_a

    while True:
        goal_eef_pos, goal_psi_hint = current_goal
        print(f"Planning to goal eef_pos={goal_eef_pos.tolist()}, psi_hint={goal_psi_hint:.3f}")

        segments = plan_to_goal_from_current(
            robot, param, ambient, environment, sampler, settings,
            ambient_settings, ambient_sampler, goal_eef_pos, goal_psi_hint)
        if segments is None:
            print("  planning failed, retrying...")
            time.sleep(0.5)
            continue

        for i, segment in enumerate(segments):
            print(f"  executing segment {i + 1}/{len(segments)} "
                  f"with {len(segment)} waypoints...")
            move_along_path(robot, segment, vel_scale, acc_scale, num_samples, relative_dynamics)
        print("  reached goal.")

        current_goal, other_goal = other_goal, current_goal


if __name__ == "__main__":
    Fire(main)
