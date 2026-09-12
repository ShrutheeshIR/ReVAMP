"""
Continuously-replanning variant of main_realtime.py for the FR3 marker maze demo.

Reuses main_realtime.py's planning helpers (branch/psi-sweep resolution, task-space
rrtc+shortcut, ambient bootstrap) unchanged, but replaces its "plan once, move
blocking, repeat" main loop with a real replanning loop:

  1. get_new_env() rebuilds the environment and reports whether it changed
     since the last check (for now it's a dummy -- it just reloads the same
     maze_cuboids.json every time -- but this is where a live perception
     update would plug in).
  2. plan_to_goal_from_current() plans (in task space) from wherever the
     robot is right now to the current goal.
  3. The plan is executed asynchronously (franky's Robot.move(..., asynchronous=True))
     so the main loop can keep polling the environment while the robot is moving.
     If get_new_env() ever reports a change, the in-flight motion is stopped
     and step 2 runs again from the new current position. If it reports no
     change, nothing is interrupted -- the robot just keeps going.
  4. Once the goal is actually reached (with no pending environment change),
     start/goal swap and the loop continues.

Usage:
    python main_realtime_continuous.py
    python main_realtime_continuous.py --host 172.16.0.3 --problem_index 0
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).parent / "vamp" / "scripts"))

import vamp
from fr3_marker_maze_example import MAZE_JSON, EEF_TO_OFFSET, TSR_LOWER, TSR_UPPER, WORLD_TO_REFERENCE

from mock_franky import Duration, JointState, JointWaypoint, JointWaypointMotion, Robot

from main_toppra import compute_toppra_trajectory, preflight_check_segments, sample_trajectory

from main_realtime import DEFAULT_PROBLEMS_JSON, plan_to_goal_from_current


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

def build_environment_from_cuboids(cuboids: list) -> vamp.Environment:
    """Matches fr3_marker_maze_example.load_maze_environment's cuboid transform
    exactly (push forward/up, pad dz, shared maze/mount frame with the iiwa rig)."""
    environment = vamp.Environment()
    for c in cuboids:
        position = [c["x"] + 0.285 * 2, c["y"], c["z"] + 0.0]
        orientation = [c.get("roll", 0.0), c.get("pitch", 0.0), c.get("yaw", 0.0)]
        half_extents = [(c["dx"] + 0.0) / 2.0, (c["dy"] + 0.0) / 2.0, (c["dz"] + 0.0) / 2.0]
        environment.add_cuboid(vamp.Cuboid(position, orientation, half_extents))
    return environment


def get_new_env(previous_cuboids=None):
    """Rebuild the environment and report whether it changed since the last
    call. For now this is a dummy: it just reloads the same
    vamp/resources/environments/maze_cuboids.json every time, so `changed`
    will be True on the very first call (previous_cuboids=None) and False
    every call after that -- this is where a live perception update (a new
    point cloud / detected obstacle set) would plug in later.

    Pass the `cuboids` this returns back in as `previous_cuboids` next call.
    Returns (environment, cuboids, changed). changed=False means: no need to
    replan, the current plan (if any) is still valid.
    """
    with open(MAZE_JSON) as f:
        cuboids = json.load(f)
    changed = cuboids != previous_cuboids
    environment = build_environment_from_cuboids(cuboids)
    return environment, cuboids, changed


# ------------------------------------------------------------------
# Mover -- same TOPPRA + franky pipeline as main_realtime.py's move_along_path,
# but with an `asynchronous` switch so the main loop below can keep polling
# get_new_env() while the robot is moving.
# ------------------------------------------------------------------

def move_along_path(robot, path: np.ndarray, vel_scale: float, acc_scale: float, num_samples: int,
                     relative_dynamics: float, asynchronous: bool = False):
    vel_limits = robot.joint_velocity_limit.max * vel_scale
    acc_limits = robot.joint_acceleration_limit.max * acc_scale

    jnt_traj = compute_toppra_trajectory(path, vel_limits, acc_limits)
    ts, qs, qds = sample_trajectory(jnt_traj, num_samples)
    segment_durations = np.diff(ts)

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
    robot.move(JointWaypointMotion(joint_waypoints), asynchronous=asynchronous)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main(
    host: str = "172.16.0.3",
    problems: str = str(DEFAULT_PROBLEMS_JSON),
    problem_index: int = 26,
    relative_dynamics: float = 1.0,
    vel_scale: float = 0.2,
    acc_scale: float = 0.2,
    num_samples: int = 200,
    range_: float = 0.42,  # RRTC range; matches fr3_maze_solver_benchmark.cc.
    max_iterations: int = 100_000,
    max_samples: int = 100_000,
    radius: float = 1.0,
    replan_poll_period: float = 0.5,  # how often (seconds) to check get_new_env while moving
):
    with open(problems) as f:
        problem_list = json.load(f)
    problem = problem_list[problem_index]

    point_a = (np.asarray(problem["start_eef_pos"], dtype=np.float32), float(problem["start_psi"]))
    point_b = (np.asarray(problem["goal_eef_pos"], dtype=np.float32), float(problem["goal_psi"]))

    ambient = vamp.fr3marker
    param = ambient.parameterized_space
    inner = param.halton()

    # TaskSpaceInformedSampler needs an environment to construct against;
    # get_new_env() is called again (and its environment used) at the top of
    # every planning iteration below, and again while a motion is executing.
    environment, cuboids, _ = get_new_env()
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
            environment, cuboids, _ = get_new_env(cuboids)
            continue

        # Any earlier segments (e.g. a bootstrap move) run to completion
        # (blocking) before the final, interruptible segment starts.
        for segment in segments[:-1]:
            print(f"  executing bootstrap segment with {len(segment)} waypoints...")
            move_along_path(robot, segment, vel_scale, acc_scale, num_samples, relative_dynamics)

        final_segment = segments[-1]
        print(f"  executing final segment with {len(final_segment)} waypoints (async)...")
        move_along_path(robot, final_segment, vel_scale, acc_scale, num_samples,
                         relative_dynamics, asynchronous=True)

        # Continuously replan: while this motion runs, keep checking the
        # environment every replan_poll_period seconds. get_new_env's bool
        # gates everything -- unchanged means the in-flight plan is still
        # valid and nothing is interrupted; changed means we stop and replan
        # from wherever the robot actually is right now.
        interrupted = False
        while True:
            finished = robot.join_motion(timeout=replan_poll_period)
            environment, cuboids, changed = get_new_env(cuboids)

            if changed:
                print("  environment changed mid-motion -- stopping to replan.")
                robot.stop()
                interrupted = True
                break
            if finished:
                print("  reached goal.")
                break

        if interrupted:
            continue  # replan toward the same current_goal from wherever we stopped

        current_goal, other_goal = other_goal, current_goal


if __name__ == "__main__":
    Fire(main)
