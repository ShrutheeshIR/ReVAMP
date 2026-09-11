import argparse
import numpy as np
from pathlib import Path

import ruckig
import toppra as ta
import toppra.algorithm as ta_algorithm
import toppra.constraint as ta_constraint

from mock_franky import (
    Duration,
    JointState,
    JointWaypoint,
    JointWaypointMotion,
    Robot,
)

# from main2 import load_txt

# FR3 velocity limits (per joint, rad/s), from
# PVAMP/cricket/resources/fr3_marker/fr3_expo_spherized.urdf. This is the
# marker/expo deployment's reduced envelope, not the raw factory FR3 spec.
FR3_MAX_JOINT_VELOCITY = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0])

# Acceleration isn't in that URDF (URDF only carries position + velocity
# limits). This is an unverified placeholder for --dry_run previews only;
# real runs query the robot's actual joint_acceleration_limit instead.
FR3_MAX_JOINT_ACCELERATION_PLACEHOLDER = np.array([10.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


def compute_toppra_trajectory(path: np.ndarray, vel_limits: np.ndarray, acc_limits: np.ndarray):
    """
    Fit a spline through the waypoints and compute a time-optimal
    parameterization of it under joint velocity/acceleration limits.
    """
    ss = np.linspace(0, 1, path.shape[0])
    spline_path = ta.SplineInterpolator(ss, path)

    pc_vel = ta_constraint.JointVelocityConstraint(vel_limits)
    pc_acc = ta_constraint.JointAccelerationConstraint(acc_limits)

    # Default gridpoint spacing can be too coarse for a dense waypoint path,
    # letting the continuous-time trajectory overshoot the limits between
    # checked points. Force at least 10x the waypoint count.
    gridpoints = np.linspace(0, 1, max(10 * path.shape[0], 1000))

    instance = ta_algorithm.TOPPRA(
        [pc_vel, pc_acc], spline_path, gridpoints=gridpoints, solver_wrapper="seidel"
    )
    jnt_traj = instance.compute_trajectory()
    if jnt_traj is None:
        raise RuntimeError("TOPPRA failed to find a feasible time parameterization for this path.")
    return jnt_traj


def sample_trajectory(jnt_traj, num_samples: int):
    ts = np.linspace(0, jnt_traj.duration, num_samples)
    qs = jnt_traj(ts)
    qds = jnt_traj(ts, 1)
    if not (np.isfinite(qs).all() and np.isfinite(qds).all()):
        bad = np.where(~np.isfinite(qs).all(axis=1) | ~np.isfinite(qds).all(axis=1))[0]
        raise RuntimeError(
            f"TOPPRA produced non-finite samples at time indices {bad.tolist()} "
            f"(t={ts[bad].tolist()}); refusing to send this to the robot."
        )
    return ts, qs, qds


def preflight_check_segments(
    qs: np.ndarray,
    qds: np.ndarray,
    segment_durations: np.ndarray,
    vel_max: np.ndarray,
    acc_max: np.ndarray,
    jerk_max: np.ndarray,
    current_position: np.ndarray,
    current_velocity: np.ndarray,
):
    """
    Replay every planned segment through the real Ruckig engine (the same
    library franky uses internally for JointWaypointMotion) using the
    robot's actual absolute joint limits and starting state. Raises a
    specific error naming the failing segment/DOF instead of letting the
    generic "Motion planner failed with error code -100" surface from
    inside franky's C++ layer.
    """
    current_position = current_position.tolist()
    current_velocity = current_velocity.tolist()
    current_acceleration = [0.0] * 7

    inst = ruckig.Ruckig(7, 0.001)
    for i, (q, qd, seg_dt) in enumerate(zip(qs, qds, segment_durations)):
        inp = ruckig.InputParameter(7)
        inp.current_position = current_position
        inp.current_velocity = current_velocity
        inp.current_acceleration = current_acceleration
        inp.target_position = q.tolist()
        inp.target_velocity = qd.tolist()
        inp.target_acceleration = [0.0] * 7
        inp.max_velocity = vel_max.tolist()
        inp.max_acceleration = acc_max.tolist()
        inp.max_jerk = jerk_max.tolist()
        inp.synchronization = ruckig.Synchronization.Time
        inp.minimum_duration = float(seg_dt)

        try:
            valid = inp.validate(False, True)
        except ruckig.RuckigError as e:
            raise RuntimeError(f"Segment {i} (waypoint index {i + 1}) is invalid: {e}\n{inp}") from e
        if not valid:
            raise RuntimeError(f"Segment {i} (waypoint index {i + 1}) failed Ruckig's input validation:\n{inp}")

        traj = ruckig.Trajectory(7)
        result = inst.calculate(inp, traj)
        if result not in (ruckig.Working, ruckig.Finished):
            raise RuntimeError(
                f"Segment {i} (waypoint index {i + 1}) could not be planned (result={result}):\n{inp}"
            )

        current_position, current_velocity, current_acceleration = traj.at_time(traj.duration)


def main():
    parser = argparse.ArgumentParser(
        description="Load TXT joint path, time-parameterize it with TOPPRA, and execute on Franka robot"
    )
    parser.add_argument("txt_path", type=Path, help="Path to TXT file")
    parser.add_argument(
        "--host",
        default="172.16.0.3",
        help="FCI IP of the robot",
    )
    parser.add_argument(
        "--relative_dynamics",
        type=float,
        default=1.0,
        help="Robot-level dynamics scaling on top of the TOPPRA limits. Keep at 1.0 so the robot "
        "isn't forced below the (already scaled-down) TOPPRA profile; use --vel_scale/--acc_scale "
        "to make the motion more conservative instead.",
    )
    parser.add_argument(
        "--vel_scale",
        type=float,
        default=0.2,
        help="Fraction of the FR3's max joint velocity to use as the TOPPRA velocity limit",
    )
    parser.add_argument(
        "--acc_scale",
        type=float,
        default=0.2,
        help="Fraction of the FR3's max joint acceleration to use as the TOPPRA acceleration limit",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples drawn from the TOPPRA trajectory to send to the robot as waypoints",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output .npz file to save the time-parameterized trajectory (t, q, qd)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute (and optionally save) the trajectory without connecting to or moving the robot",
    )

    args = parser.parse_args()

    if not args.txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {args.txt_path}")

    # -----------------------------
    # Load trajectory
    # -----------------------------
    path = load_txt(args.txt_path)[::1]

    if path.ndim != 2 or path.shape[1] != 7:
        raise ValueError(
            f"Expected joint path shape (N, 7), got {path.shape}"
        )

    print(f"Loaded trajectory with {path.shape[0]} waypoints.")

    # -----------------------------
    # Connect to robot (skipped for --dry_run) and get joint limits.
    # Connecting first lets us read the FR3's actual hardware
    # joint_velocity_limit/joint_acceleration_limit instead of guessing,
    # since acceleration/jerk limits aren't published anywhere in the repo.
    # -----------------------------
    robot = None
    if args.dry_run:
        print(
            "Dry run: using FR3_MAX_JOINT_VELOCITY (fr3_expo_spherized.urdf) and "
            "a placeholder acceleration limit, since no robot is connected."
        )
        vel_limits = FR3_MAX_JOINT_VELOCITY * args.vel_scale
        acc_limits = FR3_MAX_JOINT_ACCELERATION_PLACEHOLDER * args.acc_scale
    else:
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = args.relative_dynamics

        vel_limits = robot.joint_velocity_limit.max * args.vel_scale
        acc_limits = robot.joint_acceleration_limit.max * args.acc_scale
        print(f"Robot joint velocity limit (max): {robot.joint_velocity_limit.max}")
        print(f"Robot joint acceleration limit (max): {robot.joint_acceleration_limit.max}")
        print(f"Robot joint jerk limit (max): {robot.joint_jerk_limit.max}")

    # -----------------------------
    # TOPPRA time parameterization
    # -----------------------------
    print("Running TOPPRA time parameterization...")
    jnt_traj = compute_toppra_trajectory(path, vel_limits, acc_limits)
    print(f"TOPPRA trajectory duration: {jnt_traj.duration:.3f} s")

    ts, qs, qds = sample_trajectory(jnt_traj, args.num_samples)

    if args.out:
        np.savez(args.out, t=ts, q=qs, qd=qds)
        print(f"Time-parameterized trajectory saved to {args.out}")

    if args.dry_run:
        print("Dry run: skipping robot execution.")
        return

    # -----------------------------
    # Pre-flight check: replay every segment through the real Ruckig engine
    # before sending anything to the robot, using its actual current joint
    # state and the *effective* limits franky will actually enforce (its
    # hardware max scaled by robot.relative_dynamics_factor, same as
    # PositionWaypointMotion::setInputLimits does internally).
    # -----------------------------
    segment_durations = np.diff(ts)
    print("Pre-flight checking all segments against Ruckig...")
    preflight_check_segments(
        qs[1:],
        qds[1:],
        segment_durations,
        vel_max=robot.joint_velocity_limit.max * args.relative_dynamics,
        acc_max=robot.joint_acceleration_limit.max * args.relative_dynamics,
        jerk_max=robot.joint_jerk_limit.max * args.relative_dynamics,
        current_position=robot.current_joint_positions,
        current_velocity=robot.current_joint_velocities,
    )
    print("Pre-flight check passed.")

    # -----------------------------
    # Convert TOPPRA samples to waypoints, using each sample's velocity and
    # the TOPPRA-computed segment duration as a minimum time to reach it.
    #
    # The first waypoint is a "settle" step: target = current position,
    # velocity = 0. franky seeds Ruckig's very first segment from the
    # robot's live desired acceleration (robot_state.ddq_d), which isn't
    # exposed through franky's Python API and so can't be checked by the
    # pre-flight above. If that's ever nonzero (e.g. settling from a
    # previous motion), jumping straight into a nonzero-velocity target can
    # trip Ruckig's input validation. This waypoint absorbs any such
    # residual velocity/acceleration before the real TOPPRA ramp starts.
    # -----------------------------
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

    joint_motion = JointWaypointMotion(joint_waypoints)

    # -----------------------------
    # Execute motion
    # -----------------------------
    print("Executing TOPPRA-parameterized joint waypoint motion...")
    robot.move(joint_motion)
    print("Motion complete.")


if __name__ == "__main__":
    main()
