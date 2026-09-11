"""Drop-in mock of the handful of franky symbols main_realtime.py and
main_realtime_continuous.py use, for testing the replanning loop without a
real FR3 connected. Backed by a viser visualization instead of hardware --
Robot.move() doesn't teleport, it interpolates through the commanded
waypoints in real time on a background thread, same as the real robot would
take some time to get there.

Swap the import at the top of main_realtime(_continuous).py:
    from franky import Duration, JointState, JointWaypoint, JointWaypointMotion, Robot
becomes
    from mock_franky import Duration, JointState, JointWaypoint, JointWaypointMotion, Robot
Nothing else in either script needs to change -- Robot(host), .move(...),
.current_joint_positions, .joint_velocity_limit.max, .recover_from_errors(),
.stop(), .join_motion(timeout=...) all behave the way those scripts expect.

Open http://localhost:8080 (viser's default) after constructing a Robot to
watch it move.
"""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from viser import transforms as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vamp" / "scripts"))
from viser_utils import setup_viser_with_robot  # noqa: E402

DEFAULT_ROBOT_DIR = Path(__file__).resolve().parents[1] / "vamp" / "resources" / "fr3_marker"
DEFAULT_URDF_NAME = "fr3_expo_spherized.urdf"
MAZE_JSON = Path(__file__).resolve().parents[1] / "vamp" / "resources" / "environments" / "maze_cuboids.json"

# Franka FR3 factory joint limits (rad/s, rad/s^2, rad/s^3) -- only used to
# give the mock's vel/acc/jerk limits realistic-looking numbers; move_along_path
# scales these by vel_scale/acc_scale/relative_dynamics same as it would for
# the real robot's limits.
FR3_JOINT_VELOCITY_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_JOINT_ACCELERATION_MAX = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
FR3_JOINT_JERK_MAX = np.array([5000.0] * 7)

FR3_READY_CONFIG = np.array([
    -1.033585786819458,
    -1.3716682195663452,
    1.5653663873672485,
    -2.2768752574920654,
    1.414989948272705,
    1.695421814918518,
    -2.515864133834839
])


class Duration:
    """Mirrors franky.Duration(milliseconds)."""

    def __init__(self, milliseconds: float):
        self.milliseconds = float(milliseconds)

    def to_seconds(self) -> float:
        return self.milliseconds / 1000.0


class JointState:
    """Mirrors franky.JointState(position=..., velocity=...)."""

    def __init__(self, position, velocity=None):
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.zeros_like(self.position) if velocity is None else np.asarray(velocity, dtype=np.float64)


class JointWaypoint:
    """Mirrors franky.JointWaypoint(joint_state, minimum_time=...)."""

    def __init__(self, state: JointState, minimum_time: Duration):
        self.state = state
        self.minimum_time = minimum_time


class JointWaypointMotion:
    """Mirrors franky.JointWaypointMotion(joint_waypoints)."""

    def __init__(self, joint_waypoints):
        self.joint_waypoints = list(joint_waypoints)


def add_maze_cuboids(server) -> None:
    """Renders vamp/resources/environments/maze_cuboids.json as boxes, with the
    same offset main_realtime_continuous.py's build_environment_from_cuboids
    applies, so the rendered walls line up with what the planner sees."""
    with open(MAZE_JSON) as f:
        cuboids = json.load(f)

    for i, c in enumerate(cuboids):
        position = (c["x"] + 0.285 * 2, c["y"], c["z"] + 0.1)
        dimensions = (c["dx"], c["dy"], c["dz"])
        wxyz = tf.SO3.from_rpy_radians(
            c.get("roll", 0.0), c.get("pitch", 0.0), c.get("yaw", 0.0)).wxyz
        server.scene.add_box(
            f"/maze/cuboid_{i}",
            color=(120, 120, 130),
            dimensions=dimensions,
            wxyz=wxyz,
            position=position,
            opacity=0.9,
        )


class Robot:
    """Mock franky.Robot, visualized with viser instead of talking to hardware."""

    def __init__(
        self,
        host: str = "mock",
        home_config=FR3_READY_CONFIG,
        robot_dir: Path = DEFAULT_ROBOT_DIR,
        urdf_name: str = DEFAULT_URDF_NAME,
        tick_rate_hz: float = 60.0,
    ):
        print(f"[mock_franky] 'connecting' to '{host}' (no hardware, viser only)")
        self.relative_dynamics_factor = 1.0
        self.joint_velocity_limit = SimpleNamespace(max=FR3_JOINT_VELOCITY_MAX.copy())
        self.joint_acceleration_limit = SimpleNamespace(max=FR3_JOINT_ACCELERATION_MAX.copy())
        self.joint_jerk_limit = SimpleNamespace(max=FR3_JOINT_JERK_MAX.copy())

        self._lock = threading.Lock()
        self._position = np.asarray(home_config, dtype=np.float64).copy()
        self._velocity = np.zeros_like(self._position)

        self._motion_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._done_event.set()  # no motion in flight yet
        self._tick_dt = 1.0 / tick_rate_hz

        self._server, self._viser_robot = setup_viser_with_robot(robot_dir, urdf_name)
        add_maze_cuboids(self._server)
        self._status_text = self._server.gui.add_text("Status", initial_value="idle", disabled=True)
        self._viser_robot.update_cfg(self._position)

    # -- franky-compatible API -------------------------------------------------

    def recover_from_errors(self):
        print("[mock_franky] recover_from_errors() -- no-op")

    @property
    def current_joint_positions(self):
        with self._lock:
            return self._position.copy()

    @property
    def current_joint_velocities(self):
        with self._lock:
            return self._velocity.copy()

    def move(self, motion: JointWaypointMotion, asynchronous: bool = False):
        self.stop()  # cancel + join any previous in-flight motion first

        with self._lock:
            start_position = self._position.copy()
        segments = self._flatten_waypoints(start_position, motion.joint_waypoints)

        self._stop_event.clear()
        self._done_event.clear()
        self._motion_thread = threading.Thread(target=self._run_motion, args=(segments,), daemon=True)
        self._motion_thread.start()

        if not asynchronous:
            self._motion_thread.join()

    def stop(self):
        if self._motion_thread is not None and self._motion_thread.is_alive():
            self._stop_event.set()
            self._motion_thread.join()
        self._motion_thread = None

    def join_motion(self, timeout=None) -> bool:
        """Waits up to `timeout` seconds for the in-flight motion to finish.
        Returns True if it finished (or nothing was running), False if it's
        still going after the timeout -- same as franky's join_motion."""
        return self._done_event.wait(timeout)

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _flatten_waypoints(start_position, joint_waypoints):
        """Turns [JointWaypoint(...), ...] into a list of
        (t_start, t_end, q_start, q_end, qd_start, qd_end) segments, where
        t is seconds from motion start."""
        segments = []
        t = 0.0
        q_prev, qd_prev = start_position, np.zeros_like(start_position)
        for wp in joint_waypoints:
            dt = max(wp.minimum_time.to_seconds(), 1e-3)
            q_next, qd_next = wp.state.position, wp.state.velocity
            segments.append((t, t + dt, q_prev, q_next, qd_prev, qd_next))
            t += dt
            q_prev, qd_prev = q_next, qd_next
        return segments

    def _run_motion(self, segments):
        self._status_text.value = "moving"
        start_time = time.monotonic()
        seg_idx = 0
        try:
            while seg_idx < len(segments):
                if self._stop_event.is_set():
                    break
                elapsed = time.monotonic() - start_time
                t0, t1, q0, q1, qd0, qd1 = segments[seg_idx]
                if elapsed >= t1:
                    seg_idx += 1
                    continue
                alpha = 0.0 if t1 <= t0 else np.clip((elapsed - t0) / (t1 - t0), 0.0, 1.0)
                position = (1 - alpha) * q0 + alpha * q1
                velocity = (1 - alpha) * qd0 + alpha * qd1
                self._set_state(position, velocity)
                time.sleep(self._tick_dt)

            if not self._stop_event.is_set() and segments:
                self._set_state(segments[-1][3], segments[-1][5])
        finally:
            self._status_text.value = "idle"
            self._done_event.set()

    def _set_state(self, position, velocity):
        with self._lock:
            self._position = np.asarray(position, dtype=np.float64)
            self._velocity = np.asarray(velocity, dtype=np.float64)
        self._viser_robot.update_cfg(self._position)
