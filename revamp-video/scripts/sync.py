"""Stage B: coarse time sync between the video and the logs, from wall clocks.

On some phones the container creation_time stamps when the file was CLOSED
(rby1 capture); on THIS iPhone 17 Pro it is evidently the recording START:
with start=creation the log window [-0.26 s, +389.3 s] brackets the video
[0, 386.9 s], while start=creation-duration puts every log sample after the
video ends. Log timestamps are Unix epoch.
video_time(t_log) = t_log - video_start_epoch + delta, where delta
(initially 0, expected |delta| <~ 3 s from phone/robot clock skew) is refined
by the calibration stage; this script only writes the coarse mapping.

Validation (never used to SET the sync): a frame-difference motion profile of
the video can be compared against logged joint speed by later stages.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    creation = dt.datetime.strptime(
        common.VIDEO_CREATION_UTC, "%Y-%m-%dT%H:%M:%S.%f%z")
    creation_epoch = creation.timestamp()
    video_start_epoch = creation_epoch  # creation_time == recording start here

    js = common.load_joint_states()
    log_t0, log_t1 = float(js["t"][0]), float(js["t"][-1])

    queries = common.load_queries()
    q_t0, q_t1 = queries[0]["t"], queries[-1]["t"]

    def vt(t_epoch):
        return t_epoch - video_start_epoch

    print(f"video: start epoch {video_start_epoch:.3f} "
          f"({dt.datetime.fromtimestamp(video_start_epoch, dt.timezone.utc)})"
          f" duration {common.VIDEO_DURATION_S:.3f}s")
    print(f"joint_states: [{vt(log_t0):+8.3f}, {vt(log_t1):+8.3f}] s in video time")
    print(f"queries:      [{vt(q_t0):+8.3f}, {vt(q_t1):+8.3f}] s in video time")

    out = {
        "video_start_epoch": video_start_epoch,
        "creation_epoch": creation_epoch,
        "duration_s": common.VIDEO_DURATION_S,
        "fps": common.VIDEO_FPS,
        "delta_s": 0.0,  # refined by calibrate.py; video_time = t - start + delta
        "delta_search_window_s": 3.0,
        "joint_states_video_window": [vt(log_t0), vt(log_t1)],
        "queries_video_window": [vt(q_t0), vt(q_t1)],
        "note": "delta_s=0 is the wall-clock coarse estimate; "
                "calibrate.py refines it.",
    }
    common.write_json(os.path.join(common.CALIB_DIR, "sync.json"), out)


if __name__ == "__main__":
    main()
