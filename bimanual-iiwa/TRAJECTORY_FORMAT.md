# Bimanual IIWA trajectory file format

Twelve files total: 2 methods (`DualFollower`, `LeaderFollower`) x 6 segments
each. One start-goal planning query per file -- a single solved path, not
multiple candidates.

## Location and naming

Write files into `bimanual-iiwa/trajectories/` (relative to this file's
directory), named:

```
dualfollower_T_to_B.json
dualfollower_B_to_M.json
dualfollower_M_to_B.json
dualfollower_M_to_T.json
dualfollower_T_to_M.json
dualfollower_B_to_T.json
leaderfollower_T_to_B.json
leaderfollower_B_to_M.json
leaderfollower_M_to_B.json
leaderfollower_M_to_T.json
leaderfollower_T_to_M.json
leaderfollower_B_to_T.json
```

(`_to_` in the filename, not the literal arrow -- keep the arrow inside the
JSON's own `segment` field.)

## Schema

```json
{
  "method": "DualFollower",
  "segment": "T->B",
  "planning_time_s": 0.842,
  "dt_s": 0.01,
  "configs": [
    [q1_L, q2_L, q3_L, q4_L, q5_L, q6_L, q7_L,
     q1_R, q2_R, q3_R, q4_R, q5_R, q6_R, q7_R],
    "... one row per waypoint ..."
  ]
}
```

Field-by-field:

- **`method`** -- exactly `"DualFollower"` or `"LeaderFollower"` (used
  verbatim in on-screen captions).
- **`segment`** -- exactly one of `"T->B"`, `"B->M"`, `"M->B"`, `"M->T"`,
  `"T->M"`, `"B->T"` (used verbatim in captions; must match the file's own
  name).
- **`planning_time_s`** -- float, seconds. Wall-clock planning time for
  this one segment.
- **`configs`** -- list of waypoints, each a list of exactly **14 floats**
  (radians), in this fixed order: **left IIWA's 7 joints, then right
  IIWA's 7 joints** (`iiwa_left::iiwa_joint_1..7`, then
  `iiwa_right::iiwa_joint_1..7`). This matches the joint ordering in
  `rby1-constrained-planning/models/iiwa_bimanual_table_only.dmd.yaml`
  (`iiwa_left` added before `iiwa_right`) -- do not reorder or interleave.
  No gripper/finger DOFs -- if your planner's configuration vector
  includes finger joints, drop them before writing (or tell me, and I'll
  adjust this spec instead of silently truncating).
- **`dt_s`** -- float, seconds between consecutive waypoints, assumed
  uniform. **If the trajectory is not uniformly timed** (e.g. after
  TOPPRA retiming), omit `dt_s` and instead include:
  - **`times_s`** -- list of floats, same length as `configs`, seconds
    from the start of this segment. (Provide exactly one of `dt_s` or
    `times_s`, not both.)

## Example (2 waypoints, illustrative only)

```json
{
  "method": "DualFollower",
  "segment": "T->B",
  "planning_time_s": 0.842,
  "dt_s": 0.01,
  "configs": [
    [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0,
     0.0,  0.3, 0.0, -1.5, 0.0, 1.2, 0.0],
    [0.01, -0.29, 0.0, -1.49, 0.0, 1.2, 0.0,
     0.01, 0.29, 0.0, -1.49, 0.0, 1.2, 0.0]
  ]
}
```

## What happens downstream (context, not required of you)

The renderer loads each file, runs Drake FK over `configs` using
`iiwa_bimanual_table_only.dmd.yaml`, and computes the left-gripper/
right-gripper relative transform at every waypoint. One **global**
reference transform is fixed from waypoint 0 of the very first trajectory
loaded (not re-zeroed per segment), and every waypoint of every one of the
12 trajectories is reported as deviation from that single reference --
so a real drift at a handoff between two segments shows up as error
instead of being hidden.
