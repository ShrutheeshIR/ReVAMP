# What's in this directory

This is the output of `replay_task_space_log.py` (full explanation of what
that script does and why: `../../REPLAY_TASK_SPACE_LOG.md` at the repo root)
run against the original planning log at `../20260917_184242/` (101 rows,
80 solved). It reconstructs the joint-space trajectories and RRT search
trees that run's task-space planner actually produced — data the original
log never stored, only summary fields.

**Read this before the `.npy`/`.npz`/`.jsonl` files** — there's one
non-obvious gotcha below (`planning_queries.jsonl` has been appended to
three times) that will silently misalign your row indices if you don't
account for it.

## ⚠️ `planning_queries.jsonl` has 303 lines, not 101 — read this first

`RunLogger` opens `planning_queries.jsonl` in **append** mode, and this
replay was run **three times** against the same output directory (once
without tree data, then re-run(s) with `--store_tree`). Each run appends
its own 101 rows (one per row in the original log) rather than overwriting
the file. So:

- Lines **0–100** = 1st replay run
- Lines **101–201** = 2nd replay run
- Lines **202–302** = 3rd (most recent) replay run

`trajectories/*.npy` and `trees/*.npz`, by contrast, are written with
`np.save`/`np.savez` to filenames keyed only by row index (`000.npy`,
`001.npy`, ...) — so each re-run **overwrote** the previous run's files.
What's currently on disk in `trajectories/` and `trees/` is only ever the
**most recent** run's output.

**Bottom line: to match a `trajectories/{i:03d}.npy` or `trees/{i:03d}.npz`
file against its metadata, use `planning_queries.jsonl` line `202 + i`**
(0-indexed), not line `i`. For example, `trajectories/041.npy` corresponds
to line 243 of `planning_queries.jsonl`, not line 41. All three runs agree
with each other and with the original log (0 solved/unsolved mismatches,
all 80 solved-row waypoint counts match exactly), so which run's *metadata*
you read barely matters for correctness — but only the last run's
*trajectory/tree files* still exist on disk, so you must index into the
last 101 lines to get metadata that actually describes an existing file.

A quick way to load just the current (3rd-run) metadata in Python:

```python
import json

with open("planning_queries.jsonl") as f:
    all_rows = [json.loads(line) for line in f]
rows = all_rows[202:303]  # aligns with trajectories/*.npy and trees/*.npz on disk

row = rows[41]              # metadata for trajectories/041.npy / trees/041.npz
```

## Directory contents

```
planning_queries.jsonl   # 303 lines = 3 concatenated replay runs, see above
joint_states.csv         # header row only, no data (see below) -- ignore
trajectories/
  000.npy ... 100.npy    # 80 files exist (one per SOLVED row); missing for unsolved rows
trees/
  000.npz ... 100.npz    # 101 files exist (one per row that reached RRTC at all)
```

### `joint_states.csv`

Just the header row, no data. `RunLogger` always creates this file, but
this replay script never calls `log_joint_state` (there's no live robot in
a replay) — safe to ignore.

### `planning_queries.jsonl` (use lines 202–302 — see gotcha above)

One JSON object per row considered by the (most recent) replay run. Notable
fields (see the full field table in `../../REPLAY_TASK_SPACE_LOG.md`):

| Field | Meaning |
|---|---|
| `q_current` | The row's original logged robot joint configuration (7,) |
| `start_state` / `goal_state` | Resolved `[x, y, z, qx, qy, qz, qw, psi]` (8,) task-space poses |
| `smm` | The resolved branch selector (3,) |
| `solved` | Whether **this replay** found a path — compare to the original log's `solved` |
| `rrtc_nanoseconds` / `rrtc_iterations` | RRTC's own timing/iteration count |
| `num_waypoints` | Length of the resolved path (matches the corresponding `trajectories/*.npy` row count), or `null` if unsolved |
| `spheres` | OptiTrack obstacle spheres used for this row's environment |

### `trajectories/<row_index>.npy`

One `(n, 7)` float32 array per **solved** row (80 of 101) — an ambient
FR3 joint-space path (`q1..q7`) from start to goal, at 64-step interpolation
resolution. Missing for the 21 unsolved rows.

### `trees/<row_index>.npz`

One file per row that **reached RRTC** (all 101 here — every row resolved
a start/goal), solved or not. This is RRTC's actual bidirectional search
tree, captured via `RRTCSettings.store_tree` (a vamp C++ feature added
specifically to support this — requires a vamp build with
`PlanningResult.tree_nodes`/`tree_parents`/`tree_owner`). Arrays:

- `nodes`: `(m, 8)` float32 — every task-space-parameterized-space
  configuration (`[x, y, z, qx, qy, qz, qw, psi]`) RRTC inserted into either
  tree, start/goal roots included, in insertion order. **Not** resolved to
  ambient joint space.
- `parents`: `(m,)` int64 — `parents[i]` is the index of node `i`'s parent
  (a root is its own parent).
- `owner`: `(m,)` uint8 — `0` if `nodes[i]` is in the start tree, `1` if
  it's in the goal tree. Node 0 is always the start root; node 1 is always
  the goal root.

To draw a tree: for every `i` where `parents[i] != i`, draw a segment from
`nodes[i]` to `nodes[parents[i]]`, colored by `owner[i]`.

**For the 21 unsolved rows** (indices `41, 42, 43, 58, 59, 61, 62, 63, 64,
65, 67, 82, 83, 88, 94, 95, 96, 97, 98, 99, 100`), the `.npz` also has four
extra keys describing the start tree's closest approach to the goal:

- `closest_to_goal_index`: int, index into `nodes`/`owner` of the start-tree
  node closest to the goal.
- `closest_to_goal_distance`: float, Euclidean `(x, y, z)` distance in
  meters from that node to the goal — a **position-only proxy**, not
  RRTC's real internal distance metric (which also weighs orientation/psi
  and isn't exposed to Python).
- `closest_to_goal_state`: `(8,)` float32, that node's raw task-space state.
- `closest_to_goal_ambient_q`: `(7,)` float32, that state resolved to an
  FR3 joint configuration — i.e. "what would the robot's near-miss
  configuration actually look like".

Check with `"closest_to_goal_index" in np.load(path)` — these keys are
absent on solved rows' `.npz` files.

## Quick recipes

Load a solved row's trajectory and its metadata together:
```python
import json, numpy as np

with open("planning_queries.jsonl") as f:
    rows = [json.loads(l) for l in f][202:303]

i = 0
row = rows[i]
traj = np.load(f"trajectories/{i:03d}.npy")   # only exists if row["solved"]
```

Load an unsolved row's tree and its near-miss configuration:
```python
i = 41
row = rows[i]
assert not row["solved"]
tree = np.load(f"trees/{i:03d}.npz")
print(tree["closest_to_goal_distance"], tree["closest_to_goal_ambient_q"])
```
