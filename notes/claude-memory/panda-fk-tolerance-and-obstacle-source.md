---
name: panda-fk-tolerance-and-obstacle-source
description: "Expect few-mm Panda FK mismatch; obstacle data comes from planning_queries.jsonl, not optitrack"
metadata: 
  node_type: memory
  pinned: true
  originSessionId: 78709ea8-7edc-470d-a064-b941ac51821a
  modified: 2026-09-18T14:31:59.363Z
---

Two standing facts from Tommy for this project:

1. The Franka Panda/FR3 has loose manufacturing tolerances (a known issue), so forward kinematics may mismatch the physical arm by up to a few millimeters even with logged joint states. When calibrating the camera against robot pose or evaluating overlay accuracy, treat a few-mm 3D error floor as expected — use robust losses, don't chase sub-mm agreement, and don't interpret a few-px residual floor as a calibration bug.

2. Do NOT pursue continuous optitrack logs for obstacle visualization: the sphere (and cuboid) obstacle poses listed per planning query in final_vid_0917/planning_queries.jsonl are the intended and sufficient obstacle data source. Tommy later settled HOW to show them: do NOT interpolate sphere positions between query snapshots (it reads as the bubble laggily pursuing the wand). Instead hold each bubble static at its last snapshot and teleport it at each new query — specifically at t_query − external_duration_s, since obstacles were sampled at query submission and the plan lands after the solve time. This step-hold shows the planner's actual belief, which is the honest visualization.
