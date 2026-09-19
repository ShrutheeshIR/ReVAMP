---
name: apriltag-wrinkled-use-robot-pose
description: The AprilTag paper is wrinkled; calibrate the camera from robot pose data instead
metadata: 
  node_type: memory
  pinned: true
  originSessionId: 78709ea8-7edc-470d-a064-b941ac51821a
  modified: 2026-09-18T01:46:29.904Z
---

Tommy noted that the AprilTag in the hardware video (visible at t≈0–8 s, on a white 10 cm block right by the robot base, tag center a little below the base per apriltag_desc.txt: 23 cm in front, 19 cm above the Panda base) is printed on slightly wrinkled paper, so its detected corners should NOT be trusted for precise 3D pose / camera extrinsics. Instead, use the robot pose information, which is exact: forward kinematics on the logged joint states (final_vid_0917/joint_states.csv, ~28.7 Hz) gives ground-truth 3D positions of the end-effector (and any robot link) in the robot base frame over the whole video. Match those against tracked image positions to solve camera intrinsics + extrinsics + time offset. The AprilTag is only good as a coarse initialization or sanity check.
