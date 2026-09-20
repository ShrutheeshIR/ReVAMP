---
name: commit-and-push-as-you-go
description: Commit AND push to the remote continuously as work progresses
metadata: 
  node_type: memory
  pinned: true
  originSessionId: 78709ea8-7edc-470d-a064-b941ac51821a
  modified: 2026-09-18T02:45:48.876Z
---

Tommy wants work committed and PUSHED to the remote continuously as it progresses — not batched at the end of a session. After each meaningful unit of work (a script landing, a calibration artifact, a render fix), commit with a descriptive message and push to origin. He set up the remote after the first commit specifically so pushes can happen throughout.

Handoff convention (added 2026-09-20): when a work effort is finished, the FINAL commit of that effort must open its message with an explicit statement that the effort is complete and handed off (e.g. "EFFORT COMPLETE — kinematic calibration done; safe to build on top"). This is how Shrutheesh (or another collaborator) knows he can immediately start working on the code without overlapping in-flight work. A commit that merely describes its change is not enough to signal completion.
