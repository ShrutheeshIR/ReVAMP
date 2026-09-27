---
name: no-large-files-in-repo
description: "Never commit large files; the source video is user-provided, not tracked"
metadata: 
  node_type: memory
  pinned: true
  originSessionId: 78709ea8-7edc-470d-a064-b941ac51821a
  modified: 2026-09-18T02:14:48.151Z
---

Tommy's standing rule: never commit large files to this repo. In particular the ~2.3 GB source video (shru_revamp_vid_hanlanphone-001.MOV) must be provided by each user of the GitHub repo themselves — it is gitignored (*.MOV), and pipeline scripts should check for its presence up front and name it as a missing prerequisite rather than failing mid-run or substituting anything. The same applies to rendered .mp4 outputs, venv/, and scratch/ (all gitignored). Small derived artifacts (calib/*.json) are committed.
