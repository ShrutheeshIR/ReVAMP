---
name: deliver-visuals-in-scratch
description: Put review images/videos in repo scratch/ and print paths; don't rely on chat attachments
metadata:
    pinned: true
---

Tommy often works from a plain CLI session where chat file attachments
(SendUserFile) are not viewable. When producing images or clips for him to
review (calibration comparisons, overlay stills, preview renders), write
them into the repo's gitignored `scratch/` directory with descriptive
filenames and tell him the paths, so he can open them locally himself.
Artifact web pages remain a useful extra for phone access, but the files
must always land in `scratch/` first.
