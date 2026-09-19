---
name: subagent-token-economy
description: Delegate liberally to subagents and prefer cheap models for mechanical tasks
metadata: 
  node_type: memory
  pinned: true
  originSessionId: 78709ea8-7edc-470d-a064-b941ac51821a
  modified: 2026-09-18T01:44:44.154Z
---

The user (Tommy) wants token economy managed by delegating work to subagents rather than doing everything in the main context, and specifically reminded me to "use cheap subagents as appropriate." When spawning agents for mechanical or exploratory work (file searches, format mining, running scripts, frame extraction checks), pass a cheaper `model` override (haiku, or sonnet for moderately complex exploration) instead of inheriting the expensive session model. Reserve the session model for tasks needing frontier reasoning (design, tricky debugging, synthesis).
