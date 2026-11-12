# Task: Document the Optix orchestration trial
Agent: muse
Expect-Files: docs/orchestration-trial.md
Timeout-Min: 20
Retries: 0
Fallback: no

## Goal
Create a short trial note proving the doc-only pipeline works. Do not touch any other files, especially under app/.

## Context
Optix Phase 1 end-to-end trial. Write docs/orchestration-trial.md with exactly these sections: Purpose, Steps, Result.

## Acceptance criteria
- [file:docs/orchestration-trial.md] contains "## Purpose"
- [file:docs/orchestration-trial.md] contains "## Steps"
- [file:docs/orchestration-trial.md] contains "## Result"
- A human confirms the note reads sensibly (manual review).

## Test plan
run: powershell -NoProfile -NonInteractive -Command "if (-not (Test-Path docs/orchestration-trial.md)) { exit 1 }"
