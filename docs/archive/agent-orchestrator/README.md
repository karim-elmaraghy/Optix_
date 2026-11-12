# Optix orchestrator (Phase 1)

Paste-a-spec multi-agent coding: ChatGPT writes the spec, you paste it once,
workers execute in isolated worktrees, results return as draft PRs.

## Layout

- `optix.config.json` — lanes, budgets, timeouts, worker order. No secrets.
- `Submit-Task.ps1` — paste spec (text, `-File`, or stdin) → validates → creates
  the GitHub task issue → records state. `-WhatIf` validates without writes.
- `Dispatch.ps1` — one pass: claim → worktree+branch → worker → independent
  verification → commit/push → draft PR. `-Issue N`, `-Retry`, `-WhatIf`.
- `Get-OptixStatus.ps1` — read-only monitor: tasks, lanes, budgets.
- `adapters/` — one lane per agent. A lane runs only after `Test-*Lane`
  verifies its executable and non-interactive flags; otherwise it reports
  disabled with a reason and never pretends to run.
- `lib/` — spec parser, REST-only GitHub calls, atomic state + lock, safe
  process-tree termination, scoped git helper.
- `../.optix/` (gitignored) — `state.json`, `runs/<issue>/`, dispatcher lock.

## Run a trial (your own terminal, not the agent sandbox)

```powershell
cd C:\Karim\Code\Personal\Optix_
.\management\orchestrator\Get-OptixStatus.ps1
.\management\orchestrator\Submit-Task.ps1 -File management\orchestrator\trial-spec.md
.\management\orchestrator\Dispatch.ps1 -Issue <N>
```

## Safety rules (enforced, not advisory)

- Workers run only in orchestrator-created worktrees; the main tree is never a
  worker cwd. No reset, no force-push, no branch/worktree deletion.
- Worker exit 0 is not enough: the dispatcher re-checks changed files,
  acceptance checks, and the test command, and reports anything unverified.
- Timeouts kill only the verified worker process tree (PID + start-time
  identity check, then native taskkill /T).
- `--yolo` / `--disable-sandbox` and equivalents are banned in adapters.
  Approval-gated runs stop as blocked for a human decision.
- Nothing merges or deploys. PRs are drafts; you merge.
