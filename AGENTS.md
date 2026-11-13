# Tunix contributor instructions

- Use `tunix/` for all implementation work and imports.
- Read README.md and the relevant docs before changing behavior.
- Preserve original `results/*.json` bytes and recorded hashes. Write derived evidence separately.
- Never present a raw speedup as quality-approved or relax validation thresholds as a cosmetic cleanup.
- Never commit credentials, local environments, or generated runtime sessions.
- Run `python -m pytest tests -q` after code changes. Report skipped GPU/live-API checks accurately.
- Work on a focused branch. Preserve unrelated edits and history; do not force-push shared branches.
- Historical agent-board and orchestrator material in `docs/archive/` is reference only, not the active workflow.
