# Contributing to Tunix

Start with the [developer guide](docs/DEVELOPER.md). Install an editable checkout with `python -m pip install -e ".[test]"`, make changes on a focused branch, and run `python -m pytest tests -q` before opening a pull request.

All code and notebooks import `tunix`. Describe behavior changes, test results, and any checks that require a GPU or external credentials.

Original measured JSON in `results/` is evidence: preserve its bytes and hashes. Keep synthetic fixtures labeled, write derived reports separately, and distinguish raw performance from quality-approved outcomes. Never include credentials in commits, screenshots, logs, or result archives.
