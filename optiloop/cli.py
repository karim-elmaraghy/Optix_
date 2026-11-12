"""Compatibility alias for tunix.cli."""
import importlib
import sys

if __name__ == "__main__":
    from tunix.cli import main
    raise SystemExit(main())

sys.modules[__name__] = importlib.import_module("tunix.cli")
