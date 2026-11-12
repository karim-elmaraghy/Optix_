"""Compatibility alias for tunix.meta_planner."""
import importlib
import sys

if __name__ == "__main__":
    from tunix.meta_planner import main
    raise SystemExit(main())

sys.modules[__name__] = importlib.import_module("tunix.meta_planner")
