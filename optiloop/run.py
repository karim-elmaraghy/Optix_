"""Compatibility alias for tunix.run."""
import importlib
import sys

if __name__ == "__main__":
    from tunix.run import main
    raise SystemExit(main())

sys.modules[__name__] = importlib.import_module("tunix.run")
