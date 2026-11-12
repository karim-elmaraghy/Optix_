"""Compatibility alias for tunix.safe_apply."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.safe_apply")
