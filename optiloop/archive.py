"""Compatibility alias for tunix.archive."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.archive")
