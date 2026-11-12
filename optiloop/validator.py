"""Compatibility alias for tunix.validator."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.validator")
