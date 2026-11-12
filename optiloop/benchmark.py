"""Compatibility alias for tunix.benchmark."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.benchmark")
