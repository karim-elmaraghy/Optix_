"""Compatibility alias for tunix.workload."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.workload")
