"""Compatibility alias for tunix.workbench_ui."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.workbench_ui")
