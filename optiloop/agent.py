"""Compatibility alias for tunix.agent."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.agent")
