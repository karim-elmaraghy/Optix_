"""Compatibility alias for tunix.application."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.application")
