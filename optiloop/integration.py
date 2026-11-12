"""Compatibility alias for tunix.integration."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tunix.integration")
