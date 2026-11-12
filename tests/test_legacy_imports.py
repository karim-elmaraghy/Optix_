"""Old notebooks must resolve the same objects after the package migration."""
import importlib
import subprocess
import sys


def test_legacy_modules_preserve_identity():
    for name in ('cli', 'validator', 'profiles', 'archive', 'application', 'meta_planner'):
        assert importlib.import_module('optiloop.' + name) is importlib.import_module('tunix.' + name)


def test_legacy_cli_entry_point():
    result = subprocess.run([sys.executable, '-m', 'optiloop.cli', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'tunix' in result.stdout
