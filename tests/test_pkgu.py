import pytest

from pkgu import get_python, run_subprocess_cmd


def test_get_python():
    py_env = get_python()
    assert py_env is not None and "python3" in py_env and "\n" not in py_env


def test_run_subprocess_cmd():
    py_env = get_python()
    cmd = f"{py_env}\n -m pip list --outdated --format=json"
    _, bool_v = run_subprocess_cmd(cmd)
    assert bool_v is False

    cmd = f"{py_env} -m pip list --outdated --format=json"
    _, bool_v = run_subprocess_cmd(cmd)
    assert bool_v is True
