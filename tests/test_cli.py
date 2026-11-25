"""Help and argument validation do not require the numerical dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import mpas_init


@pytest.mark.parametrize("arguments,code,message", (
    (["--help"], 0, "--source"),
    (["prepare", "--help"], 0, "--data-root"),
    ([], 2, "required"),
    (["prepare"], 2, "required"),
    (["--profile", "native-noah"], 2, "requires --surface-source and --near-surface-source"),
    (["--ozone-polar-extension", "clamp"], 2, "requires --ozone-source"),
))
def test_cli_help_and_argument_errors_without_site_packages(tmp_path, arguments, code, message):
    if arguments[:1] not in ([], ["prepare"]) and "--help" not in arguments:
        arguments = [*arguments, "--source", "missing.nc", "--static", "missing.nc", "--output", "initial.nc"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(Path(mpas_init.__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-S", "-B", "-m", "mpas_init", *arguments],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == code, result.stderr
    assert message in result.stdout + result.stderr
    assert not (tmp_path / "initial.nc").exists()
