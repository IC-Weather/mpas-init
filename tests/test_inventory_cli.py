"""Opt-in inventory reports preserve the native file and name missing source volumes."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import mpas_init

from mpas_init.inventory import WATER_SPECIES
from mpas_init.output import inventory_record, write_inventory_report
from test_inventory import assert_inventory, case, scalar_inventory


def environment():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(Path(mpas_init.__file__).resolve().parents[1]),
            **{name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "OMP_DYNAMIC": "FALSE"}


def test_report_exclusive_creation_identity_and_unavailable_source(tmp_path):
    values = case(torch.float64)
    fields = dict(zip(("areaCell", "rho", "zgrid", *WATER_SPECIES), values))
    now = datetime(2024, 5, 21, tzinfo=timezone.utc)
    source = SimpleNamespace(valid_time=now, reference_time=now - timedelta(hours=6),
                             lead=timedelta(hours=6), member=4,
                             fields={"geopotential_height": torch.tensor([-1.e30, float("nan"), -20., 30001.])})
    output, report = tmp_path / "ic.nc", tmp_path / "inventory.json"
    output.write_bytes(b"closed IC identity")
    record = inventory_record(fields, source, output)
    write_inventory_report(report, record)
    result = json.loads(report.read_text())
    assert result["ic_identity"] == {"bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    assert result["source_sample_height_range_m"] == [-20., 30001.]
    assert result["source_inventory"]["status"] == "unavailable"
    assert not result["mass_correction_applied"]
    assert result["member"] == 4 and result["lead_seconds"] == 21600
    assert set(result["target_inventory"]["water_species"]) == set(WATER_SPECIES)
    assert_inventory({key: torch.tensor(value, dtype=torch.float64) for key, value in result["target_inventory"]["totals"].items()},
                     {key: value for key, value in scalar_inventory(values).items() if not isinstance(value, list)}, torch.float64)
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        write_inventory_report(report, record)
    assert report.read_bytes() == before
    with pytest.raises(ValueError):
        write_inventory_report(tmp_path / "nonfinite.json", {**record, "bad": float("nan")})
    assert not (tmp_path / "nonfinite.json").exists()


@pytest.mark.parametrize("defect", ("existing_report", "existing_ic", "same_path", "dangling", "parent"))
def test_inventory_path_preflight_without_scientific_imports(tmp_path, defect):
    output, report = tmp_path / "ic.nc", tmp_path / "inventory.json"
    if defect == "existing_report":
        report.write_bytes(b"keep report")
    elif defect == "existing_ic":
        output.write_bytes(b"keep IC")
    elif defect == "same_path":
        report = output
    elif defect == "dangling":
        report.symlink_to(tmp_path / "absent.json")
    else:
        report = tmp_path / "absent/inventory.json"
    command = [sys.executable, "-S", "-B", "-m", "mpas_init", "--source", "absent.nc", "--static", "absent.nc",
               "--output", str(output), "--inventory", str(report)]
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, "PYTHONPATH": environment()["PYTHONPATH"]},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "error:" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    if defect == "existing_ic":
        assert output.read_bytes() == b"keep IC"
    else:
        assert not output.exists()
    if defect == "existing_report":
        assert report.read_bytes() == b"keep report"
