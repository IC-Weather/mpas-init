from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.profiles import NATIVE_DIMENSIONS, NATIVE_STATIC_DIMENSIONS, NATIVE_ZERO_DIMENSIONS, initialize_native_noah
from test_vertical_grid import assert_stage


def assert_complete_native(actual_path, reference_path):
    with Dataset(actual_path) as actual, Dataset(reference_path) as reference:
        assert set(actual.variables) == set(reference.variables)
        assert set(actual.dimensions) == set(reference.dimensions), "dimension inventory"
        for name, expected in reference.dimensions.items():
            assert len(actual.dimensions[name]) == len(expected), name + " dimension size"
            assert actual.dimensions[name].isunlimited() == expected.isunlimited(), name + " unlimited dimension"
        for name, expected in reference.variables.items():
            value = actual[name]
            assert value.dimensions == expected.dimensions, name
            assert value.dtype == expected.dtype, name + " dtype"
            assert value.units == expected.units, name
            a, b = np.asarray(value[:]), np.asarray(expected[:])
            if b.dtype.kind == "f":
                assert_stage(torch.tensor(a), torch.tensor(b), 512)
            else:
                np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize("defect,error", (("extra_dimension", "dimension inventory"),
                                        ("fixed_time", "unlimited dimension"), ("dtype", "rho dtype")))
def test_complete_native_comparator_rejects_schema_changes(tmp_path, defect, error):
    reference, changed = tmp_path / "reference.nc", tmp_path / "changed.nc"
    for path in (reference, changed):
        with Dataset(path, "w") as ds:
            actual = path == changed
            ds.createDimension("Time", 1 if actual and defect == "fixed_time" else None)
            ds.createDimension("nCells", 2)
            if actual and defect == "extra_dimension":
                ds.createDimension("unused", 3)
            var = ds.createVariable("rho", "f4" if actual and defect == "dtype" else "f8", ("Time", "nCells"))
            var.units = "kg m-3"
            var[:] = [[1., 2.]]
    with pytest.raises(AssertionError, match=error):
        assert_complete_native(changed, reference)


@pytest.mark.parametrize("classification", ("USGS", "MODIFIED_IGBP_MODIS_NOAH"))
def test_native_noah_tensor_contract_and_static_derivatives(classification):
    sizes = {"nCells": 3, "nMonths": 12, "nlcat": 24 if classification == "USGS" else 20, "nscat": 16}
    statics = {name: torch.full(tuple(sizes[dim] for dim in dimensions), .3, dtype=torch.float64)
               for name, dimensions in NATIVE_STATIC_DIMENSIONS.items()}
    statics["landmask"] = torch.tensor([0, 1, 1])
    dycore = {name: torch.zeros(3, 5, dtype=torch.float64) for name in ("rho", "qc", "qr", "qi", "qs", "qg")}
    result = initialize_native_noah(statics, dycore, classification)
    assert set(result) == set(NATIVE_DIMENSIONS)
    assert result["iswater_lu"] == (16 if classification == "USGS" else 17)
    assert result["isice_lu"] == (24 if classification == "USGS" else 15)
    for name in statics:
        assert result[name] is statics[name]
    for name in NATIVE_ZERO_DIMENSIONS:
        assert torch.count_nonzero(result[name]) == 0
    floating = tuple(name for name in statics if name != "landmask")
    inputs = tuple(statics[name].requires_grad_() for name in floating)
    directions = tuple(torch.cos(torch.arange(value.numel(), dtype=value.dtype)).reshape_as(value) for value in inputs)
    def evaluate(*values):
        output = initialize_native_noah({**statics, **dict(zip(floating, values))}, dycore, classification)
        return torch.cat([output[name].flatten() for name in floating] + [output["canwat"]])
    value, tangent = torch.func.jvp(evaluate, inputs, directions)
    expected = torch.cat([direction.flatten() for direction in directions] + [torch.zeros(3)])
    torch.testing.assert_close(tangent, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(value.sum(), inputs)
    for actual, input_value in zip(gradient, inputs):
        torch.testing.assert_close(actual, torch.ones_like(input_value), rtol=0, atol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(tangent), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="inventory"):
        initialize_native_noah({name: value for name, value in statics.items() if name != "soilf"}, dycore, classification)
    with pytest.raises(ValueError, match="shape"):
        initialize_native_noah({**statics, "soilf": statics["soilf"][:, :2]}, dycore, classification)
    with pytest.raises(RuntimeError, match="no-analyzed-cloud"):
        initialize_native_noah(statics, {**dycore, "qc": dycore["qc"] + .01}, classification)
