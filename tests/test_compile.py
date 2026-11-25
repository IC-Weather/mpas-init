"""Dynamo capture regressions, not Inductor numerical qualification."""

from dataclasses import replace

import pytest
import torch

from mpas_init.horizontal import HorizontalGrid, remap_atmosphere
from mpas_init.thermodynamics import validate_atmospheric_support


@pytest.mark.parametrize("masked", (False, True))
def test_dynamo_captures_atmospheric_inventory_and_retains_checks(masked):
    pressure = torch.tensor([50000., 100000.], dtype=torch.float64)
    fields = {"air_temperature": torch.full((2, 3, 4), 275., dtype=torch.float64)}
    surfaces = {"air_temperature": torch.full((3, 4), 101000., dtype=torch.float64)} if masked else None
    compiled = torch.compile(validate_atmospheric_support, backend="eager", fullgraph=True)
    expected = validate_atmospheric_support(fields, pressure, surfaces)
    assert compiled(fields, pressure, surfaces) == expected
    wrong = {"air_temperature": torch.full_like(fields["air_temperature"], float("nan"))}
    with pytest.raises(RuntimeError, match="nonfinite atmospheric"):
        compiled(wrong, pressure, surfaces)


def test_dynamo_captures_per_field_grid_inventory():
    latitude = torch.tensor([90., 0., -90.], dtype=torch.float64)
    longitude = torch.tensor([0., 90., 180., 270.], dtype=torch.float64)
    values = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) + 1
    fields = {"geopotential_height": values, "air_temperature": values + 275.}
    targets = (torch.tensor([.2, -.3]), torch.tensor([.3, 1.2]))
    grids = {"air_temperature": HorizontalGrid(latitude, longitude)}

    def evaluate(fields):
        return remap_atmosphere(fields, tuple(fields), latitude, longitude, *targets, field_grids=grids)

    expected = evaluate(fields)
    actual = torch.compile(evaluate, backend="eager", fullgraph=True)(fields)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.compiler
def test_dynamo_captures_surface_calendar_and_static_inventory():
    from mpas_init.surface import initialize_surface
    from test_surface import small_surface_inputs

    source, static = small_surface_inputs()
    evaluate = lambda fields: initialize_surface(replace(source, fields=fields), static)
    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    for offset in (0., 1.):
        fields = {**source.fields, "soil_temperature": source.fields["soil_temperature"] + offset}
        actual, expected = compiled(fields), evaluate(fields)
        assert actual.keys() == expected.keys()
        for name in expected:
            assert torch.equal(actual[name], expected[name]), name
    with pytest.raises(RuntimeError, match="nonfinite surface source"):
        compiled({**source.fields, "soil_temperature": torch.full_like(source.fields["soil_temperature"], float("nan"))})
