import numpy as np
import pytest
import torch

from mpas_init.climatology import land_accumulation, leaf_area_index, monthly_albedo, snow_albedo, vegetation_fraction
from mpas_init.geotile import GeotileIndex
from test_vertical_grid import assert_stage


def fixture(field):
    cells = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 5])
    landmask = torch.tensor([1, 0, 1, 1, 0, 1])
    landuse = torch.tensor([1, 17, 2, 3, 17, 15])
    if field == "snoalb":
        raw = torch.tensor([[1000, -9999, 2000, 3000, -9999, 5000, -9999, -9999, 7000]])
        return raw, cells, landmask, landuse, -9999, None, 0.01
    raw = (torch.arange(12)[:, None] * 3 + torch.arange(9)[None] * 5 + 10)
    if field == "albedo12m":
        raw = 1000 + raw * 10
    missing = -9999 if field == "albedo12m" else 255
    raw[:, 6:8] = missing
    raw[0, 0] = missing
    raw[5, 4] = missing
    fill = 8 if field == "albedo12m" else 0
    scale = 0.01 if field in ("albedo12m", "greenfrac") else 0.1
    return raw, cells, landmask, landuse, missing, fill, scale


def outputs(field, total, count, scale, landuse):
    if field == "snoalb":
        return {field: snow_albedo(total, count, scale)}
    if field == "greenfrac":
        return vegetation_fraction(total, count)
    if field == "albedo12m":
        return {field: monthly_albedo(total, count, scale, landuse, 15)}
    return {field: leaf_area_index(total, count, scale)}


@pytest.mark.parametrize("field", ["snoalb", "greenfrac", "albedo12m", "lai12m"])
def test_land_reductions_missing_rules_and_chunking(field):
    pixels, cells, landmask, landuse, missing, fill, scale = fixture(field)
    total, count = land_accumulation(pixels, cells, landmask, missing=missing, fill=fill)
    expected = [[0] * len(landmask) for _ in range(len(pixels))]
    expected_count = [0] * len(landmask)
    for pixel, cell in enumerate(cells.tolist()):
        if landmask[cell] != 1 or (fill is None and pixels[0, pixel] == missing):
            continue
        expected_count[cell] += 1
        for layer in range(len(pixels)):
            value = int(pixels[layer, pixel])
            expected[layer][cell] += fill if value == missing else value
    assert torch.equal(total, torch.tensor(expected))
    assert torch.equal(count, torch.tensor(expected_count))
    a, ac = land_accumulation(pixels[:, :4], cells[:4], landmask, missing=missing, fill=fill)
    b, bc = land_accumulation(pixels[:, 4:], cells[4:], landmask, missing=missing, fill=fill)
    assert torch.equal(a + b, total) and torch.equal(ac + bc, count)
    result = outputs(field, total, count, scale, landuse)
    if field == "snoalb":
        assert_stage(result[field], torch.tensor([0.1, 0., 0.5, 0., 0., 0.7], dtype=torch.float64), 16)
        assert count.tolist() == [1, 0, 1, 0, 0, 1]
    else:
        assert count.tolist() == [2, 0, 2, 2, 0, 1]
        if field == "albedo12m":
            assert torch.all(result[field][:, 3] == 0.08)
            assert torch.all(result[field][:, 4] == 8.0)
            assert torch.all(result[field][:, 5] == 70.0)
        else:
            assert torch.all(result[field][:, [1, 3, 4]] == 0)
            if field == "greenfrac":
                assert result[field].max() > 1.0
                assert torch.equal(result["shdmin"], result[field].min(0).values)
                assert torch.equal(result["shdmax"], result[field].max(0).values)


@pytest.mark.parametrize("field", ["snoalb", "greenfrac", "albedo12m", "lai12m"])
def test_climatology_continuous_mean_derivatives_and_masking(field):
    raw, cells, mask, landuse, missing, fill, scale = fixture(field)
    total, count = land_accumulation(raw, cells, mask, missing=missing, fill=fill)
    x = total.double().requires_grad_()
    f = lambda t: outputs(field, t, count, scale, landuse)[field]
    assert torch.autograd.gradcheck(f, (x,), check_forward_ad=True)
    weights = torch.arange(f(x).numel(), dtype=torch.float64).reshape_as(f(x)) + 1
    derivative, = torch.autograd.grad((f(x) * weights).sum(), x)
    factor = 0.01 * scale if field == "snoalb" else 1.0 if field == "greenfrac" else scale
    active = count > 0
    if field == "albedo12m":
        active = active & (landuse != 15)
    expected = torch.where(active[None], weights * factor / count.clamp_min(1)[None], 0)
    assert_stage(derivative, expected, 16)
    assert torch.all(derivative[:, active] > 0) and torch.all(derivative[:, ~active] == 0)
    with pytest.raises(ValueError, match="one-layer snow albedo"):
        land_accumulation(raw[:1].expand(2, -1), cells, mask, missing=missing, fill=None)


def miniature_climatology(directory, field):
    directory.mkdir()
    nl = 1 if field == "snoalb" else 12
    signed = field in ("snoalb", "albedo12m")
    missing = -999 if signed else 200 if field == "greenfrac" else 255
    scale = 0.1 if field == "lai12m" else 0.01
    # Opposite latitude directions and longitude origins exercise independent geotile addressing.
    dy, lat, lon = (-7.5, 86.25, -176.25) if signed else (7.5, -86.25, 3.75)
    layers = "tile_z=1" if nl == 1 else "tile_z_start=1\ntile_z_end=12"
    (directory / "index").write_text(f"""type=continuous
projection=regular_ll
dx=7.5
dy={dy}
known_x=1
known_y=1
known_lat={lat}
known_lon={lon}
tile_x=12
tile_y=12
{layers}
tile_bdr=2
wordsize={2 if signed else 1}
signed={'yes' if signed else 'no'}
endian=little
scale_factor={scale}
missing_value={missing}
""")
    index = GeotileIndex.read(directory / "index")
    for y in range(1, 25, 12):
        for x in range(1, 49, 12):
            data = np.full((nl, 16, 16), missing, dtype="<i2" if signed else "u1")
            for k in range(nl):
                for j in range(12):
                    for i in range(12):
                        row, col = y - 1 + j, x - 1 + i
                        v = 1 + (k * 17 + row * 11 + col * 7) % 70
                        if signed:
                            v = 1500 + 50 * v
                        if row < 3 or (k + row * 48 + col) % 13 == 0:
                            v = missing
                        data[k, j + 2, i + 2] = v
            (directory / index.filename(x, y)).write_bytes(data.tobytes())
    return index
