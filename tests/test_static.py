import math

import pytest
import torch

from mpas_init.geotile import GeotileIndex
from mpas_init.static import (
    NearestCells, pixel_cartesian, pixel_coordinates, terrain_accumulation, terrain_mean, terrain_patch,
)
from test_geotile import write_index


def test_native_subpixel_positions_and_latitude_direction(tmp_path):
    path = tmp_path / "index"
    write_index(path)
    index = GeotileIndex.read(path)
    lat, lon = pixel_coordinates(index, 1, 1, torch.arange(6), torch.arange(12), 3)
    torch.testing.assert_close(lat, torch.tensor([math.radians(-45 + j * 90 / 3) for j in range(6)],
                                                dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(lon, torch.tensor([math.radians(45 + i * 90 / 3) for i in range(12)],
                                                dtype=torch.float64), rtol=0, atol=0)
    xyz = pixel_cartesian(lat, lon, lat.new_tensor(6371220.0))
    truth = [[6371220.0 * math.cos(lo) * math.cos(la),
              6371220.0 * math.sin(lo) * math.cos(la), 6371220.0 * math.sin(la)]
             for la in lat.tolist() for lo in lon.tolist()]
    torch.testing.assert_close(xyz, torch.tensor(truth, dtype=torch.float64),
                               rtol=16 * torch.finfo(torch.float64).eps, atol=1e-15)
    path.write_text(path.read_text().replace("dy=90", "dy=-90").replace("known_lat=-45", "known_lat=45"))
    descending = GeotileIndex.read(path)
    flipped, _ = pixel_coordinates(descending, 1, 1, torch.arange(6), torch.arange(12), 3)
    assert torch.equal(flipped, -lat)


def brute_force(points, centers):
    return torch.tensor([min(range(len(centers)),
                             key=lambda j: (sum((a - b) ** 2 for a, b in zip(p, centers[j])), *centers[j]))
                         for p in points], dtype=torch.int64)


def test_exact_locator_culling_ownership_and_native_ties():
    generator = torch.Generator().manual_seed(817)
    centers = torch.randn(117, 3, generator=generator, dtype=torch.float64)
    centers = centers / torch.linalg.vector_norm(centers, dim=-1, keepdim=True) * 6371220.0
    points = torch.randn(253, 3, generator=generator, dtype=torch.float64)
    points = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True) * 6371220.0
    locator = NearestCells.from_centers(centers)
    truth = brute_force(points.tolist(), centers.tolist())
    assert torch.equal(locator.locate(points), truth)
    split = torch.cat([locator.locate(part) for part in points.split(7)])
    assert torch.equal(split, truth)
    # Compact, distant query patches exercise candidate rejection more strongly.
    compact = points[0] + points / 10000.0
    assert torch.equal(locator.locate(compact), brute_force(compact.tolist(), centers.tolist()))
    tie_centers = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    tie = NearestCells.from_centers(tie_centers)
    assert tie.locate(torch.tensor([[0., 0., 1.]], dtype=torch.float64)).tolist() == [1]


def test_integer_terrain_accumulation_chunking_and_missing_coverage():
    values = torch.tensor([7, 20, -4, 9, 13], dtype=torch.int64)
    cells = torch.tensor([0, 2, 0, 2, 0], dtype=torch.int64)
    total, count = terrain_accumulation(values, cells, 4)
    assert torch.equal(total, torch.tensor([16, 0, 29, 0]))
    assert torch.equal(count, torch.tensor([3, 0, 2, 0]))
    a, ac = terrain_accumulation(values[:2], cells[:2], 4)
    b, bc = terrain_accumulation(values[2:], cells[2:], 4)
    assert torch.equal(a + b, total) and torch.equal(ac + bc, count)
    mean = terrain_mean(total, count, 0.1)
    torch.testing.assert_close(mean[[0, 2]], torch.tensor([16 / 3 * 0.1, 29 / 2 * 0.1], dtype=torch.float64),
                               rtol=0, atol=0)
    assert torch.isnan(mean[[1, 3]]).all()
    with pytest.raises(ValueError, match="int64"):
        terrain_accumulation(values.float(), cells, 4)


def test_cell_total_mean_continuous_derivatives():
    total = torch.tensor([200., 800.], dtype=torch.float64, requires_grad=True)
    count = torch.tensor([7, 9])
    f = lambda x: terrain_mean(x, count, 0.1)
    assert torch.autograd.gradcheck(f, (total,), check_forward_ad=True)
    weights = total.new_tensor([0.3, -0.4])
    derivative, = torch.autograd.grad((f(total) * weights).sum(), total)
    expected = weights * 0.1 / count
    torch.testing.assert_close(derivative, expected, rtol=1e-15, atol=0)
    assert torch.all(derivative != 0)


def test_public_terrain_patch_block_invariance_and_native_float_buffer(tmp_path):
    path = tmp_path / "index"
    write_index(path)
    index = GeotileIndex.read(path)
    centers = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]], dtype=torch.float64)
    locator = NearestCells.from_centers(centers)
    # Native C_FLOAT cannot preserve every 32-bit encoded integer.
    pixels = torch.tensor([[2**24 + 1, 2**24 + 3, 0, 5], [-2**24 - 1, 9, 6, 10]], dtype=torch.int64)
    results = [terrain_patch(pixels, index, locator, centers.new_tensor(1.),
                              start_x=1, start_y=1, supersample=3, block_side=side) for side in (1, 4, 32)]
    for result in results:
        assert result.keys() == {"coordinates", "cartesian", "owner", "pixels", "total", "count", "mean"}
        assert int(result["count"].sum()) == 72
        assert result["pixels"][0] == 2**24
        assert result["pixels"][3] == 2**24 + 4
        for name in result:
            torch.testing.assert_close(result[name], results[0][name], rtol=0, atol=0, equal_nan=True)
    with pytest.raises(ValueError, match="outside the tile"):
        terrain_patch(pixels, index, locator, centers.new_tensor(1.), start_x=1, start_y=1, row0=1)
