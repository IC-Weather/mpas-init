import numpy as np
import pytest
import torch

from mpas_init.gwd import CLASSIC_FIELDS, SG_DELTA, _ordered_sum, box_indices, box_size, classic_statistics
from test_vertical_grid import assert_stage


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(11, 17), (2, 11, 17)])
def test_ordered_sum_releases_prefix_storage_without_changing_values_or_ad(dtype, shape):
    value = (torch.arange(np.prod(shape), dtype=dtype).reshape(shape) * .37).sin() * 10000
    value = value.transpose(-2, -1).requires_grad_()
    prefix = value.flatten(-2).cumsum(-1)
    expected = prefix[..., -1]
    actual = _ordered_sum(value)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    assert expected.untyped_storage().nbytes() > actual.untyped_storage().nbytes()
    direction = torch.cos(torch.arange(value.numel(), dtype=dtype)).reshape_as(value)
    primal, tangent = torch.func.jvp(_ordered_sum, (value,), (direction,))
    assert torch.equal(primal, expected)
    assert torch.equal(tangent, direction.flatten(-2).cumsum(-1)[..., -1])
    weights = torch.arange(actual.numel(), dtype=dtype).reshape_as(actual) + .25
    reverse, = torch.autograd.grad((actual * weights).sum(), value)
    assert torch.equal(reverse, weights[..., None, None].expand_as(value))


def test_classic_continuous_field_and_spacing_derivatives():
    gen = torch.Generator().manual_seed(319)
    box = (torch.rand((6, 8), generator=gen, dtype=torch.float64) * 800 + 200).requires_grad_()
    spacing = (torch.rand((6, 8), generator=gen, dtype=torch.float64) * 500 + 400).requires_grad_()
    landuse = torch.ones_like(box, dtype=torch.int64)
    landuse[:, :3] = 16
    def f(height, distance):
        fields = classic_statistics(height, landuse, distance)
        return torch.stack([fields[n] for n in CLASSIC_FIELDS])
    assert torch.autograd.gradcheck(f, (box, spacing), check_forward_ad=True)
    # Constant elevation shifts preserve moments, slopes, lengths, and orientation.
    tangent = torch.func.jvp(lambda b: f(b, spacing), (box,), (torch.ones_like(box),))[1]
    expected = torch.zeros_like(tangent)
    expected[-1] = 1.0
    assert_stage(tangent, expected, 64)
    flat = torch.ones_like(box, requires_grad=True)
    gradient, = torch.autograd.grad(f(flat, spacing).sum(), flat)
    assert torch.isfinite(gradient).all()
    batch = classic_statistics(torch.stack((box, box * 1.1)), landuse.expand(2, -1, -1), spacing.expand(2, -1, -1))
    for i, value in enumerate((box, box * 1.1)):
        direct = classic_statistics(value, landuse, spacing)
        for name in CLASSIC_FIELDS:
            assert torch.equal(batch[name][i], direct[name])


def test_box_dimensions_poles_and_index_bounds():
    lat = torch.tensor([0.0, 0.5, 0.5 * torch.pi, -0.5 * torch.pi], dtype=torch.float64)
    diameter = torch.full_like(lat, 100000.0)
    nx, ny = box_size(lat, diameter)
    assert torch.equal(ny, torch.full((4,), 108, dtype=torch.int64))
    assert nx[0] == 108 and nx[1] == 123 and nx[2] == nx[3] == 21600
    for pole in (-0.5 * torch.pi, 0.5 * torch.pi):
        for longitude in (-torch.pi, torch.pi, 2 * torch.pi):
            row, col, spacing = box_indices(lat.new_tensor(pole), lat.new_tensor(longitude), 8, 8)
            assert row.shape == col.shape == spacing.shape == (8, 8)
            assert torch.all((row >= 0) & (row < 21600) & (col >= 0) & (col < 43200))
            assert torch.all((spacing > 0) & (spacing <= SG_DELTA))
    with pytest.raises(ValueError, match="at least 4 by 4"):
        classic_statistics(torch.zeros(3, 5), torch.zeros(3, 5), torch.ones(3, 5))


def test_batched_box_indices_match_scalar_order_at_poles_and_seam():
    latitude = torch.tensor([[0., .5, -.5], [.5*torch.pi, -.5*torch.pi, .1]], dtype=torch.float64)
    longitude = torch.tensor([[-torch.pi, torch.pi, 2*torch.pi], [.4, -.7, 0.]], dtype=torch.float64)
    actual = box_indices(latitude, longitude, 8, 10)
    for row in range(2):
        for column in range(3):
            expected = box_indices(latitude[row, column], longitude[row, column], 8, 10)
            for batched, scalar in zip(actual, expected):
                assert torch.equal(batched[row, column], scalar)
