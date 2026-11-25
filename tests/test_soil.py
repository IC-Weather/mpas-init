import numpy as np
import pytest
import torch

from mpas_init.geotile import GeotileIndex
from mpas_init.horizontal import four_point_sequence
from test_vertical_grid import assert_stage


def test_missing_sequence_continuous_field_and_coordinate_derivatives():
    params = (torch.arange(225, dtype=torch.float64).reshape(15, 15) / 10 + 250).requires_grad_()
    mask = torch.ones_like(params)
    mask[2, 2] = 0
    mask[5:7, 5:7] = 0
    mask[9:15, 9:15] = 0
    x = params.new_tensor([3.2, 1.3, 5.4, 11.3], requires_grad=True)
    y = params.new_tensor([1.4, 1.4, 5.6, 11.6], requires_grad=True)
    f = lambda p, xx, yy: four_point_sequence(p * mask, xx, yy)
    field_only = lambda p: f(p, x, y)
    assert torch.autograd.gradcheck(field_only, (params,))
    # Keep every field basis, including masked zeros, without scalar JVP loops.
    with torch.no_grad():
        perturbations = torch.eye(params.numel(), dtype=params.dtype).reshape(-1, *params.shape)*1e-6
        batch = torch.vmap(field_only)
        field_fd = ((batch(params + perturbations) - batch(params - perturbations))/(2e-6)).T.reshape(4, *params.shape)
    field_jacobian = torch.func.jacfwd(field_only)(params)
    torch.testing.assert_close(field_jacobian, field_fd, rtol=1e-3, atol=1e-5)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(field_jacobian), field_fd, rtol=1e-3, atol=1e-5)
    assert torch.autograd.gradcheck(lambda xx, yy: f(params, xx, yy), (x, y), check_forward_ad=True)
    offset = params.new_zeros((), requires_grad=True)
    assert torch.autograd.gradcheck(lambda delta: field_only(params + delta), (offset,), check_forward_ad=True)
    tangent = torch.func.jvp(lambda p: f(p, x, y), (params,), (torch.ones_like(params),))[1]
    assert_stage(tangent, torch.ones_like(tangent), 16)
    field_grad, = torch.autograd.grad(f(params, x, y).sum(), params)
    assert torch.isfinite(field_grad).all() and torch.all(field_grad[mask == 0] == 0)
    # Exact grid-point branches must not acquire NaN gradients from unused radial weights.
    exact = four_point_sequence(params * mask, params.new_tensor([3.0]), params.new_tensor([1.0]))
    derivative, = torch.autograd.grad(exact.sum(), params)
    assert torch.isfinite(derivative).all() and derivative[1, 3] == 1


def test_search_omits_resolved_and_inactive_targets(monkeypatch):
    import mpas_init.horizontal as horizontal
    field = torch.arange(100, dtype=torch.float64).reshape(10, 10) + 250
    field[3:8, 3:8] = 0
    field.requires_grad_()
    x, y = field.new_tensor([1.2, 5.2, 8.2]), field.new_tensor([1.4, 5.4, 8.4])
    active = torch.tensor([True, True, False])
    original, calls = horizontal._search_missing, []
    def observed(*args, **kwargs):
        calls.append(kwargs["active"].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(horizontal, "_search_missing", observed)
    result = four_point_sequence(field, x, y, weighted16=False, sixteen=True, active=active)
    torch.testing.assert_close(calls[-1], torch.tensor([False, True, False]), rtol=0, atol=0)
    expected = original(field, x[1:2], y[1:2], 0, 0, 0.)
    torch.testing.assert_close(result[1:2], expected, rtol=0, atol=0)
    assert result[2] == 0
    for mask in (active, torch.zeros_like(active)):
        evaluate = lambda value: four_point_sequence(value, x, y, weighted16=False, sixteen=True, active=mask)
        value, tangent = torch.func.jvp(evaluate, (field,), (torch.ones_like(field),))
        gradient, = torch.autograd.grad(value.sum(), field)
        assert torch.isfinite(tangent).all() and torch.isfinite(gradient).all()
        assert tangent[~mask].count_nonzero() == 0
        if not mask.any():
            assert value.count_nonzero() == gradient.count_nonzero() == 0
    with pytest.raises(ValueError, match="active mask"):
        four_point_sequence(field, x, y, active=torch.ones(3))


def miniature_soil_temperature(directory):
    directory.mkdir()
    (directory / "index").write_text("""type=continuous
projection=regular_ll
dx=1.0
dy=1.0
known_x=1
known_y=1
known_lat=-89.5
known_lon=-179.5
tile_x=180
tile_y=180
tile_z=1
tile_bdr=3
wordsize=2
scale_factor=0.01
missing_value=0
""")
    index = GeotileIndex.read(directory / "index")
    row = np.arange(-2, 184)[:, None]
    for x in (1, 181):
        col = np.arange(x - 3, x + 183)[None]
        data = 25000 + (row % 180) * 11 + (col % 360) * 3
        data = np.where((col >= 120) & (col <= 150) & (row >= 70) & (row <= 95), 0, data)
        (directory / index.filename(x, 1)).write_bytes(data.astype(">u2").tobytes())
