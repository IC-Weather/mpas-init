import pytest
import torch

from mpas_init.horizontal import overlapping_parabolic, sixteen_point, regular_latlon_indices


def test_quadratic_interpolation():
    y, x = torch.meshgrid(torch.arange(9, dtype=torch.float64),
                          torch.arange(12, dtype=torch.float64), indexing="ij")
    field = 3.0 + x * x + 2 * x * y + 0.5 * y * y
    tx, ty = torch.tensor([2.3, 5.6, 8.1], dtype=torch.float64), torch.tensor([2.4, 3.7, 6.1], dtype=torch.float64)
    expected = 3.0 + tx * tx + 2 * tx * ty + 0.5 * ty * ty
    torch.testing.assert_close(sixteen_point(field, tx, ty), expected, rtol=1e-14, atol=1e-14)


def test_raw_descending_longitudes_reject_before_native_remapping():
    latitude = torch.linspace(-90., 90., 5, dtype=torch.float64)
    longitude = torch.arange(315., -1., -45., dtype=torch.float64)
    with pytest.raises(RuntimeError, match="source longitude must scan eastward"):
        regular_latlon_indices(latitude, longitude, latitude.new_tensor([.2]), latitude.new_tensor([.8]))


def test_periodic_seam_poles_and_exact_zeros():
    field = torch.arange(35, dtype=torch.float64).reshape(5, 7)
    x = torch.tensor([-0.3, 6.7, 0.0, 7.0], dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 4.0, 0.0], dtype=torch.float64)
    values = sixteen_point(field, x, y)
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=1e-13)
    assert values[2] == field[4, 0] and values[3] == 0
    zeros = sixteen_point(torch.zeros_like(field), x, y)
    # The original substitutes 1e-20 before interpolation; roundoff can retain it.
    torch.testing.assert_close(zeros, torch.zeros_like(zeros), rtol=0, atol=1e-15)


def test_zero_branches_and_input_derivatives():
    values = torch.tensor([0.0, 2.0, 4.0, 0.0], dtype=torch.float64, requires_grad=True)
    x = values.new_tensor(0.25)
    result = overlapping_parabolic(x, *values.unbind())
    assert result == 2.5
    derivative, = torch.autograd.grad(result, values)
    torch.testing.assert_close(derivative, values.new_tensor([0.0, 0.75, 0.25, 0.0]), rtol=0, atol=0)
    source = (torch.arange(63, dtype=torch.float64).reshape(7, 9) + 1).requires_grad_()
    assert torch.autograd.gradcheck(lambda f: sixteen_point(f, f.new_tensor([2.3]), f.new_tensor([3.2])), (source,))


def test_rounded_north_pole_uses_original_search_fallback():
    field = torch.arange(35, dtype=torch.float64).reshape(5, 7)
    actual = sixteen_point(field, field.new_tensor([2.3]), field.new_tensor([-0.0000005]))
    torch.testing.assert_close(actual, field[0, 2:3], rtol=0, atol=0)
