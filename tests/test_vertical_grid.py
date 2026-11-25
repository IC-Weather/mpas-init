import pytest
import torch

from mpas_init.vertical_grid import (
    GlobalConnectivity, _single_surface_factor, smooth_terrain, tc_interfaces, vertical_grid,
)


def assert_stage(actual, expected, c):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    eps = torch.finfo(expected.dtype).eps
    bound = c * eps * expected.abs() + 1.0e-15
    error = (actual - expected).abs()
    assert torch.all(error <= bound), (float(error.max()),
                                      float((error / bound).max()))


def tetrahedron(dtype=torch.float64):
    # Every cell has three neighbors. Each undirected pair shares one edge.
    cells = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    edges = torch.tensor([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]])
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    return GlobalConnectivity(cells, edges, torch.full((4,), 3),
                              torch.full((6,), 1000000.0, dtype=dtype),
                              torch.full((6,), 500000.0, dtype=dtype), pairs)


def test_flat_global_grid_and_metrics():
    mesh = tetrahedron()
    terrain = torch.zeros(4, dtype=torch.float64)
    zw = tc_interfaces(55, terrain.new_tensor(30000.0))
    output = vertical_grid(smooth_terrain(terrain, mesh), mesh, zw, n_smooth=0)
    torch.testing.assert_close(output["zgrid"], zw.expand(4, -1), rtol=0, atol=0)
    torch.testing.assert_close(output["zz"], torch.ones((4, 55), dtype=zw.dtype), rtol=0, atol=0)
    assert torch.count_nonzero(output["zxu"]) == 0
    assert zw[0] == 0 and zw[-1] == 30000
    assert torch.all(zw[1:] > zw[:-1])
    torch.testing.assert_close(output["cf1"] + output["cf2"] + output["cf3"], zw.new_tensor(1.0))


def test_terrain_gradient_independent_finite_difference():
    mesh = tetrahedron()
    terrain = torch.tensor([10.0, 100.0, 1800.0, 5000.0], dtype=torch.float64, requires_grad=True)
    zw = tc_interfaces(41, terrain.new_tensor(30000.0))
    direction = terrain.new_tensor([0.3, -0.7, 0.2, 0.5])
    def evaluate(t):
        result = vertical_grid(smooth_terrain(t, mesh), mesh, zw, n_smooth=0)
        return result["zgrid"][2, 10] + 1000.0 * result["zz"][1, 8]
    derivative, = torch.autograd.grad(evaluate(terrain), terrain)
    step = 0.001
    finite_difference = (evaluate(terrain + step * direction) - evaluate(terrain - step * direction)) / (2 * step)
    torch.testing.assert_close(derivative @ direction, finite_difference, rtol=1e-7, atol=1e-8)
    assert abs(float(derivative @ direction)) > 0.01
    # A detached/zeroed derivative must fail the same active-sensitivity check.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(0.0 * derivative @ direction, finite_difference, rtol=1e-7, atol=1e-8)


def test_single_surface_factor_reference_witness_and_limits():
    # Original grid interface 23: native real-power rounding differs from x*x.
    x = torch.tensor([0.0, 0.5031241178512573, 1.0, 3.0], dtype=torch.float32)
    expected = x.new_tensor([0.0, 0.2531338930130005, 1.0, 1.0])
    torch.testing.assert_close(_single_surface_factor(x), expected, rtol=0, atol=0)
    assert x[1] * x[1] != expected[1]
    dense = torch.linspace(0, 3, 10001, dtype=torch.float32)
    assert_stage(_single_surface_factor(dense), torch.minimum(dense*dense, torch.ones_like(dense)), 1)


def test_single_surface_factor_forward_reverse_derivatives():
    # Avoid discrete reduction boundaries and the saturation kink at x=1.
    x = torch.tensor([0.0021, 0.0512, 0.3123, 0.5031241178512573, 0.8123, 1.2],
                     dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_single_surface_factor, (x,), check_forward_ad=True,
                                    check_backward_ad=True, atol=1e-8, rtol=1e-5)
    derivative, = torch.autograd.grad(_single_surface_factor(x).sum(), x)
    expected = torch.where(x < 1, 2*x, 0.0)
    torch.testing.assert_close(derivative, expected, rtol=5e-6, atol=1e-8)
    x32 = x.detach().to(torch.float32)
    direction = torch.linspace(-0.4, 0.7, x.numel(), dtype=torch.float32)
    _, tangent = torch.func.jvp(_single_surface_factor, (x32,), (direction,))
    _, pullback = torch.func.vjp(_single_surface_factor, x32)
    expected32 = torch.where(x32 < 1, 2*x32, 0.0)
    torch.testing.assert_close(tangent, expected32 * direction, rtol=5e-6, atol=1e-8)
    torch.testing.assert_close(pullback(direction)[0], expected32 * direction, rtol=5e-6, atol=1e-8)
    zero = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    dzero, = torch.autograd.grad(_single_surface_factor(zero).sum(), zero)
    torch.testing.assert_close(dzero, torch.zeros_like(zero), rtol=0, atol=0)
