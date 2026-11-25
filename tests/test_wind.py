import torch

from mpas_init.wind import pressure_level_wind, vertical_wind
from test_vertical_grid import tetrahedron


def test_wind_source_and_height_independent_derivatives():
    mesh = tetrahedron()
    latitude = torch.linspace(-90, 90, 5, dtype=torch.float64)
    longitude = torch.arange(8, dtype=torch.float64) * 45.0
    lat_cell = latitude.new_tensor([-0.7, -0.2, 0.4, 0.9])
    lon_cell = latitude.new_tensor([0.3, 1.5, 3.1, 4.9])
    lat_edge = 0.5 * (lat_cell[mesh.cells_on_edge[:, 0]] + lat_cell[mesh.cells_on_edge[:, 1]])
    lon_edge = 0.5 * (lon_cell[mesh.cells_on_edge[:, 0]] + lon_cell[mesh.cells_on_edge[:, 1]])
    angle = torch.linspace(-1.2, 2.0, 6, dtype=torch.float64)
    height = latitude.new_tensor([0.0, 3000.0, 8000.0])[:, None, None].expand(3, 5, 8)
    east = (8.0 + height / 1000.0).clone().requires_grad_()
    north = (-3.0 + height / 3000.0).clone().requires_grad_()
    zgrid = latitude.new_tensor([0.0, 500.0, 2000.0, 5000.0]).expand(4, 4).clone().requires_grad_()
    def evaluate(e, n, z):
        return pressure_level_wind({"geopotential_height": height, "eastward_wind": e, "northward_wind": n},
                                   latitude, longitude, lat_cell, lon_cell, lat_edge, lon_edge,
                                   angle, mesh.cells_on_edge, z)
    inputs = (east, north, zgrid)
    directions = tuple(torch.cos(torch.arange(v.numel(), dtype=v.dtype)).reshape(v.shape) for v in inputs)
    output, jvp = torch.func.jvp(evaluate, inputs, directions)
    weights = torch.linspace(0.3, 1.2, output.numel(), dtype=output.dtype).reshape_as(output)
    gradient = torch.autograd.grad((output * weights).sum(), inputs)
    reverse = sum((g * d).sum() for g, d in zip(gradient, directions))
    torch.testing.assert_close(reverse, (jvp * weights).sum(), rtol=1e-12, atol=1e-12)
    assert reverse.abs() > 0.1
    for step in (1e-3, 1e-4, 1e-5):
        plus = evaluate(*(v + step*d for v, d in zip(inputs, directions)))
        minus = evaluate(*(v - step*d for v, d in zip(inputs, directions)))
        finite = (plus - minus) / (2 * step)
        torch.testing.assert_close(jvp, finite, rtol=2e-6, atol=2e-9)


def test_terrain_vertical_wind_density_and_velocity_derivatives():
    mesh = tetrahedron()
    u = torch.linspace(-3.0, 4.0, 18, dtype=torch.float64).reshape(6, 3).requires_grad_()
    density = torch.linspace(0.5, 1.2, 12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    zz = torch.linspace(0.9, 1.1, 12, dtype=torch.float64).reshape(4, 3)
    fzm = u.new_tensor([0.0, 0.4, 0.45])
    fzp = u.new_tensor([0.0, 0.6, 0.55])
    zb = torch.linspace(-0.01, 0.02, 48, dtype=torch.float64).reshape(6, 2, 4)
    zb3 = 0.2 * zb
    def evaluate(velocity, rho):
        return vertical_wind(velocity, rho, zz, fzm, fzp, zb, zb3, mesh)
    output = evaluate(u, density)
    assert torch.count_nonzero(output[:, 1:-1]) == 8
    assert torch.count_nonzero(output[:, [0, -1]]) == 0
    assert torch.autograd.gradcheck(evaluate, (u, density), check_forward_ad=True)
