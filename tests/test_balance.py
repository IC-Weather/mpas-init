import torch

from mpas_init.balance import hydrostatic_state
from mpas_init.vertical_grid import tc_interfaces, vertical_grid
from test_vertical_grid import tetrahedron


def columns(n_levels=55):
    terrain = torch.tensor([0.0, 100.0, 2000.0, 4500.0], dtype=torch.float64)
    zw = tc_interfaces(n_levels, terrain.new_tensor(30000.0))
    grid = vertical_grid(terrain, tetrahedron(), zw, n_smooth=0)
    z = 0.5 * (grid["zgrid"][:, 1:] + grid["zgrid"][:, :-1])
    temperature = torch.clamp(288.0 - 0.0065 * z, min=210.0)
    pressure = 100000.0 * torch.exp(-9.80616 * z / (287.0 * 265.0))
    qv = 0.014 * torch.exp(-z / 2200.0) + 0.000005
    return temperature, pressure, qv, grid


def evaluate(temperature, pressure, qv, grid):
    return hydrostatic_state(temperature, pressure, qv,
                              *(grid[n] for n in ("zgrid", "zz", "fzm", "fzp", "dzu", "rdzw")))


def test_discrete_hydrostatic_residual():
    t, p, q, grid = columns()
    state = evaluate(t, p, q, grid)
    rho = state["rho_zz"]
    rr = rho - state["rho_base"]
    expected = (state["pressure_p"][:, :-1]
                - (grid["fzm"][1:] * rr[:, 1:] + grid["fzp"][1:] * rr[:, :-1]) * 9.80616 * grid["dzu"][1:]
                - (grid["fzm"][1:] * rho[:, 1:] * q[:, 1:]
                   + grid["fzp"][1:] * rho[:, :-1] * q[:, :-1]) * 9.80616 * grid["dzu"][1:])
    assert torch.max(torch.abs(expected - state["pressure_p"][:, 1:])) < 0.0001
    assert torch.all(state["pressure"] > 0)
    assert torch.all(state["rho"] > 0)
    torch.testing.assert_close(state["pressure"][:, 0], p[:, 0], rtol=0, atol=0)


def test_active_temperature_and_humidity_sensitivities():
    t, p, q, grid = columns(8)
    t.requires_grad_()
    q.requires_grad_()
    def objective(t, q):
        s = evaluate(t, p, q, grid)
        return s["pressure"][0, 5] + 10000.0 * s["rho"][1, 4]
    dt, dq = torch.autograd.grad(objective(t, q), (t, q))
    tv = torch.full_like(t, 0.1)
    qv = torch.full_like(q, 1e-5)
    _, jvp = torch.func.jvp(objective, (t, q), (tv, qv))
    torch.testing.assert_close(jvp, torch.sum(dt * tv) + torch.sum(dq * qv), rtol=1e-10, atol=1e-10)
    for variable, direction, expected in (("t", tv, torch.sum(dt * tv)), ("q", qv, torch.sum(dq * qv))):
        errors = []
        for step in (0.1, 0.01, 0.001):
            if variable == "t":
                fd = (objective(t + step * direction, q) - objective(t - step * direction, q)) / (2 * step)
            else:
                fd = (objective(t, q + step * direction) - objective(t, q - step * direction)) / (2 * step)
            errors.append(float(abs(fd - expected).detach()))
        assert min(errors) < 1e-6
        assert abs(float(expected)) > 1e-4
