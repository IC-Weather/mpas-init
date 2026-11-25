"""Independent scalar finite-volume inventories and their analytic derivatives."""

import math

import numpy as np
import pytest
import torch

from mpas_init.inventory import WATER_SPECIES, atmosphere_inventory


def case(dtype, hail=True):
    area = torch.tensor([2., 3., 5.], dtype=dtype)
    rho = torch.tensor([[1.2, .9], [.8, .4], [1.1, .6]], dtype=dtype)
    height = torch.tensor([[-1., 2., 6.], [1., 3., 8.], [0., 4., 7.]], dtype=dtype)
    water = tuple(torch.tensor([[.002, .001], [.003, .004], [.005, .006]], dtype=dtype) / (index + 1)
                  for index in range(7 if hail else 6))
    return (area, rho, height, *water)


def evaluate(*values):
    return atmosphere_inventory(*values[:3], dict(zip(WATER_SPECIES, values[3:], strict=False)))


def scalar_inventory(values, directions=None):
    """Integrate layer masses with math.fsum; optionally apply the product rule."""
    a, r, z, *water = [value.detach().cpu().tolist() for value in values]
    if directions is None:
        da, dr, dz, *dq = [torch.zeros_like(value).tolist() for value in values]
    else:
        da, dr, dz, *dq = [value.detach().cpu().tolist() for value in directions]
    columns, tangents = {"dry_air": []}, {"dry_air": []}
    for name in WATER_SPECIES[:len(water)]:
        columns[name], tangents[name] = [], []
    for i in range(len(a)):
        dry, ddry = [], []
        for k in range(len(r[i])):
            thickness = z[i][k + 1] - z[i][k]
            dthickness = dz[i][k + 1] - dz[i][k]
            dry.append(r[i][k] * thickness)
            ddry.append(dr[i][k] * thickness + r[i][k] * dthickness)
        columns["dry_air"].append(math.fsum(dry))
        tangents["dry_air"].append(math.fsum(ddry))
        for name, q, d in zip(WATER_SPECIES, water, dq):
            columns[name].append(math.fsum(dry[k] * q[i][k] for k in range(len(dry))))
            tangents[name].append(math.fsum(ddry[k] * q[i][k] + dry[k] * d[i][k] for k in range(len(dry))))
    for name, members in (("condensate", WATER_SPECIES[1:len(water)]),
                          ("represented_water", WATER_SPECIES[:len(water)])):
        columns[name] = [math.fsum(columns[s][i] for s in members) for i in range(len(a))]
        tangents[name] = [math.fsum(tangents[s][i] for s in members) for i in range(len(a))]
    actual, derivative = {}, {}
    for name, column in columns.items():
        actual[name + "_column_kg_m2"] = column
        derivative[name + "_column_kg_m2"] = tangents[name]
        actual[name + "_mass_kg"] = math.fsum(a[i] * column[i] for i in range(len(a)))
        derivative[name + "_mass_kg"] = math.fsum(da[i] * column[i] + a[i] * tangents[name][i] for i in range(len(a)))
    actual["represented_air_mass_kg"] = math.fsum((actual["dry_air_mass_kg"], actual["represented_water_mass_kg"]))
    derivative["represented_air_mass_kg"] = math.fsum((derivative["dry_air_mass_kg"], derivative["represented_water_mass_kg"]))
    actual["area_m2"], derivative["area_m2"] = math.fsum(a), math.fsum(da)
    actual["volume_m3"] = math.fsum(a[i] * (z[i][-1] - z[i][0]) for i in range(len(a)))
    derivative["volume_m3"] = math.fsum(da[i] * (z[i][-1] - z[i][0]) + a[i] * (dz[i][-1] - dz[i][0]) for i in range(len(a)))
    for position, index in (("bottom", 0), ("top", -1)):
        for label, select in (("min", min), ("max", max)):
            cell = select(range(len(a)), key=lambda i: z[i][index])
            actual[position + "_" + label + "_m"] = z[cell][index]
            derivative[position + "_" + label + "_m"] = dz[cell][index]
    return actual if directions is None else derivative


def assert_inventory(actual, expected, dtype):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        reference = torch.tensor(expected[name], dtype=dtype)
        assert value.shape == reference.shape, name
        assert torch.all(torch.abs(value - reference) <= 64 * torch.finfo(dtype).eps * torch.abs(reference) + 1.e-15), name


def flatten(result):
    return torch.cat([result[name].reshape(-1) for name in sorted(result)])


def scalar_flatten(result):
    return [x for name in sorted(result) for x in (result[name] if isinstance(result[name], list) else [result[name]])]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("hail", (False, True))
def test_analytic_inventory_and_zero_condensate(dtype, hail):
    values = case(dtype, hail)
    assert_inventory(evaluate(*values), scalar_inventory(values), dtype)
    clear = (*values[:4], *(torch.zeros_like(q) for q in values[4:]))
    result = evaluate(*clear)
    assert_inventory(result, scalar_inventory(clear), dtype)
    assert torch.count_nonzero(result["condensate_column_kg_m2"]) == 0
    assert torch.equal(result["represented_water_mass_kg"], result["qv_mass_kg"])


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_inventory_all_argument_jvp_vjp_and_independent_fd(dtype, tmp_path):
    values = tuple(value.requires_grad_() for value in case(dtype))
    directions = tuple(torch.cos(torch.arange(value.numel(), dtype=dtype).reshape(value.shape) + index) * .01
                       for index, value in enumerate(values))
    primal, tangent = torch.func.jvp(evaluate, values, directions)
    assert_inventory(primal, scalar_inventory(values), dtype)
    expected = torch.tensor(scalar_flatten(scalar_inventory(values, directions)), dtype=dtype)
    torch.testing.assert_close(flatten(tangent), expected, rtol=2.e-6, atol=2.e-8)
    payload = {"jvp_actual": flatten(tangent).detach().numpy(), "jvp_reference": expected.numpy()}
    for argument in range(len(values)):
        isolated = tuple(direction if i == argument else torch.zeros_like(direction) for i, direction in enumerate(directions))
        _, partial = torch.func.jvp(evaluate, values, isolated)
        reference = torch.tensor(scalar_flatten(scalar_inventory(values, isolated)), dtype=dtype)
        torch.testing.assert_close(flatten(partial), reference, rtol=2.e-6, atol=2.e-8)
        payload["partial_actual__" + str(argument)] = flatten(partial).detach().numpy()
        payload["partial_reference__" + str(argument)] = reference.numpy()
    weights = torch.linspace(-.3, .7, expected.numel(), dtype=dtype)
    reverse = torch.autograd.grad((flatten(evaluate(*values)) * weights).sum(), values)
    for argument, value in enumerate(values):
        analytic = []
        for index in range(value.numel()):
            basis = [torch.zeros_like(x) for x in values]
            basis[argument].reshape(-1)[index] = 1.
            derivative = scalar_flatten(scalar_inventory(values, basis))
            analytic.append(math.fsum(x * y for x, y in zip(derivative, weights.tolist())))
        torch.testing.assert_close(reverse[argument], torch.tensor(analytic, dtype=dtype).reshape(value.shape), rtol=2.e-6, atol=2.e-8)
        payload["vjp_actual__" + str(argument)] = reverse[argument].detach().numpy()
        payload["vjp_reference__" + str(argument)] = np.asarray(analytic).reshape(value.shape)
    # Scalar double-precision differences test the analytic rule, not float32 cancellation.
    for step in (1.e-4, 5.e-5, 2.5e-5):
        plus = [x.detach().double() + step * d.double() for x, d in zip(values, directions)]
        minus = [x.detach().double() - step * d.double() for x, d in zip(values, directions)]
        finite = (torch.tensor(scalar_flatten(scalar_inventory(plus)), dtype=torch.float64)
                  - torch.tensor(scalar_flatten(scalar_inventory(minus)), dtype=torch.float64)) / (2 * step)
        torch.testing.assert_close(flatten(tangent).double(), finite, rtol=2.e-6, atol=2.e-8)
        payload["finite_difference__" + str(step)] = finite.numpy()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(flatten(tangent) * 0, expected, rtol=2.e-6, atol=2.e-8)
    np.savez(tmp_path / "derivatives.npz", **payload)


@pytest.mark.parametrize("defect", ("area", "rho", "height", "water", "dtype", "shape", "missing", "extra", "nan", "overflow"))
def test_invalid_finite_volumes_reject(defect):
    a, r, z, *q = case(torch.float64)
    water = dict(zip(WATER_SPECIES, q))
    if defect == "area":
        a[0] = 0
    elif defect == "rho":
        r[0, 0] = -1
    elif defect == "height":
        z[0, 1] = z[0, 0]
    elif defect == "water":
        q[-1][0, 0] = -1
    elif defect == "dtype":
        a = a.float()
    elif defect == "shape":
        z = z[:, :-1]
    elif defect == "missing":
        del water["qg"]
    elif defect == "extra":
        water["ozone"] = q[0]
    elif defect == "nan":
        q[0][0, 0] = torch.nan
    else:
        r[0, 0] = torch.finfo(r.dtype).max
    with pytest.raises((ValueError, RuntimeError)):
        atmosphere_inventory(a, r, z, water)
