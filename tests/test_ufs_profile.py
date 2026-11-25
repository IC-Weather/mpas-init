import numpy as np
import pytest
import torch

from mpas_init.profiles import UFS_STATIC_FIELDS, initialize_ufs_static


def test_ufs_static_identity_and_active_maximum_derivatives(tmp_path):
    fields = {name: torch.ones(3, dtype=torch.float64) for name in UFS_STATIC_FIELDS}
    fields["gwd_elvmax"] = torch.tensor([10., 40., 80.], dtype=torch.float64)
    terrain = torch.tensor([-20., 30., 90.], dtype=torch.float64)
    names = tuple(fields)
    inputs = (*fields.values(), terrain)

    def compute(*values):
        result = initialize_ufs_static(dict(zip(names, values[:-1])), {"ter": values[-1]})
        return torch.cat([result[name] for name in names])

    expected = torch.cat([torch.tensor([10., 40., 90.]) if name == "gwd_elvmax" else fields[name] for name in names])
    torch.testing.assert_close(compute(*inputs), expected, rtol=0, atol=0)
    active = tuple(value.clone().requires_grad_() for value in inputs)
    reverse = torch.autograd.grad(compute(*active).sum(), active)
    maximum_index = names.index("gwd_elvmax")
    packet = {}
    for i, value in enumerate(inputs):
        tangent = tuple(torch.ones_like(v) if j == i else torch.zeros_like(v) for j,v in enumerate(inputs))
        _, jvp = torch.func.jvp(compute, inputs, tangent)
        analytic = torch.zeros_like(expected)
        if i == len(names):
            analytic[maximum_index*3:maximum_index*3+3] = torch.tensor([0., 0., 1.])
            grad = torch.tensor([0., 0., 1.], dtype=torch.float64)
        else:
            grad = torch.tensor([1., 1., 0.], dtype=torch.float64) if i == maximum_index else torch.ones(3, dtype=torch.float64)
            analytic[i*3:i*3+3] = grad
        torch.testing.assert_close(jvp, analytic, rtol=0, atol=0)
        torch.testing.assert_close(reverse[i], grad, rtol=0, atol=0)
        assert torch.count_nonzero(reverse[i]) > 0
        with pytest.raises(AssertionError):
            torch.testing.assert_close(jvp, torch.zeros_like(jvp), rtol=0, atol=0)
        packet.update({f"jvp_{i}": jvp.numpy(), f"analytic_{i}": analytic.numpy(),
                       f"reverse_{i}": reverse[i].numpy(), f"zero_control_{i}": torch.zeros_like(jvp).numpy()})
    np.savez(tmp_path / "static-derivatives.npz", **packet)
