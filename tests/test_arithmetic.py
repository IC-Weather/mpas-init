import ctypes
import ctypes.util
import platform

import numpy as np
import pytest
import torch

from mpas_init.arithmetic import _exp, _power, _single_power, _sincos
from test_vertical_grid import assert_stage


@pytest.mark.parametrize("operation", ("sincos", "exp", "power"))
def test_consumed_scalar_float32_primitives(tmp_path, operation):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("requires the consumed Darwin arm64 scalar math library")
    native = ctypes.CDLL(ctypes.util.find_library("m"))
    generator = torch.Generator().manual_seed(132)
    packets = {}
    if operation == "sincos":
        x = torch.cat((torch.linspace(-2*torch.pi, 2*torch.pi, 100001),
                       torch.rand(100001, generator=generator) * 4*torch.pi - 2*torch.pi))
        actuals = _sincos(x)
        for name, actual in zip(("sin", "cos"), actuals):
            routine = getattr(native, name + "f")
            routine.argtypes, routine.restype = [ctypes.c_float], ctypes.c_float
            expected = torch.tensor([routine(value) for value in x.tolist()])
            assert_stage(actual, expected, 16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.count_nonzero(getattr(torch, name)(x) != expected) > 0
            packets[name + "_reference"], packets[name + "_actual"] = expected.numpy(), actual.numpy()
    elif operation == "exp":
        x = torch.cat((torch.linspace(-80., 80., 100001), torch.rand(100001, generator=generator)*160-80))
        routine = native.expf
        routine.argtypes, routine.restype = [ctypes.c_float], ctypes.c_float
        expected = torch.tensor([routine(value) for value in x.tolist()])
        actual = _exp(x)
        assert_stage(actual, expected, 16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.count_nonzero(torch.exp(x) != expected) > 0
        packets["reference"], packets["actual"] = expected.numpy(), actual.numpy()
    else:
        x = torch.exp(torch.linspace(-12., 12., 200001))
        routine = native.powf
        routine.argtypes, routine.restype = [ctypes.c_float, ctypes.c_float], ctypes.c_float
        samples = x.tolist()
        for index, exponent in enumerate((torch.tensor(287.)/1004.5, torch.tensor(717.5)/1004.5, torch.tensor(2.))):
            scalar_exponent = float(exponent)
            expected = torch.tensor([routine(value, scalar_exponent) for value in samples])
            actual = _power(x, exponent)
            assert_stage(actual, expected, 16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.count_nonzero((x ** exponent) != expected) > 0
            packets[f"exponent_{index}"] = exponent.numpy()
            packets[f"reference_{index}"], packets[f"actual_{index}"] = expected.numpy(), actual.numpy()
    np.savez(tmp_path / (operation + ".npz"), input=x.numpy(), **packets)


@pytest.mark.parametrize("operation", ("sincos", "exp", "power"))
def test_float32_primitive_continuous_derivatives(tmp_path, operation):
    x = torch.linspace(-2.9, 2.9, 93, requires_grad=True)
    if operation == "power":
        x = torch.exp(x).requires_grad_()
    direction = torch.linspace(.1, .7, x.numel())
    if operation == "sincos":
        evaluate = lambda value: torch.stack(_sincos(value))
        expected = torch.stack((torch.cos(x.double()), -torch.sin(x.double()))) * direction
    elif operation == "exp":
        evaluate = _exp
        expected = torch.exp(x.double()) * direction
    else:
        exponent = x.new_tensor(287.)/1004.5
        evaluate = lambda value: _power(value, exponent)
        expected = exponent.double() * x.double() ** (exponent.double()-1) * direction
        wide = x.double().requires_grad_()
        assert torch.autograd.gradcheck(lambda value: _single_power(value, exponent.double()), (wide,),
                                        check_forward_ad=True)
    value, tangent = torch.func.jvp(evaluate, (x,), (direction,))
    # Reuse the prior float32 surface-factor bound for its shared exp polynomial.
    # Preserve failure of the tighter float64-path bound instead of claiming it.
    torch.testing.assert_close(tangent.double(), expected, rtol=5e-6, atol=1e-8)
    if operation == "sincos":
        torch.testing.assert_close(tangent.double(), expected, rtol=2e-6, atol=2e-8)
    else:
        with pytest.raises(AssertionError):
            torch.testing.assert_close(tangent.double(), expected, rtol=2e-6, atol=2e-8)
    np.savez(tmp_path / (operation + "-derivative.npz"), input=x.detach().numpy(),
             direction=direction.numpy(), actual=tangent.detach().numpy(), expected=expected.detach().numpy())
    assert torch.linalg.vector_norm(tangent) > 0
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(tangent).double(), expected, rtol=2e-6, atol=2e-8)
    weights = torch.linspace(.3, 1.1, value.numel()).reshape_as(value)
    gradient, = torch.autograd.grad((value*weights).sum(), x)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close((gradient*direction).sum(), (tangent*weights).sum(), rtol=2e-6, atol=2e-8)
