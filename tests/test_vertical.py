import torch

from mpas_init.vertical import interpolate_height


def test_linear_constant_and_lapse_rate():
    height = torch.tensor([[10.0, 100.0, 1000.0]], dtype=torch.float64)
    value = 280.0 - 0.0065 * height
    target = height.new_tensor([[-10.0, 10.0, 37.5, 100.0, 600.0, 1000.0, 1100.0]])
    expected = 280.0 - 0.0065 * target
    torch.testing.assert_close(interpolate_height(height, value, target, "linear"), expected, rtol=1e-14, atol=1e-15)
    constant = interpolate_height(height, value, target)
    assert constant[0, 0] == value[0, 0]
    assert constant[0, -1] == value[0, -1]
    lapse = interpolate_height(height, value, target, "lapse-rate")
    torch.testing.assert_close(lapse[:, :-2], expected[:, :-2], rtol=1e-14, atol=1e-15)
    assert torch.isnan(lapse[:, -2:]).all()


def test_batched_height_and_value_derivatives():
    height = torch.tensor([[10.0, 100.0, 1000.0], [12.0, 110.0, 1200.0]], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([[283.0, 272.0, 260.0], [290.0, 278.0, 265.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[20.0, 600.0], [50.0, 700.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda h, v, t: interpolate_height(h, v, t), (height, value, target))
