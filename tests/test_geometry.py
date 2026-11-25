import torch

from mpas_init.geometry import _inverse


def test_scaled_pivot_inverse_and_derivatives():
    matrix = torch.tensor([[[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]]], dtype=torch.float64)
    torch.testing.assert_close(matrix @ _inverse(matrix), torch.eye(3, dtype=matrix.dtype)[None])
    assert torch.autograd.gradcheck(_inverse, (matrix.requires_grad_(),), check_forward_ad=True)
