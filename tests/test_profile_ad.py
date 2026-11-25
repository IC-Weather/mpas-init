"""Full-profile local AD and bounded independent original-initializer FD checks."""


import pytest
import torch


def assert_derivative(actual, expected):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=2.e-6, atol=2.e-8)
    norm = torch.linalg.vector_norm(expected)
    error = torch.linalg.vector_norm(actual - expected)
    if norm > 0:
        assert error / norm < 2.e-6
    else:
        assert torch.equal(actual, expected)
    return {"max_abs": float((actual - expected).abs().max()),
            "relative_norm": float(error / norm) if norm > 0 else 0.}


@pytest.mark.parametrize("defect", ("zero", "nonfinite", "one_element", "zero_reference"))
def test_complete_derivative_predicate_rejects_bad_values(defect):
    expected = torch.tensor([1., 2., 3.], dtype=torch.float64)
    actual = expected.clone()
    if defect == "zero":
        actual.zero_()
    elif defect == "nonfinite":
        actual[0] = torch.nan
    elif defect == "one_element":
        actual[1] += 1.e-3
    else:
        expected.zero_()
        actual.zero_()
        actual[1] = 1.e-20
    with pytest.raises(AssertionError):
        assert_derivative(actual, expected)
