"""Consumed scalar math for float32 initialization; tensor arithmetic only."""

import torch
from torch import Tensor


def _scaled_exp(power: Tensor, nearest: Tensor) -> Tensor:
    remainder = power - nearest
    scale = torch.ldexp(torch.exp2(nearest.remainder(128) / 128),
                        torch.div(nearest, 128, rounding_mode="floor").to(torch.int32))
    return scale + scale * ((1.4662262378543872e-05 * remainder + 0.005415217308896754) * remainder)


def _power(base: Tensor, exponent: Tensor) -> Tensor:
    """Original positive-base real power in either working precision."""
    return _single_power(base, exponent) if base.dtype == torch.float32 else base ** exponent


def _single_power(base: Tensor, exponent: Tensor) -> Tensor:
    """Darwin scalar powf arithmetic, also evaluable wide for derivative tests.

    Preserve the consumed wide log/exp reduction and coefficients (P6/P29).
    Reduction centers and powers are analytic, not sampled oracle outputs.
    """
    torch._assert_async((base > 0).all(), "real-power base must be positive")
    wide = base.to(torch.float64)
    mantissa, binary_exponent = torch.frexp(wide)
    k = binary_exponent - (mantissa < 0.701171875).to(binary_exponent.dtype)
    z = torch.ldexp(wide, -k)
    index = torch.floor(torch.where(z < 1, (z - 0.701171875) * 256,
                                    (z - 1) * 128 + 76.5))
    center = torch.where(index < 76, (index + 180) / 256, (index + 52) / 128)
    r = z * (1 / center) - 1
    logarithm = (((-46.16680486880053 * r + 61.55555197023551) * (r*r)
                  + (-92.33248261492298 * r + 184.6649652327124)) * r
                 + (128 * torch.log2(center) + k * 128))
    power = exponent.to(torch.float64) * logarithm
    nearest = -torch.floor(-power + 0.5)
    return _scaled_exp(power, nearest).to(base.dtype)


def _exp(value: Tensor) -> Tensor:
    """Consumed scalar expf for finite initialization arguments within (-128,128)."""
    if value.dtype != torch.float32:
        return torch.exp(value)
    torch._assert_async((value.abs() < 128).all(), "float32 exponential argument must lie within (-128,128)")
    power = value.to(torch.float64) * 184.6649652337873
    return _scaled_exp(power, torch.round(power)).to(value.dtype)


def _sincos(angle: Tensor) -> tuple[Tensor, Tensor]:
    """Consumed scalar sinf/cosf on native mesh angles within [-2*pi,2*pi].

    Preserve the wide reduction and polynomial before float32 rounding (P6).
    Vector float32 sine can change a nearly cancelling edge wind by many ULPs.
    The coefficients describe the polynomial, not sampled reference outputs.
    """
    if angle.dtype != torch.float32:
        return torch.sin(angle), torch.cos(angle)
    torch._assert_async((angle.abs() <= 2 * torch.pi).all(), "native mesh angles must lie within [-2*pi,2*pi]")
    wide = angle.to(torch.float64)
    quadrant = torch.round(wide * (2.0 / torch.pi))
    reduced = wide - quadrant * (torch.pi / 2.0)
    square = reduced * reduced
    cube = reduced * square
    fifth = square * cube
    sine = (fifth * (-0.00019515283191485737 * square + 0.008332160761855575)
            + (cube * -0.16666654609548576 + reduced))
    cosine = (1.0 + square * (square * square * (2.43835673118386e-05 * square - 0.0013886681647984319)
                              + (0.041666620357131375 * square - 0.4999999969447599)))
    sin_sign = torch.where(torch.floor(quadrant / 2).remainder(2) == 0, 1.0, -1.0)
    cos_sign = torch.where(torch.floor((quadrant + 1) / 2).remainder(2) == 0, 1.0, -1.0)
    odd = quadrant.remainder(2) != 0
    return ((torch.where(odd, cosine, sine) * sin_sign).to(angle.dtype),
            (torch.where(odd, sine, cosine) * cos_sign).to(angle.dtype))
