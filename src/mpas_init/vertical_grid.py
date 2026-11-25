"""MPAS global terrain smoothing and terrain-following vertical coordinates.

Arrays use cell-major storage. Connectivity uses zero-based indices, with zero
in unused slots. The edge count masks those slots before ordered summation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mpas_init.arithmetic import _single_power, _sincos

@dataclass(frozen=True)
class GlobalConnectivity:
    cells_on_cell: Tensor
    edges_on_cell: Tensor
    n_edges_on_cell: Tensor
    dc_edge: Tensor
    dv_edge: Tensor
    cells_on_edge: Tensor

    def laplacian(self, value: Tensor) -> Tensor:
        result = torch.zeros_like(value)
        for j in range(self.cells_on_cell.shape[1]):
            edge = self.edges_on_cell[:, j]
            term = (self.dv_edge[edge] / self.dc_edge[edge]
                    * (value[..., self.cells_on_cell[:, j]] - value))
            result = result + torch.where(self.n_edges_on_cell > j, term, 0.0)
        return result

    def mean_dc(self) -> Tensor:
        total = torch.zeros_like(self.n_edges_on_cell, dtype=self.dc_edge.dtype)
        for j in range(self.edges_on_cell.shape[1]):
            total = total + torch.where(self.n_edges_on_cell > j,
                                        self.dc_edge[self.edges_on_cell[:, j]], 0.0)
        return total / self.n_edges_on_cell.to(total.dtype)


def smooth_terrain(terrain: Tensor, mesh: GlobalConnectivity, passes: int = 1) -> Tensor:
    """Apply the original fourth-order smoother, including its ocean-zero mask."""
    for _ in range(passes):
        first = terrain + 0.216 * torch.where(terrain != 0, mesh.laplacian(terrain), 0.0)
        terrain = first - 0.216 * torch.where(first != 0, mesh.laplacian(first), 0.0)
    return terrain


def tc_interfaces(n_levels: int, ztop: Tensor) -> Tensor:
    """Return the reference 2014 TC interface heights in the input precision."""
    als, alt, zetal = (0.075, 1.70, 0.75) if n_levels >= 55 else (0.075, 1.23, 0.31)
    k = torch.arange(n_levels + 1, dtype=ztop.dtype, device=ztop.device)
    # Typed constants preserve default-real arithmetic in either reference build.
    als, alt, zetal = (ztop.new_tensor(x) for x in (als, alt, zetal))
    dz = ztop / n_levels
    x = k / n_levels
    zl = 1.0 - alt * (1.0 - zetal)
    lower = (als * k / n_levels
             + (3.0 * (1.0 - alt) + 2.0 * (alt - als) * zetal)
             * (k * dz / (ztop * zetal)) ** 2
             - (2.0 * (1.0 - alt) + (alt - als) * zetal)
             * (k * dz / (ztop * zetal)) ** 3) * ztop
    upper = (zl + alt * (x - zetal)) * ztop
    return torch.where(x < zetal, lower, upper)


def reference_metrics(zw: Tensor, interface_projection: str = "linear_interpolation") -> dict[str, Tensor]:
    """Reference-height metrics; the unused first dzu/fzm/fzp entries are zero."""
    dzw = zw[1:] - zw[:-1]
    dzu = 0.5 * (dzw[1:] + dzw[:-1])
    fzp, fzm = 0.5 * dzw[1:] / dzu, 0.5 * dzw[:-1] / dzu
    if interface_projection == "layer_integral":
        fzp, fzm = fzm, fzp
    elif interface_projection != "linear_interpolation":
        raise ValueError(f"unsupported interface projection: {interface_projection}")
    cof1 = (2.0 * dzu[0] + dzu[1]) / (dzu[0] + dzu[1]) * dzw[0] / dzu[0]
    cof2 = dzu[0] / (dzu[0] + dzu[1]) * dzw[0] / dzu[1]
    pad = zw.new_zeros(1)
    return {"rdzw": 1.0 / dzw, "dzu": torch.cat((pad, dzu)),
            "fzm": torch.cat((pad, fzm)), "fzp": torch.cat((pad, fzp)),
            "cf1": fzp[0] + cof1, "cf2": fzm[0] - cof1 - cof2, "cf3": cof2}


def _single_surface_factor(x: Tensor) -> Tensor:
    """Reference min(powf(x, 2.0), 1) for nonnegative surface-height ratios.

    The consumed scalar libSystem powf uses wide log/exp polynomials, not a
    float32 square. Preserve its reduction and coefficients (P6/P29). Table
    centers and powers are analytic here; no sampled oracle values are used.
    """
    active = (x > 0) & (x < 1)
    safe = torch.where(active, x, 1.0)
    result = _single_power(safe, x.new_tensor(2.0))
    result = torch.where(active, result, torch.where(x == 0, 0.0, 1.0))
    return result.to(x.dtype)


def vertical_grid(terrain: Tensor, mesh: GlobalConnectivity, zw: Tensor, *,
                  smooth_surfaces: bool = True, n_smooth: int = 30,
                  dzmin: float = 0.3) -> dict[str, Tensor]:
    """Build heights, inverse Jacobians, and normal slopes from smoothed terrain."""
    angle = 0.5 * zw.new_tensor(torch.pi) * zw / 30000.0
    cosine = _sincos(angle)[1]
    square = cosine * cosine
    ah = torch.where(zw < 30000.0, square * square * square, 0.0)
    surfaces = [terrain]
    if smooth_surfaces:
        sm0 = torch.maximum(terrain.new_tensor(0.01),
                            0.125 * torch.minimum(terrain.new_tensor(1.0), 3000.0 / mesh.mean_dc()))
        ratio = 3.0 * zw / zw[-1]
        factor = (_single_surface_factor(ratio) if zw.dtype == torch.float32
                  else torch.minimum(ratio ** 2.0, zw.new_tensor(1.0)))
        for k in range(1, zw.shape[0] - 1):
            current = surfaces[-1]
            sm = sm0 * factor[k]
            for _ in range(n_smooth + k + 1):
                proposed = current + sm * mesh.laplacian(current)
                thickness = (zw[k] + ah[k] * proposed) - (zw[k-1] + ah[k-1] * surfaces[-1])
                current = torch.where(thickness > dzmin * (zw[k] - zw[k-1]), proposed, current)
            # The source sets every surface from the first pure-height interface to zero.
            surfaces.append(torch.where(zw[k] < 30000.0, current, 0.0))
        surfaces.append(torch.zeros_like(terrain))
    else:
        surfaces.extend([terrain] * (zw.shape[0] - 1))
    hx = torch.stack(surfaces, dim=-1)
    zgrid = zw + ah * hx
    zz = (zw[1:] - zw[:-1]) / (zgrid[..., 1:] - zgrid[..., :-1])
    c1, c2 = mesh.cells_on_edge[:, 0], mesh.cells_on_edge[:, 1]
    zxu = 0.5 * (zgrid[..., c2, :-1] - zgrid[..., c1, :-1]
                 + zgrid[..., c2, 1:] - zgrid[..., c1, 1:]) / mesh.dc_edge[:, None]
    return {"zgrid": zgrid, "zz": zz, "zxu": zxu, **reference_metrics(zw)}
