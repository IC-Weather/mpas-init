"""Native edge-normal and terrain-consistent vertical wind initialization."""

import torch
from torch import Tensor

from mpas_init.arithmetic import _sincos
from mpas_init.horizontal import HorizontalGrid, remap_atmosphere
from mpas_init.vertical import interpolate_height
from mpas_init.vertical_grid import GlobalConnectivity


def pressure_level_wind(fields: dict[str, Tensor], latitude: Tensor, longitude: Tensor,
                        lat_cell: Tensor, lon_cell: Tensor, lat_edge: Tensor,
                        lon_edge: Tensor, angle_edge: Tensor, cells_on_edge: Tensor,
                        zgrid: Tensor, *, grid_type: str = "regular_latlon",
                        field_grids: dict[str, HorizontalGrid] | None = None,
                        row_counts: Tensor | None = None,
                        missing_fields: set[str] | frozenset[str] = frozenset()) -> Tensor:
    """Project earth-relative winds before MPAS physical-height interpolation."""
    def horizontal(names, lat, lon):
        return remap_atmosphere(fields, names, latitude, longitude, lat, lon,
                                grid_type=grid_type, field_grids=field_grids, row_counts=row_counts, missing_fields=missing_fields)

    height_cell = horizontal(("geopotential_height",), lat_cell, lon_cell)["geopotential_height"]
    winds = horizontal(("eastward_wind", "northward_wind"), lat_edge, lon_edge)
    east, north = winds["eastward_wind"], winds["northward_wind"]
    sine, cosine = _sincos(angle_edge[:, None])
    normal = cosine * east + sine * north
    c1, c2 = cells_on_edge[:, 0], cells_on_edge[:, 1]
    height, order = torch.sort(0.5 * (height_cell[c1] + height_cell[c2]), dim=-1)
    target = 0.25 * (zgrid[c1, :-1] + zgrid[c1, 1:] + zgrid[c2, :-1] + zgrid[c2, 1:])
    return interpolate_height(height, torch.gather(normal, -1, order), target, "constant")


def terrain_flux_metrics(zgrid: Tensor, mesh: GlobalConnectivity, area_cell: Tensor,
                         deriv_two: Tensor, order: int = 3) -> dict[str, Tensor]:
    """Build MPAS edge-side metrics from geometry-derived polynomial stencils."""
    if order not in (2, 3, 4):
        raise ValueError("theta advection order must be 2, 3, or 4")
    c1, c2 = mesh.cells_on_edge[:, 0], mesh.cells_on_edge[:, 1]
    heights = zgrid[:, :-1]
    edge = 0.5 * (heights[c1] + heights[c2])
    edge3 = torch.zeros_like(edge)
    if order != 2:
        derivatives = []
        for side, cell in enumerate((c1, c2)):
            value = deriv_two[:, side, 0, None] * heights[cell]
            for j in range(mesh.cells_on_cell.shape[1]):
                term = deriv_two[:, side, j + 1, None] * heights[mesh.cells_on_cell[cell, j]]
                value = value + torch.where((mesh.n_edges_on_cell[cell] > j)[:, None], term, 0.0)
            derivatives.append(value)
        edge = edge - mesh.dc_edge[:, None].square() * (derivatives[0] + derivatives[1]) / 12.0
        if order == 3:
            edge3 = -mesh.dc_edge[:, None].square() * (derivatives[0] - derivatives[1]) / 12.0
    zb = torch.stack([(edge - heights[c]) * mesh.dv_edge[:, None] / area_cell[c, None]
                      for c in (c1, c2)], dim=1)
    zb3 = torch.stack([edge3 * mesh.dv_edge[:, None] / area_cell[c, None]
                       for c in (c1, c2)], dim=1)
    pad = zb.new_zeros((*zb.shape[:-1], 1))
    return {"zb": torch.cat((zb, pad), dim=-1), "zb3": torch.cat((zb3, pad), dim=-1)}


def vertical_wind(u: Tensor, rho_zz: Tensor, zz: Tensor, fzm: Tensor, fzp: Tensor,
                   zb: Tensor, zb3: Tensor, mesh: GlobalConnectivity, *,
                   order: int = 3, coefficient: float = 0.25) -> Tensor:
    """Cancel metric-induced mass flux, with zero bottom and top velocity."""
    c1, c2 = mesh.cells_on_edge[:, 0], mesh.cells_on_edge[:, 1]
    ru = u * 0.5 * (rho_zz[c1] + rho_zz[c2])
    rw = torch.zeros_like(rho_zz[:, 1:])
    metric = fzm[1:] * zz[:, 1:] + fzp[1:] * zz[:, :-1]
    cells = torch.arange(rho_zz.shape[0], device=u.device)
    for j in range(mesh.edges_on_cell.shape[1]):
        edge = mesh.edges_on_cell[:, j]
        side = (cells != c1[edge]).to(torch.int64)
        orientation = torch.where(side == 0, -1.0, 1.0).to(u.dtype)[:, None]
        flux = fzm[1:] * ru[edge, 1:] + fzp[1:] * ru[edge, :-1]
        active = (mesh.n_edges_on_cell > j)[:, None]
        term = metric * zb[edge, side, 1:-1] * flux
        rw = rw + torch.where(active, orientation * term, 0.0)
        if order == 3:
            flux_sign = torch.where(torch.signbit(ru[edge, 1:]), -1.0, 1.0).to(u.dtype)
            term3 = flux_sign * coefficient * metric * zb3[edge, side, 1:-1] * flux
            rw = rw - torch.where(active, orientation * term3, 0.0)
    interior = rw / (fzp[1:] * rho_zz[:, :-1] + fzm[1:] * rho_zz[:, 1:])
    pad = interior.new_zeros((interior.shape[0], 1))
    return torch.cat((pad, interior, pad), dim=-1)
