"""Classic native MPAS subgrid-orography coordinates and statistics."""

import torch
from torch import Tensor

from mpas_init.vertical_grid import GlobalConnectivity


CLASSIC_FIELDS = ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
                  "gwd_theta", "gwd_gamma", "gwd_sigma", "gwd_elvmax")
RADIUS = 6371229.0
POINTS_PER_DEGREE = 120.0
SG_DELTA = 2.0 * torch.pi * RADIUS / (360.0 * POINTS_PER_DEGREE)


def box_size(latitude: Tensor, diameter: Tensor) -> tuple[Tensor, Tensor]:
    """Original 30-arc-second footprint dimensions; latitude is in radians."""
    cosine = torch.cos(latitude)
    boundary = ((2.0 * POINTS_PER_DEGREE * diameter * 180.0)
                / (43200.0 * torch.pi * RADIUS))
    safe = torch.where(cosine > boundary, cosine, 1.0)
    nx = torch.where(cosine > boundary,
                     torch.ceil((180.0 * diameter * POINTS_PER_DEGREE) / (torch.pi * RADIUS * safe)), 21600.0)
    ny = torch.ceil((180.0 * diameter * POINTS_PER_DEGREE) / (torch.pi * RADIUS))
    return nx.to(torch.int64), ny.to(torch.int64)


def box_indices(latitude: Tensor, longitude: Tensor, nx: int, ny: int) -> tuple[Tensor, Tensor, Tensor]:
    """Zero-based USGS indices and zonal spacing, with native pole reflection."""
    if latitude.shape != longitude.shape or min(nx, ny) < 1:
        raise ValueError("GWD box indices require matching coordinates and positive dimensions")
    degrees = 180.0 / torch.pi
    x = (longitude * degrees - (-180.0)) * POINTS_PER_DEGREE
    y = (latitude * degrees - (-90.0)) * POINTS_PER_DEGREE
    ic = torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5)).to(torch.int64) + 1
    jc = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5)).to(torch.int64) + 1
    ic = torch.where(ic <= 0, ic + 43200, torch.where(ic > 43200, ic - 43200, ic))
    i = torch.arange(1, nx + 1, device=latitude.device)[None, :] - nx // 2 + ic[..., None, None]
    j = torch.arange(1, ny + 1, device=latitude.device)[:, None] - ny // 2 + jc[..., None, None]
    south = j <= 0
    j = torch.where(south, -j + 1, j)
    i = i + torch.where(south, 21600, 0)
    north = j > 21600
    j = torch.where(north, 21600 - (j - 21600 - 1), j)
    i = i + torch.where(north, 21600, 0)
    shape = latitude.shape + (ny, nx)
    rows, columns = (j - 1).expand(shape), (i - 1).remainder(43200).expand(shape)
    sg_lat = (-90.0 + (j.to(latitude.dtype) - 1.0 + 0.5) / POINTS_PER_DEGREE) / degrees
    dxm = (SG_DELTA * torch.cos(sg_lat)).expand(shape)
    return rows, columns, dxm


def _ordered_sum(value: Tensor) -> Tensor:
    # A scalar view would retain the complete footprint's cumulative-sum buffer.
    return value.flatten(-2).cumsum(-1)[..., -1].clone()


def _sqrt(value: Tensor) -> Tensor:
    positive = value > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, value, 1.0)), 0.0)


def classic_statistics(box: Tensor, landuse: Tensor, dxm: Tensor) -> dict[str, Tensor]:
    """All fourteen classic statistics before the mesh's isolated-point smoothing.

    The last two dimensions are latitude then longitude. Leading batch axes are
    permitted. USGS category 16 is water. Selection boundaries remain discrete.
    """
    if box.ndim < 2 or min(box.shape[-2:]) < 4 or box.shape != landuse.shape or box.shape != dxm.shape:
        raise ValueError("classic GWD requires matching terrain, USGS, and spacing arrays, at least 4 by 4")
    ny, nx = box.shape[-2:]
    n = nx * ny
    mean = _ordered_sum(box) / n
    centered = box - mean[..., None, None]
    s2 = _ordered_sum(centered * centered)
    variance = s2 / n
    del centered
    stddev = _sqrt(variance)
    land = landuse != 16
    count = land.sum((-2, -1))
    mean_land = _ordered_sum(torch.where(land, box, 0.0)) / count.clamp_min(1)
    mean_water = _ordered_sum(torch.where(~land, box, 0.0)) / (n - count).clamp_min(1)
    dominant = count.to(box.dtype) / n >= 0.5
    oro = torch.where(dominant, mean_land, mean_water)
    square = (box - oro[..., None, None]) ** 2
    s4 = _ordered_sum(square * square)
    del square, land
    con_active = (stddev >= 1.0) & dominant
    con = torch.where(con_active, s4 / torch.where(con_active, variance * variance * n, 1.0), 0.0)
    i = torch.arange(1, nx + 1, device=box.device)[None, :]
    j = torch.arange(1, ny + 1, device=box.device)[:, None]
    diagonal = torch.floor(i.to(box.dtype) * (ny / nx) + 0.5).to(torch.int64)
    higher = box > mean[..., None, None]
    result = {"var2d": stddev, "con": con}
    for number, upwind in enumerate((i <= nx // 2, j <= ny // 2, diagonal < ny - j, diagonal < j), 1):
        nu = (higher & upwind).sum((-2, -1))
        nd = (higher & ~upwind).sum((-2, -1))
        result[f"oa{number}"] = (nu - nd).to(box.dtype) / (nu + nd).clamp_min(1)
    del higher, upwind
    critical = 1116.2 - 0.878 * stddev
    high = box > critical[..., None, None]
    masks = ((j >= ny // 4) & (j <= 3 * ny // 4),
             (i >= nx // 4) & (i <= 3 * nx // 4),
             ((j <= ny // 2) & (i <= nx // 2)) | ((j > ny // 2) & (i > nx // 2)),
             ((j > ny // 2) & (i <= nx // 2)) | ((j <= ny // 2) & (i > nx // 2)))
    for number, region in enumerate(masks, 1):
        region = region.expand(ny, nx)
        result[f"ol{number}"] = (high & region).sum((-2, -1)).to(box.dtype) / region.sum()
    del masks, region, high
    xf = (box[..., 1:-1, 2:] - box[..., 1:-1, :-2]) / (2.0 * dxm[..., 1:-1, 1:-1])
    yf = (box[..., 2:, 1:-1] - box[..., :-2, 1:-1]) / (2.0 * SG_DELTA)
    ni = (nx - 2) * (ny - 2)
    hx2 = _ordered_sum(xf * xf) / ni
    hy2 = _ordered_sum(yf * yf) / ni
    hxy = _ordered_sum(xf * yf) / ni
    del xf, yf
    hk, hl = 0.5 * (hx2 + hy2), 0.5 * (hx2 - hy2)
    hlp = _sqrt(hl * hl + hxy * hxy)
    angle_defined = (hxy != 0) | (hl != 0)
    result["gwd_theta"] = 0.5 * torch.atan2(torch.where(angle_defined, hxy, 0.0),
                                           torch.where(angle_defined, hl, 1.0))
    anisotropy_defined = (hk + hlp > 0) & (hk - hlp >= 0)
    ratio = torch.where(anisotropy_defined, hk - hlp, 0.0) / torch.where(anisotropy_defined, hk + hlp, 1.0)
    result["gwd_gamma"] = _sqrt(ratio)
    result["gwd_sigma"] = _sqrt(hk + hlp)
    result["gwd_elvmax"] = box.amax((-2, -1))
    return result


def smooth_classic_variance(variance: Tensor, landmask: Tensor, mesh: GlobalConnectivity) -> tuple[Tensor, Tensor]:
    """Native isolated-point smoothing on a closed triangular cell graph.

    Eligible points cannot be adjacent on that graph: their common third cell
    would need opposite labels simultaneously. The I/O boundary verifies this
    precondition. Thus native ascending-cell updates read only unchanged neighbors.
    """
    total = torch.zeros_like(variance)
    land = torch.zeros_like(landmask)
    for j in range(mesh.cells_on_cell.shape[1]):
        neighbor = mesh.cells_on_cell[:, j]
        active = mesh.n_edges_on_cell > j
        total = total + torch.where(active, variance[neighbor], 0.0)
        land = land + torch.where(active, landmask[neighbor], 0)
    isolated = (((land == mesh.n_edges_on_cell) & (landmask == 0))
                | ((land == 0) & (landmask == 1)))
    return torch.where(isolated, total / mesh.n_edges_on_cell.to(variance.dtype), variance), isolated
