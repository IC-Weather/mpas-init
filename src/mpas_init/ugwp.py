"""Native UGWP terrain statistics and large-scale background interpolation."""

import torch
from torch import Tensor

from mpas_init.gwd import _ordered_sum, _sqrt


UGWP_BASE_FIELDS = ('var2d', 'con', 'oa1', 'oa2', 'oa3', 'oa4', 'ol1', 'ol2', 'ol3', 'ol4')
UGWP_FIELDS = tuple(name + suffix for suffix in ('ls', 'ss') for name in UGWP_BASE_FIELDS)
HGT_MISSING = 1.0e10


def ugwp_fine_coordinates(nx: int, ny: int, *, device=None, dtype=torch.float64) -> tuple[Tensor, Tensor]:
    """Original pixel-center coordinates, latitude then longitude, in radians."""
    latitude = (-90.0 + (180.0 / ny) * (torch.arange(1, ny + 1, device=device, dtype=dtype) - 0.5)) * torch.pi / 180.0
    longitude = (-180.0 + (360.0 / nx) * (torch.arange(1, nx + 1, device=device, dtype=dtype) - 0.5)) * torch.pi / 180.0
    return latitude, longitude


def ugwp_block_geometry(latitude: Tensor, longitude: Tensor, diameter: Tensor, *, nx: int, ny: int,
                        fine_coordinates=None):
    """Native nine-block geometry with the approved full-longitude polar extension.

    Widths spanning a full ring use every longitude once, ordered from -pi.
    Their three zonal blocks coincide; distinct centers keep background slopes
    well-defined and zonally constant. Non-polar wrapping remains unchanged.
    Absent polar neighbor rows retain native -999 indices and unmodified centers.
    """
    from mpas_init.gwd import RADIUS

    lat_fine, lon_fine = (ugwp_fine_coordinates(nx, ny, device=latitude.device, dtype=latitude.dtype)
                          if fine_coordinates is None else fine_coordinates)
    width_lat = torch.sqrt(diameter * diameter) / RADIUS
    cosine = torch.cos(latitude)
    full_ring = width_lat >= 2 * torch.pi * cosine
    width_lon = torch.sqrt(diameter * diameter) / (RADIUS * torch.where(full_ring, 1.0, cosine))
    width_lon = torch.where(full_ring, 2 * torch.pi, width_lon)
    lon = torch.where(longitude > torch.pi, longitude - 2 * torch.pi, longitude)
    lon = torch.where(full_ring, 0.0, lon)
    offsets = torch.arange(-1, 2, device=latitude.device, dtype=latitude.dtype)
    lat_blocks = latitude[..., None] + offsets * width_lat[..., None]
    lon_blocks = lon[..., None] + offsets * width_lon[..., None]
    west, east = lon_blocks - 0.5 * width_lon[..., None], lon_blocks + 0.5 * width_lon[..., None]
    for _ in range(2):
        west = torch.where(west < -torch.pi, west + 2 * torch.pi, west)
        west = torch.where(west > torch.pi, west - 2 * torch.pi, west)
        east = torch.where(east < -torch.pi, east + 2 * torch.pi, east)
        east = torch.where(east > torch.pi, east - 2 * torch.pi, east)
    start_i = torch.where(west > lon_fine[-1], 1, torch.searchsorted(lon_fine, west.contiguous()) + 1)
    end_i = torch.where((east < lon_fine[0]) | (east >= lon_fine[-1]), nx,
                        torch.searchsorted(lon_fine, east.contiguous(), right=True))
    start_i = torch.where(full_ring[..., None], 1, start_i)
    end_i = torch.where(full_ring[..., None], nx, end_i)
    south, north = lat_blocks - 0.5 * width_lat[..., None], lat_blocks + 0.5 * width_lat[..., None]
    start_j = (torch.searchsorted(lat_fine, south.contiguous()) + 1).clamp_max(ny)
    end_j = torch.searchsorted(lat_fine, north.contiguous(), right=True).clamp_min(1)
    start_j = torch.where(south.abs() > 0.5 * torch.pi, -999, start_j)
    end_j = torch.where(north.abs() > 0.5 * torch.pi, -999, end_j)
    center = offsets == 0
    shave_south = (start_j == -999) & ((end_j != -999) | center)
    start_j = torch.where(shave_south, 1, start_j)
    lat_blocks = torch.where(shave_south,
                              0.5 * (lat_fine[0] + lat_fine[(end_j - 1).clamp(0, ny - 1)]), lat_blocks)
    shave_north = (end_j == -999) & ((start_j != -999) | center)
    end_j = torch.where(shave_north, ny, end_j)
    lat_blocks = torch.where(shave_north,
                              0.5 * (lat_fine[(start_j - 1).clamp(0, ny - 1)] + lat_fine[-1]), lat_blocks)
    return lat_blocks, lon_blocks, start_i, end_i, start_j, end_j


def ugwp_statistics(box: Tensor, *, small_scale: bool) -> dict[str, Tensor]:
    """Ten native statistics of raw small-scale or detrended large-scale terrain.

    Last axes are latitude then longitude. Leading batch axes are supported.
    Terrain selection and directional counts have piecewise-constant derivatives.
    """
    if box.ndim < 2 or min(box.shape[-2:]) < 1:
        raise ValueError('UGWP statistics require a nonempty two-dimensional terrain footprint')
    ny, nx = box.shape[-2:]
    n = nx * ny
    active = (box.abs() > (0.1 if small_scale else 0.001)).any(dim=(-2, -1))
    mean = _ordered_sum(box) / n
    square = (box - mean[..., None, None]).square()
    variance = _ordered_sum(square) / n
    convexity_active = variance.abs() >= 1.0e-5
    convexity = _ordered_sum(square.square()) / torch.where(convexity_active, variance.square() * n, 1.0)
    del square
    values = {'var2d': _sqrt(variance), 'con': convexity.clamp_max(10.0)}
    values['con'] = torch.where(convexity_active, values['con'], 0.0)
    i = torch.arange(1, nx + 1, device=box.device)[None, :]
    j = torch.arange(1, ny + 1, device=box.device)[:, None]
    diagonal = torch.floor(i.to(box.dtype) * (ny / nx) + 0.5).long()
    higher = box > mean[..., None, None]
    halves = ((i <= nx // 2 + nx % 2, i >= nx // 2 + 1),
              (j <= ny // 2 + ny % 2, j >= ny // 2 + 1),
              (diagonal <= ny - j + 1, diagonal >= ny - j + 1),
              (diagonal <= j, diagonal >= j))
    for number, (upwind, downwind) in enumerate(halves, 1):
        nu = (higher & upwind).sum((-2, -1))
        nd = (higher & downwind).sum((-2, -1))
        values[f'oa{number}'] = (nu - nd).to(box.dtype) / (nu + nd).clamp_min(1)
    del halves, upwind, downwind
    bands = ((j >= max(ny // 4, 1)) & (j <= 3 * ny // 4),
             (i >= max(nx // 4, 1)) & (i <= 3 * nx // 4),
             ((j <= ny // 2) & (i <= nx // 2)) | ((j > ny // 2) & (i > nx // 2)),
             ((j > ny // 2) & (i <= nx // 2)) | ((j <= ny // 2) & (i > nx // 2)))
    for number, band in enumerate(bands, 1):
        band = band.expand(ny, nx)
        values[f'ol{number}'] = (higher & band).sum((-2, -1)).to(box.dtype) / band.sum().clamp_min(1)
    suffix = 'ss' if small_scale else 'ls'
    return {name + suffix: torch.where(active, values[name], 0.0) for name in UGWP_BASE_FIELDS}


def ugwp_background(latitude: Tensor, longitude: Tensor, lat_blocks: Tensor,
                    lon_blocks: Tensor, heights: Tensor) -> Tensor:
    """Original HGT_interpolate, including seam alignment and polar-row extrapolation.

    Block axes are latitude then longitude. Coordinates are radians. Missing block
    rows use the native 1e10 sentinel; north-row handling has original precedence.
    """
    batch = lat_blocks.shape[:-1]
    if (lat_blocks.shape != batch + (3,) or lon_blocks.shape != lat_blocks.shape
            or heights.shape != batch + (3, 3)):
        raise ValueError('UGWP background requires matching batches of three coordinates and 3 by 3 heights')
    latitude, longitude = torch.broadcast_tensors(latitude, longitude)
    if latitude.shape[:len(batch)] != batch:
        raise ValueError('UGWP background pixel coordinates must have matching leading batch axes')
    trailing = (1,) * (latitude.ndim - len(batch))
    middle_lat = lat_blocks[..., 1].reshape(batch + trailing)
    middle_lon = lon_blocks[..., 1].reshape(batch + trailing)
    # Advanced indices select each member's own background without changing the
    # original two latitude interpolations followed by longitude interpolation.
    member = tuple(torch.arange(size, device=heights.device).reshape(
        (1,) * axis + (size,) + (1,) * (latitude.ndim - axis - 1))
        for axis, size in enumerate(batch))
    seam = (longitude - middle_lon).abs() > torch.pi
    lon = torch.where(seam & (longitude > 0), longitude - 2 * torch.pi,
                      torch.where(seam & (longitude < 0), longitude + 2 * torch.pi, longitude))
    north_missing = (heights[..., 2, :] == HGT_MISSING).any(-1).reshape(batch + trailing)
    south_missing = (heights[..., 0, :] == HGT_MISSING).any(-1).reshape(batch + trailing)
    row = torch.where(north_missing, 0, torch.where(south_missing, 1, (latitude >= middle_lat).long()))
    column = (lon >= middle_lon).long()

    def interpolate(x, x1, x2, y1, y2):
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)

    south, north = lat_blocks[member + (row,)], lat_blocks[member + (row + 1,)]
    west = interpolate(latitude, south, north,
                       heights[member + (row, column)], heights[member + (row + 1, column)])
    east = interpolate(latitude, south, north,
                       heights[member + (row, column + 1)], heights[member + (row + 1, column + 1)])
    return interpolate(lon, lon_blocks[member + (column,)], lon_blocks[member + (column + 1,)], west, east)
