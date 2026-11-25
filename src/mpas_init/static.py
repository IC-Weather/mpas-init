"""Tensor kernels for native MPAS raw-geography terrain preparation."""

from dataclasses import dataclass

import torch
from torch import Tensor

from mpas_init.geotile import GeotileIndex


def pixel_coordinates(index: GeotileIndex, start_x: int, start_y: int,
                      rows: Tensor, columns: Tensor, supersample: int = 1) -> tuple[Tensor, Tensor]:
    """Native zero-based interior subpixel coordinates, returned in radians."""
    if supersample < 1:
        raise ValueError("supersample must be positive")
    rows, columns = rows.to(torch.float64), columns.to(torch.float64)
    # The native supersample starts at the original pixel center, not a recentered footprint.
    latitude = index.known_lat + (rows + supersample * (start_y - 1)) * index.dy / supersample
    longitude = index.known_lon + (columns + supersample * (start_x - 1)) * index.dx / supersample
    return latitude * (torch.pi / 180.0), longitude * (torch.pi / 180.0)


def pixel_cartesian(latitude: Tensor, longitude: Tensor, radius: Tensor) -> Tensor:
    """Original mpas_latlon_to_xyz operation order on a latitude/longitude product."""
    return pixel_cartesian_pairs(latitude[:, None], longitude[None, :], radius).reshape(-1, 3)


def pixel_cartesian_pairs(latitude: Tensor, longitude: Tensor, radius: Tensor) -> Tensor:
    """Apply the same Cartesian operations to broadcast-compatible paired coordinates."""
    x = radius * torch.cos(longitude) * torch.cos(latitude)
    y = radius * torch.sin(longitude) * torch.cos(latitude)
    z = (radius * torch.sin(latitude)).expand_as(x)
    return torch.stack((x, y, z), dim=-1)


def _distance2(delta: Tensor) -> Tensor:
    return (delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]) + delta[..., 2] * delta[..., 2]


def _nearest_owner(points: Tensor, centers: Tensor, ids: Tensor) -> Tensor:
    distance2 = _distance2(points[:, None, :] - centers[None, :, :])
    return ids[torch.argmin(distance2, dim=-1)]


@dataclass(frozen=True)
class NearestCells:
    centers: Tensor
    ids: Tensor

    @classmethod
    def from_centers(cls, centers: Tensor) -> "NearestCells":
        if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
            raise ValueError("cell centers must be a nonempty nCells by 3 tensor")
        ids = torch.arange(centers.shape[0], device=centers.device)
        # The native k-d tree resolves exact-distance ties by lexicographic position.
        for axis in (2, 1, 0):
            ids = ids[torch.argsort(centers[ids, axis], stable=True)]
        return cls(centers[ids], ids)

    def locate(self, points: Tensor) -> Tensor:
        """Exact nearest centers, with a conservative triangle-inequality candidate bound."""
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError("points must be a nonempty nPoints by 3 tensor")
        return _nearest_owner(points, *self._candidates(points))

    def _candidates(self, points: Tensor) -> tuple[Tensor, Tensor]:
        center = 0.5 * (points.amin(dim=0) + points.amax(dim=0))
        radius = torch.linalg.vector_norm(points - center, dim=-1).amax()
        distance2 = _distance2(self.centers - center)
        bound = distance2.amin().sqrt() + 2.0 * radius
        # Outward rounding only adds candidates; it does not alter compared distances.
        bound = bound + 32.0 * torch.finfo(points.dtype).eps * (bound + self.centers.abs().amax())
        keep = distance2 <= bound * bound
        return self.centers[keep], self.ids[keep]


def terrain_accumulation(pixels: Tensor, cells: Tensor, n_cells: int) -> tuple[Tensor, Tensor]:
    """Accumulate native integer terrain pixels without floating-point reduction error."""
    if pixels.dtype != torch.int64 or cells.dtype != torch.int64 or pixels.shape != cells.shape:
        raise ValueError("terrain pixels and cell IDs must be matching int64 vectors")
    total = pixels.new_zeros(n_cells).scatter_add(0, cells, pixels)
    count = pixels.new_zeros(n_cells).scatter_add(0, cells, torch.ones_like(pixels))
    return total, count


def terrain_mean(total: Tensor, count: Tensor, scale_factor: float) -> Tensor:
    """Native float64 average. Uncovered cells remain explicitly nonfinite."""
    value = (total.to(torch.float64) / count.clamp_min(1).to(torch.float64)) * scale_factor
    return torch.where(count > 0, value, torch.full_like(value, torch.nan))


def mapped_patch(raw: Tensor, index: GeotileIndex, locator: NearestCells, radius: Tensor, *,
                 start_x: int, start_y: int, row0: int = 0, col0: int = 0,
                 supersample: int = 1, block_side: int = 32) -> dict[str, Tensor]:
    """Map native integer pixels. Caller handles missing pixels at the decoding boundary.

    Geometry and quantized geotile values are discrete; this does not claim
    derivatives through them. Terrain and categories share this exact mapping.
    """
    if raw.ndim != 2 or min(raw.shape) < 1 or raw.dtype != torch.int64:
        raise ValueError("geography patch must be a nonempty two-dimensional int64 tensor")
    if supersample < 1 or block_side < 1:
        raise ValueError("supersample and block_side must be positive")
    height, width = raw.shape
    if min(row0, col0) < 0 or row0 + height > index.tile_y or col0 + width > index.tile_x:
        raise ValueError("geography patch lies outside the tile interior")
    rows = torch.arange(row0 * supersample, (row0 + height) * supersample,
                        dtype=torch.float64, device=raw.device)
    cols = torch.arange(col0 * supersample, (col0 + width) * supersample,
                        dtype=torch.float64, device=raw.device)
    lat, lon = pixel_coordinates(index, start_x, start_y, rows, cols, supersample)
    points = pixel_cartesian(lat, lon, radius)
    grid = points.reshape(len(rows), len(cols), 3)
    owners = []
    for row in range(0, len(rows), block_side):
        band = grid[row:row + block_side]
        owners.append(torch.cat([locator.locate(band[:, col:col + block_side].reshape(-1, 3))
                                 .reshape(band.shape[0], -1)
                                 for col in range(0, len(cols), block_side)], dim=1))
    owner = torch.cat(owners).reshape(-1)
    # The original manager decodes into C_FLOAT before its integer accumulator.
    pixels = raw.to(torch.float32).to(torch.int64)
    pixels = pixels.repeat_interleave(supersample, 0).repeat_interleave(supersample, 1).reshape(-1)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    return {"coordinates": torch.stack((lat_grid, lon_grid), -1).reshape(-1, 2),
            "cartesian": points, "owner": owner, "pixels": pixels}


def terrain_patch(raw: Tensor, index: GeotileIndex, locator: NearestCells, radius: Tensor, *,
                   start_x: int, start_y: int, row0: int = 0, col0: int = 0,
                   supersample: int = 1, block_side: int = 32) -> dict[str, Tensor]:
    """Map a native terrain patch and return mergeable cell sums/counts."""
    result = mapped_patch(raw, index, locator, radius, start_x=start_x, start_y=start_y,
                          row0=row0, col0=col0, supersample=supersample, block_side=block_side)
    total, count = terrain_accumulation(result["pixels"], result["owner"], locator.centers.shape[0])
    return {**result, "total": total, "count": count, "mean": terrain_mean(total, count, index.scale_factor)}
