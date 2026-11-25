"""Bounded tile I/O for classic MPAS GWD, separate from the tensor statistics."""

import math
from pathlib import Path

import torch
from torch import Tensor

from mpas_init.geotile import GeotileIndex
from mpas_init._geotile_reader import GeotileReaders
from mpas_init._gwd_schedule import footprint_batches
from mpas_init.gwd import CLASSIC_FIELDS, RADIUS, box_indices, box_size, classic_statistics, smooth_classic_variance
from mpas_init.mesh import GlobalMesh
from mpas_init.prepare import _complete_tiles


def _geography(geography: str | Path):
    root = Path(geography)
    topo, land = (GeotileIndex.read(root / name / "index") for name in ("topo_gmted2010_30s", "landuse_30s"))
    def layout(index):
        return (index.kind, index.global_shape, index.dx, index.dy, index.known_lat, index.known_lon,
                index.tile_x, index.tile_y, index.tile_bdr, index.layers, index.tile_z_start,
                index.wordsize, index.signed, index.endian, index.scale_factor, index.missing_value)
    topo_layout = ("continuous", (43200, 21600), 0.00833333, 0.00833333, -89.99583, 0.004166667,
                   1200, 1200, 3, 1, 1, 2, True, "big", 1.0, None)
    land_layout = ("categorical", (43200, 21600), 0.00833333, 0.00833333, -89.99583, -179.99583,
                   1200, 1200, 0, 1, 1, 1, False, "big", 1.0, None)
    if layout(topo) != topo_layout or layout(land) != land_layout or (land.category_min, land.category_max) != (1, 24):
        raise ValueError("classic GWD requires native GMTED2010 and 24-category USGS 30-arc-second geography")
    for name, index in (("topo_gmted2010_30s", topo), ("landuse_30s", land)):
        _complete_tiles(root / name, index, "classic GWD")
    return root, topo, land


def _box(root, topo_index, land_index, latitude, longitude, nx, ny, cache, cache_tiles):
    rows, columns, dxm = box_indices(latitude, longitude, nx, ny)
    box = torch.empty(rows.shape, dtype=torch.float64, device=latitude.device)
    landuse = torch.empty(rows.shape, dtype=torch.int8, device=latitude.device)
    boxes, categories = box.reshape(-1, ny, nx), landuse.reshape(-1, ny, nx)
    # Only fixed integer indices cross the I/O boundary. Each row is consecutive
    # in longitude; pole reflection can reverse rows and shift their longitude.
    row_lists = rows[..., :, 0].reshape(-1, ny).tolist()
    column_lists = columns[..., :, 0].reshape(-1, ny).tolist()
    del rows, columns
    for member, (row_ids, column_ids) in enumerate(zip(row_lists, column_lists)):
        first = 0
        while first < ny:
            row, column = row_ids[first], column_ids[first]
            ty, local_row = divmod(row, 1200)
            step = row_ids[first + 1] - row if first + 1 < ny else 1
            step = step if step in (-1, 1) else 1
            end = first + 1
            while (end < ny and row_ids[end] // 1200 == ty and column_ids[end] == column
                    and row_ids[end] == row + step * (end - first)):
                end += 1
            height = end - first
            local_row = min(row, row_ids[end - 1]) - ty * 1200
            written = 0
            while written < nx:
                tx, local_column = divmod(column, 1200)
                width = min(nx - written, 1200 - local_column)
                window = (local_row, local_column, height, width)
                topo_path = root / 'topo_gmted2010_30s' / topo_index.filename(((tx + 18) % 36) * 1200 + 1, ty * 1200 + 1)
                land_path = root / 'landuse_30s' / land_index.filename(tx * 1200 + 1, ty * 1200 + 1)
                topo = cache.read(topo_path, topo_index, window=window).raw[0].to(dtype=torch.float32, device=latitude.device)
                raw_land = cache.read(land_path, land_index, window=window).raw[0].to(latitude.device)
                # The original reader hardcodes signed one-byte USGS input.
                land = torch.where(raw_land >= 128, raw_land - 256, raw_land).to(torch.int8)
                if step == -1:
                    topo, land = topo.flip(0), land.flip(0)
                boxes[member, first:end, written:written + width] = topo.to(torch.float64)
                categories[member, first:end, written:written + width] = land
                written += width
                column = (column + width) % 43200
            first = end
    return box, landuse, dxm


def read_classic_box(geography: str | Path, latitude: Tensor, longitude: Tensor, diameter: Tensor):
    """Decode one full-resolution native footprint; angles are radians, diameter meters."""
    for value in (latitude, longitude, diameter):
        if value.shape != () or value.dtype != torch.float64 or not torch.isfinite(value):
            raise ValueError("classic GWD requires finite float64 scalar coordinates and diameter")
    if (latitude.device != longitude.device or latitude.device != diameter.device
            or abs(latitude) > torch.pi * 0.5 or diameter <= 0):
        raise ValueError("invalid classic GWD latitude, diameter, or devices")
    root, topo, land = _geography(geography)
    nx, ny = (int(value) for value in box_size(latitude, diameter))
    if min(nx, ny) < 4:
        raise ValueError("native classic GWD footprint needs at least four pixels in both directions")
    with GeotileReaders(16) as readers:
        return _box(root, topo, land, latitude, longitude, nx, ny, readers, 8)


def prepare_classic_gwd(mesh: GlobalMesh, geography: str | Path, *, cell_scaling: float = 1.0,
                        cache_tiles: int = 8) -> dict[str, Tensor]:
    """Prepare all fourteen classic fields, including native isolated-point smoothing."""
    if mesh.sphere_radius != RADIUS or not math.isfinite(cell_scaling) or cell_scaling <= 0 or cache_tiles < 1:
        raise ValueError("classic GWD requires native Earth-scaled geometry, positive scaling and cache size")
    root, topo, land = _geography(geography)
    latitude, longitude = mesh.fields["latCell"], mesh.fields["lonCell"]
    diameter = mesh.connectivity.mean_dc() * cell_scaling
    widths, heights = box_size(latitude, diameter)
    # Footprint shapes and tile selection are discrete file preparation, not AD
    # through a data-dependent shape. Continuous terrain statistics remain tensors.
    shapes = list(zip(widths.tolist(), heights.tolist()))
    if any(min(shape) < 4 for shape in shapes):
        raise ValueError("native classic GWD footprint needs at least four pixels in both directions")
    values = latitude.new_empty((len(CLASSIC_FIELDS), latitude.numel()))
    mask = torch.empty(latitude.shape, dtype=torch.int64, device=latitude.device)
    with GeotileReaders(2 * cache_tiles) as cache:
        for (nx, ny), cells in footprint_batches(shapes, latitude, longitude):
            ids = torch.tensor(cells, dtype=torch.int64, device=latitude.device)
            box, landuse, dxm = _box(root, topo, land, latitude[ids], longitude[ids], nx, ny, cache, cache_tiles)
            result = classic_statistics(box, landuse, dxm)
            values[:, ids] = torch.stack([result[name] for name in CLASSIC_FIELDS])
            mask[ids] = ((landuse != 16).sum((-2, -1)).to(box.dtype) / (nx * ny) >= 0.5).long()
            del box, landuse, dxm, result
    smoothed, isolated = smooth_classic_variance(values[0], mask, mesh.connectivity)
    slots = torch.arange(mesh.connectivity.cells_on_cell.shape[1], device=latitude.device)[None] < mesh.connectivity.n_edges_on_cell[:, None]
    if torch.any(isolated[:, None] & isolated[mesh.connectivity.cells_on_cell] & slots):
        raise ValueError("classic GWD smoothing requires a triangular global cell graph without adjacent isolated points")
    return {name: smoothed if name == "var2d" else values[i] for i, name in enumerate(CLASSIC_FIELDS)}
