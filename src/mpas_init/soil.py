"""Native annual soil-temperature preparation from the bounded one-degree dataset."""

from pathlib import Path

import torch
from torch import Tensor

from mpas_init.geotile import GeotileIndex, read_geotile
from mpas_init.horizontal import four_point_sequence
from mpas_init.mesh import GlobalMesh


def soil_temperature(field: Tensor, latitude: Tensor, longitude: Tensor, landmask: Tensor) -> Tensor:
    """Original interp_soiltemp coordinates and missing-value sequence.

    The field has native bounds (-2:363, -2:183), stored as row then column.
    Its inputs already include the original C_FLOAT scale multiplication.
    """
    if field.shape != (186, 366) or latitude.ndim != 1 or latitude.shape != longitude.shape or landmask.shape != latitude.shape:
        raise ValueError("soil temperature requires the native bordered field and matching cell vectors")
    degrees = 180.0 / torch.pi
    lat, lon = latitude * degrees, longitude * degrees
    x = (lon - (-179.5)) / 1.0 + 1.0
    wrapped = torch.where(x < 0.5, lon + 360.0, torch.where(x >= 360.5, lon - 360.0, lon))
    x = (wrapped - (-179.5)) / 1.0 + 1.0
    y = ((lat - (-89.5)) / 1.0 + 1.0).clamp(1.0, 179.0)
    value = four_point_sequence(field, x, y, start_x=-2, start_y=-2)
    return torch.where(landmask == 1, value, 0.0)


def _soil_input(geography):
    geography = Path(geography)
    index = GeotileIndex.read(geography / "index")
    layout = (index.kind, index.global_shape, index.tile_x, index.tile_y, index.layers, index.tile_z_start,
              index.tile_bdr, index.wordsize, index.signed, index.endian,
              index.dx, index.dy, index.known_lat, index.known_lon, index.scale_factor, index.missing_value)
    expected = ("continuous", (360, 180), 180, 180, 1, 1, 3, 2, False, "big", 1.0, 1.0, -89.5, -179.5, 0.01, 0.0)
    if layout != expected:
        raise ValueError("soil temperature requires the native bordered, unsigned one-degree geography")
    from mpas_init.prepare import _complete_tiles
    paths = [path for _, _, path in _complete_tiles(geography, index, 'soil-temperature')]
    return index, paths


def prepare_soil_temperature(mesh: GlobalMesh, geography: str | Path, landmask: Tensor) -> Tensor:
    """Decode the two native annual-mean tiles, then interpolate land cells in PyTorch."""
    index, paths = _soil_input(geography)
    if (landmask.shape != (mesh.sizes["nCells"],) or landmask.dtype not in (torch.int32, torch.int64)
            or torch.any((landmask != 0) & (landmask != 1))):
        raise ValueError("soil temperature requires a binary integer landmask on this mesh")
    first, second = (read_geotile(path, index, include_border=True).raw[0] for path in paths)
    device = mesh.fields["latCell"].device
    raw = torch.cat((first[:, :183], second[:, 3:186]), dim=1).to(device)
    field = (raw.to(torch.float32) * 0.01).to(torch.float64)
    mask = landmask.to(device)
    result = soil_temperature(field, mesh.fields["latCell"], mesh.fields["lonCell"], mask)
    if not torch.isfinite(result).all() or torch.any(result[mask == 1] <= 0):
        raise ValueError("soil-temperature geography leaves land cells without a positive annual mean")
    return result
