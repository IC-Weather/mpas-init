"""Native MPAS point interpolation and explicit missing-value selection."""

from dataclasses import dataclass

import torch
from torch import Tensor

MISSING = -1.0e30


@dataclass(frozen=True)
class HorizontalGrid:
    latitude: Tensor
    longitude: Tensor
    grid_type: str = "regular_latlon"
    row_counts: Tensor | None = None


def remap_atmosphere(fields: dict[str, Tensor], names: tuple[str, ...], latitude: Tensor,
                       longitude: Tensor, lat: Tensor, lon: Tensor, *, grid_type: str = "regular_latlon",
                       field_grids: dict[str, HorizontalGrid] | None = None,
                       row_counts: Tensor | None = None,
                       missing_fields: set[str] | frozenset[str] = frozenset()) -> dict[str, Tensor]:
    """Map each field directly to its MPAS targets; keep common level ownership.

    The primary grid owns pressure and height. Other fields may carry their own
    complete global rectilinear grid. Coordinates use degrees; targets use radians.
    """
    field_grids = {} if field_grids is None else field_grids
    if "geopotential_height" in field_grids or set(field_grids) - set(fields):
        raise ValueError("field grids must name non-height atmospheric fields; height owns the primary grid")
    if missing_fields - set(fields):
        raise ValueError("missing-field declarations name an unknown atmospheric field")
    x, y = global_latlon_indices(latitude, longitude, lat, lon, grid_type=grid_type, row_counts=row_counts)
    result = {}
    for name in names:
        source = field_grids.get(name)
        source_lat, source_lon = (latitude, longitude) if source is None else (source.latitude, source.longitude)
        counts = row_counts if source is None else source.row_counts
        value = fields[name]
        horizontal_shape = (source_lon.numel(),) if counts is not None else (source_lat.numel(), source_lon.numel())
        if (value.shape != (fields["geopotential_height"].shape[0], *horizontal_shape)):
            raise ValueError("atmospheric field dimensions differ from its grid or shared vertical coordinate: " + name)
        xx, yy = ((x, y) if source is None else global_latlon_indices(
            source.latitude, source.longitude, lat, lon, grid_type=source.grid_type, row_counts=counts))
        if name in missing_fields:
            if counts is not None or value.ndim != 3 or value.shape[-1] < 3:
                raise ValueError("masked atmospheric interpolation requires rectangular grids with at least three longitudes")
            mapped = []
            for level in value.unbind(0):
                halo = torch.cat((level[:, -3:], level, level[:, :3]), dim=-1)
                mapped.append(four_point_sequence(halo, xx + 1., yy + 1., start_x=-2, start_y=1,
                                                  missing=MISSING, weighted16=False, sixteen=True, four=False))
            mapped = torch.stack(mapped)
            torch._assert_async((torch.isfinite(mapped) & (mapped != MISSING)).all(),
                                "no valid required atmospheric source support: " + name)
        else:
            mapped = sixteen_point(value, xx, yy, row_counts=counts)
        result[name] = mapped.transpose(-1, -2)
    return result


def _longitude_index(longitude: Tensor, lon_degrees: Tensor) -> Tensor:
    dx = longitude[1] - longitude[0]
    torch._assert_async(dx > 0, "source longitude must scan eastward; normalize coordinates and fields together")
    x = (lon_degrees - longitude[0]) / dx
    # Wrap in physical degrees before division, as latlon_to_ij does.
    # A remainder after division loses bits near the periodic seam.
    wrapped = torch.where(x + 1.0 < 0.5, lon_degrees + 360.0,
                           torch.where(x + 1.0 >= longitude.shape[0] + 0.5,
                                       lon_degrees - 360.0, lon_degrees))
    x = (wrapped - longitude[0]) / dx
    return x


def regular_latlon_indices(latitude: Tensor, longitude: Tensor, lat: Tensor,
                            lon: Tensor) -> tuple[Tensor, Tensor]:
    """Original LATLON projection and wrap order; eastward source scan, radian targets."""
    # The original rounds PI to RKIND before dividing, including in float32.
    degrees = lat.new_tensor(180.0) / lat.new_tensor(torch.pi)
    x = _longitude_index(longitude, lon * degrees)
    y = (lat * degrees - latitude[0]) / (latitude[1] - latitude[0])
    y = torch.where(y + 1.0 < 0.5, 0.0,
                     torch.where(y + 1.0 >= latitude.shape[0] + 0.5, latitude.shape[0] - 1.0, y))
    return x, y


def global_latlon_indices(latitude: Tensor, longitude: Tensor, lat: Tensor,
                           lon: Tensor, *, grid_type: str = "regular_latlon",
                           row_counts: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Native LATLON/GAUSS indices; reduced rows return four longitude stencils."""
    if (grid_type == "reduced_gaussian") != (row_counts is not None):
        raise ValueError("reduced Gaussian grids require row counts; rectangular grids must not carry them")
    if grid_type == "regular_latlon":
        return regular_latlon_indices(latitude, longitude, lat, lon)
    if grid_type not in ("rectilinear_latlon", "reduced_gaussian"):
        raise ValueError("unsupported source grid: " + grid_type)
    if latitude.ndim != 1 or latitude.shape[0] < 3:
        raise ValueError("nonuniform latitude requires at least three nodes")
    torch._assert_async((latitude[:-1] > latitude[1:]).all(), "nonuniform latitude must scan north to south")
    degrees = lat.new_tensor(180.0) / lat.new_tensor(torch.pi)
    target = lat * degrees
    lower = (torch.searchsorted(-latitude, -target, right=False) - 1).clamp(0, latitude.shape[0] - 2)
    north, south = latitude[lower], latitude[lower + 1]
    # Preserve llij_gauss's one-based weighted-index expression before conversion.
    j = ((north - target) * (lower + 2) + (target - south) * (lower + 1)) / (north - south)
    nearest = torch.where((target - latitude[0]).abs() < (target - latitude[-1]).abs(), 1., float(latitude.shape[0]))
    y = torch.where(target.abs() > latitude[0].abs(), nearest, j) - 1.0
    if row_counts is None:
        return _longitude_index(longitude, lon * degrees), y
    if row_counts.shape != latitude.shape or row_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("reduced row counts must be an integer vector matching latitude")
    torch._assert_async((row_counts >= 4).all() & (row_counts.sum() == longitude.numel()),
                        "reduced rows require at least four points and complete longitude storage")
    offsets = torch.cumsum(row_counts, 0) - row_counts
    row = (torch.trunc(y + 1.0 + 0.00001).to(torch.int64)[None] - 1
           + torch.arange(-1, 3, device=y.device)[:, None]).clamp(0, latitude.numel() - 1)
    count = row_counts[row]
    origin = longitude[offsets[row]]
    dx = torch.div(360.0, count.to(longitude.dtype))
    physical = lon[None] * degrees
    x = (physical - origin) / dx
    wrapped = torch.where(x + 1.0 < 0.5, physical + 360.0,
                           torch.where(x + 1.0 >= count + 0.5, physical - 360.0, physical))
    return (wrapped - origin) / dx, y


def overlapping_parabolic(x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """Port of init_atm_hinterp.oned, including its exact-zero branches."""
    initial = torch.where(x == 0, b, torch.where(x == 1, c, 0.0))
    left = b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b))
    right = c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c))
    both = (1.0 - x) * left + x * right
    boundary = torch.where((a == 0) & (d == 0), b * (1.0 - x) + c * x,
                           torch.where(a != 0, left, right))
    return torch.where(b * c != 0, torch.where(a * d == 0, boundary, both), initial)


def sixteen_point(field: Tensor, x: Tensor, y: Tensor, *, row_counts: Tensor | None = None) -> Tensor:
    """Interpolate complete unmasked fields [..., latitude, longitude].

    x/y are zero-based fractional source indices. Longitude is periodic. The
    original polar stencil clamps latitude rather than reflecting through poles.
    Complete-field SEARCH fallback selects the nearest point at a rounded pole.
    Missing/coastal search belongs to a separate explicit selection stage.
    """
    if row_counts is not None:
        return _reduced_sixteen_point(field, x, y, row_counts)
    nx, ny = field.shape[-1], field.shape[-2]
    xx, yy = x + 1.0, y + 1.0
    i, j = torch.trunc(xx + 0.00001).to(torch.int64), torch.trunc(yy + 0.00001).to(torch.int64)
    fx, fy = xx - i, yy - j
    i, j = i - 1, j - 1
    rows = []
    for dy in range(-1, 3):
        values = []
        for dx in range(-1, 3):
            value = field[..., (j + dy).clamp(0, ny - 1), (i + dx).remainder(nx)]
            values.append(torch.where(value == 0, 1.0e-20, value))
        rows.append(overlapping_parabolic(fx, *values))
    result = overlapping_parabolic(fy, *rows)
    result = torch.where(result == 1.0e-20, 0.0, result)
    direct = field[..., j.clamp(0, ny - 1), i.remainder(nx)]
    interpolated = torch.where((fx.abs() > 0.0001) | (fy.abs() > 0.0001), result, direct)
    # sixteen_pt checks INT(yy) before its epsilon-adjusted stencil index.
    # Float32 mesh coordinates can place an exact pole just outside that range.
    outside = (torch.trunc(yy) < 1) | (torch.trunc(yy) > ny)
    nearest_i = torch.floor(xx + 0.5).to(torch.int64) - 1
    nearest_j = torch.floor(yy + 0.5).to(torch.int64) - 1
    nearest = field[..., nearest_j.clamp(0, ny - 1), nearest_i.remainder(nx)]
    return torch.where(outside, nearest, interpolated)


def _reduced_sixteen_point(field: Tensor, x: Tensor, y: Tensor, counts: Tensor) -> Tensor:
    """Apply the same native polynomials to four independently periodic rows."""
    if x.shape != (4, y.numel()) or counts.ndim != 1 or field.ndim < 1:
        raise ValueError("reduced interpolation requires four row stencils and flattened point fields")
    torch._assert_async(counts.sum() == field.shape[-1], "reduced field and row counts differ")
    offsets = torch.cumsum(counts, 0) - counts
    xx, yy = x + 1.0, y + 1.0
    i = torch.trunc(xx + 0.00001).to(torch.int64) - 1
    j = torch.trunc(yy + 0.00001).to(torch.int64) - 1
    fx, fy = xx - (i + 1), yy - (j + 1)
    rows = []
    for r, dy in enumerate(range(-1, 3)):
        row = (j + dy).clamp(0, counts.numel() - 1)
        values = []
        for dx in range(-1, 3):
            value = field[..., offsets[row] + (i[r] + dx).remainder(counts[row])]
            values.append(torch.where(value == 0, 1.0e-20, value))
        rows.append(overlapping_parabolic(fx[r], *values))
    result = overlapping_parabolic(fy, *rows)
    result = torch.where(result == 1.0e-20, 0.0, result)
    row = j.clamp(0, counts.numel() - 1)
    direct = field[..., offsets[row] + i[1].remainder(counts[row])]
    interpolated = torch.where((fx[1].abs() > 0.0001) | (fy.abs() > 0.0001), result, direct)
    nearest_row = (torch.floor(yy + 0.5).to(torch.int64) - 1).clamp(0, counts.numel() - 1)
    nearest_i = torch.floor(xx[1] + 0.5).to(torch.int64) - 1
    nearest = field[..., offsets[nearest_row] + nearest_i.remainder(counts[nearest_row])]
    outside = (torch.trunc(yy) < 1) | (torch.trunc(yy) > counts.numel())
    return torch.where(outside, nearest, interpolated)


def _bfs_priority(dx: Tensor, dy: Tensor, width: int) -> Tensor:
    group = torch.where(dx < 0, 0, torch.where(dx > 0, 1, 2))
    return ((dx.abs() + dy.abs()) * (6 * width + 4) + group * (2 * width + 1)
            + 2 * (width - dx.abs()) + (dy > 0).to(torch.int64))


def _search_missing(field: Tensor, x: Tensor, y: Tensor, start_x: int, start_y: int, missing: float,
                    active: Tensor | None = None) -> Tensor:
    """Original left/right/down/up FIFO search, including the queue after first discovery."""
    ny, nx = field.shape
    rows = torch.arange(ny, device=field.device)[:, None]
    cols = torch.arange(nx, device=field.device)[None, :]
    valid = field != missing
    infinity = 4 * (nx + ny + 1)
    left = cols + torch.cummin(torch.where(valid, -cols, infinity), dim=1).values
    right = -cols + torch.cummin(torch.where(valid, cols, infinity).flip(1), dim=1).values.flip(1)
    horizontal = torch.minimum(left, right)
    down = rows + torch.cummin(horizontal - rows, dim=0).values
    up = -rows + torch.cummin((horizontal + rows).flip(0), dim=0).values.flip(0)
    radius_map = torch.minimum(down, up)
    delta = torch.arange(1 - min(ny, nx), min(ny, nx), device=field.device)
    signs = x.new_tensor([-1, 1], dtype=torch.int64)
    outputs = []
    offset = 0
    for xx, yy in zip(x.split(128), y.split(128)):
        ix = torch.where(xx >= 0, torch.floor(xx + 0.5), torch.ceil(xx - 0.5)).to(torch.int64)
        iy = torch.where(yy >= 0, torch.floor(yy + 0.5), torch.ceil(yy - 0.5)).to(torch.int64)
        radius = radius_map[(iy - start_y).clamp(0, ny - 1), (ix - start_x).clamp(0, nx - 1)]
        radius = radius[:, None] + torch.arange(2, device=field.device)[None, :]
        orthogonal = radius[:, :, None] - delta.abs()[None, None, :]
        shape = (xx.numel(), 2, 2, delta.numel())
        along = delta[None, None, None, :].expand(shape)
        across = orthogonal[:, :, None, :] * signs[None, None, :, None]
        dx, dy = (across, along) if ny <= nx else (along, across)
        dx, dy = dx.reshape(xx.numel(), -1), dy.reshape(xx.numel(), -1)
        columns, lines = ix[:, None] + dx, iy[:, None] + dy
        values = field[(lines - start_y).clamp(0, ny - 1), (columns - start_x).clamp(0, nx - 1)]
        eligible = ((orthogonal >= 0)[:, :, None, :].expand(shape).reshape(xx.numel(), -1)
                    & (columns >= start_x) & (columns < start_x + nx)
                    & (lines >= start_y) & (lines < start_y + ny) & (values != missing))
        order = _bfs_priority(dx, dy, nx)
        limit = torch.iinfo(torch.int64).max
        # No valid point lies inside the first Manhattan shell. Before stopping,
        # the native FIFO can enqueue only that shell and its immediate successor.
        first = torch.where(eligible[:, :2 * delta.numel()], order[:, :2 * delta.numel()], limit).amin(-1)
        if active is not None:
            first = torch.where(active[offset:offset + xx.numel()], first, limit)
        offset += xx.numel()
        parent_x = torch.where(dy != 0, dx, dx - dx.sign())
        parent_y = torch.where(dy != 0, dy - dy.sign(), dy)
        queued = _bfs_priority(parent_x, parent_y, nx) <= first[:, None]
        rx, ry = columns.to(xx.dtype) - xx[:, None], lines.to(yy.dtype) - yy[:, None]
        distance = rx * rx + ry * ry
        eligible = eligible & queued
        minimum = torch.where(eligible, distance, torch.inf).amin(-1)
        selected = torch.where(eligible & (distance == minimum[:, None]), order, limit).argmin(-1)
        inside = ((ix >= start_x) & (ix < start_x + nx) & (iy >= start_y) & (iy < start_y + ny))
        value = values.gather(1, selected[:, None]).squeeze(1)
        outputs.append(torch.where(inside & (first != limit), value, missing))
    return torch.cat((field.flatten()[:0], *outputs))


def _distance_weight(dx: Tensor, dy: Tensor, radius: float) -> Tensor:
    squared = dx * dx + dy * dy
    # Zero distance in an unused branch must not poison active coordinate derivatives.
    distance = torch.sqrt(torch.where(squared > 0, squared, 1.0))
    return (radius - torch.where(squared > 0, distance, 0.0)).clamp_min(0)


def four_point_sequence(field: Tensor, x: Tensor, y: Tensor, *, start_x: int = 0, start_y: int = 0,
                         missing: float = 0.0, weighted16: bool = True, search: bool = True,
                         sixteen: bool = False, four: bool = True, active: Tensor | None = None) -> Tensor:
    """FOUR_POINT, W_AVERAGE4, W_AVERAGE16, SEARCH on a bounded two-dimensional field.

    Disable weighted16 for the native surface/soil sequences. Optional sixteen
    precedes FOUR_POINT; four=False selects SIXTEEN_POINT then SEARCH only.
    search=False ends after weighted averaging.
    Coordinates retain the native array origin. Missing patterns and discrete
    SEARCH ownership are not differentiated; active field values remain tensors.
    An explicit active mask suppresses unused target values, returning missing there.
    """
    if field.ndim != 2 or x.ndim != 1 or x.shape != y.shape or not x.numel() or min(field.shape) < 1:
        raise ValueError("four-point sequence requires a 2-D field and matching nonempty coordinate vectors")
    if active is not None and (active.shape != x.shape or active.dtype != torch.bool):
        raise ValueError("interpolation active mask must be boolean and match target coordinates")
    ny, nx = field.shape
    end_x, end_y = start_x + nx - 1, start_y + ny - 1
    fx, cx, fy, cy = (op(v).to(torch.int64) for v, op in ((x, torch.floor), (x, torch.ceil),
                                                        (y, torch.floor), (y, torch.ceil)))

    def sample(i, j):
        return field[(j - start_y).clamp(0, ny - 1), (i - start_x).clamp(0, nx - 1)]

    a, b, c, d = sample(fx, fy), sample(fx, cy), sample(cx, fy), sample(cx, cy)
    xf, xc, yf, yc = (v.to(x.dtype) for v in (fx, cx, fy, cy))
    bilinear = ((y - yf) * (b * (xc - x) + d * (x - xf))
                + (yc - y) * (a * (xc - x) + c * (x - xf)))
    bilinear = torch.where(fx == cx, torch.where(fy == cy, a, a * (yc - y) + b * (y - yf)),
                            torch.where(fy == cy, a * (xc - x) + c * (x - xf), bilinear))
    complete = ((fx >= start_x) & (cx <= end_x) & (fy >= start_y) & (cy <= end_y)
                & (a != missing) & (b != missing) & (c != missing) & (d != missing))
    left, right = (x > start_x - 0.5) & (fx < start_x), (x < end_x + 0.5) & (cx > end_x)
    bottom, top = (y > start_y - 0.5) & (fy < start_y), (y < end_y + 0.5) & (cy > end_y)
    ax, bx = (torch.where(left, start_x, torch.where(right, end_x, v)) for v in (fx, cx))
    ay, by = (torch.where(bottom, start_y, torch.where(top, end_y, v)) for v in (fy, cy))
    allowed4 = (ax >= start_x) & (bx <= end_x) & (ay >= start_y) & (by <= end_y)
    values4 = (sample(ax, ay), sample(ax, by), sample(bx, ay), sample(bx, by))
    weights4 = tuple(torch.where(value != missing, _distance_weight(x - i, y - j, 1.0), 0.0)
                     for value, i, j in zip(values4, (xf, xf, xc, xc), (yf, yc, yf, yc)))
    w0, w1, w2, w3 = weights4
    v0, v1, v2, v3 = values4
    sum4 = ((w0 + w1) + w2) + w3
    mean4 = (((w0 * v0 + w1 * v1) + w2 * v2) + w3 * v3) / torch.where(sum4 > 0, sum4, 1.0)
    searched = torch.full_like(x, missing)
    needs_search = torch.ones_like(x, dtype=torch.bool)
    fallback = searched
    if weighted16:
        sum16, total16 = torch.zeros_like(x), torch.zeros_like(x)
        for i in range(1, 5):
            for j in range(1, 5):
                column, row = fx + 3 - i, fy + 3 - j
                value = sample(column, row)
                weight = torch.where(value != missing, _distance_weight(x - column, y - row, 2.0), 0.0)
                sum16 = sum16 + weight
                total16 = total16 + weight * value
        allowed16 = (fx >= start_x + 1) & (fx <= end_x - 2) & (fy >= start_y + 1) & (fy <= end_y - 2)
        mean16 = total16 / torch.where(sum16 > 0, sum16, 1.0)
        fallback = torch.where(allowed16 & (sum16 > 0), mean16, fallback)
        needs_search = ~(allowed16 & (sum16 > 0))
    fallback = torch.where(allowed4, torch.where(sum4 > 0, mean4, fallback), missing)
    result = torch.where(complete, bilinear, fallback) if four else searched
    needs_search = (~complete & allowed4 & ~(sum4 > 0) & needs_search) if four else torch.ones_like(needs_search)
    if sixteen:
        ix = torch.trunc(x + 0.00001).to(torch.int64)
        iy = torch.trunc(y + 0.00001).to(torch.int64)
        rx, ry = x - ix, y - iy
        valid = torch.ones_like(x, dtype=torch.bool)
        rows = []
        for dy in range(-1, 3):
            values = []
            for dx in range(-1, 3):
                value = sample(ix + dx, iy + dy)
                valid = valid & (value != missing)
                values.append(torch.where((value == 0) & (missing != 0), 1.0e-20, value))
            rows.append(overlapping_parabolic(rx, *values))
        high_order = overlapping_parabolic(ry, *rows)
        high_order = torch.where(high_order == 1.0e-20, 0.0, high_order)
        direct = sample(ix, iy)
        off_point = (rx.abs() > 0.0001) | (ry.abs() > 0.0001)
        allowed = ((torch.trunc(x) >= start_x) & (torch.trunc(x) <= end_x)
                   & (torch.trunc(y) >= start_y) & (torch.trunc(y) <= end_y))
        valid_direct = ((ix >= start_x) & (ix <= end_x) & (iy >= start_y) & (iy <= end_y)
                        & (direct != missing))
        use_sixteen = allowed & torch.where(off_point, valid, valid_direct)
        result = torch.where(use_sixteen,
                             torch.where(off_point, high_order, direct), result)
        needs_search = needs_search & ~use_sixteen
    if active is not None:
        needs_search = needs_search & active
    if search:
        searched = _search_missing(field, x, y, start_x, start_y, missing, active=needs_search)
        result = torch.where(needs_search, searched, result)
    if active is not None:
        result = torch.where(active, result, missing)
    return result
