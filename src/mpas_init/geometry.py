"""Spherical MPAS polynomial geometry, with the original ordered tiny solves."""

import math

import torch
from torch import Tensor

from mpas_init.vertical_grid import GlobalConnectivity
from mpas_init.mesh import GlobalMesh


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _arc(a: Tensor, b: Tensor) -> Tensor:
    radius = torch.sqrt(_dot(a, a))
    delta = b - a
    chord = torch.sqrt(_dot(delta, delta))
    return radius * 2.0 * torch.asin(chord / (2.0 * radius))


def _sphere_angle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    side_a, side_b, side_c = _arc(b, c), _arc(a, c), _arc(a, b)
    ab, ac = b - a, c - a
    cross = torch.stack((ab[..., 1] * ac[..., 2] - ab[..., 2] * ac[..., 1],
                         -(ab[..., 0] * ac[..., 2] - ab[..., 2] * ac[..., 0]),
                         ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]), dim=-1)
    semiperimeter = 0.5 * (side_a + side_b + side_c)
    sine = torch.sqrt(((torch.sin(semiperimeter - side_b) * torch.sin(semiperimeter - side_c))
                       / (torch.sin(side_b) * torch.sin(side_c))).clamp(0.0, 1.0))
    angle = 2.0 * torch.asin(sine.clamp(-1.0, 1.0))
    return torch.where(_dot(cross, a) >= 0.0, angle, -angle)


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    value = torch.zeros((*a.shape[:-2], a.shape[-2], b.shape[-1]), dtype=a.dtype, device=a.device)
    for k in range(a.shape[-1]):
        value = value + a[..., :, k, None] * b[..., None, k, :]
    return value


def _solve(matrix: Tensor, rhs: Tensor) -> Tensor:
    """MPAS scaled-pivot solve, batched across independent cells and RHS columns."""
    batch_size, n, _ = matrix.shape
    batch = torch.arange(batch_size, device=matrix.device)
    indices = torch.arange(n, device=matrix.device).expand(batch_size, n).clone()
    rows = list(matrix.unbind(dim=-2))
    # Functional row replacement retains both reverse- and forward-mode AD.
    scale = matrix.abs().amax(dim=-1)
    for j in range(n - 1):
        a = torch.stack(rows, dim=-2)
        candidates = indices[:, j:]
        ratios = a[batch[:, None], candidates, j].abs() / scale.gather(1, candidates)
        position = j + ratios.argmax(dim=-1)
        old = indices[:, j].clone()
        pivot = indices[batch, position].clone()
        columns = torch.arange(n, device=matrix.device)[None, :]
        indices = torch.where(columns == j, pivot[:, None],
                               torch.where(columns == position[:, None], old[:, None], indices))
        for i in range(j + 1, n):
            a = torch.stack(rows, dim=-2)
            target = indices[:, i]
            multiplier = a[batch, target, j] / a[batch, pivot, j]
            updated = torch.cat((a[batch, target, :j], multiplier[:, None],
                                 a[batch, target, j+1:] - multiplier[:, None] * a[batch, pivot, j+1:]), dim=-1)
            rows = [torch.where((target == k)[:, None], updated, row) for k, row in enumerate(rows)]
    a = torch.stack(rows, dim=-2)
    b = rhs
    for i in range(n - 1):
        for j in range(i + 1, n):
            target = indices[:, j]
            value = b[batch, target] - a[batch, target, i, None] * b[batch, indices[:, i]]
            b = torch.where((torch.arange(n, device=matrix.device)[None, :] == target[:, None])[:, :, None],
                            value[:, None, :], b)
    solution = [b[batch, indices[:, -1]] / a[batch, indices[:, -1], -1, None]]
    for j in range(n - 2, -1, -1):
        value = b[batch, indices[:, j]]
        for k in range(j + 1, n):
            value = value - a[batch, indices[:, j], k, None] * solution[k - j - 1]
        solution.insert(0, value / a[batch, indices[:, j], j, None])
    return torch.stack(solution, dim=-2)


def _inverse(matrix: Tensor) -> Tensor:
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return _solve(matrix, identity.expand_as(matrix))


def advection_coefficients(mesh: GlobalConnectivity, cell_xyz: Tensor,
                            vertex_xyz: Tensor, vertices_on_edge: Tensor,
                            sphere_radius: float | Tensor, stencil_size: int = 15) -> Tensor:
    """Compute the original quadratic spherical `deriv_two` from raw mesh data."""
    slots = mesh.cells_on_cell.shape[1]
    if stencil_size < slots + 1:
        raise ValueError("advection stencil cannot hold this mesh degree")
    center = cell_xyz / sphere_radius
    neighbors = center[mesh.cells_on_cell]
    north = center.new_tensor([0.0, 0.0, 1.0]).expand_as(center)
    pole = center[:, 2] == 1.0
    safe_north = torch.where(pole[:, None], center.new_tensor([0.0, 1.0, 0.0]), north)
    pi = center.new_tensor(torch.pi)
    theta_abs = torch.where(pole, pi / 2.0, pi / 2.0 - _sphere_angle(center, neighbors[:, 0], safe_north))
    theta = [theta_abs]
    lengths = []
    for j in range(slots):
        next_slot = torch.where(mesh.n_edges_on_cell > j + 1, j + 1, 0)
        next_neighbor = neighbors.gather(1, next_slot[:, None, None].expand(-1, 1, 3)).squeeze(1)
        if j + 1 < slots:
            theta.append(theta[-1] + _sphere_angle(center, neighbors[:, j], next_neighbor))
        lengths.append(sphere_radius * _arc(center, neighbors[:, j]))
    theta = torch.stack(theta, dim=-1)
    distance = torch.stack(lengths, dim=-1)
    xp, yp = torch.cos(theta) * distance, torch.sin(theta) * distance
    zero = torch.zeros_like(xp[:, :1])
    x, y = torch.cat((zero, xp), dim=-1), torch.cat((zero, yp), dim=-1)
    active = torch.arange(slots + 1, device=center.device)[None, :] <= mesh.n_edges_on_cell[:, None]
    design = torch.stack((torch.ones_like(x), x, y, x.square(), x * y, y.square()), dim=-1)
    design = torch.where(active[:, :, None], design, 0.0)
    transpose = design.transpose(-1, -2)
    fit = _matmul(_inverse(_matmul(transpose, design)), transpose)

    vertices = vertex_xyz / sphere_radius
    first, second = vertices[vertices_on_edge[:, 0]], vertices[vertices_on_edge[:, 1]]
    midpoint = 0.5 * (first + second)
    midpoint = torch.sqrt(_dot(first, first))[:, None] * midpoint / torch.sqrt(_dot(midpoint, midpoint))[:, None]
    edge_midpoint = midpoint[mesh.edges_on_cell]
    angle = _sphere_angle(center[:, None, :], neighbors, edge_midpoint) + theta
    cosine, sine = torch.cos(angle), torch.sin(angle)
    coefficients = (2.0 * cosine.square()[:, :, None] * fit[:, None, 3, :]
                    + 2.0 * (cosine * sine)[:, :, None] * fit[:, None, 4, :]
                    + 2.0 * sine.square()[:, :, None] * fit[:, None, 5, :])
    # Each global edge has one slot in each adjacent cell. Gather without scatter sums.
    result = []
    for side in range(2):
        cell = mesh.cells_on_edge[:, side]
        edge = torch.arange(mesh.cells_on_edge.shape[0], device=center.device)
        slot = (mesh.edges_on_cell[cell] == edge[:, None]).to(torch.int64).argmax(dim=-1)
        values = coefficients[cell, slot]
        values = torch.cat((values, values.new_zeros((values.shape[0], stencil_size - values.shape[1]))), dim=-1)
        result.append(values)
    return torch.stack(result, dim=1)


def deformation_coefficients(mesh: GlobalConnectivity, cell_xyz: Tensor,
                              vertex_xyz: Tensor, vertices_on_cell: Tensor,
                              sphere_radius: float | Tensor) -> dict[str, Tensor]:
    """Original spherical tangent-plane deformation and cell-gradient weights."""
    center, vertices = cell_xyz / sphere_radius, vertex_xyz / sphere_radius
    corners = vertices[vertices_on_cell]
    north = center.new_tensor([0.0, 0.0, 1.0]).expand_as(center)
    pole = center[:, 2] == 1.0
    safe_north = torch.where(pole[:, None], center.new_tensor([0.0, 1.0, 0.0]), north)
    pi = center.new_tensor(torch.pi)
    first = pi / 2.0 - _sphere_angle(center, corners[:, 0], safe_north)
    angles = [torch.where(pole, pi / 2.0, first)]
    slots = corners.shape[1]
    active = torch.arange(slots, device=center.device)[None, :] < mesh.n_edges_on_cell[:, None]
    for j in range(1, slots):
        previous = torch.where(active[:, j, None], corners[:, j - 1], corners[:, 0])
        following = torch.where(active[:, j, None], corners[:, j], corners[:, 1])
        angles.append(angles[-1] + _sphere_angle(center, previous, following))
    theta = torch.stack(angles, dim=-1)
    distance = sphere_radius * _arc(center[:, None], corners)
    x, y = torch.cos(theta) * distance, torch.sin(theta) * distance
    area = torch.zeros_like(center[:, 0])
    normal, lengths = [], []
    for j in range(slots):
        next_slot = torch.where(mesh.n_edges_on_cell > j + 1, j + 1, 0)
        xn, yn = x.gather(1, next_slot[:, None]).squeeze(1), y.gather(1, next_slot[:, None]).squeeze(1)
        dx, dy = xn - x[:, j], yn - y[:, j]
        # Preserve the two ordered area updates, including their subtraction rounding.
        updated = (area + 0.25 * (x[:, j] + xn) * (yn - y[:, j])
                   - 0.25 * (y[:, j] + yn) * (xn - x[:, j]))
        area = torch.where(active[:, j], updated, area)
        normal.append(torch.atan2(dy, dx) - pi / 2.0)
        lengths.append(torch.sqrt(dx.square() + dy.square()))
    normal, dl = torch.stack(normal, -1), torch.stack(lengths, -1)
    sine, cosine = torch.sin(normal), torch.cos(normal)
    own = torch.arange(cell_xyz.shape[0], device=center.device)[:, None]
    sign = torch.where(mesh.cells_on_edge[mesh.edges_on_cell, 0] == own, 1.0, -1.0)
    values = {"defc_a": sign * dl * (cosine.square() - sine.square()) / area[:, None],
              "defc_b": sign * (dl * 2.0 * (sine * cosine) / area[:, None]),
              "cell_gradient_coef_x": dl * cosine / area[:, None],
              "cell_gradient_coef_y": dl * sine / area[:, None]}
    return {name: torch.where(active, value, 0.0) for name, value in values.items()}


def prepare_mesh_geometry(raw: GlobalMesh, *, sphere_radius: float = 6371229.0,
                           omega: float = 7.29212e-5, nominal_min_dc: float | None = None) -> GlobalMesh:
    """Scale imported geometry and compute native weights, without fabricating terrain."""
    if not math.isfinite(sphere_radius) or sphere_radius <= 0 or not math.isfinite(omega):
        raise ValueError("sphere_radius must be finite and positive; omega must be finite")
    if nominal_min_dc is not None:
        if isinstance(nominal_min_dc, bool) or not isinstance(nominal_min_dc, (int, float)) or not math.isfinite(nominal_min_dc) or nominal_min_dc <= 0:
            raise ValueError("nominal_min_dc must be a finite positive length in metres")
        if "nominalMinDc" in raw.fields:
            raise ValueError("nominal_min_dc is only permitted when the raw mesh omits nominalMinDc")
    elif "nominalMinDc" not in raw.fields:
        raise ValueError("raw mesh lacks nominalMinDc; supply nominal_min_dc explicitly in metres")
    factor = sphere_radius / raw.sphere_radius
    lengths = {axis + location for axis in ("x", "y", "z") for location in ("Cell", "Edge", "Vertex")}
    lengths.update(("dcEdge", "dvEdge", "nominalMinDc"))
    areas = {"areaCell", "areaTriangle", "kiteAreasOnVertex"}
    fields = {name: value * factor if name in lengths else value * factor**2.0 if name in areas else value
              for name, value in raw.fields.items()}
    connection = raw.connectivity
    mesh = GlobalConnectivity(connection.cells_on_cell, connection.edges_on_cell, connection.n_edges_on_cell,
                              fields["dcEdge"], fields["dvEdge"], connection.cells_on_edge)
    vector = lambda location: torch.stack([fields[axis + location] for axis in ("x", "y", "z")], dim=-1)
    fields.update(deformation_coefficients(mesh, vector("Cell"), vector("Vertex"),
                                            (fields["verticesOnCell"] - 1).clamp_min(0), sphere_radius))
    fields["deriv_two"] = advection_coefficients(mesh, vector("Cell"), vector("Vertex"),
                                                 fields["verticesOnEdge"] - 1, sphere_radius)
    dimensions, sizes = dict(raw.dimensions), {**raw.sizes, "FIFTEEN": 15}
    if nominal_min_dc is not None:
        fields["nominalMinDc"] = fields["dcEdge"].new_tensor(nominal_min_dc)
        dimensions["nominalMinDc"] = ()
    for name in ("defc_a", "defc_b", "cell_gradient_coef_x", "cell_gradient_coef_y"):
        dimensions[name] = ("nCells", "maxEdges")
    dimensions["deriv_two"] = ("nEdges", "TWO", "FIFTEEN")
    for location in ("Edge", "Vertex"):
        name = "f" + location
        fields[name] = 2.0 * omega * torch.sin(fields["lat" + location])
        dimensions[name] = raw.dimensions["lat" + location]
    for location, dim in (("Cell", "nCells"), ("Edge", "nEdges"), ("Vertex", "nVertices")):
        name = "bdyMask" + location
        fields[name] = fields["indexTo" + location + "ID"].new_zeros(sizes[dim])
        dimensions[name] = (dim,)
    return GlobalMesh(fields, dimensions, sizes, sphere_radius, mesh)
