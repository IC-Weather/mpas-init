"""MPAS spherical vector basis and radial-basis wind reconstruction."""

import torch
from torch import Tensor

from mpas_init.arithmetic import _sincos
from mpas_init.geometry import _dot, _solve
from mpas_init.vertical_grid import GlobalConnectivity


def _unit(vector: Tensor) -> Tensor:
    return vector / torch.sqrt(_dot(vector, vector))[..., None]


def vector_geometry(mesh: GlobalConnectivity, cell_xyz: Tensor, edge_xyz: Tensor) -> dict[str, Tensor]:
    vertical = _unit(cell_xyz)
    normals = _unit(cell_xyz[mesh.cells_on_edge[:, 1]] - cell_xyz[mesh.cells_on_edge[:, 0]])
    first = normals[mesh.edges_on_cell[:, 0]]
    xhat = _unit(first - _dot(first, vertical)[:, None] * vertical)
    yhat = _unit(torch.stack((vertical[:, 1] * xhat[:, 2] - vertical[:, 2] * xhat[:, 1],
                             vertical[:, 2] * xhat[:, 0] - vertical[:, 0] * xhat[:, 2],
                             vertical[:, 0] * xhat[:, 1] - vertical[:, 1] * xhat[:, 0]), dim=-1))
    basis = torch.stack((xhat, yhat), dim=1)
    slots = mesh.edges_on_cell.shape[1]
    result = cell_xyz.new_zeros((cell_xyz.shape[0], slots, 3))
    # Degree is a static mesh selection, not a branch on a meteorological value.
    # Mask unused batches before solving; no tensor values move to the host.
    for count in range(3, slots + 1):
        edges = mesh.edges_on_cell[:, :count]
        locations, units = edge_xyz[edges], normals[edges]
        delta = cell_xyz[:, None, :] - locations
        radius = torch.sqrt(_dot(delta, delta))
        alpha = torch.zeros_like(radius[:, 0])
        for j in range(count):
            alpha = alpha + radius[:, j]
        alpha = alpha / count
        points = torch.stack([_dot(locations, basis[:, axis, None, :]) for axis in range(2)], dim=-1)
        directions = torch.stack([_dot(units, basis[:, axis, None, :]) for axis in range(2)], dim=-1)
        destination = torch.stack([_dot(cell_xyz, basis[:, axis, :]) for axis in range(2)], dim=-1)
        difference = points[:, :, None, :] - points[:, None, :, :]
        distance = (difference[..., 0].square() + difference[..., 1].square()) / alpha[:, None, None].square()
        dot = (directions[:, :, None, 0] * directions[:, None, :, 0]
               + directions[:, :, None, 1] * directions[:, None, :, 1])
        upper = torch.cat((1.0 / torch.sqrt(1.0 + distance) * dot, directions), dim=-1)
        lower = torch.cat((directions.transpose(-1, -2), cell_xyz.new_zeros((cell_xyz.shape[0], 2, 2))), dim=-1)
        matrix = torch.cat((upper, lower), dim=-2)
        difference = destination[:, None, :] - points
        distance = (difference[..., 0].square() + difference[..., 1].square()) / alpha[:, None].square()
        rhs = torch.cat((1.0 / torch.sqrt(1.0 + distance[..., None]) * directions,
                         torch.eye(2, dtype=cell_xyz.dtype, device=cell_xyz.device).expand(cell_xyz.shape[0], 2, 2)), dim=-2)
        active = mesh.n_edges_on_cell == count
        matrix = torch.where(active[:, None, None], matrix,
                              torch.eye(count + 2, dtype=cell_xyz.dtype, device=cell_xyz.device))
        rhs = torch.where(active[:, None, None], rhs, 0.0)
        solution = _solve(matrix, rhs)[:, :count]
        coeffs = basis[:, 0, None, :] * solution[:, :, 0, None] + basis[:, 1, None, :] * solution[:, :, 1, None]
        coeffs = torch.cat((coeffs, coeffs.new_zeros((coeffs.shape[0], slots - count, 3))), dim=1)
        result = torch.where(active[:, None, None], coeffs, result)
    return {"localVerticalUnitVectors": vertical, "edgeNormalVectors": normals,
            "cellTangentPlane": basis, "coeffs_reconstruct": result}


def reconstruct_wind(u: Tensor, coefficients: Tensor, mesh: GlobalConnectivity,
                       latitude: Tensor, longitude: Tensor) -> dict[str, Tensor]:
    vector = u.new_zeros((latitude.shape[0], u.shape[-1], 3))
    for j in range(mesh.edges_on_cell.shape[1]):
        term = coefficients[:, j, None, :] * u[mesh.edges_on_cell[:, j], :, None]
        vector = vector + torch.where((mesh.n_edges_on_cell > j)[:, None, None], term, 0.0)
    slat, clat = _sincos(latitude[:, None])
    slon, clon = _sincos(longitude[:, None])
    x, y, z = vector.unbind(dim=-1)
    return {"uReconstructZonal": -x * slon + y * clon,
            "uReconstructMeridional": -(x * clon + y * slon) * slat + z * clat}
