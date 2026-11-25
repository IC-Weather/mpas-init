"""Atmospheric full levels to a native global MPAS dycore initial condition."""

from dataclasses import dataclass, fields as dataclass_fields

import torch
from torch import Tensor

from mpas_init.mesh import GlobalStatic
from mpas_init.netcdf import PressureLevelSource
from mpas_init.reconstruction import reconstruct_wind, vector_geometry
from mpas_init.thermodynamics import pressure_level_state
from mpas_init.vertical_grid import smooth_terrain, tc_interfaces, vertical_grid
from mpas_init.wind import pressure_level_wind, terrain_flux_metrics, vertical_wind


def _geometry_inputs(static):
    return tuple(sorted(static.fields.items())) + tuple(
        ("connectivity." + field.name, getattr(static.connectivity, field.name))
        for field in dataclass_fields(static.connectivity))


def _bindings(values):
    try:
        return tuple((name, value, value._version, value.requires_grad) for name, value in values)
    except RuntimeError as exc:
        raise ValueError("reusable geometry requires version-tracked tensors; prepare outside inference_mode") from exc


@dataclass(frozen=True)
class DycoreGeometry:
    """Geometry for one unchanged static object and vertical configuration.

    Reuse across meteorological states, not across geometry edits.
    To differentiate geometry, prepare it inside the differentiated function.
    """

    static: GlobalStatic
    n_levels: int
    model_top: float
    fields: dict[str, Tensor]
    _inputs: tuple
    _outputs: tuple
    _grad_enabled: bool

    def validate(self, static, n_levels, model_top):
        if static is not self.static or (n_levels, model_top) != (self.n_levels, self.model_top):
            raise ValueError("reusable geometry belongs to a different static object or vertical configuration")
        if (torch.is_grad_enabled() != self._grad_enabled
                and any(requires_grad for _, _, _, requires_grad in self._inputs)):
            raise ValueError("reusable geometry belongs to a different AD mode; prepare it in the current mode")
        for current, original in ((_geometry_inputs(static), self._inputs),
                                  (tuple(sorted(self.fields.items())), self._outputs)):
            if len(current) != len(original):
                raise ValueError("stale reusable geometry: tensor inventory changed")
            for (name, value), (old_name, old_value, version, requires_grad) in zip(current, original):
                if (name != old_name or value is not old_value or value._version != version
                        or value.requires_grad != requires_grad):
                    raise ValueError("stale reusable geometry: " + name)


def _geometry_fields(static, n_levels, model_top):
    f, mesh = static.fields, static.connectivity
    terrain = smooth_terrain(f["ter"], mesh)
    grid = vertical_grid(terrain, mesh, tc_interfaces(n_levels, terrain.new_tensor(model_top)))
    metrics = terrain_flux_metrics(grid["zgrid"], mesh, f["areaCell"], f["deriv_two"])
    xyz = lambda location: torch.stack([f[axis + location] for axis in ("x", "y", "z")], dim=-1)
    vectors = vector_geometry(mesh, xyz("Cell"), xyz("Edge"))
    return {**f, "ter": terrain, **grid, **metrics, **vectors}


def prepare_dycore_geometry(static: GlobalStatic, *, n_levels: int = 55,
                            model_top: float = 30000.0) -> DycoreGeometry:
    """Prepare target-only tensor geometry; retain its ordinary PyTorch AD graph."""
    if n_levels < 3 or not 0 < model_top <= 30000.0:
        raise ValueError("requires at least three levels and a model top within (0, 30000] m")
    inputs = _bindings(_geometry_inputs(static))
    geometry = _geometry_fields(static, n_levels, model_top)
    return DycoreGeometry(static, n_levels, model_top, geometry, inputs,
                          _bindings(tuple(sorted(geometry.items()))), torch.is_grad_enabled())


def initialize_dycore(source: PressureLevelSource, static: GlobalStatic, *,
                       n_levels: int = 55, model_top: float = 30000.0,
                       geometry: DycoreGeometry | None = None) -> dict[str, Tensor]:
    """Build dycore fields only. Surface/soil and coupled physics are not claimed."""
    if n_levels < 3 or not 0 < model_top <= 30000.0:
        raise ValueError("requires at least three levels and a model top within (0, 30000] m")
    f, mesh = static.fields, static.connectivity
    if geometry is not None:
        geometry.validate(static, n_levels, model_top)
    grid = _geometry_fields(static, n_levels, model_top) if geometry is None else geometry.fields
    state = pressure_level_state(source.fields, source.pressure, source.latitude, source.longitude,
                                  f["latCell"], f["lonCell"], grid, grid_type=source.grid_type,
                                  field_grids=source.field_grids, row_counts=source.row_counts,
                                  field_surface_pressure=source.field_surface_pressure)
    wind = pressure_level_wind(source.fields, source.latitude, source.longitude,
                               f["latCell"], f["lonCell"], f["latEdge"], f["lonEdge"],
                               f["angleEdge"], mesh.cells_on_edge, grid["zgrid"], grid_type=source.grid_type,
                               field_grids=source.field_grids, row_counts=source.row_counts,
                               missing_fields=set(source.field_surface_pressure or ()))
    w = vertical_wind(wind, state["rho_zz"], grid["zz"], grid["fzm"], grid["fzp"],
                       grid["zb"], grid["zb3"], mesh)
    reconstructed = reconstruct_wind(wind, grid["coeffs_reconstruct"], mesh, f["latCell"], f["lonCell"])
    keep = ("rho", "theta", "qv", "rho_base", "theta_base", "surface_pressure", "precipw")
    fields = {**grid, **reconstructed,
              **{name: state[name] for name in keep}, "u": wind, "w": w}
    fields["rdzu"] = torch.cat((grid["ter"].new_zeros(1), 1.0 / grid["dzu"][1:]))
    # The native dry-dycore profile uses the source's no-cloud cold start.
    for name in ("qc", "qr", "qi", "qs", "qg"):
        fields[name] = torch.zeros_like(state["qv"])
    fields["dss"] = torch.zeros_like(state["qv"])
    fields["t_init"] = torch.zeros_like(state["qv"])
    for name in ("u_init", "v_init", "qv_init"):
        fields[name] = grid["ter"].new_zeros(n_levels)
    return fields
