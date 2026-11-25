"""The original MPAS dry base state and discrete moist hydrostatic adjustment."""

import torch
from torch import Tensor

from mpas_init.arithmetic import _exp, _power

def hydrostatic_state(temperature: Tensor, pressure: Tensor, qv: Tensor,
                      zgrid: Tensor, zz: Tensor, fzm: Tensor, fzp: Tensor,
                      dzu: Tensor, rdzw: Tensor) -> dict[str, Tensor]:
    """Construct cell-major native thermodynamic fields from interpolated fields.

    qv is water-vapor mass per dry-air mass, not specific humidity. The original
    bottom pressure stays fixed. Each higher level takes at most 30 iterations.
    Converged columns stop updating through tensor masks, without host decisions.
    """
    p0, rgas, cp, cv, gravity, t0b = (temperature.new_tensor(v) for v in
                                     (100000.0, 287.0, 1004.5, 717.5, 9.80616, 250.0))
    rvord = temperature.new_tensor(461.6) / rgas
    initial_exner = _power(pressure / p0, rgas / cp)
    theta = temperature * _power(p0 / pressure, rgas / cp)
    dry_density = pressure / rgas / (initial_exner * theta * (1.0 + (rvord - 1.0) * qv)) / (1.0 + qv)
    precipw = torch.zeros_like(pressure[..., 0])
    for k in range(pressure.shape[-1]):
        precipw = precipw + dry_density[..., k] * qv[..., k] * (zgrid[..., k+1] - zgrid[..., k])

    midpoint = 0.5 * (zgrid[..., 1:] + zgrid[..., :-1])
    pressure_base = p0 * _exp(-gravity * midpoint / (rgas * t0b))
    exner_base = _power(pressure_base / p0, rgas / cp)
    rho_base_uncoupled = pressure_base / (rgas * t0b)
    theta_base = t0b / exner_base
    rtheta_base = rho_base_uncoupled * theta_base
    rho_base = rho_base_uncoupled / zz
    initial_rho = dry_density / zz
    initial_pp = pressure - pressure_base
    initial_rr = initial_rho - rho_base

    rho = _power(pressure[..., 0] / p0, cv / cp) * (p0 / rgas)
    rho = rho / (theta[..., 0] * (1.0 + 1.61 * qv[..., 0])) / zz[..., 0]
    densities = [rho]
    perturbations = [initial_pp[..., 0]]
    density_perturbations = [rho - rho_base[..., 0]]
    pressures = [pressure[..., 0]]
    exners = [exner_base[..., 0]]
    threshold = temperature.new_tensor(0.0001)
    for k in range(1, pressure.shape[-1]):
        pp = initial_pp[..., k]
        rho, rr = initial_rho[..., k], initial_rr[..., k]
        p, exner = pressure[..., k], exner_base[..., k]
        active = torch.ones_like(p, dtype=torch.bool)
        for _ in range(30):
            proposed_pp = (perturbations[-1]
                           - (fzm[k] * rr + fzp[k] * density_perturbations[-1]) * gravity * dzu[k]
                           - (fzm[k] * rho * qv[..., k]
                              + fzp[k] * densities[-1] * qv[..., k-1]) * gravity * dzu[k])
            proposed_p = proposed_pp + pressure_base[..., k]
            proposed_exner = _power(proposed_p / p0, rgas / cp)
            proposed_rho = proposed_p / rgas / (proposed_exner * theta[..., k] * (1.0 + 1.61 * qv[..., k])) / zz[..., k]
            proposed_rr = proposed_rho - rho_base[..., k]
            still_active = active & ((pp - proposed_pp).abs() > threshold)
            pp, p = torch.where(active, proposed_pp, pp), torch.where(active, proposed_p, p)
            exner = torch.where(active, proposed_exner, exner)
            rho, rr = torch.where(active, proposed_rho, rho), torch.where(active, proposed_rr, rr)
            active = still_active
        densities.append(rho)
        perturbations.append(pp)
        density_perturbations.append(rr)
        pressures.append(p)
        exners.append(exner)
    rho_zz = torch.stack(densities, dim=-1)
    pressure_p = torch.stack(perturbations, dim=-1)
    rho_p = torch.stack(density_perturbations, dim=-1) * zz
    theta_m = theta * (1.0 + 1.61 * qv)
    surface_pressure = 0.5 * gravity / rdzw[0] * (
        1.25 * rho_zz[..., 0] * (1.0 + qv[..., 0]) - 0.25 * rho_zz[..., 1] * (1.0 + qv[..., 1]))
    surface_pressure = surface_pressure + pressure_p[..., 0] + pressure_base[..., 0]
    return {"pressure": torch.stack(pressures, dim=-1), "pressure_base": pressure_base,
            "pressure_p": pressure_p, "exner": torch.stack(exners, dim=-1),
            "exner_base": exner_base, "rho_zz": rho_zz, "rho_base": rho_base,
            "rho_p": rho_p, "theta_m": theta_m, "theta_base": theta_base,
            "rtheta_base": rtheta_base, "surface_pressure": surface_pressure,
            "rho": rho_zz * zz, "theta": theta_m / (1.0 + 1.61 * qv), "precipw": precipw}
