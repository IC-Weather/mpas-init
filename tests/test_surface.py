from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from mpas_init.surface import (SurfaceSource, initialize_surface, monthly_interpolation, noah_soil_layers,
                               SURFACE_FIELDS)


def test_monthly_calendar_matches_datetime_over_gregorian_cycle():
    monthly = torch.arange(36, dtype=torch.float64).reshape(3, 12).sin()
    for year in (1900, 1999, 2000, 2023, 2024, 2100):
        middle = [datetime(year, month, 15).timetuple().tm_yday for month in range(1, 13)]
        middle = [middle[0] - 31, *middle, middle[-1] + 31]
        date = datetime(year, 1, 1)
        while date.year == year:
            day = date.timetuple().tm_yday
            month = next(i for i in range(13) if middle[i] < day <= middle[i + 1])
            left, right = (11, 0) if month in (0, 12) else (month - 1, month)
            expected = ((monthly[..., right] * (day - middle[month])
                         + monthly[..., left] * (middle[month + 1] - day))
                        / (middle[month + 1] - middle[month]))
            assert torch.equal(monthly_interpolation(monthly, date), expected), date
            date += timedelta(days=1)


def test_soil_field_and_coordinate_derivatives():
    temp = (torch.arange(8, dtype=torch.float64).reshape(2, 4) + 270).requires_grad_()
    moist = (temp.detach() / 1000).requires_grad_()
    skin = torch.tensor([285., 275.], dtype=torch.float64, requires_grad=True)
    deep = torch.tensor([280., 270.], dtype=torch.float64, requires_grad=True)
    bounds = torch.tensor([[0., .2], [.2, .4], [.4, .8], [.8, 1.8]], dtype=torch.float64)
    land = torch.ones(2, dtype=torch.int64)
    f = lambda t, m, s, d: torch.cat(tuple(noah_soil_layers(t, m, bounds, s, d, land)[name]
                                           for name in ("tslb", "smois")), -1)
    assert torch.autograd.gradcheck(f, (temp, moist, skin, deep), check_forward_ad=True)
    for index in range(4):
        inputs = (temp, moist, skin, deep)
        directions = tuple(torch.ones_like(v) if i == index else torch.zeros_like(v)
                           for i, v in enumerate(inputs))
        tangent = torch.func.jvp(f, inputs, directions)[1]
        assert torch.count_nonzero(tangent) > 0


def small_surface_inputs():
    latitude = torch.linspace(-90, 90, 7, dtype=torch.float64)
    longitude = torch.arange(12, dtype=torch.float64) * 30
    lat, lon = torch.meshgrid(latitude, longitude, indexing="ij")
    wave = torch.sin(torch.deg2rad(lon)) + torch.deg2rad(lat) / 5
    fields = {"landmask": (lon < 180).to(torch.float64), "surface_altitude": 200 + 10 * wave,
              "skin_temperature": 280 + wave, "snow_water_equivalent": 20 + wave,
              "snow_depth": .2 + wave / 100, "sea_ice_fraction": .1 + wave / 100}
    fields["soil_temperature"] = torch.stack([275 + wave + k for k in range(4)])
    fields["soil_moisture"] = torch.stack([.2 + wave / 100 + k / 100 for k in range(4)])
    bounds = latitude.new_tensor([[0., .2], [.2, .4], [.4, .8], [.8, 1.8]])
    source = SurfaceSource(latitude, longitude, bounds, fields, datetime(2024, 5, 21, tzinfo=timezone.utc))
    static = {"latCell": latitude.new_tensor([-.65, .27, -.33, .71]),
              "lonCell": torch.deg2rad(latitude.new_tensor([43., 107., 223., 287.])),
              "landmask": torch.tensor([1, 1, 0, 0]), "ter": latitude.new_tensor([350., 280., 0., 0.]),
              "soiltemp": latitude.new_tensor([285., 280., 290., 295.]),
              "ivgtyp": torch.tensor([3, 4, 17, 17]), "isltyp": torch.tensor([5, 6, 14, 14]),
              "snoalb": latitude.new_tensor([.5, .6, .7, .8]),
              "greenfrac": torch.arange(48, dtype=torch.float64).reshape(4, 12) + 20,
              "albedo12m": torch.arange(48, dtype=torch.float64).reshape(4, 12) / 10 + 10}
    return source, static


@pytest.mark.parametrize("name", ["skin_temperature", "surface_altitude", "snow_depth",
                                  "snow_water_equivalent", "sea_ice_fraction",
                                  "soil_temperature", "soil_moisture"])
def test_missing_required_surface_support_rejected_before_fills(name):
    source, static = small_surface_inputs()
    source.fields[name].fill_(-1.e30)
    with pytest.raises(RuntimeError, match="no valid required surface source support: " + name):
        initialize_surface(source, static)


def test_surface_masks_and_soil_bounds_are_not_silently_repaired():
    source, static = small_surface_inputs()
    source.fields["landmask"][0, 0] = .5
    with pytest.raises(RuntimeError, match="source surface landmask must be binary"):
        initialize_surface(source, static)
    temp, moisture = torch.full((2, 4), 280.), torch.full((2, 4), .2)
    for bounds in ([[0., .1], [.1, .1], [.1, 1.], [1., 2.]],
                   [[0., .1], [.2, .4], [.4, 1.], [1., 2.]],
                   [[0., .1], [.1, .4], [.4, 1.], [1., 8.]]):
        with pytest.raises(RuntimeError, match="soil depth coverage"):
            noah_soil_layers(temp, moisture, torch.tensor(bounds), temp[:, 0], temp[:, 0], torch.ones(2))


def test_complete_surface_field_and_coordinate_jvp_vjp(tmp_path):
    source, static = small_surface_inputs()
    names = tuple(sorted(set(source.fields) - {"landmask"}))
    static_names = ("ter", "soiltemp", "latCell", "lonCell", "greenfrac", "albedo12m")
    inputs = tuple(v.clone().requires_grad_() for v in
                   (*(source.fields[name] for name in names), *(static[name] for name in static_names)))
    # Keep selections fixed. The coordinate perturbations stay well inside each stencil.
    directions = tuple(torch.cos(torch.arange(v.numel(), dtype=v.dtype)).reshape(v.shape)
                       * (.01 if name in ("soil_moisture", "snow_depth", "sea_ice_fraction", "latCell", "lonCell") else 1)
                       for name, v in zip((*names, *static_names), inputs))

    def evaluate(*values):
        fields = {**source.fields, **dict(zip(names, values[:len(names)]))}
        surface = initialize_surface(SurfaceSource(source.latitude, source.longitude, source.soil_bounds,
                                                    fields, source.valid_time),
                                     {**static, **dict(zip(static_names, values[len(names):]))})
        return torch.cat([surface[name].flatten() for name in sorted(SURFACE_FIELDS)
                          if surface[name].is_floating_point()])

    output, tangent = torch.func.jvp(evaluate, inputs, directions)
    weights = torch.linspace(.3, 1.2, output.numel(), dtype=output.dtype)
    gradient = torch.autograd.grad((output * weights).sum(), inputs)
    reverse = sum((g * d).sum() for g, d in zip(gradient, directions))
    torch.testing.assert_close(reverse, (tangent * weights).sum(), rtol=1e-12, atol=1e-12)
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in gradient)
    finite = []
    for step in (1e-3, 1e-4, 1e-5):
        plus = evaluate(*(v + step * d for v, d in zip(inputs, directions)))
        minus = evaluate(*(v - step * d for v, d in zip(inputs, directions)))
        value = (plus - minus) / (2 * step)
        finite.append(value.detach().numpy())
        torch.testing.assert_close(tangent, value, rtol=2e-6, atol=2e-8)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(torch.zeros_like(tangent), value, rtol=2e-6, atol=2e-8)
    np.savez(tmp_path / "derivatives.npz", tangent=tangent.detach().numpy(),
             finite=np.stack(finite), reverse=reverse.detach().numpy(),
             **{f"gradient_{name}": grad.numpy() for name, grad in zip((*names, *static_names), gradient)})
    wrong = tmp_path / "zeroed-gradient.npz"
    np.savez(wrong, tangent=np.zeros(tangent.shape))
    with np.load(wrong) as packet:
        assert not np.array_equal(packet["tangent"], tangent.detach().numpy())
        for reference in finite:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(torch.tensor(packet["tangent"]), torch.tensor(reference), rtol=2e-6, atol=2e-8)
    wrong.unlink()
