import math
from types import SimpleNamespace

import pytest
import torch

from mpas_init.geometry import deformation_coefficients, prepare_mesh_geometry
from mpas_init.vertical_grid import GlobalConnectivity
from test_vertical_grid import assert_stage


def quadrilateral(slots=4):
    counts = torch.tensor([4])
    return GlobalConnectivity(torch.zeros((1, slots), dtype=torch.int64),
                              torch.tensor([[0, 1, 2, 3] + [0] * (slots - 4)]), counts,
                              torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64),
                              torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]))


def test_deformation_polar_tangent_polygon():
    radius = 6371229.0
    center = torch.tensor([[0., 0., radius]], dtype=torch.float64)
    vertices = torch.tensor([[0.5 * math.cos(t), 0.5 * math.sin(t), math.sqrt(3) / 2]
                             for t in (0., math.pi / 2, math.pi, 3 * math.pi / 2)], dtype=torch.float64) * radius
    result = deformation_coefficients(quadrilateral(), center, vertices, torch.arange(4)[None], radius)
    length = radius * math.pi / 6
    expected_x = torch.tensor([[-1., -1., 1., 1.]], dtype=torch.float64) / (2 * length)
    expected_y = torch.tensor([[1., -1., -1., 1.]], dtype=torch.float64) / (2 * length)
    expected_b = torch.tensor([[-1., 1., -1., 1.]], dtype=torch.float64) / (math.sqrt(2) * length)
    for name, expected in (("defc_a", torch.zeros_like(expected_b)), ("defc_b", expected_b),
                            ("cell_gradient_coef_x", expected_x), ("cell_gradient_coef_y", expected_y)):
        assert_stage(result[name], expected, 256)


def test_deformation_padding_and_continuous_derivatives():
    center = torch.tensor([[1., 0., 0.]], dtype=torch.float64, requires_grad=True)
    vertices = torch.tensor([[1., -0.2, 0.3], [1., -0.3, -0.2], [1., 0.2, -0.3], [1., 0.3, 0.2]],
                             dtype=torch.float64)
    vertices = (vertices / torch.linalg.vector_norm(vertices, dim=-1, keepdim=True)).requires_grad_()
    def compute(c, v):
        result = deformation_coefficients(quadrilateral(6), c, v, torch.tensor([[0, 1, 2, 3, 0, 0]]), 1.)
        return torch.stack(tuple(result.values()))
    result = compute(center, vertices)
    assert torch.isfinite(result).all() and torch.equal(result[..., 4:], torch.zeros_like(result[..., 4:]))
    assert torch.autograd.gradcheck(compute, (center, vertices), check_forward_ad=True)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "160000"])
def test_geometry_rejects_invalid_explicit_nominal_length(value):
    with pytest.raises(ValueError, match="finite positive length in metres"):
        prepare_mesh_geometry(SimpleNamespace(fields={}), nominal_min_dc=value)


def miniature_terrain(directory):
    directory.mkdir()
    (directory / "index").write_text("""type=continuous
projection=regular_ll
dx=7.5
dy=7.5
known_x=1
known_y=1
known_lat=-86.25
known_lon=3.75
tile_x=12
tile_y=12
tile_z=1
tile_bdr=1
wordsize=2
signed=yes
units="meters MSL"
""")
    from mpas_init.geotile import GeotileIndex
    index = GeotileIndex.read(directory / "index")
    paths, pixels = [], []
    for y in range(1, 25, 12):
        for x in range(1, 49, 12):
            path = directory / index.filename(x, y)
            values = []
            for j in range(-1, 13):
                for i in range(-1, 13):
                    value = (y - 1 + j) * 48 + x - 1 + i - 128
                    if 0 <= i < 12 and 0 <= j < 12:
                        pixels.append((-86.25 + (y - 1 + j) * 7.5, 3.75 + (x - 1 + i) * 7.5, value))
                    else:
                        value = 9999
                    values.append(value)
            path.write_bytes(b"".join(v.to_bytes(2, "big", signed=True) for v in values))
            paths.append(path)
    return paths, pixels
