import pytest
import torch

from mpas_init.categories import category_counts, category_mode_fractions, surface_categories
from mpas_init.geotile import GeotileIndex


def surface_pixels():
    land = [[1, 1, 1, 1, 1, 0, 21], [17, 17, 17, 17, 17, 0, 21], [15, 15, 15, 15, 15, 0, 21],
            [2, 2, 2, 2, 2, 0, 21], [3, 4, 3, 4, 5, 0, 21]]
    soil = [[14, 14, 14, 14, 14, 0, 17], [1, 1, 1, 1, 1, 0, 17], [14, 14, 14, 14, 14, 0, 17],
            [16, 16, 16, 16, 16, 0, 17], [2, 3, 2, 3, 4, 0, 17]]
    return torch.tensor(land).reshape(-1), torch.tensor(soil).reshape(-1), torch.arange(5).repeat_interleave(7)


def test_category_counts_offset_ties_empty_and_chunk_invariance():
    pixels = torch.tensor([-1, 2, 3, 4, 3, 4, 5, 5, 6])
    cells = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
    expected = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 0]])
    counts = category_counts(pixels, cells, 4, 3, 5)
    assert torch.equal(counts, expected)
    first = category_counts(pixels[:4], cells[:4], 4, 3, 5)
    second = category_counts(pixels[4:], cells[4:], 4, 3, 5)
    assert torch.equal(counts, first + second)
    mode, fractions = category_mode_fractions(counts, 3)
    assert mode.tolist() == [3, 3, 5, 3]
    assert torch.equal(fractions[:, :3], torch.tensor([[0.5, 0.5, 0.], [0.5, 0.5, 0.], [0., 0., 1.]],
                                                     dtype=torch.float64))
    assert torch.isnan(fractions[:, 3]).all()
    for bad in (pixels.float(), pixels[:, None]):
        with pytest.raises(ValueError, match="matching int64 vectors"):
            category_counts(bad, cells, 4, 3, 5)
    with pytest.raises(ValueError, match="range or cell count"):
        category_counts(pixels, cells, 4, 5, 3)
    with pytest.raises(ValueError, match="category-by-cell"):
        category_mode_fractions(counts.float(), 3)


def test_ordered_surface_corrections_keep_original_fractions():
    land, soil, cells = surface_pixels()
    lc, sc = category_counts(land, cells, 5, 1, 20), category_counts(soil, cells, 5, 1, 16)
    result = surface_categories(lc, sc, land_min=1, soil_min=1, land_water=17, land_ice=15, soil_water=14)
    assert result["lu_index"].tolist() == [17, 17, 15, 2, 3]
    assert result["soilcat_top"].tolist() == [14, 14, 16, 16, 2]
    assert result["landmask"].tolist() == [0, 0, 1, 1, 1]
    assert result["landusef"][0, 0] == 1 and result["landusef"][16, 0] == 0
    assert result["soilf"][13, 2] == 1 and result["soilf"][15, 2] == 0
    assert torch.equal(result["landusef"].sum(0), torch.ones(5, dtype=torch.float64))
    assert torch.equal(result["soilf"].sum(0), torch.ones(5, dtype=torch.float64))


def miniature_categories(directory, n_categories):
    directory.mkdir()
    (directory / "index").write_text(f"""type=categorical
category_min=1
category_max={n_categories}
projection=regular_ll
dx=7.5
dy=7.5
known_x=1
known_y=1
known_lat=-86.25
known_lon=-176.25
tile_x=12
tile_y=12
tile_z=1
tile_bdr=1
wordsize=1
missing_value=255
""" + ('iswater=17\nisice=15\nmminlu="MODIFIED_IGBP_MODIS_NOAH"\n' if n_categories == 20 else ""))
    index = GeotileIndex.read(directory / "index")
    pixels = []
    for y in range(1, 25, 12):
        for x in range(1, 49, 12):
            values = []
            for j in range(-1, 13):
                for i in range(-1, 13):
                    value = 255
                    if 0 <= i < 12 and 0 <= j < 12:
                        row, col = y - 1 + j, x - 1 + i
                        value = 1 + (row * 11 + col * 7) % n_categories
                        if (row * 48 + col) % 37 == 0:
                            value = 0
                        pixels.append((-86.25 + row * 7.5, -176.25 + col * 7.5, value))
                    values.append(value)
            (directory / index.filename(x, y)).write_bytes(bytes(values))
    return pixels
