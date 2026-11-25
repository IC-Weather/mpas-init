import pytest

from mpas_init.prepare import prepare_static
from test_categories import miniature_categories
from test_climatology import miniature_climatology
from test_raw_mesh import miniature_terrain
from test_soil import miniature_soil_temperature


def geography(root):
    root.mkdir()
    paths, pixels = miniature_terrain(root / "topo_gmted2010_30s")
    miniature_categories(root / "modis_landuse_20class_30s", 20)
    miniature_categories(root / "soiltype_top_30s", 16)
    miniature_categories(root / "landuse_30s", 24)
    for name, field in (("maxsnowalb_modis", "snoalb"), ("greenfrac_fpar_modis", "greenfrac"),
                        ("albedo_modis", "albedo12m"), ("lai_modis_30s", "lai12m")):
        miniature_climatology(root / name, field)
    miniature_soil_temperature(root / "soiltemp_1deg")
    return paths, pixels


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
@pytest.mark.parametrize("control", ["supersample", "landuse_supersample", "soil_supersample",
                                     "vegetation_supersample", "block_side", "row_chunk"])
def test_prepare_rejects_invalid_controls_before_input_reads(tmp_path, bad, control):
    with pytest.raises(ValueError, match="positive integers"):
        prepare_static("absent", "absent", tmp_path / "static.nc", **{control: bad})
    assert not (tmp_path / "static.nc").exists()


def test_prepare_rejects_unknown_profile_and_existing_symlinks(tmp_path):
    with pytest.raises(ValueError, match="supports dycore"):
        prepare_static("absent", "absent", tmp_path / "static.nc", profile="unknown")
    output = tmp_path / "static.nc"
    output.symlink_to(tmp_path / "absent-target")
    with pytest.raises(FileExistsError):
        prepare_static("absent", "absent", output)
    assert output.is_symlink() and not output.exists()
