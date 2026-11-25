import pytest
import torch

from mpas_init.geotile import GeotileIndex, read_geotile


def write_index(path, *, wordsize=2, signed="yes", endian="big", extra=""):
    path.write_text(f"""# Independent miniature global geotile
type=continuous
projection=regular_ll
dx=90
dy=90
known_x=1
known_y=1
known_lat=-45
known_lon=45
tile_x=4
tile_y=2
tile_z=2
tile_bdr=1
wordsize={wordsize}
signed={signed}
endian={endian}
scale_factor=0.1
missing_value=7
units="meters MSL"
{extra}""")


@pytest.mark.parametrize("wordsize", [1, 2, 3, 4])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("endian", ["big", "little"])
def test_integer_codec_extremes_borders_and_layers(tmp_path, wordsize, signed, endian):
    from mpas_init._geotile_reader import GeotileReader

    path = tmp_path / "index"
    write_index(path, wordsize=wordsize, signed="yes" if signed else "no", endian=endian)
    index = GeotileIndex.read(path)
    bits = 8 * wordsize
    lower, upper = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    truth = [lower, upper, 0, 7, 1, -1 if signed else 2] * 8
    tile_path = tmp_path / index.filename(1, 1)
    tile_path.write_bytes(b"".join(v.to_bytes(wordsize, endian, signed=signed) for v in truth))
    complete = read_geotile(tile_path, index, include_border=True)
    expected = torch.tensor(truth, dtype=torch.int64).reshape(2, 4, 6)
    assert torch.equal(complete.raw, expected)
    assert torch.equal(complete.valid, expected != 7)
    interior = read_geotile(tile_path, index)
    assert torch.equal(interior.raw, expected[:, 1:-1, 1:-1])
    assert torch.equal(interior.valid, interior.raw != 7)
    # Decoding preserves raw integers; applying physical scale belongs to the tensor stage.
    assert interior.raw.dtype == torch.int64 and index.scale_factor == 0.1
    with GeotileReader(tile_path, index) as reader:
        for window in ((0, 0, 2, 4), (1, 1, 1, 2)):
            row, column, height, width = window
            slab = reader.read(window=window, layers=slice(1, 2))
            truth = expected[1:2, 1 + row:1 + row + height, 1 + column:1 + column + width]
            assert torch.equal(slab.raw, truth)
            assert torch.equal(slab.valid, truth != 7)
        bordered = reader.read(include_border=True)
        assert torch.equal(bordered.raw, expected)
        rows, columns = torch.tensor([1, 0, 1]), torch.tensor([3, 0, 3])
        points = reader.points(rows, columns)
        assert torch.equal(points.raw, expected[:, rows + 1, columns + 1])
        assert torch.equal(points.valid, points.raw != 7)
    assert torch.equal(bordered.raw, expected)  # Returned buffers do not retain the mapping.
    with pytest.raises(ValueError, match='closed'):
        reader.read()
    for invalid in (tile_path.read_bytes()[:-1], tile_path.read_bytes() + b"\0"):
        broken = tmp_path / "broken"
        broken.write_bytes(invalid)
        with pytest.raises(ValueError, match="incomplete or oversized"):
            read_geotile(broken, index)


def test_index_defaults_layer_bounds_and_rejections(tmp_path):
    path = tmp_path / "index"
    write_index(path)
    text = path.read_text().replace("tile_z=2", "tile_z_start=3\ntile_z_end=4")
    text = text.replace("signed=yes\n", "").replace("endian=big\n", "").replace("tile_bdr=1\n", "")
    path.write_text(text)
    index = GeotileIndex.read(path)
    assert index.layers == 2 and index.tile_z_start == 3
    assert index.endian == "big" and not index.signed and index.tile_bdr == 0
    assert index.global_shape == (4, 2)
    assert index.metadata["units"] == '"meters MSL"'
    for before, after in (("dx=90", "dx=0"), ("dy=90", "dy=nan"),
                          ("tile_x=4", "tile_x=3"), ("known_x=1", "known_x=2"),
                          ("regular_ll", "lambert"), ("wordsize=2", "wordsize=8"),
                          ("type=continuous", "type=categorical")):
        path.write_text(text.replace(before, after))
        with pytest.raises(ValueError):
            GeotileIndex.read(path)
    path.write_text(text + "dx=90\n")
    with pytest.raises(ValueError, match="invalid geotile index"):
        GeotileIndex.read(path)
    for x, y in ((0, 1), (2, 1), (5, 1), (1, 0), (1, 3)):
        with pytest.raises(ValueError, match="tile starts"):
            index.filename(x, y)


def test_descending_latitude_index_and_category_range(tmp_path):
    path = tmp_path / "index"
    write_index(path)
    text = path.read_text().replace("dy=90", "dy=-90").replace("known_lat=-45", "known_lat=45")
    text = text.replace("type=continuous", "type=categorical\ncategory_min=2\ncategory_max=20")
    path.write_text(text + "row_order=top_bottom\n")
    index = GeotileIndex.read(path)
    assert index.dy == -90 and index.category_min == 2 and index.category_max == 20
    path.write_text(text + "row_order=bottom_top\n")
    with pytest.raises(ValueError, match="row_order"):
        GeotileIndex.read(path)


@pytest.mark.parametrize("missing", ["4294967295", "4294967294.5", "1e100", "-1"])
def test_missing_mask_preserves_adjacent_large_integers(tmp_path, missing):
    from mpas_init._geotile_reader import GeotileReader

    path = tmp_path / "index"
    write_index(path, wordsize=4, signed="no")
    path.write_text(path.read_text().replace("missing_value=7", f"missing_value={missing}"))
    index = GeotileIndex.read(path)
    truth = [4294967294, 4294967295] * 24
    tile_path = tmp_path / index.filename(1, 1)
    tile_path.write_bytes(b"".join(value.to_bytes(4, "big") for value in truth))
    tile = read_geotile(tile_path, index, include_border=True)
    expected = [True, False] * 24 if missing == "4294967295" else [True] * 48
    assert torch.equal(tile.valid, torch.tensor(expected).reshape(2, 4, 6))
    with GeotileReader(tile_path, index) as reader:
        actual = reader.read(window=(0, 1, 2, 2), layers=slice(1, 2))
        assert torch.equal(actual.raw, tile.raw[1:2, 1:3, 2:4])
        assert torch.equal(actual.valid, tile.valid[1:2, 1:3, 2:4])


@pytest.mark.parametrize('arguments', [
    {'window': (-1, 0, 1, 1)}, {'window': (0, 0, 0, 1)},
    {'window': (1, 0, 2, 1)}, {'window': (0, 3, 1, 2)},
    {'window': (0, 0, 1)}, {'window': (False, 0, 1, 1)},
    {'window': (0, 0, 1., 1)}, {'layers': slice(-1, 2)},
    {'layers': slice(1, 1)}, {'layers': slice(0, 3)},
    {'layers': slice(0, 2, 2)}, {'layers': slice(False, 2)},
    {'layers': 1},
])
def test_bounded_reader_rejects_clipped_or_ambiguous_selections(tmp_path, arguments):
    from mpas_init._geotile_reader import GeotileReader

    write_index(tmp_path / 'index')
    index = GeotileIndex.read(tmp_path / 'index')
    path = tmp_path / index.filename(1, 1)
    path.write_bytes(bytes(96))
    with GeotileReader(path, index) as reader:
        with pytest.raises(ValueError, match='window|layer'):
            reader.read(**arguments)
