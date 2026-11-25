import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpas_init._geography_index import convert_terrain, indexed_manifest
from mpas_init._geotile_reader import GeotileReader
from mpas_init.geotile import GeotileIndex
from test_geotile import write_index


@pytest.mark.parametrize('budget', [24, 240, 2400])
def test_sparse_monthly_reads_use_bounded_contiguous_stripes(monkeypatch, budget):
    from mpas_init._geography_index import IndexedTile

    raw = np.arange(2 * 12 * 180 * 180, dtype=np.int32).reshape(2, 12, 180, 180)
    requests = []

    class Variable:
        shape, dtype = raw.shape, raw.dtype

        def chunking(self):
            return (1, 12, 128, 128)

        def __getitem__(self, selection):
            assert all(isinstance(v, slice) for v in selection[1:])
            result = raw[selection]
            assert result.size <= budget
            requests.append(result.size)
            return result

    store = object.__new__(IndexedTile)
    store.raw, store.number = Variable(), 1
    store.raw_values_requested = store.raw_read_calls = 0
    rows = np.array([[179, 0, 129, 0], [64, 64, 130, 2]])
    columns = np.array([[0, 179, 4, 179], [3, 171, 170, 0]])
    monkeypatch.setattr('mpas_init._geography_index.MAX_PAIRED_SAMPLES', budget)
    actual = store.values((slice(2, 8), rows, columns))
    assert np.array_equal(actual, raw[1, 2:8, rows, columns].transpose(2, 0, 1))
    assert store.raw_read_calls == len(requests)
    assert store.raw_values_requested == sum(requests)


@pytest.mark.parametrize('wordsize', [1, 2, 3, 4])
@pytest.mark.parametrize('signed', [False, True])
@pytest.mark.parametrize('endian', ['big', 'little'])
def test_lossless_index_raw_borders_native_quantization(tmp_path, wordsize, signed, endian, monkeypatch):
    source = tmp_path / 'raw'
    source.mkdir()
    write_index(source / 'index', wordsize=wordsize, signed='yes' if signed else 'no', endian=endian)
    text = (source / 'index').read_text().replace('tile_z=2', 'tile_z=1').replace('missing_value=7', 'missing_value=-1e100')
    (source / 'index').write_text(text)
    index = GeotileIndex.read(source / 'index')
    bits = 8 * wordsize
    lower, upper = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    truth = [lower, upper, 0, 7, 1, 0] * 4
    tile = source / index.filename(1, 1)
    tile.write_bytes(b''.join(v.to_bytes(wordsize, endian, signed=signed) for v in truth))
    target = tmp_path / 'indexed'
    record = convert_terrain(source, target, radius=1., chunk_side=256, compress=True)
    assert record['sources'][0]['name'] == tile.name
    assert (source / 'index').read_bytes() == (target / 'index').read_bytes()
    # The index remains usable after the original paths disappear.
    tile.unlink()
    with GeotileReader(target / tile.name, index) as reader:
        expected = torch.tensor(truth, dtype=torch.int64).reshape(1, 4, 6)
        assert torch.equal(reader.read(include_border=True).raw, expected)
        assert torch.equal(reader.read().raw, expected[:, 1:-1, 1:-1])
        rows, cols = torch.tensor([1, 0, 1]), torch.tensor([3, 0, 3])
        assert torch.equal(reader.points(rows, cols).raw, expected[:, rows + 1, cols + 1])
        monkeypatch.setattr('mpas_init._geography_index.MAX_PAIRED_SAMPLES', 4)
        assert torch.equal(reader.points(rows.reshape(1, 3), cols.reshape(1, 3)).raw,
                           expected[:, (rows + 1).reshape(1, 3), (cols + 1).reshape(1, 3)])
        for side in (128, 64, 32):
            sums, counts, _ = reader._indexed.level(side)
            assert int(sums.sum()) == int(expected[:, 1:-1, 1:-1].to(torch.float32).to(torch.int64).sum())
            assert int(counts.sum()) == 8
    with pytest.raises(FileExistsError):
        convert_terrain(source, target, tiles=[(1, 1, tile)])


def _partial_source(tmp_path, width=120, height=60, *, convert=True):
    source = tmp_path / 'raw'
    source.mkdir()
    (source / 'index').write_text(f'''type=continuous
projection=regular_ll
dx=0.5
dy=0.5
known_x=1
known_y=1
known_lat=-89.75
known_lon=-179.75
tile_x={width}
tile_y={height}
tile_z=1
tile_bdr=1
wordsize=4
signed=yes
endian=big
scale_factor=0.1
''')
    index = GeotileIndex.read(source / 'index')
    raw = np.arange((height + 2) * (width + 2), dtype=np.int32).reshape(1, height + 2, width + 2) * 7919 - 1000000
    tile = source / index.filename(1, 1)
    tile.write_bytes(raw.astype('>i4').tobytes())
    target = tmp_path / 'indexed'
    if convert:
        convert_terrain(source, target, tiles=[(1, 1, tile)], radius=1.)
    return index, raw, target, tile.name


@pytest.mark.parametrize('capacity', [1, 2])
def test_indexed_container_reuse_eviction_and_exception_cleanup(tmp_path, monkeypatch, capacity):
    from mpas_init import _geography_index as indexed
    from mpas_init._geotile_reader import GeotileReaders

    index, raw, target, name = _partial_source(tmp_path, 4, 2, convert=False)
    source = tmp_path / 'raw'
    second = source / index.filename(5, 1)
    second.write_bytes((raw + 123).astype('>i4').tobytes())
    tiles = [(1, 1, source / name), (5, 1, second)]
    targets = [target, tmp_path / 'indexed2', tmp_path / 'indexed3']
    for directory in targets:
        convert_terrain(source, directory, tiles=tiles, radius=1.)
    handles, original = [], indexed.Dataset

    def opened(*args, **kwargs):
        ds = original(*args, **kwargs)
        handles.append(ds)
        return ds

    monkeypatch.setattr(indexed, 'Dataset', opened)
    with GeotileReaders(capacity) as cache:
        for tile_name, expected in ((name, raw), (second.name, raw + 123), (name, raw)):
            reader = cache._reader(target / tile_name, index)
            assert torch.equal(reader.read(include_border=True).raw, torch.from_numpy(expected.astype(np.int64)))
        assert len(handles) == 1
        for directory in targets[1:]:
            cache.read(directory / name, index, window=(0, 0, 1, 1))
            assert len(cache._indexed_sources) <= capacity
            assert sum(ds.isopen() for ds in handles) <= capacity
    assert not any(ds.isopen() for ds in handles)
    with GeotileReaders(capacity) as cache:
        with pytest.raises(ValueError, match='indexed tile'):
            cache.read(target / 'absent', index, window=(0, 0, 1, 1))
        assert not cache._indexed_sources
    assert not any(ds.isopen() for ds in handles)


@pytest.mark.parametrize('damage', ['building', 'manifest', 'metadata', 'payload'])
def test_recognized_collection_rejects_damage_without_fallback(tmp_path, damage):
    _, _, target, name = _partial_source(tmp_path)
    if damage == 'building':
        (target / 'geography.json').write_text('{"schema":1,"status":"building"}')
    elif damage == 'manifest':
        record = json.loads((target / 'geography.json').read_text())
        record['radius'] = 2.
        (target / 'geography.json').write_text(json.dumps(record))
    elif damage == 'metadata':
        with (target / 'index').open('a') as stream:
            stream.write('# changed\n')
    else:
        with (target / 'data.nc').open('ab') as stream:
            stream.write(b'x')
    with pytest.raises(ValueError, match='index|manifest|metadata|payload'):
        indexed_manifest(target)


@pytest.mark.parametrize('record', [None, 1, [], 'ready'])
def test_recognized_manifest_requires_an_object(tmp_path, record):
    (tmp_path / 'geography.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='manifest'):
        indexed_manifest(tmp_path)


def test_interrupted_collection_rejects_before_mesh_or_numerical_work(tmp_path, monkeypatch):
    from mpas_init.prepare import prepare_geography, prepare_static
    from mpas_init._geography_index import collection_manifest
    from test_prepare_cli import geography

    source, target = tmp_path / 'raw', tmp_path / 'indexed'
    geography(source)
    import mpas_init._geography_index as indexed
    original = indexed.convert_dataset
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('interrupted conversion')
        return original(*args, **kwargs)
    monkeypatch.setattr(indexed, 'convert_dataset', interrupted)
    with pytest.raises(RuntimeError, match='interrupted conversion'):
        prepare_geography(source, target, profile='native-noah')
    assert indexed_manifest(target / 'topo_gmted2010_30s')['status'] == 'ready'
    with pytest.raises(ValueError, match='manifest'):
        collection_manifest(target)
    with pytest.raises(ValueError, match='manifest'):
        prepare_static('must-not-read-mesh', target, tmp_path / 'static.nc')
    assert not (tmp_path / 'static.nc').exists()


def test_conversion_rejects_source_metadata_change_before_publication(tmp_path, monkeypatch):
    import mpas_init._geography_index as indexed

    source = tmp_path / 'raw'
    source.mkdir()
    write_index(source / 'index')
    (source / 'index').write_text((source / 'index').read_text().replace('tile_z=2', 'tile_z=1').replace('missing_value=7', 'missing_value=9999'))
    index = GeotileIndex.read(source / 'index')
    path = source / index.filename(1, 1)
    path.write_bytes(np.arange(24, dtype='>i2').tobytes())
    original = indexed._verify_raw
    def changed(*args):
        original(*args)
        with (source / 'index').open('a') as stream:
            stream.write('# changed source metadata\n')
    monkeypatch.setattr(indexed, '_verify_raw', changed)
    target = tmp_path / 'indexed'
    with pytest.raises(ValueError, match='metadata changed'):
        convert_terrain(source, target)
    with pytest.raises(ValueError, match='manifest'):
        indexed_manifest(target)


def test_conversion_rejects_source_change_during_final_verification(tmp_path, monkeypatch):
    import mpas_init._geography_index as indexed

    source = tmp_path / 'raw'
    source.mkdir()
    write_index(source / 'index')
    text = (source / 'index').read_text().replace('tile_z=2', 'tile_z=1').replace('missing_value=7', 'missing_value=9999')
    (source / 'index').write_text(text)
    index = GeotileIndex.read(source / 'index')
    path = source / index.filename(1, 1)
    path.write_bytes(np.arange(24, dtype='>i2').tobytes())
    original = indexed._verify_raw
    def changed(*args):
        original(*args)
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    monkeypatch.setattr(indexed, '_verify_raw', changed)
    target = tmp_path / 'indexed'
    with pytest.raises(ValueError, match='source tile changed'):
        convert_terrain(source, target)
    with pytest.raises(ValueError, match='manifest'):
        indexed_manifest(target)


@pytest.mark.parametrize('kind', ['terrain', 'category', 'snoalb'])
def test_accumulator_bounds_reject_huge_sampling_before_pixel_work(tmp_path, kind):
    from mpas_init.prepare import terrain_reductions, category_reductions, climatology_reductions

    write_index(tmp_path / 'index')
    text = (tmp_path / 'index').read_text().replace('tile_z=2', 'tile_z=1')
    if kind == 'category':
        text = text.replace('type=continuous', 'type=categorical\ncategory_min=1\ncategory_max=4').replace('scale_factor=0.1', 'scale_factor=1')
    (tmp_path / 'index').write_text(text)
    index = GeotileIndex.read(tmp_path / 'index')
    (tmp_path / index.filename(1, 1)).write_bytes(bytes(48))
    mesh = SimpleNamespace(sizes={'nCells': 1})
    mapping = SimpleNamespace(centers=torch.zeros((1, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match='accumulator bound'):
        if kind == 'terrain':
            terrain_reductions(mesh, tmp_path, supersample=2**40, _mapping=mapping)
        elif kind == 'category':
            category_reductions(mesh, tmp_path, supersample=2**40, _mapping=mapping)
        else:
            climatology_reductions(mesh, tmp_path, torch.ones(1, dtype=torch.int64), kind, supersample=2**40, _mapping=mapping)
