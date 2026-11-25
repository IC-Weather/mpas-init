"""Local lossless geography storage and fixed, mesh-independent terrain summaries.

NetCDF access and collection publication are offline I/O. Integer summaries and
coordinate enclosures use the same PyTorch operations as preparation.
"""

import hashlib
import json
from pathlib import Path

from netCDF4 import Dataset
import torch

from mpas_init.geotile import GeotileIndex
from mpas_init._mapping import block_points, enclosing_blocks, geometry_descriptor, block_owners
from mpas_init.static import pixel_coordinates, pixel_cartesian, pixel_cartesian_pairs, terrain_accumulation
from mpas_init.categories import category_counts
from mpas_init.climatology import land_accumulation


MANIFEST = 'geography.json'
COLLECTION = 'geography-collection.json'
RECIPE = 'native-f32-i64;sphere-all-points;integer-summaries-v1'
SIDES = (128, 64, 32)
CACHE_BYTES = 2 * 1024**2
MAX_PAIRED_SAMPLES = 2 * 1024**2


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _ready_record(path):
    try:
        record = json.loads(Path(path).read_text())
        if not isinstance(record, dict):
            raise ValueError('indexed geography manifest must be an object')
        identity = record.pop('identity')
        if identity != _identity(record) or record['schema'] != 1 or record['status'] != 'ready':
            raise ValueError('incomplete or damaged indexed geography manifest')
        record['identity'] = identity
        return record
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid indexed geography manifest: {path}') from exc


def collection_manifest(directory):
    """Check collection publication and child identities before any preparation."""
    directory = Path(directory)
    path = directory / COLLECTION
    if not path.exists() and not path.is_symlink():
        return None
    record = _ready_record(path)
    try:
        for name, identity in record['datasets'].items():
            if Path(name).name != name or indexed_manifest(directory / name)['identity'] != identity:
                raise ValueError('indexed geography collection dataset identity differs')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid indexed geography collection inventory') from exc
    return record


def indexed_manifest(directory):
    """Reject incomplete recognized collections; never fall back to packed tiles."""
    directory = Path(directory)
    path = directory / MANIFEST
    if not path.exists() and not path.is_symlink():
        return None
    try:
        record = _ready_record(path)
        if _sha256(directory / 'index') != record['index_sha256']:
            raise ValueError('indexed geography metadata changed')
        state = (directory / 'data.nc').stat()
        if (state.st_size, state.st_mtime_ns) != (record['payload']['size'], record['payload']['mtime_ns']):
            raise ValueError('immutable indexed geography payload changed')
        return record
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid indexed geography collection: {directory}') from exc


def _cache(variable, size=CACHE_BYTES):
    variable.set_auto_maskandscale(False)
    variable.set_var_chunk_cache(size=size, nelems=1009, preemption=.5)
    return variable


def _bounds(index, x, y, row, height, side, radius):
    rows = torch.arange(row, row + height, dtype=torch.float64)
    columns = torch.arange(index.tile_x, dtype=torch.float64)
    lat, lon = pixel_coordinates(index, x, y, rows, columns)
    points = block_points(pixel_cartesian(lat, lon, radius).reshape(height, index.tile_x, 3), side)
    bounds = enclosing_blocks(points)
    if not torch.all(torch.linalg.vector_norm(points - bounds[:, None, :3], dim=-1) <= bounds[:, 3, None]):
        raise ValueError('source enclosure does not contain its native samples')
    return bounds.reshape((height + side - 1) // side, (index.tile_x + side - 1) // side, 4)


def _integer_summary(raw, side):
    """Exclude padding while preserving native integer conversion before summation."""
    value = raw.to(torch.float32).to(torch.int64)
    height, width = value.shape[-2:]
    nr, nc = (height + side - 1) // side, (width + side - 1) // side
    padded = value.new_zeros((*value.shape[:-2], nr * side, nc * side))
    padded[..., :height, :width] = value
    sums = padded.reshape(*value.shape[:-2], nr, side, nc, side).sum((-3, -1))
    rows = (height - torch.arange(nr) * side).clamp(0, side)
    cols = (width - torch.arange(nc) * side).clamp(0, side)
    return sums, rows[:, None] * cols[None, :]


def _verify_raw(tiles, index, payload):
    """Compare every bordered value with a separate streamed packed-file decoder."""
    import numpy as np
    nx = index.tile_x + 2 * index.tile_bdr
    ny = index.tile_y + 2 * index.tile_bdr
    with Dataset(payload) as ds:
        variable = _cache(ds.variables['raw'])
        for number, (_, _, path) in enumerate(tiles):
            with Path(path).open('rb') as source:
                for layer in range(index.layers):
                    for row in range(0, ny, 128):
                        height = min(128, ny - row)
                        packed = source.read(height * nx * index.wordsize)
                        if index.wordsize == 3:
                            octets = np.frombuffer(packed, dtype='u1').reshape(height, nx, 3).astype(np.int64)
                            weights = np.array([65536, 256, 1] if index.endian == 'big' else [1, 256, 65536])
                            expected = (octets * weights).sum(-1)
                            if index.signed:
                                expected = np.where(expected >= 8388608, expected - 16777216, expected)
                        else:
                            dtype = ('>' if index.endian == 'big' else '<') + ('i' if index.signed else 'u') + str(index.wordsize)
                            expected = np.frombuffer(packed, dtype=dtype).reshape(height, nx).astype(np.int64)
                        actual = variable[number, layer, row:row + height].astype(np.int64)
                        if not np.array_equal(actual, expected):
                            raise ValueError(f'lossless geography verification failed: {path}, layer {layer}, row {row}')
                if source.read(1):
                    raise ValueError(f'source tile grew during geography verification: {path}')


def convert_terrain(geography, output, *, tiles=None, radius=6371229., chunk_side=128, compress=False):
    """Build terrain through the common local converter."""
    return convert_dataset(geography, output, summary='terrain', tiles=tiles,
                           radius=radius, chunk_side=chunk_side, compress=compress)


def convert_dataset(geography, output, *, summary=None, tiles=None, radius=6371229., chunk_side=128, compress=False,
                    _fingerprints=None):
    """Build a lossless local dataset. Optional tiles support bounded real-slice checks.

    An interrupted output remains recognizable and rejected. Existing outputs are
    never overwritten. Full-collection orchestration supplies the complete inventory.
    """
    from mpas_init._geotile_reader import GeotileReader
    from mpas_init.prepare import _complete_tiles, _check_integer_accumulator

    geography, output = Path(geography), Path(output)
    index_bytes = (geography / 'index').read_bytes()
    index = GeotileIndex.read(geography / 'index')
    if summary not in (None, 'terrain', 'category', 'climatology'):
        raise ValueError('unknown geography summary kind')
    if summary == 'terrain' and (index.kind != 'continuous' or index.layers != 1):
        raise ValueError('terrain summaries require one continuous layer')
    if summary == 'category' and (index.kind != 'categorical' or index.layers != 1 or index.scale_factor != 1.):
        raise ValueError('category summaries require one unscaled categorical layer')
    if summary == 'climatology' and (index.kind != 'continuous' or index.missing_value is None
                                   or not index.missing_value.is_integer()):
        raise ValueError('climatology summaries require continuous layers and an integer missing value')
    if chunk_side not in (128, 256):
        raise ValueError('indexed geography chunk side must be 128 or 256')
    tiles = _complete_tiles(geography, index, 'terrain') if tiles is None else list(tiles)
    if not tiles:
        raise ValueError('indexed geography requires source tiles')
    # A global reduction cannot exceed int64 after the native float32 conversion.
    _check_integer_accumulator(index, 1, category=summary == 'category')
    output.mkdir(parents=True, exist_ok=False)
    (output / MANIFEST).write_text(json.dumps(dict(schema=1, status='building')) + '\n')
    (output / 'index').write_bytes(index_bytes)
    ny, nx = index.tile_y + 2 * index.tile_bdr, index.tile_x + 2 * index.tile_bdr
    sources = []
    with Dataset(output / 'data.nc', 'w', format='NETCDF4') as ds:
        ds.createDimension('tile', len(tiles))
        ds.createDimension('layer', index.layers)
        if summary == 'category':
            ds.createDimension('category', index.category_max - index.category_min + 1)
        ds.createDimension('raw_y', ny)
        ds.createDimension('raw_x', nx)
        ds.createDimension('four', 4)
        dtype = ('i' if index.signed else 'u') + str(4 if index.wordsize == 3 else index.wordsize)
        raw = _cache(ds.createVariable('raw', dtype, ('tile', 'layer', 'raw_y', 'raw_x'),
            chunksizes=(1, index.layers, min(ny, chunk_side), min(nx, chunk_side)),
            zlib=compress, complevel=1, shuffle=compress, fletcher32=True, fill_value=False))
        levels = {}
        for side in SIDES if summary is not None else ():
            dims = (f'y{side}', f'x{side}')
            ds.createDimension(dims[0], (index.tile_y + side - 1) // side)
            ds.createDimension(dims[1], (index.tile_x + side - 1) // side)
            fields = [('count', 'i8', (), ()), ('bounds', 'f8', (), ('four',))]
            if summary == 'terrain':
                fields += [('sum', 'i8', (), ())]
            elif summary == 'category':
                fields += [('hist', 'i8', ('category',), ())]
            else:
                fields += [('sum', 'i8', ('layer',), ()), ('valid', 'i8', ('layer',), ())]
            levels[side] = {name: _cache(ds.createVariable(name + str(side), dtype, ('tile', *leading, *dims, *tail),
                chunksizes=(1, *(len(ds.dimensions[d]) for d in (*leading, *dims, *tail))),
                zlib=compress, complevel=1, fletcher32=True, fill_value=False), 65536)
                for name, dtype, leading, tail in fields}
        for number, (x, y, path) in enumerate(tiles):
            path = Path(path)
            if path.name != index.filename(x, y):
                raise ValueError('source tile identity differs from its native grid')
            before = path.stat()
            previous = (_fingerprints or {}).get(str(path.absolute()))
            digest = (previous['sha256'] if previous is not None
                      and (previous['size'], previous['mtime_ns']) == (before.st_size, before.st_mtime_ns)
                      else _sha256(path))
            sources.append(dict(name=path.name, x=x, y=y, size=before.st_size,
                mtime_ns=before.st_mtime_ns, sha256=digest))
            with GeotileReader(path, index) as reader:
                for row in range(0, ny, chunk_side):
                    slab = reader.read(include_border=True, window=(row, 0, min(chunk_side, ny - row), nx))
                    raw[number, :, row:row + slab.raw.shape[1]] = slab.raw.numpy()
                for row in range(0, index.tile_y, SIDES[0]) if summary is not None else ():
                    height = min(SIDES[0], index.tile_y - row)
                    slab = reader.read(window=(row, 0, height, index.tile_x))
                    if summary == 'terrain' and not slab.valid.all():
                        raise ValueError(f'terrain contains missing pixels: {path}')
                    values = slab.raw.to(torch.float32).to(torch.int64)
                    for side, variables in levels.items():
                        sums, counts = _integer_summary(values, side)
                        selection = (number, slice(row // side, row // side + counts.shape[0]))
                        variables['count'][selection] = counts.numpy()
                        variables['bounds'][selection] = _bounds(index, x, y, row, height, side, torch.tensor(radius, dtype=torch.float64)).numpy()
                        if summary == 'terrain':
                            variables['sum'][selection] = sums[0].numpy()
                        elif summary == 'category':
                            pixels = torch.where(slab.valid[0], slab.raw[0], index.category_min - 1).to(torch.float32).to(torch.int64)
                            nc = counts.shape[1]
                            owners = (torch.arange(height)[:, None] // side * nc
                                      + torch.arange(index.tile_x)[None, :] // side).flatten()
                            hist = category_counts(pixels.flatten(), owners, counts.numel(), index.category_min, index.category_max)
                            variables['hist'][number, :, selection[1], :] = hist.reshape(-1, *counts.shape).numpy()
                        else:
                            valid = values != int(index.missing_value)
                            sums, _ = _integer_summary(torch.where(valid, values, 0), side)
                            valid_count, _ = _integer_summary(valid.to(torch.int64), side)
                            variables['sum'][number, :, selection[1], :] = sums.numpy()
                            variables['valid'][number, :, selection[1], :] = valid_count.numpy()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('source tile changed during geography conversion')
    payload = output / 'data.nc'
    _verify_raw(tiles, index, payload)
    for (_, _, path), source in zip(tiles, sources):
        state = Path(path).stat()
        if (state.st_size, state.st_mtime_ns) != (source['size'], source['mtime_ns']):
            raise ValueError('source tile changed during geography verification')
    if (geography / 'index').read_bytes() != index_bytes:
        raise ValueError('source geography metadata changed during conversion')
    state = payload.stat()
    record = dict(schema=1, status='ready', recipe=RECIPE, torch=torch.__version__,
        radius=radius, sides=list(SIDES) if summary is not None else [], summary=summary,
        descriptor=geometry_descriptor(index, 1, SIDES[-1]),
        index_sha256=_sha256(output / 'index'), sources=sources,
        payload=dict(size=state.st_size, mtime_ns=state.st_mtime_ns, sha256=_sha256(payload)))
    record['identity'] = _identity(record)
    staged = output / '.geography.json'
    staged.write_text(json.dumps(record, sort_keys=True) + '\n')
    staged.replace(output / MANIFEST)
    return record


class IndexedDataset:
    """One validated container and bounded variable caches for open virtual tiles."""

    def __init__(self, directory, index):
        self.directory, self.index = Path(directory), index
        self.record = indexed_manifest(directory)
        if self.record is None:
            raise ValueError('indexed tile has no collection manifest')
        self.numbers = {item['name']: number for number, item in enumerate(self.record['sources'])}
        if GeotileIndex.read(Path(directory) / 'index') != index:
            raise ValueError('indexed source descriptor does not match')
        self.dataset = Dataset(Path(directory) / 'data.nc')
        try:
            for name, variable in self.dataset.variables.items():
                _cache(variable, CACHE_BYTES if name == 'raw' else 65536)
            expected = (len(self.numbers), index.layers, index.tile_y + 2 * index.tile_bdr, index.tile_x + 2 * index.tile_bdr)
            if self.dataset.variables['raw'].shape != expected:
                raise ValueError('indexed raw array differs from native tile/layer dimensions')
        except BaseException:
            self.dataset.close()
            raise

    def close(self):
        self.dataset.close()


class IndexedTile:
    """Bounded NetCDF window I/O behind the existing GeotileReader interface."""

    def __init__(self, directory, name, index, *, source=None):
        self._owns_source = source is None
        self.source = IndexedDataset(directory, index) if source is None else source
        if self.source.directory != Path(directory) or self.source.index != index or name not in self.source.numbers:
            if self._owns_source:
                self.source.close()
            raise ValueError('indexed tile or source descriptor does not match')
        self.record, self.dataset = self.source.record, self.source.dataset
        self.raw, self.number = self.dataset.variables['raw'], self.source.numbers[name]
        self.raw_values_requested = self.raw_read_calls = self.summary_bytes_requested = 0

    def values(self, selection):
        import numpy as np
        layers, rows, columns = selection
        if isinstance(rows, np.ndarray):
            row0, row1 = int(rows.min()), int(rows.max()) + 1
            col0, col1 = int(columns.min()), int(columns.max()) + 1
            start, stop, _ = layers.indices(self.raw.shape[1])
            if (row1 - row0) * (col1 - col0) * (stop - start) <= MAX_PAIRED_SAMPLES:
                # NetCDF orthogonal fancy indexing creates many tiny HDF5 reads.
                # Coalesce local requests into one bounded rectangle, then pair.
                slab = self.raw[self.number, layers, row0:row1, col0:col1]
                self.raw_values_requested += slab.size
                self.raw_read_calls += 1
                data = slab[:, rows - row0, columns - col0]
            else:
                # Coalesce sparse points into bounded contiguous stripes. NetCDF
                # fancy columns otherwise issue a separate HDF5 read per run.
                rr, cc = rows.reshape(-1), columns.reshape(-1)
                data = np.empty((stop - start, rr.size), dtype=self.raw.dtype)
                width = min(self.raw.shape[-1], max(1, MAX_PAIRED_SAMPLES // (stop - start)))
                height = max(1, min(self.raw.chunking()[2], MAX_PAIRED_SAMPLES // (width * (stop - start))))
                column_bands = (self.raw.shape[-1] + width - 1) // width
                bands = rr // height * column_bands + cc // width
                for band in np.unique(bands):
                    positions = np.flatnonzero(bands == band)
                    r, c = rr[positions], cc[positions]
                    r0, r1, c0, c1 = int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1
                    slab = self.raw[self.number, layers, r0:r1, c0:c1]
                    self.raw_values_requested += slab.size
                    self.raw_read_calls += 1
                    data[:, positions] = slab[:, r - r0, c - c0]
                data = data.reshape((stop - start, *rows.shape))
        else:
            data = self.raw[self.number, layers, rows, columns]
            self.raw_values_requested += data.size
            self.raw_read_calls += 1
        return data.astype(np.int64)

    def level(self, side):
        return tuple(self.array(name, side) for name in ('sum', 'count', 'bounds'))

    def array(self, name, side):
        values = self.dataset.variables[name + str(side)][self.number].copy()
        self.summary_bytes_requested += values.nbytes
        return torch.from_numpy(values)

    def close(self):
        if self._owns_source:
            self.source.close()


def terrain_tile_reductions(reader, mapping, x, y, *, window=None, supersample=1):
    """Traverse certified summaries; use exact native pixel ownership at ambiguous leaves."""
    store, index = reader._indexed, reader.index
    row, column, height, width = window or (0, 0, index.tile_y, index.tile_x)
    if (min(row, column) < 0 or min(height, width, supersample) < 1
            or row + height > index.tile_y or column + width > index.tile_x):
        raise ValueError('terrain reduction window lies outside its source tile')
    from mpas_init.prepare import _check_integer_accumulator
    _check_integer_accumulator(index, supersample)
    total = torch.zeros(len(mapping.centers), dtype=torch.int64)
    count = torch.zeros_like(total)
    if store is None or supersample != 1 or store.record['summary'] != 'terrain':
        for row0 in range(row, row + height, 128):
            h = min(128, row + height - row0)
            slab = reader.read(window=(row0, column, h, width))
            if not slab.valid.all():
                raise ValueError(f'terrain contains missing pixels: {reader.path}')
            owner = mapping.owners(index, start_x=x, start_y=y, row0=row0, col0=column,
                height=h, width=width, supersample=supersample)
            pixels = slab.raw[0].to(torch.float32).to(torch.int64)
            pixels = pixels.repeat_interleave(supersample, 0).repeat_interleave(supersample, 1).flatten()
            a, b = terrain_accumulation(pixels, owner, len(total))
            total += a
            count += b
        return total, count
    return summary_reductions([reader], mapping, x, y, window=window)[0]


def _certified_nodes(reader, mapping, x, y, window, *, landmask=None):
    """Yield whole certified nodes or exact raw sample owners for every source kind."""
    store, index = reader._indexed, reader.index
    row, column, height, width = window or (0, 0, index.tile_y, index.tile_x)
    record = store.record
    if (record['recipe'] != RECIPE or record['torch'] != torch.__version__
            or record['radius'] != float(mapping.radius)
            or tuple(record['descriptor']) != geometry_descriptor(index, 1, SIDES[-1])):
        raise ValueError('indexed source enclosure recipe or geometry differs')
    mask_identity = None if landmask is None else hashlib.sha256(landmask.contiguous().numpy().tobytes()).hexdigest()
    active = None
    for side in SIDES:
        bounds = store.array('bounds', side)
        shape = bounds.shape[:2]
        r0, c0 = torch.arange(shape[0])[:, None] * side, torch.arange(shape[1])[None, :] * side
        if active is None:
            active = (r0 < row + height) & (r0 + side > row) & (c0 < column + width) & (c0 + side > column)
        else:
            active = active.repeat_interleave(2, 0).repeat_interleave(2, 1)[:shape[0], :shape[1]].contiguous()
            active &= (r0 < row + height) & (r0 + side > row) & (c0 < column + width) & (c0 + side > column)
        selected = torch.nonzero(active.flatten()).flatten()
        if len(selected) == 0:
            break
        descriptor = (*geometry_descriptor(index, 1, side), 'source-hierarchy-v2', record['recipe'], record['torch'], mask_identity)
        key = (descriptor, x, y, row, column, height, width)
        offsets, candidates = mapping.candidate_plan(key, len(selected), lambda: bounds.reshape(-1, 4)[selected])
        single = offsets[1:] - offsets[:-1] == 1
        excluded = torch.zeros_like(single)
        if landmask is not None:
            # Every native owner is in its conservative candidate set. A set
            # containing only ocean cells contributes no land climatology.
            prefix = torch.cat((offsets.new_zeros(1), (landmask[candidates] == 1).to(torch.int64).cumsum(0)))
            excluded = prefix[offsets[1:]] == prefix[offsets[:-1]]
        inside = ((r0 >= row) & ((r0 + side).clamp_max(index.tile_y) <= row + height)
                  & (c0 >= column) & ((c0 + side).clamp_max(index.tile_x) <= column + width))
        single = single & inside.flatten()[selected] & ~excluded
        accepted = selected[single]
        owners = candidates[offsets[:-1][single]]
        if len(accepted):
            yield ('summary', side, accepted, owners)
        mapping.summary_accepted += len(accepted)
        mapping.summary_excluded += int(excluded.sum())
        mapping.summary_refined += int((~(single | excluded)).sum())
        active.view(-1)[selected[single | excluded]] = False
        if side != SIDES[-1]:
            continue
        remaining = torch.nonzero(~(single | excluded)).flatten()
        for first in range(0, len(remaining), 64):
            batch = remaining[first:first + 64]
            nodes = selected[batch]
            row0, col0 = nodes // shape[1] * side, nodes % shape[1] * side
            rr = row0[:, None, None] + torch.arange(side)[None, :, None]
            cc = col0[:, None, None] + torch.arange(side)[None, None, :]
            valid = ((rr < index.tile_y) & (cc < index.tile_x)
                     & (rr >= row) & (rr < row + height) & (cc >= column) & (cc < column + width))
            rr, cc = rr.clamp_max(index.tile_y - 1), cc.clamp_max(index.tile_x - 1)
            lat, lon = pixel_coordinates(index, x, y, rr, cc)
            points = pixel_cartesian_pairs(lat, lon, mapping.radius).reshape(len(batch), -1, 3)
            lengths = offsets[batch + 1] - offsets[batch]
            positions = torch.cat([torch.arange(offsets[k], offsets[k + 1]) for k in batch])
            local_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
            owner = block_owners(points, mapping.centers, local_offsets, candidates[positions])
            rr, cc = torch.broadcast_tensors(rr, cc)
            yield ('pixels', rr[valid], cc[valid], owner[valid.reshape_as(owner)])
            mapping.raw_samples += int(valid.sum())


def summary_reductions(readers, mapping, x, y, *, landmask=None, fields=None, window=None):
    """Apply exact field summaries and the existing pixel kernels to one traversal."""
    kind = readers[0]._indexed.record['summary']
    if kind not in ('terrain', 'category', 'climatology') or any(r._indexed.record['summary'] != kind for r in readers):
        raise ValueError('incompatible indexed summary kinds')
    if kind == 'climatology' and (landmask is None or fields is None or len(fields) != len(readers)):
        raise ValueError('climatology summaries require field names and a landmask')
    cells = len(mapping.centers)
    totals = [torch.zeros((r.index.category_max - r.index.category_min + 1, cells) if kind == 'category'
                         else (r.index.layers, cells) if kind == 'climatology' else (cells,), dtype=torch.int64)
              for r in readers]
    counts = [torch.zeros(cells, dtype=torch.int64) for r in readers]
    for node in _certified_nodes(readers[0], mapping, x, y, window,
                                 landmask=landmask if kind == 'climatology' else None):
        if node[0] == 'summary':
            _, side, selected, owner = node
            for slot, reader in enumerate(readers):
                store = reader._indexed
                if kind == 'terrain':
                    totals[slot].scatter_add_(0, owner, store.array('sum', side).flatten()[selected])
                    counts[slot].scatter_add_(0, owner, store.array('count', side).flatten()[selected])
                elif kind == 'category':
                    hist = store.array('hist', side).flatten(1)[:, selected]
                    totals[slot].scatter_add_(1, owner[None].expand_as(hist), hist)
                else:
                    sums = store.array('sum', side).flatten(1)[:, selected]
                    valid = store.array('valid', side).flatten(1)[:, selected]
                    count = store.array('count', side).flatten()[selected]
                    fill = None if fields[slot] == 'snoalb' else 8 if fields[slot] == 'albedo12m' else 0
                    if fill is None:
                        count = valid[0]
                    else:
                        sums = sums + fill * (count[None] - valid)
                    retained = landmask[owner] == 1
                    totals[slot].scatter_add_(1, owner[None].expand_as(sums), torch.where(retained[None], sums, 0))
                    counts[slot].scatter_add_(0, owner, torch.where(retained, count, 0))
        else:
            _, rows, columns, owner = node
            if kind == 'climatology':
                retained = landmask[owner] == 1
                if not retained.any():
                    continue
                rows, columns, owner = rows[retained], columns[retained], owner[retained]
            for slot, reader in enumerate(readers):
                index = reader.index
                tile = reader.points(rows, columns)
                pixels = tile.raw.to(torch.float32).to(torch.int64)
                if kind == 'terrain':
                    a, b = terrain_accumulation(pixels[0], owner, cells)
                elif kind == 'category':
                    pixels = torch.where(tile.valid[0], tile.raw[0], index.category_min - 1).to(torch.float32).to(torch.int64)
                    a = category_counts(pixels, owner, cells, index.category_min, index.category_max)
                    b = counts[slot].new_zeros(cells)
                else:
                    fill = None if fields[slot] == 'snoalb' else 8 if fields[slot] == 'albedo12m' else 0
                    a, b = land_accumulation(pixels, owner, landmask, missing=int(index.missing_value), fill=fill)
                totals[slot] += a
                counts[slot] += b
    return list(zip(totals, counts))
