"""Compact NetCDF storage for fixed block candidates, outside tensor numerics."""

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from netCDF4 import Dataset
import numpy as np
import torch


def _digest(identity, arrays):
    digest = hashlib.sha256(identity.encode())
    for name, values in arrays.items():
        digest.update(name.encode())
        digest.update(values.tobytes(order='C'))
    return digest.hexdigest()


def plan_path(directory, identity):
    return Path(directory) / ('geometry-' + hashlib.sha256(identity.encode()).hexdigest() + '.nc')


def read_plans(path, identity, cells, max_bytes):
    """Reject identity, layout and integrity failures before returning candidates."""
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError('geometry cache must be a regular file')
        with Dataset(path) as ds:
            ds.set_auto_maskandscale(False)
            if (ds.identity != identity or set(ds.dimensions) != {'plans', 'coordinates', 'plan_edges', 'blocks', 'block_edges', 'entries'}
                    or set(ds.variables) != {'windows', 'plan_offsets', 'block_offsets', 'candidates'}):
                raise ValueError('geometry cache identity or inventory mismatch')
            n, b, k = (len(ds.dimensions[name]) for name in ('plans', 'blocks', 'entries'))
            if (min(n, b, k) < 1 or len(ds.dimensions['coordinates']) != 6
                    or len(ds.dimensions['plan_edges']) != n + 1 or len(ds.dimensions['block_edges']) != b + 1
                    or (6 * n + n + 1 + b + 1 + k) * 8 > max_bytes):
                raise ValueError('geometry cache dimensions exceed bounds')
            layout = {'windows': ('i8', ('plans', 'coordinates')), 'plan_offsets': ('i8', ('plan_edges',)),
                      'block_offsets': ('i8', ('block_edges',)), 'candidates': ('u4', ('entries',))}
            arrays = {}
            for name, (dtype, dimensions) in layout.items():
                if ds[name].dimensions != dimensions or ds[name].dtype != np.dtype(dtype):
                    raise ValueError('geometry cache variable layout mismatch')
                arrays[name] = np.asarray(ds[name][:], dtype='<' + dtype)
            if ds.payload_sha256 != _digest(identity, arrays):
                raise ValueError('geometry cache payload checksum mismatch')
        windows, plans, offsets, ids = arrays.values()
        if (plans[0] != 0 or plans[-1] != b or np.any(np.diff(plans) <= 0)
                or offsets[0] != 0 or offsets[-1] != k or np.any(np.diff(offsets) <= 0)
                or np.any(ids >= cells) or np.any(windows[:, :2] < 1)
                or np.any(windows[:, 2:4] < 0) or np.any(windows[:, 4:] < 1)):
            raise ValueError('geometry cache offsets, windows or candidate IDs are invalid')
        result = {}
        for window, begin, end in zip(windows, plans[:-1], plans[1:]):
            key = tuple(int(v) for v in window)
            if key in result:
                raise ValueError('geometry cache contains duplicate windows')
            first, last = offsets[begin], offsets[end]
            result[key] = (torch.from_numpy((offsets[begin:end + 1] - first).copy()),
                           torch.from_numpy(ids[first:last].astype(np.int64)))
        return result
    except (OSError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError(f'invalid geometry cache: {path}') from exc


def write_plans(path, identity, plans, cells, max_bytes):
    """Verify a privately staged replacement before atomic publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        previous = read_plans(path, identity, cells, max_bytes)
        previous.update(plans)
        plans = previous
    windows, plan_offsets, offsets, ids = [], [0], [0], []
    for key, (block_offsets, candidates) in sorted(plans.items()):
        windows.append(key)
        plan_offsets.append(plan_offsets[-1] + len(block_offsets) - 1)
        offsets.extend((block_offsets[1:] + offsets[-1]).tolist())
        ids.append(candidates.numpy().astype('<u4'))
    arrays = dict(windows=np.asarray(windows, dtype='<i8'), plan_offsets=np.asarray(plan_offsets, dtype='<i8'),
                  block_offsets=np.asarray(offsets, dtype='<i8'), candidates=np.concatenate(ids))
    memory = sum(v.size * 8 for v in arrays.values())
    if memory > max_bytes:
        raise ValueError('geometry cache exceeds its declared memory budget')
    with TemporaryDirectory(prefix='.geometry-', dir=path.parent) as temporary:
        staged = Path(temporary) / 'plan.nc'
        with Dataset(staged, 'w') as ds:
            for name, size in dict(plans=len(windows), coordinates=6, plan_edges=len(plan_offsets),
                                   blocks=len(offsets) - 1, block_edges=len(offsets), entries=len(arrays['candidates'])).items():
                ds.createDimension(name, size)
            for name, dimensions in dict(windows=('plans', 'coordinates'), plan_offsets=('plan_edges',),
                                         block_offsets=('block_edges',), candidates=('entries',)).items():
                ds.createVariable(name, arrays[name].dtype, dimensions)[:] = arrays[name]
            ds.identity, ds.payload_sha256 = identity, _digest(identity, arrays)
        read_plans(staged, identity, cells, max_bytes)
        os.replace(staged, path)
