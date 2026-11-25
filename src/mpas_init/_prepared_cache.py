"""Explicit immutable-input cache for complete prepared statics, outside numerics."""

import errno
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import pkgutil
import shutil
from tempfile import TemporaryDirectory

from netCDF4 import Dataset
import torch

from mpas_init.geotile import GeotileIndex
from mpas_init.mesh import (CLASSIC_FIELDS, CLIMATOLOGY_DIMENSIONS, MESH_FIELDS, SURFACE_NAMES,
                            UGWP_FIELDS, read_classic_gwd, read_native_noah_static, read_static, read_ugwp)
from mpas_init.mesh_assets import _sha256


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def recipe_identity():
    """Bind installed numerical sources, including constants and polar conventions."""
    import mpas_init
    names = ['mpas_init'] + ['mpas_init.' + item.name for item in pkgutil.iter_modules(mpas_init.__path__)]
    sources = {name: hashlib.sha256(find_spec(name).loader.get_source(name).encode()).hexdigest()
               for name in names}
    return _digest(dict(schema=1, sources=sources, torch=torch.__version__))


def input_paths(mesh, root, profile):
    from mpas_init.prepare import _complete_tiles
    from mpas_init._geography_index import COLLECTION, MANIFEST, collection_manifest, indexed_manifest
    names = ['topo_gmted2010_30s']
    if profile != 'dycore':
        names += ['modis_landuse_20class_30s', 'soiltype_top_30s', 'maxsnowalb_modis',
                  'greenfrac_fpar_modis', 'albedo_modis', 'lai_modis_30s', 'soiltemp_1deg', 'landuse_30s']
    if profile == 'ufs-ms':
        names += ['topo_ugwp_30s', 'topo_ugwp_2.5m']
    paths = [Path(mesh)]
    if collection_manifest(root) is not None:
        paths.append(Path(root) / COLLECTION)
    for name in names:
        directory = Path(root) / name
        index_path = directory / 'index'
        index = GeotileIndex.read(index_path)
        paths.append(index_path)
        tiles = _complete_tiles(directory, index, 'cached preparation')
        if indexed_manifest(directory) is None:
            paths.extend(path for _, _, path in tiles)
        else:
            paths.extend((directory / MANIFEST, directory / 'data.nc'))
    return paths


def _stat(path):
    status = path.stat()
    if not path.is_file():
        raise ValueError(f'cached preparation input is not a file: {path}')
    return dict(size=status.st_size, mtime_ns=status.st_mtime_ns)


def _fingerprint(path):
    before = _stat(path)
    digest = _sha256(path)
    if _stat(path) != before:
        raise ValueError(f'preparation input changed while hashing: {path}')
    return dict(path=str(path.absolute()), **before, sha256=digest)


def _unchanged(inputs):
    for item in inputs:
        path = Path(item['path'])
        try:
            unchanged = _stat(path) == {key: item[key] for key in ('size', 'mtime_ns')}
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError(f'cached preparation input changed or disappeared: {path}; use a new cache directory')


def _value(value):
    return value.tolist() if hasattr(value, 'tolist') else value


def _signature(path, profile):
    expected = set(MESH_FIELDS)
    if profile != 'dycore':
        expected |= set(SURFACE_NAMES.values()) | {'mminlu', 'isice_lu', 'iswater_lu', 'soiltemp'}
        expected |= set(CLIMATOLOGY_DIMENSIONS) | set(CLASSIC_FIELDS)
    if profile == 'ufs-ms':
        expected |= set(UGWP_FIELDS)
    with Dataset(path) as ds:
        if set(ds.variables) != expected:
            raise ValueError('prepared-static cache has an incomplete or unexpected variable inventory')
        signature = dict(dimensions={name: [len(dim), dim.isunlimited()] for name, dim in ds.dimensions.items()},
            variables={name: dict(dtype=variable.dtype.str, dimensions=list(variable.dimensions),
                attributes={key: _value(variable.getncattr(key)) for key in variable.ncattrs()})
                for name, variable in ds.variables.items()},
            attributes={key: _value(ds.getncattr(key)) for key in ds.ncattrs()})
    read_static(path)
    if profile != 'dycore':
        read_native_noah_static(path)
        read_classic_gwd(path)
    if profile == 'ufs-ms':
        read_ugwp(path)
        for key in ('mpas_init_ugwp_polar_extension', 'mpas_init_ugwp_polar_directional_convention'):
            if not signature['attributes'].get(key):
                raise ValueError('prepared-static cache lacks the approved UGWP polar convention')
    return signature


class PreparedStaticCache:
    """A hit trusts unchanged size/mtime only for the declared immutable input collection.

    Deliberate edits that preserve both values violate this contract. The complete
    cached output is hashed and validated on every hit, including its metadata.
    """

    def __init__(self, directory, mesh, root, parameters):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request = dict(schema=1, mesh=str(Path(mesh).absolute()), root=str(Path(root).absolute()),
                            parameters=parameters, recipe=recipe_identity())
        self.entry = self.directory / ('static-' + _digest(self.request))
        self.profile = parameters['profile']
        self.inputs = None
        self.geometry = None

    def load(self, output=None):
        if not self.entry.exists():
            return False
        try:
            receipt = json.loads((self.entry / 'manifest.json').read_text())
            if receipt['request'] != self.request or receipt['identity'] != _digest({
                    key: receipt[key] for key in ('request', 'inputs', 'geometry', 'payload')}):
                raise ValueError('prepared-static cache identity does not match')
            geometry = receipt['geometry']
            if (not isinstance(geometry, dict) or set(geometry) != {
                    'schema', 'distance', 'mesh', 'radius', 'compiler', 'compiler_sha256'}
                    or not geometry['compiler'].startswith('mpas-atlas-candidates protocol=1 ')
                    or any(len(geometry[key]) != 64 for key in ('mesh', 'compiler_sha256'))):
                raise ValueError('prepared-static cache geometry/compiler provenance is incomplete')
            _unchanged(receipt['inputs'])
            path = self.entry / 'static.nc'
            payload = receipt['payload']
            if path.stat().st_size != payload['size'] or _sha256(path) != payload['sha256']:
                raise ValueError('prepared-static cache payload checksum does not match')
            if _signature(path, self.profile) != payload['signature']:
                raise ValueError('prepared-static cache dimensions, types or metadata do not match')
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'invalid prepared-static cache entry: {self.entry}') from exc
        if output is not None:
            # Exclusive creation retains the public refusal to overwrite outputs.
            with Path(output).open('xb') as target, path.open('rb') as source:
                try:
                    shutil.copyfileobj(source, target, length=1024**2)
                except BaseException:
                    target.close()
                    Path(output).unlink()
                    raise
        return True

    def fingerprint(self, mesh, root):
        index_path = self.directory / ('inputs-' + _digest(str(Path(root).absolute())) + '.json')
        previous = {}
        if index_path.exists():
            try:
                record = json.loads(index_path.read_text())
                previous = record['inputs']
                if record['identity'] != _digest(previous):
                    raise ValueError('preparation input fingerprint checksum does not match')
            except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                raise ValueError('invalid preparation input fingerprint cache') from exc
        self.inputs = []
        for path in input_paths(mesh, root, self.profile):
            name = str(path.absolute())
            item = previous.get(name)
            if item is None or _stat(path) != {key: item[key] for key in ('size', 'mtime_ns')}:
                item = _fingerprint(path)
            self.inputs.append(item)
            previous[name] = item
        with TemporaryDirectory(prefix='.inputs-', dir=self.directory) as temporary:
            staged = Path(temporary) / 'inputs.json'
            staged.write_text(json.dumps(dict(inputs=previous, identity=_digest(previous)), sort_keys=True) + '\n')
            os.replace(staged, index_path)

    def publish(self, output):
        if self.inputs is None or self.geometry is None:
            raise ValueError('prepared-static cache publication requires input fingerprints and geometry identity')
        _unchanged(self.inputs)
        with TemporaryDirectory(prefix='.static-', dir=self.directory) as temporary:
            staging = Path(temporary)
            path = staging / 'static.nc'
            shutil.copyfile(output, path)
            payload = dict(size=path.stat().st_size, sha256=_sha256(path), signature=_signature(path, self.profile))
            receipt = dict(request=self.request, inputs=self.inputs, geometry=self.geometry, payload=payload)
            receipt['identity'] = _digest(receipt)
            (staging / 'manifest.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n')
            _unchanged(self.inputs)
            try:
                os.rename(staging, self.entry)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not self.load():
                    raise
