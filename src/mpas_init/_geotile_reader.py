"""Bounded, read-only geography decoding. This is outside tensor numerics."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from mpas_init.geotile import Geotile, GeotileIndex


class GeotileReader:
    """Keep packed bytes; expand only requested layers, windows or footprint pixels."""

    def __init__(self, path: str | Path, index: GeotileIndex, *, _indexed_source=None):
        self.path, self.index = Path(path), index
        self.nx, self.ny = index.tile_x + 2 * index.tile_bdr, index.tile_y + 2 * index.tile_bdr
        self._indexed = None
        from mpas_init._geography_index import MANIFEST, IndexedTile
        if (self.path.parent / MANIFEST).exists() or (self.path.parent / MANIFEST).is_symlink():
            self._indexed = IndexedTile(self.path.parent, self.path.name, index, source=_indexed_source)
            self._buffer = True
            return
        expected = self.nx * self.ny * index.layers * index.wordsize
        if self.path.stat().st_size != expected:
            raise ValueError(f'incomplete or oversized geotile: {self.path}; expected {expected} bytes')
        # mmap creation dominates the measured legacy-file workload on this filesystem.
        self._buffer = self.path.read_bytes()
        if len(self._buffer) != expected:
            raise ValueError(f'incomplete or oversized geotile: {self.path}; expected {expected} bytes')

    def read(self, *, window: tuple[int, int, int, int] | None = None,
             include_border: bool = False, layers: slice | None = None) -> Geotile:
        """Read (row, column, height, width), relative to the selected interior or bordered tile."""
        if self._buffer is None:
            raise ValueError('geotile reader is closed')
        index = self.index
        b = 0 if include_border else index.tile_bdr
        height, width = (self.ny, self.nx) if include_border else (index.tile_y, index.tile_x)
        if window is None:
            row, column = 0, 0
        else:
            if (len(window) != 4 or any(not isinstance(v, int) or isinstance(v, bool) for v in window)):
                raise ValueError('geotile window requires four integer coordinates')
            row, column, h, w = window
            if min(row, column) < 0 or min(h, w) < 1 or row + h > height or column + w > width:
                raise ValueError('geotile window lies outside the selected tile')
            height, width = h, w
        if layers is None:
            layers = slice(0, index.layers)
        if (not isinstance(layers, slice) or layers.step not in (None, 1)
                or any(v is not None and (not isinstance(v, int) or isinstance(v, bool))
                       for v in (layers.start, layers.stop))):
            raise ValueError('geotile layers require a contiguous integer slice')
        start, stop = 0 if layers.start is None else layers.start, index.layers if layers.stop is None else layers.stop
        if not 0 <= start < stop <= index.layers:
            raise ValueError('geotile layer slice lies outside the tile')
        selection = (slice(start, stop), slice(row + b, row + b + height), slice(column + b, column + b + width))
        return self._decode(selection)

    def points(self, rows: torch.Tensor, columns: torch.Tensor) -> Geotile:
        """Decode paired interior indices without expanding an entire tile."""
        if self._buffer is None:
            raise ValueError('geotile reader is closed')
        if (rows.shape != columns.shape or rows.numel() < 1
                or any(v.dtype != torch.int64 or v.device.type != 'cpu' for v in (rows, columns))
                or torch.any((rows < 0) | (rows >= self.index.tile_y))
                or torch.any((columns < 0) | (columns >= self.index.tile_x))):
            raise ValueError('geotile points require matching valid CPU int64 interior indices')
        b = self.index.tile_bdr
        return self._decode((slice(None), rows.numpy() + b, columns.numpy() + b))

    def _decode(self, selection):
        index = self.index
        if self._indexed is not None:
            values = self._indexed.values(selection)
        elif index.wordsize == 3:
            packed = np.ndarray((index.layers, self.ny, self.nx, 3), dtype=np.uint8, buffer=self._buffer)
            octets = packed[selection].astype(np.int64)
            if index.endian == 'little':
                octets = octets[..., ::-1]
            values = (octets[..., 0] << 16) | (octets[..., 1] << 8) | octets[..., 2]
            if index.signed:
                values = np.where(values >= 1 << 23, values - (1 << 24), values)
        else:
            code = ('>' if index.endian == 'big' else '<') + ('i' if index.signed else 'u') + str(index.wordsize)
            packed = np.ndarray((index.layers, self.ny, self.nx), dtype=np.dtype(code), buffer=self._buffer)
            values = packed[selection].astype(np.int64)
        raw = torch.from_numpy(values)
        valid = torch.ones_like(raw, dtype=torch.bool)
        bits, missing = index.wordsize * 8, index.missing_value
        lower, upper = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if index.signed else (0, (1 << bits) - 1)
        if missing is not None and missing.is_integer() and lower <= missing <= upper:
            valid = raw != int(missing)
        return Geotile(raw, valid)

    def close(self):
        if self._indexed is not None:
            self._indexed.close()
            self._indexed = None
        self._buffer = None

    def __enter__(self):
        return self

    def __exit__(self, error, value, traceback):
        self.close()


class GeotileReaders:
    """Bound open tiles and share physical indexed containers until eviction."""

    def __init__(self, capacity):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError('geotile reader capacity must be a positive integer')
        self.capacity, self._readers = capacity, OrderedDict()
        self._indexed_sources = {}

    def _release_unused(self):
        active = {reader._indexed.source for reader in self._readers.values() if reader._indexed is not None}
        for directory, source in list(self._indexed_sources.items()):
            if source not in active:
                source.close()
                del self._indexed_sources[directory]

    def _reader(self, path, index):
        key = Path(path)
        reader = self._readers.pop(key, None)
        if reader is None:
            from mpas_init._geography_index import MANIFEST, IndexedDataset
            manifest = key.parent / MANIFEST
            source = self._indexed_sources.get(key.parent)
            if source is None and (manifest.exists() or manifest.is_symlink()):
                source = IndexedDataset(key.parent, index)
                self._indexed_sources[key.parent] = source
            try:
                reader = GeotileReader(key, index, _indexed_source=source)
            except BaseException:
                self._release_unused()
                raise
        elif reader.index != index:
            reader.close()
            raise ValueError('cached geography reader index changed')
        self._readers[key] = reader
        while len(self._readers) > self.capacity:
            self._readers.popitem(last=False)[1].close()
        self._release_unused()
        return reader

    def points(self, path, index, rows, columns):
        return self._reader(path, index).points(rows, columns)

    def read(self, path, index, *, window):
        return self._reader(path, index).read(window=window)

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._release_unused()

    def __enter__(self):
        return self

    def __exit__(self, error, value, traceback):
        self.close()
