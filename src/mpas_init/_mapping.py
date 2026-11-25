"""Offline ownership planning shared by raw-geography preparation stages."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess

import torch

from mpas_init._atlas import AtlasCandidates
from mpas_init._geometry_cache import plan_path, read_plans, write_plans
from mpas_init._ownership import nearest_block_owners
from mpas_init.static import pixel_cartesian, pixel_coordinates


def block_points(grid, side):
    """Pack equal-sized blocks, repeating edge points only in incomplete blocks."""
    height, width = grid.shape[:2]
    nr, nc = (height + side - 1) // side, (width + side - 1) // side
    rows = torch.arange(nr * side, device=grid.device).clamp_max(height - 1)
    columns = torch.arange(nc * side, device=grid.device).clamp_max(width - 1)
    return grid[rows[:, None], columns[None, :]].reshape(nr, side, nc, side, 3).permute(0, 2, 1, 3, 4).reshape(-1, side * side, 3)


def enclosing_blocks(points):
    center = .5 * (points.amin(1) + points.amax(1))
    radius = torch.linalg.vector_norm(points - center[:, None], dim=-1).amax(1)
    return torch.cat((center, radius[:, None]), 1)


def geometry_descriptor(index, supersample, side):
    """Use actual formulas, not rounded resolution labels or dataset names."""
    return (index.dx, index.dy, index.known_lat, index.known_lon,
            index.tile_x, index.tile_y, supersample, side)


def block_owners(points, centers, offsets, candidates):
    """Batch independent equal-candidate blocks within a 32 MiB distance workspace."""
    lengths = offsets[1:] - offsets[:-1]
    owners = torch.empty(points.shape[:2], dtype=torch.int64, device=points.device)
    for length in torch.unique(lengths).tolist():
        blocks = torch.nonzero(lengths == length).flatten()
        if length == 1:
            owners[blocks] = candidates[offsets[blocks], None]
            continue
        pixels = max(1, min(points.shape[1], 1048576 // length))
        batch = max(1, 1048576 // (pixels * length))
        for start in range(0, len(blocks), batch):
            selected = blocks[start:start + batch]
            positions = offsets[selected, None] + torch.arange(length)
            ids = candidates[positions]
            for first in range(0, points.shape[1], pixels):
                owners[selected, first:first + pixels] = nearest_block_owners(points[selected, first:first + pixels], centers[ids], ids)
    return owners


class PreparationMapping:
    """One Atlas tree and bounded candidate reuse for a complete preparation call."""

    def __init__(self, centers, radius, *, block_side=32, executable=None, cache_dir=None, max_bytes=256 * 1024**2):
        if centers.requires_grad or radius.requires_grad:
            raise ValueError('static ownership requires fixed geometry')
        executable = executable or os.environ.get('MPAS_INIT_ATLAS_EXECUTABLE') or shutil.which('mpas-atlas-candidates')
        if not executable or not Path(executable).is_file():
            raise ValueError('new-mesh preparation requires mpas-atlas-candidates on PATH or MPAS_INIT_ATLAS_EXECUTABLE')
        if (centers.dtype != torch.float64 or centers.device.type != 'cpu' or not torch.isfinite(centers).all()
                or radius.dtype != torch.float64 or radius.device.type != 'cpu' or radius.shape != () or not torch.isfinite(radius)):
            raise ValueError('preparation mapping requires finite fixed CPU float64 geometry')
        version = subprocess.run([str(executable), '--version'], check=True, capture_output=True, timeout=60).stdout
        if not version.startswith(b'mpas-atlas-candidates protocol=1 ') or len(version) > 1024:
            raise ValueError('unknown Atlas candidate compiler version')
        self._identity = dict(schema=1, distance='(dx*dx+dy*dy)+dz*dz;lexicographic-xyz-stable-native-ties',
                              mesh=hashlib.sha256(centers.contiguous().numpy().astype('<f8').tobytes()).hexdigest(),
                              radius=radius.item(), compiler=version.decode().strip(),
                              compiler_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest())
        self.centers, self.radius = centers, radius
        self.side, self.max_bytes = block_side, max_bytes
        self.cache_dir, self.executable = cache_dir, executable
        self._tree = None
        self._plans, self._bytes = OrderedDict(), 0
        self._loaded_grids = set()
        self.compiled_blocks = self.reused_blocks = self.loaded_blocks = 0
        self.summary_accepted = self.summary_refined = self.summary_excluded = self.raw_samples = 0

    def candidates(self, bounds):
        """Query precomputed source enclosures without expanding their sample coordinates."""
        if self._tree is None:
            self._tree = AtlasCandidates(self.executable, self.centers)
        offsets, candidates = self._tree.query(bounds)
        self.compiled_blocks += len(bounds)
        return offsets, candidates

    def _grid_identity(self, descriptor):
        return json.dumps({**self._identity, 'grid': descriptor}, sort_keys=True, separators=(',', ':'))

    def _retain(self, key, offsets, candidates):
        size = offsets.numel() * offsets.element_size() + candidates.numel() * candidates.element_size()
        if size <= self.max_bytes:
            while self._plans and self._bytes + size > self.max_bytes:
                old_offsets, old_ids = self._plans.popitem(last=False)[1]
                self._bytes -= (old_offsets.numel() * old_offsets.element_size() + old_ids.numel() * old_ids.element_size())
            self._plans[key] = (offsets, candidates)
            self._bytes += size

    def candidate_plan(self, key, blocks, bounds):
        """Share checked, bounded candidate reuse between raw and indexed geometry."""
        if self.cache_dir is not None and key[0] not in self._loaded_grids:
            identity = self._grid_identity(key[0])
            path = plan_path(self.cache_dir, identity)
            if path.exists() or path.is_symlink():
                for window, (offsets, candidates) in read_plans(path, identity, len(self.centers), self.max_bytes).items():
                    self._retain((key[0], *window), offsets, candidates)
                    self.loaded_blocks += len(offsets) - 1
            self._loaded_grids.add(key[0])
        if key in self._plans:
            offsets, candidates = self._plans.pop(key)
            self._plans[key] = (offsets, candidates)
            if len(offsets) != blocks + 1:
                raise ValueError('geometry cache block count differs from its window')
            self.reused_blocks += blocks
            return offsets, candidates
        offsets, candidates = self.candidates(bounds())
        self._retain(key, offsets, candidates)
        return offsets, candidates

    def owners(self, index, *, start_x, start_y, height, width, row0=0, col0=0, supersample=1):
        """Return only owners; keep decoded field values outside geometry planning."""
        if min(height, width, supersample) < 1 or min(row0, col0) < 0 or row0 + height > index.tile_y or col0 + width > index.tile_x:
            raise ValueError('ownership window lies outside the tile interior')
        rows = torch.arange(row0 * supersample, (row0 + height) * supersample, dtype=torch.float64, device=self.centers.device)
        columns = torch.arange(col0 * supersample, (col0 + width) * supersample, dtype=torch.float64, device=self.centers.device)
        lat, lon = pixel_coordinates(index, start_x, start_y, rows, columns, supersample)
        grid = pixel_cartesian(lat, lon, self.radius).reshape(len(rows), len(columns), 3)
        side = min(self.side, max(len(rows), len(columns)))
        points = block_points(grid, side)
        del grid
        key = (geometry_descriptor(index, supersample, side), start_x, start_y, row0, col0, height, width)
        offsets, candidates = self.candidate_plan(key, len(points), lambda: enclosing_blocks(points))
        owners = block_owners(points, self.centers, offsets, candidates)
        self.raw_samples += len(rows) * len(columns)
        nr, nc = (len(rows) + side - 1) // side, (len(columns) + side - 1) // side
        return owners.reshape(nr, nc, side, side).permute(0, 2, 1, 3).reshape(nr * side, nc * side)[:len(rows), :len(columns)].reshape(-1)

    def close(self):
        if self._tree is not None:
            self._tree.close()
        if self.cache_dir is not None:
            for descriptor in self._loaded_grids:
                plans = {key[1:]: value for key, value in self._plans.items() if key[0] == descriptor}
                if plans:
                    identity = self._grid_identity(descriptor)
                    write_plans(plan_path(self.cache_dir, identity), identity, plans, len(self.centers), self.max_bytes)

    def __enter__(self):
        return self

    def __exit__(self, error, value, traceback):
        if error is None:
            self.close()
        elif self._tree is not None:
            self._tree.close(abort=True)
