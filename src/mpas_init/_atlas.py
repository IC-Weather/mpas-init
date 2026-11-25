"""Offline binary transport for fixed Atlas geometry; never used in tensor numerics."""

import struct
import subprocess
import select
import time
import os

import numpy as np
import torch


def _geometry(values, width):
    if (values.ndim != 2 or values.shape[1] != width or values.dtype != torch.float64
            or values.device.type != 'cpu' or values.requires_grad or not torch.isfinite(values).all()):
        raise ValueError('candidate compiler requires finite, fixed CPU float64 geometry')
    return values.contiguous().numpy().astype('<f8', copy=False).tobytes()


def _decode(data, blocks, cells):
    if len(data) < 16:
        raise ValueError('truncated Atlas candidate response header')
    queries, entries = struct.unpack_from('<QQ', data)
    if queries != len(blocks) or not queries <= entries <= 16777216:
        raise ValueError('invalid Atlas candidate response counts')
    expected = 16 + 8 * (queries + 1) + 4 * entries
    if len(data) != expected:
        raise ValueError('truncated or trailing Atlas candidate response')
    offsets = np.frombuffer(data, '<u8', queries + 1, 16).astype(np.int64)
    ids = np.frombuffer(data, '<u4', entries, 16 + 8 * (queries + 1)).astype(np.int64)
    if offsets[0] != 0 or offsets[-1] != entries or np.any(np.diff(offsets) <= 0) or np.any(ids >= cells):
        raise ValueError('invalid Atlas candidate offsets or IDs')
    return torch.from_numpy(offsets), torch.from_numpy(ids)


class AtlasCandidates:
    """One bounded offline compiler process per fixed mesh, reused across query batches."""

    def __init__(self, executable, centers):
        encoded = _geometry(centers, 3)
        if not 0 < len(centers) < 2**31:
            raise ValueError('candidate compiler cell count exceeds protocol bounds')
        self.cells = len(centers)
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
        try:
            self._write(b'MPASAT01' + struct.pack('<Q', self.cells) + encoded)
            if self._read(8) != b'MPASOK01':
                raise ValueError('invalid Atlas candidate response header')
        except BaseException:
            self.close(abort=True)
            raise

    def _write(self, data):
        remaining = memoryview(data)
        deadline = time.monotonic() + 60
        while remaining:
            if not select.select([], [self.process.stdin], [], max(0., deadline - time.monotonic()))[1]:
                raise TimeoutError('Atlas candidate request exceeded 60 seconds')
            sent = os.write(self.process.stdin.fileno(), remaining[:4096])
            remaining = remaining[sent:]

    def _read(self, size):
        parts, remaining = [], size
        deadline = time.monotonic() + 60
        while remaining:
            if not select.select([self.process.stdout], [], [], max(0., deadline - time.monotonic()))[0]:
                raise TimeoutError('Atlas candidate response exceeded 60 seconds')
            part = os.read(self.process.stdout.fileno(), min(remaining, 1024 * 1024))
            if not part:
                raise RuntimeError('Atlas candidate compiler closed its response early')
            parts.append(part)
            remaining -= len(part)
        return b''.join(parts)

    def query(self, blocks):
        encoded = _geometry(blocks, 4)
        if not 0 < len(blocks) <= 65536 or torch.any(blocks[:, 3] < 0):
            raise ValueError('candidate compiler query geometry exceeds protocol bounds')
        self._write(struct.pack('<Q', len(blocks)) + encoded)
        header = self._read(16)
        queries, entries = struct.unpack('<QQ', header)
        if queries != len(blocks) or not queries <= entries <= 16777216:
            raise ValueError('invalid Atlas candidate response counts')
        return _decode(header + self._read(8 * (queries + 1) + 4 * entries), blocks, self.cells)

    def close(self, *, abort=False):
        if self.process.stdin.closed:
            return
        try:
            if abort:
                self.process.kill()
            else:
                self._write(struct.pack('<Q', 0))
            self.process.wait(timeout=10)
            if not abort and self.process.returncode:
                raise RuntimeError('Atlas candidate compiler failed: ' + self.process.stderr.read(4096).decode(errors='replace'))
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, error, value, traceback):
        self.close(abort=error is not None)


def compile_candidates(executable, centers, blocks):
    _geometry(blocks, 4)
    if not 0 < len(blocks) <= 65536 or torch.any(blocks[:, 3] < 0):
        raise ValueError('candidate compiler query geometry exceeds protocol bounds')
    with AtlasCandidates(executable, centers) as compiler:
        return compiler.query(blocks)
