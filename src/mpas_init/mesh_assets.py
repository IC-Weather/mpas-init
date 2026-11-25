"""Explicit acquisition and offline reuse of small NCAR global mesh archives."""

from contextlib import suppress
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from mpas_init.mesh import read_raw_mesh


# NCAR's published small quasi-uniform meshes; larger downloads remain explicit local imports.
STANDARD_MESHES = {"x1.2562": 2562, "x1.4002": 4002, "x1.10242": 10242}
_BASE_URL = "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes/"
_DOWNLOAD_LIMIT = 16 * 1024**2
_EXPANDED_LIMIT = 128 * 1024**2
_MESH_LIMIT = 32 * 1024**2


class _NCARRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if (target.scheme != "https" or target.netloc != "www2.mmm.ucar.edu"):
            raise ValueError("managed mesh redirects must retain the NCAR HTTPS origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download(url, output):
    with build_opener(_NCARRedirect()).open(url, timeout=30) as source, output.open("xb") as target:
        total = 0
        digest = hashlib.sha256()
        while block := source.read(1024**2):
            total += len(block)
            if total > _DOWNLOAD_LIMIT:
                raise ValueError("managed mesh archive exceeds the 16 MiB download limit")
            target.write(block)
            digest.update(block)
    return total, digest.hexdigest()


class _ExpandedArchive:
    def __init__(self, stream):
        self.stream, self.total = stream, 0

    def read(self, size):
        block = self.stream.read(min(size, _EXPANDED_LIMIT - self.total + 1))
        self.total += len(block)
        if self.total > _EXPANDED_LIMIT:
            raise ValueError("managed mesh archive exceeds the 128 MiB expanded limit")
        return block


def _unpack(archive, output):
    selected = None
    # Never extract archive paths, links, ownership or permissions into the cache.
    with gzip.open(archive, "rb") as compressed:
        stream = _ExpandedArchive(compressed)
        with tarfile.open(fileobj=stream, mode="r|") as bundle:
            for member in bundle:
                path = PurePosixPath(member.name)
                if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                        or not (member.isfile() or member.isdir()) or member.issparse()):
                    raise ValueError("unsafe managed mesh archive member")
                if member.size < 0 or member.size > _EXPANDED_LIMIT:
                    raise ValueError("invalid managed mesh archive member size")
                if member.isfile() and path.suffix == ".nc":
                    if selected is not None:
                        raise ValueError("managed mesh archive must contain exactly one NetCDF mesh")
                    if member.size > _MESH_LIMIT:
                        raise ValueError("managed mesh NetCDF exceeds the 32 MiB mesh limit")
                    with bundle.extractfile(member) as source, output.open("xb") as target:
                        while block := source.read(1024**2):
                            target.write(block)
                    selected = member.name
        # Read through the gzip trailer, so truncation and CRC errors cannot publish a cache entry.
        while stream.read(1024**2):
            pass
    if selected is None:
        raise ValueError("managed mesh archive must contain exactly one NetCDF mesh")
    return selected


def _validate_mesh(path, cells):
    mesh = read_raw_mesh(path)
    if mesh.sizes["nCells"] != cells or mesh.sphere_radius != 1.0:
        raise ValueError("managed mesh identity requires the listed cell count and a unit sphere")


def _cached(entry, name, cells, url):
    mesh, receipt = entry / "mesh.nc", entry / "mesh.json"
    if (entry.is_symlink() or mesh.is_symlink() or receipt.is_symlink()
            or not mesh.is_file() or not receipt.is_file() or receipt.stat().st_size > 16384):
        raise ValueError(f"incomplete or invalid managed mesh cache: {entry}")
    try:
        record = json.loads(receipt.read_text())
        valid = (isinstance(record, dict) and record.get("schema") == 1 and record.get("name") == name
                 and record.get("url") == url and record.get("nCells") == cells
                 and record.get("mesh_bytes") == mesh.stat().st_size
                 and mesh.stat().st_size <= _MESH_LIMIT and record.get("mesh_sha256") == _sha256(mesh))
    except (ValueError, OSError) as exc:
        raise ValueError(f"invalid managed mesh cache receipt: {receipt}") from exc
    if not valid:
        raise ValueError(f"managed mesh cache identity or checksum mismatch: {entry}")
    _validate_mesh(mesh, cells)
    return mesh


def resolve_mesh(mesh: str | Path, *, cache_dir: str | Path | None = None,
                 allow_download: bool = False) -> Path:
    """Resolve a local path or a managed name. Never download or repair a cache implicitly.

    NCAR does not publish an independent digest here. HTTPS and native topology
    checks establish the first cache entry; recorded hashes detect later changes.
    """
    path = Path(mesh)
    name = str(mesh)
    if path.is_file() or name not in STANDARD_MESHES:
        if not path.is_file():
            raise ValueError(f"mesh is not a local file or a supported managed name: {mesh}")
        return path
    if cache_dir is None:
        raise ValueError("managed mesh names require an explicit mesh cache directory")
    cells, url = STANDARD_MESHES[name], _BASE_URL + name + ".tar.gz"
    cache = Path(cache_dir)
    entry = cache / name
    if entry.exists() or entry.is_symlink():
        return _cached(entry, name, cells, url)
    if not allow_download:
        raise ValueError("managed mesh is not cached; explicitly permit a mesh download")
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".mesh-", dir=cache) as temporary:
        stage = Path(temporary)
        archive, staged_mesh = stage / "archive.tar.gz", stage / "mesh.nc"
        archive_bytes, archive_hash = _download(url, archive)
        member = _unpack(archive, staged_mesh)
        _validate_mesh(staged_mesh, cells)
        record = dict(schema=1, name=name, url=url, nCells=cells, member=member,
                      acquired_utc=datetime.now(timezone.utc).isoformat(), archive_bytes=archive_bytes,
                      archive_sha256=archive_hash, mesh_bytes=staged_mesh.stat().st_size,
                      mesh_sha256=_sha256(staged_mesh))
        staged_receipt = stage / "mesh.json"
        staged_receipt.write_text(json.dumps(record, indent=2) + "\n")
        # mkdir claims this entry without replacing even an empty existing directory.
        entry.mkdir()
        published = []
        try:
            for source in (staged_mesh, staged_receipt):
                target = entry / source.name
                os.link(source, target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink()
            with suppress(OSError):
                entry.rmdir()
            raise
    return entry / "mesh.nc"
