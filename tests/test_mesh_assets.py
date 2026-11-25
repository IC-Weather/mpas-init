import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
from urllib.request import Request

import pytest

from mpas_init import mesh_assets
from mpas_init.mesh_assets import resolve_mesh


def archive_bytes(payload=b"transport fixture, not a numerical mesh", *, names=None, kind=None):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as archive:
        for name in names or ["x1.10242/mesh.grid.nc", "x1.10242/graph.info"]:
            member = tarfile.TarInfo(name)
            if kind is not None:
                member.type, member.linkname = kind, "outside.nc"
            else:
                content = payload if name.endswith(".nc") else b"2 0\n"
                member.size = len(content)
            archive.addfile(member, io.BytesIO(content) if member.isfile() else None)
    return result.getvalue()


@pytest.fixture
def transport(monkeypatch):
    """Stub transport/topology only for archive/cache protocol tests; real meshes are tested below."""
    state = dict(body=archive_bytes(), calls=[], cells=10242, radius=1.0, reads=0)

    class Opener:
        def open(self, url, timeout):
            state["calls"].append((url, timeout))
            return io.BytesIO(state["body"])

    def read(path):
        assert path.read_bytes() == b"transport fixture, not a numerical mesh"
        state["reads"] += 1
        return SimpleNamespace(sizes={"nCells": state["cells"]}, sphere_radius=state["radius"])

    monkeypatch.setattr(mesh_assets, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(mesh_assets, "read_raw_mesh", read)
    return state


def test_managed_download_requires_explicit_permission_then_reuses_offline(transport, tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="explicit mesh cache"):
        resolve_mesh("x1.10242", allow_download=True)
    with pytest.raises(ValueError, match="explicitly permit"):
        resolve_mesh("x1.10242", cache_dir=cache)
    assert not cache.exists() and not transport["calls"]
    path = resolve_mesh("x1.10242", cache_dir=cache, allow_download=True)
    assert transport["calls"] == [(mesh_assets._BASE_URL + "x1.10242.tar.gz", 30)]
    record = json.loads(path.with_name("mesh.json").read_text())
    assert record["mesh_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["archive_sha256"] == hashlib.sha256(transport["body"]).hexdigest()
    assert record["archive_bytes"] == len(transport["body"])
    assert record["member"] == "x1.10242/mesh.grid.nc"
    assert record["mesh_bytes"] == path.stat().st_size
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    assert resolve_mesh("x1.10242", cache_dir=cache) == path
    assert len(transport["calls"]) == 1 and transport["reads"] == 2
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
    assert list(cache.iterdir()) == [path.parent]


@pytest.mark.parametrize("name,cells", mesh_assets.STANDARD_MESHES.items())
def test_catalog_identity(transport, tmp_path, name, cells):
    transport["cells"] = cells
    path = resolve_mesh(name, cache_dir=tmp_path, allow_download=True)
    assert json.loads(path.with_name("mesh.json").read_text())["nCells"] == cells
    assert transport["calls"][0][0] == mesh_assets._BASE_URL + name + ".tar.gz"


def test_local_path_never_downloads(transport, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("x1.10242")
    path.write_bytes(b"local mesh; the prepare reader validates this later")
    assert resolve_mesh(path, allow_download=True) == path
    assert not transport["calls"] and transport["reads"] == 0
    with pytest.raises(ValueError, match="not a local file"):
        resolve_mesh("unsupported", cache_dir=tmp_path, allow_download=True)


@pytest.mark.parametrize("problem", ["traversal", "absolute", "backslash", "symlink", "hardlink", "fifo",
                                     "duplicate", "absent", "gzip", "truncated", "crc", "cells", "radius"])
def test_bad_archives_leave_no_published_or_temporary_mesh(transport, tmp_path, problem):
    if problem in ("traversal", "absolute", "backslash"):
        name = {"traversal": "../escape.nc", "absolute": "/escape.nc", "backslash": "..\\escape.nc"}[problem]
        transport["body"] = archive_bytes(names=[name])
    elif problem in ("symlink", "hardlink", "fifo"):
        transport["body"] = archive_bytes(kind={"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                                                "fifo": tarfile.FIFOTYPE}[problem])
    elif problem == "duplicate":
        transport["body"] = archive_bytes(names=["a.nc", "a.nc"])
    elif problem == "absent":
        transport["body"] = archive_bytes(names=["graph.info"])
    elif problem == "gzip":
        transport["body"] = b"not gzip"
    elif problem == "truncated":
        transport["body"] = transport["body"][:-8]
    elif problem == "crc":
        transport["body"] = transport["body"][:-8] + b"\0" * 8
    elif problem == "cells":
        transport["cells"] = 162
    else:
        transport["radius"] = 6371229.0
    with pytest.raises((ValueError, OSError, EOFError, tarfile.TarError)):
        resolve_mesh("x1.10242", cache_dir=tmp_path, allow_download=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("limit", ["download", "mesh", "expanded"])
def test_archive_limits_before_publication(transport, tmp_path, monkeypatch, limit):
    constant = {"download": "_DOWNLOAD_LIMIT", "mesh": "_MESH_LIMIT", "expanded": "_EXPANDED_LIMIT"}[limit]
    monkeypatch.setattr(mesh_assets, constant, 16)
    with pytest.raises(ValueError, match="limit"):
        resolve_mesh("x1.10242", cache_dir=tmp_path, allow_download=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", ["mesh", "schema", "name", "url", "nCells", "mesh_bytes", "mesh_sha256",
                                      "json", "symlink", "missing", "empty"])
def test_bad_cache_never_repairs_or_downloads_implicitly(transport, tmp_path, mutation):
    path = resolve_mesh("x1.10242", cache_dir=tmp_path, allow_download=True)
    receipt = path.with_name("mesh.json")
    if mutation == "mesh":
        body = path.read_bytes()
        path.write_bytes(b"X" + body[1:])
    elif mutation == "json":
        receipt.write_text("{")
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "outside.nc")
    elif mutation in ("missing", "empty"):
        receipt.unlink()
        if mutation == "empty":
            path.unlink()
    else:
        record = json.loads(receipt.read_text())
        record[mutation] = "wrong"
        receipt.write_text(json.dumps(record))
    before = {p.name: p.read_bytes() for p in path.parent.iterdir() if not p.is_symlink()}
    with pytest.raises(ValueError, match="cache"):
        resolve_mesh("x1.10242", cache_dir=tmp_path, allow_download=True)
    assert len(transport["calls"]) == 1
    assert {p.name: p.read_bytes() for p in path.parent.iterdir() if not p.is_symlink()} == before


def test_publication_failure_removes_only_its_entry(transport, tmp_path, monkeypatch):
    original = os.link
    def fail_receipt(source, target):
        if source.name == "mesh.json":
            raise OSError("simulated failed publication")
        original(source, target)
    monkeypatch.setattr(mesh_assets.os, "link", fail_receipt)
    with pytest.raises(OSError, match="simulated"):
        resolve_mesh("x1.10242", cache_dir=tmp_path, allow_download=True)
    assert list(tmp_path.iterdir()) == []


def test_racing_entry_is_not_overwritten(transport, tmp_path, monkeypatch):
    unpack = mesh_assets._unpack
    def racing_unpack(archive, output):
        member = unpack(archive, output)
        entry = tmp_path / "x1.10242"
        entry.mkdir()
        (entry / "keep").write_bytes(b"other writer")
        return member
    monkeypatch.setattr(mesh_assets, "_unpack", racing_unpack)
    with pytest.raises(FileExistsError):
        resolve_mesh("x1.10242", cache_dir=tmp_path, allow_download=True)
    assert (tmp_path / "x1.10242/keep").read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [tmp_path / "x1.10242"]


def test_https_redirect_origin_is_preserved():
    handler = mesh_assets._NCARRedirect()
    request = Request(mesh_assets._BASE_URL + "x1.10242.tar.gz")
    for target in ("http://www2.mmm.ucar.edu/mesh", "https://other.example/mesh", "file:///mesh"):
        with pytest.raises(ValueError, match="HTTPS origin"):
            handler.redirect_request(request, None, 302, "redirect", {}, target)
    target = mesh_assets._BASE_URL + "new-name.tar.gz"
    assert handler.redirect_request(request, None, 302, "redirect", {}, target).full_url == target
