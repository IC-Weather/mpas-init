from netCDF4 import Dataset
import pytest
import torch

from mpas_init._geometry_cache import plan_path, read_plans, write_plans


def candidates():
    return {(1, 1, 0, 0, 32, 64): (torch.tensor([0, 2, 3]), torch.tensor([3, 1, 2])),
            (1, 33, 0, 0, 16, 64): (torch.tensor([0, 1, 3]), torch.tensor([0, 2, 1]))}


def test_compact_geometry_cache_roundtrip_and_budget(tmp_path):
    expected = candidates()
    path = plan_path(tmp_path, 'fixed geometry identity')
    write_plans(path, 'fixed geometry identity', expected, 4, 1024)
    actual = read_plans(path, 'fixed geometry identity', 4, 1024)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert all(torch.equal(a, b) for a, b in zip(actual[key], expected[key]))
    with Dataset(path) as ds:
        assert ds['candidates'].dtype.itemsize == 4
    with pytest.raises(ValueError, match='bounds'):
        read_plans(path, 'fixed geometry identity', 4, 64)
    with pytest.raises(ValueError, match='identity'):
        read_plans(path, 'different ordered mesh', 4, 1024)


@pytest.mark.parametrize('variable', ['windows', 'plan_offsets', 'block_offsets', 'candidates'])
def test_geometry_cache_corruption_rejects_same_loader(tmp_path, variable):
    path = plan_path(tmp_path, 'fixed')
    write_plans(path, 'fixed', candidates(), 4, 1024)
    with Dataset(path, 'a') as ds:
        value = ds[variable][:]
        value.flat[-1] += 1
        ds[variable][:] = value
    with pytest.raises(ValueError, match='checksum'):
        read_plans(path, 'fixed', 4, 1024)


def test_failed_geometry_publication_preserves_verified_previous(tmp_path, monkeypatch):
    path = plan_path(tmp_path, 'fixed')
    write_plans(path, 'fixed', candidates(), 4, 1024)
    previous = path.read_bytes()

    def interrupted(*args):
        raise OSError('interrupted before atomic publication')

    monkeypatch.setattr('mpas_init._geometry_cache.os.replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        write_plans(path, 'fixed', candidates(), 4, 1024)
    assert path.read_bytes() == previous
    assert read_plans(path, 'fixed', 4, 1024).keys() == candidates().keys()
    assert not list(tmp_path.glob('.geometry-*'))
