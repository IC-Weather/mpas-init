import numpy as np
import pytest
import torch

from mpas_init.ugwp import (HGT_MISSING, UGWP_BASE_FIELDS, UGWP_FIELDS, ugwp_background, ugwp_block_geometry,
                            ugwp_statistics)
from test_vertical_grid import assert_stage


def test_statistics_derivatives_and_batching():
    gen = torch.Generator().manual_seed(192)
    box = (torch.rand((6,8), generator=gen, dtype=torch.float64)*700 + 50).requires_grad_()
    for small_scale in (False, True):
        def f(x):
            return torch.stack(tuple(ugwp_statistics(x, small_scale=small_scale).values()))
        assert torch.autograd.gradcheck(f, (box,), check_forward_ad=True)
        translated = torch.func.jvp(f, (box,), (torch.ones_like(box),))[1]
        assert_stage(translated, torch.zeros_like(translated), 64)
        scaled = torch.func.jvp(f, (box,), (box,))[1]
        expected = torch.zeros_like(scaled)
        expected[0] = f(box)[0]
        assert_stage(scaled, expected, 64)
        batch = ugwp_statistics(torch.stack((box,box*1.3)), small_scale=small_scale)
        for i,x in enumerate((box,box*1.3)):
            direct = ugwp_statistics(x, small_scale=small_scale)
            assert all(torch.equal(v[i],direct[n]) for n,v in batch.items())
        for value in (0.0, 13.0):
            flat = torch.full_like(box, value, requires_grad=True)
            gradient, = torch.autograd.grad(f(flat).sum(), flat)
            assert torch.isfinite(gradient).all()
    assert len(set(UGWP_FIELDS)) == 20
    with pytest.raises(ValueError, match='nonempty'):
        ugwp_statistics(torch.empty(0,3), small_scale=True)


def test_background_derivatives_and_affine_truth():
    lat_blocks = torch.tensor([-0.2,0.1,0.4],dtype=torch.float64,requires_grad=True)
    lon_blocks = torch.tensor([-0.3,0.2,0.7],dtype=torch.float64,requires_grad=True)
    latitude = torch.tensor([[-0.08,0.27]],dtype=torch.float64,requires_grad=True)
    longitude = torch.tensor([[0.02,0.38]],dtype=torch.float64,requires_grad=True)
    heights = torch.tensor([[30.,70.,20.],[120.,-5.,430.],[210.,320.,85.]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(ugwp_background,(latitude,longitude,lat_blocks,lon_blocks,heights),check_forward_ad=True)
    affine = 30+13*lat_blocks[:,None]+17*lon_blocks[None,:]
    expected = 30+13*latitude+17*longitude
    assert_stage(ugwp_background(latitude,longitude,lat_blocks,lon_blocks,affine),expected,16)
    for row in (0,2):
        missing = heights.detach().clone()
        missing[row,1] = HGT_MISSING
        missing.requires_grad_()
        fixed_missing = missing.detach() == HGT_MISSING
        assert torch.autograd.gradcheck(lambda h: ugwp_background(latitude,longitude,lat_blocks,lon_blocks,
                                          torch.where(fixed_missing,HGT_MISSING,h)),
                                        (missing,),check_forward_ad=True)
    with pytest.raises(ValueError,match='3 by 3'):
        ugwp_background(latitude,longitude,lat_blocks,lon_blocks,torch.zeros(2,2))


def test_background_batches_preserve_scalar_results_and_derivatives():
    lat_blocks = torch.tensor([[-.2, .1, .4], [-.3, .0, .3]], dtype=torch.float64)
    lon_blocks = torch.tensor([[2.7, 3.0, 3.3], [-3.4, -3.1, -2.8]], dtype=torch.float64)
    heights = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3).square()
    latitude = torch.tensor([[[-.1], [.2]], [[-.15], [.15]]], dtype=torch.float64)
    longitude = torch.tensor([[[2.9, -3.1]], [[3.1, -3.0]]], dtype=torch.float64)
    for missing in (None, 0, 2):
        values = heights.clone()
        if missing is not None:
            values[0, missing, 1] = HGT_MISSING
        batched = ugwp_background(latitude, longitude, lat_blocks, lon_blocks, values)
        expected = torch.stack([ugwp_background(latitude[i], longitude[i], lat_blocks[i], lon_blocks[i], values[i])
                                for i in range(2)])
        assert torch.equal(batched, expected)
    arguments = tuple(value.clone().requires_grad_() for value in
                       (latitude, longitude, lat_blocks, lon_blocks, heights))
    assert torch.autograd.gradcheck(ugwp_background, arguments, check_forward_ad=True)
    directions = tuple(torch.ones_like(value)*.03 for value in arguments)
    batched_jvp = torch.func.jvp(ugwp_background, arguments, directions)[1]
    scalar_jvp = torch.stack([torch.func.jvp(ugwp_background,
        tuple(value[i] for value in arguments), tuple(value[i] for value in directions))[1] for i in range(2)])
    assert torch.equal(batched_jvp, scalar_jvp)


def test_ugwp_accepts_poles_and_keeps_other_input_checks():
    from mpas_init.ugwp_prepare import read_ugwp_footprint
    scalar = lambda v: torch.tensor(v,dtype=torch.float64)
    for lat in (-0.5*torch.pi,0.5*torch.pi):
        with pytest.raises(FileNotFoundError):
            read_ugwp_footprint('does-not-exist',scalar(lat),scalar(0),scalar(8000),small_scale=True)
    with pytest.raises(ValueError,match='duplicate'):
        read_ugwp_footprint('does-not-exist',scalar(0),scalar(0),scalar(7499),small_scale=False)


@pytest.mark.parametrize('single_batches', (False, True))
def test_preparation_duplication_protocol(monkeypatch, single_batches):
    from types import SimpleNamespace
    import weakref
    import mpas_init.ugwp_prepare as prepare
    from mpas_init.gwd import RADIUS
    diameter = torch.tensor([7499.,7500.,7501.],dtype=torch.float64)
    mesh = SimpleNamespace(sphere_radius=RADIUS,sizes={'nCells':3},
                           fields={'latCell':torch.tensor([0.,0.1,0.2],dtype=torch.float64),
                                   'lonCell':torch.zeros(3,dtype=torch.float64)},
                           connectivity=SimpleNamespace(mean_dc=lambda:diameter))
    calls = []
    previous = []
    box = torch.arange(48,dtype=torch.float64).reshape(6,8).sin()*500
    monkeypatch.setattr(prepare,'_geography',lambda root: {
        scale: (None, SimpleNamespace(global_shape=shape))
        for scale,shape in ((True,(43200,21600)),(False,(8640,4320)))})
    if single_batches:
        monkeypatch.setattr(prepare, 'footprint_batches',
                            lambda shapes, *args: [(shape, [cell]) for cell, shape in enumerate(shapes)])
        statistics = prepare.ugwp_statistics
        def tracked_statistics(boxes, **kwargs):
            previous.append(weakref.ref(boxes))
            return statistics(boxes, **kwargs)
        monkeypatch.setattr(prepare, 'ugwp_statistics', tracked_statistics)
    def footprint(lat,lon,diameter,geography,small_scale,cache,cache_tiles,**kwargs):
        if single_batches:
            assert all(value() is None for value in previous), 'previous footprint batch remains allocated'
        calls.append((float(lat),small_scale))
        terrain = box.clone() if small_scale else box*0.3
        if single_batches:
            previous.append(weakref.ref(terrain))
        return {'terrain':terrain, 'latitude':torch.zeros(6,1,dtype=torch.float64),
                'longitude':torch.zeros(1,8,dtype=torch.float64),
                'lat_blocks':torch.tensor([-1.,0.,1.],dtype=torch.float64),
                'lon_blocks':torch.tensor([-1.,0.,1.],dtype=torch.float64),
                'heights':torch.zeros(3,3,dtype=torch.float64)}
    monkeypatch.setattr(prepare,'_footprint',footprint)
    fields = prepare.prepare_ugwp(mesh,'unused')
    assert set(fields) == set(UGWP_FIELDS)
    assert sorted(calls) == sorted([(0.,True),(0.1,True),(0.2,True),(0.1,False),(0.2,False)])
    for name in UGWP_BASE_FIELDS:
        assert fields[name+'ls'][0] == fields[name+'ss'][0]
        expected = ugwp_statistics(box*0.3,small_scale=False)[name+'ls']
        assert torch.equal(fields[name+'ls'][1:],expected.expand(2))


@pytest.mark.parametrize('latitude', [-90., -89.999, 89.999, 90.])
@pytest.mark.parametrize('small_scale', [False, True])
def test_polar_geometry_uses_each_longitude_once(latitude, small_scale):
    from mpas_init.gwd import RADIUS
    nx, ny = (43200, 21600) if small_scale else (8640, 4320)
    target_lat = torch.full((4,), latitude * np.pi / 180., dtype=torch.float64)
    target_lon = torch.tensor([-np.pi, -.4, 1.8, 2*np.pi], dtype=torch.float64)
    diameter = torch.full_like(target_lat, 120000.)
    blocks = ugwp_block_geometry(target_lat, target_lon, diameter, nx=nx, ny=ny)
    lat_blocks, lon_blocks, start_i, end_i, start_j, end_j = blocks
    assert torch.equal(start_i, torch.ones_like(start_i))
    assert torch.equal(end_i, torch.full_like(end_i, nx))
    np.testing.assert_array_equal(lon_blocks, np.tile([-2*np.pi, 0., 2*np.pi], (4, 1)))
    fine = np.deg2rad(-90. + (np.arange(ny) + .5) * (180./ny))
    centers = np.deg2rad(latitude) + np.arange(-1, 2) * (120000. / RADIUS)
    expected_s, expected_e = [], []
    for center in centers:
        south, north = center - 60000./RADIUS, center + 60000./RADIUS
        if south > np.pi/2 or north < -np.pi/2:
            expected_s.append(-999); expected_e.append(-999)
        else:
            expected_s.append(min(np.count_nonzero(fine < south) + 1, ny))
            expected_e.append(max(np.count_nonzero(fine <= north), 1))
    np.testing.assert_array_equal(start_j, np.tile(expected_s, (4, 1)))
    np.testing.assert_array_equal(end_j, np.tile(expected_e, (4, 1)))
    for component in blocks:
        assert torch.equal(component, component[0].expand_as(component))
    assert torch.isfinite(lat_blocks).all() and torch.isfinite(lon_blocks).all()


def test_polar_geometry_coordinate_derivatives_are_finite():
    latitude = torch.tensor([np.pi/2, -np.pi/2], dtype=torch.float64)
    longitude = torch.tensor([.7, 2.1], dtype=torch.float64)
    diameter = torch.full_like(latitude, 120000.)
    def centers(lat, lon, size):
        return ugwp_block_geometry(lat, lon, size, nx=720, ny=360)[:2]
    _, tangent = torch.func.jvp(centers, (latitude, longitude, diameter),
                                (torch.zeros_like(latitude), torch.ones_like(longitude), torch.ones_like(diameter)))
    _, pullback = torch.func.vjp(centers, latitude, longitude, diameter)
    gradients = pullback(tuple(torch.ones_like(a) for a in centers(latitude, longitude, diameter)))
    assert all(torch.isfinite(a).all() for a in (*tangent, *gradients))
    assert torch.count_nonzero(gradients[1]) == 0
    assert torch.count_nonzero(tangent[1]) == 0


def test_full_ring_transition_keeps_finite_native_centers_below_threshold():
    from mpas_init.gwd import RADIUS
    cutoff = np.arccos(120000./(2*np.pi*RADIUS))
    latitude = torch.tensor([cutoff-1.e-5, cutoff+1.e-5], dtype=torch.float64)
    longitude = torch.full_like(latitude, .7)
    diameter = torch.full_like(latitude, 120000.)
    centers = ugwp_block_geometry(latitude, longitude, diameter, nx=43200, ny=21600)[1]
    assert centers[0, 1] == .7 and centers[1, 1] == 0
    np.testing.assert_array_equal(centers[1], [-2*np.pi, 0., 2*np.pi])
    assert torch.isfinite(centers).all()


@pytest.mark.parametrize('sign', [-1, 1])
def test_polar_ring_detrending_against_independent_latitude_interpolation(monkeypatch, sign):
    from collections import OrderedDict
    from types import SimpleNamespace
    from mpas_init.gwd import RADIUS
    import mpas_init.ugwp_prepare as prepare

    nx, ny = 144, 72
    fine = (-90. + (np.arange(ny)+.5)*(180./ny))*np.pi/180.
    column = np.r_[np.arange(nx//2), -np.arange(nx//2)]
    terrain = torch.tensor(1000. + 20*np.arange(ny)[:, None] + 3*column[None, :], dtype=torch.float64)
    calls = []
    def read_block(directory, index, si, ei, sj, ej, device, cache, cache_tiles):
        assert si == 1 and ei == nx
        calls.append((sj, ej))
        return terrain[sj-1:ej]
    monkeypatch.setattr(prepare, '_read_block', read_block)
    latitude, longitude, diameter = (torch.tensor(v, dtype=torch.float64) for v in (sign*np.pi/2, 1.23, 1900000.))
    footprint = prepare._footprint(latitude, longitude, diameter,
                    {False: (None, SimpleNamespace(global_shape=(nx, ny)))}, False, OrderedDict(), 2)
    assert set(footprint) == {'terrain', 'box', 'lat_blocks', 'lon_blocks', 'bounds', 'heights', 'background'}
    centers = sign*np.pi/2 + np.arange(-1, 2)*(1900000./RADIUS)
    means = np.full(3, HGT_MISSING)
    bounds = []
    for j, center in enumerate(centers.copy()):
        south, north = center-950000./RADIUS, center+950000./RADIUS
        selected = np.nonzero((fine >= max(south, -np.pi/2)) & (fine <= min(north, np.pi/2)))[0]
        if selected.size:
            bounds.append((int(selected[0]+1), int(selected[-1]+1)))
            means[j] = 1000. + 20*selected.mean()
            if south < -np.pi/2 or north > np.pi/2:
                centers[j] = .5*(fine[selected[0]]+fine[selected[-1]])
        else:
            bounds.append(None)
    central = np.arange(bounds[1][0]-1, bounds[1][1])
    south = 0 if sign > 0 else 1
    expected_background = means[south] + ((means[south+1]-means[south])/(centers[south+1]-centers[south]))*(fine[central]-centers[south])
    expected_background = torch.tensor(np.broadcast_to(expected_background[:, None], (central.size, nx)).copy())
    assert_stage(footprint['background'], expected_background, 16)
    assert_stage(footprint['box'], terrain[central]-expected_background, 16)
    np.testing.assert_array_equal(footprint['heights'], np.repeat(means[:, None], 3, axis=1))
    assert set(calls) == {bound for bound in bounds if bound is not None}
    wrong = terrain[central] - expected_background*0
    with pytest.raises(AssertionError):
        assert_stage(footprint['box'], wrong, 16)


@pytest.mark.parametrize('policy', [('ring',), ('', 'direction'), ('ring', 1), ['ring', 'direction'], ('ring', 'direction')])
def test_writer_rejects_incomplete_or_unattached_ugwp_provenance(tmp_path, policy):
    from mpas_init.output import write_dycore
    from datetime import datetime
    with pytest.raises(ValueError, match='UGWP polar provenance'):
        write_dycore(tmp_path/'invalid.nc', {}, None, datetime(2024, 5, 21),
                     ugwp_polar_extension=policy, ufs_static={} if policy != ('ring', 'direction') else None)
    assert not (tmp_path/'invalid.nc').exists()
