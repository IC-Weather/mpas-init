from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from mpas_init.diagnostics import initialize_diagnostics, relative_humidity_wrt_ice, near_surface_mixing_ratio
from mpas_init.netcdf import NearSurfaceSource, PressureLevelSource


FIELDS = ("relhum", "t2m", "rh2", "q2", "u10", "v10")


def test_humidity_kernels_gradcheck():
    temperature = torch.tensor([150., 249., 260., 275., 290.],dtype=torch.float64,requires_grad=True)
    humidity = torch.tensor([10., 20., 30., 70., 90.],dtype=torch.float64,requires_grad=True)
    pressure = torch.tensor([50000., 70000., 80000., 90000., 100000.],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(relative_humidity_wrt_ice,(temperature,humidity),check_forward_ad=True)
    assert torch.autograd.gradcheck(near_surface_mixing_ratio,(temperature,humidity,pressure),check_forward_ad=True)


def test_complete_diagnostic_field_coordinate_and_pressure_derivatives(tmp_path):
    latitude = torch.linspace(-90,90,7,dtype=torch.float64)
    longitude = torch.arange(12,dtype=torch.float64)*30
    lat,lon = torch.meshgrid(latitude,longitude,indexing='ij')
    wave = torch.sin(torch.deg2rad(lon)) + torch.deg2rad(lat)/5
    shape = (4,7,12)
    height = latitude.new_tensor([-1000.,5000.,15000.,35000.])[:,None,None].expand(shape) + wave*100
    temp = latitude.new_tensor([290.,265.,230.,210.])[:,None,None].expand(shape) + wave
    rh = latitude.new_tensor([80.,60.,40.,10.])[:,None,None].expand(shape) + wave
    near_values = (280+wave,55+wave,5+wave,-2+wave)
    pressure = latitude.new_tensor([100000.,50000.,15000.,1000.])
    zgrid = latitude.new_tensor([0.,2000.,8000.,20000.]).expand(4,4)
    target_lat = latitude.new_tensor([-.65,.27,-.33,.71])
    target_lon = torch.deg2rad(latitude.new_tensor([43.,107.,223.,287.]))
    inputs = tuple(value.clone().requires_grad_() for value in (temp,height,rh,*near_values,pressure,zgrid,target_lat,target_lon))
    scales = (.1,20.,.1,.1,.1,.1,.1,100.,10.,.001,.001)
    directions = tuple(torch.cos(torch.arange(v.numel(),dtype=v.dtype)).reshape(v.shape)*scale for v,scale in zip(inputs,scales))
    instant = datetime(2024,5,21,tzinfo=timezone.utc)
    def evaluate(*args):
        temp,height,rh,t2,rh2,u,v,p,z,la,lo = args
        source = PressureLevelSource(p,latitude,longitude,{'air_temperature':temp,'geopotential_height':height,
                                      'relative_humidity':rh},instant,instant,timedelta(0))
        near = NearSurfaceSource(latitude,longitude,{'air_temperature':t2,'relative_humidity':rh2,
                                                   'eastward_wind':u,'northward_wind':v},instant,instant,timedelta(0))
        fields = initialize_diagnostics(source,near,{'zgrid':z,'latCell':la,'lonCell':lo})
        return torch.cat([fields[name].flatten() for name in FIELDS])
    output,tangent = torch.func.jvp(evaluate,inputs,directions)
    weights = torch.linspace(.3,1.2,output.numel(),dtype=output.dtype)
    gradient = torch.autograd.grad((output*weights).sum(),inputs)
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g)>0 for g in gradient)
    torch.testing.assert_close(sum((g*d).sum() for g,d in zip(gradient,directions)),(tangent*weights).sum(),rtol=1e-12,atol=1e-12)
    finite = []
    for step in (1e-3,1e-4,1e-5):
        value = (evaluate(*(v+step*d for v,d in zip(inputs,directions)))
                 - evaluate(*(v-step*d for v,d in zip(inputs,directions)))) / (2*step)
        torch.testing.assert_close(tangent,value,rtol=2e-6,atol=2e-8)
        finite.append(value.detach().numpy())
    np.savez(tmp_path / 'derivatives.npz',tangent=tangent.detach().numpy(),finite=np.stack(finite),
             **{f'gradient_{i}':g.numpy() for i,g in enumerate(gradient)})
