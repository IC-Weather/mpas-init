"""Bounded native UGWP geography I/O around the shared tensor kernels."""

import math
from pathlib import Path

import torch
from torch import Tensor

from mpas_init.geotile import GeotileIndex
from mpas_init._geotile_reader import GeotileReaders
from mpas_init._gwd_schedule import footprint_batches
from mpas_init.gwd import RADIUS, _ordered_sum
from mpas_init.mesh import GlobalMesh
from mpas_init.prepare import _complete_tiles
from mpas_init.ugwp import (HGT_MISSING, UGWP_BASE_FIELDS, ugwp_background, ugwp_block_geometry,
                            ugwp_fine_coordinates, ugwp_statistics)


def _geography(root):
    root = Path(root)
    result = {}
    for small_scale, name in ((True, 'topo_ugwp_30s'), (False, 'topo_ugwp_2.5m')):
        index = GeotileIndex.read(root/name/'index')
        geometry = ((43200,21600),0.0083333333,-89.9958333334,-179.9958333333) if small_scale else (
                    (8640,4320),0.0416666667,-89.9791666667,-179.9791666667)
        layout = (index.kind,index.global_shape,index.dx,index.dy,index.known_lat,index.known_lon,
                  index.tile_x,index.tile_y,index.tile_bdr,index.layers,index.tile_z_start,
                  index.wordsize,index.signed,index.endian,index.scale_factor,index.missing_value)
        expected = ('continuous',geometry[0],geometry[1],geometry[1],geometry[2],geometry[3],
                    8640,4320,0,1,1,4,True,'big',0.1,None)
        if layout != expected:
            raise ValueError('UGWP requires native signed, float32-scaled geography: '+name)
        _complete_tiles(root/name,index,'UGWP')
        result[small_scale] = (root/name,index)
    return result


def _coordinates(latitude,longitude,diameter):
    if (latitude.shape != longitude.shape or latitude.shape != diameter.shape
            or any(a.dtype != torch.float64 or a.device != latitude.device for a in (latitude,longitude,diameter))
            or any(not torch.isfinite(a).all() for a in (latitude,longitude,diameter))
            or torch.any(latitude.abs() > 0.5*torch.pi) or torch.any((longitude < -torch.pi) | (longitude > 2*torch.pi))
            or torch.any((diameter <= 0) | (diameter >= torch.pi*RADIUS))):
        raise ValueError('UGWP requires finite float64 coordinates, radians, and positive Earth-scale diameters')


def _read_block(directory,index,start_i,end_i,start_j,end_j,device,cache,cache_tiles):
    nx, _ = index.global_shape
    width = end_i-start_i+1 if start_i <= end_i else nx-start_i+1+end_i
    if min(width,end_j-start_j+1) < 1 or min(start_j,end_j) < 1:
        raise ValueError('UGWP geography has an empty or invalid native block')
    values = torch.empty((end_j-start_j+1,width),dtype=torch.float64,device=device)
    # Native blocks are rectangles, possibly split at the seam. Their integer
    # bounds select byte windows; no pixel sorting or boolean gather is needed.
    row = start_j-1
    while row < end_j:
        ty, local_row = divmod(row,index.tile_y)
        height = min(end_j-row,index.tile_y-local_row)
        column, written = (start_i-1) % nx, 0
        while written < width:
            tx, local_column = divmod(column,index.tile_x)
            count = min(width-written,index.tile_x-local_column)
            path = directory/index.filename(tx*index.tile_x+1,ty*index.tile_y+1)
            raw = cache.read(path,index,window=(local_row,local_column,height,count)).raw[0]
            raw = raw.to(device=device,dtype=torch.float32)
            values[row-start_j+1:row-start_j+1+height,written:written+count] = (raw*0.1).to(torch.float64)
            written += count
            column = (column+count) % nx
        row += height
    return values


def _footprint(latitude,longitude,diameter,geography,small_scale,cache,cache_tiles,
               *, geometry=None, fine_coordinates=None, background=True):
    directory,index = geography[small_scale]
    nx,ny = index.global_shape
    if geometry is None:
        geometry = ugwp_block_geometry(latitude,longitude,diameter,nx=nx,ny=ny,
                                        fine_coordinates=fine_coordinates)
    lat_blocks,lon_blocks,*bounds = geometry
    starts_i,ends_i,starts_j,ends_j = [a.tolist() for a in bounds]
    def block(j,i):
        return _read_block(directory,index,starts_i[i],ends_i[i],starts_j[j],ends_j[j],
                           latitude.device,cache,cache_tiles)
    terrain = block(1,1)
    result = {'terrain':terrain,'box':terrain,'lat_blocks':lat_blocks,'lon_blocks':lon_blocks,
               'bounds':torch.stack(bounds)}
    if not small_scale:
        heights = []
        for j in range(3):
            row = []
            for i in range(3):
                if starts_j[j] == ends_j[j] == -999:
                    row.append(latitude.new_tensor(HGT_MISSING))
                else:
                    values = terrain if i == j == 1 else block(j,i)
                    row.append(_ordered_sum(values)/values.numel())
            heights.append(torch.stack(row))
        heights = torch.stack(heights)
        lat_fine,lon_fine = (ugwp_fine_coordinates(nx,ny,device=latitude.device,dtype=latitude.dtype)
                             if fine_coordinates is None else fine_coordinates)
        rows = torch.arange(starts_j[1]-1,ends_j[1],device=latitude.device)
        columns = torch.arange(starts_i[1]-1,starts_i[1]-1+terrain.shape[1],device=latitude.device).remainder(nx)
        pixel_latitude, pixel_longitude = lat_fine[rows,None], lon_fine[None,columns]
        result.update(heights=heights)
        if background:
            plane = ugwp_background(pixel_latitude,pixel_longitude,lat_blocks,lon_blocks,heights)
            result.update(background=plane,box=terrain-plane)
        else:
            result.update(latitude=pixel_latitude,longitude=pixel_longitude)
    return result


def read_ugwp_footprint(geography: str | Path, latitude: Tensor, longitude: Tensor, diameter: Tensor,
                        *, small_scale: bool) -> dict[str,Tensor]:
    """Decode one complete native footprint and large-scale detrending neighborhood."""
    if latitude.shape != ():
        raise ValueError('UGWP footprint requires scalar coordinates')
    _coordinates(latitude,longitude,diameter)
    if not small_scale and diameter < 7500:
        raise ValueError('native large-scale fields duplicate small-scale values below 7500 m')
    with GeotileReaders(2) as readers:
        return _footprint(latitude,longitude,diameter,_geography(geography),small_scale,readers,2)


def prepare_ugwp(mesh: GlobalMesh, geography: str | Path, *, cell_scaling: float = 1.0,
                  cache_tiles: int = 2) -> dict[str,Tensor]:
    """Prepare all twenty fields, including canonical full-ring polar footprints."""
    if mesh.sphere_radius != RADIUS or not math.isfinite(cell_scaling) or cell_scaling <= 0 or cache_tiles < 1:
        raise ValueError('UGWP requires native Earth-scaled geometry, positive scaling and cache size')
    latitude,longitude = mesh.fields['latCell'],mesh.fields['lonCell']
    diameter = mesh.connectivity.mean_dc()*cell_scaling
    _coordinates(latitude,longitude,diameter)
    geography = _geography(geography)
    duplicate = torch.sqrt(diameter*diameter) < 7500
    fields = {}
    with GeotileReaders(cache_tiles) as cache:
        for small_scale in (True,False):
            nx,ny = geography[small_scale][1].global_shape
            fine = ugwp_fine_coordinates(nx,ny,device=latitude.device,dtype=latitude.dtype)
            geometry = ugwp_block_geometry(latitude,longitude,diameter,nx=nx,ny=ny,fine_coordinates=fine)
            start_i,end_i,start_j,end_j = geometry[2:]
            widths = torch.where(start_i[:,1] <= end_i[:,1], end_i[:,1]-start_i[:,1]+1,
                                  nx-start_i[:,1]+1+end_i[:,1])
            heights = end_j[:,1]-start_j[:,1]+1
            selected = torch.arange(latitude.numel(),device=latitude.device)
            if not small_scale:
                selected = selected[~duplicate]
            cells = selected.tolist()
            shapes = list(zip(widths[selected].tolist(),heights[selected].tolist()))
            suffix = 'ss' if small_scale else 'ls'
            for name in UGWP_BASE_FIELDS:
                fields[name+suffix] = (latitude.new_empty(latitude.shape) if small_scale
                                       else fields[name+'ss'].clone())
            for _, members in footprint_batches(shapes,latitude[selected],longitude[selected]):
                native = [cells[member] for member in members]
                footprints = [_footprint(latitude[cell],longitude[cell],diameter[cell],geography,
                    small_scale,cache,cache_tiles,geometry=tuple(value[cell] for value in geometry),
                    fine_coordinates=fine,background=False) for cell in native]
                boxes = torch.stack([value['terrain'] for value in footprints])
                if not small_scale:
                    plane = ugwp_background(*(torch.stack([value[key] for value in footprints])
                        for key in ('latitude','longitude','lat_blocks','lon_blocks','heights')))
                    boxes = boxes-plane
                result = ugwp_statistics(boxes,small_scale=small_scale)
                ids = torch.tensor(native,dtype=torch.int64,device=latitude.device)
                for name,value in result.items():
                    fields[name][ids] = value
                del footprints, boxes, result
                if not small_scale:
                    del plane
    return fields
