"""File orchestration for implemented global geometry, terrain, and land statics."""

from pathlib import Path

from netCDF4 import Dataset
import torch

from mpas_init._geotile_reader import GeotileReaders
from mpas_init._mapping import PreparationMapping, geometry_descriptor
from mpas_init.categories import category_counts, surface_categories
from mpas_init.climatology import land_accumulation, leaf_area_index, monthly_albedo, snow_albedo, vegetation_fraction
from mpas_init.geotile import GeotileIndex
from mpas_init.geometry import prepare_mesh_geometry
from mpas_init.mesh import GlobalMesh, GlobalStatic, read_raw_mesh, write_static
from mpas_init.static import terrain_mean


def prepare_geography(data_root: str | Path, output: str | Path, *, profile: str = 'ufs-ms'):
    """Convert a complete local collection once; publish only verified integer data."""
    import json
    from mpas_init._geography_index import COLLECTION, _identity, convert_dataset

    if profile not in ('dycore', 'native-noah', 'ufs-ms'):
        raise ValueError('geography conversion supports dycore, native-noah and ufs-ms')
    root, output = Path(data_root), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'geography output already exists: {output}')
    names = {'topo_gmted2010_30s': 'terrain'}
    if profile != 'dycore':
        names.update(modis_landuse_20class_30s='category', soiltype_top_30s='category',
            maxsnowalb_modis='climatology', greenfrac_fpar_modis='climatology',
            albedo_modis='climatology', lai_modis_30s='climatology', soiltemp_1deg=None, landuse_30s=None)
    if profile == 'ufs-ms':
        names.update({'topo_ugwp_30s': None, 'topo_ugwp_2.5m': None})
    inventories = {}
    for name in names:
        index = GeotileIndex.read(root / name / 'index')
        inventories[name] = _complete_tiles(root / name, index, name)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / COLLECTION
    manifest.write_text(json.dumps(dict(schema=1, status='building')) + '\n')
    datasets = {}
    for name, kind in names.items():
        record = convert_dataset(root / name, output / name, summary=kind, tiles=inventories[name])
        datasets[name] = record['identity']
    record = dict(schema=1, status='ready', profile=profile, datasets=datasets)
    record['identity'] = _identity(record)
    staged = output / ('.' + COLLECTION)
    staged.write_text(json.dumps(record, sort_keys=True) + '\n')
    staged.replace(manifest)
    return record


def _mesh_mapping(mesh, block_side, cache_dir=None):
    centers = torch.stack([mesh.fields[axis + 'Cell'] for axis in ('x', 'y', 'z')], -1)
    return PreparationMapping(centers, centers.new_tensor(mesh.sphere_radius),
                              block_side=block_side, cache_dir=cache_dir)


def _check_integer_accumulator(index, factor, *, category=False):
    bound = 1 if category else 1 << (index.wordsize * 8 - int(index.signed))
    nx, ny = index.global_shape
    if bound * nx * ny * factor**2 >= 1 << 63:
        raise ValueError('geography integer accumulator bound exceeds int64')


def _complete_tiles(geography: Path, index: GeotileIndex, label: str):
    nx, ny = index.global_shape
    tiles = [(x, y, geography / index.filename(x, y))
             for y in range(1, ny + 1, index.tile_y) for x in range(1, nx + 1, index.tile_x)]
    from mpas_init._geography_index import indexed_manifest
    record = indexed_manifest(geography)
    if record is not None:
        if [(v['x'], v['y'], v['name']) for v in record['sources']] != [(x, y, p.name) for x, y, p in tiles]:
            raise ValueError(f'missing or incomplete indexed {label} tiles: {geography}')
        return tiles
    expected = (index.tile_x + 2 * index.tile_bdr) * (index.tile_y + 2 * index.tile_bdr) * index.layers * index.wordsize
    # Discover incomplete geography before starting a worldwide pixel calculation.
    for _, _, path in tiles:
        if not path.is_file() or path.stat().st_size != expected:
            raise ValueError(f"missing or incomplete {label} tile: {path}")
    return tiles


def _terrain_input(geography):
    geography = Path(geography)
    index = GeotileIndex.read(geography / "index")
    if index.kind != "continuous" or index.layers != 1:
        raise ValueError("terrain requires one continuous geography layer")
    return index, _complete_tiles(geography, index, "terrain")


def _category_input(geography):
    geography = Path(geography)
    index = GeotileIndex.read(geography / "index")
    if index.kind != "categorical" or index.layers != 1 or index.scale_factor != 1.0:
        raise ValueError("categories require one unscaled categorical geography layer")
    return index, _complete_tiles(geography, index, "category")


def terrain_reductions(mesh: GlobalMesh, geography: str | Path, *, supersample: int = 1,
                         block_side: int = 32, _mapping=None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Stream complete terrain into mergeable integer sums/counts and physical scale.

    This I/O boundary checks decoded masks and final coverage. Pixel mapping and
    reductions use the same tensor kernels as the independently checked patches.
    """
    if supersample < 1 or block_side < 1:
        raise ValueError("supersample and block_side must be positive")
    geography = Path(geography)
    index, tiles = _terrain_input(geography)
    _check_integer_accumulator(index, supersample)
    if _mapping is None:
        with _mesh_mapping(mesh, block_side) as mapping:
            return terrain_reductions(mesh, geography, supersample=supersample, block_side=block_side, _mapping=mapping)
    total = torch.zeros(mesh.sizes["nCells"], dtype=torch.int64, device=_mapping.centers.device)
    count = torch.zeros_like(total)
    with GeotileReaders(1) as cache:
        for x, y, path in tiles:
            reader = cache._reader(path, index)
            from mpas_init._geography_index import terrain_tile_reductions
            sums, counts = terrain_tile_reductions(reader, _mapping, x, y, supersample=supersample)
            total, count = total + sums, count + counts
    return total, count, index.scale_factor


def prepare_terrain(mesh: GlobalMesh, geography: str | Path, *, supersample: int = 1,
                      block_side: int = 32, _mapping=None) -> GlobalStatic:
    """Prepare terrain from the same complete tile reductions used by reference checks."""
    total, count, scale = terrain_reductions(mesh, geography, supersample=supersample, block_side=block_side, _mapping=_mapping)
    terrain = terrain_mean(total, count, scale)
    if not torch.isfinite(terrain).all():
        raise ValueError("terrain geography leaves uncovered global cells; increase supersample explicitly")
    return GlobalStatic({**mesh.fields, "ter": terrain}, {**mesh.dimensions, "ter": ("nCells",)},
                         mesh.sizes, mesh.sphere_radius, mesh.connectivity)


def prepare_dycore_static(mesh_path: str | Path, terrain_path: str | Path, *,
                            sphere_radius: float = 6371229.0, supersample: int = 1,
                            block_side: int = 32, nominal_min_dc: float | None = None) -> GlobalStatic:
    """Raw global mesh and terrain to dycore statics, not complete land/physics statics."""
    mesh = prepare_mesh_geometry(read_raw_mesh(mesh_path), sphere_radius=sphere_radius, nominal_min_dc=nominal_min_dc)
    return prepare_terrain(mesh, terrain_path, supersample=supersample, block_side=block_side)


def category_reductions(mesh: GlobalMesh, geography: str | Path, *, supersample: int = 1,
                        block_side: int = 32, _mapping=None) -> tuple[torch.Tensor, GeotileIndex]:
    """Stream complete category geography into exact, mergeable category-by-cell counts."""
    if supersample < 1 or block_side < 1:
        raise ValueError("supersample and block_side must be positive")
    geography = Path(geography)
    index, tiles = _category_input(geography)
    if _mapping is None:
        with _mesh_mapping(mesh, block_side) as mapping:
            return category_reductions(mesh, geography, supersample=supersample, block_side=block_side, _mapping=mapping)
    return _category_group(mesh, [(index, tiles, supersample)], _mapping)[0], index


def _category_group(mesh, sources, mapping):
    """Apply one exact ownership map to each compatible categorical source."""
    for index, _, factor in sources:
        _check_integer_accumulator(index, factor, category=True)
    counts = [torch.zeros((index.category_max - index.category_min + 1, mesh.sizes['nCells']),
                          dtype=torch.int64, device=mapping.centers.device) for index, _, _ in sources]
    index, tiles, factor = sources[0]
    with GeotileReaders(len(sources)) as cache:
        for number, (x, y, _) in enumerate(tiles):
            readers = [cache._reader(other_tiles[number][2], other_index)
                       for other_index, other_tiles, _ in sources]
            if factor == 1 and all(r._indexed is not None and r._indexed.record['summary'] == 'category' for r in readers):
                from mpas_init._geography_index import summary_reductions
                for slot, (hist, _) in enumerate(summary_reductions(readers, mapping, x, y)):
                    counts[slot] += hist
                continue
            for row in range(0, index.tile_y, 128):
                height = min(128, index.tile_y - row)
                owner = mapping.owners(index, start_x=x, start_y=y, row0=row, height=height,
                                        width=index.tile_x, supersample=factor)
                for slot, ((other, _, _), reader) in enumerate(zip(sources, readers)):
                    tile = reader.read(window=(row, 0, height, index.tile_x))
                    pixels = torch.where(tile.valid[0], tile.raw[0], other.category_min - 1).to(torch.float32).to(torch.int64)
                    pixels = pixels.repeat_interleave(factor, 0).repeat_interleave(factor, 1).reshape(-1)
                    counts[slot] += category_counts(pixels, owner, mesh.sizes['nCells'], other.category_min, other.category_max)
    return counts


def prepare_surface_categories(mesh: GlobalMesh, landuse: str | Path, soil: str | Path, *,
                                landuse_supersample: int = 1, soil_supersample: int = 1,
                                block_side: int = 32, _mapping=None) -> dict[str, torch.Tensor]:
    """Prepare native dominant types, fractions, and landmask, not a complete physics static file."""
    if min(landuse_supersample, soil_supersample, block_side) < 1:
        raise ValueError('supersample and block_side must be positive')
    (li, lt), (si, st) = (_category_input(path) for path in (landuse, soil))
    # These are the bound original geotile manager's defaults, overridden by index metadata.
    land_water = int(li.metadata.get("iswater", "16"))
    land_ice = int(li.metadata.get("isice", "24"))
    soil_water = int(si.metadata.get("isoilwater", "14"))
    if not (li.category_min <= land_water <= li.category_max
            and si.category_min <= soil_water <= si.category_max and si.category_min <= 16 <= si.category_max):
        raise ValueError("category ranges do not contain the native water/ice soil codes")
    if _mapping is None:
        with _mesh_mapping(mesh, block_side) as mapping:
            return prepare_surface_categories(mesh, landuse, soil, landuse_supersample=landuse_supersample,
                                               soil_supersample=soil_supersample, block_side=block_side, _mapping=mapping)
    if geometry_descriptor(li, landuse_supersample, block_side) == geometry_descriptor(si, soil_supersample, block_side):
        land, soil_counts = _category_group(mesh, [(li, lt, landuse_supersample), (si, st, soil_supersample)], _mapping)
    else:
        land, _ = category_reductions(mesh, landuse, supersample=landuse_supersample, block_side=block_side, _mapping=_mapping)
        soil_counts, _ = category_reductions(mesh, soil, supersample=soil_supersample, block_side=block_side, _mapping=_mapping)
    if torch.any(land.sum(0) == 0) or torch.any(soil_counts.sum(0) == 0):
        raise ValueError("category geography leaves uncovered global cells; increase supersample explicitly")
    return surface_categories(land, soil_counts, land_min=li.category_min, soil_min=si.category_min,
                               land_water=land_water, land_ice=land_ice, soil_water=soil_water)


def _climatology_input(geography: str | Path, field: str):
    layers = {"snoalb": 1, "greenfrac": 12, "albedo12m": 12, "lai12m": 12}
    if field not in layers:
        raise ValueError(f"unsupported climatology field: {field}")
    geography = Path(geography)
    index = GeotileIndex.read(geography / "index")
    if index.kind != "continuous" or index.layers != layers[field] or index.tile_z_start != 1:
        raise ValueError(f"{field} requires {layers[field]} continuous layers starting at one")
    if index.missing_value is None or not index.missing_value.is_integer():
        raise ValueError("climatology requires an explicit integer missing_value")
    return geography, index, _complete_tiles(geography, index, field)


def climatology_reductions(mesh: GlobalMesh, geography: str | Path, landmask: torch.Tensor, field: str, *,
                            supersample: int = 1, block_side: int = 32,
                            row_chunk: int = 128, _mapping=None) -> tuple[torch.Tensor, torch.Tensor, GeotileIndex]:
    """Stream native land-only sums/counts, sharing each pixel map across all months.

    At most one decoded tile and one row chunk's mapped monthly pixels are live.
    Native fills, rather than a new coverage policy, handle uncovered land cells.
    """
    if min(supersample, block_side, row_chunk) < 1:
        raise ValueError("supersample, block_side, and row_chunk must be positive")
    if field == "lai12m" and supersample != 1:
        raise ValueError("native LAI does not supersample")
    if (landmask.shape != (mesh.sizes["nCells"],) or landmask.dtype not in (torch.int32, torch.int64)
            or torch.any((landmask != 0) & (landmask != 1))):
        raise ValueError("climatology requires a binary integer landmask on this mesh")
    _, index, tiles = _climatology_input(geography, field)
    if _mapping is None:
        with _mesh_mapping(mesh, block_side) as mapping:
            return climatology_reductions(mesh, geography, landmask, field, supersample=supersample,
                                           block_side=block_side, row_chunk=row_chunk, _mapping=mapping)
    total, count = _climatology_group(mesh, [(field, index, tiles, supersample)], landmask, _mapping, row_chunk)[0]
    return total, count, index


def _climatology_group(mesh, sources, landmask, mapping, row_chunk):
    """Share owners across exact grids, retaining each field's own missing-value rule."""
    for _, index, _, factor in sources:
        _check_integer_accumulator(index, factor)
    mask = landmask.to(mapping.centers.device)
    totals = [torch.zeros((index.layers, mesh.sizes['nCells']), dtype=torch.int64, device=mask.device)
              for _, index, _, _ in sources]
    counts = [torch.zeros_like(mask, dtype=torch.int64) for _ in sources]
    _, index, tiles, factor = sources[0]
    with GeotileReaders(len(sources)) as cache:
        for number, (x, y, _) in enumerate(tiles):
            readers = [cache._reader(other_tiles[number][2], other_index)
                       for _, other_index, other_tiles, _ in sources]
            if factor == 1 and all(r._indexed is not None and r._indexed.record['summary'] == 'climatology' for r in readers):
                from mpas_init._geography_index import summary_reductions
                reduced = summary_reductions(readers, mapping, x, y, landmask=mask, fields=[s[0] for s in sources])
                for slot, (sums, count) in enumerate(reduced):
                    totals[slot] += sums
                    counts[slot] += count
                continue
            for row in range(0, index.tile_y, row_chunk):
                height = min(row_chunk, index.tile_y - row)
                owner = mapping.owners(index, start_x=x, start_y=y, row0=row, height=height,
                                        width=index.tile_x, supersample=factor)
                if not (mask[owner] == 1).any():
                    continue
                for slot, ((field, other, _, _), reader) in enumerate(zip(sources, readers)):
                    raw = reader.read(window=(row, 0, height, index.tile_x)).raw
                    # The original manager stores every layer in C_FLOAT before integer reduction.
                    pixels = raw.to(torch.float32).to(torch.int64)
                    pixels = pixels.repeat_interleave(factor, 1).repeat_interleave(factor, 2)
                    fill = None if field == 'snoalb' else 8 if field == 'albedo12m' else 0
                    sums, count = land_accumulation(pixels.flatten(1), owner, mask, missing=int(other.missing_value), fill=fill)
                    totals[slot] += sums
                    counts[slot] += count
    return list(zip(totals, counts))


def prepare_climatology(mesh: GlobalMesh, surface: dict[str, torch.Tensor], *, snow_albedo_path: str | Path,
                        vegetation_path: str | Path, albedo_path: str | Path, lai_path: str | Path,
                        ice_category: int, supersample: int = 1, vegetation_supersample: int = 1,
                        block_side: int = 32, row_chunk: int = 128, _mapping=None) -> dict[str, torch.Tensor]:
    """Prepare six native MODIS climatology fields, without soil temperature or GWD statics."""
    paths = {"snoalb": snow_albedo_path, "greenfrac": vegetation_path,
             "albedo12m": albedo_path, "lai12m": lai_path}
    if min(supersample, vegetation_supersample, block_side, row_chunk, ice_category) < 1:
        raise ValueError("climatology controls and ice_category must be positive")
    mask, landuse = surface["landmask"], surface["lu_index"]
    if (landuse.shape != (mesh.sizes["nCells"],) or landuse.dtype not in (torch.int32, torch.int64)
            or torch.any(landuse < 1)):
        raise ValueError("climatology requires integer land-use IDs on this mesh")
    # Check every layer inventory before any global reduction starts.
    groups = {}
    for field, path in paths.items():
        _, index, tiles = _climatology_input(path, field)
        factor = vegetation_supersample if field == 'greenfrac' else 1 if field == 'lai12m' else supersample
        key = geometry_descriptor(index, factor, block_side)
        groups.setdefault(key, []).append((field, index, tiles, factor))
    if _mapping is None:
        with _mesh_mapping(mesh, block_side) as mapping:
            return prepare_climatology(mesh, surface, **dict(snow_albedo_path=snow_albedo_path, vegetation_path=vegetation_path,
                                        albedo_path=albedo_path, lai_path=lai_path), ice_category=ice_category,
                                        supersample=supersample, vegetation_supersample=vegetation_supersample,
                                        block_side=block_side, row_chunk=row_chunk, _mapping=mapping)
    reduced = {}
    for sources in groups.values():
        for (field, index, _, _), (total, count) in zip(sources, _climatology_group(mesh, sources, mask, _mapping, row_chunk)):
            reduced[field] = total, count, index
    result = {}
    for field in paths:
        total, count, index = reduced[field]
        if field == "snoalb":
            result[field] = snow_albedo(total, count, index.scale_factor)
        elif field == "greenfrac":
            result.update(vegetation_fraction(total, count))
        elif field == "albedo12m":
            result[field] = monthly_albedo(total, count, index.scale_factor, landuse.to(total.device), ice_category)
        else:
            result[field] = leaf_area_index(total, count, index.scale_factor)
    return result


def prepare_static(mesh_path: str | Path, data_root: str | Path, output: str | Path, *,
                   profile: str = "native-noah", supersample: int = 1,
                   landuse_supersample: int = 1, soil_supersample: int = 1,
                   vegetation_supersample: int = 1, block_side: int = 32,
                   row_chunk: int = 128, mesh_cache: str | Path | None = None,
                   download_mesh: bool = False, nominal_min_dc: float | None = None,
                   cache_dir: str | Path | None = None) -> None:
    """Prepare reusable global statics; managed mesh downloads require explicit permission."""
    # Local imports avoid the GWD reader's dependency on the shared tile checker.
    from mpas_init.gwd_prepare import _geography, prepare_classic_gwd
    from mpas_init.mesh_assets import resolve_mesh
    from mpas_init.soil import _soil_input, prepare_soil_temperature
    from mpas_init.ugwp_prepare import _geography as ugwp_geography, prepare_ugwp
    from mpas_init._geography_index import collection_manifest

    if profile not in ("dycore", "native-noah", "ufs-ms"):
        raise ValueError("static preparation supports dycore, native-noah and ufs-ms")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in
           (supersample, landuse_supersample, soil_supersample, vegetation_supersample, block_side, row_chunk)):
        raise ValueError("static preparation controls must be positive integers")
    output, root = Path(output), Path(data_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"static output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"static output directory does not exist: {output.parent}")
    collection_manifest(root)
    mesh_path = resolve_mesh(mesh_path, cache_dir=mesh_cache, allow_download=download_mesh)
    cache = None
    if cache_dir is not None:
        from mpas_init._prepared_cache import PreparedStaticCache
        parameters = dict(profile=profile, supersample=supersample, landuse_supersample=landuse_supersample,
            soil_supersample=soil_supersample, vegetation_supersample=vegetation_supersample,
            block_side=block_side, row_chunk=row_chunk, nominal_min_dc=nominal_min_dc)
        cache = PreparedStaticCache(cache_dir, mesh_path, root, parameters)
        if cache.load(output):
            return
        cache.fingerprint(mesh_path, root)
    mesh = prepare_mesh_geometry(read_raw_mesh(mesh_path), nominal_min_dc=nominal_min_dc)
    terrain = root / "topo_gmted2010_30s"
    _terrain_input(terrain)
    landuse, soil = root / "modis_landuse_20class_30s", root / "soiltype_top_30s"
    climate_paths = dict(snow_albedo_path=root / "maxsnowalb_modis",
                         vegetation_path=root / "greenfrac_fpar_modis",
                         albedo_path=root / "albedo_modis", lai_path=root / "lai_modis_30s")
    if profile in ("native-noah", "ufs-ms"):
        li, si = (_category_input(path)[0] for path in (landuse, soil))
        if ((li.category_min, li.category_max, li.metadata.get("iswater"), li.metadata.get("isice"))
                != (1, 20, "17", "15") or (si.category_min, si.category_max) != (1, 16)
                or si.metadata.get("isoilwater", "14") != "14"
                or li.metadata.get("mminlu", "").strip('"') != "MODIFIED_IGBP_MODIS_NOAH"):
            raise ValueError("native-noah preparation requires explicit 20-category MODIS and 16-category soil conventions")
        for field, path in zip(("snoalb", "greenfrac", "albedo12m", "lai12m"), climate_paths.values()):
            _climatology_input(path, field)
        _soil_input(root / "soiltemp_1deg")
        _geography(root)
        if profile == "ufs-ms":
            ugwp_geography(root)
    with _mesh_mapping(mesh, block_side, cache_dir=cache_dir) as mapping:
        if cache is not None:
            cache.geometry = dict(mapping._identity)
        static = prepare_terrain(mesh, terrain, supersample=supersample, block_side=block_side, _mapping=mapping)
        if profile == "dycore":
            write_static(output, static)
        else:
            surface = prepare_surface_categories(static, landuse, soil, landuse_supersample=landuse_supersample,
                                                 soil_supersample=soil_supersample, block_side=block_side, _mapping=mapping)
            climate = prepare_climatology(static, surface, **climate_paths, ice_category=15,
                                          supersample=supersample, vegetation_supersample=vegetation_supersample,
                                          block_side=block_side, row_chunk=row_chunk, _mapping=mapping)
    if profile == "dycore":
        if cache is not None:
            cache.publish(output)
        return
    temperature = prepare_soil_temperature(static, root / "soiltemp_1deg", surface["landmask"])
    gwd = prepare_classic_gwd(static, root)
    ugwp = prepare_ugwp(static, root) if profile == "ufs-ms" else None
    write_static(output, static, surface=surface, landuse_name="MODIFIED_IGBP_MODIS_NOAH",
                  climatology=climate, soil_temperature=temperature, classic_gwd=gwd, ugwp=ugwp)
    if ugwp is not None:
        with Dataset(output, "a") as ds:
            ds.mpas_init_ugwp_polar_extension = "Full longitude ring when diameter/R >= 2*pi*cos(latitude); each longitude once from -pi; original latitude shaving and statistics"
            ds.mpas_init_ugwp_polar_directional_convention = "Canonical geographic longitude ordering; no intrinsic pole-relative direction or isotropic closure"
    if cache is not None:
        cache.publish(output)
