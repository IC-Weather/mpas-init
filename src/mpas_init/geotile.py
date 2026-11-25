"""WPS geotile decoding boundary. Numerical remapping lives in static.py."""

import configparser
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class GeotileIndex:
    kind: str
    dx: float
    dy: float
    known_lat: float
    known_lon: float
    tile_x: int
    tile_y: int
    tile_z_start: int
    tile_z_end: int
    tile_bdr: int
    wordsize: int
    signed: bool
    endian: str
    scale_factor: float
    missing_value: float | None
    category_min: int | None
    category_max: int | None
    metadata: Mapping[str, str]

    @classmethod
    def read(cls, path: str | Path) -> "GeotileIndex":
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        try:
            parser.read_string("[geotile]\n" + Path(path).read_text(encoding="ascii"))
            if parser.sections() != ["geotile"] or parser.defaults():
                raise ValueError("geotile indexes contain key=value records, without sections or defaults")
            p = parser["geotile"]
            if p["projection"].lower() != "regular_ll":
                raise ValueError("only regular_ll geotiles are implemented")
            if float(p["known_x"]) != 1.0 or float(p["known_y"]) != 1.0:
                raise ValueError("native static geotiles require known_x=known_y=1")
            if ("tile_z" in p) == ("tile_z_start" in p or "tile_z_end" in p):
                raise ValueError("provide tile_z or both tile_z_start and tile_z_end")
            z0 = 1 if "tile_z" in p else int(p["tile_z_start"])
            z1 = int(p["tile_z"]) if "tile_z" in p else int(p["tile_z_end"])
            signed = p.get("signed", "no").lower()
            if signed not in ("yes", "no"):
                raise ValueError("signed must be yes or no")
            index = cls(p["type"].lower(), float(p["dx"]), float(p["dy"]),
                        float(p["known_lat"]), float(p["known_lon"]), int(p["tile_x"]), int(p["tile_y"]),
                        z0, z1, int(p.get("tile_bdr", "0")), int(p["wordsize"]), signed == "yes",
                        p.get("endian", "big").lower(), float(p.get("scale_factor", "1")),
                        float(p["missing_value"]) if "missing_value" in p else None,
                        int(p["category_min"]) if "category_min" in p else None,
                        int(p["category_max"]) if "category_max" in p else None,
                        MappingProxyType(dict(p)))
            row_order = p.get("row_order", "").lower()
        except (configparser.Error, KeyError) as exc:
            raise ValueError(f"invalid geotile index: {path}: {exc}") from exc
        numbers = (index.dx, index.dy, index.known_lat, index.known_lon, index.scale_factor)
        if not all(math.isfinite(x) for x in numbers) or not index.dx or not index.dy:
            raise ValueError("nonfinite geotile geometry or zero pixel spacing")
        if index.scale_factor <= 0 or not -90 <= index.known_lat <= 90:
            raise ValueError("invalid geotile scale or known latitude")
        if index.missing_value is not None and not math.isfinite(index.missing_value):
            raise ValueError("geotile missing_value must be finite")
        if index.kind not in ("continuous", "categorical") or index.endian not in ("big", "little"):
            raise ValueError("unsupported geotile type or endian")
        if index.wordsize not in (1, 2, 3, 4) or min(index.tile_x, index.tile_y, index.tile_z_start) < 1:
            raise ValueError("invalid geotile wordsize or dimensions")
        if index.tile_z_end < index.tile_z_start or index.tile_bdr < 0:
            raise ValueError("invalid geotile layer range or border")
        expected_order = "bottom_top" if index.dy > 0 else "top_bottom"
        if row_order and row_order != expected_order:
            raise ValueError("row_order must agree with the signed dy coordinate")
        if index.kind == "categorical" and (index.category_min is None or index.category_max is None
                                            or index.category_min > index.category_max):
            raise ValueError("categorical geotiles require a valid category range")
        nx, ny = index.global_shape
        if nx % index.tile_x or ny % index.tile_y or min(nx, ny) < 1:
            raise ValueError("geotiles must cover a complete global lattice")
        return index

    @property
    def global_shape(self) -> tuple[int, int]:
        return math.floor(360.0 / abs(self.dx) + 0.5), math.floor(180.0 / abs(self.dy) + 0.5)

    @property
    def layers(self) -> int:
        return self.tile_z_end - self.tile_z_start + 1

    def filename(self, start_x: int, start_y: int) -> str:
        nx, ny = self.global_shape
        if (not 1 <= start_x <= nx or not 1 <= start_y <= ny
                or (start_x - 1) % self.tile_x or (start_y - 1) % self.tile_y):
            raise ValueError("tile starts must be one-based global tile boundaries")
        return f"{start_x:05d}-{start_x+self.tile_x-1:05d}.{start_y:05d}-{start_y+self.tile_y-1:05d}"


@dataclass(frozen=True)
class Geotile:
    raw: Tensor  # layer, row, column; scale_factor has not been applied
    valid: Tensor


def read_geotile(path: str | Path, index: GeotileIndex, *, include_border: bool = False,
                 window: tuple[int, int, int, int] | None = None, layers: slice | None = None) -> Geotile:
    """Decode exact two's-complement integers and masks before physical scaling."""
    from mpas_init._geotile_reader import GeotileReader

    with GeotileReader(path, index) as reader:
        return reader.read(include_border=include_border, window=window, layers=layers)
