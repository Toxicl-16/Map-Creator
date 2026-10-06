"""Elevation data provider for fetching real topographic data.

Uses SRTM 1 Arc-Second Global tiles (~30m resolution).
No API key required. Data courtesy of NASA/USGS (public domain).
"""

import asyncio
import gzip
import logging
import math
import pathlib
from typing import Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


SRTM_TILE_SIZE = 3601  # Each hgt tile is 3601×3601 samples
SRTM_TILE_BYTES = SRTM_TILE_SIZE * SRTM_TILE_SIZE * 2  # big-endian int16, no header

# On-disk cache for downloaded tiles (data/ is gitignored).
TILE_CACHE_DIR = pathlib.Path("data") / "elevation"

# Value used by SRTM/skadi datasets to flag voids (ocean / no data).
VOID_SENTINEL = -32768


class SRTMTileFetcher:
    """Fetch individual SRTM hgt tiles from public mirrors.

    Tile naming: N{lat}W{lon} for northern/western hemisphere tiles.
    For other quadrants: prefix is NS, suffix is EW based on quadrant.

    Example filenames:
      N40W105.hgt → 40°N to 41°N, 105°W to 106°W
      S30E28.hgt  → 31°S to 30°S, 28°E to 29°E
    """

    # Mirrors are tried in order; `gzip` marks sources that need decompression.
    MIRRORS = [
        ("https://srtm.glues.ac.uk/data/{tile}.hgt", False),
        ("https://elevation.data/SRTMGL1_{tile}.hgt", False),
        ("https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}/{tile}.hgt.gz", True),
    ]

    _USER_AGENT = "Map-Creator/0.1 (terrain model generator)"

    # ── disk cache ────────────────────────────────────────────────────────

    @classmethod
    def _cache_path(cls, tile_name: str) -> pathlib.Path:
        return TILE_CACHE_DIR / f"{tile_name}.npy"

    @classmethod
    def load_cached_tile(cls, tile_name: str) -> Optional[np.ndarray]:
        """Return a previously downloaded tile, or None when not cached."""
        path = cls._cache_path(tile_name)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (ValueError, OSError):
            logger.warning("Discarding corrupt elevation cache %s", path)
            path.unlink(missing_ok=True)
            return None

    @classmethod
    def store_tile(cls, tile_name: str, data: np.ndarray) -> None:
        """Persist a downloaded tile so repeat requests never re-download."""
        try:
            TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # np.save appends ".npy" when missing, so the temp name must end
            # in .npy or the atomic rename below silently targets a missing file.
            tmp = cls._cache_path(tile_name).with_name(f"{tile_name}.tmp.npy")
            np.save(tmp, data.astype(np.float32))
            tmp.replace(cls._cache_path(tile_name))
        except OSError as exc:
            logger.warning("Could not cache tile %s: %s", tile_name, exc)

    # ── network ───────────────────────────────────────────────────────────

    @staticmethod
    def _mirror_url(template: str, tile_name: str) -> str:
        # The skadi bucket shards by latitude band, e.g. "N37" for N37W120.
        ns = tile_name[:3]
        return template.format(tile=tile_name, ns=ns)

    @classmethod
    async def fetch_tile(cls, tile_name: str, *, timeout: float = 45.0) -> Optional[np.ndarray]:
        """Fetch a single SRTM hgt tile and return as numpy array.

        Returns None if no mirror has the tile or all fail.
        """
        cached = cls.load_cached_tile(tile_name)
        if cached is not None:
            logger.debug("Elevation tile %s served from cache", tile_name)
            return cached

        for url_template, needs_gzip in cls.MIRRORS:
            full_url = cls._mirror_url(url_template, tile_name)

            try:
                async with aiohttp.ClientSession() as session:
                    resp = await session.get(
                        full_url,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                        headers={"User-Agent": cls._USER_AGENT},
                    )

                    if resp.status == 404:
                        continue

                    if resp.status != 200:
                        logger.debug("Mirror %s returned %d for %s", full_url, resp.status, tile_name)
                        continue

                    raw_data = await resp.read()
                    if needs_gzip:
                        raw_data = gzip.decompress(raw_data)

                    if len(raw_data) != SRTM_TILE_BYTES:
                        logger.debug("Tile %s has wrong size: %d bytes", tile_name, len(raw_data))
                        continue

                    grid = _decode_tile(raw_data)
                    cls.store_tile(tile_name, grid)
                    return grid

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, gzip.BadGzipFile):
                logger.debug("Mirror failed for tile %s: %s", full_url, tile_name)
                continue

        return None

    @classmethod
    async def fetch_elevation(
        cls,
        west: float,
        south: float,
        east: float,
        north: float,
        resolution_m: float = 30.0,
    ) -> np.ndarray:
        """Fetch a height grid in metres for the given bounding box.

        This is the single entry point used by the API layer.
        """
        from app.models import GeoBounds

        processor = ElevationDataProcessor()
        grid = await processor.to_grid(
            GeoBounds(west=west, south=south, east=east, north=north),
            resolution_m=resolution_m,
        )
        if grid is None or not np.isfinite(grid).any():
            raise ElevationUnavailableError(
                f"No elevation data available for {north:.4f}°N {west:.4f}°W "
                f"→ {south:.4f}°S {east:.4f}°E. SRTM coverage spans 60°S–56°N."
            )
        return grid

    @classmethod
    def get_tile_names(cls, west: float, south: float, east: float, north: float):
        """Get the 1°×1° hgt tile names that cover the bounding box."""
        tiles = []

        for lat in range(math.floor(south), math.ceil(north)):
            for lon in range(math.floor(west), math.ceil(east)):
                # Standard SRTM naming for northern/western hemisphere (most common)
                if lat >= 0 and lon < 0:
                    tile = f"N{abs(lat):02d}W{abs(lon):03d}"
                elif lat >= 0 and lon >= 0:
                    tile = f"N{abs(lat):02d}E{lon:03d}"
                elif lat < 0 and lon < 0:
                    tile = f"S{abs(lat):02d}W{abs(lon):03d}"
                else:
                    tile = f"S{abs(lat):02d}E{lon:03d}"

                tiles.append(tile)

        return tiles


class ElevationUnavailableError(Exception):
    """Raised when no usable elevation raster exists for a bounding box."""


def _decode_tile(raw_data: bytes) -> np.ndarray:
    """Decode raw ``.hgt`` bytes into a float64 grid with row 0 = southern edge.

    ``.hgt`` samples are big-endian int16 laid out north→south, east→west. The
    transpose flips both axes so row 0 is the southern edge and column 0 the
    western edge, matching :class:`ElevationDataProcessor`'s documented order.
    No-data sentinels become NaN.
    """
    flat = np.frombuffer(raw_data, dtype=">i2").reshape(
        (SRTM_TILE_SIZE, SRTM_TILE_SIZE)
    )
    grid = flat.T.astype(np.float64)
    grid[grid <= VOID_SENTINEL] = np.nan
    return grid


class ElevationDataProcessor:
    """Combine SRTM tiles into a uniform elevation grid at requested resolution.

    Pipeline:
    1. Identify all tiles covering the bounding box
    2. Download each tile (parallel HTTP requests, disk-cached)
    3. Create output grid at desired resolution
    4. Fill each cell by bilinear sampling from the source tiles
    5. Return as numpy array E[i, j] where i = south→north, j = west→east

    All elevation values are in metres above the WGS84 ellipsoid. Cells with no
    source data are NaN; use :meth:`fill_nan` before meshing.
    """

    def __init__(self):
        self._tile_fetcher = SRTMTileFetcher()

    async def to_grid(
        self,
        bounds,
        resolution_m: float = 30.0,
        output_dir: Optional[pathlib.Path] = None,
    ) -> np.ndarray:
        """Fetch elevation data and return a grid at the requested resolution.

        Args:
            bounds: GeoBounds object with west/south/east/north attributes
            resolution_m: Target cell size in metres
            output_dir: Optional cache directory for downloaded tiles

        Returns:
            2D numpy array of elevation values in metres, NaN where uncovered.
        """
        from app.utils.projection import bounds_to_meters

        width_m, height_m = bounds_to_meters(bounds)
        cols = max(2, int(round(width_m / resolution_m)) + 1)
        rows = max(2, int(round(height_m / resolution_m)) + 1)

        tile_names = SRTMTileFetcher.get_tile_names(
            bounds.west, bounds.south, bounds.east, bounds.north
        )
        if not tile_names:
            raise ValueError(
                f"No SRTM tiles cover {bounds.north:.4f}°→{bounds.south:.4f}° lat, "
                f"{bounds.west:.4f}°→{bounds.east:.4f}° lon."
            )

        tiles = await self._fetch_tiles(tile_names)
        if not tiles:
            raise ValueError(
                "No SRTM data available for the selected area "
                "(coverage extends from 60°S to 56°N)."
            )

        return self._resample(tiles, bounds, rows, cols)

    async def _fetch_tiles(self, tile_names: list[str]) -> dict[str, np.ndarray]:
        """Download tiles with bounded concurrency; failures are skipped."""
        semaphore = asyncio.Semaphore(3)
        tiles: dict[str, np.ndarray] = {}

        async def fetch(tile_name: str):
            async with semaphore:
                return tile_name, await SRTMTileFetcher.fetch_tile(tile_name)

        results = await asyncio.gather(
            *(fetch(name) for name in tile_names), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Elevation tile fetch failed: %s", result)
                continue
            name, data = result
            if data is not None:
                tiles[name] = data
        return tiles

    def _resample(self, tiles, bounds, rows: int, cols: int) -> np.ndarray:
        """Bilinearly resample the source tiles onto a rows×cols grid."""
        from app.utils.projection import bounds_to_meters

        width_m, height_m = bounds_to_meters(bounds)
        # Row 0 is the southern edge, matching this class's documented contract
        # and the row order the terrain mesh and feature rasters expect.
        lat_axis = np.linspace(bounds.south, bounds.north, rows)
        lon_axis = np.linspace(bounds.west, bounds.east, cols)

        # Group target cells by the 1° tile they fall in so each tile is sampled once.
        lat_floor, t_lat = _tile_index(lat_axis, math.floor(bounds.south))
        lon_floor, t_lon = _tile_index(lon_axis, math.floor(bounds.west))

        grid = np.full((rows, cols), np.nan, dtype=np.float64)
        last = SRTM_TILE_SIZE - 1

        for tile_name, tile in tiles.items():
            tile_lat = int(tile_name[1:3]) * (1 if tile_name[0] == "N" else -1)
            tile_lon = int(tile_name[4:7]) * (-1 if tile_name[3] == "W" else 1)

            row_sel = np.flatnonzero(lat_floor == tile_lat)
            if row_sel.size == 0:
                continue
            col_sel = np.flatnonzero(lon_floor == tile_lon)
            if col_sel.size == 0:
                continue

            fr = np.clip(t_lat[row_sel] * last, 0, last)
            fc = np.clip(t_lon[col_sel] * last, 0, last)
            grid[np.ix_(row_sel, col_sel)] = _bilinear(tile, fr, fc)

        return grid

    @staticmethod
    def fill_nan(grid: np.ndarray) -> np.ndarray:
        """Replace void cells by iterative nearest-valid-neighbour dilation.

        SRTM voids cluster around water and deep shadow; the model needs a
        continuous surface, so gaps are filled from the surrounding terrain.
        """
        filled = grid.copy()
        mask = np.isnan(filled)
        if not mask.any():
            return filled

        for _ in range(max(grid.shape)):
            if not mask.any():
                break
            padded = np.pad(filled, 1, mode="edge")
            stack = np.stack(
                [
                    padded[0:-2, 1:-1],
                    padded[2:, 1:-1],
                    padded[1:-1, 0:-2],
                    padded[1:-1, 2:],
                ]
            )
            # Explicit masked mean: np.nanmean warns on all-NaN neighbourhoods,
            # which are expected inside a void cluster.
            valid = ~np.isnan(stack)
            counts = valid.sum(axis=0)
            totals = np.where(valid, stack, 0.0).sum(axis=0)
            neighbour_mean = np.divide(
                totals,
                counts,
                out=np.full(totals.shape, np.nan),
                where=counts > 0,
            )
            grow = mask & ~np.isnan(neighbour_mean)
            filled[grow] = neighbour_mean[grow]
            mask &= ~grow

        # Anything still void (fully surrounded region) collapses to 0 m.
        filled[np.isnan(filled)] = 0.0
        return filled


def _tile_index(axis: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    """Split sample coordinates into (tile index, fraction within the tile).

    A coordinate sitting exactly on a tile boundary (a selection whose edge
    lands on a whole degree) floors into the *next* tile, which was never
    fetched, leaving that row or column empty. Such samples are attributed to
    the tile below at fraction 1.0 instead - the same physical point.
    """
    base = np.floor(axis).astype(np.int64)
    frac = axis - base

    shifted = base - 1
    use_shifted = (frac <= 1e-12) & (shifted >= minimum)

    return np.where(use_shifted, shifted, base), np.where(use_shifted, 1.0, frac)


def _bilinear(tile: np.ndarray, fr: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2-D tile at fractional row/col positions."""
    r0 = np.floor(fr).astype(int)
    c0 = np.floor(fc).astype(int)
    r0 = np.clip(r0, 0, tile.shape[0] - 2)
    c0 = np.clip(c0, 0, tile.shape[1] - 2)
    dr = (fr - r0)[:, None]
    dc = (fc - c0)[None, :]

    e00 = tile[np.ix_(r0, c0)]
    e01 = tile[np.ix_(r0, c0 + 1)]
    e10 = tile[np.ix_(r0 + 1, c0)]
    e11 = tile[np.ix_(r0 + 1, c0 + 1)]

    top = e00 * (1 - dr) + e10 * dr
    bottom = e01 * (1 - dr) + e11 * dr
    return top * (1 - dc) + bottom * dc


async def fetch_elevation_grid(
    bounds,
    resolution_m: float = 30.0,
    output_dir: Optional[pathlib.Path] = None,
) -> np.ndarray:
    """Fetch and return elevation as a 2D NumPy array for the given bounding box."""
    processor = ElevationDataProcessor()
    return await processor.to_grid(bounds, resolution_m, output_dir)
