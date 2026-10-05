"""Elevation data provider for fetching real topographic data.

Uses SRTM 1 Arc-Second Global tiles (~30m resolution).
No API key required. Data courtesy of NASA/USGS (public domain).
"""

import asyncio
import logging
import math
import os
import pathlib
import struct
from typing import Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


SRTM_TILE_SIZE = 3601  # Each hgt tile is 3601×3601 samples
SRTM_TILE_BYTES = SRTM_TILE_SIZE * SRTM_TILE_SIZE * 2  # big-endian int16, no header


class SRTMTileFetcher:
    """Fetch individual SRTM tiles from public mirrors.
    
    Tile naming: N{lat}W{lon} for northern/western hemisphere tiles.
    For other quadrants: prefix is NS, suffix is EW based on quadrant.
    
    Example filenames:
      N40W105.hgt → 40°N to 41°N, 105°W to 106°W
      S30E28.hgt  → 31°S to 30°S, 28°E to 29°E
    """

    MIRRORS = [
        "https://elevation.data/SRTMGL1_{tile}.hgt",
        "https://srtm.glues.ac.uk/data/{tile}.hgt",
    ]

    @classmethod
    async def fetch_tile(cls, tile_name: str, *, timeout: float = 30.0) -> Optional[np.ndarray]:
        """Fetch a single SRTM hgt tile and return as numpy array.
        
        Returns None if no mirror has the tile or all fail.
        """
        for url_template in cls.MIRRORS:
            full_url = url_template.format(tile=tile_name)
            
            # Validate URL format
            if not all(c.isalnum() or c in '/._-' for c in full_url[:80]):
                continue
                
            try:
                async with aiohttp.ClientSession() as session:
                    resp = await session.get(full_url, timeout=aiohttp.ClientTimeout(total=timeout))
                    
                    if resp.status == 200:
                        raw_data = await resp.read()
                        
                        if len(raw_data) != SRTM_TILE_BYTES:
                            logger.debug("Tile %s has wrong size: %d bytes", tile_name, len(raw_data))
                            continue
                        
                        # Parse big-endian int16
                        flat = np.frombuffer(raw_data, dtype=">i2")
                        return flat.reshape((3601, 3601)).T.astype(np.float64)
                        
                    elif resp.status == 404:
                        continue
                        
            except (aiohttp.ClientError, asyncio.TimeoutError):
                logger.warning("Mirror failed for tile %s", tile_name)
                continue
        
        return None

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


class ElevationDataProcessor:
    """Combine SRTM tiles into a uniform elevation grid at requested resolution.
    
    Pipeline:
    1. Identify all tiles covering the bounding box  
    2. Download each tile (parallel HTTP requests)
    3. Create output grid at desired resolution
    4. Fill each pixel by sampling from input tiles
    5. Return as numpy array E[i,j] where i = south→north, j = west→east
    
    All elevation values are in meters above WGS84 ellipsoid.
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
            resolution_m: Target pixel size in meters (default 30m)
            output_dir: Optional cache directory for downloaded tiles
            
        Returns:
            2D numpy array of elevation values in meters
        """
        # Calculate output dimensions
        from app.utils.projection import bounds_to_meters
        
        width_m, height_m = bounds_to_meters(bounds)
        cols = max(3, int(round(width_m / resolution_m)) + 1)
        rows = max(3, int(round(height_m / resolution_m)) + 1)
        
        if cols < 2 or rows < 2:
            raise ValueError(f"Bounding box too small for resolution: {width_m:.1f}m × {height_m:.1f}m")
        
        # Calculate grid spacing
        dx = width_m / (cols - 1)
        dy = height_m / (rows - 1)
        
        # Get all tile filenames that cover the bounding box  
        tile_names = SRTMTileFetcher.get_tile_names(bounds.west, bounds.south, bounds.east, bounds.north)
        
        if not tile_names:
            raise ValueError(f"No SRTM tiles cover the selected area: {bounds.west:.2f}° to {bounds.east:.2f}°, {bounds.south:.2f}° to {bounds.north:.2f}°")
        
        # Fetch all tiles in parallel (max 4 concurrent)
        semaphore = asyncio.Semaphore(4)
        tiles = {}
        
        async def fetch_with_semaphore(tile_name):
            async with semaphore:
                return tile_name, await SRTMTileFetcher.fetch_tile(tile_name)
        
        fetch_tasks = [fetch_with_semaphore(name) for name in tile_names]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process fetched tiles
        for result in results:
            if isinstance(result, Exception):
                logger.error("Tile fetch failed: %s", result)
                continue
            
            tile_name, tile_data = result
            if tile_data is not None:
                tiles[tile_name] = tile_data
        
        if not tiles:
            raise ValueError("No SRTM data available for the selected area. "
                           "SRTM coverage extends from 60°S to 56°N.")
        
        # Build output grid by sampling from input tiles
        grid = np.zeros((rows, cols), dtype=np.float64)
        
        # Create a lookup of which tile covers each grid cell
        for i in range(rows):
            # North-south position (inverse since array[0] is north/SRTM top)
            lat = bounds.north - dy * i
            
            if lat < -83 or lat > 84:
                # Outside SRTM coverage
                grid[i, :] = 0
                continue
                
            for j in range(cols):
                # West-east position  
                lon = bounds.west + dx * j
                
                # Find the right tile and interpolate
                tile_lat = math.floor(lat)
                tile_lon = math.floor(lon)
                
                # Get tile name
                if tile_lat >= 0 and tile_lon < 0:
                    tile_key = f"N{abs(tile_lat):02d}W{abs(tile_lon):03d}"  
                elif tile_lat >= 0 and tile_lon >= 0:
                    tile_key = f"N{abs(tile_lat):02d}E{tile_lon:03d}"
                elif tile_lat < 0 and tile_lon < 0:
                    tile_key = f"S{abs(tile_lat):02d}W{abs(tile_lon):03d}"
                else:
                    tile_key = f"S{abs(tile_lat):02d}E{tile_lon:03d}"
                
                if tile_key in tiles:
                    tile_data = tiles[tile_key]
                    
                    # Local coordinates within the tile (0 to 1)
                    t_lat = lat - tile_lat
                    t_lon = lon - tile_lon
                    
                    # Find row/col in tile 
                    tile_row = int(round((1 - t_lat) * SRTM_TILE_SIZE))  # Invert for hgt orientation
                    tile_col = int(round(t_lon * SRTM_TILE_SIZE))
                    
                    if (0 <= tile_row < SRTM_TILE_SIZE and 0 <= tile_col < SRTM_TILE_SIZE):
                        # Bilinear interpolation from adjacent pixels
                        elev = self._interpolate_elevation(
                            tile_data, float(tile_row), float(tile_col)
                        )
                        
                        grid[i, j] = elev
        
        return grid
    
    def _interpolate_elevation(self, tile: np.ndarray, row: float, col: float) -> float:
        """Bilinear interpolation from a single point in an SRTM tile."""
        r = int(row)
        c = int(col)
        
        # Check bounds
        if not (0 <= r < SRTM_TILE_SIZE - 1 and 0 <= c < SRTM_TILE_SIZE - 1):
            return float(tile[min(r, SRTM_TILE_SIZE-2), min(c, SRTM_TILE_SIZE-2)])
        
        # Extract surrounding pixels
        e00 = tile[r, c]
        e10 = tile[r, c + 1]
        e01 = tile[r + 1, c]
        e11 = tile[r + 1, c + 1]
        
        # Handle missing data (-9999 or NaN)
        if all(abs(e) > 5000 for e in [e00, e10, e01, e11]):
            return 0.0
        
        # Bilinear interpolation formula
        dr = row - r
        dc = col - c
        
        top = (1 - dr) * e00 + dr * e10
        bot = (1 - dr) * e01 + dr * e11
        
        return float((1 - dc) * top + dc * bot)


async def fetch_elevation_grid(
    bounds,
    resolution_m: float = 30.0,
    output_dir: Optional[pathlib.Path] = None,
) -> np.ndarray:
    """Fetch and return elevation as a 2D NumPy array for the given bounding box."""
    processor = ElevationDataProcessor()
    return await processor.to_grid(bounds, resolution_m, output_dir)
