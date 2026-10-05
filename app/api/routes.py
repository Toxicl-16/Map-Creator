"""FastAPI application: Map-Creator terrain model generator.

Endpoints:
    GET  /api/health              -- health check
    POST /api/geocode             -- resolve place name to bounding box
    POST /api/settings            -- validate & return terrain generation settings
    POST /api/generate/stl        -- generate STL from current selection (blocking)
    GET  /api/download/{token}    -- stream generated binary STL file

Data sources:
    Elevation  : SRTM GL1 30m hgt tiles (no API key required)
    Geocoding  : Nominatim + Photon fallback (ODbL, no API key required)
    Vector     : Overpass API for roads/buildings (ODbL, no API key required)
"""

import asyncio
import logging
import math
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Paths (relative to project root)
DATA_DIR = Path("data")  # shared cache for elevation / vector tiles
DOWNLOADS_DIR = DATA_DIR.joinpath("downloads")  # generated STL output
OUTPUT_DIR = Path("output")

MAX_MODEL_WIDTH_MM = 600
MAX_MODEL_DEPTH_MM = 600


# -- Pydantic models ----------------------------------------------------------


class GeoQuery(BaseModel):
    """Search query: a place name or address."""
    q: str = Field(..., min_length=1, description="Search string", examples=["San Francisco, CA"])

    @field_validator("q")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class BoundingBox(BaseModel):
    """Geographic bounding box (WGS84 degrees)."""
    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)

    @field_validator("west", "east")
    @classmethod
    def validate_lon(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v

    @field_validator("south", "north")
    @classmethod
    def validate_lat(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError("latitude must be between -90 and -90")
        return v


class GridSize(BaseModel):
    """Grid resolution parameters."""
    rows: int = Field(..., ge=32, le=1024)
    cols: int = Field(..., ge=32, le=1024)

    @field_validator("rows", "cols")
    @classmethod
    def validate_positive(cls, v):
        if v < 32:
            raise ValueError("minimum grid size is 32")
        return v


class CreateMeshRequest(BaseModel):
    """Parameters for generating a terrain STL mesh."""
    # Geographic bounds
    west: float = Field(-122.5, ge=-180, le=180)
    south: float = Field(37.7, ge=-90, le=90)
    east: float = Field(-122.4, ge=-180, le=180)
    north: float = Field(37.8, ge=-90, le=90)

    # Model dimensions in mm
    model_width_mm: float = Field(300.0, gt=0, le=600)
    model_depth_mm: float = Field(300.0, gt=0, le=600)

    # Elevation grid resolution (meters per cell)
    resolution_m: float = Field(50.0, gt=5, le=128)

    # Height mapping
    min_altitude_mm: float = Field(2.0, ge=0, lt=100)
    max_altitude_mm: float = Field(50.0, gt=0, lt=1000)
    vertical_exaggeration: float = Field(1.0, gt=0, le=50)

    # Base plate
    base_thickness_mm: float = Field(3.0, ge=0, lt=100)
    base_extension_mm: float = Field(0.0, ge=0, lt=100)

    # Feature inclusion toggles
    include_roads: bool = False
    include_buildings: bool = False
    include_contours: bool = False

    @field_validator("min_altitude_mm")
    @classmethod
    def validate_min_height(cls, v: float, info):
        if hasattr(info, "data"):
            max_h = info.data.get("max_altitude_mm", 100)
            if v >= max_h:
                raise ValueError("min_altitude must be less than max_altitude")
        return v

    @field_validator("model_width_mm", "model_depth_mm")
    @classmethod
    def validate_model_size(cls, v, info):
        if hasattr(info, "data"):
            if not (0 < v <= MAX_MODEL_WIDTH_MM):
                raise ValueError(f"must be between 0 and {MAX_MODEL_WIDTH_MM}")
        return v


class GeocodeResponse(BaseModel):
    """Result of a geocoding lookup."""
    display_name: str
    center_lat: float
    center_lon: float
    bounds: BoundingBox | None = None
    confidence: float = 0.0


class MeshInfo(BaseModel):
    """Metadata about a generated mesh (not the binary data)."""
    token: str
    vertices: int
    triangles: int
    width_mm: float
    depth_mm: float
    height_range_mm: float
    file_size_bytes: int
    duration_ms: float
    bbox: BoundingBox


# -- App lifespan ---------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure data directories exist on startup and clean temp files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Cleanup stale downloads (older than 1 hour) — prevent disk fill
    asyncio.create_task(_cleanup_old_downloads())

    logger.info("Map-Creator starting up")
    yield
    logger.info("Map-Creator shutting down")


async def _cleanup_old_downloads():
    """Remove download files older than ONE_HOUR_SECONDS."""
    ONE_HOUR_SECONDS = 3600
    import os

    now = time.time()
    if not DOWNLOADS_DIR.exists():
        return
    for f in DOWNLOADS_DIR.iterdir():
        age = now - f.stat().st_mtime
        if age > ONE_HOUR_SECONDS:
            f.unlink(missing_ok=True)

# -- App instance ---------------------------------------------------------------


app = FastAPI(
    title="Map-Creator",
    description="Terrain model generator — select a location, generate an STL for 3D printing.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow localhost frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _bbox_width_metres(west, east, lat_mid):
    """Return the approximate west-east ground distance in metres at the given latitude."""
    return abs(east - west) * 111_000.0 * math.cos(math.radians(lat_mid))


def _bbox_height_metres(south, north):
    """Return the approximate south-north ground distance in metres."""
    return (north - south) * 111_320.0


# ---- Endpoint: health check -----------------------------------------------

@app.get("/api/health", tags=["system"])
async def health_check():
    return {"ok": True, "service": "map-creator", "status": "running"}


# ---- Endpoint: geocode ----------------------------------------------------

@app.post("/api/geocode", response_model=GeocodeResponse, tags=["geocoding"])
async def geocode_endpoint(payload: GeoQuery):
    """Resolve a place name or address to geographic coordinates."""
    import os
    from app.geospatial.providers.geocoding import NominatimClient

    user_agent = os.environ.get("NOMINATIM_USER_AGENT", "Map-Creator <support@example.com>")
    nominatim = NominatimClient(user_agent=user_agent)

    result = None
    try:
        search_result = await nominatim.search(payload.q, limit=1)
        if search_result is not None and hasattr(search_result, "lat"):
            # SearchResult is a dataclass from geocoding.py line 28-35:
            # bounding_box stores [south, west, north, east]
            result = {
                "lat": float(search_result.lat),
                "lon": float(search_result.lon),
                "display_name": search_result.display_name or payload.q,
                "boundingbox": [float(b) for b in (search_result.bounding_box or [])],
            }
    except Exception as exc:
        logger.warning("Nominatim/Photon search failed for '%s': %s", payload.q, exc)

    if not result or "lat" not in (result or {}):
        # Fallback: safe default coordinates
        return GeocodeResponse(
            display_name=payload.q,
            center_lat=37.7749, center_lon=-122.4194,
            confidence=0.0,
        )

    bounds_list = result.get("boundingbox") or []
    bbox = None
    if isinstance(bounds_list, list) and len(bounds_list) >= 4:
        try:
            # Nominatim bounding_box order: [south, west, north, east]
            bbox = BoundingBox(
                west=float(bounds_list[1]), north=float(bounds_list[2]),
                east=float(bounds_list[3]), south=float(bounds_list[0]),
            )
        except (ValueError, TypeError):
            pass

    return GeocodeResponse(
        display_name=result.get("display_name", payload.q),
        center_lat=result.get("lat", 37.7749),
        center_lon=result.get("lon", -122.4194),
        bounds=bbox, confidence=0.5,
    )


# ---- Endpoint: generate STL -----------------------------------------------

@app.post("/api/generate/stl", response_model=MeshInfo, tags=["mesh"])
async def generate_mesh(payload: CreateMeshRequest):
    """Generate a terrain mesh and return metadata (not binary data)."""
    import time as _time

    start_ms = _time.time() * 1000
    token = uuid.uuid4().hex[:12]
    out_file = DOWNLOADS_DIR / f"{token}.stl"

    try:
        from app.services.terrain import TerrainSettings, MeshBuilder

        ts = TerrainSettings(
            west=payload.west, south=payload.south,
            east=payload.east, north=payload.north,
            model_width_mm=payload.model_width_mm,
            model_depth_mm=payload.model_depth_mm,
            resolution_m=payload.resolution_m,
            min_altitude_mm=payload.min_altitude_mm,
            max_altitude_mm=payload.max_altitude_mm,
            vertical_exaggeration=payload.vertical_exaggeration,
            base_thickness_mm=payload.base_thickness_mm,
            base_extension_mm=payload.base_extension_mm,
        )

        # Fetch elevation data (blocking — SRTM tiles)
        from app.geospatial.providers.elevation import SRTMTileFetcher

        fetcher = SRTMTileFetcher()
        elev_grid = await fetcher.fetch_elevation(
            west=payload.west, south=payload.south,
            east=payload.east, north=payload.north,
        )
        if elev_grid is None:
            raise HTTPException(status_code=502, detail="Unable to fetch elevation tiles for the selected area.")

        builder = MeshBuilder(ts)
        vertices, triangles = builder.build(elev_grid, elevation_data_available=True)

        out_path = await export_stl_mesh(vertices, triangles, str(out_file))
        file_size = out_file.stat().st_size if out_file.exists() else 0

        elapsed_ms = _time.time() * 1000 - start_ms
        height_range = payload.max_altitude_mm - payload.min_altitude_mm

        return MeshInfo(
            token=token, vertices=len(vertices), triangles=len(triangles),
            width_mm=payload.model_width_mm, depth_mm=payload.model_depth_mm,
            height_range_mm=height_range, file_size_bytes=file_size,
            duration_ms=round(elapsed_ms, 1),
            bbox=BoundingBox(west=payload.west, south=payload.south,
                             east=payload.east, north=payload.north),
        )

    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing import: {exc}")
    except Exception as exc:
        logger.exception("Mesh generation failed for token=%s", token)
        raise HTTPException(status_code=500, detail=str(exc))


# ---- Endpoint: download STL (binary) --------------------------------------

@app.get("/api/download/{token}", tags=["mesh"])
async def download_stl(token: str):
    """Stream the generated STL file back to the client."""
    stl_file = DOWNLOADS_DIR / f"{token}.stl"
    if not stl_file.exists():
        raise HTTPException(status_code=404, detail="Mesh file not found or expired.")

    return Response(
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="terrain_model.stl"'},
        content=stl_file.read_bytes(),
    )


# ---- Internal helpers -----------------------------------------------------


async def get_settings():
    """Load environment settings (lazy, single-call)."""
    from dotenv import load_dotenv

    # Resolve .env relative to project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path, override=False)

    class _Settings:
        pass

    s = _Settings()
    s.open_topography_key = ""  # optional — set OPEN_TOPOGRAPHY_API_KEY
    s.nominatim_user_agent = "Map-Creator <your@email>"
    user_agent = __import__("os").environ.get("NOMINATIM_USER_AGENT", s.nominatim_user_agent)
    if user_agent:
        s.nominatim_user_agent = user_agent
    return s


async def export_stl_mesh(vertices, triangles, output_path: str):
    """Write vertices/triangles to a binary STL file."""
    import struct
    import numpy
    from pathlib import Path as _P

    out = _P(output_path)
    header_str = "Terrain mesh generated by Map-Creator" + "\x00" * (80 - len("Terrain mesh generated by Map-Creator"))
    n_tri = len(triangles)

    data = header_str.encode()[:80] + struct.pack("<I", n_tri)
    for i0, i1, i2 in triangles:
        v0 = numpy.asarray(vertices[i0]) if not isinstance(vertices[i0], numpy.ndarray) else vertices[i0]
        v1 = numpy.asarray(vertices[i1]) if not isinstance(vertices[i1], numpy.ndarray) else vertices[i1]
        v2 = numpy.asarray(vertices[i2]) if not isinstance(vertices[i2], numpy.ndarray) else vertices[i2]

        # Unnormalized face normal
        nx_val = (v0[1] - v1[1]) * (v0[2] - v2[2]) - (v0[2] - v1[2]) * (v0[0] - v2[0])
        ny_val = (v0[2] - v1[2]) * (v0[0] - v2[0]) - (v0[0] - v1[0]) * (v0[1] - v2[1])
        nz_val = (v0[0] - v1[0]) * (v0[1] - v2[1]) - (v0[1] - v1[1]) * (v0[2] - v2[2])

        data += struct.pack("<ffff", float(nx_val), float(ny_val), float(nz_val), 0.0)
        for vert in [vertices[i0], vertices[i1], vertices[i2]]:
            arr = numpy.asarray(vert) if not isinstance(vert, numpy.ndarray) else vert
            data += struct.pack("<ffff", float(arr[0]), float(arr[1]), float(arr[2]), 0.0)

    out.write_bytes(data)
    return out
