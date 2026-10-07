"""FastAPI application: Map-Creator terrain model generator.

Endpoints:
    GET  /api/health              -- health check
    POST /api/geocode             -- resolve place name to bounding box
    POST /api/generate/stl        -- build, export and validate an STL model
    GET  /api/download/{token}    -- download the generated binary STL

Data sources (all keyless):
    Elevation  : SRTM GL1 hgt tiles
    Geocoding  : Nominatim (OpenStreetMap)
    Vector     : Overpass API for roads / building footprints
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from app.models import GeoBounds

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "frontend" / "static"
DATA_DIR = ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"

MAX_MODEL_WIDTH_MM = 600
MAX_MODEL_DEPTH_MM = 600
DOWNLOAD_TTL_SECONDS = 3600

# Overpass and Nominatim both ask for an identifying User-Agent.
DEFAULT_USER_AGENT = "Map-Creator terrain model generator (https://github.com/local/map-creator)"


# -- Pydantic models ----------------------------------------------------------


class GeoQuery(BaseModel):
    """Search query: a place name or address."""

    q: str = Field(..., min_length=1, examples=["Yosemite Valley, CA"])

    @field_validator("q")
    @classmethod
    def strip_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("search query must not be blank")
        return cleaned


class BoundingBox(BaseModel):
    """Geographic bounding box (WGS84 degrees)."""

    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)

    @model_validator(mode="after")
    def check_extent(self):
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        return self

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2.0


class CreateMeshRequest(BaseModel):
    """Parameters for generating a terrain STL model."""

    # Geographic bounds
    west: float = Field(-122.5, ge=-180, le=180)
    south: float = Field(37.7, ge=-90, le=90)
    east: float = Field(-122.4, ge=-180, le=180)
    north: float = Field(37.8, ge=-90, le=90)

    # Physical model dimensions (mm)
    model_width_mm: float = Field(300.0, gt=0, le=MAX_MODEL_WIDTH_MM)
    model_depth_mm: float = Field(300.0, gt=0, le=MAX_MODEL_DEPTH_MM)

    # Elevation grid resolution (metres per cell)
    resolution_m: float = Field(50.0, ge=10, le=128)

    # Height mapping
    min_altitude_mm: float = Field(2.0, ge=0, lt=100)
    max_altitude_mm: float = Field(50.0, gt=0, lt=1000)
    vertical_exaggeration: float = Field(1.0, gt=0, le=50)
    smoothing_passes: int = Field(2, ge=0, le=8)

    # Base plate
    base_thickness_mm: float = Field(3.0, ge=0, lt=100)
    base_extension_mm: float = Field(0.0, ge=0, lt=100)

    # Feature toggles
    include_roads: bool = False
    include_buildings: bool = False
    include_contours: bool = False
    road_width_mm: float = Field(3.0, gt=0, le=20)
    road_height_mm: float = Field(2.0, gt=0, le=20)
    building_height_mm: float = Field(12.0, gt=0, le=50)
    contour_interval_m: float = Field(20.0, gt=0, le=2000)
    contour_thickness_mm: float = Field(0.6, gt=0, le=10)
    contour_height_mm: float = Field(0.8, ge=-10, le=10)
    contours_engraved: bool = False

    @model_validator(mode="after")
    def check_request(self):
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("selection must have west < east and south < north")
        if self.min_altitude_mm >= self.max_altitude_mm:
            raise ValueError("min_altitude_mm must be less than max_altitude_mm")

        from app.utils.projection import bounds_to_meters

        width_m, height_m = bounds_to_meters(self.to_geo_bounds())
        if width_m <= 0 or height_m <= 0:
            raise ValueError("selection has no ground area")
        if width_m < 50 or height_m < 50:
            raise ValueError("selection is too small to mesh (minimum 50 m across)")
        if width_m > 60_000 or height_m > 60_000:
            raise ValueError("selection is too large (maximum 60 km across)")
        return self

    def to_geo_bounds(self) -> GeoBounds:
        return GeoBounds(
            west=self.west, south=self.south, east=self.east, north=self.north
        )


class GeocodeResponse(BaseModel):
    """Result of a geocoding lookup."""

    display_name: str
    center_lat: float
    center_lon: float
    bounds: BoundingBox
    confidence: float = 0.0


class FeatureStats(BaseModel):
    """How much of each optional feature ended up in the heightfield."""

    roads: int = 0
    buildings: int = 0
    contours: int = 0


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
    validation: dict
    elevation: dict
    features: FeatureStats
    requested_features: list[str] = []
    ground_area_m: float = 0.0


class JobAccepted(BaseModel):
    """Acknowledgement that a generation job has started."""

    job_id: str
    token: str
    status_url: str
    stages: list[str]


class ProgressResponse(BaseModel):
    """Live progress for a generation job."""

    job_id: str
    stage: Optional[str] = None
    message: str = ""
    stages: list[str] = []
    completed: int = 0
    percent: int = 0
    done: bool = False
    error: Optional[str] = None
    result: Optional[dict] = None


class PipelineError(Exception):
    """A failure that should be reported to the user, not as a 500."""


class Job:
    """Tracks the real state of one background generation run."""

    def __init__(self, token: str, stages: list[str]):
        self.token = token
        self.stages = list(stages)
        self.stage: Optional[str] = None
        self.message = "Queued"
        self.completed = 0
        self.done = False
        self.error: Optional[str] = None
        self.result: Optional[dict] = None
        self.created = time.time()
        self.history: list[str] = []

    def enter(self, stage: str, message: str) -> None:
        """Mark *stage* as started."""
        self.stage = stage
        self.message = message
        self.history.append(stage)

    def complete(self, stage: str, message: str) -> None:
        """Mark *stage* as finished and advance the counter."""
        self.completed += 1
        self.stage = stage
        self.message = message

    def fail(self, message: str) -> None:
        self.error = message
        self.message = message


# In-memory job registry. Entries are small and short-lived.
_JOBS: dict[str, Job] = {}
_TASKS: dict[str, asyncio.Task] = {}


# -- App lifespan -------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare data directories and prune expired downloads on startup."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(_cleanup_old_downloads())
    logger.info("Map-Creator starting up")
    yield
    logger.info("Map-Creator shutting down")


def _prune_downloads(max_age_seconds: float = DOWNLOAD_TTL_SECONDS) -> int:
    """Delete expired STLs and finished jobs. Returns files removed."""
    now = time.time()
    cutoff = now - max_age_seconds
    removed = 0

    if DOWNLOADS_DIR.is_dir():
        for path in DOWNLOADS_DIR.glob("*.stl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.debug("Could not remove %s", path, exc_info=True)

    for token, job in list(_JOBS.items()):
        if now - job.created > max_age_seconds:
            _JOBS.pop(token, None)

    return removed


async def _cleanup_old_downloads() -> None:
    """Periodically prune expired downloads for the lifetime of the app."""
    try:
        while True:
            await asyncio.sleep(600)
            removed = _prune_downloads()
            if removed:
                logger.info("Pruned %d expired download(s)", removed)
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - background task must never crash
        logger.debug("Download cleanup stopped", exc_info=True)


# -- App instance -------------------------------------------------------------

app = FastAPI(
    title="Map-Creator",
    description="Terrain model generator — select a location, generate an STL for 3D printing.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -- Helpers ------------------------------------------------------------------


def _user_agent() -> str:
    return os.environ.get("NOMINATIM_USER_AGENT", DEFAULT_USER_AGENT)


def _bbox_from_list(values) -> Optional[BoundingBox]:
    """Convert a Nominatim ``boundingbox`` array to a :class:`BoundingBox`.

    Nominatim orders the array ``[south, north, west, east]``.
    """
    if not values or len(values) < 4:
        return None
    try:
        south, north, west, east = (float(v) for v in values[:4])
        return BoundingBox(west=west, south=south, east=east, north=north)
    except (TypeError, ValueError):
        return None


def _default_box(lat: float, lon: float, size_m: float = 1500.0) -> BoundingBox:
    """A box around a point for results that carry no boundary of their own."""
    from app.utils.projection import meters_to_degrees

    deg_lon, deg_lat = meters_to_degrees(size_m, size_m, lat)
    return BoundingBox(
        west=lon - deg_lon / 2,
        east=lon + deg_lon / 2,
        south=lat - deg_lat / 2,
        north=lat + deg_lat / 2,
    )


# -- Endpoints ----------------------------------------------------------------


@app.get("/api/health", tags=["system"])
async def health_check():
    return {
        "ok": True,
        "service": "map-creator",
        "status": "running",
        "version": app.version,
    }


@app.get("/", include_in_schema=False)
async def index():
    """Serve the single-page frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="frontend is not installed")
    return FileResponse(str(index_file))


@app.post("/api/geocode", response_model=GeocodeResponse, tags=["geocoding"])
async def geocode_endpoint(payload: GeoQuery):
    """Resolve a place name to coordinates and a selectable bounding box."""
    from app.geospatial.providers.geocoding import NominatimClient

    client = NominatimClient(user_agent=_user_agent())

    try:
        result = await client.search(payload.q, limit=1)
    except Exception as exc:
        logger.warning("Geocoding failed for %r: %s", payload.q, exc)
        raise HTTPException(status_code=502, detail="location search is unavailable") from exc

    if result is None:
        raise HTTPException(
            status_code=404, detail=f'no location found for "{payload.q}"'
        )

    lat = float(result.lat)
    lon = float(result.lon)

    try:
        bbox = _bbox_from_list(result.bounding_box) or _default_box(lat, lon)
    except ValueError:
        bbox = _default_box(lat, lon)

    return GeocodeResponse(
        display_name=result.display_name or payload.q,
        center_lat=lat,
        center_lon=lon,
        bounds=bbox,
        confidence=0.9 if result.bounding_box else 0.5,
    )


@app.post("/api/generate/stl", response_model=JobAccepted, status_code=202, tags=["mesh"])
async def generate_mesh(payload: CreateMeshRequest):
    """Start a generation job.

    A full run takes several seconds (SRTM downloads, Overpass queries, mesh
    assembly), so the work happens in the background and reports genuine stage
    transitions through ``GET /api/progress/{token}``. Poll until ``done``,
    then download ``token``.
    """
    token = uuid.uuid4().hex[:12]

    stages = ["elevation"]
    if payload.include_roads or payload.include_buildings or payload.include_contours:
        stages.append("features")
    stages += ["mesh", "export", "validate"]

    job = Job(token=token, stages=stages)
    _JOBS[token] = job

    task = asyncio.create_task(_run_pipeline(job, payload))
    _TASKS[token] = task
    task.add_done_callback(lambda _t, key=token: _TASKS.pop(key, None))

    return JobAccepted(
        job_id=token,
        token=token,
        status_url=f"/api/progress/{token}",
        stages=stages,
    )


@app.get("/api/progress/{token}", response_model=ProgressResponse, tags=["mesh"])
async def generation_progress(token: str):
    """Report real stage transitions for a running generation job."""
    if not token.isalnum():
        raise HTTPException(status_code=400, detail="invalid token")

    job = _JOBS.get(token)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    percent = round(100 * job.completed / len(job.stages)) if job.stages else 0
    return ProgressResponse(
        job_id=job.token,
        stage=job.stage,
        message=job.message,
        stages=job.stages,
        completed=job.completed,
        percent=100 if job.done else percent,
        done=job.done,
        error=job.error,
        result=job.result,
    )


# -- Pipeline -----------------------------------------------------------------


async def _run_pipeline(job: "Job", payload: CreateMeshRequest) -> None:
    """Run a generation job, recording each stage as it actually completes."""
    started = time.perf_counter()
    token = job.token

    try:
        from app.geospatial.providers.elevation import (
            ElevationUnavailableError,
            SRTMTileFetcher,
        )
        from app.services.features import (
            ModelTransform,
            build_buildings,
            build_contours,
            build_roads,
        )
        from app.services.terrain import (
            MeshBuilder,
            TerrainGenerationError,
            TerrainSettings,
            export_to_stl,
        )
        from app.services.validation import validate_stl_file
        from app.utils.projection import bounds_to_meters

        bounds = payload.to_geo_bounds()
        width_m, height_m = bounds_to_meters(bounds)

        settings = TerrainSettings(
            west=bounds.west,
            south=bounds.south,
            east=bounds.east,
            north=bounds.north,
            model_width_mm=payload.model_width_mm,
            model_depth_mm=payload.model_depth_mm,
            resolution_m=payload.resolution_m,
            min_altitude_mm=payload.min_altitude_mm,
            max_altitude_mm=payload.max_altitude_mm,
            vertical_exaggeration=payload.vertical_exaggeration,
            smoothing_passes=payload.smoothing_passes,
            base_thickness_mm=payload.base_thickness_mm,
            base_extension_mm=payload.base_extension_mm,
            include_roads=payload.include_roads,
            include_buildings=payload.include_buildings,
            include_contours=payload.include_contours,
            road_width_mm=payload.road_width_mm,
            road_height_mm=payload.road_height_mm,
            building_height_mm=payload.building_height_mm,
            contour_interval_m=payload.contour_interval_m,
            contour_thickness_mm=payload.contour_thickness_mm,
            contour_height_mm=payload.contour_height_mm,
            contours_engraved=payload.contours_engraved,
        )

        # -- Stage 1: elevation -------------------------------------------
        job.enter("elevation", "Downloading SRTM elevation tiles…")
        try:
            elev_grid = await SRTMTileFetcher.fetch_elevation(
                west=bounds.west,
                south=bounds.south,
                east=bounds.east,
                north=bounds.north,
                resolution_m=payload.resolution_m,
            )
        except ElevationUnavailableError as exc:
            raise PipelineError(str(exc)) from exc
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc

        elev_grid = _finite(elev_grid)
        builder = MeshBuilder(settings)
        surface = builder.surface_of(elev_grid)
        job.complete("elevation", f"Elevation ready ({elev_grid.shape[0]}×{elev_grid.shape[1]} samples)")

        # -- Stage 2: map features ----------------------------------------
        roads = buildings = contours = None
        if "features" in job.stages:
            job.enter("features", "Loading roads and buildings from OpenStreetMap…")
            transform = ModelTransform.create(
                bounds, payload.model_width_mm, payload.model_depth_mm
            )

            if payload.include_roads:
                ways = await _fetch_roads(bounds)
                roads = build_roads(
                    ways,
                    transform,
                    surface,
                    width_mm=settings.road_width_mm,
                    height_mm=settings.road_height_mm,
                    min_width_mm=settings.road_width_mm / 2,
                    smoothing_passes=settings.smoothing_passes,
                )
                logger.info(
                    "Rasterised %d road cell(s) from %d way(s)",
                    int(roads["mask"].sum()), len(ways),
                )

            if payload.include_buildings:
                ways = await _fetch_buildings(bounds)
                buildings = build_buildings(
                    ways,
                    transform,
                    surface,
                    height_mm=settings.building_height_mm,
                    min_height_mm=settings.building_height_mm / 2,
                    footprint_scale=settings.building_footprint_scale,
                )
                logger.info(
                    "Rasterised %d building cell(s) from %d way(s)",
                    int(buildings["mask"].sum()), len(ways),
                )

            if payload.include_contours:
                contours = build_contours(
                    elev_grid,
                    width_mm=payload.model_width_mm,
                    depth_mm=payload.model_depth_mm,
                    interval_m=settings.contour_interval_m,
                    thickness_mm=settings.contour_thickness_mm,
                    height_mm=settings.contour_height_mm,
                    engraved=settings.contours_engraved,
                )
                logger.info("Rasterised %d contour cell(s)", int(contours["mask"].sum()))

            job.complete("features", "Map features loaded")

        # -- Stage 3: mesh -------------------------------------------------
        job.enter("mesh", "Building terrain surface and solid…")
        try:
            vertices, triangles, _surface, stats = builder.build_with_features(
                elev_grid, roads=roads, buildings=buildings, contours=contours
            )
        except TerrainGenerationError as exc:
            raise PipelineError(str(exc)) from exc
        job.complete("mesh", f"Mesh assembled ({len(triangles):,} triangles)")

        # -- Stage 4: export ------------------------------------------------
        job.enter("export", "Writing binary STL…")
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = DOWNLOADS_DIR / f"{token}.stl"
        try:
            export_to_stl(vertices, triangles, out_file)
        except OSError as exc:
            raise PipelineError("could not write the STL file") from exc
        job.complete("export", "STL written")

        # -- Stage 5: validate ----------------------------------------------
        job.enter("validate", "Checking the mesh is watertight…")
        report = validate_stl_file(out_file, len(triangles))
        if not report.watertight or not report.manifold:
            out_file.unlink(missing_ok=True)
            logger.error("Generated mesh failed validation: %s", report.issues)
            raise PipelineError(
                "the generated mesh did not pass validation: " + "; ".join(report.issues)
            )

        job.result = MeshInfo(
            token=token,
            vertices=report.vertices,
            triangles=report.triangles,
            width_mm=report.bbox_mm[0],
            depth_mm=report.bbox_mm[1],
            height_range_mm=report.bbox_mm[2],
            file_size_bytes=out_file.stat().st_size,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            bbox=BoundingBox(
                west=bounds.west,
                south=bounds.south,
                east=bounds.east,
                north=bounds.north,
            ),
            validation=report.as_dict(),
            elevation={
                "min_m": round(float(elev_grid.min()), 1),
                "max_m": round(float(elev_grid.max()), 1),
                "rows": int(elev_grid.shape[0]),
                "cols": int(elev_grid.shape[1]),
            },
            features=FeatureStats(**stats),
            requested_features=[
                name
                for name, wanted in (
                    ("roads", payload.include_roads),
                    ("buildings", payload.include_buildings),
                    ("contours", payload.include_contours),
                )
                if wanted
            ],
            ground_area_m=round(max(width_m, height_m), 1),
        ).model_dump()
        job.complete("validate", "Validation passed")

    except PipelineError as exc:
        job.fail(str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Generation job %s failed", token)
        job.fail(f"unexpected error: {exc}")
    finally:
        job.done = True


@app.get("/api/download/{token}", tags=["mesh"])
async def download_stl(token: str):
    """Download a previously generated binary STL file."""
    if not token.isalnum():
        raise HTTPException(status_code=400, detail="invalid download token")

    stl_file = DOWNLOADS_DIR / f"{token}.stl"
    if not stl_file.is_file():
        raise HTTPException(status_code=404, detail="model file not found or expired")

    return Response(
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="terrain-{token}.stl"',
            "Content-Length": str(stl_file.stat().st_size),
        },
        content=stl_file.read_bytes(),
    )


# -- Pipeline helpers ---------------------------------------------------------


def _finite(grid: np.ndarray) -> np.ndarray:
    """Replace elevation voids with a continuous surface."""
    from app.geospatial.providers.elevation import ElevationDataProcessor

    voids = int(np.isnan(grid).sum())
    if voids:
        logger.warning("Elevation grid contained %d void cell(s)", voids)
        grid = ElevationDataProcessor.fill_nan(grid)
    return grid


async def _fetch_roads(bounds: GeoBounds) -> list:
    from app.geospatial.providers.vector import RoadFeatureProvider

    provider = RoadFeatureProvider(user_agent=_user_agent())
    return await provider.fetch_roads(bounds.west, bounds.south, bounds.east, bounds.north)


async def _fetch_buildings(bounds: GeoBounds) -> list:
    from app.geospatial.providers.vector import BuildingFeatureProvider

    provider = BuildingFeatureProvider(user_agent=_user_agent())
    return await provider.fetch_buildings(
        bounds.west, bounds.south, bounds.east, bounds.north
    )
