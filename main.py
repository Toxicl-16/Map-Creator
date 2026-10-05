"""Entry point for Map-Creator — terrain model generator.

Runs the FastAPI application via uvicorn.

Usage:
    python main.py                       # development with hot-reload
    uvicorn main:app --host 0.0.0.0      # production

Required environment variables (see .env.example):
    NOMINATIM_USER_AGENT   - Required by Nominatim public API ("AppName (email)")
    OPEN_TOPOGRAPHY_API_KEY - Optional: for OpenTopography elevation data override
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager, closing
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Map-Creator — terrain model generator")
    yield
    logger.info("Shutting down Map-Creator")


app = FastAPI(
    title="Map-Creator",
    description="Interactive terrain model generator for 3D printing.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the frontend UI."""
    return FileResponse("frontend/static/index.html")


@app.get("/api/health")
async def health_check():
    """Health check — used by monitors and the frontend."""
    import platform

    plat = platform.platform()
    return {
        "ok": True,
        "service": "map-creator",
        "platform": plat,
        "python": platform.python_version(),
    }


@app.post("/api/geocode")
async def geocode(q: Optional[str] = None, search: Optional[str] = Query(None)):
    """Geocode a location name to lat/lon using OpenStreetMap Nominatim."""

    if not (q or search):
        raise HTTPException(400, "Provide 'q' query parameter for the place name.")
    term = q or search

    url = f"https://nominatim.openstreetmap.org/search?format=json&q={term.encode('utf-8').hex()}&addressdetails=1"
    headers = {"User-Agent": os.environ.get("NOMINATIM_USER_AGENT", "Map-Creator-dev")}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status in (403, 429):
                    raise HTTPException(429, "Nominatim rate limit reached. Try again shortly.")
                data = await resp.json()
        except aiohttp.ClientError as e:
            logger.warning("Nominatim request failed: %s", str(e))
            data = []

    if not data or not isinstance(data, list) or len(data) == 0:
        raise HTTPException(404, f"No results for '{term}'")

    feat = data[0]  # top result
    try:
        lat = float(feat["lat"])
        lon = float(feat["lon"])
    except (KeyError, ValueError) as e:
        raise HTTPException(502, "Nominatim returned an invalid result.") from e

    bbox = None
    raw_bbox = feat.get("boundingbox")
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        try:
            min_lon, max_lon = float(raw_bbox[0]), float(raw_bbox[1])
            min_lat, max_lat = float(raw_bbox[2]), float(raw_bbox[3])
            bbox = {"west": min_lon, "south": min_lat, "east": max_lon, "north": max_lat}
        except ValueError:
            pass

    result = {
        "lat": lat,
        "lon": lon,
        "bbox": bbox,
        "place_rank": feat.get("place_rank"),
        "importance": feat.get("importance"),
    }

    return result


@app.post("/api/settings")
async def get_settings(
    west: float = Query(-122.5, ge=-180, le=180),
    north: float = Query(37.9, ge=-90, le=90),
    east: float = Query(-122.4, ge=-180, le=180),
    south: float = Query(37.7, ge=-90, le=90),
):
    """Compute and return derived settings for the selected area."""

    width_m = (east - west) * 111_320.49536891712408 * abs(math.cos(math.radians((north + south) / 2)))
    height_m = (north - south) * 111_320.49536891712408

    max_tiles = max(2, math.ceil(max(width_m, height_m) / 1))

    return {
        "width_m": round(width_m, 2),
        "height_m": round(height_m, 2),
        "center_lat": round((north + south) / 2, 6),
        "center_lon": round((west + east) / 2, 6),
        "min_zoom": max(10, int(math.ceil(math.log2(max(width_m, height_m) / 50)))),
        "max_tiles_needed": max_tiles,
    }


@app.post("/api/generate/stl")
async def generate_stl(
    west: float = Query(-122.4773, ge=-180, le=180),
    south: float = Query(37.7562, ge=-90, le=90),
    east: float = Query(-122.4453, ge=-180, le=180),
    north: float = Query(37.7896, ge=-90, le=90),
    model_width_mm: float = Query(300.0, gt=0, le=600),
    model_depth_mm: float = Query(300.0, gt=0, le=600),
    resolution_m: float = Query(50.0, gt=1, le=128),
    min_altitude_mm: float = Query(1.0, ge=0, le=99),
    max_altitude_mm: float = Query(50.0, gt=0, lt=100),
    vertical_exaggeration: float = Query(30.0, gt=0, le=200),
    base_thickness_mm: float = Query(3.0, ge=0, le=50),
):
    """Synchronous generation of an STL file."""

    # 1. Compute grid dimensions (rows x cols)
    rows_raw = int(round((north - south) * 111_320 / resolution_m)) + 2
    cols_raw = int(round((east - west) * 111_320 / resolution_m)) + 2

    # Clamp to prevent huge meshes
    if rows_raw > 256 or cols_raw > 256:
        logger.warning("Grid clamped: %.0f m → %.0f m", resolution_m, max((north - south) * 111_320, (east - west) * 111_320))

    cols = min(cols_raw, 256)
    rows = min(rows_raw, 256)
    actual_res_m = max(
        (north - south) * 111_320.4953689 / (rows - 1),
        (east - west) * 111_320.4953689 / (cols - 1),
    )

    # 2. Load elevation grid lazily; fall back to numpy random + smooth noise
    elev_data = await _load_elevation(west, south, east, north, cols, rows)
    if elev_data is None:
        height_m = (north - south) * 111_320.495
        width_m = (east - west) * 111_320.495
        elev_data = _fallback_elevation(cols, rows, height_m, width_m)

    # 3. Build mesh and export STL via MeshBuilder
    mesh_builder = MeshBuilder(
        west=west, south=south, east=east, north=north,
        model_width_mm=model_width_mm, model_depth_mm=model_depth_mm,
    )

    result = await mesh_builder.build(elev_data)
    if not result or "error" in result:
        msg = result.get("error", "MeshBuilder returned no usable data.")
        raise HTTPException(502, msg)

    out_dir = Path(".")
    out_path_str = f"{out_dir}/{result['stl_file']}"
    out_path = Path(out_path_str)

    if not out_path.exists():
        # Write from raw mesh builder vertices / triangles
        await _write_stl_directly(result["vertices"], result["triangles"], str(out_path))

    file_size = int(out_path.stat().st_size) if out_path.exists() else 0

    return {
        "filename": result.get("stl_file", "terrain.stl"),
        "width_m": model_width_mm,
        "height_m": model_depth_mm,
        "vertices": result.get("vertex_count", 0),
        "triangles": result.get("triangle_count", 0),
        "file_size_bytes": file_size,
    }


# ── Streaming STL (download) endpoints ───────────────────────

@app.get("/api/download/{filename}")
async def download_stl(filename: str):
    """Stream a generated STL file to the client."""
    out_dir = Path(".")
    out_path = out_dir / filename

    if not out_path.exists():
        raise HTTPException(404, "File not found.")

    return FileResponse(str(out_path), media_type="application/sla", filename=filename)


# ── Internal helpers (STL write, elevation loading) ──────────

async def _write_stl_directly(vertices: list[dict], triangles: list[tuple[int, int, int]], path: str):
    import numpy as np

    # Prepare arrays
    N = len(triangles)
    v0_arr = np.zeros((N, 3), dtype=np.float32)
    v1_arr = np.zeros((N, 3), dtype=np.float32)
    v2_arr = np.zeros((N, 3), dtype=np.float32)

    # Convert each triangle vertex to [x y z] in millimeters
    for i, tri in enumerate(triangles):
        v0 = np.array([v[i]["x"], v[i]["y"], v[i]["z"]], dtype=np.float32) if isinstance(v[i], dict) else np.asarray(vertices[v[0]])
        v1 = np.array([v[i]["x"], v[i]["y"], v[i]["z"]], dtype=np.float32) if isinstance(v[i], dict) else np.asarray(vertices[v[1]])
        v2 = np.array([v[i]["x"], v[i]["y"], v[i]["z"]], dtype=np.float32) if isinstance(v[i], dict) else np.asarray(vertices[v[2]])

        v0_arr[i] = v0
        v1_arr[i] = v1
        v2_arr[i] = v2

    # Compute face normals for each triangle (non-unit)
    # n = 1/2 * cross(v1 - v0, v2 - v0)
    normal_arr = np.cross(v1_arr - v0_arr, v2_arr - v0_arr) / 2.0

    import array
    import struct as st

    header_bytes = b"""Solid MapCreator STLMesh by terrain_service""" + (b"\x00" * 76)
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(st.pack("<I", N))

        for i in range(N):
            nx, ny, nz = float(normal_arr[i][0]), float(normal_arr[i][1]), float(normal_arr[i][2])
            # STL is little-endian IEEE754 (32-bit floats)
            f.write(st.pack("<ffff", nx, ny, nz, 0.0))

            for vert in [v0_arr[i], v1_arr[i], v2_arr[i]]:
                coords = map(float, vert)
                f.write(st.pack("<fff", *coords))


async def _load_elevation(west: float, south: float, east: float, north: float, cols: int, rows: int):
    """Load or download elevation raster data for the given bounding box."""

    # Try from local cache if available (future)
    try:
        cached = await _load_cached_elevation(
            west=west, south=south, east=east, north=north,
            cols=cols, rows=rows,
        )
        if cached is not None:
            return cached
    except FileNotFoundError:
        pass

    # Return nothing — let MeshBuilder handle fallback or noise generation
    return None


# ── Fallback elevation (if no real data) ─────────────────────

def _fallback_elevation(cols: int, rows: int, width_m: float, height_m: float):
    import numpy as np

    dx = width_m / cols
    dy = height_m / rows

    y_grid, x_grid = np.mgrid[:rows, :cols]
    x = x_grid * dx
    y = y_grid * dy

    # Multi-scale perlin-like (no external deps needed)
    base_hills = 0.3 * np.exp(-((x - width_m / 4) ** 2 + (y - height_m / 4) ** 2) / (width_m**2))
    noise = 0.1 * np.random.randn(rows, cols)
    elev_map = base_hills * 250 + noise * 30

    return {
        "data": elev_map,
        "dx": dx,
        "dy": dy,
        "width_m": width_m,
        "height_m": height_m,
    }


# ── Entry point / CLI startup ────────────────────────────────

async def main():
    """Start the server synchronously (for uv run or direct call)."""
    import sys
    import argparse as ag
    from typing import cast

    parser = ag.ArgumentParser(description="Map-Creator Terrain Model Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Launch via uvicorn's AsyncWorker (uvicorn must be installed — see requirements.txt)
    import asyncio
    import logging as lib_log

    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/static",
        cast("StaticFiles", StaticFiles(static_dir="frontend/static")),
        "frontend/static",
    )

    await _serve(args.host, args.port)


async def _serve(host: str, port: int):
    import logging as lib_log

    # uvicorn's run() is async — wrap in a task
    import asyncio
    import sys
    from multiprocessing import Process, Event

    proc = Process(
        target=_uvicorn_process,
        args=(host, port, True),
    )
    proc.start()

    def _wait_or_stop():
        try:
            while proc.is_alive():
                asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()

    task = asyncio.create_task(_wait_or_stop())
    try:
        await task  # runs forever until Ctrl+C or exception
    except Exception:
        proc.terminate()
        raise


def _uvicorn_process(host: str, port: int, debug: bool):
    """UVICORN ENTRY (runs inside a Process)."""
    import logging as lib_log

    lib_log.basicConfig(level=lib_log.DEBUG if debug else lib_log.INFO)
    import uvicorn  # type: ignore[import]

    cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        reload=True,
    )
    server = uvicorn.Server(cfg)
    server.run()


if __name__ == "__main__":
    asyncio.run(main())
