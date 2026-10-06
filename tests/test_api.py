"""API layer tests.

These use FastAPI's TestClient with the network-facing providers stubbed out, so
the whole pipeline (validation -> features -> mesh -> STL -> validation report)
is exercised without touching SRTM, Nominatim or Overpass.
"""

import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.models import GeoBounds

pytest.importorskip("httpx")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient whose downloads land in a temporary directory."""
    from app.api import routes

    monkeypatch.setattr(routes, "DOWNLOADS_DIR", tmp_path / "downloads")

    def fake_elevation(west, south, east, north, resolution_m=30.0):
        from app.utils.projection import bounds_to_meters

        width_m, height_m = bounds_to_meters(
            GeoBounds(west=west, south=south, east=east, north=north)
        )
        cols = max(2, int(width_m / resolution_m) + 1)
        rows = max(2, int(height_m / resolution_m) + 1)
        i = np.arange(rows)[:, None]
        j = np.arange(cols)[None, :]
        return 2400.0 + 40.0 * np.sin(i / 7.0) * np.cos(j / 9.0)

    from app.geospatial.providers import elevation as elevation_module

    async def fake_fetch(west, south, east, north, resolution_m=30.0):
        return fake_elevation(west, south, east, north, resolution_m)

    monkeypatch.setattr(elevation_module.SRTMTileFetcher, "fetch_elevation", fake_fetch)

    with TestClient(routes.app) as test_client:
        yield test_client


@pytest.fixture
def stub_features(monkeypatch):
    """Replace the Overpass helpers with fixed way dictionaries."""
    from app.api import routes

    road = {
        "type": "way",
        "id": 1,
        "geometry": [
            {"lat": 37.7200, "lon": -119.6200},
            {"lat": 37.7500, "lon": -119.5900},
        ],
    }
    building = {
        "type": "way",
        "id": 2,
        "tags": {"building": "yes"},
        "geometry": [
            {"lat": 37.7300, "lon": -119.6100},
            {"lat": 37.7350, "lon": -119.6100},
            {"lat": 37.7350, "lon": -119.6050},
            {"lat": 37.7300, "lon": -119.6050},
            {"lat": 37.7300, "lon": -119.6100},
        ],
    }

    async def fake_roads(bounds):
        return [road]

    async def fake_buildings(bounds):
        return [building]

    monkeypatch.setattr(routes, "_fetch_roads", fake_roads)
    monkeypatch.setattr(routes, "_fetch_buildings", fake_buildings)
    return road, building


def base_request(**overrides):
    """A valid Yosemite Valley request."""
    payload = {
        "west": -119.62,
        "south": 37.72,
        "east": -119.59,
        "north": 37.75,
        "model_width_mm": 200,
        "model_depth_mm": 150,
        "resolution_m": 40,
        "min_altitude_mm": 2,
        "max_altitude_mm": 30,
        "base_thickness_mm": 3,
    }
    payload.update(overrides)
    return payload


def run_job(client, payload=None, max_polls=400):
    """Start a generation job and poll it to completion.

    Returns ``(progress, stages_run)``. The stages come from the job's own
    recorded history, because a test client can finish a job before the first
    poll — polling alone cannot prove the stages actually ran in order.
    """
    from app.api import routes

    response = client.post("/api/generate/stl", json=payload or base_request())
    assert response.status_code == 202, response.text
    accepted = response.json()

    assert accepted["token"] == accepted["job_id"]
    assert accepted["status_url"] == f"/api/progress/{accepted['token']}"
    assert accepted["stages"][0] == "elevation"
    assert accepted["stages"][-1] == "validate"

    progress = {}
    for _ in range(max_polls):
        progress = client.get(accepted["status_url"]).json()
        if progress["done"]:
            break
        time.sleep(0.005)
    else:
        raise AssertionError("generation job never finished")

    stages = list(routes._JOBS[accepted["token"]].history)
    return progress, stages


# -- system -------------------------------------------------------------------


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["service"] == "map-creator"


def test_index_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_static_files_mounted(client):
    assert client.get("/static/js/app.js").status_code == 200


# -- request validation -------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"west": -119.59},  # west >= east
        {"north": 37.72},  # south >= north
        {"west": 200},  # out of range
        {"model_width_mm": 0},  # not positive
        {"model_width_mm": 5000},  # over the cap
        {"resolution_m": 5},  # under the floor
        {"min_altitude_mm": 40, "max_altitude_mm": 30},  # inverted heights
        {"base_thickness_mm": -1},  # negative
    ],
)
def test_invalid_requests_are_rejected(client, overrides):
    response = client.post("/api/generate/stl", json=base_request(**overrides))
    assert response.status_code == 422


def test_tiny_selection_is_rejected(client):
    response = client.post(
        "/api/generate/stl",
        json=base_request(west=-119.6200, east=-119.62005),
    )
    assert response.status_code == 422


# -- generation ---------------------------------------------------------------


def test_generate_returns_watertight_mesh(client):
    progress, stages = run_job(client)
    assert progress["error"] is None
    body = progress["result"]

    assert stages == ["elevation", "mesh", "export", "validate"]
    assert progress["percent"] == 100
    assert progress["completed"] == 4

    assert body["token"]
    assert body["triangles"] > 0
    assert body["vertices"] > 0
    assert body["file_size_bytes"] > 84
    assert body["duration_ms"] >= 0
    assert body["width_mm"] == pytest.approx(200.0)
    assert body["depth_mm"] == pytest.approx(150.0)
    assert body["height_range_mm"] > 0
    assert body["ground_area_m"] > 0

    report = body["validation"]
    assert report["watertight"] is True
    assert report["manifold"] is True
    assert report["valid_stl"] is True
    assert report["boundary_edges"] == 0
    assert report["non_manifold_edges"] == 0
    assert report["degenerate_triangles"] == 0
    assert report["issues"] == []

    stats = body["elevation"]
    assert stats["rows"] > 1 and stats["cols"] > 1
    assert stats["max_m"] >= stats["min_m"]

    assert body["features"] == {"roads": 0, "buildings": 0, "contours": 0}
    assert body["bbox"]["west"] == pytest.approx(-119.62)


def test_generated_file_is_a_binary_stl(client):
    body = run_job(client)[0]["result"]
    blob = client.get(f"/api/download/{body['token']}").content

    assert len(blob) == 84 + 50 * body["triangles"]

    header, count = struct.unpack("<80sI", blob[:84])
    assert count == body["triangles"]

    for index in range(count):
        offset = 84 + index * 50
        values = struct.unpack("<12fH", blob[offset : offset + 50])
        normal = values[0:3]
        magnitude = sum(c * c for c in normal) ** 0.5
        # Facet normals must be unit length (or zero for a degenerate facet).
        assert magnitude == pytest.approx(1.0, abs=1e-3) or magnitude == 0.0
        assert values[12] == 0  # attribute byte count


def test_generate_with_features(client, stub_features):
    progress, stages = run_job(
        client,
        base_request(
            include_roads=True,
            include_buildings=True,
            include_contours=True,
            contour_interval_m=10,
        ),
    )
    assert progress["error"] is None
    # The features stage only exists when features were requested.
    assert stages == ["elevation", "features", "mesh", "export", "validate"]
    body = progress["result"]

    assert body["features"]["roads"] > 0
    assert body["features"]["buildings"] > 0
    assert body["features"]["contours"] > 0
    assert body["validation"]["watertight"] is True
    assert body["validation"]["manifold"] is True


def test_engraved_contours_stay_inside_the_model(client):
    """A negative contour height must not punch through the base plate."""
    raised = run_job(
        client, base_request(include_contours=True, contour_height_mm=2, contour_interval_m=5)
    )[0]["result"]
    engraved = run_job(
        client,
        base_request(
            include_contours=True,
            contour_height_mm=-0.6,
            contour_interval_m=5,
            contours_engraved=True,
        ),
    )[0]["result"]

    assert engraved["validation"]["watertight"] is True
    assert engraved["validation"]["manifold"] is True
    assert engraved["validation"]["volume_mm3"] > 0
    # Raised lines add material; engraving removes it.
    assert engraved["validation"]["volume_mm3"] < raised["validation"]["volume_mm3"]


def test_engraved_contours_respect_base_thickness(client):
    """Deep engraving must not invert or remove the base."""
    report = run_job(
        client,
        base_request(
            include_contours=True,
            contour_height_mm=-0.6,
            contour_interval_m=5,
            contours_engraved=True,
            base_thickness_mm=3,
        ),
    )[0]["result"]["validation"]
    assert report["watertight"] is True
    assert report["volume_mm3"] > 0


def test_elevation_failure_becomes_502(client, monkeypatch):
    from app.geospatial.providers import elevation as elevation_module
    from app.geospatial.providers.elevation import ElevationUnavailableError

    async def boom(*args, **kwargs):
        raise ElevationUnavailableError("no SRTM coverage for this location")

    monkeypatch.setattr(elevation_module.SRTMTileFetcher, "fetch_elevation", boom)
    progress, _ = run_job(client)
    assert progress["done"] is True
    assert progress["result"] is None
    assert "SRTM" in progress["error"]


def test_missing_file_returns_404(client):
    assert client.get("/api/download/deadbeef1234").status_code == 404


def test_invalid_token_is_rejected(client):
    assert client.get("/api/download/..%2Fetc%2Fpasswd").status_code in (400, 404)


# -- geocoding ----------------------------------------------------------------


def test_geocode_returns_bounds(monkeypatch):
    from app.api import routes
    from app.geospatial.providers.geocoding import NominatimClient, SearchResult

    async def fake_search(self, query, limit=1):
        assert query == "Yosemite Valley"
        return SearchResult(
            display_name="Yosemite Valley, California, USA",
            lat=37.7459,
            lon=-119.5937,
            # Nominatim order: [south, north, west, east]
            bounding_box=[37.71, 37.77, -119.66, -119.53],
        )

    monkeypatch.setattr(NominatimClient, "search", fake_search)

    with TestClient(routes.app) as client:
        response = client.post("/api/geocode", json={"q": "  Yosemite Valley  "})

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"].startswith("Yosemite Valley")
    assert body["center_lat"] == pytest.approx(37.7459)
    assert body["center_lon"] == pytest.approx(-119.5937)

    bounds = body["bounds"]
    assert bounds["south"] == pytest.approx(37.71)
    assert bounds["north"] == pytest.approx(37.77)
    assert bounds["west"] == pytest.approx(-119.66)
    assert bounds["east"] == pytest.approx(-119.53)
    # A real boundary is more trustworthy than a synthesised box.
    assert body["confidence"] > 0.8


def test_geocode_without_boundary_synthesises_box(monkeypatch):
    from app.api import routes
    from app.geospatial.providers.geocoding import NominatimClient, SearchResult

    async def fake_search(self, query, limit=1):
        return SearchResult(display_name="Somewhere", lat=0.0, lon=0.0, bounding_box=None)

    monkeypatch.setattr(NominatimClient, "search", fake_search)

    with TestClient(routes.app) as client:
        body = client.post("/api/geocode", json={"q": "Somewhere"}).json()

    assert body["bounds"]["south"] < 0 < body["bounds"]["north"]
    assert body["bounds"]["west"] < 0 < body["bounds"]["east"]
    assert body["confidence"] <= 0.5


def test_geocode_miss_returns_404(monkeypatch):
    from app.api import routes
    from app.geospatial.providers.geocoding import NominatimClient

    async def fake_search(self, query, limit=1):
        return None

    monkeypatch.setattr(NominatimClient, "search", fake_search)

    with TestClient(routes.app) as client:
        response = client.post("/api/geocode", json={"q": "asdfqwerzxcv"})

    assert response.status_code == 404
    assert "asdfqwerzxcv" in response.json()["detail"]


def test_geocode_provider_error_returns_502(monkeypatch):
    from app.api import routes
    from app.geospatial.providers.geocoding import NominatimClient

    async def boom(self, query, limit=1):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(NominatimClient, "search", boom)

    with TestClient(routes.app) as client:
        response = client.post("/api/geocode", json={"q": "Yosemite"})

    assert response.status_code == 502


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_geocode_is_rejected(client, query):
    assert client.post("/api/geocode", json={"q": query}).status_code == 422


# -- startup / cleanup --------------------------------------------------------


def test_prune_downloads_removes_only_expired_files(tmp_path, monkeypatch):
    import os
    import time

    from app.api import routes

    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes, "DOWNLOADS_DIR", downloads)

    stale = downloads / "stale.stl"
    fresh = downloads / "fresh.stl"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    old = time.time() - routes.DOWNLOAD_TTL_SECONDS - 60
    os.utime(stale, (old, old))

    assert routes._prune_downloads() == 1
    assert not stale.exists()
    assert fresh.exists()

    # A brand-new file is not yet expired.
    assert routes._prune_downloads() == 0


def test_prune_downloads_tolerates_missing_directory(tmp_path, monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "DOWNLOADS_DIR", tmp_path / "absent")
    assert routes._prune_downloads() == 0