"""Elevation provider: tile decoding, caching, and grid orientation.

These tests never touch the network; tiles are synthesised so the geometry and
row-ordering contract can be checked deterministically.
"""

import pathlib

import numpy as np
import pytest

from app.geospatial.providers import elevation as elev
from app.models import GeoBounds


def flat_tile(value: float = 500.0) -> np.ndarray:
    """A decoded tile with row 0 as the southern edge."""
    return np.full((elev.SRTM_TILE_SIZE, elev.SRTM_TILE_SIZE), value, dtype=np.float64)


def encode_tile(grid: np.ndarray) -> bytes:
    """Encode a grid into raw ``.hgt`` bytes, undoing _decode_tile's transpose."""
    return grid.T.astype(">i2").tobytes()


def test_tile_names_cover_the_box():
    assert elev.SRTMTileFetcher.get_tile_names(-119.62, 37.72, -119.59, 37.75) == ["N37W120"]


def test_tile_names_cover_every_quadrant():
    assert elev.SRTMTileFetcher.get_tile_names(10.0, 40.0, 10.5, 40.5) == ["N40E010"]
    assert elev.SRTMTileFetcher.get_tile_names(-10.0, -40.5, -9.5, -40.0) == ["S41W010"]
    assert elev.SRTMTileFetcher.get_tile_names(-0.5, 5.0, 0.5, 5.5) == ["N05W001", "N05E000"]


def test_aws_mirror_url_uses_a_two_digit_latitude_bucket():
    """Regression: the bucket was 'N' instead of 'N37', so AWS returned 404."""
    template = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}/{tile}.hgt.gz"
    url = elev.SRTMTileFetcher._mirror_url(template, "N37W120")

    assert url == (
        "https://s3.amazonaws.com/elevation-tiles-prod/skadi/N37/N37W120.hgt.gz"
    )


def test_mirror_url_leaves_plain_templates_alone():
    url = elev.SRTMTileFetcher._mirror_url(
        "https://srtm.glues.ac.uk/data/{tile}.hgt", "N37W120"
    )
    assert url == "https://srtm.glues.ac.uk/data/N37W120.hgt"


def test_decode_returns_south_to_north_rows():
    """Regression: rows stayed north→south, which flipped every model."""
    grid = flat_tile()
    grid[-1, :] = 900.0  # northern edge

    decoded = elev._decode_tile(encode_tile(grid))

    assert decoded[0, 0] == pytest.approx(500.0)
    assert decoded[-1, 0] == pytest.approx(900.0)
    assert decoded[0, 0] < decoded[-1, 0]


def test_decode_returns_west_to_east_columns():
    grid = flat_tile()
    grid[:, -1] = 1234.0  # eastern edge

    decoded = elev._decode_tile(encode_tile(grid))

    assert decoded[0, -1] == pytest.approx(1234.0)
    assert decoded[0, 0] == pytest.approx(500.0)


def test_decode_converts_voids_to_nan():
    grid = flat_tile()
    grid[0, 0] = elev.VOID_SENTINEL

    decoded = elev._decode_tile(encode_tile(grid))

    assert np.isnan(decoded[0, 0])
    assert np.isfinite(decoded[1, 1])


def test_decode_reads_big_endian_int16():
    raw = encode_tile(np.full((elev.SRTM_TILE_SIZE, elev.SRTM_TILE_SIZE), 1000.0))
    assert len(raw) == elev.SRTM_TILE_BYTES

    decoded = elev._decode_tile(raw)
    assert decoded.mean() == pytest.approx(1000.0)


def test_cache_round_trip(tmp_path, monkeypatch):
    """Regression: np.save appended .npy, so the atomic rename silently failed."""
    monkeypatch.setattr(elev, "TILE_CACHE_DIR", tmp_path / "cache")
    tile = flat_tile(4321.0)

    elev.SRTMTileFetcher.store_tile("N37W120", tile)

    assert (tmp_path / "cache" / "N37W120.npy").exists()
    # No temp files left behind.
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["N37W120.npy"]

    loaded = elev.SRTMTileFetcher.load_cached_tile("N37W120")
    assert loaded is not None
    assert loaded.shape == tile.shape
    assert np.allclose(loaded, tile)


def test_missing_cache_entry_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(elev, "TILE_CACHE_DIR", tmp_path / "cache")
    assert elev.SRTMTileFetcher.load_cached_tile("N99W999") is None


def test_corrupt_cache_entry_is_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr(elev, "TILE_CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "N37W120.npy").write_bytes(b"not a numpy file")

    assert elev.SRTMTileFetcher.load_cached_tile("N37W120") is None
    assert not (tmp_path / "cache" / "N37W120.npy").exists()


def test_grid_row_zero_is_the_southern_edge():
    """The resampler must produce south→north rows to match the mesh."""
    bounds = GeoBounds(west=10.0, south=40.0, east=10.5, north=41.0)
    tile = flat_tile(100.0)
    tile[-1, :] = 900.0  # northern edge of the tile

    grid = elev.ElevationDataProcessor()._resample({"N40E010": tile}, bounds, 9, 5)

    assert grid[0, 0] == pytest.approx(100.0, abs=1.0)
    assert grid[-1, 0] == pytest.approx(900.0, abs=1.0)
    assert grid[0, 0] < grid[-1, 0]


def test_bounds_aligned_to_whole_degrees_leave_no_gaps():
    """Regression: floor() put the edge sample in a tile that was never fetched."""
    bounds = GeoBounds(west=10.0, south=40.0, east=11.0, north=41.0)

    # A tile whose value tracks row index, so the sampled fraction is visible.
    ramp = np.tile(
        np.arange(elev.SRTM_TILE_SIZE, dtype=np.float64)[:, None],
        (1, elev.SRTM_TILE_SIZE),
    )
    tiles = {"N40E010": ramp}

    grid = elev.ElevationDataProcessor()._resample(tiles, bounds, 11, 11)

    assert np.isfinite(grid).all()
    # south edge is the tile's row 0, north edge its last row
    assert grid[0, 0] == pytest.approx(0.0, abs=2.0)
    assert grid[-1, 0] == pytest.approx(3600.0, abs=2.0)
    assert grid[0, 0] < grid[-1, 0]


def test_resample_returns_the_requested_shape():
    bounds = GeoBounds(west=-119.62, south=37.72, east=-119.59, north=37.75)
    grid = elev.ElevationDataProcessor()._resample(
        {"N37W120": flat_tile(300.0)}, bounds, 17, 23
    )
    assert grid.shape == (17, 23)


def test_to_grid_derives_shape_from_resolution(monkeypatch):
    from app.utils.projection import bounds_to_meters

    bounds = GeoBounds(west=-119.62, south=37.72, east=-119.59, north=37.75)
    width_m, height_m = bounds_to_meters(bounds)

    async def fake_fetch(self, tile_names):
        return {"N37W120": flat_tile(300.0)}

    monkeypatch.setattr(
        elev.ElevationDataProcessor, "_fetch_tiles", fake_fetch, raising=True
    )
    grid = pytest.importorskip("asyncio").run(
        elev.ElevationDataProcessor().to_grid(bounds, resolution_m=30.0)
    )

    assert grid.shape == (
        max(2, int(round(height_m / 30.0)) + 1),
        max(2, int(round(width_m / 30.0)) + 1),
    )


def test_fill_nan_produces_a_continuous_surface():
    grid = np.full((5, 5), np.nan)
    grid[2, 2] = 100.0

    filled = elev.ElevationDataProcessor.fill_nan(grid)

    assert np.isfinite(filled).all()
    assert filled[2, 2] == 100.0


def test_fill_nan_leaves_complete_grids_untouched():
    grid = flat_tile(250.0)
    assert np.array_equal(elev.ElevationDataProcessor.fill_nan(grid), grid)


def test_fetch_elevation_is_the_api_entry_point():
    """The route used to call a method that did not exist (HTTP 500)."""
    assert hasattr(elev.SRTMTileFetcher, "fetch_elevation")

    tile = flat_tile(500.0)

    async def fake_fetch(cls, tile_name, **kwargs):
        return tile

    original = elev.SRTMTileFetcher.fetch_tile
    elev.SRTMTileFetcher.fetch_tile = classmethod(fake_fetch)
    try:
        grid = pytest.importorskip("asyncio").run(
            elev.SRTMTileFetcher.fetch_elevation(10.0, 40.0, 10.5, 40.5, resolution_m=60.0)
        )
    finally:
        elev.SRTMTileFetcher.fetch_tile = original

    assert grid.ndim == 2
    assert np.isfinite(grid).all()
    assert grid.mean() == pytest.approx(500.0)


def test_fetch_elevation_raises_when_no_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(elev, "TILE_CACHE_DIR", tmp_path / "cache")

    async def missing(cls, tile_name, **kwargs):
        return None

    original = elev.SRTMTileFetcher.fetch_tile
    elev.SRTMTileFetcher.fetch_tile = classmethod(missing)
    try:
        with pytest.raises((elev.ElevationUnavailableError, ValueError)):
            pytest.importorskip("asyncio").run(
                elev.SRTMTileFetcher.fetch_elevation(10.0, 40.0, 10.5, 40.5)
            )
    finally:
        elev.SRTMTileFetcher.fetch_tile = original


def test_cache_dir_lives_under_data():
    assert isinstance(elev.TILE_CACHE_DIR, pathlib.Path)
    assert str(elev.TILE_CACHE_DIR).startswith("data")