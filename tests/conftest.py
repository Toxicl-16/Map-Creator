"""Shared pytest fixtures and import path setup."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def yosemite_bounds():
    """A small real-world box: Yosemite Valley."""
    from app.models import GeoBounds

    return GeoBounds(west=-119.62, south=37.72, east=-119.59, north=37.75)


@pytest.fixture
def synthetic_elevation():
    """Smooth 60x90 synthetic terrain in metres, row 0 = southern edge."""
    rows, cols = 60, 90
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    return 2400.0 + 300.0 * np.sin(i / 9.0) * np.cos(j / 11.0)


@pytest.fixture
def builder(yosemite_bounds):
    """MeshBuilder configured for the Yosemite box."""
    from app.services.terrain import MeshBuilder, TerrainSettings

    settings = TerrainSettings(
        west=yosemite_bounds.west,
        south=yosemite_bounds.south,
        east=yosemite_bounds.east,
        north=yosemite_bounds.north,
        model_width_mm=300,
        model_depth_mm=200,
        resolution_m=30,
        max_altitude_mm=40,
        base_thickness_mm=3,
    )
    return MeshBuilder(settings)