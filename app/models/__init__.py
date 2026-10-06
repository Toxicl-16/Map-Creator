"""Data models for the terrain model generator."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RoadType(str, Enum):
    """OSM highway classifications to include."""
    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    RESIDENTIAL = "residential"
    UNCLASSIFIED = "unclassified"

    @property
    def min_level(self) -> int:
        """Minimum highway class (1=highest priority)."""
        return {
            self.MOTORWAY: 1,
            self.TRUNK: 2,
            self.PRIMARY: 3,
            self.SECONDARY: 4,
            self.TERTIARY: 5,
            self.RESIDENTIAL: 6,
            self.UNCLASSIFIED: 7,
        }[self]


class ElevationSourceType(str, Enum):
    """Elevation data source priority."""
    COPERNICUS_30M = "copernicus_30m"
    SRTM_GLSL = "srtm_gl1"
    NASADEM = "nasadem"
    NED_HQD = "ned_hq"


class ContourStyle(str, Enum):
    """Contour rendering style."""
    RAISED = "raised"
    ENGRAVED = "engraved"


class OutputFormat(str, Enum):
    """STL output format."""
    BINARY = "binary"
    ASCII = "ascii"


@dataclass
class GeoBounds:
    """WGS84 bounding box (lat/lon degrees)."""
    west: float
    south: float
    east: float
    north: float

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2

    @property
    def width_deg(self) -> float:
        return abs(self.east - self.west)

    @property
    def height_deg(self) -> float:
        return abs(self.north - self.south)

    def intersects(self, other: "GeoBounds") -> bool:
        return not (self.east <= other.west or self.west >= other.east or self.south >= other.north or self.north <= other.south)


@dataclass
class ModelDimensions:
    """Physical model dimensions in millimeters."""
    width_mm: float = 300.0
    depth_mm: float = 300.0
    base_thickness_mm: float = 5.0
    min_terrain_thickness_mm: float = 2.0


@dataclass
class ElevationSettings:
    """Controls for the elevation data layer."""
    vertical_exaggeration: float = 1.0
    elevation_offset_m: float = 0.0
    min_elevation_clamp_m: Optional[float] = None
    max_elevation_clamp_m: Optional[float] = None
    smoothing_passes: int = 0
    smoothing_radius_pixels: int = 1
    min_height_mm: float = 0.0
    resolution_m: float = 30.0


@dataclass
class RoadSettings:
    """Controls for road rendering."""
    enabled: bool = True
    max_level: int = 5
    min_width_mm: float = 0.5
    height_multiplier: float = 0.2
    smoothing_passes: int = 0


@dataclass
class BuildingSettings:
    """Controls for building footprint rendering."""
    enabled: bool = True
    min_footprint_area_mm2: float = 50.0
    default_height_m: float = 5.0
    height_scale: float = 1.0
    smoothing_passes: int = 0


@dataclass
class ContourSettings:
    """Controls for topographic contour lines."""
    enabled: bool = False
    interval_m: float = 25.0
    minor_interval_m: Optional[float] = None
    line_thickness_mm: float = 0.3
    contour_height_ratio: float = 0.8
    style: ContourStyle = ContourStyle.RAISED
    min_feature_size_m: float = 50.0


@dataclass
class GenerationSettings:
    """Complete user-facing configuration for one generation job."""
    bounds: GeoBounds | None = None
    model: ModelDimensions = field(default_factory=ModelDimensions)
    elevation: ElevationSettings = field(default_factory=ElevationSettings)
    roads: RoadSettings = field(default_factory=RoadSettings)
    buildings: BuildingSettings = field(default_factory=BuildingSettings)
    contours: ContourSettings = field(default_factory=ContourSettings)
    output_format: OutputFormat = OutputFormat.BINARY
    grid_width_pixels: int = 192
    grid_height_pixels: int = 192


@dataclass
class GenerationProgress:
    """Track progress through the generation pipeline."""
    stage: str = "pending"
    message: str = "Ready"
    percent: float = 0.0


@dataclass
class StlFileInfo:
    """Information about a generated STL file."""
    path: str
    size_bytes: int
    format: OutputFormat
    triangle_count: int
    bounding_box_mm: tuple[float, float, float]


VALID_ROAD_TYPES = list(RoadType)


@dataclass
class ValidationError:
    """A single mesh validation issue."""
    message: str
    severity: str  # "error", "warning", "info"


def validate_bounds(bounds: GeoBounds) -> Optional[str]:
    """Return an error string if bounds are invalid, else None."""
    if not bounds or not isinstance(bounds, GeoBounds):
        return "No geographic area selected."
    if bounds.width_deg <= 0 or bounds.height_deg <= 0:
        return "Area has zero width or height."
    if bounds.west >= bounds.east:
        return "West must be less than east."
    if bounds.south >= bounds.north:
        return "South must be less than north."
    if abs(bounds.center_lat) > 85:
        return "Latitude exceeds ±85°; terrain accuracy declines at high latitudes."
    if bounds.width_deg > 80 or bounds.height_deg > 80:
        return f"Area ({bounds.width_deg:.1f}°×{bounds.height_deg:.1f}°) is unusually large; consider a smaller region."
    return None


def validate_settings(settings: GenerationSettings) -> list[ValidationError]:
    """Validate generation settings for plausibility. Return list of issues."""
    errors: list[ValidationError] = []

    if not settings.bounds:
        errors.append(ValidationError("No geographic area defined", "error"))
    else:
        bounds_err = validate_bounds(settings.bounds)
        if bounds_err:
            errors.append(ValidationError(bounds_err, "error"))

    s = settings.elevation
    if s.vertical_exaggeration <= 0:
        errors.append(ValidationError("Vertical exaggeration must be > 0", "error"))
    if s.smoothing_passes < 0:
        errors.append(ValidationError("Smoothing passes cannot be negative", "warning"))

    m = settings.model
    if m.width_mm <= 0 or m.depth_mm <= 0:
        errors.append(ValidationError("Model dimensions must be > 0", "error"))
    if m.base_thickness_mm < 0:
        errors.append(ValidationError("Base thickness cannot be negative", "warning"))

    if settings.roads.enabled:
        if settings.roads.max_level < 0 or settings.roads.max_level > 7:
            errors.append(ValidationError(f"road max_level {settings.roads.max_level} out of range [0,7]", "warning"))
        if settings.roads.min_width_mm < 0.25:
            errors.append(ValidationError("Road min width below 0.25 mm may be unprintable", "warning"))

    return errors
