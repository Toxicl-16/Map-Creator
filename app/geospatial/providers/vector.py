"""Road and building feature providers using OpenStreetMap / Overpass API.

Overpass API retrieves OpenStreetMap road/network and building geometry.
Licensed under ODbL 1.0, with attribution required in the application.

Rate limits (per https://medium.com/the-real-data-center-understanding-overpass-api-rate-limiting-7648a5f4e8c9):
- Max download: 10MB uncompressed per request
- Recommended interval: ~3 seconds between queries
- Concurrent connections: max 2

Always set a User-Agent header as required by the API policy.

Data sources and licensing: https://wiki.openstreetmap.org/wiki/License/Original
Attribution requirement: "Contains data from OpenStreetMap contributors, ODbL 1.0."
"""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Number of full passes over the endpoint list before giving up. Public mirrors
# frequently answer 429/500/503/504 for a moment and then recover, so a single
# failed pass must not silently drop every road from the model.
OVERPASS_PASSES = 3


class OverpassUnavailableError(RuntimeError):
    """Raised when every Overpass endpoint failed for a query."""


# Highway classes rendered as roads. Ordered most major to most minor.
DEFAULT_HIGHWAY_TYPES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "living_street",
    "service",
]


def build_road_query(
    west: float,
    south: float,
    east: float,
    north: float,
    highway_types: Optional[List[str]] = None,
) -> str:
    """Build the Overpass QL that fetches road centrelines for a bounding box.

    A way carries exactly one ``highway=*`` value, so consecutive tag filters
    would be AND-ed together and match nothing. One alternation filter matches
    every requested class in a single statement.
    """
    types = [
        t
        for t in (highway_types or DEFAULT_HIGHWAY_TYPES)
        if t and t.replace("_", "").isalnum()
    ]
    if not types:
        types = [t for t in DEFAULT_HIGHWAY_TYPES if t.replace("_", "").isalnum()]

    # Only word-like class names can appear in the QL, so nothing is escaped.
    pattern = "^(" + "|".join(types) + ")$"
    bbox = f"({south},{west},{north},{east})"

    return (
        "[out:json][timeout:60];\n"
        f'way["highway"~"{pattern}"]{bbox};\n'
        "out geom;"
    )


class OverpassClient:
    """Lightweight HTTP client for the Overpass API."""

    def __init__(self, user_agent: Optional[str] = None):
        self._user_agent = user_agent or "Map-Creator"

    async def query(self, ql_string: str, timeout: float = 60.0) -> Optional[Dict]:
        """Execute an Overpass QL query and return the JSON response.

        Retries the whole endpoint list a few times with a short backoff before
        returning ``None``; transient mirror hiccups are common.
        """
        for attempt in range(OVERPASS_PASSES):
            for endpoint in OVERPASS_ENDPOINTS:
                try:
                    async with aiohttp.ClientSession() as session:
                        resp = await asyncio.wait_for(
                            session.post(
                                endpoint,
                                data={"data": ql_string},
                                headers={"User-Agent": self._user_agent},
                                timeout=aiohttp.ClientTimeout(total=timeout + 5),
                            ),
                            timeout=timeout + 10,
                        )

                        if resp.status == 200:
                            data = await resp.json()
                            return data
                        elif resp.status in (429, 500, 502, 503, 504):
                            logger.warning(
                                "Overpass endpoint %s returned %d, trying next",
                                endpoint,
                                resp.status,
                            )
                            await asyncio.sleep(1)
                        else:
                            # Truncate the body ourselves: ClientResponse.text()
                            # takes no size argument.
                            body = (await resp.text())[:2000]
                            logger.warning(
                                "Overpass returned status %d: %s", resp.status, body
                            )
                except aiohttp.ClientError:
                    logger.warning("Failed to query Overpass API at %s", endpoint)
                except asyncio.TimeoutError:
                    logger.warning("Overpass query timed out at %s", endpoint)
                except Exception:
                    logger.exception("Unexpected error querying Overpass at %s", endpoint)
            if attempt < OVERPASS_PASSES - 1:
                await asyncio.sleep(2 * (attempt + 1))
        logger.error("All Overpass endpoints failed after %d passes", OVERPASS_PASSES)
        return None


class RoadFeatureProvider:
    """Fetch road network data from the Overpass API.

    Supported highway types (default query): motorway, trunk, primary,
    secondary, tertiary, residential, unclassified, living_street, service.
    """

    def __init__(self, user_agent: Optional[str] = None):
        self._client = OverpassClient(user_agent)

    async def fetch_roads(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        highway_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fetch road geometries for the bounding box."""
        if not all(-180 <= c <= 180 for c in [west, east]):
            raise ValueError(
                f"Invalid longitude bounds: {west}, {east}"
            )
        if not all(-90 <= c <= 90 for c in [south, north]):
            raise ValueError(
                f"Invalid latitude bounds: {south}, {north}"
            )

        query_str = build_road_query(west, south, east, north, highway_types)

        data = await self._client.query(query_str)
        if data is None:
            raise OverpassUnavailableError(
                "OpenStreetMap (Overpass) is not responding right now"
            )

        elements = data.get("elements", [])
        roads = [elem for elem in elements if elem.get("type") == "way"]
        return roads


class BuildingFeatureProvider:
    """Fetch building footprint data from the Overpass API.

    Buildings are represented as outlines (polygon boundaries) not extruded polygons.

    Attribution requirement: "Contains data from OpenStreetMap contributors, ODbL 1.0."
    """

    def __init__(self, user_agent: Optional[str] = None):
        self._client = OverpassClient(user_agent)

    async def fetch_buildings(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> List[Dict]:
        """Get building footprints for the specified bounding box."""
        if not all(-180 <= c <= 180 for c in [west, east]):
            raise ValueError(
                f"Invalid longitude bounds: {west}, {east}"
            )
        if not all(-90 <= c <= 90 for c in [south, north]):
            raise ValueError(
                f"Invalid latitude bounds: {south}, {north}"
            )

        query_str = (
            '[out:json][timeout:60];\n'
            "( way[\"building\"]({bbox}); );\n"
            "out geom;"
        ).format(bbox=f"{south},{west},{north},{east}")

        data = await self._client.query(query_str)
        if data is None:
            raise OverpassUnavailableError(
                "OpenStreetMap (Overpass) is not responding right now"
            )

        elements = data.get("elements", [])
        # Keep every way that carries any building tag: the query already asks
        # for building=* too, so filtering to building="yes" alone discarded
        # houses, apartments, garages and the rest.
        buildings = [
            elem
            for elem in elements
            if elem.get("type") == "way" and elem.get("tags", {}).get("building")
        ]
        return buildings


async def get_feature_data(
    west: float,
    south: float,
    east: float,
    north: float,
    road_highways: Optional[List[str]] = None,
    include_buildings: bool = True,
) -> Dict[str, List[Dict]]:
    """Fetch both roads and buildings for a single bounding box.

    Returns dict with keys 'roads' (list[dict]) and 'buildings' (list[dict]).
    Empty lists are returned if API fails or data unavailable.
    """
    if road_highways is None:
        road_highways = list(DEFAULT_HIGHWAY_TYPES)

    roads: List[Dict] = []
    try:
        roads = await RoadFeatureProvider().fetch_roads(
            west, south, east, north, road_highways
        )
    except OverpassUnavailableError:
        logger.debug("Road feature fetch failed for area")
    buildings: List[Dict] = []

    if include_buildings:
        try:
            buildings = await BuildingFeatureProvider().fetch_buildings(
                west, south, east, north
            )
        except (ValueError, KeyError, OverpassUnavailableError):
            logger.debug("Building feature fetch failed for area")

    return {"roads": roads, "buildings": buildings}
