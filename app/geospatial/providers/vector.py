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
]


class OverpassClient:
    """Lightweight HTTP client for the Overpass API."""

    def __init__(self, user_agent: Optional[str] = None):
        self._user_agent = user_agent or "Map-Creator"

    async def query(self, ql_string: str, timeout: float = 60.0) -> Optional[Dict]:
        """Execute an Overpass QL query and return the JSON response.

        Returns None if all endpoints fail.
        """
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
                    elif resp.status in (429, 503):
                        logger.warning(
                            "Overpass API rate limited (%d), trying next endpoint",
                            resp.status,
                        )
                        await asyncio.sleep(1)
                    else:
                        text = await resp.text(max_size=8000)
                        logger.debug("Overpass returned status %d: %s", resp.status, text)

            except (aiohttp.ClientError, asyncio.TimeoutError):
                logger.warning("Failed to query Overpass API at %s", endpoint)
                continue

        logger.error("All Overpass endpoints failed")
        return None


class RoadFeatureProvider:
    """Fetch road network data from the Overpass API.

    Supported highway types (default query fetches "major" roads):
      motorway, trunk, primary, secondary, tertiary, residential
    """

    def __init__(self, user_agent: Optional[str] = None):
        self._client = OverpassClient(user_agent)

    async def fetch_roads(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
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

        highway_types = ["motorway", "trunk", "primary", "secondary"]
        filters = []
        for htype in highway_types:
            filters.append('["highway"="' + htype + '"]')

        filter_str = " ".join(filters)

        query_str = (
            "[out:json][timeout:60];\n"
            f"( way {filter_str} ({south},{west},{north},{east}); );\n"
            "out;"
        )

        data = await self._client.query(query_str)
        if data is None:
            return []

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
            '( way["building"="yes"]('
            f"{south},{west},{north},{east}"
            ');'
            ' way["building"="*"]('
            f"{south},{west},{north},{east}"
            '); ); out;'
        )

        data = await self._client.query(query_str)
        if data is None:
            return []

        elements = data.get("elements", [])
        buildings = [
            elem for elem in elements
            if elem.get("tags", {}).get("building") == "yes"
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
        road_highways = ["motorway", "trunk", "primary", "secondary"]

    roads = await RoadFeatureProvider().fetch_roads(west, south, east, north)
    buildings: List[Dict] = []

    if include_buildings:
        try:
            buildings = await BuildingFeatureProvider().fetch_buildings(
                west, south, east, north
            )
        except (ValueError, KeyError):
            logger.debug("Building feature fetch failed for area")

    return {"roads": roads, "buildings": buildings}
