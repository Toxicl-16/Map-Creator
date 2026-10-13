"""Overpass query construction for roads and buildings."""

import asyncio

import pytest

from app.geospatial.providers.vector import (
    DEFAULT_HIGHWAY_TYPES,
    BuildingFeatureProvider,
    OverpassUnavailableError,
    RoadFeatureProvider,
    build_road_query,
)


def test_road_query_uses_one_alternation_filter():
    """A way carries a single highway value, so class filters must be OR-ed."""
    query = build_road_query(-119.62, 37.72, -119.59, 37.75)

    assert query.count("way[") == 1
    assert '["highway"~"' in query
    # The regression this guards: consecutive tag filters are AND-ed in QL.
    assert '["highway"="primary"] ["highway"="secondary"]' not in query
    for highway in DEFAULT_HIGHWAY_TYPES:
        assert highway in query
    assert "out geom;" in query


def test_road_query_covers_minor_streets():
    """Residential and service roads must be fetched, not just trunk roads."""
    query = build_road_query(-1.0, 51.0, 0.0, 52.0)
    assert "residential" in query
    assert "service" in query
    assert "tertiary" in query


def test_road_query_places_the_bbox_south_west_north_east():
    query = build_road_query(-119.62, 37.72, -119.59, 37.75)
    assert "(37.72,-119.62,37.75,-119.59)" in query


def test_road_query_restricts_to_the_requested_classes():
    query = build_road_query(-1.0, 51.0, 0.0, 52.0, ["primary", "secondary"])
    assert "^(primary|secondary)$" in query
    assert "residential" not in query


def test_road_query_ignores_hostile_class_names():
    query = build_road_query(-1.0, 51.0, 0.0, 52.0, ['"); out tags; (', "primary"])
    assert '"); out tags; (' not in query
    assert "^(primary)$" in query


def test_road_query_falls_back_to_the_default_classes():
    query = build_road_query(-1.0, 51.0, 0.0, 52.0, ["", None])
    assert "^(motorway|" in query


def test_fetch_roads_returns_only_ways(monkeypatch):
    provider = RoadFeatureProvider(user_agent="test")
    captured = {}

    async def fake_query(ql_string, timeout=60.0):
        captured["ql"] = ql_string
        return {
            "elements": [
                {"type": "way", "id": 1},
                {"type": "node", "id": 2},
            ]
        }

    monkeypatch.setattr(provider._client, "query", fake_query)

    roads = asyncio.run(
        provider.fetch_roads(-119.62, 37.72, -119.59, 37.75, ["primary"])
    )

    assert roads == [{"type": "way", "id": 1}]
    assert '["highway"~"^(primary)$"]' in captured["ql"]


def test_fetch_roads_rejects_out_of_range_bounds():
    provider = RoadFeatureProvider(user_agent="test")
    with pytest.raises(ValueError):
        asyncio.run(provider.fetch_roads(200.0, 0.0, 201.0, 1.0))
    with pytest.raises(ValueError):
        asyncio.run(provider.fetch_roads(-1.0, -99.0, 0.0, 1.0))


def test_fetch_roads_raises_when_every_endpoint_failed(monkeypatch):
    """A total outage must be distinguishable from 'no roads mapped here'."""
    provider = RoadFeatureProvider(user_agent="test")

    async def dead_query(ql_string, timeout=60.0):
        return None

    monkeypatch.setattr(provider._client, "query", dead_query)

    with pytest.raises(OverpassUnavailableError):
        asyncio.run(provider.fetch_roads(-119.62, 37.72, -119.59, 37.75))


def test_fetch_buildings_raises_when_every_endpoint_failed(monkeypatch):
    provider = BuildingFeatureProvider(user_agent="test")

    async def dead_query(ql_string, timeout=60.0):
        return None

    monkeypatch.setattr(provider._client, "query", dead_query)

    with pytest.raises(OverpassUnavailableError):
        asyncio.run(provider.fetch_buildings(-119.62, 37.72, -119.59, 37.75))


def test_building_query_targets_building_ways(monkeypatch):
    provider = BuildingFeatureProvider(user_agent="test")
    captured = {}

    async def fake_query(ql_string, timeout=60.0):
        captured["ql"] = ql_string
        return {"elements": [{"type": "way", "id": 7, "tags": {"building": "yes"}}]}

    monkeypatch.setattr(provider._client, "query", fake_query)

    buildings = asyncio.run(
        provider.fetch_buildings(-119.62, 37.72, -119.59, 37.75)
    )

    assert len(buildings) == 1
    assert 'way["building"]' in captured["ql"]
    assert "(37.72,-119.62,37.75,-119.59)" in captured["ql"]
