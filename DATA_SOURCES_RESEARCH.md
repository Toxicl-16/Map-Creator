# Geospatial Data Sources for Local Terrain STL Generator

Complete research on free, no-API-key geospatial data sources suitable for integrating into a local web application that generates 3D terrain STL models.

---

## 1. Elevation Data

### A. SRTM (Shuttle Radar Topography Mission) — 30m Global Coverage ✅ **BEST FREE/NO-KEY OPTION**

| Item | Detail |
|------|--------|
| **Coverage** | Global (~80% land mass at full global coverage; the HDEM fill-in covers virtually all land). Misses extreme polar regions. |
| **Resolution** | 1 arc-second ≈ 30m (GL1/90m variant) or 3 arc-second ≈ 90m (GL3/CGIAR). Use GL1 (~30m). |
| **Endpoint / Download Pattern** | **NASA open-data S3** (no account needed):<br>`https://elevation-tiles-prod.s3.amazonaws.com/hgt/<lat>/<lon>.hgt.gz`<br>Or OpenTopography mirror:<br>`https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/<lat>/<lon>.hgt.zip'<br>AWS CLI (no signing): `aws s3 cp s3://elevation-tiles-prod/... --no-sign-request` |
| **Rate Limits** | None (static S3 bucket, standard AWS public data limits). No registration or key required. For programmatic bulk download, use `--no-sign-request`. |
| **License** | Public domain in the US (NASA/NGA data). No fee-based license needed. Cite NASA JPL and NGA in any publication. |
| **Attribution** | "Data: © NASA / NGA — SRTM" |
| **Notes for tropical/high-relief terrain** | Adequate fidelity for most use cases; known issues in dense canopy areas (Amazon, Congo basin) where radar penetrates to ground but vegetation obscures true surface. Better than CGIAR-30 but inferior to ALOS World 3D or LIDAR-derived DEMs when available. The HDEM fill-in is superior to original SRTM in tropical/hilly regions. |

---

### B. NASADEM ✅ **RECOMMENDED — IMPROVED OVER SRTM**

| Item | Detail |
|------|--------|
| **Coverage** | Global land surface (fills the same gaps as HDEM fill-in but uses improved processing). |
| **Resolution** | 1 arc-second ≈ 30m. |
| **Endpoint / Download Pattern** | NASA AWS open-data S3:<br>`https://elevation-tiles-prod.s3.amazonaws.com/nasadem/<hgt-file>.nc'<br>AWS CLI: `aws s3 ls --no-sign-request s3://elevation-tiles-prod/`<br>Or CloudFront CDN endpoints for direct HTTP access. |
| **Rate Limits** | None (public AWS open data). No account or key needed. |
| **License** | US Government public domain (NASA data per U.S. Code § 105 — no copyright). Free to use without restrictions. |
| **Attribution** | "Data: © NASA — NASADEM" |
| **Improvement over SRTM** | Uses Phase-Corrected SRTM Plus Topography (PSPT) algorithm; corrects phase-correction errors, fills the HDEM voids, and fuses additional topographic data for improved accuracy especially in tropical/mountainous regions. Released in 2021 — supersedes SRTM GL1 where available. |

---

### C. USGS 3DEP (National Map) ✅ **BEST FOR USA/ALASKA**

| Item | Detail |
|------|--------|
| **Coverage** | United States, Alaska, Hawaii, Puerto Rico, and U.S. territories primarily. Limited international coverage. |
| **Resolution** | 1/3 arc-second (~10m) LiDAR-derived where available; lower resolution in less-mapped areas. Excellent for terrain detail (urban canyons, bridges, rivers). |
| **Endpoint / Download Pattern** | **USGS National Map Downloader API** (no key needed for bulk download):<br>`https://download.3dep.nationalmap.gov/service/v1/...`<br>Daily Elevation tiles available as GeoTIFF via direct HTTP.<br>Also: Bulk download from `https://nationalmap.gov/epqs/` |
| **Rate Limits** | No explicit per-minute cap documented; generous daily quotas for bulk. For very large batches, contact USGS. |
| **License** | Public domain (U.S. Government work). Unrestricted use. |
| **Attribution** | "Data: © USGS 3DEP / National Map" |
| **Notes** | Highest resolution option for U.S. domains. For global coverage, pair with NASADEM/SRTM for non-U.S. areas. Not ideal for tropical regions outside the U.S. (Puerto Rico/USVI have good LiDAR coverage). |

---

### D. Copernicus DEM ⚠️ **REQUIRES ACCOUNT / API KEY**

| Item | Detail |
|------|--------|
| **Coverage** | Global land surface (GLO-30 at 30m where available; GLO-90 at 90m fills gaps). |
| **Resolution** | 1 arc-second ≈ 30m (GLO-30) or ~2.5 arc-seconds ≈ 90m (GLO-90). |
| **Endpoint / Download Pattern** | Sentinel Hub API requires OAuth token (`https://services.sentinel-hub.com/api/v1/process`).<br>EU Copernicus Data Space (dataspace.copernicus.eu) — requires free account registration.<br>**NOT KEYLESS.** Requires authentication for direct access. |
| **Rate Limits** | Varies by provider tier. Sentinel Hub limited tiers available. |
| **License** | EU Data Space terms — use license varies by region; GLO-30 Public has permissive terms, but WorldDEM-derived tiles may have limitations. |
| **Attribution** | "Data © Copernicus Programme 2024 (EU) — Source: processed by ESA" |
| **Conclusion** | Not suitable for a strictly keyless/local deployment. Skip unless user registers with Copernicus or Sentinel Hub. |

---

## 2. Geocoding / Location Search

### A. Photon ⚠️ **REQUIRES SELF-HOSTING FOR HEAVY USE (KEYLESS BUT WITH LIMITS)**

| Item | Detail |
|------|--------|
| **Coverage** | World-wide. Powered by OpenStreetMap data. |
| **Endpoint** | `https://photon.komoot.io/api/` — GET `/api/?q=QUERY&lat=LAT&lon=LON&limit=N`.<br>Reverse: `https://photon.komoot.io/reverse/?lat=LAT&lon=LON&limit=N` |
| **Rate Limits** | Public instance rate-limited but **no documented per-minute cap**. Komoot requires self-hosting for heavy production use due to load on their infrastructure. No API key needed. |
| **License** | ODbL (Open Data Commons Open Database License) — must attribute and share alike.<br>OSM data under ODbL; Photon UI/API code under BSD/MIT variants. |
| **Attribution** | "© <a href='https://photon.komoot.io'>photon</a> | © <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap contributors</a>" |
| **Notes** | Komoot's public instance is convenient for prototyping. For production with moderate-to-heavy traffic, self-host Photon (Docker container `komoot/photon` pulls OSM data independently). Query syntax supports filtering by country (`&countrycodes=XX`), distance (`&limit=N&bbox=`), and geospatial parameters. |
| **Recommended for Map-Creator** | ✅ Good starting point for keyless local development. Self-host if needed later. |

---

### B. Nominatim (OSM geocoder) ⚠️ **KEYLESS BUT STRICT RATE LIMITS**

| Item | Detail |
|------|--------|
| **Coverage** | World-wide (OpenStreetMap data). |
| **Endpoint** | Primary public: `https://nominatim.openstreetmap.org/search?format=json&q=QUERY&limit=N`<br>Reverse: `search?format=json&viewbox=...&bounded=1&q=LAT,LON`<br>(Also reverse: `reverse?format=json&lat=LAT&lon=LON&distance=DIST`) |
| **Rate Limits** | **Strict:** max **1 request per second**. Requires `User-Agent` header identifying your application (must include a valid contact URL or email). No API key needed. 5-second delay between successive calls required from same IP for anonymous use. |
| **License** | ODbL — OpenStreetMap contributors. Must attribute and share alike. |
| **Attribution** | "© OpenStreetMap contributors" (visible on map; included in geocoder output) |
| **Headers Required** | `User-Agent: YourAppName/1.0 (+https://yourapp.example.org; contact: email@example.org)`<br>`Accept-Language: en-US,en;q=0.9` |
| **Nominatim Instances** | Multiple public instances exist (e.g., nominatim.cloud). Some are self-managed and have more generous limits, but none is guaranteed keyless for production. Consider self-hosting from OSM Planet file (~20–80 GB compressed, hundreds of GB expanded). |
| **Recommended for Map-Creator** | ✅ For local development / low-volume use. Self-host if production volume exceeds public instance limits. |

---

### C. Pelias (Self-Hosted Geocoder) ✅ **FLEXIBLE BUT REQUIRES SETUP**

| Item | Detail |
|------|--------|
| **Coverage** | World-wide (uses OSM + US TIGER for USA). |
| **Endpoint** | Self-hosted: `http://localhost:4000/v1/search?text=QUERY`<br>Pelias API follows standard geocoder patterns. |
| **Rate Limits** | None — runs locally, governed by your local resource limits. |
| **License** | MIT / BSD (Pelias codebase itself). Data under ODbL (OSM) and public domain (TIGER). |
| **Attribution** | "© Pelia contributors | © OpenStreetMap contributors" |
| **Recommended for Map-Creator** | ⭐ Excellent choice for a keyless local deployment. Full control, zero external dependencies after initial data load. |

---

## 3. Vector Features (Roads, Buildings, Land Use)

### A. Overpass API ✅ **BEST KEYLESS OPTION FOR RAW OSM DATA**

| Item | Detail |
|------|--------|
| **Coverage** | World-wide OpenStreetMap data. |
| **Endpoint (Public)** | Primary: `https://overpass-api.de/api/interpreter` (GET with `data=` parameter) or `https://overpass-api.de/api/map?bbox=MINLON,MINLAT,MAXLON,MAXLAT`<br>Alternative servers: `https://overpass.komoot.io/api/...`, `https://overpass.openstreetmap.fr/cgi/...`, `https://overpass.anti.team/api/interpreter` |
| **Query Pattern** | Get all nodes/ways/relations within bounding box:<br>`[out:json][bbox:MINLON,MINLAT,MAXLON,MAXLAT];(node["highway"]["highway"~"(motorway|primary|residential)"];way["highway"]["highway"~"(motorway|primary|residential)"];);out geom;`<br>Get ways (road segments): `[out:json][bbox:..];(way["highway"];>;);out body;`<br>Building polygons: `[out:json][bbox:..];(way[building]->.b;node(w.b)->.n;);(._;>;);out skel;` |
| **Rate Limits** | Public instance (`overpass-api.de`): **max 1 query every 10 seconds** from IP. No explicit byte-size limit but oversized queries are rejected (very large bounding boxes may fail or time out). Self-hosting recommended for heavy use.<br>Alternative servers have different limits — check each server's status page at `https://overpass-api.de/status` |
| **License** | ODbL — OpenStreetMap contributors. Must attribute and share alike. |
| **Attribution** | "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap contributors</a>" (must appear in your application) |
| **Data Types Available** | Roads (highway tags), buildings (building=*), land_use, waterways, boundaries, poi nodes, power lines, rail, and ~50+ tag categories. Can filter by highway type, building presence, natural features, etc. |
| **Recommended for Map-Creator** | ✅ Great for quick data fetch in development. For production, self-host Overpass Turbo or an Overpass instance from the latest OSM planet file to avoid rate limits and gain full control. |

---

### B. OpenStreetMap Data Directly ⭐ **BEST FOR FULL CONTROL**

| Item | Detail |
|------|--------|
| **Coverage** | Full global dataset (~1B nodes, 800M+ ways). |
| **Download Pattern** | OsmAnd extracts: `https://download.geofabrik.de/osm/` (country/region extract as .osm.pbf — compressed OpenStreetMap binary format).<br>Bulk planet: `http://planet.openstreetmap.org/pkg/version/weekly/planet-latest.osm.pbf` (~80-100 GB, grows continuously) |
| **Processing** | Load into local Overpass Turbo instance or process with Python libraries (osmnx, pyrosm). Filter to needed features at data load time — no rate limits. |
| **License** | Creative Commons Attribution-ShareAlike 2.0 + ODbL (OpenStreetMap contributors). Must attribute. |
| **Recommended for Map-Creator** | ⭐ Best long-term approach: download regional extract once, process locally, serve vector features from local files/API with zero external dependencies. |

---

## 4. Map Tile Providers

### A. OpenStreetMap Default Tiles ✅ **STANDARD FREE OPTION**

| Item | Detail |
|------|--------|
| **Endpoint** | `https://tile.openstreetmap.org/{z}/{x}/{y}.png` (standard raster tiles, 256×256px PNG) |
| **Coverage** | Global. |
| **Rate Limits** | No strict per-minute count, but OSMF Tile Usage Policy is enforced by blocking abusive traffic:<br>- Must send identifiable `User-Agent` header.<br>- Must not bulk-download/scrape tiles or serve offline features.<br>- Must cache tiles (≥ 7 day TTL if server headers unreadable).<br>- Commercial services risk access being withdrawn at any time. |
| **License** | Tile content: CC-BY-SA 2.0 (not all tile servers use identical content — check specific instance). OSM data itself: ODbL / CC BY-SA 2.0+ |
| **Attribution** | `© OpenStreetMap contributors` (must be visible on every map view, typically bottom-right) |
| **Recommendation for Map-Creator** | ⚠️ Acceptable for prototyping and moderate use but not ideal for heavy production tile serving. Self-host a tile server instead (below). |

---

### B. CartoDB / Carto Tiles ✅ **FREE TIER WITH ATTRIBUTION**

| Item | Detail |
|------|--------|
| **Endpoint** | Light basemap: `https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png`<br>Dark basemap: `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png`<br>({s} = a.basemaps.cartocdn.com) |
| **Coverage** | Global. Based on OSM data with Carto styling. |
| **Rate Limits** | No strict per-minute cap documented for public tiles, but reasonable use expected (standard CDN caching applies). |
| **License** | CC-BY 3.0 US — free to use with attribution. Different styles may have different restrictions — verify per style. |
| **Attribution** | `© OpenStreetMap contributors | © CARTO` (both credits required) |
| **Recommendation for Map-Creator** | ✅ Clean basemaps that render well over terrain. Good production alternative to OSM tile servers. CartoDB-Dark is especially readable on dark-themed 3D map views. |

---

### C. Self-Hosted Tile Server ⭐ **RECOMMENDED FOR PRODUCTION**

| Item | Detail |
|------|--------|
| **Option** | Serve tiles from local Mapnik/mbcache (e.g., `tileserver-gl`, `openlayers` tile cache, or `leaflet` with `.pbf` vector tiles). Use regional OSM extract + `osm2pgsql` to load into PostgreSQL/PostGIS. |
| **Benefit** | Zero external dependency after initial data load. Full control over styling, zoom levels, and caching. No risk of service interruption. |
| **Recommendation for Map-Creator** | ⭐ Best long-term choice. Serve tiles directly from local disk via a simple Node.js (Express) or Python (FastAPI/Fask) static file server pointing at exported PNG/WebP tile cache. |

---

### D. OpenFreeMap / Stadia / Thunderforest ⚠️ **VARIES**

| Provider | No-Key? | Notes |
|----------|---------|-------|
| **OpenFreeMap** | Sometimes. Public tiles available but rate limits and availability are not guaranteed for production. Self-hostable (Docker) from OSM data. Check current licensing/terms at `https://openfreemap.org`. |
| **Stadia Maps** | Requires API key. Free tier has generous monthly limits (~10,000 tile loads/month). |
| **Thunderforest** | Requires API key. 2,500 free requests/day (OpenCycle map); paid for other maps. |

---

## 5. Python Libraries for Geospatial Processing

### Recommended Stack for Map-Creator Terrain Pipeline:

```
core_data = pd.DataFrame([
    ("rasterio", "Raster I/O + CRS handling for DEM tiles (GeoTIFF/HGT)", "Essential"),
    ("numpy", "Array operations, interpolation, grid creation", "Essential"),
    ("pyproj", "CRS transforms (EPSG:4326 ↔ EPSG:3857 / local projections)", "Essential"),
    ("geopandas", "Vector data loading (Shapefile/GeoJSON), spatial joins", "Essential for vector processing"),
    ("osmnx", "Pulls OSM networks, processes building footprints, road centerlines from Overpass", "Highly recommended for Map-Creator"),
    ("shapely", "Geometry primitives - create, clip, union polygons/lines", "Essential"),
    ("matplotlib / plotly", "3D visualization of terrain mesh before STL export", "Visualization"),
    ("trimesh", "STL file I/O, mesh processing, 3D geometry operations", "Essential for STL export"),
    ("meshio", "Alternative STL mesh I/O - supports additional formats", "Optional"),
    ("pyogrio / fiona", "OGR-based vector I/O (GeoJSON, Shapefile)", "Essential"),
    ("scipy", "NDInterp, Delaunay triangulation for terrain surface fitting", "Important"),
    ("numba", "Numba JIT acceleration for computationally intensive DEM processing", "Performance"),
])
```

| Library | Purpose | License | Pip Install |
|---------|---------|---------|-------------|
| **rasterio** | Read DEM tiles (HGT/GeoTIFF), reproject, extract elevation values → numpy arrays. Fast GDAL-backed I/O. | MIT | `pip install rasterio` |
| **numpy** | Core array handling for grid construction from DEM cells. Creates 2D mesh grids (`np.linspace`, `np.mgrid`). | BSD | `pip install numpy` |
| **pyproj** | CRS coordinate transformations (WGS84 to Web Mercator, Lambert, UTM). Critical for projecting lat/lon → local coords. | MIT | `pip install pyproj` |
| **geopandas** | Vector feature processing — building footprints from Overpass, road centerlines, boundary extraction. Spatial joins and clipping. | BSD | `pip install geopandas` |
| **osmnx** | Python interface for OpenStreetMap data — pulls roads, buildings, POIs within bbox via Overpass API. Converts GeoJSON to Graph/GeoDataFrame.<br>`import osmnx as ox`<br>`ox.geometries_from_bbox(lat, lon, lat2, lon2, tags={"building":True})` | MIT | `pip install osmnx` |
| **shapely** | Geometry operations — create polygons from building footprints, clip terrain to study area bounds, compute areas/perimeters. GEOS backend. | BSD | `pip install shapely` |
| **trimesh** | STL export (`mesh.export('terrain.stl', file_type='stl')`), mesh loading/visual validation, surface normals, bounding box extraction. Supports .stl, .off, .obj, .glb, + 100+ formats.<br>`mesh = trimesh.Trimesh(vertices=verts, faces=faces)` | MIT | `pip install trimesh` |
| **plotly** | Interactive 3D scatter/surface visualization of terrain before STL export. Validate mesh topology visually. | MIT | `pip install plotly` |
| **scipy** | `NDGridInterpolator`, Delaunay triangulation (`scipy.spatial.Delaunay`) for filling gaps in DEM tiles, smooth-surfacing. | BSD | `pip install scipy` |
| **hgt** / **elevation** | Python packages specifically designed to download SRTM/GL1 tiles by lat/lon coordinates.<br>`pip install elevation`<br>`elevation.download(bounding_box=min_lon, min_lat, max_lon, max_lat)` — handles tile pattern, HTTP calls, decompression to GeoTIFF automatically. | AGPL-3.0 (elevation); varies for hgt | `pip install elevation` or `hgt` |
| **meshio** | Additional mesh file format support beyond trimesh (VTU, XDMF, Exodus). Useful if output target changes to simulation formats. | MIT | `pip install meshio` |

---

## Summary Comparison Matrix for Primary Decisions

### Elevation Data Recommendation:

1. **Default / Global:** NASADEM (public domain, S3, no key, ~30m improved over SRTM)
2. **U.S.-only (highest res):** USGS 3DEP (1/3-arc-second LiDAR, public domain)
3. **Fallback:** SRTM GL1 from OpenTopography mirror (~30m)

### Geocoding Recommendation:

1. **Prototype / Dev:** Photon (`https://photon.komoot.io/api/`) — no key, generous limits for low volume
2. **Production Local:** Pelias (self-hosted Docker — zero external deps after setup)
3. **Alternative Nominatim:** `nominatim.openstreetmap.org` with proper User-Agent (strict 1 req/s rate limit)

### Vector Features Recommendation:

1. **Development Quick:** Overpass API public instance — `[bbox=...]` queries, 1 query per 10s
2. **Production Local:** Download regional OSM extract from `download.geofabrik.de`, process via osmnx or pyrosm

### Map Tiles Recommendation:

1. **Prototype:** CartoDB basemap (`basemaps.cartocdn.com`) — no key, clean rendering
2. **Production Local:** Serve cached PNG tiles from local disk (leaflet.js + Express/FastAPI) or load OSM data into tileserver-gl / OpenLayers cache

### Key Insight:
The entire pipeline can operate with **zero API keys** at development/prototyping stage using public endpoints. Migration to fully self-hosted (Pelias, Overpass Turbo, local tile server from regional OSM PBF extract) eliminates all external dependencies for production deployment.
