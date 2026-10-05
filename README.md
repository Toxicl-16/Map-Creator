# Map-Creator

Interactive terrain model generator — select a geographic area on a map, generate a 3D-printable STL mesh, and download it.

## Quick Start

```bash
cd /path/to/Map-Creator

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variables
cp .env.example .env  # edit as needed
export NOMINATIM_USER_AGENT="AppName (your@email)"

# Run the development server
python main.py --host 127.0.0.1 --port 8080 --debug
```

Open `http://127.0.0.1:8080` in your browser.

## API

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/api/health`    | Health check                             |
| POST   | `/api/geocode`   | Resolve place name → lat/lon + bounds    |
| POST   | `/api/generate/stl` | Generate STL mesh for selected area   |
| GET    | `/api/download/{token}` | Stream back the generated STL file |

## Data Sources

- **Geocoding**: Nominatim (OpenStreetMap) with Photon fallback — ODbL, no API key required. 1 request/second rate limit.
- **Elevation**: SRTM GL1 30m hgt tiles from public mirrors — Public Domain, no API key required.
- **Vector Features**: Overpass API for roads/buildings — ODbL, no API key required.

See the full data source analysis in [`DATA_SOURCES_RESEARCH.md`](DATA_SOURCES_RESEARCH.md).

## Configuration

Required:

- `NOMINATIM_USER_AGENT` — Required by Nominatim public API. Format: `AppName (your@email)`

Optional:

- `OPEN_TOPOGRAPHY_API_KEY` — Optional override for elevation data source.

## Project Structure

```
Map-Creator/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── geospatial/providers/  # Nominatim, SRTM, Overpass clients
│   ├── models/                # Pydantic & domain data classes
│   ├── services/              # MeshBuilder (STL generation)
│   └── utils/                 # Projection utilities (UTM/WGS84)
├── frontend/static/           # Vanilla HTML/CSS/JS UI + Leaflet
├── tests/                     # Test suite
├── main.py                    # Entry point
├── requirements.txt
└── .env.example
```

## License

MIT — see [LICENSE](LICENSE) for details.

This product also uses data under different licenses:

- OpenStreetMap data: [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/)
- SRTM elevation data: Public Domain (NASA/USGS)

Attribution — OpenStreetMap data:
> Contains data © OpenStreetMap contributors, ODbL. You are free to copy, distribute, 
> transmit and adapt it. If you alter, transform, or build upon this work, you must 
> distribute the resulting work under the same license.

## Development Notes

- Grid resolution is capped at 256×1024 nodes to prevent excessive mesh sizes in production.
- Mesh files are cached in `downloads/` and auto-cleaned after 1 hour during startup.
- UTM projection is used for all physical (millimetre) coordinate calculations; direct 
  lat/lon computation is intentionally avoided to prevent spherical distortion errors.
