# Map-Creator

Turn any location on Earth into a print-ready terrain model. Draw an area on a map,
tune the physical dimensions, and Map-Creator builds a **watertight, manifold** STL
from real NASA SRTM elevation data with optional OpenStreetMap roads and buildings.

No API keys. No accounts. Everything runs locally.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --host 127.0.0.1 --port 8080
```

Then open <http://127.0.0.1:8080>.

| Flag | Purpose |
|------|---------|
| `--host`, `--port` | Bind address (defaults `127.0.0.1:8080`) |
| `--debug` | Auto-reload and debug logging |
| `--log-level LEVEL` | Explicit log level, overrides `--debug` |

To run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## How it works

1. **Find** — search a place name (Nominatim geocoding) or navigate the map.
2. **Select** — drag a rectangle on the map. Drag it to move, use any of the eight
   handles to resize, or click *Use whole view*.
3. **Configure** — physical size in millimetres, relief height range, grid detail,
   and which map features to fold in.
4. **Generate** — the backend runs five real stages and reports each one as it
   completes.
5. **Preview** — the browser parses the exact STL the server produced and renders it
   in an orthographic view. What you see is what you download.
6. **Validate & download** — every model is checked for watertightness and manifold
   edges before it is offered for download.

## The mesh

Terrain, skirt, base plate, roads, buildings and contour lines are all folded into a
single millimetre heightfield, then extruded as one closed solid. That is why the
export is always a single watertight mesh rather than several overlapping shells.

Physical coordinates use a local UTM projection, so millimetre dimensions are not
distorted by latitude. Road and building footprints are projected with the same
transform as the terrain, so features always line up with the ground they sit on.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/geocode` | Resolve a place name to coordinates and bounds |
| `POST` | `/api/generate/stl` | Start a generation job — returns `202` and a token |
| `GET` | `/api/progress/{token}` | Poll real stage transitions until `done` |
| `GET` | `/api/download/{token}` | Download the generated binary STL |

Generation is asynchronous because a real run takes several seconds. Start it, then
poll progress:

```bash
TOKEN=$(curl -sX POST localhost:8080/api/generate/stl \
  -H 'Content-Type: application/json' \
  -d '{"west":-119.62,"south":37.72,"east":-119.59,"north":37.75,
       "model_width_mm":300,"model_depth_mm":220,"resolution_m":30,
       "min_altitude_mm":2,"max_altitude_mm":42}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')

curl -s localhost:8080/api/progress/$TOKEN     # {"stage":"mesh","percent":60,...}
curl -s -o model.stl localhost:8080/api/download/$TOKEN
```

Interactive docs are served at `/docs`.

## Data sources

All keyless and public:

- **Elevation** — NASA SRTM GL1 tiles via the AWS Open Data mirror, cached on disk
  after first download. Public domain. Coverage runs 60°S–56°N; outside that band the
  API returns a clear message instead of a broken model.
- **Geocoding** — [Nominatim](https://nominatim.org/), with
  [Photon](https://photon.komoot.io/) as fallback. ODbL.
- **Roads & buildings** — [Overpass API](https://overpass-api.de/), queried with
  `out geom` so way geometry arrives with real coordinates. ODbL.

Set `NOMINATIM_USER_AGENT` to identify your deployment, as Nominatim's usage policy
requires: `NOMINATIM_USER_AGENT="My App (me@example.com)"`.

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `NOMINATIM_USER_AGENT` | No | Identification for Nominatim/Overpass. A sensible default is built in. |

## Project layout

```
app/
├── api/routes.py            # FastAPI app, job pipeline, static serving
├── geospatial/providers/    # elevation.py, geocoding.py, vector.py
├── models/                  # Pydantic request/response models
├── services/
│   ├── terrain.py           # MeshBuilder, TerrainSettings, STL export
│   ├── features.py          # ModelTransform, roads, buildings, contours
│   └── validation.py        # mesh / STL structural validation
└── utils/projection.py      # WGS84, UTM, local projection
frontend/static/             # index.html, css/main.css, js/app.js
tests/                       # pytest suite (115 tests)
main.py                      # CLI entry point
```

## Development notes

- Downloaded SRTM tiles are cached under `data/` and pruned after an hour, along
  with finished jobs. Neither is committed to git.
- The real bound on geometry is grid resolution (10–128 m) against the selection size
  (50 m–60 km), both validated in the request model. The coarsest setting on the
  largest allowed selection is the practical ceiling.
- Selections outside those limits are rejected up front, because they either produce
  a degenerate mesh or an unusable amount of geometry.

## License

MIT — see [LICENSE](LICENSE).

This product also uses data under other licenses:

- OpenStreetMap data © OpenStreetMap contributors, [ODbL](https://opendatacommons.org/licenses/odbl/)
- SRTM elevation data: NASA/USGS, public domain

> Contains data © OpenStreetMap contributors, ODbL. You are free to copy, distribute,
> transmit and adapt it. If you alter, transform, or build upon this work, you must
> distribute the resulting work under the same license.