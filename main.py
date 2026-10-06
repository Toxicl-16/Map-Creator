"""Command-line entry point for Map-Creator.

The application itself lives in :mod:`app.api.routes`; this module only parses
arguments and hands over to Uvicorn, so there is exactly one app serving both
the JSON API and the static frontend.

Usage:
    python main.py --host 127.0.0.1 --port 8080 --debug
"""

import argparse
import logging

import uvicorn

logger = logging.getLogger("map-creator")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="map-creator",
        description="Terrain model generator — select a location and export an STL.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable auto-reload and debug logging")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Explicit log level (overrides --debug)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    level = args.log_level or ("debug" if args.debug else "info")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    )

    logger.info("Serving Map-Creator on http://%s:%d", args.host, args.port)
    uvicorn.run(
        "app.api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level=level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())