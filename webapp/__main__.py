"""Run the AIR-SDK chat website:  python -m webapp [--port 8000]"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="AIR-SDK chat website")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Local-only by design: the app exposes machine telemetry and local models.
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
