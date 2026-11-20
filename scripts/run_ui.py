"""Serve the local Drone Vision test interface."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, help="Directory for uploaded videos and run outputs")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        import uvicorn
        from src.web.api import create_app
    except ImportError as error:
        parser.exit(1, f"Install the UI dependencies with uv run --group ui.\n{error}\n")

    uvicorn.run(create_app(output_root=args.output_dir), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
