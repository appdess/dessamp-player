from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone DessMetal conditioned-model player")
    parser.add_argument("manifest", type=Path, help="Path to a .dess amp package or legacy player-manifest.json")
    parser.add_argument("--renderer", type=Path, required=True, help="NAM Core render_model executable")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Library" / "Application Support" / "DessMetal Player",
        help="Player input and render cache directory",
    )
    parser.add_argument("--input", type=Path, help="Optional mono WAV loaded when the player starts")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    app = create_app(
        manifest_path=args.manifest,
        renderer_path=args.renderer,
        data_dir=args.data_dir,
        default_input_path=args.input,
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"DessMetal Player: {url}")
    print(f"Model artifact: {app.state.artifact.manifest_path}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
