from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone DessMetal capture and conditioned-model trainer")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Library" / "Application Support" / "DessMetal Capture",
        help="Persistent application data directory",
    )
    parser.add_argument("--repository-root", type=Path, help="DessMetal checkout used for development fixtures")
    parser.add_argument("--demo-root", type=Path, help="Root containing DessMetal's existing capture WAV folders")
    parser.add_argument("--loader", type=Path, help="Hardened NAM Core loadmodel executable")
    parser.add_argument("--trainer-root", type=Path, help="PANAMA-compatible conditioned trainer source")
    parser.add_argument("--trainer-python", type=Path, default=Path(sys.executable), help="Python used for local training")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    inferred_repository_root = Path(__file__).resolve().parents[3]
    repository_root = args.repository_root.resolve() if args.repository_root else inferred_repository_root
    loader = args.loader
    if loader is None and repository_root is not None:
        candidate = repository_root / "NeuralAmpModelerCore" / "build" / "tools" / "loadmodel"
        if candidate.is_file():
            loader = candidate
    trainer_root = args.trainer_root
    if trainer_root is None:
        candidate = repository_root / "trainer_source"
        if candidate.is_dir():
            trainer_root = candidate
    app = create_app(
        data_dir=args.data_dir,
        repository_root=repository_root,
        demo_root=args.demo_root,
        loader_path=loader,
        trainer_root=trainer_root,
        trainer_python=args.trainer_python,
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"DessMetal Trainer: {url}")
    print(f"Persistent data: {app.state.store.data_dir}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
