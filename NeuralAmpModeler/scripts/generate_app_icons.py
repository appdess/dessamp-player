#!/usr/bin/env python3

"""Generate the DessAmp Apple icon catalogue from one hash-bound master."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MASTER = (
    PROJECT_ROOT
    / "resources"
    / "Brand"
    / "AppIconSource"
    / "DessAmp-AppIcon-master.png"
)
DEFAULT_ASSET_DIR = (
    PROJECT_ROOT / "resources" / "Images.xcassets" / "AppIcon.appiconset"
)
EXPECTED_MASTER_SHA256 = (
    "6ce25aee237089ef50d14fa841e2e93458ced864fb7f2b780fec838725cf1ca6"
)

TARGETS_BY_SIZE = {
    1024: ("iOSAppIcon.png", "icon_512x512@2x.png"),
    512: ("icon_512x512.png", "icon_256x256@2x.png"),
    256: ("icon_256x256.png", "icon_128x128@2x.png"),
    128: ("icon_128x128.png",),
    64: ("icon_32x32@2x.png",),
    32: ("icon_32x32.png", "icon_16x16@2x.png"),
    16: ("icon_16x16.png",),
}
TARGET_FILENAMES = tuple(
    filename
    for size in sorted(TARGETS_BY_SIZE, reverse=True)
    for filename in TARGETS_BY_SIZE[size]
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sips() -> str:
    executable = shutil.which("sips")
    if executable is None:
        raise RuntimeError("Apple sips is required to generate the icon catalogue")
    return executable


def icon_metadata(path: pathlib.Path) -> dict[str, object]:
    result = subprocess.run(
        [_sips(), "-g", "pixelWidth", "-g", "pixelHeight", "-g", "hasAlpha", os.fspath(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.strip().partition(": ")
        if separator and key in {"pixelWidth", "pixelHeight", "hasAlpha"}:
            properties[key] = value
    if set(properties) != {"pixelWidth", "pixelHeight", "hasAlpha"}:
        raise RuntimeError(f"could not read complete icon metadata: {path}")
    return {
        "width": int(properties["pixelWidth"]),
        "height": int(properties["pixelHeight"]),
        "has_alpha": properties["hasAlpha"].lower() == "yes",
    }


def verify_master(master: pathlib.Path = DEFAULT_MASTER) -> str:
    master = master.resolve()
    if not master.is_file():
        raise RuntimeError(f"icon master is missing: {master}")
    digest = sha256(master)
    if digest != EXPECTED_MASTER_SHA256:
        raise RuntimeError(
            "icon master hash does not match the owner-approved composition: "
            + digest
        )
    metadata = icon_metadata(master)
    if metadata != {"width": 1254, "height": 1254, "has_alpha": False}:
        raise RuntimeError(f"unexpected icon master metadata: {metadata}")
    return digest


def validate_catalog(asset_dir: pathlib.Path = DEFAULT_ASSET_DIR) -> None:
    contents_path = asset_dir / "Contents.json"
    if not contents_path.is_file():
        raise RuntimeError(f"asset catalogue metadata is missing: {contents_path}")
    contents = json.loads(contents_path.read_text(encoding="utf-8"))
    filenames = tuple(
        image["filename"]
        for image in contents.get("images", [])
        if "filename" in image
    )
    if set(filenames) != set(TARGET_FILENAMES) or len(filenames) != len(TARGET_FILENAMES):
        raise RuntimeError(
            "asset catalogue filenames differ from the deterministic target set"
        )


def _generate(master: pathlib.Path, output_dir: pathlib.Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for size, filenames in TARGETS_BY_SIZE.items():
        sized_path = output_dir / f"icon-{size}.png"
        subprocess.run(
            [
                _sips(),
                "-z",
                str(size),
                str(size),
                os.fspath(master),
                "--out",
                os.fspath(sized_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = icon_metadata(sized_path)
        expected = {"width": size, "height": size, "has_alpha": False}
        if metadata != expected:
            raise RuntimeError(
                f"generated icon metadata mismatch for {size}: {metadata}"
            )
        for filename in filenames:
            target = output_dir / filename
            shutil.copyfile(sized_path, target)
            hashes[filename] = sha256(target)
    return hashes


def _replace_atomically(source: pathlib.Path, destination: pathlib.Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_icon_catalogue(
    master: pathlib.Path = DEFAULT_MASTER,
    asset_dir: pathlib.Path = DEFAULT_ASSET_DIR,
    *,
    check: bool = False,
) -> dict[str, object]:
    master = master.resolve()
    asset_dir = asset_dir.resolve()
    master_digest = verify_master(master)
    validate_catalog(asset_dir)

    with tempfile.TemporaryDirectory(prefix="dessamp-app-icons-") as temporary:
        generated_dir = pathlib.Path(temporary)
        hashes = _generate(master, generated_dir)
        mismatches = [
            filename
            for filename in TARGET_FILENAMES
            if not (asset_dir / filename).is_file()
            or (asset_dir / filename).read_bytes()
            != (generated_dir / filename).read_bytes()
        ]
        if check:
            if mismatches:
                raise RuntimeError(
                    "asset catalogue differs from the hash-bound master: "
                    + ", ".join(mismatches)
                )
        else:
            for filename in TARGET_FILENAMES:
                _replace_atomically(
                    generated_dir / filename,
                    asset_dir / filename,
                )

    return {
        "asset_dir": os.fspath(asset_dir),
        "checked": check,
        "master": os.fspath(master),
        "master_sha256": master_digest,
        "targets": {filename: hashes[filename] for filename in TARGET_FILENAMES},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--master", type=pathlib.Path, default=DEFAULT_MASTER)
    parser.add_argument("--asset-dir", type=pathlib.Path, default=DEFAULT_ASSET_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_icon_catalogue(
        args.master,
        args.asset_dir,
        check=args.check,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
