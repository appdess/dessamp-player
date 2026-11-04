#!/usr/bin/env python3
"""Stage a reviewable DessAmp public-source candidate without Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable
from urllib.parse import unquote


ROOT_FILES = (
    "HeadRush/README.md",
    "HeadRush/packages/dessfire/INSTALL.md",
    "HeadRush/packages/installer.sha256",
    "HeadRush/prototypes/dessfire/PERFORMANCE.md",
    "HeadRush/prototypes/dessfire/build_renderer.sh",
    "HeadRush/targets/hg12/runtime/fire.cpp",
    "HeadRush/targets/hg12/runtime/fire.h",
    ".clang-format",
    ".gitignore",
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "THIRD_PARTY_NOTICES.md",
    "common-ios.xcconfig",
    "common-mac.xcconfig",
    "common-web.mk",
    "common-win.props",
    "dessamp-signing.local.xcconfig.example",
)

ROOT_DIRECTORIES = (
    ".agents/skills/dessamp-conditioned-training",
    ".github",
    "AudioDSPTools",
    "NeuralAmpModeler",
    "NeuralAmpModelerCore",
    "capture_companion",
    "docs",
    "iPlug2",
    "ios",
    "licenses",
    "tools",
    "trainer_source",
)

EIGEN_SOURCE_ROOTS = (
    "NeuralAmpModelerCore/Dependencies/eigen",
)

EIGEN_NON_PRODUCT_SUBTREES = (
    ".gitlab",
    "bench",
    "blas",
    "ci",
    "cmake",
    "debug",
    "demos",
    "doc",
    "failtest",
    "lapack",
    "scripts",
    "test",
    "unsupported",
)

EXCLUDED_PREFIXES = (
    "AudioDSPTools/Dependencies",
    "iPlug2/Scripts/SetFileIcon",
    "docs/experiments",
    "docs/qa",
    "eigen",
    "iPlug2/Documentation",
    "iPlug2/Examples",
    "iPlug2/Tests",
    "NeuralAmpModeler/resources/models",
    "NeuralAmpModeler/resources/private_validation_irs",
    "NeuralAmpModeler/resources/FactoryBank",
    "NeuralAmpModeler/resources/PrivateValidation",
) + tuple(
    f"{root}/{subtree}"
    for root in EIGEN_SOURCE_ROOTS
    for subtree in EIGEN_NON_PRODUCT_SUBTREES
)

CONDITIONAL_RUNTIME_PREFIXES = (
    "NeuralAmpModeler/resources/Brand/AppIconSource",
    "NeuralAmpModeler/resources/DessAmp",
    "NeuralAmpModeler/resources/Images.xcassets",
    "NeuralAmpModeler/resources/fonts",
)

EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".idea",
        ".planning",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vscode",
        "DerivedData",
        "__pycache__",
        "build",
        "build-ios",
        "build-mac",
        "build-win",
        "node_modules",
        "xcuserdata",
    }
)

# A renamed build directory must not turn installed products or user settings
# into source. Keep these exclusions independent of the build directory name.
GENERATED_DIRECTORY_SUFFIXES = (".app", ".component", ".vst3", ".dSYM", ".egg-info", ".xcarchive")
COMPILED_SUFFIXES = {".o", ".obj", ".so", ".dylib", ".dll", ".exe", ".a", ".lib"}
LOCAL_STATE_SUFFIXES = {".user", ".desspreset", ".dmg", ".pkg", ".sqlite", ".sqlite3"}

EXPLICIT_BINARY_FIXTURES = frozenset(
    {
        "NeuralAmpModelerCore/example_models/lstm.nam",
        "NeuralAmpModelerCore/example_models/my_model.nam",
        "NeuralAmpModelerCore/example_models/wavenet.nam",
        "NeuralAmpModelerCore/example_models/wavenet_a1_standard.nam",
        "NeuralAmpModelerCore/example_models/wavenet_a2_max.nam",
        "NeuralAmpModelerCore/example_models/wavenet_condition_dsp.nam",
        "trainer_source/demo_ckpt.ckpt",
        "trainer_source/nam/models/_resources/loudness_input.wav",
    }
)

PRIVATE_ARTIFACT_SUFFIXES = frozenset(
    {
        ".aif",
        ".aiff",
        ".ckpt",
        ".dess",
        ".flac",
        ".m4a",
        ".mp3",
        ".nam",
        ".ogg",
        ".wav",
    }
)

CREDENTIAL_SUFFIXES = frozenset(
    {
        ".cer",
        ".jks",
        ".key",
        ".keystore",
        ".mobileprovision",
        ".p12",
        ".pem",
        ".pfx",
        ".provisionprofile",
    }
)

SECRET_PATTERNS = (
    ("private-key marker", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("RunPod token", re.compile(rb"rpa_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("GitHub token", re.compile(rb"gh[opsu]_[A-Za-z0-9]{30,}")),
    ("Google API key", re.compile(rb"AIza[A-Za-z0-9_-]{30,}")),
    (
        "owner cloud-drive path",
        re.compile(b"Library/CloudStorage/Google" + b"Drive-"),
    ),
)

MARKDOWN_LINK_SCAN_EXCLUDED_PREFIXES = (
    "AudioDSPTools/Dependencies",
    "NeuralAmpModelerCore/Dependencies",
    "iPlug2",
)

MARKDOWN_INLINE_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\n]+)\)")
URI_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _under(relative: str, prefix: str) -> bool:
    return relative == prefix or relative.startswith(prefix + "/")


def classify_relative_file(
    relative: str, *, include_conditional_runtime_assets: bool
) -> str:
    """Return include/exclude, or fail for a secret-bearing file class."""
    path = Path(relative)
    if path.is_absolute() or relative != path.as_posix() or ".." in path.parts:
        raise ValueError(f"Unsafe source-relative path: {relative!r}")
    if any(_under(relative, prefix) for prefix in EXCLUDED_PREFIXES):
        return "exclude"
    if any(part in EXCLUDED_PARTS or part.endswith(GENERATED_DIRECTORY_SUFFIXES) for part in path.parts):
        return "exclude"
    if path.suffix.lower() in LOCAL_STATE_SUFFIXES or path.suffix.lower() in COMPILED_SUFFIXES:
        return "exclude"
    if path.name == ".DS_Store" or path.suffix.lower() in {".pyc", ".pyo", ".log"}:
        return "exclude"
    if path.name in {".env", ".netrc"} or path.name.startswith(".env."):
        raise ValueError(f"Credential-bearing filename is not stageable: {relative}")
    if path.suffix.lower() in CREDENTIAL_SUFFIXES:
        raise ValueError(f"Credential-bearing file class is not stageable: {relative}")
    if path.suffix.lower() in PRIVATE_ARTIFACT_SUFFIXES:
        if relative not in EXPLICIT_BINARY_FIXTURES:
            raise ValueError(f"Private model/audio/checkpoint is not stageable: {relative}")
    if any(_under(relative, prefix) for prefix in CONDITIONAL_RUNTIME_PREFIXES):
        return "include" if include_conditional_runtime_assets else "conditional"
    return "include"


def _iter_directory_files(root: Path, relative_root: str) -> Iterable[Path]:
    absolute_root = root / relative_root
    if absolute_root.is_symlink() or not absolute_root.is_dir():
        raise ValueError(f"Required source directory is missing or unsafe: {relative_root}")
    for current, directory_names, file_names in os.walk(absolute_root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in directory_names:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                raise ValueError(f"Symlinked source directory is not stageable: {relative}")
            if any(_under(relative, prefix) for prefix in EXCLUDED_PREFIXES):
                continue
            if any(part in EXCLUDED_PARTS or part.endswith(GENERATED_DIRECTORY_SUFFIXES) for part in candidate.relative_to(root).parts):
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"Non-regular source file is not stageable: {relative}")
            yield candidate


def collect_source_files(
    root: Path, *, include_conditional_runtime_assets: bool
) -> list[Path]:
    root = root.resolve(strict=True)
    files: list[Path] = []
    for relative in ROOT_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required source file is missing or unsafe: {relative}")
        files.append(path)
    for relative_root in ROOT_DIRECTORIES:
        for path in _iter_directory_files(root, relative_root):
            relative = path.relative_to(root).as_posix()
            classification = classify_relative_file(
                relative,
                include_conditional_runtime_assets=include_conditional_runtime_assets,
            )
            if classification == "include":
                files.append(path)
    relative_files = [path.relative_to(root).as_posix() for path in files]
    if len(relative_files) != len(set(relative_files)):
        raise RuntimeError("Public-source allowlist produced duplicate paths")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_staged_files(destination: Path, relative_files: Iterable[str]) -> None:
    for relative in relative_files:
        path = destination / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Staged path is not a regular file: {relative}")
        data = path.read_bytes()
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise RuntimeError(f"Staged source matches {label}: {relative}")


def validate_staged_markdown_links(
    destination: Path, relative_files: Iterable[str]
) -> None:
    """Reject missing first-party relative links in the staged tree.

    Vendored upstream documentation is intentionally outside this policy: it
    can describe upstream repository layouts that are not part of DessAmp's
    product-source allowlist.
    """

    destination = destination.resolve(strict=True)
    for relative in relative_files:
        if not relative.endswith(".md"):
            continue
        if any(_under(relative, prefix) for prefix in MARKDOWN_LINK_SCAN_EXCLUDED_PREFIXES):
            continue
        markdown_path = destination / relative
        text = markdown_path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for raw_destination in MARKDOWN_INLINE_LINK_PATTERN.findall(line):
                target = raw_destination.strip()
                if target.startswith("<"):
                    closing = target.find(">")
                    if closing < 0:
                        raise RuntimeError(
                            f"Malformed Markdown link in {relative}:{line_number}"
                        )
                    target = target[1:closing]
                else:
                    target = target.split(maxsplit=1)[0]
                target = unquote(target)
                if (
                    not target
                    or target.startswith(("#", "//"))
                    or URI_SCHEME_PATTERN.match(target)
                ):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                resolved = (markdown_path.parent / target).resolve()
                try:
                    resolved.relative_to(destination)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Markdown link escapes staged source in "
                        f"{relative}:{line_number}: {target}"
                    ) from exc
                if not resolved.exists():
                    raise RuntimeError(
                        f"Missing staged Markdown link target in "
                        f"{relative}:{line_number}: {target}"
                    )


def stage_repository(
    root: Path,
    destination: Path,
    *,
    include_conditional_runtime_assets: bool = False,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Destination must not already exist: {destination}")
    if destination == root or root in destination.parents:
        raise ValueError("Destination must be outside the source repository")

    source_files = collect_source_files(
        root,
        include_conditional_runtime_assets=include_conditional_runtime_assets,
    )
    destination.mkdir(parents=True)
    inventory: list[tuple[str, str, int]] = []
    for source in source_files:
        relative = source.relative_to(root)
        staged = destination / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staged, follow_symlinks=False)
        inventory.append((relative.as_posix(), _sha256(staged), staged.stat().st_size))

    relative_files = [relative for relative, _, _ in inventory]
    scan_staged_files(destination, relative_files)
    validate_staged_markdown_links(destination, relative_files)
    inventory_text = "".join(
        f"{digest}  {relative}\n" for relative, digest, _ in inventory
    )
    inventory_path = destination / "SOURCE-INVENTORY.sha256"
    inventory_path.write_text(inventory_text, encoding="utf-8")
    status = {
        "schema_version": 1,
        "release_ready": False,
        "conditional_runtime_assets_included_for_local_validation": (
            include_conditional_runtime_assets
        ),
        "first_party_markdown_links_validated": True,
        "file_count": len(inventory),
        "total_bytes": sum(size for _, _, size in inventory),
        "inventory_sha256": _sha256(inventory_path),
        "release_blockers": [
            "accepted and redistribution-approved final model catalogue",
            "file-level approval or replacement of conditional runtime assets",
            "history-aware secret and dependency scan in a new Git directory",
            "clean-tree desktop, mobile, host, package, and UI validation",
            "separate owner approval for publication and binary release",
        ],
    }
    (destination / "PUBLIC-STAGE-STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--include-conditional-assets-for-local-validation",
        action="store_true",
        help=(
            "copy unapproved UI/icon/font assets for local build validation; "
            "the generated status remains release_ready=false"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    status = stage_repository(
        repository_root,
        args.destination,
        include_conditional_runtime_assets=(
            args.include_conditional_assets_for_local_validation
        ),
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
