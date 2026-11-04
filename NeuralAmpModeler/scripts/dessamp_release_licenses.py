#!/usr/bin/env python3

"""Build the deterministic third-party license bundle for DessAmp products."""

import hashlib
import json
import os
import shutil
import stat


LICENSE_BUNDLE_ENTRIES = (
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ("LICENSE", "DessAmp-MIT.txt"),
    ("NeuralAmpModelerCore/LICENSE", "NeuralAmpModelerCore-MIT.txt"),
    ("AudioDSPTools/LICENSE", "AudioDSPTools-MIT.txt"),
    ("iPlug2/LICENSE.txt", "iPlug2.txt"),
    ("NeuralAmpModelerCore/Dependencies/eigen/COPYING.APACHE", "Eigen-COPYING.APACHE"),
    ("NeuralAmpModelerCore/Dependencies/eigen/COPYING.BSD", "Eigen-COPYING.BSD"),
    ("NeuralAmpModelerCore/Dependencies/eigen/COPYING.MINPACK", "Eigen-COPYING.MINPACK"),
    ("NeuralAmpModelerCore/Dependencies/eigen/COPYING.MPL2", "Eigen-COPYING.MPL2"),
    ("NeuralAmpModelerCore/Dependencies/eigen/COPYING.README", "Eigen-COPYING.README"),
    ("iPlug2/Dependencies/IPlug/VST3_SDK/LICENSE.txt", "VST3-SDK-MIT.txt"),
    ("iPlug2/Dependencies/IPlug/REAPER_SDK/LICENSE", "REAPER-SDK-zlib.txt"),
    ("iPlug2/Dependencies/IGraphics/NanoVG/LICENSE.txt", "NanoVG-zlib.txt"),
    ("iPlug2/Dependencies/IGraphics/NanoSVG/LICENSE.txt", "NanoSVG-zlib.txt"),
    ("iPlug2/Dependencies/IGraphics/MetalNanoVG/LICENSE", "MetalNanoVG-MIT.txt"),
    ("iPlug2/Dependencies/IGraphics/yoga/LICENSE", "Yoga-MIT.txt"),
    ("licenses/nlohmann-json-MIT.txt", "nlohmann-json-MIT.txt"),
    ("iPlug2/Dependencies/IPlug/RTAudio/doc/doxygen/license.txt", "RtAudio.txt"),
    ("licenses/RtMidi-MIT.txt", "RtMidi-MIT.txt"),
    ("licenses/stb-MIT.txt", "stb-MIT.txt"),
    ("licenses/CHOC-ISC.txt", "CHOC-ISC.txt"),
    ("licenses/SST-LanczosResampler-dual.txt", "SST-LanczosResampler-dual.txt"),
    ("licenses/Roboto-Regular-Apache-2.0.txt", "Roboto-Regular-Apache-2.0.txt"),
    ("NeuralAmpModeler/resources/fonts/OFL-Roboto.txt", "Roboto-SemiBold-OFL-1.1.txt"),
    ("licenses/Michroma-OFL-1.1.txt", "Michroma-OFL-1.1.txt"),
)


def _bounded_source_path(source_path, source_root, source_anchor=None):
    source_root = os.path.abspath(os.path.expanduser(os.fspath(source_root)))
    source_path = os.path.abspath(os.path.expanduser(os.fspath(source_path)))
    source_anchor = os.path.abspath(
        os.path.expanduser(os.fspath(source_anchor or source_root))
    )
    try:
        root_is_anchored = (
            os.path.commonpath([source_anchor, source_root]) == source_anchor
        )
        source_is_bounded = (
            os.path.commonpath([source_root, source_path]) == source_root
        )
    except ValueError:
        root_is_anchored = False
        source_is_bounded = False
    if not root_is_anchored:
        raise RuntimeError(
            "approved source root escapes its traversal anchor: " + source_root
        )
    if not source_is_bounded or source_path == source_root:
        raise RuntimeError("source file escapes its approved root: " + source_path)
    return source_anchor, source_root, source_path


def open_regular_source_no_follow(source_path, source_root, source_anchor=None):
    """Open one regular source without following the anchor, parents, or leaf.

    Resource and license roots can be external for explicitly gated private
    listening builds. The caller supplies the exact approved subtree and may
    additionally anchor repository-owned traversal at the project root. This
    rejects a symlink above a narrow resources/fonts or resources/DessAmp root
    without rejecting unrelated system aliases above an explicitly approved
    external private root.
    """

    source_anchor, _source_root, source_path = _bounded_source_path(
        source_path, source_root, source_anchor
    )
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure no-follow resource copying is unavailable")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptors = []
    try:
        current = os.open(source_anchor, directory_flags)
        descriptors.append(current)
        relative_parts = os.path.relpath(source_path, source_anchor).split(os.sep)
        for part in relative_parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        source_descriptor = os.open(relative_parts[-1], file_flags, dir_fd=current)
        metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(source_descriptor)
            raise RuntimeError("resource source is not a regular file: " + source_path)
        return os.fdopen(source_descriptor, "rb"), metadata
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "resource source must be a regular file in a symlink-free approved tree: "
            + source_path
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def copy_open_regular_source(source, metadata, destination_path):
    """Copy an already secured descriptor to a non-symlink destination."""

    destination_path = os.path.abspath(os.fspath(destination_path))
    os.makedirs(os.path.dirname(destination_path), mode=0o0755, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        destination_descriptor = os.open(
            destination_path,
            flags,
            stat.S_IMODE(metadata.st_mode) or 0o644,
        )
    except OSError as error:
        raise RuntimeError(
            "resource destination must be a regular non-symlink file: "
            + destination_path
        ) from error
    try:
        source.seek(0)
        with os.fdopen(destination_descriptor, "wb", closefd=False) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
        os.fchmod(destination_descriptor, stat.S_IMODE(metadata.st_mode) or 0o644)
    finally:
        os.close(destination_descriptor)
    os.utime(
        destination_path,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        follow_symlinks=False,
    )


def copy_regular_file_no_follow(
    source_path, destination_path, source_root, source_anchor=None
):
    """Copy one approved regular source without any source symlink traversal."""

    source, metadata = open_regular_source_no_follow(
        source_path, source_root, source_anchor
    )
    try:
        copy_open_regular_source(source, metadata, destination_path)
    finally:
        source.close()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reset_bounded_directory(resource_root, target_dir):
    resource_root = os.path.abspath(resource_root)
    target_dir = os.path.abspath(target_dir)
    if target_dir == resource_root or os.path.commonpath([resource_root, target_dir]) != resource_root:
        raise RuntimeError("refusing to reset a license path outside the current bundle")
    if os.path.islink(target_dir):
        os.unlink(target_dir)
    elif os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    elif os.path.exists(target_dir):
        raise RuntimeError("license bundle target exists but is not a directory")


def copy_release_license_bundle(
    repository_root,
    resource_root,
    entries=LICENSE_BUNDLE_ENTRIES,
):
    """Copy the exact allowlisted notices and write a hash manifest.

    Every source is verified before the existing generated bundle is reset, so
    a missing checkout dependency cannot replace a previously complete bundle
    with a partial one.
    """

    repository_root = os.path.abspath(repository_root)
    resource_root = os.path.abspath(resource_root)
    normalized_entries = []
    destinations = set()
    try:
        for source_relative, destination_name in entries:
            if os.path.isabs(source_relative) or os.path.isabs(destination_name):
                raise RuntimeError("license bundle paths must be repository-relative")
            if destination_name != os.path.basename(destination_name):
                raise RuntimeError("license bundle destinations must be flat filenames")
            if destination_name in destinations:
                raise RuntimeError("duplicate license bundle destination: " + destination_name)
            destinations.add(destination_name)
            source_path = os.path.abspath(os.path.join(repository_root, source_relative))
            if os.path.commonpath([repository_root, source_path]) != repository_root:
                raise RuntimeError("license source escapes the repository root")
            try:
                source, metadata = open_regular_source_no_follow(
                    source_path, repository_root
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "required release license is missing or unsafe: " + source_relative
                ) from error
            normalized_entries.append(
                (source_relative, source, metadata, destination_name)
            )

        target_dir = os.path.join(resource_root, "THIRD_PARTY_LICENSES")
        _reset_bounded_directory(resource_root, target_dir)
        os.makedirs(target_dir, mode=0o0755, exist_ok=True)

        manifest_files = []
        for source_relative, source, metadata, destination_name in normalized_entries:
            destination_path = os.path.join(target_dir, destination_name)
            copy_open_regular_source(source, metadata, destination_path)
            manifest_files.append(
                {
                    "bundle_path": destination_name,
                    "sha256": _sha256(destination_path),
                    "source_repository_path": source_relative,
                }
            )
    finally:
        for _source_relative, source, _metadata, _destination_name in normalized_entries:
            source.close()

    manifest = {
        "schema_version": 1,
        "files": manifest_files,
    }
    manifest_path = os.path.join(target_dir, "LICENSE-MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest
