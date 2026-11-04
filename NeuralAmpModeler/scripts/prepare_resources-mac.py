#!/usr/bin/env python3

# this script will copy the project's resources (pngs, ttfs, svgs etc) to the correct place
# depending on the value of PLUG_SHARED_RESOURCES in config.h
# resources can either be copied into the plug-in bundle or into a shared path
# since the shared path should be accesible from the mac app sandbox,
# the path used is ~/Music/SHARED_RESOURCES_SUBPATH

import os
import shutil
import sys

scriptpath = os.path.dirname(os.path.realpath(__file__))
projectpath = os.path.abspath(os.path.join(scriptpath, os.pardir))

IPLUG2_ROOT = "../../iPlug2"

sys.path.insert(0, os.path.join(projectpath, "../iPlug2/Scripts"))
sys.path.insert(0, scriptpath)

from parse_config import parse_config
from dessamp_factory_defaults import stage_factory_bank
from dessamp_release_licenses import (
    copy_regular_file_no_follow,
    copy_release_license_bundle,
)


DRIVE_MODEL_FILENAMES = (
    "OD808.nam",
    "SD1.nam",
    "TS9.nam",
    "aesahaettr.nam",
)


def copy_drive_bank(
    source_dir, target_dir, description, *, source_anchor=None
):
    """Copy the exact runtime drive bank from one explicitly approved source."""
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    if not os.path.isdir(source_dir):
        raise RuntimeError(description + " drive directory does not exist")
    missing = [
        filename
        for filename in DRIVE_MODEL_FILENAMES
        if not os.path.isfile(os.path.join(source_dir, filename))
    ]
    if missing:
        raise RuntimeError(
            description + " drive bank is incomplete: " + ", ".join(missing)
        )
    os.makedirs(target_dir, mode=0o0755, exist_ok=True)
    for filename in DRIVE_MODEL_FILENAMES:
        print("copying " + description + " drive slot " + filename)
        copy_regular_file_no_follow(
            os.path.join(source_dir, filename),
            os.path.join(target_dir, filename),
            source_dir,
            source_anchor,
        )


def resolve_drive_source(project_path, environment=None):
    """Return one rights-cleared or explicitly gated private drive source.

    The former resources/models/DessDrive path is deliberately not consulted:
    it contains provenance-restricted historical captures and must never enter
    an ordinary source or release build by accident.
    """
    if environment is None:
        environment = os.environ
    public_dir = os.path.join(project_path, "resources", "Drives")
    private_dir = environment.get("DESSAMP_PRIVATE_DRIVE_DIR", "")
    if private_dir and os.path.isdir(public_dir):
        raise RuntimeError(
            "use either licensed resources/Drives or DESSAMP_PRIVATE_DRIVE_DIR, not both"
        )
    if private_dir:
        if environment.get("DESSAMP_PRIVATE_DRIVE_VALIDATION") != "YES":
            raise RuntimeError(
                "DESSAMP_PRIVATE_DRIVE_DIR requires DESSAMP_PRIVATE_DRIVE_VALIDATION=YES"
            )
        return private_dir, "private validation"
    if os.path.isdir(public_dir):
        return public_dir, "licensed"
    return None, None


def copy_amp_bank(source_dir, target_dir, description, *, source_anchor=None):
    """Copy a bounded .dess catalog without exposing source filenames."""
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    if not os.path.isdir(source_dir):
        raise RuntimeError(description + " directory does not exist")
    packages = [
        os.path.join(source_dir, filename)
        for filename in sorted(os.listdir(source_dir))
        if filename.lower().endswith(".dess")
    ]
    if not packages:
        raise RuntimeError(description + " directory contains no .dess packages")
    if len(packages) > 32:
        raise RuntimeError(description + " catalog exceeds the 32-amplifier bundle limit")
    os.makedirs(target_dir, mode=0o0755, exist_ok=True)
    for index, package in enumerate(packages, start=1):
        destination = os.path.join(target_dir, "amp-{:02d}.dess".format(index))
        print("copying " + description + " amplifier to neutral bundled slot " + os.path.basename(destination))
        copy_regular_file_no_follow(
            package, destination, source_dir, source_anchor
        )


def reset_staged_directory(resource_root, target_dir):
    """Remove one exact generated resource subtree before repopulating it.

    Xcode reuses bundle directories between builds. Without this reset, a
    private listening model from an earlier opt-in build can silently survive
    into an ordinary build. The containment checks keep cleanup bounded to the
    current bundle's resource root.
    """

    resource_root = os.path.abspath(resource_root)
    target_dir = os.path.abspath(target_dir)
    if target_dir == resource_root or os.path.commonpath([resource_root, target_dir]) != resource_root:
        raise RuntimeError("refusing to reset a resource path outside the current bundle")
    if os.path.islink(target_dir):
        os.unlink(target_dir)
    elif os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    elif os.path.exists(target_dir):
        raise RuntimeError("staged resource target exists but is not a directory")


def main():
    config = parse_config(projectpath)

    print("Copying resources ...")

    if config["PLUG_SHARED_RESOURCES"]:
        dst = (
            os.path.expanduser("~")
            + "/Music/"
            + config["SHARED_RESOURCES_SUBPATH"]
            + "/Resources"
        )
    else:
        dst = os.path.join(
            os.environ["TARGET_BUILD_DIR"],
            os.environ["UNLOCALIZED_RESOURCES_FOLDER_PATH"]
        )

    if os.path.exists(dst) == False:
        os.makedirs(dst + "/", 0o0755)

    # Original DessAmp player artwork is the complete active visual resource
    # surface. Preserve the DessAmp/ subdirectory expected by LoadBitmap.
    dessamp_image_root = os.path.join(projectpath, "resources", "DessAmp")
    for filename in [
        "amp-head-shell@2x.png",
        "amp-head-shell-dual-red-modern-v1@2x.png",
        "amp-head-shell-forge-streamlined-v1@2x.png",
        "amp-head-shell-fire-chrome-v1@2x.png",
        "amp-knob-face@2x.png",
        "dess-badge@2x.png",
        "dess-amp-wordmark@2x.png",
        "studio-oxide@2x.png",
        "studio-amber@2x.png",
        "studio-verdant@2x.png",
        "studio-nocturne@2x.png",
        "studio-cab@2x.png",
        "studio-cab-dzl@2x.png",
        "studio-cab-blend@2x.png",
        "studio-cab-recto@2x.png",
        "studio-cab-green@2x.png",
    ]:
        source = os.path.join(dessamp_image_root, filename)
        target_dir = os.path.join(dst, "DessAmp")
        os.makedirs(target_dir, mode=0o0755, exist_ok=True)
        print("copying DessAmp/" + filename + " to " + target_dir)
        copy_regular_file_no_follow(
            source,
            os.path.join(target_dir, filename),
            dessamp_image_root,
            projectpath,
        )

    if os.path.exists(projectpath + "/resources/fonts/"):
        for font in [
            "Michroma-Regular.ttf",
            "Roboto-Regular.ttf",
            "Roboto-SemiBold.ttf",
            "OFL-Roboto.txt",
        ]:
            print("copying " + font + " to " + dst)
            font_root = os.path.join(projectpath, "resources", "fonts")
            copy_regular_file_no_follow(
                os.path.join(font_root, font),
                os.path.join(dst, font),
                font_root,
                projectpath,
            )

    print("copying deterministic third-party license bundle to " + dst)
    copy_release_license_bundle(os.path.dirname(projectpath), dst)

    # PRE drive captures are a separate rights boundary. Ordinary builds carry
    # none unless a complete bank is placed in the explicitly licensed
    # resources/Drives directory. Local listening builds may opt into an
    # external private bank, but only through the matching validation gate.
    drive_target = os.path.join(dst, "models", "DessDrive")
    reset_staged_directory(dst, drive_target)
    drive_source, drive_description = resolve_drive_source(projectpath)
    if drive_source:
        drive_anchor = projectpath if drive_description == "licensed" else drive_source
        copy_drive_bank(
            drive_source,
            drive_target,
            drive_description,
            source_anchor=drive_anchor,
        )

    # Rights-cleared product packages may be added to resources/Amplifiers.
    # Their filenames are neutralized in the bundle; the validated package
    # manifest remains the authoritative identity and control schema.
    public_amp_dir = os.path.join(projectpath, "resources", "Amplifiers")
    public_amp_target = os.path.join(dst, "Amplifiers")
    reset_staged_directory(dst, public_amp_target)
    if os.path.isdir(public_amp_dir):
        copy_amp_bank(
            public_amp_dir,
            public_amp_target,
            "licensed",
            source_anchor=projectpath,
        )

    # Private, provenance-restricted .dess packages may be baked into an
    # explicitly marked local listening build without ever entering Git or an
    # ordinary release bundle. The directory form supports an instant catalog;
    # the original single-package variable remains backward compatible.
    private_amp_package = os.environ.get("DESSAMP_PRIVATE_AMP_PACKAGE", "")
    private_amp_dir = os.environ.get("DESSAMP_PRIVATE_AMP_DIR", "")
    if private_amp_package and private_amp_dir:
        raise RuntimeError(
            "use either DESSAMP_PRIVATE_AMP_PACKAGE or DESSAMP_PRIVATE_AMP_DIR, not both"
        )
    if (private_amp_package or private_amp_dir) and os.environ.get(
        "DESSAMP_PRIVATE_AMP_VALIDATION"
    ) != "YES":
        raise RuntimeError(
            "private amplifier bundling requires DESSAMP_PRIVATE_AMP_VALIDATION=YES"
        )
    private_amp_target = os.path.join(dst, "PrivateValidation", "Amplifiers")
    reset_staged_directory(dst, private_amp_target)
    if private_amp_package:
        private_amp_package = os.path.abspath(os.path.expanduser(private_amp_package))
        if not os.path.isfile(private_amp_package) or not private_amp_package.lower().endswith(".dess"):
            raise RuntimeError("private validation amplifier must be one existing .dess package")
        os.makedirs(private_amp_target, mode=0o0755, exist_ok=True)
        print("copying private validation amplifier to neutral bundled slot")
        copy_regular_file_no_follow(
            private_amp_package,
            os.path.join(private_amp_target, "amp-01.dess"),
            os.path.dirname(private_amp_package),
        )
    elif private_amp_dir:
        copy_amp_bank(private_amp_dir, private_amp_target, "private validation")

    reset_staged_directory(dst, os.path.join(dst, "IRs"))
    stage_factory_bank(dst)


if __name__ == "__main__":
    main()
