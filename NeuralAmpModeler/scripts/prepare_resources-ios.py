#!/usr/bin/env python3

# this script will create/update info plist files based on config.h

kAudioUnitType_MusicDevice = "aumu"
kAudioUnitType_MusicEffect = "aumf"
kAudioUnitType_Effect = "aufx"
kAudioUnitType_MIDIProcessor = "aumi"

import plistlib, os, datetime, fileinput, glob, sys, string, shutil

scriptpath = os.path.dirname(os.path.realpath(__file__))
projectpath = os.path.abspath(os.path.join(scriptpath, os.pardir))

IPLUG2_ROOT = "../../iPlug2"

sys.path.insert(0, os.path.join(projectpath, "../iPlug2/Scripts"))
sys.path.insert(0, scriptpath)

from parse_config import parse_config, parse_xcconfig
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


def copy_drive_bank(source_dir, target_dir, description, *, source_anchor=None):
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
    """Return one rights-cleared or explicitly gated private drive source."""
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
    """Copy a bounded .dess catalog into neutral, deterministic slots."""
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
    """Remove one exact generated subtree from a reused app bundle."""
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
    if len(sys.argv) == 2:
        if sys.argv[1] == "app":
            print("Copying resources ...")

            dst = (
                os.environ["TARGET_BUILD_DIR"]
                + "/"
                + os.environ["UNLOCALIZED_RESOURCES_FOLDER_PATH"]
            )

            # The conditioned player shell is original DessAmp artwork. Do not
            # carry the retired DessMetal catalogue or its product identity into
            # the mobile product.
            dessamp_art_dir = os.path.join(projectpath, "resources", "DessAmp")
            if os.path.isdir(dessamp_art_dir):
                for item in os.listdir(dessamp_art_dir):
                    item_path = os.path.join(dessamp_art_dir, item)
                    if os.path.isfile(item_path):
                        print("copying DessAmp/" + item + " to " + dst)
                        copy_regular_file_no_follow(
                            item_path,
                            os.path.join(dst, item),
                            dessamp_art_dir,
                            projectpath,
                        )

            if os.path.exists(projectpath + "/resources/fonts/"):
                fonts = os.listdir(projectpath + "/resources/fonts/")
                for font in fonts:
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

            # Keep the PRE drive asset boundary identical to macOS. Ordinary
            # mobile builds remain model-free; private listening builds must
            # opt into one complete bank through the explicit validation gate.
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

            # Rights-cleared release packages may be added explicitly. Private
            # packages remain outside Git and require a positive validation
            # gate, matching the macOS build boundary.
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
