#!/usr/bin/env python3

# this script will create/update info plist files based on config.h

import plistlib, os, re, sys, shutil

IPLUG2_ROOT = "../../iPlug2"

scriptpath = os.path.dirname(os.path.realpath(__file__))
projectpath = os.path.abspath(os.path.join(scriptpath, os.pardir))
repositorypath = os.path.abspath(os.path.join(projectpath, os.pardir))

kAudioUnitType_MusicDevice = "aumu"
kAudioUnitType_MusicEffect = "aumf"
kAudioUnitType_Effect = "aufx"
kAudioUnitType_MIDIProcessor = "aumi"

sys.path.insert(0, os.path.join(repositorypath, "iPlug2", "Scripts"))

from parse_config import parse_config, parse_xcconfig


def write_plist_if_changed(path, value):
    """Avoid dirtying source metadata during an ordinary no-op build."""
    encoded = plistlib.dumps(value, fmt=plistlib.FMT_XML, sort_keys=True)
    try:
        with open(path, "rb") as source:
            if source.read() == encoded:
                return False
    except FileNotFoundError:
        pass
    with open(path, "wb") as destination:
        destination.write(encoded)
    return True


def write_text_if_changed(path, value):
    """Write generated text only when its bytes would actually change."""
    try:
        with open(path, "r", encoding="utf-8") as source:
            if source.read() == value:
                return False
    except FileNotFoundError:
        pass
    with open(path, "w", encoding="utf-8") as destination:
        destination.write(value)
    return True


def main():
    config = parse_config(projectpath)
    xcconfig = parse_xcconfig(os.path.join(repositorypath, "common-ios.xcconfig"))

    CFBundleGetInfoString = (
        config["BUNDLE_NAME"]
        + " v"
        + config["FULL_VER_STR"]
        + " "
        + config["PLUG_COPYRIGHT_STR"]
    )
    CFBundleVersion = config["FULL_VER_STR"]
    CFBundlePackageType = "BNDL"
    CSResourcesFileMapped = True
    LSMinimumSystemVersion = xcconfig["DEPLOYMENT_TARGET"]

    print("Processing Info.plist files...")

    # AUDIOUNIT v3

    if config["PLUG_TYPE"] == 0:
        if config["PLUG_DOES_MIDI_IN"]:
            COMPONENT_TYPE = kAudioUnitType_MusicEffect
        else:
            COMPONENT_TYPE = kAudioUnitType_Effect
    elif config["PLUG_TYPE"] == 1:
        COMPONENT_TYPE = kAudioUnitType_MusicDevice
    elif config["PLUG_TYPE"] == 2:
        COMPONENT_TYPE = kAudioUnitType_MIDIProcessor

    if config["PLUG_HAS_UI"] == 1:
        NSEXTENSIONPOINTIDENTIFIER = "com.apple.AudioUnit-UI"
    else:
        NSEXTENSIONPOINTIDENTIFIER = "com.apple.AudioUnit"

    # Keep the plist filenames stable even when the shipping bundle name differs
    # from the historical Xcode project name.
    plistpath = projectpath + "/resources/NeuralAmpModeler-iOS-AUv3-Info.plist"

    NSEXTENSIONATTRDICT = dict(
        NSExtensionAttributes=dict(AudioComponents=[{}]),
        NSExtensionPointIdentifier=NSEXTENSIONPOINTIDENTIFIER,
    )

    with open(plistpath, "rb") as f:
        auv3 = plistlib.load(f)
        auv3["CFBundleExecutable"] = config["BUNDLE_NAME"] + "AppExtension"
        auv3["CFBundleIdentifier"] = "$(PRODUCT_BUNDLE_IDENTIFIER)"
        auv3["CFBundleName"] = config["BUNDLE_NAME"] + "AppExtension"
        auv3["CFBundleDisplayName"] = config["BUNDLE_NAME"] + "AppExtension"
        auv3["CFBundleVersion"] = CFBundleVersion
        auv3["CFBundleShortVersionString"] = CFBundleVersion
        auv3["CFBundlePackageType"] = "XPC!"
        auv3["NSExtension"] = NSEXTENSIONATTRDICT
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"] = [{}]
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
            "description"
        ] = config["PLUG_NAME"]
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
            "manufacturer"
        ] = config["PLUG_MFR_ID"]
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["name"] = (
            config["PLUG_MFR"] + ": " + config["PLUG_NAME"]
        )
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
            "subtype"
        ] = config["PLUG_UNIQUE_ID"]
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
            "type"
        ] = COMPONENT_TYPE
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
            "version"
        ] = config["PLUG_VERSION_INT"]
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
            "sandboxSafe"
        ] = True
        auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["tags"] = [
            "",
            "",
        ]

        if config["PLUG_TYPE"] == 1:
            auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["tags"][
                0
            ] = "Synth"
        else:
            auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["tags"][
                0
            ] = "Effects"

        if config["PLUG_HAS_UI"] == 1:
            auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["tags"][
                1
            ] = (
                "size:{"
                + str(config["PLUG_WIDTH"])
                + ","
                + str(config["PLUG_HEIGHT"])
                + "}"
            )
            auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0][
                "factoryFunction"
            ] = "IPlugAUViewController_vDessAmp"
            auv3["NSExtension"]["NSExtensionMainStoryboard"] = (
                config["BUNDLE_NAME"] + "-iOS-MainInterface"
            )
        else:
            auv3["NSExtension"][
                "NSExtensionPrincipalClass"
            ] = "IPlugAUViewController_vDessAmp"

        write_plist_if_changed(plistpath, auv3)

    # Standalone APP

    plistpath = projectpath + "/resources/NeuralAmpModeler-iOS-Info.plist"
    with open(plistpath, "rb") as f:
        iOSapp = plistlib.load(f)
        iOSapp["CFBundleExecutable"] = config["BUNDLE_NAME"]
        iOSapp["CFBundleIdentifier"] = "$(PRODUCT_BUNDLE_IDENTIFIER)"
        iOSapp["CFBundleName"] = config["BUNDLE_NAME"]
        iOSapp["CFBundleVersion"] = CFBundleVersion
        iOSapp["CFBundleShortVersionString"] = CFBundleVersion
        iOSapp["CFBundlePackageType"] = "APPL"
        iOSapp["LSApplicationCategoryType"] = "public.app-category.music"

        write_plist_if_changed(plistpath, iOSapp)

    # The shared AU framework is a separately versioned bundle in both the
    # standalone and AUv3 products. Keep its metadata on the same source-of-
    # truth version as the two launchers.
    plistpath = projectpath + "/resources/NeuralAmpModeler-iOS-AUv3Framework-Info.plist"
    with open(plistpath, "rb") as f:
        framework = plistlib.load(f)
    framework["CFBundleVersion"] = CFBundleVersion
    framework["CFBundleShortVersionString"] = CFBundleVersion
    write_plist_if_changed(plistpath, framework)

    # Xcode exposes these values in archives even though the generated plists
    # are authoritative. Leaving them at the template's 1/1.0 is misleading
    # release evidence, so synchronize every iOS project occurrence.
    project_file = projectpath + "/projects/NeuralAmpModeler-iOS.xcodeproj/project.pbxproj"
    with open(project_file, "r", encoding="utf-8") as f:
        project = f.read()
    project = re.sub(
        r"((?:CURRENT_PROJECT_VERSION|MARKETING_VERSION) = )\d+(?:\.\d+){0,2}(;)",
        rf"\g<1>{CFBundleVersion}\g<2>",
        project,
    )
    write_text_if_changed(project_file, project)


if __name__ == "__main__":
    main()
