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
        with open(path, "rb") as f:
            if f.read() == encoded:
                return False
    except FileNotFoundError:
        pass
    with open(path, "wb") as f:
        f.write(encoded)
    return True


def write_text_if_changed(path, value):
    """Write generated text only when its bytes would actually change."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == value:
                return False
    except FileNotFoundError:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)
    return True


def main():
    config = parse_config(projectpath)
    xcconfig = parse_xcconfig(os.path.join(repositorypath, "common-mac.xcconfig"))

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

    # VST3

    plistpath = projectpath + "/resources/" + "DessAmp" + "-VST3-Info.plist"
    with open(plistpath, "rb") as f:
        vst3 = plistlib.load(f)
        vst3["CFBundleExecutable"] = config["BUNDLE_NAME"]
        vst3["CFBundleGetInfoString"] = CFBundleGetInfoString
        vst3["CFBundleIdentifier"] = (
            config["BUNDLE_DOMAIN"]
            + "."
            + config["BUNDLE_MFR"]
            + ".vst3."
            + config["BUNDLE_NAME"]
            + ""
        )
        vst3["CFBundleName"] = config["BUNDLE_NAME"]
        vst3["CFBundleVersion"] = CFBundleVersion
        vst3["CFBundleShortVersionString"] = CFBundleVersion
        vst3["LSMinimumSystemVersion"] = LSMinimumSystemVersion
        vst3["CFBundlePackageType"] = CFBundlePackageType
        vst3["CFBundleSignature"] = config["PLUG_UNIQUE_ID"]
        vst3["CSResourcesFileMapped"] = CSResourcesFileMapped

        write_plist_if_changed(plistpath, vst3)

    # VST2

    plistpath = projectpath + "/resources/" + "DessAmp" + "-VST2-Info.plist"
    with open(plistpath, "rb") as f:
        vst2 = plistlib.load(f)
        vst2["CFBundleExecutable"] = config["BUNDLE_NAME"]
        vst2["CFBundleGetInfoString"] = CFBundleGetInfoString
        vst2["CFBundleIdentifier"] = (
            config["BUNDLE_DOMAIN"]
            + "."
            + config["BUNDLE_MFR"]
            + ".vst."
            + config["BUNDLE_NAME"]
            + ""
        )
        vst2["CFBundleName"] = config["BUNDLE_NAME"]
        vst2["CFBundleVersion"] = CFBundleVersion
        vst2["CFBundleShortVersionString"] = CFBundleVersion
        vst2["LSMinimumSystemVersion"] = LSMinimumSystemVersion
        vst2["CFBundlePackageType"] = CFBundlePackageType
        vst2["CFBundleSignature"] = config["PLUG_UNIQUE_ID"]
        vst2["CSResourcesFileMapped"] = CSResourcesFileMapped

        write_plist_if_changed(plistpath, vst2)

    # AUDIOUNIT v2

    plistpath = projectpath + "/resources/" + "DessAmp" + "-AU-Info.plist"
    with open(plistpath, "rb") as f:
        auv2 = plistlib.load(f)
        auv2["CFBundleExecutable"] = config["BUNDLE_NAME"]
        auv2["CFBundleGetInfoString"] = CFBundleGetInfoString
        auv2["CFBundleIdentifier"] = (
            config["BUNDLE_DOMAIN"]
            + "."
            + config["BUNDLE_MFR"]
            + ".audiounit."
            + config["BUNDLE_NAME"]
            + ""
        )
        auv2["CFBundleName"] = config["BUNDLE_NAME"]
        auv2["CFBundleVersion"] = CFBundleVersion
        auv2["CFBundleShortVersionString"] = CFBundleVersion
        auv2["LSMinimumSystemVersion"] = LSMinimumSystemVersion
        auv2["CFBundlePackageType"] = CFBundlePackageType
        auv2["CFBundleSignature"] = config["PLUG_UNIQUE_ID"]
        auv2["CSResourcesFileMapped"] = CSResourcesFileMapped

        if config["PLUG_TYPE"] == 0:
            if config["PLUG_DOES_MIDI_IN"]:
                COMPONENT_TYPE = kAudioUnitType_MusicEffect
            else:
                COMPONENT_TYPE = kAudioUnitType_Effect
        elif config["PLUG_TYPE"] == 1:
            COMPONENT_TYPE = kAudioUnitType_MusicDevice
        elif config["PLUG_TYPE"] == 2:
            COMPONENT_TYPE = kAudioUnitType_MIDIProcessor

        auv2["AudioUnit Version"] = config["PLUG_VERSION_HEX"]
        auv2["AudioComponents"] = [{}]
        auv2["AudioComponents"][0]["description"] = config["PLUG_NAME"]
        auv2["AudioComponents"][0]["factoryFunction"] = config["AUV2_FACTORY"]
        auv2["AudioComponents"][0]["manufacturer"] = config["PLUG_MFR_ID"]
        auv2["AudioComponents"][0]["name"] = (
            config["PLUG_MFR"] + ": " + config["PLUG_NAME"]
        )
        auv2["AudioComponents"][0]["subtype"] = config["PLUG_UNIQUE_ID"]
        auv2["AudioComponents"][0]["type"] = COMPONENT_TYPE
        auv2["AudioComponents"][0]["version"] = config["PLUG_VERSION_INT"]
        auv2["AudioComponents"][0]["sandboxSafe"] = True
        auv2["NSPrincipalClass"] = config["AUV2_VIEW_CLASS_STR"]

        write_plist_if_changed(plistpath, auv2)

    # AUDIOUNIT v3

    if config["PLUG_HAS_UI"]:
        NSEXTENSIONPOINTIDENTIFIER = "com.apple.AudioUnit-UI"
    else:
        NSEXTENSIONPOINTIDENTIFIER = "com.apple.AudioUnit"

    plistpath = (
        projectpath + "/resources/" + "DessAmp" + "-macOS-AUv3-Info.plist"
    )

    with open(plistpath, "rb") as f:
        auv3 = plistlib.load(f)
        auv3["CFBundleExecutable"] = config["BUNDLE_NAME"]
        auv3["CFBundleGetInfoString"] = CFBundleGetInfoString
        auv3["CFBundleIdentifier"] = (
            config["BUNDLE_DOMAIN"]
            + "."
            + config["BUNDLE_MFR"]
            + ".app."
            + config["BUNDLE_NAME"]
            + ".AUv3"
        )
        auv3["CFBundleName"] = config["BUNDLE_NAME"]
        auv3["CFBundleVersion"] = CFBundleVersion
        auv3["CFBundleShortVersionString"] = CFBundleVersion
        auv3["LSMinimumSystemVersion"] = "10.12.0"
        auv3["CFBundlePackageType"] = "XPC!"
        auv3["NSExtension"] = dict(
            NSExtensionAttributes=dict(
                AudioComponentBundle="com.AlexanderDess.app."
                + config["BUNDLE_NAME"]
                + ".AUv3Framework",
                AudioComponents=[{}],
            ),
            #                               NSExtensionServiceRoleType = "NSExtensionServiceRoleTypeEditor",
            NSExtensionPointIdentifier=NSEXTENSIONPOINTIDENTIFIER,
            NSExtensionPrincipalClass="IPlugAUViewController_vDessAmp",
        )
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
            {}
        ]

        if config["PLUG_TYPE"] == 1:
            auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["tags"][
                0
            ] = "Synth"
        else:
            auv3["NSExtension"]["NSExtensionAttributes"]["AudioComponents"][0]["tags"][
                0
            ] = "Effects"

        write_plist_if_changed(plistpath, auv3)

    # AAX

    plistpath = projectpath + "/resources/" + "DessAmp" + "-AAX-Info.plist"
    with open(plistpath, "rb") as f:
        aax = plistlib.load(f)
        aax["CFBundleExecutable"] = config["BUNDLE_NAME"]
        aax["CFBundleGetInfoString"] = CFBundleGetInfoString
        aax["CFBundleIdentifier"] = (
            config["BUNDLE_DOMAIN"]
            + "."
            + config["BUNDLE_MFR"]
            + ".aax."
            + config["BUNDLE_NAME"]
            + ""
        )
        aax["CFBundleName"] = config["BUNDLE_NAME"]
        aax["CFBundleVersion"] = CFBundleVersion
        aax["CFBundleShortVersionString"] = CFBundleVersion
        aax["LSMinimumSystemVersion"] = LSMinimumSystemVersion
        aax["CSResourcesFileMapped"] = CSResourcesFileMapped

        write_plist_if_changed(plistpath, aax)

    # APP

    plistpath = (
        projectpath + "/resources/" + config["BUNDLE_NAME"] + "-macOS-Info.plist"
    )

    with open(plistpath, "rb") as f:
        macOSapp = plistlib.load(f)
        macOSapp["CFBundleExecutable"] = config["BUNDLE_NAME"]
        macOSapp["CFBundleGetInfoString"] = CFBundleGetInfoString
        macOSapp["CFBundleIdentifier"] = (
            config["BUNDLE_DOMAIN"]
            + "."
            + config["BUNDLE_MFR"]
            + ".app."
            + config["BUNDLE_NAME"]
            + ""
        )
        macOSapp["CFBundleName"] = config["BUNDLE_NAME"]
        macOSapp["CFBundleVersion"] = CFBundleVersion
        macOSapp["CFBundleShortVersionString"] = CFBundleVersion
        macOSapp["LSMinimumSystemVersion"] = LSMinimumSystemVersion
        macOSapp["CFBundlePackageType"] = "APPL"
        macOSapp["CFBundleSignature"] = config["PLUG_UNIQUE_ID"]
        macOSapp["CSResourcesFileMapped"] = CSResourcesFileMapped
        macOSapp["NSPrincipalClass"] = "SWELLApplication"
        macOSapp["NSMainNibFile"] = "DessAmp" + "-macOS-MainMenu"
        macOSapp["LSApplicationCategoryType"] = "public.app-category.music"
        macOSapp[
            "NSMicrophoneUsageDescription"
        ] = "This app needs mic access to process audio."

        write_plist_if_changed(plistpath, macOSapp)

    # Keep the remaining version-bearing project and installer metadata in
    # lockstep with config.h. These files are not consumed by the plist blocks
    # above, but stale values make Xcode archives and installer copy misleading.
    plistpath = projectpath + "/resources/DessAmp-macOS-AUv3Framework-Info.plist"
    with open(plistpath, "rb") as f:
        auv3framework = plistlib.load(f)
    auv3framework["CFBundleVersion"] = CFBundleVersion
    auv3framework["CFBundleShortVersionString"] = CFBundleVersion
    write_plist_if_changed(plistpath, auv3framework)

    project_file = projectpath + "/projects/DessAmp-macOS.xcodeproj/project.pbxproj"
    with open(project_file, "r", encoding="utf-8") as f:
        project = f.read()
    project = re.sub(
        r"((?:CURRENT_PROJECT_VERSION|MARKETING_VERSION) = )\d+(?:\.\d+){0,2}(;)",
        rf"\g<1>{CFBundleVersion}\g<2>",
        project,
    )
    write_text_if_changed(project_file, project)

    iss_path = projectpath + "/installer/DessAmp.iss"
    with open(iss_path, "r", encoding="utf-8") as f:
        iss = f.read()
    iss = re.sub(r"^(AppVersion|VersionInfoVersion)=.*$", rf"\g<1>={CFBundleVersion}", iss, flags=re.MULTILINE)
    write_text_if_changed(iss_path, iss)

    intro_path = projectpath + "/installer/intro.rtf"
    with open(intro_path, "r", encoding="utf-8") as f:
        intro = f.read()
    intro = re.sub(
        r"(Welcome to )(?:DessAmp|DessMetal) \d+(?:\.\d+){0,2}",
        rf"\g<1>{config['BUNDLE_NAME']} {CFBundleVersion}",
        intro,
    )
    write_text_if_changed(intro_path, intro)


if __name__ == "__main__":
    main()
