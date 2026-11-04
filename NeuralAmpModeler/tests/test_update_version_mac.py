import importlib.util
import pathlib
import plistlib
import re
import shutil
import tempfile
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT / "NeuralAmpModeler"
SCRIPT = PROJECT_ROOT / "scripts" / "update_version-mac.py"
SPEC = importlib.util.spec_from_file_location("dessamp_update_version_mac", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

VERSION_PLISTS = (
    "DessAmp-VST3-Info.plist",
    "DessAmp-VST2-Info.plist",
    "DessAmp-AU-Info.plist",
    "DessAmp-macOS-AUv3-Info.plist",
    "DessAmp-AAX-Info.plist",
    "DessAmp-macOS-Info.plist",
    "DessAmp-macOS-AUv3Framework-Info.plist",
)


def configured_version():
    source = (PROJECT_ROOT / "config.h").read_text(encoding="utf-8")
    match = re.search(r'^#define PLUG_VERSION_STR "([^"]+)"$', source, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


class UpdateVersionMacTests(unittest.TestCase):
    def test_repository_version_surfaces_match_config(self):
        version = configured_version()
        for name in VERSION_PLISTS:
            with (PROJECT_ROOT / "resources" / name).open("rb") as source:
                metadata = plistlib.load(source)
            self.assertEqual(metadata["CFBundleVersion"], version)
            self.assertEqual(metadata["CFBundleShortVersionString"], version)

        project = (
            PROJECT_ROOT / "projects" / "DessAmp-macOS.xcodeproj" / "project.pbxproj"
        ).read_text(encoding="utf-8")
        versions = re.findall(
            r"(?:CURRENT_PROJECT_VERSION|MARKETING_VERSION) = ([^;]+);", project
        )
        self.assertTrue(versions)
        self.assertEqual(set(versions), {version})

        installer = (PROJECT_ROOT / "installer" / "DessAmp.iss").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"AppVersion={version}", installer)
        self.assertIn(f"VersionInfoVersion={version}", installer)
        intro = (PROJECT_ROOT / "installer" / "intro.rtf").read_text(encoding="utf-8")
        self.assertIn(f"Welcome to DessAmp {version}", intro)

    def test_updater_changes_every_surface_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "NeuralAmpModeler"
            resources = root / "resources"
            project_dir = root / "projects" / "DessAmp-macOS.xcodeproj"
            installer_dir = root / "installer"
            resources.mkdir(parents=True)
            project_dir.mkdir(parents=True)
            installer_dir.mkdir(parents=True)

            for name in VERSION_PLISTS:
                shutil.copy2(PROJECT_ROOT / "resources" / name, resources / name)
            shutil.copy2(
                PROJECT_ROOT / "projects" / "DessAmp-macOS.xcodeproj" / "project.pbxproj",
                project_dir / "project.pbxproj",
            )
            for name in ("DessAmp.iss", "intro.rtf"):
                shutil.copy2(PROJECT_ROOT / "installer" / name, installer_dir / name)

            version = "9.8.7"
            config = dict(MODULE.parse_config(str(PROJECT_ROOT)))
            config["FULL_VER_STR"] = version
            config["PLUG_VERSION_INT"] = 0x090807
            config["PLUG_VERSION_HEX"] = "0x00090807"

            MODULE.projectpath = str(root)
            MODULE.repositorypath = str(pathlib.Path(temporary))
            MODULE.parse_config = lambda _: config
            MODULE.parse_xcconfig = lambda _: {"DEPLOYMENT_TARGET": "10.15"}

            MODULE.main()
            first_pass = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            MODULE.main()
            second_pass = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(second_pass, first_pass)

            for name in VERSION_PLISTS:
                with (resources / name).open("rb") as source:
                    metadata = plistlib.load(source)
                self.assertEqual(metadata["CFBundleVersion"], version)
                self.assertEqual(metadata["CFBundleShortVersionString"], version)

            project = (project_dir / "project.pbxproj").read_text(encoding="utf-8")
            versions = re.findall(
                r"(?:CURRENT_PROJECT_VERSION|MARKETING_VERSION) = ([^;]+);", project
            )
            self.assertTrue(versions)
            self.assertEqual(set(versions), {version})

            installer = (installer_dir / "DessAmp.iss").read_text(encoding="utf-8")
            self.assertIn(f"AppVersion={version}", installer)
            self.assertIn(f"VersionInfoVersion={version}", installer)
            intro = (installer_dir / "intro.rtf").read_text(encoding="utf-8")
            self.assertIn(f"Welcome to DessAmp {version}", intro)


if __name__ == "__main__":
    unittest.main()
