import importlib.util
import pathlib
import plistlib
import re
import shutil
import tempfile
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT / "NeuralAmpModeler"
SCRIPT = PROJECT_ROOT / "scripts" / "update_version-ios.py"
SPEC = importlib.util.spec_from_file_location("dessamp_update_version_ios", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def configured_version():
    source = (PROJECT_ROOT / "config.h").read_text(encoding="utf-8")
    match = re.search(r'^#define PLUG_VERSION_STR "([^"]+)"$', source, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


class UpdateVersionIOSTests(unittest.TestCase):
    def test_repository_version_surfaces_match_config(self):
        version = configured_version()
        for name in (
            "NeuralAmpModeler-iOS-Info.plist",
            "NeuralAmpModeler-iOS-AUv3-Info.plist",
            "NeuralAmpModeler-iOS-AUv3Framework-Info.plist",
        ):
            with (PROJECT_ROOT / "resources" / name).open("rb") as source:
                metadata = plistlib.load(source)
            self.assertEqual(metadata["CFBundleVersion"], version)
            self.assertEqual(metadata["CFBundleShortVersionString"], version)

        project = (
            PROJECT_ROOT / "projects" / "NeuralAmpModeler-iOS.xcodeproj" / "project.pbxproj"
        ).read_text(encoding="utf-8")
        versions = re.findall(
            r"(?:CURRENT_PROJECT_VERSION|MARKETING_VERSION) = ([^;]+);", project
        )
        self.assertTrue(versions)
        self.assertEqual(set(versions), {version})

    def test_updater_changes_every_surface_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "NeuralAmpModeler"
            resources = root / "resources"
            project_dir = root / "projects" / "NeuralAmpModeler-iOS.xcodeproj"
            resources.mkdir(parents=True)
            project_dir.mkdir(parents=True)

            for name in (
                "NeuralAmpModeler-iOS-Info.plist",
                "NeuralAmpModeler-iOS-AUv3-Info.plist",
                "NeuralAmpModeler-iOS-AUv3Framework-Info.plist",
            ):
                shutil.copy2(PROJECT_ROOT / "resources" / name, resources / name)
            shutil.copy2(
                PROJECT_ROOT / "projects" / "NeuralAmpModeler-iOS.xcodeproj" / "project.pbxproj",
                project_dir / "project.pbxproj",
            )

            version = "9.8.7"
            MODULE.projectpath = str(root)
            MODULE.repositorypath = str(pathlib.Path(temporary))
            MODULE.parse_xcconfig = lambda _: {"DEPLOYMENT_TARGET": "14"}
            MODULE.parse_config = lambda _: {
                "BUNDLE_NAME": "DessAmp",
                "FULL_VER_STR": version,
                "PLUG_COPYRIGHT_STR": "Copyright",
                "PLUG_TYPE": 0,
                "PLUG_DOES_MIDI_IN": 0,
                "PLUG_HAS_UI": 1,
                "PLUG_WIDTH": 1280,
                "PLUG_HEIGHT": 720,
                "PLUG_NAME": "DessAmp",
                "PLUG_MFR_ID": "AdMs",
                "PLUG_MFR": "Alexander Dess",
                "PLUG_UNIQUE_ID": "DsAP",
                "PLUG_VERSION_INT": 0x090807,
            }

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

            for name in (
                "NeuralAmpModeler-iOS-Info.plist",
                "NeuralAmpModeler-iOS-AUv3-Info.plist",
                "NeuralAmpModeler-iOS-AUv3Framework-Info.plist",
            ):
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


if __name__ == "__main__":
    unittest.main()
