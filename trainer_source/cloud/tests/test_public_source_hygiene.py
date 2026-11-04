from __future__ import annotations

import hashlib
import json
import tomllib
import unittest
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".mm",
    ".py",
    ".rtf",
    ".sh",
    ".txt",
}


def string_values(value: Any):
    if isinstance(value, dict):
        for nested in value.values():
            yield from string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from string_values(nested)
    elif isinstance(value, str):
        yield value


class PublicSourceHygieneTests(unittest.TestCase):
    def test_primary_runtime_visual_and_font_assets_are_hash_bound(self):
        expected_hashes = {
            "NeuralAmpModeler/resources/Brand/AppIconSource/DessAmp-AppIcon-master.png":
                "6ce25aee237089ef50d14fa841e2e93458ced864fb7f2b780fec838725cf1ca6",
            "NeuralAmpModeler/resources/DessAmp/amp-head-shell-dual-red-modern-v1@2x.png":
                "312630120ed236af06f9536d23514dafc68dbc2764d213b02bd33d6a73de03bd",
            "NeuralAmpModeler/resources/DessAmp/amp-head-shell-fire-chrome-v1@2x.png":
                "6b14615bd85f4a4c7604652456043e39d9f1ded75c92559009cb0402f22501d4",
            "NeuralAmpModeler/resources/DessAmp/amp-head-shell-forge-streamlined-v1@2x.png":
                "0d3192970c8df1ecfb0a6d83ea60d70d263fe1d4f4539fc3fbbe9bc77d8838a1",
            "NeuralAmpModeler/resources/DessAmp/amp-head-shell@2x.png":
                "ed5e4e2313d9c786a7a2c84320a1e37a36fdfe5ce9c4f625b89d64477cc9b66a",
            "NeuralAmpModeler/resources/DessAmp/amp-knob-face@2x.png":
                "3a887fadde80b6ed893d93464a1ff603b22a7d7ef02941751079ad026cf67902",
            "NeuralAmpModeler/resources/DessAmp/dess-badge@2x.png":
                "254a0ab3e59a9f9eb54e1a0a5e33a3802a5e9b93c049211dcb5d3a550ed74a15",
            "NeuralAmpModeler/resources/fonts/Michroma-Regular.ttf":
                "8e2522a39055cfee1bba775892d44141fbc048ece71e931f3c33fcd47446dfa1",
            "NeuralAmpModeler/resources/fonts/Roboto-Regular.ttf":
                "b2efabca5ea4bc56eea829713706b5cd0788b82aca153bd4adde9b1573933b4f",
            "NeuralAmpModeler/resources/fonts/Roboto-SemiBold.ttf":
                "9da280a8e5b3b0a6ae98f8994a6d33d7f5f59f38020bbd8a541bd6c48856588d",
        }
        present = {
            relative: (REPOSITORY_ROOT / relative).is_file()
            for relative in expected_hashes
        }
        self.assertIn(
            sum(present.values()),
            (0, len(expected_hashes)),
            msg=f"conditional runtime assets are only partially present: {present}",
        )
        if not any(present.values()):
            return
        for relative, expected_sha256 in expected_hashes.items():
            with self.subTest(asset=relative):
                self.assertEqual(
                    hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest(),
                    expected_sha256,
                )

    def test_native_targets_share_the_core_owned_eigen_snapshot(self):
        expected = "NeuralAmpModelerCore/Dependencies/eigen"
        for relative in (
            "NeuralAmpModeler/config/DessAmp-mac.xcconfig",
            "NeuralAmpModeler/config/NeuralAmpModeler-ios.xcconfig",
        ):
            with self.subTest(relative=relative):
                text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn(expected, text)
                self.assertNotIn("$(IPLUG2_ROOT)/../eigen", text)

        windows = (
            REPOSITORY_ROOT / "NeuralAmpModeler/config/NeuralAmpModeler-win.props"
        ).read_text(encoding="utf-8")
        self.assertIn(r"NeuralAmpModelerCore\Dependencies\eigen", windows)
        self.assertNotIn(r"$(IPLUG2_ROOT)\..\eigen", windows)

        audio_dsp = (REPOSITORY_ROOT / "AudioDSPTools/CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("../NeuralAmpModelerCore/Dependencies", audio_dsp)
        self.assertNotIn('CMAKE_CURRENT_SOURCE_DIR}/Dependencies', audio_dsp)

        common_mac = (REPOSITORY_ROOT / "common-mac.xcconfig").read_text(
            encoding="utf-8"
        )
        self.assertIn("NeuralAmpModelerCore/Dependencies", common_mac)
        self.assertNotIn("AudioDSPTools/Dependencies", common_mac)
        self.assertNotIn("RTNEURAL_PATH", common_mac)

    def test_native_release_tools_preserve_ieee_nan_and_infinity_behavior(self):
        for relative in (
            "AudioDSPTools/tools/CMakeLists.txt",
            "NeuralAmpModelerCore/tools/CMakeLists.txt",
        ):
            with self.subTest(relative=relative):
                text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
                self.assertNotIn("-Ofast", text)
                self.assertIn("$<$<CONFIG:RELEASE>:-O3>", text)

    def test_python_distributions_declare_and_ship_their_licenses(self):
        for relative_project in ("capture_companion", "trainer_source"):
            with self.subTest(project=relative_project):
                project_root = REPOSITORY_ROOT / relative_project
                metadata = tomllib.loads(
                    (project_root / "pyproject.toml").read_text(encoding="utf-8")
                )["project"]
                self.assertEqual(metadata.get("license"), "MIT")
                self.assertEqual(metadata.get("license-files"), ["LICENSE"])
                self.assertTrue((project_root / "LICENSE").is_file())

        self.assertEqual(
            (REPOSITORY_ROOT / "capture_companion/LICENSE").read_bytes(),
            (REPOSITORY_ROOT / "LICENSE").read_bytes(),
        )

    def test_public_text_has_no_owner_paths_accounts_or_live_cloud_ids(self):
        forbidden = (
            "/Users/" + "adess",
            "dess" + ".alexander" + "@gmail.com",
            "dessmetal-training-" + "20260818",
            "319" + "970029981",
        )
        violations: list[str] = []
        for relative_root in (
            ".planning",
            "capture_companion",
            "docs",
            "NeuralAmpModeler",
            "NeuralAmpModelerCore",
            "trainer_source",
        ):
            root = REPOSITORY_ROOT / relative_root
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                if any(part in {".git", ".venv", "build", "node_modules"} for part in path.parts):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for value in forbidden:
                    if value in text:
                        violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {value}")
        self.assertEqual(violations, [])

    def test_committed_cloud_data_fixtures_use_portable_paths(self):
        fixture_root = REPOSITORY_ROOT / "trainer_source/cloud/fixtures"
        for path in fixture_root.glob("*-data.json"):
            with self.subTest(path=path.name):
                value = json.loads(path.read_text(encoding="utf-8"))
                referenced_paths = [
                    entry
                    for entry in string_values(value)
                    if entry.lower().endswith(".wav")
                ]
                self.assertTrue(referenced_paths)
                self.assertTrue(
                    all(not Path(entry).is_absolute() for entry in referenced_paths)
                )

    def test_model_package_audio_and_checkpoint_files_are_explicit_fixtures_only(self):
        artifact_suffixes = {
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
        expected_hashes = {
            "NeuralAmpModelerCore/example_models/lstm.nam":
                "df9f78c49f49c2bb32411df47e3f53746075adb206b92d017e06379d1e56234a",
            "NeuralAmpModelerCore/example_models/my_model.nam":
                "ceb53469a19ce278e2235da982ae676cb8d5451a8de22a7ecc7a2617d07224d1",
            "NeuralAmpModelerCore/example_models/wavenet.nam":
                "66bda2b379289eff079c0755588bc9a92760654d9cc9af1b97cf30d0e92b167d",
            "NeuralAmpModelerCore/example_models/wavenet_a1_standard.nam":
                "ceb53469a19ce278e2235da982ae676cb8d5451a8de22a7ecc7a2617d07224d1",
            "NeuralAmpModelerCore/example_models/wavenet_a2_max.nam":
                "e0d04507c3d39545e3b4ff034ef495706ade0fc5a0270a7fdc66ae1791711a09",
            "NeuralAmpModelerCore/example_models/wavenet_condition_dsp.nam":
                "99fcb223729795e2870c75c923ff8c86eeb23ed648ab71ac799efaeffaf3fee6",
            "trainer_source/demo_ckpt.ckpt":
                "475fa5533feae501e19afece67fa2ff1d428447c099591b0e39bf29ed6b857bf",
            "trainer_source/nam/models/_resources/loudness_input.wav":
                "84f3c44ddb228909d7ed7bbfeb4c12d0bb761cfd24c2b1925a96071df9392be5",
        }
        bank = REPOSITORY_ROOT / "NeuralAmpModeler/resources/FactoryBank"
        if bank.exists():  # Private checkout; the public source exporter omits this directory.
            import json
            profile = json.loads((bank / "FactoryDefaults.json").read_text())
            cabinets = {
                "Desfire.wav": "a4b4ceb38783f181b7d8615ad8d302b10b34608fb0ce3d1fc2ea8b2415d8dc05",
                "Gleichrichter.wav": "ca7f0aeb657bde3afc4f1f1e4a6792255dc6cb722007ceceaeb4162bb8d3c8d7",
                "Green.wav": "f84295fc00c3172179d0cb6505cb21421907db052992c96a3423a092a34d4697",
                "SHFLD-Mix.wav": "a905249c3c17d6a7a103fe130149d2a4ab81fde0f760056a2e121b368c0a88cd",
            }
            self.assertEqual(profile["ir_sha256"], cabinets["Desfire.wav"])
            expected_hashes.update({f"NeuralAmpModeler/resources/FactoryBank/{name}": digest
                                    for name, digest in cabinets.items()})
        observed = {
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in REPOSITORY_ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower() in artifact_suffixes
            and ".git" not in path.parts
        }
        self.assertEqual(observed, set(expected_hashes))
        for relative, expected_sha256 in expected_hashes.items():
            with self.subTest(fixture=relative):
                self.assertEqual(
                    hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest(),
                    expected_sha256,
                )

        ignore_lines = {
            line.strip()
            for line in (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue({f"*{suffix}" for suffix in artifact_suffixes}.issubset(ignore_lines))
        self.assertIn("!NeuralAmpModelerCore/example_models/*.nam", ignore_lines)
        self.assertIn(
            "!trainer_source/nam/models/_resources/loudness_input.wav",
            ignore_lines,
        )
        self.assertIn("!trainer_source/demo_ckpt.ckpt", ignore_lines)


if __name__ == "__main__":
    unittest.main()
