from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPOSITORY_ROOT / "tools/stage_public_source.py"
SPEC = importlib.util.spec_from_file_location("dessamp_stage_public_source", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicSourceStageTests(unittest.TestCase):
    def test_installed_products_and_user_state_are_never_source(self):
        for relative in (
            "NeuralAmpModeler/review/DessAmp.app/Contents/MacOS/DessAmp",
            "NeuralAmpModeler/review/DessAmp.component/Contents/Resources/amp.dess",
            "NeuralAmpModeler/review/DessAmp.vst3/Contents/MacOS/DessAmp",
            "NeuralAmpModeler/review/DessAmp.xcarchive/Info.plist",
            "NeuralAmpModeler/resources/PrivateValidation/FactoryDefaults.json",
            "tools/review/bh4.desspreset",
            "tools/review/capture.sqlite3",
            "tools/review/DessAmp.dmg",
            ".planning/private-handoff.md",
        ):
            with self.subTest(relative=relative):
                self.assertEqual(MODULE.classify_relative_file(
                    relative, include_conditional_runtime_assets=True), "exclude")

    def test_classification_keeps_only_explicit_private_artifact_fixtures(self):
        self.assertEqual(
            MODULE.classify_relative_file(
                "NeuralAmpModelerCore/example_models/lstm.nam",
                include_conditional_runtime_assets=False,
            ),
            "include",
        )
        with self.assertRaisesRegex(ValueError, "Private model/audio/checkpoint"):
            MODULE.classify_relative_file(
                "NeuralAmpModeler/resources/Amplifiers/private.dess",
                include_conditional_runtime_assets=True,
            )
        with self.assertRaisesRegex(ValueError, "Credential-bearing"):
            MODULE.classify_relative_file(
                "release/signing-key.p12",
                include_conditional_runtime_assets=False,
            )

    def test_conditional_and_internal_evidence_paths_fail_closed(self):
        self.assertEqual(
            MODULE.classify_relative_file(
                "docs/qa/private-report.md",
                include_conditional_runtime_assets=True,
            ),
            "exclude",
        )
        path = "NeuralAmpModeler/resources/DessAmp/amp-head-shell@2x.png"
        self.assertEqual(
            MODULE.classify_relative_file(
                path, include_conditional_runtime_assets=False
            ),
            "conditional",
        )
        self.assertEqual(
            MODULE.classify_relative_file(
                path, include_conditional_runtime_assets=True
            ),
            "include",
        )

    def test_vendored_examples_benchmarks_and_tests_are_not_public_product_source(self):
        excluded = (
            "AudioDSPTools/Dependencies/nlohmann/json.hpp",
            "iPlug2/Examples/IPlugEffect/IPlugEffect.cpp",
            "iPlug2/Tests/IGraphicsTest/IGraphicsTest.cpp",
            "iPlug2/Documentation/quickstart.md",
            "eigen/bench/btl/COPYING",
            "AudioDSPTools/Dependencies/eigen/test/basicstuff.cpp",
            "NeuralAmpModelerCore/Dependencies/eigen/unsupported/Eigen/CXX11/Tensor",
        )
        for relative in excluded:
            with self.subTest(relative=relative):
                self.assertEqual(
                    MODULE.classify_relative_file(
                        relative, include_conditional_runtime_assets=True
                    ),
                    "exclude",
                )

        for relative in (
            "NeuralAmpModelerCore/Dependencies/eigen/Eigen/Dense",
            "iPlug2/IPlug/IPlug_include_in_plug_hdr.h",
        ):
            with self.subTest(relative=relative):
                self.assertEqual(
                    MODULE.classify_relative_file(
                        relative, include_conditional_runtime_assets=True
                    ),
                    "include",
                )

    def test_minimal_stage_is_inventory_bound_and_never_release_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "source"
            source.mkdir()
            (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
            (source / "code").mkdir()
            (source / "code/main.txt").write_text("safe\n", encoding="utf-8")
            destination = temporary_path / "staged"
            with (
                mock.patch.object(MODULE, "ROOT_FILES", ("LICENSE",)),
                mock.patch.object(MODULE, "ROOT_DIRECTORIES", ("code",)),
                mock.patch.object(MODULE, "EXCLUDED_PREFIXES", ()),
                mock.patch.object(MODULE, "CONDITIONAL_RUNTIME_PREFIXES", ()),
            ):
                status = MODULE.stage_repository(source, destination)

            self.assertFalse(status["release_ready"])
            self.assertEqual(status["file_count"], 2)
            inventory = (destination / "SOURCE-INVENTORY.sha256").read_text(
                encoding="utf-8"
            )
            self.assertIn("  LICENSE\n", inventory)
            self.assertIn("  code/main.txt\n", inventory)
            self.assertTrue((destination / "PUBLIC-STAGE-STATUS.json").is_file())

    def test_staged_secret_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "source"
            source.mkdir()
            (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
            (source / "code").mkdir()
            (source / "code/main.txt").write_text(
                "-----BEGIN " + "PRIVATE KEY-----\n", encoding="utf-8"
            )
            with (
                mock.patch.object(MODULE, "ROOT_FILES", ("LICENSE",)),
                mock.patch.object(MODULE, "ROOT_DIRECTORIES", ("code",)),
                mock.patch.object(MODULE, "EXCLUDED_PREFIXES", ()),
                mock.patch.object(MODULE, "CONDITIONAL_RUNTIME_PREFIXES", ()),
            ):
                with self.assertRaisesRegex(RuntimeError, "private-key marker"):
                    MODULE.stage_repository(source, temporary_path / "staged")

    def test_missing_first_party_markdown_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "source"
            source.mkdir()
            (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
            (source / "docs").mkdir()
            (source / "docs/README.md").write_text(
                "See [missing](not-staged.md).\n", encoding="utf-8"
            )
            with (
                mock.patch.object(MODULE, "ROOT_FILES", ("LICENSE",)),
                mock.patch.object(MODULE, "ROOT_DIRECTORIES", ("docs",)),
                mock.patch.object(MODULE, "EXCLUDED_PREFIXES", ()),
                mock.patch.object(MODULE, "CONDITIONAL_RUNTIME_PREFIXES", ()),
                mock.patch.object(
                    MODULE, "MARKDOWN_LINK_SCAN_EXCLUDED_PREFIXES", ()
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Missing staged Markdown link target"
                ):
                    MODULE.stage_repository(source, temporary_path / "staged")

    def test_existing_first_party_markdown_link_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "source"
            source.mkdir()
            (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
            (source / "docs").mkdir()
            (source / "docs/README.md").write_text(
                "See [license](../LICENSE).\n", encoding="utf-8"
            )
            with (
                mock.patch.object(MODULE, "ROOT_FILES", ("LICENSE",)),
                mock.patch.object(MODULE, "ROOT_DIRECTORIES", ("docs",)),
                mock.patch.object(MODULE, "EXCLUDED_PREFIXES", ()),
                mock.patch.object(MODULE, "CONDITIONAL_RUNTIME_PREFIXES", ()),
                mock.patch.object(
                    MODULE, "MARKDOWN_LINK_SCAN_EXCLUDED_PREFIXES", ()
                ),
            ):
                status = MODULE.stage_repository(
                    source, temporary_path / "staged"
                )
            self.assertTrue(status["first_party_markdown_links_validated"])


if __name__ == "__main__":
    unittest.main()
