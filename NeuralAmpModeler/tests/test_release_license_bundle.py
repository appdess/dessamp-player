import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "dessamp_release_licenses.py"
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dessamp_release_licenses_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReleaseLicenseBundleTests(unittest.TestCase):
    def test_installer_chrome_does_not_reintroduce_legacy_product_identity(self):
        installer = REPOSITORY_ROOT / "NeuralAmpModeler" / "installer"
        text_suffixes = {".in", ".iss", ".rtf", ".txt"}
        for path in installer.rglob("*"):
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            self.assertNotIn(
                "DessMetal",
                path.read_text(encoding="utf-8"),
                msg=f"legacy product identity remains in {path.relative_to(REPOSITORY_ROOT)}",
            )

    def test_source_and_installer_preserve_the_upstream_2022_notice(self):
        required_notice = "Copyright (c) 2022-2025 Steven Atkinson (Neural Amp Modeler)"
        source_license = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
        installer_license = (
            REPOSITORY_ROOT / "NeuralAmpModeler" / "installer" / "license.rtf"
        ).read_text(encoding="utf-8")
        third_party_notice = (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text(
            encoding="utf-8"
        )

        self.assertIn(required_notice, source_license)
        self.assertIn(required_notice, installer_license)
        self.assertIn("NeuralAmpModelerPlugin source lineage", third_party_notice)

    def test_copy_is_exact_hash_manifested_and_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            resources = root / "Resources"
            repository.mkdir()
            resources.mkdir()
            (repository / "NOTICE.md").write_text("notice\n", encoding="utf-8")
            (repository / "LICENSE.txt").write_text("license\n", encoding="utf-8")
            target = resources / "THIRD_PARTY_LICENSES"
            target.mkdir()
            (target / "stale.txt").write_text("stale", encoding="utf-8")

            manifest = MODULE.copy_release_license_bundle(
                repository,
                resources,
                (("NOTICE.md", "NOTICE.md"), ("LICENSE.txt", "LICENSE.txt")),
            )

            self.assertFalse((target / "stale.txt").exists())
            self.assertEqual((target / "NOTICE.md").read_text(encoding="utf-8"), "notice\n")
            self.assertEqual((target / "LICENSE.txt").read_text(encoding="utf-8"), "license\n")
            on_disk = json.loads((target / "LICENSE-MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk, manifest)
            self.assertEqual(
                [item["bundle_path"] for item in manifest["files"]],
                ["NOTICE.md", "LICENSE.txt"],
            )
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

    def test_missing_source_does_not_destroy_existing_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            resources = root / "Resources"
            repository.mkdir()
            target = resources / "THIRD_PARTY_LICENSES"
            target.mkdir(parents=True)
            existing = target / "complete.txt"
            existing.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "required release license is missing"):
                MODULE.copy_release_license_bundle(
                    repository,
                    resources,
                    (("missing.txt", "missing.txt"),),
                )

            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")

    def test_symlinked_leaf_or_parent_does_not_destroy_existing_bundle(self):
        for attack in ("leaf", "parent"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                repository = root / "repository"
                resources = root / "Resources"
                repository.mkdir()
                target = resources / "THIRD_PARTY_LICENSES"
                target.mkdir(parents=True)
                existing = target / "complete.txt"
                existing.write_text("preserve", encoding="utf-8")
                external = root / "external"
                external.mkdir()
                (external / "notice.txt").write_text("external", encoding="utf-8")

                if attack == "leaf":
                    (repository / "notice.txt").symlink_to(external / "notice.txt")
                    source_relative = "notice.txt"
                else:
                    (repository / "licenses").symlink_to(
                        external, target_is_directory=True
                    )
                    source_relative = "licenses/notice.txt"

                with self.assertRaisesRegex(RuntimeError, "missing or unsafe"):
                    MODULE.copy_release_license_bundle(
                        repository,
                        resources,
                        ((source_relative, "notice.txt"),),
                    )

                self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")

    def test_duplicate_or_escaping_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            resources = root / "Resources"
            repository.mkdir()
            resources.mkdir()
            (repository / "one.txt").write_text("one", encoding="utf-8")
            (repository / "two.txt").write_text("two", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "duplicate license bundle destination"):
                MODULE.copy_release_license_bundle(
                    repository,
                    resources,
                    (("one.txt", "same.txt"), ("two.txt", "same.txt")),
                )
            with self.assertRaisesRegex(RuntimeError, "flat filenames"):
                MODULE.copy_release_license_bundle(
                    repository,
                    resources,
                    (("one.txt", "nested/license.txt"),),
                )


if __name__ == "__main__":
    unittest.main()
