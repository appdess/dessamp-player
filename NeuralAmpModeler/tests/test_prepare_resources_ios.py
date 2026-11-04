import importlib.util
import os
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "prepare_resources-ios.py"
SPEC = importlib.util.spec_from_file_location("dessamp_prepare_resources_ios", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareResourcesIOSTests(unittest.TestCase):
    def test_reset_clears_only_the_requested_bundle_subtree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "Resources"
            target = root / "PrivateValidation" / "Amplifiers"
            sibling = root / "keep.txt"
            target.mkdir(parents=True)
            (target / "stale.dess").write_bytes(b"stale")
            sibling.write_text("keep", encoding="utf-8")

            MODULE.reset_staged_directory(root, target)

            self.assertFalse(target.exists())
            self.assertEqual(sibling.read_text(encoding="utf-8"), "keep")

    def test_reset_refuses_root_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            root = base / "Resources"
            root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "outside the current bundle"):
                MODULE.reset_staged_directory(root, root)
            with self.assertRaisesRegex(RuntimeError, "outside the current bundle"):
                MODULE.reset_staged_directory(root, base / "Outside")

    def test_private_catalog_is_sorted_and_neutralized(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            (source / "02-brit.dess").write_bytes(b"brit")
            (source / "01-us.dess").write_bytes(b"us")

            MODULE.copy_amp_bank(source, target, "test")

            self.assertEqual((target / "amp-01.dess").read_bytes(), b"us")
            self.assertEqual((target / "amp-02.dess").read_bytes(), b"brit")

    def test_catalog_rejects_symlinked_package_and_source_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            external = base / "external.dess"
            external.write_bytes(b"external")
            source = base / "source"
            source.mkdir()
            (source / "01-link.dess").symlink_to(external)
            with self.assertRaisesRegex(RuntimeError, "symlink-free"):
                MODULE.copy_amp_bank(source, base / "target-leaf", "test")

            real_source = base / "real-source"
            real_source.mkdir()
            (real_source / "01-real.dess").write_bytes(b"model")
            source_link = base / "source-link"
            source_link.symlink_to(real_source, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink-free"):
                MODULE.copy_amp_bank(source_link, base / "target-root", "test")

    def test_catalog_rejects_symlink_above_narrow_source_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            project = base / "project"
            project.mkdir()
            external_resources = base / "external-resources"
            source = external_resources / "Amplifiers"
            source.mkdir(parents=True)
            (source / "01-external.dess").write_bytes(b"external")
            (project / "resources").symlink_to(
                external_resources, target_is_directory=True
            )

            with self.assertRaisesRegex(RuntimeError, "symlink-free"):
                MODULE.copy_amp_bank(
                    project / "resources" / "Amplifiers",
                    base / "target-ancestor",
                    "test",
                    source_anchor=project,
                )

    def test_catalog_limit_is_enforced_before_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            for index in range(33):
                (source / ("{:02d}.dess".format(index))).write_bytes(b"model")

            with self.assertRaisesRegex(RuntimeError, "32-amplifier bundle limit"):
                MODULE.copy_amp_bank(source, target, "test")
            self.assertFalse(target.exists())

    def test_drive_bank_copies_only_the_exact_runtime_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            for filename in MODULE.DRIVE_MODEL_FILENAMES:
                (source / filename).write_bytes(filename.encode("utf-8"))
            (source / "unrelated.nam").write_bytes(b"ignore")

            MODULE.copy_drive_bank(source, target, "test")

            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                sorted(MODULE.DRIVE_MODEL_FILENAMES),
            )
            self.assertFalse((target / "unrelated.nam").exists())

    def test_drive_bank_rejects_an_incomplete_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "source"
            source.mkdir()
            (source / MODULE.DRIVE_MODEL_FILENAMES[0]).write_bytes(b"one")

            with self.assertRaisesRegex(RuntimeError, "drive bank is incomplete"):
                MODULE.copy_drive_bank(source, pathlib.Path(temporary) / "target", "test")

    def test_private_drive_source_requires_the_explicit_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = pathlib.Path(temporary)
            private = project / "private"
            private.mkdir()

            with self.assertRaisesRegex(RuntimeError, "DESSAMP_PRIVATE_DRIVE_VALIDATION=YES"):
                MODULE.resolve_drive_source(
                    project,
                    {"DESSAMP_PRIVATE_DRIVE_DIR": os.fspath(private)},
                )

            source, description = MODULE.resolve_drive_source(
                project,
                {
                    "DESSAMP_PRIVATE_DRIVE_DIR": os.fspath(private),
                    "DESSAMP_PRIVATE_DRIVE_VALIDATION": "YES",
                },
            )
            self.assertEqual(source, os.fspath(private))
            self.assertEqual(description, "private validation")


if __name__ == "__main__":
    unittest.main()
