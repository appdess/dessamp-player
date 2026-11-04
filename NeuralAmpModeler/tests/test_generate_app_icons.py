import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_app_icons.py"
SPEC = importlib.util.spec_from_file_location("dessamp_generate_app_icons", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SOURCE_ONLY = (SCRIPT.parents[2] / "SOURCE-INVENTORY.sha256").is_file()


class GenerateAppIconsTests(unittest.TestCase):
    @unittest.skipIf(SOURCE_ONLY and not MODULE.DEFAULT_MASTER.exists(), "Icon master excluded from public source")
    def test_checked_in_catalogue_matches_hash_bound_master(self):
        result = MODULE.build_icon_catalogue(check=True)

        self.assertTrue(result["checked"])
        self.assertEqual(result["master_sha256"], MODULE.EXPECTED_MASTER_SHA256)
        self.assertEqual(set(result["targets"]), set(MODULE.TARGET_FILENAMES))

    def test_master_hash_prevents_unreviewed_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            altered = pathlib.Path(temporary) / "altered.png"
            altered.write_bytes(b"not the approved master")

            with self.assertRaisesRegex(RuntimeError, "master hash"):
                MODULE.verify_master(altered)

    @unittest.skipIf(SOURCE_ONLY and not MODULE.DEFAULT_ASSET_DIR.exists(), "Icon catalogue excluded from public source")
    def test_catalogue_metadata_references_exact_target_set(self):
        MODULE.validate_catalog()

        self.assertEqual(len(MODULE.TARGET_FILENAMES), 11)


if __name__ == "__main__":
    unittest.main()
