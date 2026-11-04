import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "select_conditioned_member_by_internal_validation.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "select_conditioned_member_by_internal_validation_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class InternalMemberSelectionTests(unittest.TestCase):
    def make_member(self, root: Path, label: str, esr: float, data_hash: str = "d" * 64):
        directory = root / label
        directory.mkdir()
        model = directory / "conditioned-a2-full.nam"
        model.write_text(json.dumps({"model": label}))
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        (directory / "training-manifest.json").write_text(
            json.dumps(
                {
                    "artifacts": {
                        "model": {"path": model.name, "sha256": digest},
                        "data_config": {"path": "config_data.json", "sha256": data_hash},
                    },
                    "best_internal_validation_esr": esr,
                    "best_epoch": 42,
                    "child": "full",
                    "receptive_field": 6347,
                    "control_order": ["gain_index"],
                }
            )
        )
        return directory

    def test_selects_lower_internal_esr_without_hidden_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seed91 = self.make_member(root, "91", 0.02)
            seed92 = self.make_member(root, "92", 0.01)
            result = MODULE.select([f"91={seed91}", f"92={seed92}"])
            self.assertEqual(result["selected_member"], "92")
            self.assertFalse(result["selection_opened_hidden_evidence"])

    def test_cli_order_breaks_exact_tie(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seed91 = self.make_member(root, "91", 0.01)
            seed92 = self.make_member(root, "92", 0.01)
            result = MODULE.select([f"91={seed91}", f"92={seed92}"])
            self.assertEqual(result["selected_member"], "91")

    def test_rejects_contract_drift(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seed91 = self.make_member(root, "91", 0.01)
            seed92 = self.make_member(root, "92", 0.02, data_hash="e" * 64)
            with self.assertRaisesRegex(ValueError, "contract"):
                MODULE.select([f"91={seed91}", f"92={seed92}"])


if __name__ == "__main__":
    unittest.main()
