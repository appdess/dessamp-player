import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import dessamp_factory_defaults as factory


class FactoryDefaultsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = pathlib.Path(temp.name)
        self.bank = root / "bank"
        self.bank.mkdir()
        self.resources = root / "resources"
        self.cabinet = b"selected cabinet"
        self.hashes = {"Desfire.wav": hashlib.sha256(self.cabinet).hexdigest()}
        self.profile = {"schema_version": 1, "amplifiers": [{}], "ir_sha256": self.hashes["Desfire.wav"]}
        (self.bank / "Desfire.wav").write_bytes(self.cabinet)
        (self.bank / "FactoryDefaults.json").write_text(json.dumps(self.profile))
        patch = mock.patch.multiple(factory, PROJECT=root, BANK=self.bank, CABINET_HASHES=self.hashes)
        patch.start()
        self.addCleanup(patch.stop)

    def test_only_pinned_files_enter_bundle_and_profile_follows_amp_gate(self):
        (self.bank / "older.wav").write_bytes(b"unused")
        factory.stage_factory_bank(self.resources, {"DESSAMP_PRIVATE_AMP_VALIDATION": "YES"})
        self.assertEqual([p.name for p in (self.resources / "IRs").iterdir()], ["Desfire.wav"])
        self.assertEqual((self.resources / "IRs/Desfire.wav").read_bytes(), self.cabinet)
        target = self.resources / "PrivateValidation/FactoryDefaults.json"
        self.assertEqual(json.loads(target.read_text()), self.profile)
        factory.stage_factory_bank(self.resources, {})
        self.assertFalse(target.exists())

    def test_only_packaged_cabinet_names_can_be_factory_defaults(self):
        for name in ("other.wav", "../Desfire.wav", None, ["Desfire.wav"]):
            with self.subTest(name=name):
                profile = {**self.profile, "amplifiers": [{"cabinet": name}]}
                self.assertFalse(factory.profile_matches_bank(profile))
        self.assertTrue(factory.profile_matches_bank({**self.profile, "amplifiers": [{"cabinet": "Desfire.wav"}]}))
        self.assertTrue(factory.profile_matches_bank(self.profile))

    def test_packaged_factory_rigs(self):
        bank = pathlib.Path(__file__).resolve().parents[1] / "resources/FactoryBank/FactoryDefaults.json"
        if not bank.exists():
            self.skipTest("Private factory bank omitted from the source distribution")
        amps = json.loads(bank.read_text())["amplifiers"]
        self.assertEqual([a["cabinet"] for a in amps], ["SHFLD-Mix.wav", "Desfire.wav", "Green.wav", "Green.wav"])
        params = [a["parameters"] for a in amps]
        self.assertEqual([(p[2], p[9]) for p in params], [(-85, 1)] * 4)
        self.assertEqual([p[11] for p in params], [0, 1, 1, 1])
        self.assertEqual([p[14] for p in params[1:]], [2, 3, 3])
        self.assertEqual(params[1][8], 7.8)
        for p, maximum in zip(params[2:], (10, 8)):
            self.assertAlmostEqual(1 + p[21] * (maximum - 1) / 100, 7.0)
            self.assertEqual(p[29:33], [0, 0, 0, 7])  # Only Nocturne gets +7 dB.

    def test_rejects_wrong_audio_and_mismatched_profile(self):
        (self.bank / "FactoryDefaults.json").write_text(json.dumps({**self.profile, "ir_sha256": "0"*64}))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            factory.stage_factory_bank(self.resources, {"DESSAMP_PRIVATE_AMP_VALIDATION": "YES"})
        (self.bank / "Desfire.wav").write_bytes(b"another voicing")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            factory.stage_factory_bank(self.resources)

    def test_rejects_symlink_even_with_matching_bytes(self):
        original = self.bank / "Desfire.wav"
        moved = self.bank.parent / "outside.wav"
        original.rename(moved)
        original.symlink_to(moved)
        with self.assertRaises((RuntimeError, OSError)):
            factory.stage_factory_bank(self.resources)

    def test_source_only_export_can_build_without_private_bank(self):
        with mock.patch.object(factory, "BANK", self.bank.parent / "omitted"):
            factory.stage_factory_bank(self.resources)
        self.assertFalse((self.resources / "IRs").exists())

    def test_preset_default_is_separate_from_user_files(self):
        source = (pathlib.Path(__file__).resolve().parents[1] / "NeuralAmpModeler.cpp").read_text()
        self.assertIn('mDessPresetLabel = "Model Default"', source)
        self.assertIn("_ApplyBundledFactoryDefaults(manifest, true)", source)
        self.assertIn("_ApplyBundledFactoryDefaults(mBundledAmplifierPaths[target], true)", source)
        self.assertIn("Factory default identity mismatch", source)
        self.assertIn("mBundledFactoryDefaultsValid.assign(mBundledAmplifierPaths.size(), true)", source)


if __name__ == "__main__":
    unittest.main()
