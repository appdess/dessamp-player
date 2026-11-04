import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTROLS = REPO_ROOT / "NeuralAmpModeler" / "NeuralAmpModelerControls.h"
PLUGIN = REPO_ROOT / "NeuralAmpModeler" / "NeuralAmpModeler.cpp"


class PresetUISourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controls = CONTROLS.read_text(encoding="utf-8")
        cls.plugin = PLUGIN.read_text(encoding="utf-8")

    def test_save_and_save_as_new_are_separate_visible_actions(self):
        self.assertIn('"SAVE", saveBounds', self.controls)
        self.assertIn('"NEW", saveAsBounds', self.controls)
        self.assertIn("PresetSaveHitBounds().Contains(x, y)", self.controls)
        self.assertIn("PresetSaveAsHitBounds().Contains(x, y)", self.controls)
        self.assertIn("SavePresetAs();", self.controls)

    def test_smallest_layout_keeps_two_independent_touch_targets(self):
        area_width = 100.0
        split = area_width * 0.55
        save_width = split - 1.0
        save_as_width = area_width - (split + 1.0)
        self.assertGreaterEqual(save_width + 1.0e-6, 44.0)
        self.assertGreaterEqual(save_as_width + 1.0e-6, 44.0)
        self.assertIn("IsPhoneCompact() ? 100.f", self.controls)
        self.assertIn("row.R - 150.f", self.controls)

    def test_plugin_constrains_save_and_recall_to_active_amplifier_scope(self):
        self.assertIn("ScopedSavePath(mDessPresetDirectory", self.plugin)
        self.assertIn("storedAmplifierId != mDessPresetAmplifierId", self.plugin)
        self.assertIn("_SetDessPresetAmplifierScope(", self.plugin)
        self.assertIn("mDessPresetPath.clear();", self.plugin)
        self.assertIn('mDessPresetLabel = "Model Default";', self.plugin)


if __name__ == "__main__":
    unittest.main()
