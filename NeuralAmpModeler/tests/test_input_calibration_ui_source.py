from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_settings_panel_exposes_saved_calibration_parameters_and_howto():
    source = (ROOT / "NeuralAmpModeler" / "NeuralAmpModelerControls.h").read_text(
        encoding="utf-8"
    )
    assert "IControl(bounds, {kCalibrateInput, kInputCalibrationLevel, kInputLevel})" in source
    assert "Interface maximum input level (dBu)" in source
    assert "fallback, unverified" in source
    assert "Total gain " in source
    assert "RESET INPUT" in source


def test_dsp_and_matching_share_the_active_model_reference():
    source = (ROOT / "NeuralAmpModeler" / "NeuralAmpModeler.cpp").read_text(encoding="utf-8")
    assert "mModelMetadataInputLevel.load(std::memory_order_relaxed) : 0.0" in source
    assert "GetParam(kInputCalibrationLevel)->Value() - *reference" in source
    assert "TrimDB(peak, mAutoCalibrationBefore - *reference)" in source
