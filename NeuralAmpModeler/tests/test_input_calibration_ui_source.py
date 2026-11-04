from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_settings_panel_exposes_saved_calibration_parameters_and_howto():
    source = (ROOT / "NeuralAmpModeler" / "NeuralAmpModelerControls.h").read_text(
        encoding="utf-8"
    )
    assert "IControl(bounds, {kCalibrateInput, kInputCalibrationLevel})" in source
    assert "maximum input level" in source
    assert 'M2 1/4\\\"  +16' in source
    assert "M2 XLR  +10" in source
    assert "0 dBu fallback, unverified" in source
    assert "before Gate / Drive / Amp" in source
    assert "calibrationEnabled ? storedCorrectionDB : 0.0" in source
    assert "OFF; stored" in source
    assert "panel.H() < 500.f ? 110.f : 142.f" in source
    assert "compact ? 17.f : 22.f" in source


def test_dsp_uses_visible_zero_dbu_fallback_for_uncalibrated_models():
    source = (ROOT / "NeuralAmpModeler" / "NeuralAmpModeler.cpp").read_text(
        encoding="utf-8"
    )
    assert "const double modelInputReferenceDBU" in source
    assert ": 0.0;" in source
    assert "GetParam(kInputCalibrationLevel)->Value() - modelInputReferenceDBU" in source
