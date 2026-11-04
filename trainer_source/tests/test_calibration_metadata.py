import pytest

from trainer_source.calibration_metadata import calibration_model_metadata


def test_measured_physical_calibration_becomes_native_model_metadata():
    assert calibration_model_metadata(
        {
            "metadata": {
                "physical_amplifier": True,
                "calibration": {
                    "status": "measured",
                    "input_level_dbu": -0.54,
                },
            }
        }
    ) == {"input_level_dbu": -0.54}


def test_physical_calibration_requires_finite_input_reference():
    with pytest.raises(ValueError, match="requires input_level_dbu"):
        calibration_model_metadata(
            {
                "metadata": {
                    "physical_amplifier": True,
                    "calibration": {"status": "measured"},
                }
            }
        )


def test_uncalibrated_workflow_model_exports_no_false_reference():
    assert calibration_model_metadata(
        {"metadata": {"physical_amplifier": False}}
    ) == {}
