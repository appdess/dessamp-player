from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_latency_readout_is_plugin_delay_not_interface_round_trip():
    controls = (ROOT / "NeuralAmpModeler/NeuralAmpModelerControls.h").read_text()
    panel = controls.split("class DessAmpSettingsPanelControl", 1)[1].split("class ", 1)[0]
    assert "Plugin delay " in panel
    assert "1000.0 * mPluginLatency / mHostRate" in panel
    assert "key.VK != kVK_ESCAPE" in panel
    assert "rate == mHostRate && blockFrames == mHostBlock" in panel


def test_latency_notifications_stay_outside_realtime_path():
    source = (ROOT / "NeuralAmpModeler/NeuralAmpModeler.cpp").read_text()
    update = source.split("void NeuralAmpModeler::_UpdateLatency()", 1)[1].split(
        "void NeuralAmpModeler::_UpdateMeters", 1
    )[0]
    assert "latency += boost->GetLatency()" in update
    assert "mBoostActiveTarget.load" in update
    assert "std::lock_guard<std::mutex> lock(mDSPLoadMutex)" in update
    idle = source.split("void NeuralAmpModeler::OnIdle()", 1)[1].split(
        "bool NeuralAmpModeler::SerializeState", 1
    )[0]
    assert "SetAudioStatus(GetSampleRate(), GetBlockSize(), GetLatency())" in idle
    assert "_UpdateLatency();" in idle


def test_fixed_hpf_coefficients_are_prepared_on_reset():
    source = (ROOT / "NeuralAmpModeler/NeuralAmpModeler.cpp").read_text()
    reset = source.split("void NeuralAmpModeler::OnReset()", 1)[1].split(
        "void NeuralAmpModeler::_ServiceDSPRequests", 1
    )[0]
    assert "mHighPass.SetParams" in reset
    render = source.split("void NeuralAmpModeler::ProcessBlock", 1)[1].split(
        "void NeuralAmpModeler::OnReset", 1
    )[0]
    assert "mHighPass.SetParams" not in render
    assert "std::fill(mBoostOutputArray" not in render


def test_auto_input_uses_raw_audio_and_main_thread_parameter_gesture():
    source = (ROOT / "NeuralAmpModeler/NeuralAmpModeler.cpp").read_text()
    render = source.split("void NeuralAmpModeler::ProcessBlock", 1)[1].split(
        "void NeuralAmpModeler::OnReset", 1
    )[0]
    assert render.index("mAutoInputMatch.Process(inputs") < render.index("_ProcessInput(inputs")
    assert "_SetParameterValueFromMainThread(kInputLevel" in source
    assert "mAutoInputBefore != GetParam(kInputLevel)->Value()" in source
    assert "mAutoInputAmpRequest != mAmpLoadRequest.load()" in source
