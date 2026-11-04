import json
import wave
from argparse import Namespace
from types import SimpleNamespace

import pytest

from train_fixed_nam_a2 import core, run


def settings(tmp_path):
    stimulus = tmp_path / "dry.wav"
    with wave.open(str(stimulus), "wb") as wav:
        wav.setparams((1, 3, 48000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x40\x00\x00" * 64)
    return Namespace(
        input=stimulus, output=stimulus, work_dir=tmp_path / "work",
        export_dir=tmp_path / "export", name="Reference", gear_make="Test",
        gear_model="Amp", gear_type="amp", modeled_by=None, tone_type="hi_gain",
        minimum_epochs=200, threshold_esr=0.005, epochs=500,
        output_trim_db=0.0, capture_metadata=None,
    )


@pytest.mark.parametrize("final_esr", [0.004, 0.006])
def test_fixed_reference_checkpoint_gate(tmp_path, monkeypatch, final_esr):
    args = settings(tmp_path)
    upstream_callback = core._PackedBestCheckpoint

    def save_checkpoint(self, trainer, module):
        row = {"submodel_name": "channels_8", "epoch": trainer.current_epoch,
               "best_metric": trainer.esr}
        self.best[0] = row
        (args.work_dir / "packed_best.json").write_text(json.dumps({"submodels": [row]}))

    def export(folder, *, basename, user_metadata, other_metadata):
        assert other_metadata["capture"]["input_wav_sha256"] == other_metadata["capture"]["output_wav_sha256"]
        assert user_metadata.modeled_by is None
        (folder / f"{basename}.nam").write_text("test export")

    def fit(*_args, **_kwargs):
        callback = core._PackedBestCheckpoint()
        trainer = SimpleNamespace(current_epoch=199, esr=0.001, should_stop=False)
        callback.on_validation_end(trainer, None)
        assert not callback.best and not trainer.should_stop
        trainer.current_epoch = 200
        trainer.esr = final_esr
        callback.on_validation_end(trainer, None)
        assert trainer.should_stop == (final_esr <= args.threshold_esr)
        return SimpleNamespace(
            model=SimpleNamespace(net=SimpleNamespace(export=export)),
            metadata=SimpleNamespace(model_dump=lambda: {}, validation_esr=0.007),
        )

    monkeypatch.setattr(upstream_callback, "on_validation_end", save_checkpoint)
    monkeypatch.setattr(core, "train", fit)
    if final_esr <= args.threshold_esr:
        model = run(args)
        assert model.read_text() == "test export"
        summary = json.loads((args.export_dir / "Reference__training-result.json").read_text())
        assert summary["qualified"] and summary["high_tier_epoch"] == 200
        with pytest.raises(FileExistsError):
            run(args)
    else:
        with pytest.raises(RuntimeError, match="checkpoint gate"):
            run(args)
        assert not args.export_dir.exists()
        assert (args.work_dir / "packed_best.json").is_file()
    assert core._PackedBestCheckpoint is upstream_callback
