# Suppress harmless urllib3 OpenSSL warning on macOS (uses LibreSSL)
# Must be before any imports that might trigger urllib3
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

from nam.train import full as _full
from pathlib import Path as _Path
import json as _json
from nam.util import timestamp as _timestamp
import argparse

print("script started")

def make_outdir(base_outdir: str, suffix: str) -> _Path:
    outdir = _Path(base_outdir, _timestamp() + suffix)
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with custom configurations.")
    parser.add_argument(
        "--base-outdir",
        type=str,
        default="outputs",
        help="Base output directory for saving the model and logs."
    )
    parser.add_argument(
        "--outdir-suffix",
        type=str,
        default="",
        help="Suffix to append to the output directory name."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="training_data/nam_train_data/metadata.json",
        help="Path to the data configuration file."
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="default_config_files/models/wavenet.json",
        help="Path to the model configuration file."
    )
    parser.add_argument(
        "--learning-config",
        type=str,
        default="default_config_files/learning/default.json",
        help="Path to the learning configuration file."
    )
    parser.add_argument(
        "--get-test-set-metrics",
        action="store_true",
        help="If set, will compute metrics on the test set after training. You should then also provide a test data configuration file using --test-data-config."
    )
    parser.add_argument(
        "--test-data-config",
        type=str,
        default="training_data/30min_val_data/ckpt_964.json",
        help="Path to the test data configuration file. Only used if --get-test-set-metrics is set. Only the validation split of this data will be used."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=5,
        help="Check validation every n epochs."
    )
    parser.add_argument(
        "--max-val-datapoints",
        type=int,
        default=1000,
        help="Maximum number of validation data points to use. If the validation set has more than this many data points, it will be randomly sampled down to this size."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=96,
        help="Batch size for testing the model. This is only used if --get-test-set-metrics is set."
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=8192,
        help="Chunk length in samples for training. This is the length of the input/output sequences."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Initialize from checkpoint model weights. Optimizer, scheduler, "
            "and training-loop state are not resumed."
        )
    )
    parser.add_argument(
        "--threshold-esr",
        type=float,
        default=None,
        help="Stop training early when ESR drops below this threshold (e.g., 0.009 for 0.9%%)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Use distinct seeds when training an uncertainty ensemble."
    )
    args = parser.parse_args()

    import pytorch_lightning as _pl
    _pl.seed_everything(args.seed, workers=True)

    outdir = make_outdir(args.base_outdir, args.outdir_suffix)
    with open(args.model_config, "r") as fp:
        model_config = _json.load(fp)
        model_config["checkpoint_path"] = args.checkpoint_path
        print("model config loaded from", args.model_config)
        print("checkpoint path set to", args.checkpoint_path)
    with open(args.learning_config, "r") as fp:
        learning_config = _json.load(fp)
        learning_config["train_dataloader"]["batch_size"] = args.batch_size
        learning_config["trainer"]["check_val_every_n_epoch"] = args.check_val_every_n_epoch
        learning_config["trainer"]["max_epochs"] = args.epochs
        print("learning config loaded from", args.learning_config)
        print("batch size set to", args.batch_size)
        print("check validation every n epochs set to", args.check_val_every_n_epoch)
        print("max epochs set to", args.epochs)
    with open(args.data_config, "r") as fp:
        data_config = _json.load(fp)
        # only keep the first 300 train data points
        data_config["train"]["outputs"] = data_config["train"]["outputs"][:300]
        # drop data_config.train/validation.outputs.for_reference
        for s in ["train", "validation"]:
            data_config[s]["outputs"] = [
                {k: v for k, v in output.items() if k != "for_reference"}
                for output in data_config[s]["outputs"]
            ]
        data_config["validation"]["outputs"] = data_config["validation"]["outputs"][:args.max_val_datapoints]
        data_config["train"]["ny"] = args.ny
        if model_config["net"]["name"] == "LSTM":
            data_config["validation"]["ny"] = args.ny
        print("data config loaded from", args.data_config)
        print("validation set size limited to", args.max_val_datapoints, "data points")

    # Train
    best_ckpt_path = _Path(_full.main(data_config, model_config, learning_config, outdir, threshold_esr=args.threshold_esr))

    # Test
    if not args.get_test_set_metrics:
        print("Skipping test set metrics computation.")
        exit(0)
    print("Computing test set metrics...")
    from nam.testing_framework import test_model
    test_model(
        data_config_path=args.test_data_config,
        model_config_path=args.model_config,
        ckpt_path=best_ckpt_path,
        metrics_path=outdir / "metrics.json",
        plot_path=outdir / "plot.png",
        batch_size=args.test_batch_size
    )
