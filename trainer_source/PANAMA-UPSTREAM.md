# PANAMA: PArametric Neural Amp Modeling with Active learning

This is the implementation of our parametric amp modelling framework. Our code is based on [NAM](https://www.neuralampmodeler.com). Demos can be found [here](https://eth-disco.github.io/PANAMA/).

## Runtime scope

This fork creates DessMetal's continuous models by feeding ordered amp-knob
values through `global_condition_size` (or an LSTM conditioning vector).
PANAMA disagreement chooses capture settings and remains independent from the
final inference architecture.

`trainer_source.conditioned_a2` is the isolated conditioned-A2 experiment. It
uses the official `neural-amp-modeler==0.13.0` child recipe, adds a block-
constant global vector to A2 Lite and Full, and exports models for the matching
global-parameter-aware Core fast path. It may reuse the same labeled WAV corpus
and frozen hidden split, but it always retrains new weights; it is not a
conversion of an existing `.nam` file. The current desktop gate trains Full
only. Lite remains available for a later embedded target.

See the current [DessAmp architecture decision](../docs/architecture.md),
[training runbook](../docs/training-runbook.md), and
[player manifest contract](../docs/player-manifest-v1.md).

## Setup

The reviewed default for this retained trainer is the separate external `uv`
environment created from `trainer_source/uv.lock`; see
[`README.md`](README.md). The files under `trainer_source/environments/` remain
legacy Conda convenience alternatives for NVIDIA/CUDA and macOS inspection,
but they are not the reproducibility authority. The default learning template
still targets `gpu`; the standalone Trainer copies that template into each job
and explicitly freezes `accelerator: mps` or `accelerator: cpu`, one local
device, and zero data-loader workers. It does not silently edit the checked-in
template.

For the persistent capture-to-player workflow, run the standalone Trainer in
[`capture_companion/`](../capture_companion/README.md). It derives the global
condition size from the project's ordered controls, checks that the recording
length can satisfy the WaveNet receptive field and chosen chunk size, holds out
one captured setting when at least three are available, records source and
license hashes, runs `custom_train_full.py`, exports the best checkpoint, and
packages the result for the separate Player. Its five-second one-epoch option
is explicitly workflow validation only and must not be used as model-quality
evidence.

Conditioned exports use a tiered external-audit contract. Every hidden setting
must remain below ESR `0.035` to pass the normal NAM **Not bad** numerical gate.
The workflow still aims for mean ESR `<= 0.01` with p95/worst `<= 0.02`, and
prefers mean ESR `<= 0.005`. Capture repeatability remains independently fixed
at ESR `<= 0.01`; do not loosen the recording path because a model qualifies for
the normal band. Numerical acceptance advances a model to host and listening
gates—it is not by itself a release or physical-amplifier claim.

## DessMetal single-parameter training

Run the following commands from the repository root. The checked-in
`trainer_source/parametric_wavenet.json` uses a WaveNet with
`global_condition_size = 1`, so each training point supplies exactly one
normalized value in `g_vector`.

The historical upstream workflow expected a 190-second mono, 48 kHz, 24-bit
PCM file at `trainer_source/training_data/input.wav`. That audio is
intentionally not redistributed in this repository. A new DessAmp project
supplies and hash-binds its owned standardized input through the generic
initializer. The historical DessMetal data configs used these mappings:

- Amp gain captures use `g_vector` values from `0.2` through `0.8`, matching
  plug-in Gain values 2 through 8.
- TS9 and aesahaettr tone captures use values from `0.0` through `1.0`,
  matching the plug-in's 0 through 10 control range.

The checked-in historical `parametric_data_*.json` paths are relative to the
repository root, but neither their original shared input nor commercial
re-amped capture WAVs are included. They are provenance examples, not runnable
new-project inputs. Use the generic runbook instead of reconstructing them by
guessing paths.

The cloud helper contains capture-machine calibration defaults. They are job
inputs, not authoritative metadata for the currently shipped NAM files; for
example, the shipped Blue model records a different return level and the
shipped Red model has null calibration fields. Confirm the actual send/return
chain for the restored capture before recording or exporting a replacement,
then verify the exported NAM metadata explicitly.

Five model-specific configs are currently present:

- `parametric_data_dessblock_green.json`
- `parametric_data_dessblock_red.json`
- `parametric_data_sickdess.json`
- `parametric_data_ts9.json`
- `parametric_data_aesahaettr.json`

Train one restored dataset with:

```bash
python3 trainer_source/custom_train_full.py \
  --base-outdir outputs/parametric_training_dessblock_green \
  --data-config trainer_source/parametric_data_dessblock_green.json \
  --model-config trainer_source/parametric_wavenet.json \
  --learning-config trainer_source/default_config_files/learning/default.json \
  --epochs 800 \
  --threshold-esr 0.009 \
  --check-val-every-n-epoch 5
```

`0.009` is the existing project target, not a promise that every capture will
reach that ESR. Training writes a timestamped directory below
`--base-outdir`. Export from that directory into a review location first:

```bash
python3 trainer_source/export_model.py \
  --input-level SEND_DBU \
  --output-level RETURN_DBU \
  outputs/parametric_training_MODEL_NAME/TIMESTAMP \
  outputs/exported/MODEL_NAME
```

The exporter selects the checkpoint with the lowest ESR encoded in its
filename, falling back to the most recently modified checkpoint when the
metric cannot be parsed. Review the exported model and stats before replacing
any shipped resource. After an intentional replacement, rebuild all macOS
formats through the credential-free release gate:

```bash
./package_mac.sh --unsigned
```

For DessAmp's production RunPod workflow, including GCS durability, credential
requirements, recovery, and billing effects, see
[`cloud/README.md`](cloud/README.md).

The upstream PANAMA examples below assume the current directory is
`trainer_source/`; the DessMetal commands above intentionally run from the
repository root.

## Config JSONs

There are three types of config JSONs:

- **Model config**, which defines the model architecture as well as the loss function. Default ones are provided in `default_config_files/models/`. `wavenet-mel-mrstft.json` and `lstm-mel-mrstft.json` are recommended. The global condition size (i.e. number of amp knobs) is by default 6, but you can change it using the JSON keys `global_condition_size` for WaveNet and `conditioning_dim` for LSTM.
- **Learning config**, which provides the batch size, number of epochs and other training hyperparameters. `default_config_files/learning/default.json` shows the default learning config. We observed that in general, a larger batch size works better with a larger dataset, and vice versa.
- **Data config**, which supplies the paths of audio files in a dataset. The
  dataset format requires a fixed input signal and wet signals with different
  amp settings. The upstream PANAMA distribution provided a mini example in
  `default_config_files/data/skeleton.json` with two training points and one
  validation point. The JSON key `ny` is the length of each batch clip. Its
  historical `training_data/input.wav` is intentionally not redistributed by
  DessAmp; new projects supply and hash-bind owned input audio through the
  generic runbook.

You can always use `--help` to see all command-line options.

## Inference

Inference can be done with `inference_w_ckpt.py`. Use a checkpoint produced by
the current trainer together with its matching model config. The tracked
`demo_ckpt.ckpt` is retained as a legacy deserialization fixture, but its layer
layout does not match the current model configs.

Checkpoint files are deserialized in PyTorch's restricted `weights_only=True`
mode and must contain a non-empty tensor `state_dict` plus a positive sample
rate. Framework-managed resume options that bypass this loader are rejected,
and the utilities do not fall back to pickle-capable loading. PyTorch 2.6.0 or
newer is required because older releases are affected by
[CVE-2025-32434](https://github.com/advisories/GHSA-53q9-r3pm-6pq6), including
when `weights_only=True` is requested; older or unverifiable versions fail
closed. Treat model weights as untrusted data nonetheless, and separately
verify a checkpoint's SHA-256 against a trusted source before using it when
provenance matters.

Usage example:

```bash
python3 inference_w_ckpt.py --ckpt-path "path/to/current-model.ckpt" --input-path "my_input.wav" --g-vector 0.5 0.2 0.5 0.7 0.5 0.8 --output-dir .
```

## Single Model Training

Training is done with `custom_train_full.py`. The results will be in a subdirectory inside of `--base-outdir`.

Usage example:

```bash
python3 custom_train_full.py --base-outdir "outputs" --data-config "my_dataset.json" --model-config "default_config_files/models/wavenet-mel-mrstft.json" --learning-config "default_config_files/learning/default.json"
```

## Active Learning

`active_learner_multi_gpu.py` performs one round of our active learning method. We recommend making an output directory specifically for active learning. This ***same*** directory, for example `my_active_learning_folder`, should be supplied each round along with the current round number. Round indices start at 0. The 0-th round needs a starter dataset supplied in `--starter-data-config-path`; starting from round 1 this is no longer needed. 10 random datapoints should work fine for the starter. The signal x used during optimization of cross-model disagreement with regard to g is supplied via `--x-path-for-g-opt`. By default, the ensemble size is 4, but can be adjusted via `--ensemble-size`. Make sure you have a multi-GPU setup, and that the number of GPUs is at least ensemble size + 1.

When the script terminates, it logs the g-vectors of the new datapoints to be gathered. After recording and supplying the new audio clips in `active_learning_inputs`, rerun it with the ***same*** output directory and the new round index. Rinse and repeat.

Usage example:

```bash
python3 active_learner_multi_gpu.py --starter-data-config-path "my_starter_dataset.json" --current-round-idx 0 --x-path-for-g-opt "my_input_for_g_optimization.wav" --output-dir "my_active_learning_folder" --ensemble-size 4 --model-config-path "default_config_files/models/lstm-mel-mrstft.json" --learning-config-path "default_config_files/learning/active_learning_LSTM.json"
# Record new datapoints
python3 active_learner_multi_gpu.py --current-round-idx 1 --x-path-for-g-opt "my_input_for_g_optimization.wav" --output-dir "my_active_learning_folder" --ensemble-size 4 --model-config-path "default_config_files/models/lstm-mel-mrstft.json" --learning-config-path "default_config_files/learning/active_learning_LSTM.json"
# Record new datapoints, then repeat with the next round index.
```

### Single-device MPS/CPU adapter

The upstream command above is hard-coded for CUDA and requires at least
`ensemble_size + 1` GPUs. DessMetal's `local_active_learner.py` preserves the
conditioned-model ensemble-disagreement score while training ensemble members
sequentially on one Apple MPS or CPU device. It ranks an explicit finite knob
grid rather than performing continuous gradient optimization, then writes a
Trainer-compatible `new_points.json` including the disagreement scores.

Deterministic validation, run IDs, hidden metrics, and limitations belong in
internal `docs/experiments/` evidence, which is deliberately excluded from the
staged public tree. This adapter is currently a development worker, not a
background job launched by the standalone Trainer UI.

### Fortin Nameless seven-control capture harness

`run_fortin_conditioned_active_learning.py` is the real-plug-in development
harness for the complete continuous Nameless amp section: Master, Presence,
Bass, Middle, Treble, Gain I, and Gain II. It freezes High input, Pull Gain
off, Grind at 35%, a flat graphic EQ, and all cabinet/time effects off. The
first phase creates a persistent Trainer project, asks its coverage strategy
for ten starters, renders the full standard NAM input through the AU, keeps
the unity float masters, applies one measured safety trim to every capture,
derives PCM24 without further gain, and runs the official NAM validator.
Before any starter is accepted, it also renders the low and high endpoint of
every declared control on a short independent DI and refuses the run if any
control is audio-inactive. This is necessary because AU indexed parameters do
not necessarily encode their first label as `1`: Fortin publishes Amp Input as
`0 = High`, `1 = Low`.

The second phase trains a sequential ensemble, imports its highest-disagreement
point through the ordinary `new_points.json` boundary, captures only that
point, retrains, evaluates both ensembles on a separate DI, and packages the
result for the Player. It refuses to overwrite an existing run directory.
Build the Core loader and renderer as described in
[`capture_companion/README.md`](../capture_companion/README.md), then use:

```bash
PYTHONPATH=capture_companion/src \
python3 trainer_source/run_fortin_conditioned_active_learning.py prepare \
  --run-root /path/to/new-fortin-run \
  --renderer-binary /path/to/verified/render_fortin_au \
  --renderer-source /path/to/matching/render_fortin_au.swift \
  --delay-samples 0

PYTHONPATH=capture_companion/src:trainer_source \
python3 trainer_source/run_fortin_conditioned_active_learning.py reserve-evaluation \
  --run-root /path/to/new-fortin-run \
  --hidden-count 8 \
  --minimum-distance 0.35

PYTHONPATH=capture_companion/src \
python3 trainer_source/run_fortin_conditioned_active_learning.py train \
  --run-root /path/to/new-fortin-run \
  --backend mps \
  --loader /tmp/dessmetal-core-build/tools/loadmodel \
  --manifest-loader /tmp/dessmetal-core-build/tools/loadmodel_manifest \
  --renderer /tmp/dessmetal-core-build/tools/render_model
```

After a cloud/native ensemble emits an audit-safe disagreement report, capture
a separated batch through the same persistent queue:

```bash
PYTHONPATH=capture_companion/src:trainer_source \
python3 trainer_source/run_fortin_conditioned_active_learning.py capture-batch \
  --run-root /path/to/existing-fortin-run \
  --selection-report /path/to/disagreement-audit-safe.json \
  --round-index 2 \
  --maximum 8 \
  --minimum-distance 0.35
```

This subcommand hash-binds the current candidate state, verifies that captured
and frozen audit vectors were excluded before scoring, backs up SQLite through
its backup API, and refuses to replace existing round evidence.

`reserve-evaluation` freezes and renders the separate-DI audit settings before
cloud training. Its hash-bound references are unavailable to acquisition.
Do not use the legacy 2026-08-18 Low-input captures as evidence for a true-High
seven-control model: an inverted indexed-value mapping left Gain I inactive in
that corpus. The workflow/player plumbing from that run remains useful, but its
model-quality curve is mode-specific and not transferable to true High input.

Materialize the next immutable full-corpus config from the persistent database
instead of maintaining capture paths by hand:

```bash
PYTHONPATH=capture_companion/src \
python3 trainer_source/materialize_conditioned_data_config.py \
  --run-root /path/to/existing-fortin-run \
  --output /path/to/new-full-data-config.json
```

The command requires exactly one accepted `starter-00` validation WAV, excludes
failed captures, preserves the fixed control order, and refuses duplicate
accepted vectors or missing audio.

Select and package exactly one externally evaluated player member—not the
arithmetic ensemble—with `package_conditioned_export.py`. The command binds the
model SHA-256 to the held-out report, performs another database backup, restores
the project's control defaults, and validates the self-contained `.dess`
through current NAM Core. A member that misses the NAM-normal numerical gate is
refused unless `--allow-failed-gate` is explicitly supplied; that path is
permanently labelled `workflow_validation_only` for listening and must not
replace a shipped model. The package evidence separately records whether the
member also reached the stricter project target and preferred tier.

After at least two like-for-like external audit rounds, calculate a diagnostic
capture-count curve with `analyze_conditioned_learning_curve.py`:

```bash
python3 trainer_source/analyze_conditioned_learning_curve.py \
  --point 10=/path/to/round-00-held-out.json \
  --point 11=/path/to/round-01-held-out.json \
  --point 19=/path/to/round-02-held-out.json \
  --preferred-mean-esr 0.005 \
  --output /path/to/learning-curve.json
```

The report keeps ensemble acquisition metrics separate from the best single
player member. Its log-log projections are planning diagnostics, especially
with only two or three points; they are not a physical-amplifier capture-count
guarantee.

`--delay-samples` follows the NAM Dataset convention and is frozen into the
project state. Zero is correct for the validated offline Fortin plug-in path.
Measure a physical send/return chain and supply its actual signed value; do not
copy an official-validator recommendation or a previous interface's latency.

The default 15-second training view is deliberately a functional integration
profile; it does not turn the retained 190-second captures into a release-
quality model claim. Use a fresh run and an explicitly reviewed full-data
profile after the production accelerator and stopping criteria are available.

## Testing

Test a model with `test_model.py`. This (ideally) requires a separate data config from the training one. The config should still be the same JSON format, but only the `validation` split will be used. The result metrics will be stored as a JSON file and visualized as a plot.

Usage example:

```bash
python3 test_model.py --data-config-path "my_test_data_config.json" --ckpt-path "model_to_be_tested.ckpt" --metrics-path "results.json" --plot-path "results.png"
```
