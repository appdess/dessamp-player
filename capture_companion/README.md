# DessAmp capture companion

This is DessAmp's persistent local capture/training companion. The inherited
Python modules and compatibility console commands still use the
`dessmetal_capture` and `dessmetal_player` namespaces; that internal naming is
not the current product boundary and should be migrated only as a dedicated,
tested compatibility change.

This directory contains two local-first standalone development applications for
conditioned NAM/A2 models. The Trainer owns capture projects and conditioned
training jobs. The Player opens the Trainer's portable artifact and exposes the
exact learned controls. The two standalone applications remain separate from
each other; DessAmp APP/AUv2/VST3 is an additional consumer of
their durable boundary: one self-contained `.dess` amp model. The package
embeds the byte-exact NAM payload, its control schema, provenance, and hashes;
the older `player-manifest.json + model.nam` directory remains a development
migration format.

The first vertical slice supports:

- persistent amp/model projects in a local SQLite database;
- arbitrary continuous knob schemas and finite candidate grids;
- persistent capture recommendations using a deterministic coverage baseline;
- persistent import of PANAMA `new_points.json` batches with explicit knob
  ordering and disagreement provenance;
- direct duplex playback/recording through PortAudio-compatible interfaces;
- Logic or other DAW capture import;
- optional replay of externally supplied legacy Gain-stage recordings for
  workflow testing (those assets are not part of this repository);
- frozen, restart-safe local PANAMA WaveNet jobs on Apple MPS or CPU, including
  ordered `g_vector` values, source/config/license hashes, explicit alignment,
  a held-out control setting, and functional-only run labeling;
- checkpoint export and packaging into a self-contained `.dess` amp model;
- validation through the same hardened NAM Core loader used by DessAmp;
- a separate manifest-driven Player with arbitrary learned controls, canonical
  WAV input, and native NAM Core offline rendering;
- a DessAmp DAW/standalone player with eight append-only automation
  slots for conditioned models and snapshot/A2 compatibility.

The coverage strategy is an integration baseline, not PANAMA's ensemble-
disagreement algorithm. The Trainer can run the ordinary conditioned PANAMA
WaveNet locally. A development MPS/CPU ensemble worker now lives in
`trainer_source/local_active_learner.py`; it trains independent members
sequentially, scores the project's finite knob grid by output disagreement, and
emits `new_points.json` for the persistent queue. It is not yet launched or
resumed by the Trainer UI. The original PANAMA multi-GPU worker remains usable
on CUDA infrastructure.
See the [architecture decision](../docs/architecture.md) for the runtime
boundaries and the [current conditioned-A2
experiment](../docs/experiments/2026-08-20-fortin-a2-full-ensemble.md) for exact
evidence.
For a new physical project, use the repository
[training runbook](../docs/training-runbook.md) and its explicit project
initializer rather than creating a high-dimensional Cartesian grid in the UI.

## Development setup

Build the Core loader and renderer from the repository root:

```bash
cmake -S NeuralAmpModelerCore -B /tmp/dessamp-core-build -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/dessamp-core-build --target loadmodel render_model -j 8
```

The Python environment needs the web/audio dependencies from `pyproject.toml`.
For the retained local PANAMA trainer, reproduce the separate external
environment from `trainer_source/uv.lock` as documented in
[`trainer_source/README.md`](../trainer_source/README.md). The Conda files under
`trainer_source/environments/` are legacy convenience alternatives, not the
reproducibility authority. The current macOS development path supports Apple
MPS; create a `local-cpu` job through the API when MPS is unavailable.

## Run the standalone Trainer

From the repository root:

```bash
PYTHONPATH=capture_companion/src \
python3 -m dessmetal_capture \
  --data-dir /absolute/persistent/path/dessamp-trainer \
  --loader /tmp/dessamp-core-build/tools/loadmodel \
  --trainer-root trainer_source \
  --demo-root "/path/to/01-NAM-Capture/DessMetal"
```

Then open the loopback URL printed by the command. The server binds to
`127.0.0.1` only and protects state-changing API calls with a per-launch token
that is injected into its own UI.

`--demo-root` is optional and legacy-only. When the externally held fixture
model, input, config, and matching WAVs are all present, the UI can replay them
instead of operating audio hardware. Those product assets are intentionally
excluded from DessAmp. `dessmetal-trainer` is an equivalent compatibility
console command;
`dessmetal-capture` remains as a compatibility alias.

The five-second profile deliberately proves only the plumbing. It clips a
short excerpt, allows the known dataset-length warning under an explicit
truncate policy, trains one epoch, exports, validates, and labels the artifact
`workflow_validation_only`. A normal job uses the complete recordings, rejects
capture warnings, and holds the nearest-to-centre captured setting out of the
training split. Neither profile is permission to replace a shipped model
without listening tests and calibration evidence.

## Run the standalone Player

After a training job completes, download the `.dess` amp shown in the Trainer
and run:

```bash
PYTHONPATH=capture_companion/src \
python3 -m dessmetal_player \
  /path/to/Fortin-Nameless.dess \
  --renderer /tmp/dessamp-core-build/tools/render_model \
  --input /path/to/short-mono-48000.wav
```

The Player verifies the package length, manifest and model SHA-256 values, and
manifest/model agreement before it starts.
Its controls are generated from the manifest, not hard-coded to Gain. Releasing
a knob renders the loaded mono WAV with the exact ordered parameter vector and
updates the audio preview. This first slice is an offline audition path; a
real-time audio-interface callback and native signed app packaging are still
future work.

## Run the development DAW Player

The current DessAmp standalone app, AUv2, and VST3 open the `.dess` amp and
present its active learned controls. The current DAW surface supports up to
eight controls; the Fortin validation artifact uses seven. Legacy artifact
directories and `player-manifest.json` remain accepted during migration.

This is an unreleased development path. The local arm64 APP/AU/VST3 builds have
passed Core, state, real-AU render/reopen/automation, strict/stress `auval`,
offline-render, and official VST3 validation. Universal packaging, a Logic
listening session, production-quality multi-knob training, and signing remain
separate gates.

## Capture paths

### Direct interface

The Trainer plays the project's standardized input WAV through a selected
output device and records one mono return channel from a selected input device.
It writes exact-length 24-bit PCM at the input signal's sample rate. Starting a
capture is always a visible, explicit user action.

### Logic/manual import

The Trainer shows the recommended control settings. Capture the standardized
input through the rig in Logic, bounce the mono return, and import that WAV
against the recommendation. Format, length, peak, and hash evidence is retained
with the capture.

### PANAMA learner bridge

Import the learner's `new_points.json`, enter its round, and confirm the exact
control order used by every `g_vector`. The Trainer hashes and stores the batch,
rejects unsafe or duplicate points, and promotes each suggestion through the
ordinary capture workflow. It does not start the PANAMA ensemble learner or
claim that the coverage baseline is active learning. This is separate from the
ordinary conditioned model-training job that the Trainer can run locally.

For development on a single Apple GPU, `trainer_source/local_active_learner.py`
can now produce the imported file with a real PANAMA ensemble-disagreement
score. The official upstream learner is CUDA-only and expects at least one GPU
per ensemble member plus another optimization GPU. The local adapter uses the
same uncertainty objective but executes members sequentially and selects from
the declared physical knob grid.

### Replay dataset

The development-only replay adapter maps recommended Gain positions to
externally supplied legacy reamps. It can validate capture persistence without
making sound and can then feed the explicitly labelled functional training
profile. It never touches cloud resources, and its missing product assets make
that one optional test skip in this repository.

## Tests

```bash
PYTHONPATH=capture_companion/src \
python3 -m unittest discover -s capture_companion/tests -v
```

Tests use temporary databases and generated WAVs. They never access an audio
device.
