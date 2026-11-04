# DessAmp trainer source

This directory contains the acquisition, conditioned-model training, cloud
transport, evaluation, and packaging code used by DessAmp. The inherited
PANAMA/NAM implementation and its original usage notes are retained in
[`PANAMA-UPSTREAM.md`](PANAMA-UPSTREAM.md).

## Train your first knob-controlled amp

With the repo open, ask “Help me capture my amp for DessAmp in Logic Pro, one
setting at a time.” The [capture skill](../.agents/skills/dessamp-conditioned-training/SKILL.md)
walks you from your amp and load box to **CUSTOM → open the .dess**.

1. **Pick the sound you need.** Choose one channel, one or two controls and
   their useful physical ranges. Freeze the remaining controls and switches.
   For Gain 3–7, the example maps model values 0, 0.5 and 1 to Gain 3, 5
   and 7. Its nine candidate settings are spaced half a dial marking apart.
   Its four audit positions lie between them. Adjust both files to marks you
   can reproduce on your amp. Copy them outside Git, replace the rig settings
   and calibration with measurements, then use the runbook's initializer and
   reservation commands. [Gain-only project](examples/gain-3-to-7-project.json) ·
   [Matching audit template](examples/gain-3-to-7-evaluation.json) ·
   [Initialize and launch](../docs/training-runbook.md#3-initialize-a-persistent-capture-project)
2. **Record pairs.** Play the verified, 190-second NAM v3.0.0 `input.wav`
   through the amp at each recommended setting. The training example is the
   dry file, the raw mono return and the knobs held fixed for that take. Prove
   the routing, repeat the centre and check that each learned control changes
   the sound. Keep the same trim and measured alignment throughout the project.
   [Sweep download and hash](../docs/physical-amp-capture.md#2-get-the-training-sweep-and-check-the-signal-path) ·
   [Logic Pro track setup](../docs/physical-amp-capture.md#logic-pro-example)
3. **Let the recordings guide the next take.** The Trainer starts with
   `coverage-v1`, first near the centre and then in the gaps. At a checkpoint,
   train independent models on the same accepted captures. Play a separate
   acquisition DI through them, score where they disagree, and import the next
   batch from the remaining candidate settings. The owner sees one pending
   physical setting at a time. Reserve hidden knob positions before the first
   batch and keep evaluation audio out of the scorer. [Active-round commands](../docs/training-runbook.md#10-select-and-queue-the-next-active-captures)
4. **Train and select.** Materialize the corpus from the project's SQLite
   records, hold out one capture for internal validation, and freeze the
   ordered controls. For the RunPod path, confirm the GPU, rate, runtime
   limit and budget before upload or submission. Each GPU trains one A2 Full
   member from its own seed. Inputs, the best checkpoint, recovery checkpoints
   and exports live in GCS. Choose a single member on internal validation,
   before opening the frozen audit. [Corpus and GPU commands](../docs/training-runbook.md#6-materialize-the-authoritative-training-corpus)
5. **Check the amp in the player.** Render the chosen member against real
   withheld settings on a different dry performance. Check the worst ESR and
   listen for the direction and strength of each knob sweep. If the run passes,
   package its single model as `.dess`, open it via **CUSTOM**, and verify a
   saved session reopens with the same sound. [Evaluation and export](../docs/training-runbook.md#9-evaluate-the-frozen-hidden-settings)

## What the Fireball run took

The August 2026 physical Fireball lead run gives a measured example of the
work involved. The figures below describe its final production round.

| | Recorded result |
|---|---|
| Playable range | Lead Gain 9 o’clock–3 o’clock, Bass/Middle/Treble detents 4–9, Presence 8 o’clock–2 o’clock. Bright on, Bottom and Mid Boost off. Lead and master volumes fixed. |
| Recording | Logic Pro played NAM v3 through MOTU M2 Output 1 and a Palmer DACCAPO. The raw return came from a Two Notes Torpedo Captor into M2 Input 2. |
| Corpus | 46 training settings plus one fixed internal validation setting. At 190 seconds each, those 47 full takes alone represent **2 h 28 min 50 s** of playback. Setup, repeats, rejected takes and external audit recordings are additional. |
| Compute | Four secure on-demand NVIDIA A40s on RunPod, one for each seed 98–101. A2 Full, 60 epochs per member, batch size 16, training windows of 8,192 samples. GCS held the frozen bundle, checkpoints and final models. |
| Elapsed time | About **10 h 54 min** from creation of the run ledger (28 August, 19:05:50 UTC) through final verification of all four members (29 August, 05:59:32 UTC). This includes provisioning and verification. |
| Compute cost | Historical rate **$0.44 per GPU-hour**, or **$1.76/hour** for four. The ledger projected **$15–18** for this round and set a **$21.12** compute ceiling. The ledger contains no itemized final invoice or total for the earlier rounds. |
| Selection | Seed **100** had the best internal validation result at epoch index 59. Its five reserved physical settings scored mean ESR **0.0220**, worst **0.0305**. All passed the 0.035 gate. The stricter mean 0.01 and worst 0.02 targets remained unmet. |
| Scope of that result | The five audit positions were new, but used the same dry NAM recording as training. The single selected member was packaged for private integration. For new projects, use the independent-performance audit above. |

The private evidence can be located without copying audio, credentials or
weights into Git. Run ID `engl-fireball-r2-prod07-epoch-aligned47-a40-20260828`.
Within that run, `cloud-execution-plan.json` records the rate, budget and
member timestamps, `internal-member-selection.json` records selection, and
`QUALITY-REPORT.md` summarizes the physical and numerical checks. The package
decision is in `06_PACKAGE/fire-package-report-v2.json`.

## Supported paths

- `conditioned_a2/`: A2 Lite/Full training with a block-constant global
  condition vector. DessAmp currently deploys **A2 Full**.
- `cloud/conditioned_a2_runpod_job.py`: prepares an immutable GCS bundle, then
  submits or resumes one digest-pinned RunPod member. The launcher requires
  on-demand capacity. Secure A40 is the proven starting point. GCS holds
  the input bundle, member contracts, checkpoints, and exported models.
- `cloud/conditioned_a2_vertex_job.py`: legacy compatibility helper for
  historical Vertex runs. Do not use it for new production training.
- `cloud/conditioned_a2_entrypoint.py`: downloads the immutable bundle,
  resumes the same member when a durable checkpoint exists, exports the best
  model, and mirrors best/resume checkpoints after each validation epoch.
- `cloud/local_test_env/`: portable Python 3.12 project and lock for the
  official NAM conditioned-A2, capture, cloud-helper, and player/resource test
  surfaces. Materialize it outside the repository and keep it separate from
  the retained PANAMA trainer environment.
- `evaluate_conditioned_exports.py`: native external holdout evaluation.
- `score_conditioned_exports.py`: target-free ensemble disagreement over only
  uncaptured, non-audit candidates.
- `materialize_conditioned_data_config.py`: derives one authoritative training
  config from the persistent capture database using a stable validation ID.
- `init_conditioned_project.py`: creates a generic evidence-bearing project
  from an explicit control/candidate schema.
- `create_evaluation_reservation.py`: freezes hidden vectors, unseen DI, and
  target hashes before training.
- `materialize_teacher_distillation_terminal_audit_spec.py`: binds an
  internal-only member-selection report, current renderer, frozen unseen DI,
  and already-rendered teacher targets into a leakage-safe external-audit spec.
  It does not render targets or inspect/select a model.
- `materialize_conditioned_candidate_config.py` and
  `import_conditioned_suggestions.py`: target-free generic disagreement and
  transactionally backed-up queue import.
- `validate_capture_pair.py`: repeatability and low/high sensitivity evidence.
- `run_fortin_conditioned_active_learning.py`: owned amp-sim workflow harness
  used to validate the full capture/acquisition loop.
- `package_conditioned_export.py`: packages one selected member into `.dess`
  and records gate evidence.

The previous conditioned WaveNet remains useful as a quality/CPU baseline and
as an acquisition committee option. Its checkpoints do not resume A2 Full:
the topology and optimizer state are different.

## Where the training code lives

For the knob-controlled model, [model.py](conditioned_a2/model.py) adapts the
pinned NAM 0.13.0 A2 architecture to accept audio plus an ordered control
vector. [dataset.py](conditioned_a2/dataset.py) pairs each return with that
vector. [train.py](conditioned_a2/train.py) trains and exports the child. The
[RunPod image](cloud/Dockerfile.conditioned-a2) copies these modules from this
checkout and installs upstream NAM from the locked dependency list. The
retained PANAMA code under `nam/` uses its own environment. The older
standalone LSTM/ONNX prototype is a separate model format.

### Optional fixed-setting reference

The two early Fireball snapshots used a separate local trainer. Its reusable
wrapper now lives in [fixed_a2/train_fixed_nam_a2.py](fixed_a2/train_fixed_nam_a2.py).
It derives from the Fireball baseline script of the same name (source SHA-256
`c51dcd05f82a57f0d4d9dfc12d2854c7bfaa07a75e7e638a00354cf42ea9ec13`).
Use it for a single fixed setting when you want a listening reference. It
exports a `.nam`. For playable controls and `.dess`, follow the
[conditioned workflow](../docs/physical-amp-capture.md).

From the repo root, use the official NAM environment from runbook §2, an
already validated head-only NAM input/return pair, and fresh directories.

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/fixed_a2/train_fixed_nam_a2.py \
  --input="$DESSAMP_STIMULUS" --output=/absolute/path/to/fixed-return.wav \
  --work-dir=/absolute/new/path/work --export-dir=/absolute/new/path/export \
  --name=Lead-reference --gear-make='Your make' --gear-model='Your amp' \
  --capture-metadata=/absolute/path/to/capture-calibration.json
```

This wrapper uses the original packed `channels_8` checkpoint floor (epoch
index 200), ESR ceiling 0.005, and maximum 500 epochs. The repository version
stops at a qualifying validation callback and exports only after checking that
tier. These gates belong to the fixed baseline. The conditioned example uses
the separate held-out settings and release criteria in the runbook.

## Current Nameless validation contract

The current Nameless model receives seven normalized values in this exact order:

```text
amp_mvc
amp_presence
amp_bass
amp_middle
amp_treble
amp_gain_1
amp_gain_2
```

The `.dess` manifest preserves these IDs and model indices. A UI may reorder
the controls but must never reorder the vector passed to inference.

This list is not the generic product schema. A new amplifier declares one to
twelve normalized model controls in its persistent project. The materialized
config, immutable cloud manifest, A2 entrypoint, export, `.dess` package, and
runtime must all agree on that exact order. Conditioned A2 uses a 6,347-sample
receptive field; new exports declare it in metadata.

## Checkpoint contract

Every conditioned-A2 validation epoch mirrors two files below the member's
result prefix:

- `durable/best.ckpt`: the lowest internal validation score so far;
- `durable/resume.ckpt`: the latest model, epoch, global step, optimizer, and
  learning-rate scheduler.

Continuation preserves the same seed and member identity. Starting another
seed from a previous member's checkpoint is not an independent ensemble member
and must not be used for disagreement.

## Quality contract

Internal validation drives optimization only. Final evaluation is external,
native, and frozen:

- every held-out setting below ESR `0.035` to enter the normal NAM acceptance
  band;
- mean ESR `<= 0.01` and p95/worst `<= 0.02` as the project target;
- mean ESR `<= 0.005` preferred, without a worse corner;
- finite/non-silent output, deterministic recall, host validation, and
  level-matched listening are separate mandatory gates.

An artifact that misses the normal gate may be packaged only with the explicit
failed-gate path and remains `workflow_validation_only`.

## Reproducible retained trainer environment

`uv.lock` is the reviewed dependency lock for the retained PANAMA/NAM trainer
package in this directory. Keep this environment outside the repository and
separate from the official NAM 0.13 conditioned-A2 environment documented in
the runbook: both distributions intentionally use the top-level `nam` import
name and must not be installed together.

```bash
export DESSAMP_PYTHON=/absolute/path/to/python3.12
export DESSAMP_TRAINER_ENV=/tmp/dessamp-trainer-venv

UV_PROJECT_ENVIRONMENT="$DESSAMP_TRAINER_ENV" \
uv sync --project trainer_source --locked --extra test \
  --python "$DESSAMP_PYTHON"
```

The lock binds the adjacent `capture_companion` project through a relative
editable source. It contains no workstation-specific absolute path, so the
same repository layout can reproduce it in a clean checkout.

Build release-neutral Python artifacts outside the source tree:

```bash
uv build --project trainer_source --out-dir /tmp/dessamp-trainer-dist
```

The wheel must import from an otherwise empty environment, expose its packaged
`nam.models._resources/loudness_input.wav`, and declare every import-time
dependency. A source-tree import is not wheel-install evidence.

Generate a default-runtime CycloneDX inventory from the same lock for release
review. The compatibility-only Transformers extra is deliberately excluded:

```bash
uv export --preview-features sbom-export --project trainer_source --locked \
  --format cyclonedx1.5 --no-dev --no-extra transformers-compat \
  --output-file /tmp/dessamp-trainer-runtime.cdx.json
```

Hash and archive the exact release SBOM, then inspect license metadata and
retained license texts for the resolved platform environment. A CycloneDX
dependency graph is not by itself a license approval.

## Tests

From the repository root, first export `DESSAMP_LOCAL_PYTHON` using the locked
setup in [`cloud/README.md`](cloud/README.md), and export the separately locked
`DESSAMP_TRAINER_ENV` described above:

```bash
"$DESSAMP_LOCAL_PYTHON" -m unittest discover \
  -s trainer_source/cloud/tests \
  -p 'test_*.py' -v

PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m pytest \
  trainer_source/tests/test_conditioned_a2.py \
  trainer_source/tests/test_import_teacher_distillation_plan.py \
  trainer_source/tests/test_materialize_conditioned_data_config.py -q

"$DESSAMP_TRAINER_ENV/bin/python" -m pytest trainer_source/tests -q \
  --ignore=trainer_source/tests/test_conditioned_a2.py \
  --ignore=trainer_source/tests/test_import_teacher_distillation_plan.py \
  --ignore=trainer_source/tests/test_materialize_conditioned_data_config.py
```

Keep those pytest invocations in separate environments and keep
`trainer_source` out of `PYTHONPATH`. The retained package and the official NAM
0.13 dependency both provide top-level `nam`; combining them can produce
module-order false failures or silently test the wrong implementation.

For setup, submission, cost, and recovery details, see
[`cloud/README.md`](cloud/README.md).
