---
name: dessamp-conditioned-training
description: Guide someone capturing an amp for the DessAmp plugin, from the NAM training sweep and one recommended knob setting at a time to a playable .dess file. Use for physical amps or controllable amp sims, active-learning rounds, conditioned A2 Full training, RunPod/GCS recovery, held-out evaluation, and export. Route standalone fixed-setting NAM snapshots to nam-a2-capture.
---

# DessAmp Conditioned Training

Treat the capture project, training job, and player artifact as three separate
durable boundaries. Never call a plumbing run a production model.

## Route the task

- For a conditioned amp or amp-sim project, follow this skill.
- For a fixed-setting head-only NAM/A2 capture, use `nam-a2-capture` instead.
- PANAMA-style disagreement is an acquisition strategy, not a model
  architecture or amp-control axis. Keep acquisition and final-model choices
  separate.
- Conditioned A2 Full is a supported final-model candidate for the desktop
   player. Compare it with the conditioned WaveNet baseline using the same frozen
  hidden settings and listening material. Defer A2 Lite unless measured target
  hardware constraints require it.

## Guide an owner through a session

- **Start at their question.** For “which sweep?” answer with the full NAM v3.0.0 `input.wav` (formerly `v3_0_0.wav`), its 190-second duration and the [download and hash check](../../../docs/physical-amp-capture.md#2-get-the-training-sweep-and-check-the-signal-path). For “walk me through capturing my amp”, ask only for missing details about the channel, load/reamp path, interface send/return, Logic or direct capture, and which controls they actually use. Ask for each control’s lowest, usual and highest playable setting. Freeze the other controls. Map each chosen physical subrange to model values 0–1 and build candidates and holdouts only inside that range. For Logic, walk through the [two-track setup](../../../docs/physical-amp-capture.md#logic-pro-example) using the owner’s real interface channels. A how-to question calls for guidance. Begin recording only when the owner is ready for the take.
- **Set up or resume.** For a new project, verify the stimulus and measured calibration, reserve the hidden settings outside the candidate grid, and initialize the persistent project outside Git. For an existing project, open its data directory and ID, verify the frozen input SHA-256 and inspect accepted captures, the pending recommendation and queued suggestions. Resume the same take. Give the owner the project ID and directory at handoff.
- **Present one physical setting.** Once the routing, centre repeats and low/high checks pass, read the pending recommendation or use **Recommend next capture**. Use its ID, strategy and `display_settings` in the project's declared order. Translate switches using the project's verified physical labels. Give amp-face values, the controls to keep fixed and a cue to reply when ready. For example, if the pending values match, say “Set Lead Gain to 4/10 and Bright to Off. Keep the other controls at the recorded positions. Reply ‘ready’ when set.” Use the pending values even if another project used a similar amp.
- **Record and verify.** With the owner's ready signal, guide one pass of the frozen input through **Logic / WAV** or the verified **Audio interface** route. Follow the [capture protocol](references/capture-protocol.md) for validation, import against that recommendation ID, failure recovery and when setup of the next queued setting can overlap validation. Confirm the accepted take before starting another recording.
- **Run an active round at the starter checkpoint.** Starter recommendations use `coverage-v1`. For disagreement, use [runbook section 10](../../../docs/training-runbook.md#10-select-and-queue-the-next-active-captures) to materialize remaining candidates, score on a separate acquisition DI with at least two independent exports, and import the ranked batch. The Trainer presents each imported point through the same queue. The button uses coverage again if the queue is empty. Pause at an active-round boundary and rescore with updated models before requesting another setting.
- **Explain training in plain language.** Show how the dry file, raw returns and knob settings reach the trainer. For “how did we train it?”, use the [recorded training time, cost and outcome](../../../trainer_source/README.md#what-the-fireball-run-took) and distinguish its measured results from the budget for a new amp. Name RunPod as GPU compute, GCS as checkpoint storage, and the selected single A2 Full member as the plugin model.
- **Finish the handoff.** After independent evaluation and listening, give the owner the selected file at `package.path` and the instruction **CUSTOM → choose the .dess**. On interruption, report the last accepted take and next pending action from the project.

Before operating audio hardware or a DAW, read
[`references/capture-protocol.md`](references/capture-protocol.md) completely.
Before training, evaluating, estimating capture count, or accepting a model,
also read [`references/evaluation.md`](references/evaluation.md) completely.
For executable commands and current repository boundaries, read
[`../../../docs/training-runbook.md`](../../../docs/training-runbook.md)
completely. Do not substitute a remembered DessMetal/Fortin command.

## Establish current evidence

1. Inspect the repository status, running trainers, free disk, available audio
   devices, and existing project/capture manifests. Preserve unrelated work.
2. Record what is observed, what is only planned, and what has passed. Do not
   interrupt another training job or contend for its accelerator.
3. Confirm the amp boundary: head only, one fixed input mode, cab/time effects
   disabled, and every non-conditioned switch or knob frozen and recorded.
   For indexed AU parameters, record the host-published value strings and map
   their numeric values explicitly; do not infer that `true` or `1` means the
   first label. Parameter readback proves storage, not DSP semantics.
4. Start with one or two learned controls for the first real end-to-end proof.
   Expand to all continuous controls only after the evaluation contract passes.

## Create the persistent project

For a new physical amplifier, create the project from a reviewed explicit JSON
schema with `trainer_source/init_conditioned_project.py`. For a first Gain-only
project, use the [bounded project and matching audit template](../../../trainer_source/README.md#train-your-first-knob-controlled-amp).
For Gain plus Bright, use `docs/examples/physical-lead-two-control-project.json`.
Replace the calibration placeholders with measurements before initialization. Keep model ranges in
`[0,1]`; put native 0–10/0–100 values in display ranges. Use an explicit finite
candidate set rather than expanding a high-dimensional Cartesian grid. Keep
every frozen audit vector out of that candidate set so the built-in coverage
strategy cannot recommend it before the ensemble workflow begins.

When the first proof intentionally learns Lead Gain, Bright, and Bottom
together, use `docs/examples/physical-lead-three-control-project.json` plus
`docs/examples/evaluation-reservation-spec-lead-three-control.json`; do not
invent a Cartesian grid or reuse the two-control reservation.

Launch the local Trainer from the repository root with a persistent data
directory and the hardened NAM Core loader. First export
`DESSAMP_LOCAL_PYTHON` from the locked external-environment setup in
`docs/training-runbook.md`:

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m dessmetal_capture \
  --data-dir "/path/to/persistent/trainer-data" \
  --loader "/path/to/core-build/tools/loadmodel" \
  --trainer-root trainer_source
```

`dessmetal_capture` is the retained internal Python namespace. The product,
repository, skill, and new documentation are DessAmp; do not rename the module
as part of an unrelated training run.

Define controls in exact model-vector order. Record native labels, native and
normalized ranges, capture steps, defaults, units, and frozen controls. The
capture grid is discrete; the trained/player knobs remain continuous inside
their declared ranges.

## Capture and learn

1. Freeze the dry stimulus, interface rate, send/return routing, calibration,
   latency/alignment policy, and one shared safety trim before the first useful
   capture. Never peak-normalize individual reamps.
2. Capture a space-filling starter design plus independent repeats/holdouts.
   Use the preliminary budgets in the evaluation reference; counts are a
   stopping budget, not a claim that every amp needs exactly that number.
3. Validate each capture before marking it complete. Persist settings, paths,
   hashes, format, duration, peak, alignment, warnings, and provenance.
   For an established owner-operated manual batch, use the capture protocol's
   two-stage pipeline: hand off the next queued knob setting immediately after
   the fast safety/file gate, run official NAM validation during owner knob
   setup, and forbid the next Logic recording until the preceding capture is
   fully validated and transactionally persisted.
   Before accepting the starter set, render a low/high axis probe for every
   declared conditioned control in the frozen discrete mode. Every axis must
   produce finite, non-identical audio with a measured non-trivial delta. A
   duplicate output at different knob vectors is a hard investigation gate:
   determine whether the control is inactive, the host mapping is wrong, or
   the capture was duplicated before training or cloud upload.
   For a tone-shaping axis, compare level-matched passages and a relevant band-energy
   ratio between low and high takes. Check sample residuals against a centre repeat
   if the endpoints sound identical. Preserve the original common-gain recordings.
   Keep centre-repeat and frozen-audit WAVs outside the training project. Use
   `validate_capture_pair.py` for the repeatability and low/high sensitivity
   evidence; the tool intentionally does not optimize alignment.
4. Prefer a small conditioned LSTM committee for acquisition speed. Rank the
   finite physical knob grid by ensemble output disagreement, persist
   `new_points.json`, then capture only the chosen setting. Never select from
   the frozen audit set.
5. Retrain from scratch at fixed capture-count checkpoints and score the same
   independent holdout settings and unseen DI every time. Train the selected
   deployable architecture rather than assuming the acquisition committee is
   the player model. For the current desktop path, target conditioned A2 Full
   and retain the accepted conditioned WaveNet as the quality/CPU baseline.
6. Stop only when NAM-normal error acceptance, repeatability,
   curve-flattening, runtime, and human listening gates in the evaluation
   reference all pass. Record the stricter project target and preferred tier
   separately; continue toward them when the measured curve, listening result,
   and capture budget justify it. More disagreement captures cannot repair
   noisy, clipped, drifting, or misaligned reamps.

For cloud ensemble or final-model training, follow
[`../../../trainer_source/cloud/README.md`](../../../trainer_source/cloud/README.md).
Use a dedicated non-production GCP project and bucket for immutable inputs,
member contracts, checkpoints, and final artifacts. RunPod is the default GPU
compute plane. Choose hardware by the lowest measured USD per completed
validation epoch that satisfies memory and checkpoint-cadence requirements;
longer wall time is acceptable unless the owner sets a deadline. Secure
on-demand A40 is the current proven default, with RTX A4000 as the lower-price
benchmark challenger when capacity is available. Do not choose H100 merely for
peak speed or VRAM: require a representative benchmark showing that its
wall-time reduction lowers total cost, or a separate owner-approved deadline
justification. The contract-aware launcher currently requires on-demand
capacity and rejects interruptible submissions.
Require the exact GCP project confirmation twice and verify the RunPod account,
funded balance, GPU/cloud type, image digest, registry credential ID, and GCS
service-account identity before submission. Store the RunPod key in the OS
credential store and the Google service-account file outside Git with mode
`0600`; never persist credential values in ledgers, logs, or memory.

Select the GPU before creating member contracts. GPU type is hash-bound to the
member, so a checkpoint cannot migrate across GPU types. If economics require
a hardware change, preserve the old run as evidence and create a new run ID
with fresh independent seeds against the identical frozen corpus. Record the
live offers, expected wall time, USD-per-epoch basis, aggregate hourly rate,
funded balance, and hard per-member and ensemble ceilings in the run ledger.

Every independent seed owns one immutable member contract and one durable GCS
checkpoint namespace. Mirror `resume.ckpt` after every validation epoch, keep
`best.ckpt` separately, and resume a terminated Pod only from that exact state
after proving globally that no writer still owns the seed. Google Cloud SDK is
required locally for bundle preparation, GCS inspection/download, and artifact
verification; the GPU compute itself runs on RunPod.

For every resumed cloud acquisition round:

1. Reconcile SQLite captures, `candidate-config.json`, and the capture
   manifests. Add every accepted capture exactly once and fail on duplicates.
   Materialize the cloud data config from the persistent store with
   `trainer_source/materialize_conditioned_data_config.py --data-dir ...
   --project-id ... --validation-capture-id ...`; do not maintain a parallel
   handwritten capture list or select validation by filename.
2. Persist frozen audit vectors separately and remove them before disagreement
   rendering or optimization. Merely withholding their wet WAVs is not enough.
3. Keep the validation capture, hidden DI/settings, architecture, learning
   rate, epoch count, alignment, and metric implementation fixed when measuring
   the incremental value of a new active capture.
4. Upload one immutable hash-bound GCS bundle whose declared `control_order`
   matches every vector, submit independently seeded RunPod members, and
   require every member to be quiesced with stable final artifacts before
   acquisition.
   After each validation epoch, mirror `resume.ckpt` to the seed's
   contract-hash GCS namespace with model, optimizer, scheduler, epoch, and
   global-step state. That checkpoint continues the same member after
   timeout/preemption; it does not create a new independent seed. Retain
   `best.ckpt` separately for export comparison. Before every continuation,
   prove globally that no writer for that seed is pending or running.
5. Recompute held-out ESR outside the trainer from the exported NAM payloads.
   Preserve every export, trainer statistic, job ID, active interval, bundle
   hash, and estimated cost; delayed billing is not an immediate invoice.
   Conditioned A2 uses a 6,347-sample receptive field. New exports declare it;
   pass `--receptive-field=6347` for older exports and never inherit the legacy
   WaveNet 4,093 default.
6. Use ensemble output only for acquisition and evaluation. The player ships a
   separately selected single model unless a real-time ensemble runtime has
   been explicitly designed and benchmarked.

For a generic project, materialize a target-free candidate config with
`materialize_conditioned_candidate_config.py`, score it with at least two
independent exports, then import it through
`import_conditioned_suggestions.py`. The importer must verify current capture
and reservation hashes and create a transactionally consistent SQLite backup
before queue mutation.

For the Fortin validation harness only, import an audit-safe native disagreement
report as a spatially separated persistent capture batch with:

```bash
PYTHONPATH=capture_companion/src:trainer_source \
python3 trainer_source/run_fortin_conditioned_active_learning.py capture-batch \
  --run-root "/absolute/path/to/capture-project" \
  --selection-report "/absolute/path/to/disagreement-audit-safe.json" \
  --round-index 2 \
  --maximum 8 \
  --minimum-distance 0.35
```

The command must refuse a stale candidate-config hash, captured or frozen
vectors, an existing round, pending recommendations, or pre-existing capture
evidence. It creates a SQLite backup through the backup API before queue
mutation, activates every point through the ordinary Trainer queue, renders
full standardized captures, runs official NAM validation, and records the
batch in `run-state.json`. It checkpoints candidate and run state after every
accepted capture so a partial batch remains auditable. Never bypass these
checks with direct file copies.

The Fortin harness is a regression tool, not the entry point for a new physical
amplifier. Its checked-in control schema and historical path defaults describe
that owned amp-sim experiment only. Build a new persistent Trainer project and
pass explicit paths for a different amp.

## Export and prove the player boundary

Export one self-contained `.dess` file. JSON manifests remain a legacy and
debugging boundary; the user-facing amp is the binary package containing the
canonical manifest and byte-exact `model.nam`, each protected by SHA-256.

Require all of these before calling the workflow complete:

```bash
/path/to/core-build/tools/loadmodel_manifest "/path/to/Amp.dess"
export DESSAMP_AU_BINARY="/absolute/staged/DessAmp.component/Contents/MacOS/DessAmp"
bash NeuralAmpModeler/tests/run_au_player_test.sh \
  "/path/to/Amp.dess" \
  --component-binary "$DESSAMP_AU_BINARY"
bash NeuralAmpModeler/tests/run_unserialization_layout_tests.sh
bash NeuralAmpModeler/tests/run_desspreset_tests.sh
bash NeuralAmpModeler/tests/run_au_state_migration_test.sh \
  --component-binary "$DESSAMP_AU_BINARY"
```

- Return the exported file at `package.path` from the packaging report. The owner can open
  it with **CUSTOM**, switch to a factory head and back, and reopen a saved host session.
  A `.desspreset` holds settings. A raw `.nam` drive capture has its own loading path.
- Core accepts the model and exact parameter count.
- The package still loads without adjacent source JSON/NAM files.
- Corruption is rejected.
- APP/AU/VST3 controls map in declared order; IDs 0–19 remain unchanged and
  Player Mode/slots remain append-only.
- A save/reopen render is deterministic, finite, and non-silent.
- Moving at least one learned knob changes output; snapshot controls do not.
- A level-matched Logic listening session judges tone, feel, and transitions.

Label short excerpts, reduced epochs, synthetic amps, or amp-sim black boxes as
`WORKFLOW-PROOF` or `workflow_validation_only`. They prove architecture, not a
physical amplifier or release-ready tone. Do not replace a shipped model,
package, sign, publish, or operate a physical amp without the corresponding
owner gate.
