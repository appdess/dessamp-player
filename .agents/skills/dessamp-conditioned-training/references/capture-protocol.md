# Conditioned capture protocol

Use this checklist for direct-interface and Logic/manual captures. Preserve a
machine-readable project record; Markdown notes alone are not the capture
database.

## 1. Freeze the experiment

Record:

- amp, load box/reamp chain, interface, channel names, clock, sample rate, and
  buffer size;
- dry stimulus path and SHA-256;
- exact head-only boundary and every frozen knob, switch, input mode, pedal,
  cab, EQ, gate, and time effect;
- learned controls in ordered model-vector form, including native range,
  normalized range, capture step, default, and unit;
- send calibration, return calibration, one shared post-amp trim, and polarity;
- warm-up time, knob-settling time, latency/alignment policy, and session ID.

For a new physical project use the full, unmodified NAM v3.0.0 training input
from the [owner guide](../../../../docs/physical-amp-capture.md#2-get-the-training-sweep-and-check-the-signal-path).
Verify the original file's SHA-256 and its 9,120,000 mono PCM24 frames at
48 kHz before playback. Keep the dry reference, start marker, duration and
calibration fixed for all training takes. Reserve different DI audio for
acquisition scoring and the hidden audit.

## 2. Prove the path before the starter set

1. With the amplifier path bypassed or looped back, prove routing, polarity,
   latency, duration, and that the return is neither clipped nor silent.
2. Render the expected loudest safe amp setting first. Choose one safety trim
   for the entire project; leave headroom for active suggestions.
3. Capture the same centre setting at least twice after resetting the controls.
   Calculate repeatability ESR. Investigate drift, noise, clipping, or alignment
   before proceeding if it exceeds the evaluation gate.
4. Do not feed speaker output into a line input. A physical tube amplifier must
   always see its intended cabinet or rated reactive load.

## 3. Starter design

For one or two controls, cover corners, centre, and axis/interior points without
using the audit settings. For six or seven controls, use at least ten
space-filling starters; avoid a full Cartesian grid. Persist the random or
low-discrepancy seed and the candidate grid.

Hold back settings that are deliberately distinct from the starters and use a
different performance/DI segment for evaluation. Audit settings must never
become training captures, disagreement candidates, early-stopping data, or
model-selection data.

## 4. Capture in Logic or another DAW

For each active recommendation:

1. Read the recommendation ID and ordered native knob values from the Trainer.
2. Set only the declared learned controls. Verify every frozen control.
3. Route the exact dry stimulus through the frozen amp path, with plug-in cab
   and time effects disabled for a head model.
4. Bounce the mono return from the same start marker and exact duration. Disable
   normalization, dither changes, master processing, and export resampling.
5. Import the WAV against that recommendation; never infer its settings from a
   filename alone.
6. Confirm format, duration, peak, finite samples, alignment, and hash evidence.
   Reject or explicitly quarantine warnings instead of silently truncating.

### Two-stage manual-capture pipeline

For an owner-operated physical amplifier with a pre-queued acquisition batch,
overlap knob setup with validation without weakening acceptance:

1. Immediately after recording, stop Logic, return it to the frozen start
   marker, and record-disable the raw track. Run the fast gate: fresh device and
   routing readback, stable source file, mono 48 kHz PCM24, enough frames for
   the exact standardized prefix, finite/non-silent samples, no clipping, and
   byte-exact prefix materialization plus hashes.
2. If the fast gate passes, present exactly the next queued physical setting
   immediately so the owner can move the knobs while the deeper validation
   runs. This is a setup handoff, not acceptance of the preceding capture.
3. In parallel with owner knob setup, run the full official NAM input-version,
   length, latency, data-check, PyTorch-loader, and repeatability gates. Then
   create the integrity-checked SQLite backup, import the accepted capture
   transactionally, persist all evidence, and activate the next ordinary
   Trainer recommendation.
4. Do not operate Logic for the next `capture` reply until the preceding take
   has passed the deep gate and is transactionally persisted. If the deep gate
   fails after the next setting was shown, quarantine the failed take, withdraw
   that provisional handoff, and recapture or repair the preceding setting
   before advancing the queue.

This two-stage overlap is the default for established deterministic capture
paths. It does not apply before path calibration/repeatability is proven, and
it never relaxes the separation of training, hidden audit, acquisition, and
repeatability evidence.

Computer-use automation may operate an owned amp-sim UI and Logic when the user
has requested it. Keep recording actions visible and bounded. For a physical
amplifier, stop at each knob-setting handoff unless safe motorized control and
the complete load/routing chain have already been proven.

## 5. Active round

Train acquisition members with different seeds on the same current training
set. Prefer LSTM members for acquisition throughput. Score only uncaptured,
non-audit settings from the declared physical grid. Persist:

- ensemble configuration and member seeds;
- source/config/checkpoint hashes;
- optimization DI path/hash;
- every ranked candidate and disagreement score;
- selected vector, round, control order, and duplicate checks.

For a generic physical project, run the [candidate, scoring and guarded import
commands](../../../../docs/training-runbook.md#10-select-and-queue-the-next-active-captures).
The resulting external suggestions enter the ordinary Trainer queue. A separate
PANAMA learner can also supply `new_points.json` for the Trainer bridge.
On resume, reuse the pending recommendation first. Check the imported queue
before requesting the next active take. After the batch is exhausted the
Trainer can suggest another coverage point.

## 6. Export

Train the final conditioned model from the accepted training captures, inspect
the exported NAM metadata, and package it as `.dess`. The package must embed:

- the byte-exact `model.nam`;
- schema version, artifact identity, sample rate, architecture, condition size,
  runtime quality, capture evidence, and provenance;
- every control's id, label, range, default, and unique model parameter index;
- SHA-256 for both embedded sections.

Keep the original capture masters, project database, training configuration,
checkpoints, and holdout report. The `.dess` file is the portable player
artifact, not the sole scientific record.
