# Evaluation and stopping contract

## Dataset separation

Maintain four disjoint roles:

1. training captures;
2. acquisition candidates scored by disagreement;
3. frozen setting holdouts that are never offered to acquisition;
4. repeat captures used to estimate the physical noise/drift floor.

Evaluate every capture-count checkpoint on the same frozen settings and a dry
performance/DI not used for training crops or acquisition scoring. Record mean,
median, 95th percentile, and worst-setting metrics; a mean alone can hide a bad
corner.

## Metrics

For aligned target `y` and prediction `y_hat`, report error-to-signal ratio:

```text
ESR = sum((y - y_hat)^2) / sum(y^2)
```

Trim only the frozen alignment/receptive-field boundary. Do not optimize time,
gain, polarity, or EQ independently per evaluated model. Also report a
spectral/perceptual metric and level-matched listening because ESR does not
fully represent guitar tone or feel.

Use these provisional engineering gates until a larger real-amp benchmark
justifies changing them:

- capture repeatability ESR: `<= 0.01`; otherwise fix the capture path first;
- NAM-normal acceptable conditioned model: every held-out setting has ESR below
  `0.035`. This is the normal NAM **Not bad** band and lets a candidate advance
  to the remaining release/listening gates;
- project quality target: holdout mean ESR `<= 0.01` and 95th
  percentile/worst-setting ESR `<= 0.02`; keep acquiring and training toward
  this target when the curve and budget justify it;
- preferred release candidate: holdout mean ESR `<= 0.005`, without a worse
  corner or listening regression;
- finite/non-silent output, deterministic state recall, no clipping, and no
  unresolved capture warnings are mandatory.

Numerical acceptance is only one release gate. A two-epoch export with high ESR
must retain a `WORKFLOW-PROOF` label even when the loader and player pass, and a
NAM-normal candidate still needs deterministic host, real-time, level-matched
listening, provenance, and physical-capture gates before release.

### Interpreting an audible knob

Compare the physical and rendered endpoints on the same passage and at the native
knob values. For a brightness control, record a relevant high-to-low band-energy
ratio at both ends, then listen to level-matched versions. A strong ESR can coexist
with a sweep that is too gentle. A near-null capture pair calls for a routing or
control check. A single frequency ratio cannot establish that a control is inaudible.

Record the distinction between a withheld knob position on the training stimulus,
a withheld time segment, and an independent performance. Once a measured holdout
informs another training pass it is a reused holdout. Obtain fresh hidden evidence
before using it to make a release claim.

## Capture-count planning

Do not promise a count before measuring the target amp. Use a checkpoint curve
and reserve captures for independent evidence:

| Scope | Training budget | Audit/repeat reserve | Initial total plan |
| --- | ---: | ---: | ---: |
| 1 continuous control | 7–12 | 4–6 | 11–18 |
| 2 interacting controls | 12–20 | 6–8 | 18–28 |
| 6–7 continuous controls | 60–75 | 10–15 | 70–90 |
| 6 continuous + Bright/Bottom switches | 75–100 | 15–20 | 90–120 |

For a full six- or seven-control amp, start with ten space-filling points, then
add roughly six to ten disagreement-selected settings per round. Train and
evaluate at approximately 10, 25, 50, and 75 training captures. The current
PANAMA research supports 75 as the serious planning point for a six-control
amp, not as a universal guarantee.

For one lead channel with six continuous controls plus Bright and Bottom,
stratify the starter set across all four switch combinations (roughly 12–16
training settings total), then let disagreement choose both the continuous
location and switch state. Check at about 25, 50, 75, and 100 training captures.
Do not multiply the continuous budget blindly by four, but do not treat the
switches as free: each discrete mode needs both training support and hidden
corners. Add Mid Boost later only after this four-mode curve is understood.

Stop early only when all of these hold:

- the NAM-normal absolute ESR gate passes on the frozen setting/DI audit set;
- the stricter project target and preferred tier are recorded explicitly even
  when they have not yet passed;
- the last active round improves mean and worst-setting error by less than 5%
  relative, or the curve has otherwise flattened across two checkpoints;
- repeatability remains materially below model error;
- acquisition no longer reveals a concentrated bad region;
- the target player meets callback, memory, thermal, and power constraints;
- level-matched human listening passes across clean/transient and sustained
  playing, corners, centre, knob sweeps, and save/reopen.

If 75 training captures still miss the gate, diagnose model capacity,
conditioning, alignment, stimulus coverage, amp drift, and discrete modes
before collecting a blind Cartesian expansion.
