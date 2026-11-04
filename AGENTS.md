# DessAmp repository guidance

DessAmp is an independent conditioned-amplifier product and research repository
based on the MIT-licensed NAM/iPlug2 source lineage. It currently keeps the
player, its conditioned-A2 Core runtime, capture companion, and trainer together
so the model/package boundary can evolve atomically. If a later split becomes
useful, keep Core with the player and separate only the trainer workflow.

Before capture, training, active-learning, cloud-job, model-evaluation, or
`.dess` export work, use the repository-local
`$dessamp-conditioned-training` skill and follow
[`docs/training-runbook.md`](docs/training-runbook.md). Route plain-language
requests such as “help me capture my amp for the plugin”, “which sweep do I
play?” and “what do I set next?” to the same
[skill](.agents/skills/dessamp-conditioned-training/SKILL.md).
For a walkthrough, point the owner to the verified NAM v3.0.0 sweep in the
guide and give the next step for their rig. For a live session, use the
persistent Trainer queue and present one physical setting at a time. Start with the
[owner guide](docs/physical-amp-capture.md).
Ask which knobs and physical ranges the owner actually plays. Freeze the rest
before designing the capture set. The guide includes the Logic Pro routing and
the recorded Fireball RunPod training example.

Architecture ownership and the future split are documented in
[`docs/architecture.md`](docs/architecture.md).

## Product boundary

- The product loads portable `.dess` amplifier packages and targets conditioned A2 Full inference.
- Do not reintroduce the legacy DessMetal amplifier catalogue, scene artwork, or product identity.
- Preserve manifest-defined model-vector order even when controls are visually reordered.
- Keep model loading, hash validation, allocation, and model replacement off the audio thread.
- Generated art is a non-interactive shell. Labels, values, pointers, arcs, focus, availability, and hit targets must remain code-bound.

## Repository boundary

- `NeuralAmpModeler/` is the APP/AUv2/VST3 product surface.
- `NeuralAmpModelerCore/` owns native `.nam`/`.dess` loading and real-time
  conditioned A2 inference.
- `capture_companion/` owns persistent capture projects, recommendations,
  SQLite evidence, and the local Trainer/diagnostic Player. Its Python import
  namespace remains `dessmetal_capture` for compatibility; do not rename it as
  an incidental training change.
- `trainer_source/conditioned_a2/` owns conditioned A2 Full/Lite training and
  export. Full is the current desktop target. Lite is deferred until measured
  hardware constraints justify it.
- `trainer_source/cloud/` owns immutable GCS bundles, digest-pinned RunPod
  jobs, provider guards, and durable best/resume checkpoints. RunPod is the
  default compute plane; Google Cloud Storage remains the evidence plane.
- Capture masters, project databases, checkpoints, cloud results, and trained
  commercial-amp artifacts are evidence outside Git. Never add them merely to
  make a command convenient.
- `trainer_source/demo_ckpt.ckpt` is the sole checkpoint exception: it is a
  fixed, non-product restricted-deserialization test fixture. Never replace it
  with a run output.

## Operational memory handoff

- During long capture, training, evaluation, recovery, or architecture work,
  keep a concise durable handoff in the approved local Codex memory-extension
  folder. Record verified current state, exact run/ledger paths, immutable
  contracts and hashes, active writers, recovery rules, decisions, failures,
  reusable learnings, and the next uncrossed gate.
- Refresh the handoff at meaningful milestones such as corpus freeze, cloud
  submission, recovery, terminal artifact validation, hidden evaluation, and
  acquisition-queue creation. Do not log routine heartbeat noise or raw chat.
- Distinguish observed facts, plans, and verified completion. The persistent
  SQLite project, run ledger, GCS artifacts, and current cloud queries remain
  authoritative; a memory handoff is routing context, not completion proof.
- Never store credentials, tokens, private capture audio, or model binaries in
  memory. Link to their durable evidence paths and preserve provenance instead.

The current Nameless validation model's seven-element vector is, in exact DSP order:
`amp_mvc`, `amp_presence`, `amp_bass`, `amp_middle`, `amp_treble`,
`amp_gain_1`, `amp_gain_2`. A new physical amplifier must declare its own
control order in the persistent project and hash-bind that same order in the A2
bundle; do not copy these Fortin IDs. The trainer can research up to 12 axes,
but a directly loadable desktop artifact currently supports at most eight
learned controls. The intended six continuous plus Bright/Bottom lead model
fits that player boundary. The UI may present controls in a more natural amp
order, but model indices and manifest IDs are authoritative.

## Safety

- Preserve unrelated work in a dirty tree.
- Do not stage, commit, tag, push, install, sign, notarize, upload, or publish without explicit owner authorization.
- Treat source, model, IR, font, icon, and generated-image provenance separately.
- Do not use third-party product names, logos, copied faceplates, or bundled commercial model assets in release chrome.
- Cloud uploads and RunPod Pod submissions are billable external actions.
  Recheck the exact GCP project/bucket, immutable bundle, image digest, RunPod
  GPU/cloud type, interruptibility choice, funded balance, and owner ceiling
  before every submission.
- Choose cloud hardware by measured USD per completed validation epoch. Longer
  wall time is acceptable unless the owner sets a deadline. Secure on-demand
  A40 is the current proven default and RTX A4000 is the lower-price benchmark
  challenger; H100 requires measured total-cost advantage or an explicit
  owner-approved deadline premium. GPU type is member-contract-bound, so a
  cheaper-hardware switch requires a new run ID and fresh independent seeds.
  Delete identity-verified terminal Pods after stable GCS handoff so retained
  Pod disks do not continue billing.
- A checkpoint continuation is not an independent ensemble member. Preserve
  its seed, dataset, optimizer, scheduler, architecture, and immutable input
  prefix. Use the exact contract-aware `--resume-existing-member` flow for new
  jobs; never extend or reinterpret its absolute epoch target. Independent
  disagreement members must start from independent seeds.

## Training acceptance

- Keep training captures, disagreement candidates, frozen audit settings, and
  physical repeatability captures disjoint.
- Every held-out setting below ESR `0.035` is the normal NAM numerical gate.
  Continue aiming for mean ESR `<= 0.01` and p95/worst `<= 0.02`; prefer mean
  ESR `<= 0.005` without a worse corner.
- Recompute holdout ESR outside the trainer with the native current-source
  renderer. An internal validation score cannot release a model.
- Conditioned A2 Full/Lite uses a 6,347-sample receptive field. New exports
  declare it; older A2 exports require the explicit evaluator/scorer flag.
- Export one selected member to `.dess`; ensemble averaging is for audit and
  acquisition unless a real-time ensemble runtime is separately designed.
- A failed-gate artifact may be packaged only as
  `workflow_validation_only` for listening. It must not replace a shipped
  model.

## Validation

- Build fresh APP, AUv2, and VST3 outputs from current source.
- Run source/core tests and the exact staged AU conditioned-player test.
- Pass the exact current staged AU executable through `--component-binary`;
  installed component discovery is not acceptable release evidence.
- A successful build does not prove the UI: open the current standalone and exact plugin editors, exercise every active control, and capture visual evidence.
- Automated output checks do not judge tone; require human listening for final sound approval.
- Run the capture-companion, conditioned-A2, cloud-helper, manifest/package,
  state-layout, and exact staged AU player tests described in the runbook.
