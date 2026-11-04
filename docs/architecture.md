# DessAmp architecture

## Current decision

DessAmp remains one repository while the conditioned-model contract is still
changing. The player, native runtime, capture workflow, A2 trainer, evaluator,
and package schema must currently change and be tested together. A premature
repository split would add version coordination without creating a stable
public boundary.

The repository root is the authoritative source boundary. A hosting remote does
not become a publication boundary until the reviewed allowlist, clean history,
license/provenance audit, and release gates in `docs/release-process.md` pass.

The legacy DessMetal repository is a source/reference checkout, not the active
DessAmp training workspace. New player and conditioned-A2 changes belong here.

## Data flow

1. `capture_companion/` persists a model project, ordered controls, candidate
   grid, recommendations, captures, hashes, warnings, and job evidence.
2. PANAMA-style acquisition members score only uncaptured, non-audit settings
   by ensemble output disagreement.
3. `trainer_source/conditioned_a2/` trains one conditioned A2 Full child from
   the accepted capture corpus and one separate internal validation capture.
4. `trainer_source/cloud/` stores one immutable hash-bound bundle in GCS and
   runs its digest-pinned trainer on RunPod. Every validation epoch mirrors
   `best.ckpt` plus `resume.ckpt` back to the seed's contract-bound GCS prefix.
   The reusable Google signer stays on the operator workstation; a RunPod
   worker receives only expiring GET/PUT capabilities for the exact objects in
   that member contract.
5. Native current-source rendering evaluates every exported child on frozen
   external settings and a separate DI. Those settings are never acquisition
   candidates or model-selection inputs.
6. `trainer_source/package_conditioned_export.py` binds one selected model and its control
   schema into a self-contained `.dess` artifact.
7. `NeuralAmpModelerCore/` validates and runs the embedded model;
   `NeuralAmpModeler/` presents manifest-defined controls in APP/AUv2/VST3 and
   the shared iOS/iPadOS standalone/AUv3 editor.
8. `ios/` is the mobile workspace entry point. It references the shared player,
   Core, iPlug2, and AudioDSPTools sources rather than copying them into a second
   application tree.

## HeadRush pedal startup

On the verified Flex Prime HG12, a one-time installer puts the FIRE adapter on
internal storage. A local NetworkManager hostname event prepares it during boot.
The model loads when the saved rig selects FIRE. The decision record explains
how the first two boot trials led to this hook and how an unplugged cold start
verified it. [HeadRush startup decision](headrush-startup-decision.md)

## Conditioned A2 Full

The current deployable target is A2 Full at 48 kHz:

- upstream A2 Full topology: specialized eight-channel fast path;
- DessAmp extension: a block-constant global knob vector mixed into every A2
  layer;
- current global condition size: seven;
- current trainable model size: approximately 13.4k parameters;
- controls are normalized by their manifest ranges before entering the model;
- control work is precomputed per audio block, not repeated as UI work on each
  sample.

The exact current Nameless validation vector is:

1. `amp_mvc` (Master / UI Lead Volume)
2. `amp_presence`
3. `amp_bass`
4. `amp_middle`
5. `amp_treble`
6. `amp_gain_1` (retained at its saved/default value for compatibility; hidden
   in the lead-only UI)
7. `amp_gain_2` (UI Lead Gain)

The UI is deliberately free to reorder these controls visually. The manifest
IDs and `model_parameter_index` values—not screen position—define DSP order.
Bright, Bottom, and Mid Boost remain disabled placeholders until a trained
package maps them with `taper: "switch"`. They then become operable binary
editors on the existing learned parameter slots; no host parameter is added or
reordered. Clean Gain and Clean/Lead are intentionally absent from the product
surface: DessAmp's first physical target is the lead channel only.

That seven-ID list is not a global product schema. The generic trainer and cloud
entrypoint accept one to twelve declared controls and verify the same order in
the persistent project, materialized data config, immutable input manifest,
container arguments, and exported NAM metadata. The current host/player surface
has eight append-only learned parameter slots, so a directly loadable `.dess`
must contain at most eight controls. A first physical lead-channel proof should
use one or two controls; the intended expanded lead model is exactly eight: six
continuous controls plus Bright and Bottom. Add Mid Boost only by replacing or
versioning that player schema after the four-mode dataset passes its curve and
audit gates.

Conditioned A2 Full and Lite currently have a 6,347-sample receptive field. New
exports record this in metadata so evaluation cannot silently inherit the
legacy conditioned-WaveNet value of 4,093.

A2 Lite remains implemented for experiments but is not the desktop product
target. Revisit it only after an embedded device benchmark proves that Full
cannot meet callback, memory, thermal, or power limits.

## Package and real-time boundary

`.dess` is the user-facing amp file and the common boundary for factory and
user-loaded amplifiers. Version 1 embeds a canonical manifest and the byte-exact
`model.nam`; version 2 may additionally embed one hash-bound Morph model. Lengths
and SHA-256 values protect every payload. The manifest supplies identity, sample
rate, architecture metadata, control IDs, ranges, defaults, display ranges, and
exact model indices. The player remains backward compatible with version 1.

The product editor selects a fixed bank of factory heads. Each product maps to a
bundled package or a grouped family such as REDLINE. External package parsing
remains available to the trainer, QA tools, and saved sessions that already
reference a path. Users can edit learned controls, save scoped presets, and
choose a cabinet IR. They cannot replace a factory head through the editor.

A Morph package declares the two endpoints, alignment/polarity/level correction,
and the host slot used by the Morph control. The real-time path runs both
prepared heads and applies a smoothed continuous blend without allocation in the
audio callback. This is an explicit two-head morph, not a claim that the two
source amplifiers were retrained into one joint neural manifold.

An optional bounded `presentation.head_shell` token may select one of the
player's own original, compiled shell images. It is not a path and cannot make
the player read package-supplied artwork; unknown safe tokens fall back to the
standard shell. Labels, values, pointers, focus, availability, hit targets, and
Morph-dependent tube light remain code-rendered. A provenance-restricted `.dess`
may enter an explicitly marked local macOS listening build through
`DESSAMP_PRIVATE_AMP_PACKAGE` plus `DESSAMP_PRIVATE_AMP_VALIDATION=YES`;
ordinary builds contain no such private slot. A future release catalogue may
contain only packages with separately verified redistribution rights and asset
provenance.

Shell identity is resolved independently from model topology. An explicit
`dual-red-modern-v1` presentation remains DESS REDLINE even when the package
contains a Morph axis; an explicit Forge shell, or the compatibility fallback
for an otherwise unclassified dual-model Morph package, remains DESS FORGE;
the Fire shell remains DESS FIRE. This precedence prevents a learned or
dual-model Morph implementation detail from renaming a product.

The owner-approved three-product listening catalogue is ordered DESS FIRE,
DESS FORGE, then DESS REDLINE. Resource preparation neutralizes package names,
so a private integration directory must establish that order lexically before
copying. DESS REDLINE currently has two discrete package variants that share one
catalogue root, so the private layout is `01-fire.dess`, `02-forge.dess`,
`03-redline-us.dess`, and `04-redline-brit.dess`; the shared Redline product
family groups the final two behind the code-bound US/BRIT selector. The manifest
remains authoritative for identity, controls, and runtime behavior.
The final private listening directory is not assembled by hand. Once an
externally accepted Fire package exists,
`tools/assemble_private_listening_catalogue.py` requires its accepted package
report, exact hashes for all four packages, the expected product/control/fixed-
control contracts, and a fresh current-Core smoke load before atomically
creating those four lexical slots outside the repository. Its generated lock
remains private workflow evidence and explicitly leaves listening, signing,
installation, and redistribution unapproved.
This ordering contract does not grant redistribution rights: ordinary public
builds remain model-free until each package passes its independent fidelity,
runtime, provenance, and release gates.

The PRE drive selector is a separate capture-asset boundary. Ordinary builds
reset and omit its model bank unless the exact four-file bank is present in the
rights-cleared `NeuralAmpModeler/resources/Drives` directory. Private listening
builds may instead set `DESSAMP_PRIVATE_DRIVE_DIR` together with
`DESSAMP_PRIVATE_DRIVE_VALIDATION=YES`. The legacy
`NeuralAmpModeler/resources/models` tree is never a build source.

Parsing, hashing, allocation, model construction, and replacement happen off
the audio thread. The callback receives an already validated model and one
coherent block-constant control vector. UI controls, presets, input/output
levels, gate, and PRE processing remain outside the learned amplifier vector
unless a future model explicitly conditions them.

## Cabinet boundary

The conditioned `.dess` artifact remains a head model. Cabinet convolution is
an independent player stage after the learned amplifier, with its enabled state
and selected WAV path persisted by the existing host/preset state. The CAB page exposes the bundled IR, custom mono WAV loading, and cabinet bypass.
Switching factory heads recalls their defaults while retaining an independently chosen custom IR.

The private product stages four hash-pinned cabinets. The public source
checkout supports custom WAVs. Cabinet selection is part of the saved rig.

## Possible future split

Keep everything in this repository for the first physical-amplifier iterations.
The native loader and real-time inference Core belong to the player: they share
one release, one host-state contract, and one `.dess` compatibility boundary.
Creating a separate Core repository would add version coordination without a
separate product today.

After at least one physical-amplifier release cycle, the only split worth
considering is:

- `dessamp-player`: APP/AUv2/VST3 UI, presets, host state, product assets,
  model/package parser, conditioned A2 inference, and renderer;
- `dessamp-trainer`: capture database, acquisition, training, cloud transport,
  evaluation, and export.

Make that split only if trainer development and distribution genuinely need an
independent lifecycle. Before splitting, require versioned `.dess`
compatibility tests and a published player/trainer dependency strategy. Until
then, one repository is the lower-risk design.
