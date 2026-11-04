# DessAmp Player Manifest v1

The player manifest describes the model, controls, and runtime mapping embedded
in the user-facing [`.dess` amp package](dess-package-v1.md). The capture
companion keeps a richer private evidence manifest beside its model database,
but packages only the explicit runtime allowlist.
A capture companion, DAW-assisted workflow, cloud trainer, desktop player, or
future hardware player may have different implementations; they agree on this
artifact boundary.

Current development consumers are the diagnostic Python Player and DessAmp
standalone, AUv2, and VST3. The plug-in currently has eight append-only learned
control slots; a manifest with more than eight controls remains valid for a
future larger standalone or hardware player but is rejected by this DAW
implementation.

## Binary envelope

The user-facing artifact is one binary file:

```text
Fortin-Nameless.dess
```

Its little-endian 96-byte header is followed by canonical UTF-8 JSON and the
byte-exact `.nam` payload:

```text
magic[8] = "DESSAMP\0"
uint32 format_version = 1
uint32 flags = 0
uint64 manifest_size
uint64 model_size
byte manifest_sha256[32]
byte model_sha256[32]
manifest bytes
model bytes
```

Readers must require the declared lengths to equal the complete file length,
reject trailing data, and verify both hashes before parsing or loading the
model. The v1 implementation limits the complete package to 64 MiB and the
manifest to 1 MiB.

The embedded manifest retains `model.path = "model.nam"` as its canonical
logical payload name. The legacy development layout below remains readable for
migration but is no longer the user-facing handoff:

```text
artifact/
├── player-manifest.json
└── model.nam
```

## Required fields

```json
{
  "schema_version": 1,
  "artifact_id": "uuid",
  "name": "Human-readable model name",
  "model": {
    "path": "model.nam",
    "format": "nam",
    "sha256": "lowercase hex",
    "format_version": "0.5.4",
    "architecture": "WaveNet",
    "sample_rate": 48000,
    "parameter_mode": "global_condition",
    "global_condition_size": 1,
    "runtime_quality": {
      "mode": "fixed",
      "default_size": 1.0,
      "breakpoints": []
    }
  },
  "controls": [
    {
      "id": "gain",
      "label": "Gain",
      "minimum": 0.2,
      "maximum": 0.8,
      "default_value": 0.5,
      "display_minimum": 20,
      "display_maximum": 80,
      "unit": "%",
      "taper": "linear",
      "model_parameter_index": 0
    }
  ],
  "fixed_controls": [
    {
      "id": "bright",
      "label": "Bright",
      "taper": "switch",
      "value": 1
    }
  ]
}
```

### Control display ranges

`display_minimum` and `display_maximum` are an optional pair for presenting a
control in physical or user-facing units. If either field is present, both are
required; both values must be finite numbers and `display_minimum` must be less
than `display_maximum`. They affect labels, values, and tooltips only. The DSP
mapping always remains `minimum + normalized * (maximum - minimum)`, and the
normalized default remains `(default_value - minimum) / (maximum - minimum)`.

For backwards compatibility, a `%` control whose model range is within
`0.0..1.0` falls back to `minimum * 100` and `maximum * 100` when the display
pair is absent. Other controls fall back to their model minimum and maximum.

The desktop amp editor presents amplifier controls on a conventional `0..10`
face. Legacy `35..80 %` is shown as `3.5..8.0`; `detent` values already use the
same face directly; and `clock h` maps `07:00..17:00` to `0..10`. The normalized
host parameter still spans only the manifest's trained model interval, and the
pointer is drawn only across that interval on the full dial. A package can
declare an explicit non-percent `display_minimum`/`display_maximum` pair with an
empty unit when a physical amplifier needs a different face calibration.

### Control taper and switch semantics

`taper` is optional and accepts `linear`, `log`, or `switch`. Omitting it is
wire-compatible with older manifests and means `linear`. `linear` and `log`
controls use a continuous knob in the current desktop editor; the v1 DSP
mapping remains the normalized linear formula above.

`switch` selects a binary toggle bound to the same learned host slot as a
continuous control would use. Its `default_value` must equal either `minimum`
or `maximum`. The editor and host integration emit normalized `0` or `1`
exactly, which the runtime maps to those two model endpoints. A typical switch
entry is:

```json
{
  "id": "bright",
  "label": "Bright",
  "minimum": 0.0,
  "maximum": 1.0,
  "default_value": 0.0,
  "unit": "",
  "taper": "switch",
  "model_parameter_index": 1
}
```

The manifest label is authoritative for generic and switch controls. Known
legacy Nameless IDs may retain their established physical-head label and
position, while an unknown ID is never silently relabeled as a different amp
function.

### Fixed controls

`fixed_controls` is an optional array for recorded amplifier controls that are
not model parameters. A `taper: "switch"` entry uses an exact numeric `value` of
`0` or `1`. A `taper: "linear"` entry supplies finite `minimum`, `maximum`, and
`value` fields plus an optional short `unit`; its value must lie in the declared
range. These entries are display and provenance state: they do not consume a
host parameter or enter the model vector. Their IDs must not duplicate learned
control IDs. The editor renders fixed controls visibly but non-interactively.

Packaging derives the recognized `bright`, `bottom`, `mid_boost`, and
`lead_volume` entries from the persistent project's frozen-control record. This
lets the Fireball package enforce Bright ON, Bottom and Mid Boost OFF, and Lead
Volume at `3.5` without hard-coding those states into the generic player.

`parameter_mode` is:

- `global_condition` when the NAM model accepts a block-constant parameter
  vector. Every index from zero through `global_condition_size - 1` must occur
  exactly once in `controls`.
- `snapshot` when a model represents one fixed rig setting. Controls may still
  describe the capture project, but their `model_parameter_index` is null.

`runtime_quality.mode` is independent from the amplifier controls:

- `fixed` means the model has one runtime size.
- `slimmable` means the NAM file is a `SlimmableContainer`. `default_size`
  selects its initial quality/CPU point and `breakpoints` must exactly match
  the thresholds reported by NAM Core. When A2 Lite and Full are packed this
  way, both WaveNet children retain the same global-condition size and control
  order; the size setting is not an amp Gain or tone knob. A value below the A2
  `0.5` breakpoint selects Lite, while `0.5` and above select Full.

The capture companion additionally records capture IDs, training provenance,
local evidence paths, and the configured-loader result in its private source
manifest. They are deliberately omitted from `.dess`; a package never serves
as the authoritative training ledger. This allowlist-by-default boundary keeps
new evidence fields private unless the runtime contract explicitly adopts a
bounded public field.

## Runtime obligations

A conforming player must:

1. validate the `.dess` magic, version, flags, finite sizes, exact total length,
   and both SHA-256 values before using either payload;
2. parse the embedded JSON without executing code and require its model hash to
   agree with the binary envelope;
3. for a legacy manifest, confine `model.path` to the artifact directory;
4. pass the verified `.nam` payload through the hardened NAM Core loader;
5. validate the control mapping before passing a parameter vector;
6. validate and apply the declared runtime-quality selection;
7. reset the DSP at the declared sample rate and reject non-finite output.

The `loadmodel_manifest` Core tool implements the structural, path, parameter,
model-load, and finite-render gates. The standalone DessAmp diagnostic Player adds the
independent SHA-256 and manifest/model agreement checks, then uses the native
`render_model` tool for reset, prewarm, block processing, parameter delivery,
finite-output validation, and PCM24 output. It canonicalizes supported input
WAVs before that call because DAWs can encode otherwise equivalent PCM in
different RIFF variants. Future real-time and hardware players must retain the
same checks before accepting an untrusted artifact.

## DessAmp DAW mapping

DessAmp currently preserves the inherited parameter IDs 0–19 for state
compatibility and uses eight normalized learned-condition slots at IDs 21–28.
The custom UI relabels only the active slots from the manifest; host-visible
names remain stable so saved automation does not change when a different
artifact is loaded.

Binary controls do not add parameters: a control mapped to model index `i`
uses the existing learned-condition parameter at host ID `21 + i`. Unmapped
Bright, Bottom, and Mid Boost stay visible but disabled; once a manifest maps
one of them with `taper: "switch"`, its parameter-bound toggle replaces the
inert placeholder. The lead-only product surface intentionally omits Clean Gain
and Clean/Lead even when a compatibility model retains those vector entries.

For `global_condition`, including conditioned A2 Full and conditioned A2
SlimmableContainer models, each normalized host value is mapped to the
manifest's declared model range and supplied as one block-constant vector. For
`snapshot`, the slots are inactive and no parameter vector is sent. The
manifest path and all 29 plug-in values are retained in the inherited state
layout. Migration tests must continue to prove deterministic restore without
reordering released parameters.
