# DESS AMP Preset v1 (`.desspreset`)

`.desspreset` is a small binary state file for DESS AMP. It is deliberately
separate from a `.dess` amplifier package: the amp package contains the learned
model and its control schema, while the preset remembers the exact plug-in
state that selects that model and sets all 33 host parameters. Parameters
0 through 28 retain their previous IDs. The PRE Level knobs use four appended
slots (29 Oxide, 30 Amber, 31 Verdant, 32 Nocturne). All four are stored,
including levels for pedals that are bypassed.

Existing 29-value presets and factory profiles reopen with each individual
trim at 0 dB. Their original Boost Level (ID 7) still supplies the common
post-drive gain, including existing host automation. The selected pedal's own
level adds to that gain before the amplifier.

Verdant Tone (ID 8) drives the physical tone axis when the private drive bank
contains the conditioned TS9 capture. A fixed-setting TS9 has no tone axis.
The private listening build uses the one-axis capture with SHA-256
`d7021b1bbbcce0cc4f6438c89fbf20d86f18138d7dacf64e70670da06043aa8f`
in its gated `DESSAMP_PRIVATE_DRIVE_DIR` on Mac and iOS.

The v1 file layout is:

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `DESSPRST` |
| 8 | 4 | little-endian version (`1`) |
| 12 | 4 | little-endian state size |
| 16 | state size | byte-exact output of the existing `SerializeState` method |

Readers require the declared size to consume the complete file, reject unknown
versions, truncation, trailing bytes, and state payloads larger than 4 MiB.
Saving writes and flushes a sibling temporary file before atomically renaming it
over the requested destination. Loading validates the complete preset envelope
before touching plug-in state. After validation, it snapshots the current
serialized state and restores that snapshot if state recall fails.

Preset files intentionally retain the model path stored by the existing plug-in
state. Moving a `.dess` model can therefore require selecting that amplifier
again and resaving the preset.
