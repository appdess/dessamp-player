# DessAmp Package v1 (`.dess`)

`.dess` is the self-contained user-facing format for a complete DessAmp
model. It is a real binary envelope, not a renamed JSON file. It embeds the
canonical [Player Manifest v1](player-manifest-v1.md) and the byte-exact NAM
payload so a desktop plug-in or future hardware player needs only one file.

## Binary layout

All integer fields are unsigned little-endian values. The fixed header is 96
bytes:

```text
offset  size  field
0       8     magic = DESSAMP followed by NUL
8       4     format_version = 1
12      4     flags = 0
16      8     manifest_size
24      8     model_size
32      32    SHA-256 of the exact manifest bytes
64      32    SHA-256 of the exact model bytes
96      ...   canonical UTF-8 JSON manifest
...     ...   byte-exact NAM model
```

The runtime-only manifest is serialized as compact UTF-8 JSON with sorted keys.
Its logical model path remains `model.nam`, and `model.sha256` must equal both
the header's model digest and the embedded payload digest. Package creation
uses an explicit allowlist: private capture IDs, training provenance, local
paths, cloud identifiers, and validation logs remain in the external evidence
ledger and are never copied into the distributable artifact.

## Limits and validation

The v1 implementation permits a maximum 64 MiB package and a maximum 1 MiB
manifest. A conforming reader must, before model construction:

1. require the exact magic, version `1`, and flags `0`;
2. validate both declared sizes and require header + manifest + model to equal
   the complete file length, with no trailing bytes;
3. verify both SHA-256 digests;
4. parse the manifest as an object and require its model hash to agree;
5. validate control indices and the declared runtime-quality mapping;
6. pass the extracted NAM payload through the hardened NAM Core loader.

Writers use an adjacent temporary file, flush it, and atomically replace the
destination. Readers extract a hash-named model into a local cache and verify it
again. Corrupt manifests, corrupt models, truncation, unsupported flags, and
trailing data are rejected in both the Python Player and native Core loader
tests.

## Compatibility

The older two-file manifest-plus-model directory remains accepted during
development and migration. New Trainer downloads and end-user
model libraries should expose `.dess`. A future compact hardware encoding may
be added as a new format version; v1 readers must reject rather than guess at
unknown versions or flags.
