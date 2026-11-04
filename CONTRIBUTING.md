# Contributing to DessAmp

DessAmp is preparing for an initial public source release. Until that boundary
is finalized, discuss substantial changes with the maintainer before investing
in them and never attach private captures or model artifacts to an issue or
change proposal.

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Source and asset boundary

- Source contributions are intended for the root MIT-licensed codebase.
- Do not add commercial amplifier captures, `.dess`/`.nam` run outputs,
  cabinet IRs, hidden evaluation WAVs, checkpoints, credentials, signing
  files, copied product artwork, third-party logos, or unverified fonts.
- New third-party source or assets require an upstream URL, exact version or
  hash, compatible license, retained license text, and an entry in
  `THIRD_PARTY_NOTICES.md`.
- Preserve manifest-defined model-vector order. A UI reorder must not change
  DSP indices.
- Keep file/model loading, allocation, hash validation, and model replacement
  off the audio thread.

## Development setup

Read `AGENTS.md`, `docs/architecture.md`, and `docs/training-runbook.md` before
changing capture, training, evaluation, packaging, or cloud code. Build Core
tests from the `NeuralAmpModelerCore` working directory because its fixtures
resolve relative paths there.

Run the focused tests for the changed component and the complete repository
gate before proposing a release change. For conditioned A2 tests, retain the
separate Python environments and processes documented in the runbook so the
historical local `nam` package cannot shadow the official dependency. Recreate
the retained trainer with `uv sync --project trainer_source --locked --extra
test` in an external environment; do not commit a virtual environment.

## Change quality

- Keep changes focused and preserve unrelated work in a dirty tree.
- Add regression tests for behavior changes.
- Record user-visible or architecture changes in the relevant evergreen doc.
- Treat build, external numerical evaluation, host validation, visual QA,
  human listening, signing, installation, and publication as separate gates.
- Do not claim tone quality from internal training metrics.

By submitting a contribution, you agree that your contribution may be
distributed under the repository's MIT license and that you have the right to
submit it.
