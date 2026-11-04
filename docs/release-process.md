# Release process

This checklist prepares a DessAmp release; it does not authorize publication,
model redistribution, signing, or notarization. Each gate needs its own durable
evidence and owner decision.

## 1. Freeze source and asset scope

- Build the release from the explicit fail-closed allowlist implemented by
  `tools/stage_public_source.py` and a clean history, not the current local
  commitless object store. Internal construction manifests and audit evidence
  are intentionally not copied into the staged public tree.
- Hash the exact source archive and dependency lockfiles.
- For a conditioned trainer image, retain the exact Cloud Build source object,
  strict upload inventory, direct requirements, observed-image constraints,
  hash lock, build ID, tag-to-digest resolution, and image digest. Prove the
  lock regenerates byte-for-byte for the declared Python/Linux/CUDA target,
  Docker installs it with `--require-hashes`, and local environments, caches,
  tests, checkpoints, private evidence, and unrelated trainer sources are
  absent from the uploaded context.
- Keep captures, hidden WAVs, SQLite projects, checkpoints, cloud ledgers,
  private `.dess`/`.nam` packages, and signing material outside that history.
- Record file-level provenance and redistribution approval for every bundled
  model, IR, font, icon, and artwork layer.
- Unset all `DESSAMP_PRIVATE_*` variables for a public build. Assert that no
  `PrivateValidation` subtree exists and that a PRE drive bank is present only
  when the exact files in `NeuralAmpModeler/resources/Drives` have documented
  release approval.

## 2. Verify licenses and security

- Review `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the dependency/license
  inventory against the frozen source.
- Export a CycloneDX 1.5 default-runtime inventory from each reviewed Python
  lock. For the retained trainer, use `uv export --preview-features
  sbom-export --project trainer_source --locked --format cyclonedx1.5 --no-dev
  --no-extra transformers-compat`, hash the exact JSON, and review the licenses
  from the resolved release-platform environment.
- Build each first-party Python distribution and verify that its wheel and
  source archive declare the MIT expression and contain their exact license
  file. Do not infer package metadata from the repository-level license alone.
- Run the resource-preparation tests and verify the generated third-party
  license manifest in every final APP/component, then copy the same verified
  bundle into the installer/DMG.
- Run a history-aware secret scan and dependency/security audit on the clean
  history. Resolve findings or document an explicit, evidence-backed decision.
- Rebuild the release trainer image from the reviewed strict source context and
  run `pip check`, the image's pinned runtime assertions, and the enabled
  registry/container scanner. An unavailable or disabled scanner is a release
  blocker, not a no-findings result. A macOS lock-generation check is not a
  substitute for executing the Linux/CUDA image.

## 3. Prove model eligibility

- Select a single release member without consulting frozen external targets.
- Pass stable-generation, manifest-hash, immutable-contract, current-Core load,
  and native external-render gates.
- Preserve training, frozen audit, repeatability, acquisition, packaging,
  player, and listening evidence as separate roles.
- Require explicit redistribution approval in addition to numerical acceptance.
- For the private owner-listening build, assemble FIRE, FORGE, and both REDLINE
  voice packages with `tools/assemble_private_listening_catalogue.py`. Supply
  the accepted Fire package report and every expected SHA-256 plus the exact
  freshly built `loadmodel_manifest`; do not hand-copy or rename the catalogue.
  The generated lock is private integration evidence, not release approval.

## 4. Build and test exact artifacts

- Build fresh macOS APP, AUv2, and VST3 plus iPhone/iPad targets from the frozen
  source and exact packages.
- Redirect Xcode products, intermediates, and staging roots to a new external
  build directory; a clean build must not consume or recreate in-tree
  `build-mac`, `build-ios`, DerivedData, or CMake output.
- Run capture companion, trainer, cloud-helper, resource-preparation, Core,
  package/manifest, state-layout, preset, and exact staged AU tests from the
  runbook.
- Build the final iOS/iPadOS archive with the framework-scoped
  `PrivacyInfo.xcprivacy`, generate Xcode's privacy report, and reconcile every
  reported required-reason API, collected-data type, tracking declaration, and
  listed third-party SDK against the exact archive. An unsigned simulator build
  is useful preflight evidence but is not this archive gate.
- Prove that `config.h`, all iOS and macOS format bundle plists, installer
  metadata, and each Xcode project's `CURRENT_PROJECT_VERSION` /
  `MARKETING_VERSION` surfaces agree. Resolve or parameterize inherited
  template signing-team metadata before selecting the owner team; do not bake
  a personal signing account into the public template. DessAmp leaves the
  active-project signing team blank and optionally reads the ignored
  `dessamp-signing.local.xcconfig`; copy the checked-in `.example` file or pass
  `DEVELOPMENT_TEAM` and `CODE_SIGN_IDENTITY` as private CI/command-line
  overrides.
- Pass the exact staged AU binary path to component tests; installed discovery
  is not release evidence.
- Record SHA-256 hashes for executables, embedded packages, license manifests,
  installer, and disk image.

## 5. Exercise product behavior

- Open the exact standalone and plugin editors. Exercise every active control,
  amplifier navigation, model relinking, presets, cabinet slots, bypass, custom
  IR loading, and state recall.
- Inspect macOS, iPhone, and iPad layouts at supported sizes and orientations;
  retain screenshots tied to exact build hashes.
- Perform level-matched owner listening in Logic. Automated ESR and render
  checks cannot approve tone.

## 6. Sign, package, and notarize

- Use an owner-approved Apple identity and clean keychain context. Never store
  credentials or exported identities in Git, logs, fixtures, or memory.
- Sign nested code from the inside out, verify with `codesign`, and validate the
  Audio Unit before creating the installer/DMG.
- Verify the license bundle and build-information/checksum files inside the
  final package.
- Submit the exact signed artifact for notarization, staple the accepted ticket,
  and verify Gatekeeper on a clean system.

## 7. Publish only after approval

- Prepare release notes with supported systems/formats, known issues, package
  provenance, checksums, and support/security links.
- Obtain separate owner approvals for source publication, each bundled model or
  asset, signing/notarization, and public release.
- Retain the immutable source archive, artifact hashes, test/evaluation reports,
  signing/notarization evidence, and release notes.
