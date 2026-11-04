# Repository hygiene and clean first launch

## Keep three boundaries separate

- Source: player, Core, capture/trainer code, locked dependencies, licenses,
  portable tests and documented test fixtures.
- Private evidence: captures, model catalogue, IRs, run ledgers, checkpoints,
  build reports and owner presets. Keep these outside Git and preserve them.
- Rebuildable output: external Xcode/CMake build directories and Python
  environments. Never build into the source checkout for a release proof.

`tools/stage_public_source.py --destination /absolute/new/external/source`
creates a hash-inventoried local source candidate without Git history. It
excludes upstream framework examples/tests/documentation, unused Eigen
development subtrees, internal reports, caches, installed bundles and user
presets. This is a source copy, not `git add`, and never marks a release ready.
Conditional artwork/fonts require the explicitly named local-validation flag
for build testing; that flag does not grant publication rights.

The pruned vendor snapshots retain the headers, runtime code, build dependencies
and licenses used by DessAmp. Their upstream framework example CI and Eigen's
standalone development CMake project are not supported in this reduced tree.
Use DessAmp's own targets and native Core CMake entry point. Do not remove
legacy host parameters, package formats, named test fixtures, alternate-platform
source or the retained trainer merely because the main UI does not expose them.

Before a source upload, review the generated inventory, run the source-hygiene
tests and clean-source builds, and perform the independent license, dependency
and secret reviews in [release-process.md](release-process.md). An old commitless
Git directory can still contain private unreachable objects; preserve it as a
recovery archive and initialize a separate clean history only when authorized.

## Reversible clean-install QA on macOS

1. Close DessAmp and DAWs hosting it. Do not reset another audio product or a
   DAW's global preferences/cache as part of a DessAmp reset.
2. Inventory and hash the exact installed app/AU/VST3 and their resources.
3. Move the DessAmp application-support folder (including its preset library),
   DessAmp-specific preference domains and DessAmp AU user-preset folder to an
   external dated recovery directory. Clear only those backed-up preference
   domains from the preference service. Preserve old Logic projects unchanged.
4. Install the exact newly tested bundles at the canonical locations, keeping
   the replaced bundles recoverable outside the active plugin search paths.
5. Open a fresh standalone instance or insert a new plugin in a new session.
   An old Logic project recalls its embedded state and is not a first-run test.
6. Verify one canonical `settings.ini`, private per-user directory permissions,
   an empty user preset list, Model Default, the bundled cabinet, device setup
   and audio. Preserve test-generated state separately if the owner wants the
   next launch to remain pristine.

The macOS host's first-directory branch uses `settings.ini` without a Windows
separator and creates the directory with owner-only permissions. The regression
test compiles and executes that actual branch without operating audio hardware.
Operating-system microphone permissions and audio-device routing are separate
from plugin preferences; do not erase or grant permissions automatically.

Legacy DessMetal is a distinct plugin identity. Removing it may break old DAW
projects, so archive it only with explicit scope confirmation. A new DessAmp
installation does not migrate or replace those older inserts.
