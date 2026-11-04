# Third-party and asset notices

DessAmp source code is distributed under the root MIT license. The binary also
incorporates or links the following projects. Their full license files are
retained in the source tree. The macOS and iOS resource-preparation scripts
copy this notice and the exact allowlisted license files into a
`THIRD_PARTY_LICENSES` directory and generate its license manifest with the
source-relative path and SHA-256 of every copied file. Final APP, component,
installer, and disk-image builds must still verify that generated bundle.

| Component | License | Source license file |
| --- | --- | --- |
| NeuralAmpModelerPlugin source lineage | MIT | Root `LICENSE`, preserving Steven Atkinson's upstream 2022 copyright notice |
| NeuralAmpModelerCore | MIT | `NeuralAmpModelerCore/LICENSE` |
| AudioDSPTools | MIT | `AudioDSPTools/LICENSE` |
| iPlug2 and Cockos WDL portions | zlib-style / component notices | `iPlug2/LICENSE.txt` |
| Eigen | MPL-2.0 plus listed BSD/Apache/Minpack portions | `NeuralAmpModelerCore/Dependencies/eigen/COPYING.*` |
| VST3 SDK | MIT | `iPlug2/Dependencies/IPlug/VST3_SDK/LICENSE.txt` |
| REAPER SDK headers retained with iPlug2 (not linked into current DessAmp targets) | zlib-style | `iPlug2/Dependencies/IPlug/REAPER_SDK/LICENSE` |
| NanoVG | zlib | `iPlug2/Dependencies/IGraphics/NanoVG/LICENSE.txt` |
| NanoSVG | zlib | `iPlug2/Dependencies/IGraphics/NanoSVG/LICENSE.txt` |
| MetalNanoVG | MIT | `iPlug2/Dependencies/IGraphics/MetalNanoVG/LICENSE` |
| Yoga | MIT | `iPlug2/Dependencies/IGraphics/yoga/LICENSE` |
| nlohmann/json | MIT | `licenses/nlohmann-json-MIT.txt` |
| RtAudio | MIT-style | `iPlug2/Dependencies/IPlug/RTAudio/doc/doxygen/license.txt` |
| RtMidi | MIT-style | `licenses/RtMidi-MIT.txt` |
| stb_textedit | MIT (selected from dual-license terms) | `licenses/stb-MIT.txt` |
| CHOC warning-control headers retained in source | ISC | `licenses/CHOC-ISC.txt` |
| SST Basic Blocks Lanczos resampler adaptation | file-specific MIT/BSD alternative to GPL-3.0-or-later | `licenses/SST-LanczosResampler-dual.txt` and the full notice in `AudioDSPTools/dsp/ResamplingContainer/Dependencies/LanczosResampler.h` |
| Roboto Regular | Apache-2.0 | Font data copyright Google 2012; `licenses/Roboto-Regular-Apache-2.0.txt` |
| Roboto SemiBold static instance | SIL Open Font License 1.1 | Copyright 2011 The Roboto Project Authors; `NeuralAmpModeler/resources/fonts/OFL-Roboto.txt` |
| Michroma Regular | SIL Open Font License 1.1 | Copyright 2011 The Michroma Project Authors; `licenses/Michroma-OFL-1.1.txt` |

The deterministic public-source staging tool omits upstream iPlug2 examples,
tests, and documentation plus Eigen benchmarks, BLAS/LAPACK adapters, tests,
demos, documentation, unsupported modules, and build-system extras. It also
omits two superseded duplicate Eigen/nlohmann snapshots: every DessAmp player,
Core evaluator, and AudioDSP utility resolves the single Core-owned dependency
root. Those uncompiled upstream-only and duplicate trees are not part of the
DessAmp source candidate or binary notice bundle. The retained canonical Eigen
headers and applicable MPL/BSD/Apache/Minpack license files remain complete for
every DessAmp target.

The development-only standalone Trainer also reuses the PANAMA conditioned-
model training source from ETH DISCO under the MIT License retained at
`trainer_source/LICENSE`. The Trainer is not included in the current DessAmp
plug-in DMG; any future packaged Trainer must carry that copyright and license
text with the distribution. Its default Python graph is frozen by
`trainer_source/uv.lock`; generate and review the CycloneDX and resolved
platform license inventories as described in `trainer_source/README.md` before
distributing a Trainer environment. A dependency name in the lock is not a
substitute for the applicable license/notice text shipped by its wheel.

## Bundled captures, IRs, and artwork

DessAmp treats source licensing, trained captures, cabinet IRs, fonts, icons,
and UI artwork as separate rights boundaries. The root MIT license does not
license those binary assets.

The public-release boundary contains no approved fixed amp catalogue, bundled
`.dess` model, or PRE drive capture yet. Private validation builds may embed
explicitly selected packages, but that does not grant redistribution rights.
The four historical DSP-wired PRE drive captures (`OD808.nam`, `SD1.nam`,
`TS9.nam`, and `aesahaettr.nam`) remain outside the repository and ordinary
builds. A local build may include the complete private bank only through
`DESSAMP_PRIVATE_DRIVE_DIR` plus
`DESSAMP_PRIVATE_DRIVE_VALIDATION=YES`. A distributable bank must instead be
placed in `NeuralAmpModeler/resources/Drives` after file-level approval. Normal
builds source cabinet IRs from `NeuralAmpModeler/resources/IRs`; private
listening candidates require the explicit `DESSAMP_PRIVATE_IR_VALIDATION=YES`
build gate and remain outside the source-release boundary.

The copied DessMetal history records an owner declaration and a public-release
authorization for that earlier product's v0.1.1 asset set. That historical
authorization is useful provenance evidence, but it does not automatically
authorize publication from this separate DessAmp repository. Before a public,
signed DessAmp release, record explicit file-level redistribution approval for
every bundled drive capture, release IR, standalone icon, and each of the six
current DessAmp artwork files. Private validation builds are not public-release
evidence.

Third-party product names and marks, where present in private provenance or
test evidence, belong to their respective owners and are descriptive only.
DessAmp is not affiliated with, sponsored by, or endorsed by those owners.
