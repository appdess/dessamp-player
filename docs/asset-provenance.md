# UI asset provenance

## 2026-11-04 pedal and cabinet shells

Nine original transparent images were generated specifically for DessAmp and resized to a 720-pixel longest edge for the native editor. The source images depict unbranded enclosures. Labels, level knobs, switch state, hit targets and selection are drawn in code. Color identifies the drive slot. The assets are conditional listening-build resources pending owner review for redistribution.

| File in the DessAmp artwork directory | SHA-256 |
| --- | --- |
| `studio-amber@2x.png` | `a380b7ef351665d24e2e383499bc996b74780352fbfcd07ee376d386e2957c6b` |
| `studio-cab@2x.png` | `64cdf923cbfec51d64f49b6538aa58d9af5c1a9865706f05f43d04269b411507` |
| `studio-nocturne@2x.png` | `432a4fcdf750a9c69b6f75ab3ca7ee79874ecb272ffcdc29b18faaaa13d0e68b` |
| `studio-oxide@2x.png` | `3b39770b6cd7b82356b9bbbaebadffd7bab80d26c44805d5272079f16f766e74` |
| `studio-verdant@2x.png` | `50940586068a10c726b1764ee4f42f3239ca8d6aa63322cf3b644e5726cb4e6d` |
| `studio-cab-blend@2x.png` | `ce019e7714926cf4383db7dda4e162b84f870c51571eb0f0f2b842ca15c76a16` |
| `studio-cab-dzl@2x.png` | `dc8034930b21581f7b75071a796d761dda23f1d8a5e1e1d918a2cce8d562227e` |
| `studio-cab-green@2x.png` | `446f7d88bb9e88ac68a5bd8f787485c844d0ff4e5e666b678a7913c281a545a7` |
| `studio-cab-recto@2x.png` | `d9635053ff5a83734c018d0e8aec4e5c8b87318f2b5a16f0ccf487a2153b1c95` |

The owner described the four captures as DZL, SHFLD Mix, Gleichrichter and Green. SHFLD Mix combines two cabinets in one captured impulse response. Its illustration pairs steel mesh and worn cloth in a single enclosure. The remaining shells depict a modern metal grille, an oversized straight cabinet and a vintage basketweave cabinet. Gleichrichter follows the DZL camera angle, with a shallow visible top panel and a continuous straight roofline above its woven black grille. Loose WAV files use a separate generic shell. The original WAVs stay outside Git. The four owner-selected Version C derivatives are pinned in the private repository under `NeuralAmpModeler/resources/FactoryBank/`.

## DESS AMP header mark

The [transparent vector master](../NeuralAmpModeler/resources/Brand/dess-amp-wordmark.svg)
was drawn for this player. The seven custom letter outlines borrow the squared
shoulders and angled cuts of the existing DESS grille badge. Gradients provide
the silver face and bevel. No typeface or third-party artwork is embedded.
The app and plug-in use the same 960 × 124 raster, `dess-amp-wordmark@2x.png`,
from the optional runtime artwork. The separate DESS badge appears on the amplifier itself.

## 2026-08-19 player shell

The first DESS AMP head shell, knob face, and grille badge were generated specifically for this repository with OpenAI image generation. The prompts required original, unbranded hardware imagery and prohibited copied third-party logos, names, faceplate text, controls, labels, switch positions, and exact silhouettes.

Repository copies:

- `NeuralAmpModeler/resources/DessAmp/amp-head-shell@2x.png`, SHA-256
  `ed5e4e2313d9c786a7a2c84320a1e37a36fdfe5ce9c4f625b89d64477cc9b66a`;
- `NeuralAmpModeler/resources/DessAmp/amp-knob-face@2x.png`, SHA-256
  `3a887fadde80b6ed893d93464a1ff603b22a7d7ef02941751079ad026cf67902`;
- `NeuralAmpModeler/resources/DessAmp/dess-badge@2x.png`, SHA-256
  `254a0ab3e59a9f9eb54e1a0a5e33a3802a5e9b93c049211dcb5d3a550ed74a15`.

Review notes:

- transparent backgrounds are present;
- the head faceplate is intentionally blank;
- the knob face contains no baked pointer, numbers, or label;
- the badge spells `DESS` and remains a separate layer;
- all interactive state is rendered in code;
- reference material informed only generic high-gain-head vocabulary and functional control topology.

This note records creation provenance. It is not a trademark clearance or a decision to redistribute trained models, impulse responses, or third-party artwork.

## Retired NeuralAmpModelerPlugin UI resources

The former 30-file `NeuralAmpModeler/resources/img/` folder came from official
`sdatkinson/NeuralAmpModelerPlugin` commit
`16be869746b8915885c7a35bafdcc0061faeb50e`. A 2026-08-25 audit proved every
file byte-identical to upstream. The active DessAmp UI did not load those
files: only a compiled-out migration block and obsolete resource-copy metadata
still referenced them.

That dead block, its configuration aliases, Apple project references, Windows
resource entries, and Apple resource-copy paths were removed. The web resource
packager now consumes only the six repository-owned DessAmp shell assets. The
30 upstream files were moved intact and recoverably to
`$HOME/.codex/quarantine/dessamp-retired-upstream-img-20260825.OOq8gJ/img`;
they are no longer public-source or binary inputs. The DessAmp root and
installer licenses still preserve upstream NeuralAmpModelerPlugin source
attribution, independently of those retired images.

## 2026-08-25 application icon

The owner-selected DessAmp application icon was generated specifically for
this repository with the built-in OpenAI image generation tool and refined
through owner-directed visual review. The selected composition uses an
original silver `D` monogram, a smaller black amplifier knob, and a red
waveform/accent on a dark textured field. The final revision deliberately
makes the `D` brighter and the knob smaller so the product mark remains
legible at Dock, launcher, iPhone, and iPad sizes. It contains no third-party
name, logo, faceplate, or copied product artwork.

The single retained design master is:

- `NeuralAmpModeler/resources/Brand/AppIconSource/DessAmp-AppIcon-master.png`
- dimensions: `1254 x 1254`, RGB, no alpha channel;
- SHA-256:
  `6ce25aee237089ef50d14fa841e2e93458ced864fb7f2b780fec838725cf1ca6`.

`NeuralAmpModeler/scripts/generate_app_icons.py` verifies that exact master
hash and metadata, mechanically resizes it with the macOS `sips` tool, and
atomically writes the eleven filenames declared by the shared Apple asset
catalogue. `--check` fails if either the master or a checked-in derivative
changes without review. The macOS 1024-pixel derivative and the unified iOS
1024-pixel source are byte-identical with SHA-256
`1b768b7b212cbbd75374dcd812afcc0f2d72f2a8e33176871c4e7eaadcc23f61`.
Fresh macOS and iOS simulator builds report every compiled icon rendition as
opaque.

Eight unused design variants were moved recoverably outside the repository to
`$HOME/.codex/quarantine/dessamp-player-20260825T0005-generated/private-assets/unused-app-icon-candidates-20260825`.
They are not clean-history inputs. This section records original creation and
deterministic derivation; final public redistribution remains an explicit
owner release decision rather than an implication of a successful build.

## 2026-08-24 dual-red-modern shell

`NeuralAmpModeler/resources/DessAmp/amp-head-shell-dual-red-modern-v1@2x.png`
was generated specifically for the repository with the built-in OpenAI image
generation tool, then mechanically resized to the existing shell contract
(`1862 x 845`). Repository SHA-256:
`312630120ed236af06f9536d23514dafc68dbc2764d213b02bd33d6a73de03bd`.

The prompt requested an original, unbranded, front-orthographic modern
high-gain dual-channel head with black and gunmetal materials, restrained red
accents, warm tube glow, a clean uninterrupted perforated center grille, and a
completely blank lower fascia. The final revision deliberately removed the
intermediate center slats so the code-rendered `DESS` / `REDLINE` identity
remains legible. It explicitly prohibited knobs, switches, text, labels, logos,
badges, manufacturer marks, watermarks, copied faceplates, recognizable trade
dress, brand-specific silhouettes, and chrome diamond plate. The DESS badge
and every interactive control remain separate code-bound layers.

The package manifest selects this repository-owned shell only through the
bounded token `dual-red-modern-v1`; no third-party or package-supplied image is
loaded. This provenance record does not authorize redistribution of any model
trained or distilled from third-party captures.

## 2026-08-24 Forge and Fire product shells

The following two blank, non-interactive amplifier shells were generated
specifically for this repository with the built-in OpenAI image generation
tool and normalized to the shared `1862 x 845` shell contract:

- `NeuralAmpModeler/resources/DessAmp/amp-head-shell-forge-streamlined-v1@2x.png`
  - SHA-256: `0d3192970c8df1ecfb0a6d83ea60d70d263fe1d4f4539fc3fbbe9bc77d8838a1`
- `NeuralAmpModeler/resources/DessAmp/amp-head-shell-fire-chrome-v1@2x.png`
  - SHA-256: `6b14615bd85f4a4c7604652456043e39d9f1ded75c92559009cb0402f22501d4`

The Forge prompt requested an original streamlined black head with a circular
perforated grille, angular dark braces, restrained red accent rails, unlit
glass valves on the left and right, and a blank lower panel. The Fire prompt
requested an original black head with an elongated hex grille, a thin bright
metal grille perimeter, more prominent wraparound corner protectors, dark
braces with restrained red accents, symmetric warm valves, and a blank lower
panel. Both prompts prohibited copied faceplates, third-party names, logos,
controls, labels, pointers, values, switches, badges, watermarks, and
recognizable brand-specific trade dress.

The `DESS` badge, `FORGE` / `FIRE` names, every control, and all interactive
state remain separate code-bound layers. Forge valve colour and intensity are
also rendered continuously in code from the live morph value. The bounded
manifest tokens are `forge-streamlined-v1` and `fire-chrome-v1`; package-supplied
image paths are never loaded. This creation record is not authorization to
redistribute any trained amplifier model or third-party capture.

## Restricted checkpoint-loading fixture

`trainer_source/demo_ckpt.ckpt` is retained solely as the fixed legacy test
fixture for PyTorch `weights_only=True` checkpoint loading. It is not a player
model, capture result, cloud resume point, or distributable amplifier asset.
Its repository SHA-256 is
`475fa5533feae501e19afece67fa2ff1d428447c099591b0e39bf29ed6b857bf`.
All real training checkpoints remain ignored and external.

## Hash-bound native and trainer fixtures

The public source retains exactly eight small model/audio/checkpoint files for
native parser, runtime, benchmark, packaging, loudness, and restricted-loader
tests. None is a product amplifier, owner capture, cabinet IR, cloud result, or
candidate catalogue package. The public-source hygiene test enumerates this
closed set and verifies every SHA-256 so a run output cannot silently replace a
fixture.

| Fixture | SHA-256 | Provenance and bounded role |
| --- | --- | --- |
| `NeuralAmpModelerCore/example_models/lstm.nam` | `df9f78c49f49c2bb32411df47e3f53746075adb206b92d017e06379d1e56234a` | NeuralAmpModelerCore MIT source fixture; native parser/runtime test only |
| `NeuralAmpModelerCore/example_models/my_model.nam` | `ceb53469a19ce278e2235da982ae676cb8d5451a8de22a7ecc7a2617d07224d1` | NeuralAmpModelerCore MIT source fixture; A1 runtime/benchmark test only |
| `NeuralAmpModelerCore/example_models/wavenet.nam` | `66bda2b379289eff079c0755588bc9a92760654d9cc9af1b97cf30d0e92b167d` | NeuralAmpModelerCore MIT source fixture; native WaveNet parser test only |
| `NeuralAmpModelerCore/example_models/wavenet_a1_standard.nam` | `ceb53469a19ce278e2235da982ae676cb8d5451a8de22a7ecc7a2617d07224d1` | Byte-identical named benchmark alias of `my_model.nam`, retained for Core scripts |
| `NeuralAmpModelerCore/example_models/wavenet_a2_max.nam` | `e0d04507c3d39545e3b4ff034ef495706ade0fc5a0270a7fdc66ae1791711a09` | NeuralAmpModelerCore MIT source fixture; A2 parser/weight-generation test only |
| `NeuralAmpModelerCore/example_models/wavenet_condition_dsp.nam` | `99fcb223729795e2870c75c923ff8c86eeb23ed648ab71ac799efaeffaf3fee6` | NeuralAmpModelerCore MIT source fixture; conditioned-DSP parser test only |
| `trainer_source/demo_ckpt.ckpt` | `475fa5533feae501e19afece67fa2ff1d428447c099591b0e39bf29ed6b857bf` | Retained PANAMA MIT test fixture; restricted `weights_only=True` rejection/loading tests only |
| `trainer_source/nam/models/_resources/loudness_input.wav` | `84f3c44ddb228909d7ed7bbfeb4c12d0bb761cfd24c2b1925a96071df9392be5` | Retained PANAMA MIT trainer resource; import-time loudness calculation and package completeness tests only |

The applicable source licenses remain
`NeuralAmpModelerCore/LICENSE` (Steven Atkinson, MIT) and
`trainer_source/LICENSE` (ETH DISCO, MIT). The duplicate A1 bytes are retained
because current benchmark and loader scripts address both stable filenames;
their 398 KiB duplication is intentional test compatibility, not an amplifier
catalogue.

## 2026-08-19 operational semibold face

`NeuralAmpModeler/resources/fonts/Roboto-SemiBold.ttf` is a static 600-weight,
100-width instance generated from the official Roboto 3.015 variable font. The
source archive was downloaded from the Google Fonts Roboto release:

- source: `https://github.com/googlefonts/roboto-3-classic/releases/download/v3.015/Roboto_v3.015.zip`
- archive SHA-256: `bee3e9334ea86dd63e184598f31fb16750881c2da1a6f097a66e0f66a95b3d54`
- source font: `web/split/Roboto[wdth,wght].ttf`
- source-font SHA-256: `d7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134`
- instance command: `python3 -m fontTools.varLib.instancer Roboto-variable.ttf wght=600 wdth=100 --update-name-table --output Roboto-SemiBold.ttf`
- repository font SHA-256: `9da280a8e5b3b0a6ae98f8994a6d33d7f5f59f38020bbd8a541bd6c48856588d`

The resulting font has no variable axes, reports `OS/2.usWeightClass=600`, and
uses the `Roboto-SemiBold` PostScript name. It is distributed under the SIL
Open Font License 1.1 retained at
`NeuralAmpModeler/resources/fonts/OFL-Roboto.txt`.
