# DessFire renderer on Flex Prime HG12

The selected renderer, `mem3`, groups three frames in the eight-channel ARM NEON convolution, reuses weights across frames and refreshes only the changed span of mirrored history. The output convolution also uses NEON. This is a Cortex-A17 build of DessAmp Core with the original five inputs and weights. Each 48-sample callback at 48 kHz has 1 ms to finish. [Renderer source and build](build_renderer.sh)

## Measured on firmware 5.2.0.2a405ee

| Identical rig configuration | Original DSP, median CPU | Selected DSP, median CPU | Selected DSP, peak CPU |
| --- | ---: | ---: | ---: |
| DessFire and matched cab | 91% | 53% | 61% |
| Add AIR Delay | 92% | 55.5% | 63% |
| Add Eleven Reverb as well | 99% | 62% | 71% |

With both effects enabled, the original amp callback took a sampled median of 706 µs. The selected renderer took 309 µs. Across about 75 seconds, none of 75,118 native callbacks exceeded 1 ms. The highest callback time recorded since the final boot was 471 µs. The original combined-effects window was 14 seconds. The executable code and allocated file-backed data sections of the selected ARM library can be reproduced from the repository build script.

The physical pedal ran 24 screening cases and 18 confirmation cases across 12 configurations. None of the alternatives consistently beat the three-frame renderer. Its SHA-256 is `e080b9ecc6f632dcb6cd33f943c0b3b0672f6ead311eef7a448e4733fabb1e1f`. Whole-rig CPU above is the pedal meter. Callback timing measures only the DessFire amp.

## Qualify a changed renderer

* Compare ARM output with the original model at defaults, each axis extreme and during sweeps. Include varied buffer lengths and in-place processing. Require finite audio, error-to-signal ratio below 1e-7 and absolute sample error below 0.001.
* Benchmark on the target with fixed controls, sweeps and stepped controls. Repeat the finalists in alternating orders. Require consistent p95 improvement without additional deadline misses or a material p99 regression.
* Confirm the selected library inside the stock app. Repeat the amp/cab and combined-effects tests and collect both whole-rig CPU and native callback times. The private checkpoint contains the exact replay image and results. The full development checkout documents the package build in `HeadRush/packages/dessfire/BUILD.md`.
