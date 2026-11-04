# Vendored license extracts

This directory holds license texts for vendored single-header or source-only
dependencies whose upstream trees do not include a standalone license file in
this checkout. Each text is copied from the corresponding vendored source:

- `nlohmann-json-MIT.txt` from
  `NeuralAmpModelerCore/Dependencies/nlohmann/json.hpp`;
- `RtMidi-MIT.txt` from `iPlug2/Dependencies/IPlug/RTMidi/RtMidi.h`;
- `stb-MIT.txt` from the MIT alternative in
  `iPlug2/Dependencies/IGraphics/STB/stb_textedit.h`;
- `Roboto-Regular-Apache-2.0.txt` from the embedded font's name-table
  copyright and license metadata plus the Apache-2.0 text retained in
  `NeuralAmpModelerCore/Dependencies/eigen/COPYING.APACHE`;
- `Michroma-OFL-1.1.txt` from the embedded font's name-table copyright
  metadata plus the SIL Open Font License 1.1 text retained in
  `iPlug2/Dependencies/IGraphics/NanoVG/example/LICENSE_OFL.txt`.
- `CHOC-ISC.txt` from the complete ISC notice retained in
  `NeuralAmpModeler/choc_DisableAllWarnings.h` and
  `NeuralAmpModeler/choc_ReenableAllWarnings.h`;
- `SST-LanczosResampler-dual.txt` from the file-specific dual-license notice
  retained in
  `AudioDSPTools/dsp/ResamplingContainer/Dependencies/LanczosResampler.h`.

Other dependencies keep their full upstream license files in their vendored
directories; `THIRD_PARTY_NOTICES.md` is the authoritative index.
