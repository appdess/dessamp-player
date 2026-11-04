# Input calibration

DessAmp input calibration matches the plugin to the analog headroom of the
interface jack receiving the guitar. It is separate from the main **Input**
knob: calibration establishes the capture reference, while **Input** remains a
creative trim.

## Quick setup: Auto Input

Open **Settings > AUTO INPUT**, enable live monitoring in the host, and play
strong chords for five seconds. Keep the interface's clip indicator out of the
red. DessAmp measures the incoming signal before its Input/calibration gain,
then sets the existing Input knob once so the measured strongest peak reaches
approximately **-12 dBFS before Gate / Drive / Amp**. It leaves room for stronger
future transients; it does not normalize continuously or compress your playing.
Future peaks and the nonlinear amp's output can still clip, so check both the
interface and output meters and turn down Output if necessary.

Auto Input rejects near-full-scale/non-finite samples and very quiet input,
supports cancellation, and times out if live audio is not running. Changing
the amp, Input, or calibration during measurement cancels the result. It never
changes the interface's analog gain. A clipped recording cannot be repaired by
turning down a plugin; lower the interface gain or use an appropriate pad first.
The result is limited to the existing -20 to +20 dB Input range and reports when
that limit prevents the target. The resulting trim is saved with the preset;
the transient measurement itself is not saved and adds no host parameters.

**Auto Input is a convenient playing-level target, not physical calibration.**
Pickup voltage and playing strength are unknown, so a strum cannot reveal an
interface's dBu reference. With calibration enabled, Auto Input accounts for
that correction when setting Input; it may deliberately offset it to hit the
playing-level target. For the measured hardware response, use manual interface
calibration and leave Input at 0 dB instead. FORGE/REDLINE remain unverified in
absolute analog level even after Auto Input.

## Interface reference setup

1. Look up the interface specification named **maximum input level** for the
   exact input type you use (instrument/Hi-Z, line, or microphone) at minimum
   analog gain. Use dBu, not dynamic range or maximum output level.
2. Enter that value in **Settings > Input Calibration**. If you deliberately
   add analog input gain, subtract that gain from the published maximum input
   level.
3. Enable calibration and leave DessAmp's main **Input** knob at `0 dB` for the
   captured response. Adjust the main knob afterward only when you want a
   hotter or cleaner sound.

For a MOTU M2, the published maximum input level at minimum gain is `+16 dBu`
for the 1/4-inch instrument input and `+10 dBu` for XLR. Select the jack you
actually plugged into; “M2 has 16 dBu headroom” is only correct for the
1/4-inch path.

DESS FIRE contains its measured capture reference of `-0.54 dBu`. DESS FORGE
and DESS REDLINE do not have physical calibration evidence, so the plugin
clearly labels and uses a `0 dBu` fallback. This intentionally makes those
models hotter when calibration is enabled; it does not claim they are
physically calibrated.

Calibration is off by default so old sessions retain their sound. It is saved
inside the normal plugin/preset state and applies before Gate, Drive, and Amp.
