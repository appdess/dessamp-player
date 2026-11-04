# Input calibration

DessAmp input calibration matches the plugin to the analog headroom of the
interface jack receiving the guitar. It is separate from the main **Input**
knob: calibration establishes the capture reference, while **Input** remains a
creative trim.

## Match your playing level

1. Open **Settings** and set **Interface maximum input level** to the specification for your jack and gain setting. The number in **dBu** describes the analog signal your interface can accept.
2. Enable input monitoring, tap **Match Input**, and play strong chords for five seconds. On success, Settings shows **Calibration on** and **Input matched**. DessAmp enables the interface correction and sets the Input trim once, aiming for a peak of **−12 dBFS** before Gate, Drive and Amp.
3. Read **Input trim** and **Total gain** in **dB**. For example, a +10 dBu interface and the Fireball’s −0.54 dBu reference give a +10.54 dB correction. If the measured guitar peak is already around −12 dBFS, the match can set the trim to −10.5 dB. The two adjustments add up to about 0 dB, so the guitar stays near its original level.
4. Tap **Reset Input** to turn calibration off and return the Input knob to 0 dB. DessAmp keeps the interface setting for your next match.

The amp, your playing and the interface gain all affect the measured peak. The match cannot measure an interface’s analog headroom. For the response at the physical capture reference, use manual interface calibration and leave Input at 0 dB.

**Cancel Match** keeps the previous settings. Clipping, silence, interrupted audio, or a result outside the Input knob’s ±20 dB range also leave them as they were. If the interface clips, reduce its analog gain before trying again. Calibration, headroom and trim belong to RIG: switching amps or recalling a sound leaves them alone. The displayed total gain can still change because each amp has its own capture reference. The five-second measurement ends once the trim is set.

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

Calibration is off on first use. The standalone remembers RIG between launches;
a plug-in instance restores it with its DAW session. Calibration applies before the drive, gate detector and amp.
