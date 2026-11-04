# DessAmp for iOS and iPadOS

This folder is the mobile product entry point inside the DessAmp monorepo.
It intentionally does not copy the DSP, package parser, or player UI sources.
The workspace references the iPlug2 App/AUv3 project while those shared sources
remain owned by `NeuralAmpModeler/`, `NeuralAmpModelerCore/`, `AudioDSPTools/`,
and `iPlug2/`.

For store readiness, see the [App Store preflight](APP-STORE-PREFLIGHT.md).

Open `DessAmp.xcworkspace` and use:

- `iOS-APP with AUv3` for the standalone live player and embedded extension.
- `iOS-AUv3` when validating the extension in a host.

The first mobile release keeps the exact conditioned A2 Full model at 48 kHz.
Do not substitute Lite, pruning, quantization, or a fixed snapshot to satisfy a
device benchmark. Workflow-validation packages may be used for integration and
performance testing, but they are not bundled release content unless their
external numerical and listening gates have passed.

## Interface contract

The standalone and AUv3 share one responsive editor. On iPhone, start in
portrait: the complete amp head sits above a compact grid of live controls. Landscape puts the
same controls on the amp face. Select a head at the bottom of either layout.
Use **PRE** for drives, **AMP** for learned controls, **CAB** for cabinet IRs
and bypass, and **RIG** for input, gate and output. The preset picker, **SAVE**
and **NEW** stay on the playing pages. Swipe left on the amp picture, page
background or tab strip for the next page; swipe right to go back. The ends
stop at PRE and RIG on iPhone, or PRE and CAB on full-size iPad. A drag that
starts on a knob still edits that knob. The four factory cabinet tiles stay
in place: tap one to load it and change the large preview. Rotation changes
where the controls appear without changing the selected amp or its parameters.
On a full-size iPad the rig controls remain visible above the three stage pages.
RIG is shared by every amp. Input trim, calibration, gate, threshold and output
stay in place across factory and named presets. The standalone remembers them
between launches; each AUv3 instance restores its own RIG from its host.

Tap **TUNE** in the header and play one open string. The chromatic tuner shows
the nearest note, octave and cents relative to A4 = 440 Hz. The marker moves
left for flat and right for sharp; within three cents the note turns green.
Tap **DONE** to return to the same page. The tuner listens to the clean input
and leaves the audible rig, recording and presets alone. In a host, live
audio must be reaching the plug-in. If the host stops rendering, the note clears.

Test each of the following separately:

- iPhone: portrait and landscape standalone layout, Files import, model load,
  control edits, audio-route stability, sustained real-time performance and
  thermal behavior.
- iPad: native portrait and landscape standalone layouts plus AUv3 host layout,
  Split View/window resizing where the host supports it, model load, control
  edits, and sustained real-time performance.

Simulator acceptance covers layout, navigation, file import, and model-package
integration. It is not a substitute for physical-device audio, latency,
dropout, interruption, and thermal acceptance.

The touch regression suite in `ios/UITests` uses a disposable iPhone 17
simulator with the private four-cabinet build installed. With XcodeGen,
generate the test project in a temporary directory and run:

```sh
qa=$(mktemp -d)
cp -R ios/UITests/. "$qa/"
xcodegen generate --spec "$qa/project.yml"
xcodebuild test -project "$qa/PhoneCheck.xcodeproj" -scheme PhoneCheck \
  -destination "platform=iOS Simulator,id=$DESSAMP_SIMULATOR_UDID" \
  -parallel-testing-enabled NO -resultBundlePath "$qa/Player.xcresult"
```

It exercises both swipe directions, the four cabinet slots, amp preset recall,
and a global gate change across amp switches, a preset load and app relaunch.
The factory rig test saves three uniquely named sounds in the simulator. Verify
their actual amp, cabinet, pedal and level settings using the **Factory QA run**
ID printed in the test log:

```sh
app_data=$(xcrun simctl get_app_container "$DESSAMP_SIMULATOR_UDID" com.dessdynamics.dessamp data)
python3 ios/UITests/check_factory_sounds.py "$app_data" "$FACTORY_QA_RUN"
```

### Recheck custom amp import and AUv3 playback

1. In the standalone, tap **CUSTOM**, choose a known `.dess` package in
   Files, and check its name, controls and ranges. Change a control, switch
   to FIRE, then return to CUSTOM. The imported amp should reopen from the
   app’s own `Library/Application Support/DessAmp/Amplifiers` directory.
   Its cached bytes must match the selected file. The AUv3 has its own
   sandbox, so test extension recall separately.
2. In a separate AU host, instantiate the DessAmp effect, open its editor,
   and check that the factory amp appears with its artwork and controls.
   Play a dry signal through the active effect at 48 kHz. Check for finite,
   non-silent output that differs from the dry signal. The standalone and
   extension share factory resources in the containing app bundle.
3. On iPhone, edit Lead Gain, tap **SAVE**, name the sound, then use the
   picker to switch between **Model Default** and the saved sound. **SAVE**
   updates a named preset; **NEW** writes a separate copy. FIRE, FORGE and
   REDLINE have separate banks. Rotate and check the value, then select a
   cabinet from the four fixed tiles; only the large preview and selection
   outline should change. On iPad repeat the rotation and navigation checks, with the rig
   controls in the shared utility strip.

## Local signing

The reusable project does not embed an Apple team or personal certificate.
For Xcode GUI work, copy `../dessamp-signing.local.xcconfig.example` to
`../dessamp-signing.local.xcconfig` and fill the ignored local copy. CI and
command-line archives may instead override the standard `DEVELOPMENT_TEAM` and
`CODE_SIGN_IDENTITY` build settings. Never commit signing identities,
provisioning profiles, exported certificates, or account credentials.
