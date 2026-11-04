# DessFire on HeadRush Flex Prime

DessFire adds a touchscreen entry under **Amp → DessFire → DessFire**. Its five faders control Lead Gain, Bass, Middle, Treble and Presence within the ranges captured for this model. This setup targets **Flex Prime HG12, firmware 5.2.0.2a405ee**.

## Install from a Mac

1. Use the separately supplied `DessFire-HG12-Mac.zip` for this exact device and firmware.
   The public source includes the Core renderer and operating instructions. The installer,
   model, stock firmware and recovery images are separate inputs. Compare the ZIP digest
   with [installer.sha256](packages/installer.sha256) before opening it.

2. Unzip it and run **1 - Verify.command**. Connect the pedal over USB-B, open **Menu → USB Transfer**, then run **2 - Stage user files.command**. Eject the HeadRush drive, tap **Sync**, and load **DESS FIRE - Native**.
3. Open **Menu → Firmware Update → Via USB**. Run **3 - Install DessFire.command** and let the pedal reboot. The Mac sends a one-time installer into RAM. It puts the app and its startup hook on the pedal’s data partition.
4. Use the included reference rig to play immediately. To add the amp to another rig, tap an empty slot, open **Amp**, tap the **DessFire** provider, then tap the **DessFire** model. Open the amp to reach all five controls. For a new installation, check the receipt, advancing audio counter and playback after an unplugged cold start. [Installation checks and rollback](packages/dessfire/INSTALL.md)

**Verification status.** The picker, five-control editor and selected renderer ran on the physical pedal in a one-boot trial. The v3 startup hook separately passed an unplugged cold start. The combined persistent installer has been built and tested offline. Its install and unplugged cold start on hardware are still pending.

## How it fits into the pedal

The stock HeadRush app owns the signal chain, slot, parameter storage and rig recall. DessFire adds the Amp picker row and five native-style faders in process memory. The rig stores a primary Neural Amp Modeler block and its selector file. When the adapter recognizes those exact selector bytes, it loads the original conditioned `.dess` off the audio thread. The callback passes all five controls to DessAmp Core. On stock startup the same selector plays the captured default setting. The cabinet is a separate IR block. [Conditioned renderer](targets/hg12/runtime/fire.cpp)

The installed startup hook runs locally at power-on. It checks the stock application and model, then launches the app with the adapter. The verified firmware and bootloader stay in place. The installer also carries the original v3 runtime for rollback. [Startup design and device checks](../docs/headrush-startup-decision.md)

The Cortex-A17 renderer processes three frames together with NEON and updates only changed history samples. With AIR Delay and Eleven Reverb active, whole-rig median CPU fell from **99% to 62%** on the tested pedal. The peak in that run was **71%**. The amp callback stayed inside its 1 ms deadline for all **75,118** calls in the sustained test. [Measurements and build settings](prototypes/dessfire/PERFORMANCE.md)

The full development checkout contains the model onboarding guide (`HeadRush/MODELS.md`), the Mac packager and the device setup skill.
