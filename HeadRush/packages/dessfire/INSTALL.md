# Install DessFire on HeadRush Flex Prime

Use this private Mac package with a **HeadRush Flex Prime HG12 on firmware 5.2.0.2a405ee**, an Apple silicon Mac with Python 3.9+ and Homebrew libusb, and a USB-B data cable. Confirm the ZIP hash against the repository’s package manifest before opening it.

1. Unzip the package and run **1 - Verify.command**. This checks both maintenance images offline.
2. On the pedal, open **Menu → USB Transfer**. Run **2 - Stage user files.command** on the Mac. It copies the amp, selector, matched cab and reference rig, skips identical files and stops on conflicts. Eject the HeadRush drive, tap **Sync**, and load **DESS FIRE - Native**.
3. Open **Menu → Firmware Update → Via USB** on the pedal. Run **3 - Install DessFire.command**. Keep power connected through the automatic reboot.
4. On the returned rig screen, tap an empty slot and choose **Amp → DessFire → DessFire**. Open the block to use Lead Gain, Bass, Middle, Treble and Presence. Add the matched cab for the sound used in testing.

**Confirm startup.** The user drive’s `DessAmp/FIRE-HG12-dev1/dessfire-app-maintenance.json` must report `install` and result `0`. With the saved reference rig enabled, the local editor’s `/static/dess-fire-hg12-entry.json` must show `installed=1` and `dessfire_ui=1`; two reads of `/static/dess-fire-hg12-live.json` must show an increasing `blocks` count. Then disconnect USB and Wi-Fi, power down normally, remove power, restart and play. Reconnect Wi-Fi for readback if needed. The packaged persistent upgrade has passed offline checks; a physical install and disconnected cold boot of this package are still pending.

**Roll back.** Enter **Firmware Update → Via USB** and run **Roll back to v3.command**. Require `rollback` and result `0` in the same receipt. This restores the v3 runtime. The inactive app and user-drive assets stay on the device. For complete v3 startup-hook removal, follow the v3 recovery instructions after rollback. To bypass the adapter on the next boot, put an empty `DessAmp/Disable FIRE.txt` on the user drive.

The ZIP is a Mac installer. The Mac sends a checked maintenance session to RAM. The installer writes the app to the pedal’s data partition and uses the local v3 startup hook on each boot. The stock firmware and bootloader stay in place.
