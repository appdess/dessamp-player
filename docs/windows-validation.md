# Windows x64 validation

The Windows targets are the standalone and VST3. The player targets x64 in both environments below. This workflow has a source and packaging preflight on macOS. Windows compilation, installation, upgrade, uninstall and live host playback require native Windows evidence.

## Choose the host

On an Apple silicon Mac, use Windows 11 Pro Arm with a VM product supported for that Mac. Install Visual Studio 2022 Desktop development with C++, v143 x64 tools, a Windows SDK, Python 3.12 and Inno Setup 6.3 or newer. Start in the Visual Studio Developer Command Prompt. The Arm-hosted MSVC compiler can target x64. Run the resulting standalone and an **x64 DAW** in the guest. The kernel requires Arm64-compatible drivers. A native x64 Windows machine with the supported interface and driver is the final real-time audio test. [Microsoft on Apple silicon](https://support.microsoft.com/en-us/windows/experience/platform-variants/options-for-using-windows-11-with-mac-computers-with-apple-m1-m2-and-m3-chips), [Microsoft compiler targets](https://learn.microsoft.com/en-us/visualstudio/install/visual-studio-on-arm-devices), [Inno architecture identifiers](https://jrsoftware.org/ishelp/topic_archidentifiers.htm).

Budget a 128 GB guest disk as a planning allowance. An external SSD is a practical local destination when the internal disk is nearly full. A native x64 Windows Server GCE VM can run build and installer checks. Confirm the non-production project, cost ceiling, snapshot transfer scope, image licensing and disk teardown before provisioning. [Google Cloud Windows licensing](https://cloud.google.com/compute/docs/instances/windows/ms-licensing-faq).

## Build from a reviewed export

On the Mac, use a fresh private validation export of the current source. Conditional artwork and fonts are required to compile this local editor. Review the inventory before any transfer.

```sh
python3 tools/stage_public_source.py --destination /absolute/new/windows-source \
  --include-conditional-assets-for-local-validation
```

On Windows, extract on the guest's local disk. Keep the inventory alongside the source root. From that root run the following in the Developer Command Prompt.

```bat
NeuralAmpModeler\scripts\makedist-win.bat full installer
```

This builds both Release x64 targets, stages the VST3 under `Contents\x86_64-win`, and prepares notices from the inventory. Set `ISCC` to the full compiler path if it is outside the default Inno Setup 6 directory. The unsigned installer is in `NeuralAmpModeler\build-win\installer`. ZIP outputs and separate symbols are in `build-win\out` under the player directory. A full source-only installer has no listening bank.

For local listening QA, use a fresh guest snapshot. Supply the reviewed external bank and the SHA-256 of its **independently retained** input ledger. The command verifies an exact ten-file set and generates explicit installer entries for the APP and VST3. The three heads use four packages, including the two Redline voices. The bank also contains four drives, the Desfire IR, and the matching defaults. All of these files require separate redistribution approval before a public release.

The listening input ledger lists the four Version C WAVs under `IRs/`
and the matching profile under `Defaults/FactoryDefaults.json`, alongside the
amp and drive models. Copy the WAVs and profile from the private checkout's
`NeuralAmpModeler/resources/FactoryBank/` when preparing those inputs.

```bat
set "DESSAMP_WINDOWS_BANK=C:\DessAmpQA\private-build-inputs"
set "DESSAMP_WINDOWS_BANK_LEDGER=C:\DessAmpQA\private-input-hashes.json"
set "DESSAMP_WINDOWS_BANK_SHA256=replace-with-reviewed-ledger-digest"
NeuralAmpModeler\scripts\makedist-win.bat listening installer
```

Keep this installer and its ZIP in the private QA environment. They contain the listening assets. Restore the clean guest between public and private installer tests.

## Acceptance evidence

1. Hash the installer and both executables with PowerShell `Get-FileHash -Algorithm SHA256`. Use `dumpbin /headers` to confirm x64. Build twice from separate exports and check the inventory, contents and version metadata in each run.
2. On a clean Windows snapshot, exercise full install, standalone-only and VST3-only. Check installed files, shortcut ownership, notices and VST3 discovery in an x64 host. Open both editors. Select the three factory heads and both Redline voices, try all four drive switches, load a custom mono WAV, change heads, bypass the cabinet, save over a preset, then reopen the session.
3. Exercise a non-ASCII preset path, a missing custom IR, host automation and an upgrade from the preceding installer. Uninstall and verify that application-owned files are removed and user presets survive. Reinstall from the retained installer hash.
4. Measure the final native x64 build at 44.1, 48 and 96 kHz with 32, 64 and 128-frame buffers. Include the head, drive, cabinet and Morph paths. Check deadlines, reported latency, physical round-trip latency and dropouts separately. Retain the host, driver, interface and artifact hashes with the results.
