# Design decision: Start FIRE from the pedal’s own storage

**Accepted on 4 November 2026 for the HeadRush Flex Prime HG12 running 5.2.0.2a405ee.** The one-time USB installer persists a NetworkManager dispatcher script and the FIRE adapter on the pedal. On a normal boot, a local hostname event starts the adapter before the HeadRush rig application launches. The player can power on and open **DESS FIRE - Native** without USB, Wi-Fi or a Mac.

## Why we chose this hook

The first prototype already ran the conditioned amp on the pedal. Its rig and model survived power loss, but its launcher lived only in RAM, so each restart required USB reactivation. Inspecting the stock firmware showed that Linux mounts a writable settings layer from its data partition. That gave us a place to install our own files while keeping the signed firmware and bootloader intact. The remaining question was which stock event would reliably run our code early enough.

| Physical trial | What we observed | What it told us |
| --- | --- | --- |
| Internal rig-drive event | The install receipt reported success and the saved rig reopened. The startup marker returned HTTP 404. | Seeing the rig name proved that the rig survived. It did not establish that the conditioned amp was running. |
| Earlier device rule and rig-service override | The boot report found the rules on disk, but the service manager listed no override for the rig service and recorded no preparation call. A diagnostic script registered with NetworkManager **did** run. | The service manager had already loaded the rig service before the writable settings were visible. The diagnostic exposed a stock hook that actually executed on this pedal. |
| NetworkManager dispatcher | On both the normal restart and the unplugged cold start, the dispatcher recorded `hostname` at eight seconds of uptime and completed preparation before the rig launcher ran. | The local hostname event met the startup deadline even when the owner powered on with Wi-Fi off. Neither tested boot needed to relaunch the rig application. |

## What persists and what runs

1. **Once, during installation.** The USB maintenance session validates the product, firmware build and FIRE files. It writes `10-dessamp-fire-local` under `/etc/NetworkManager/dispatcher.d/` and its `pre-up.d/` subdirectory. The runtime lives under `/data/dessamp-fire-hg12-v1/` and the entry helper under `/data/dessamp-fire-entry-hg12-v2/`. Those paths are on the pedal. They survive power loss.
2. **Once per normal boot.** The local event runs our script. It checks the target and prepares temporary views of the stock launcher and status directory in RAM. The normal HeadRush rig app then starts with the adapter available. NetworkManager’s role ends at startup.
3. **When FIRE is selected.** The adapter recognizes the verified `.nam` selector and loads the original conditioned `.dess` model. The pedal’s ARM processor runs inference. Selecting a different amp stops FIRE processing. Recalling the FIRE rig resumes it with its saved controls.

## How we proved independent playback

The owner unplugged USB, turned Wi-Fi off, removed power, restarted and played the rig offline. Wi-Fi was restored afterward for readback. The pedal reported a new boot ID and the same eight-second trigger. The FIRE audio counter advanced from 65,538 to 67,539, held at 82,524 on another saved rig, and reached 83,359 after FIRE was recalled. That last check distinguishes the live conditioned amp from the fixed capture that the same selector can play on stock firmware.

The physical evidence covers this pedal and firmware build. If the trigger arrives late, the implementation permits one rig-service relaunch in the first 45 seconds only after confirming that the saved FIRE rig is clean. That fallback passed a local harness test. If an attempted launch never reaches playback, a persistent marker makes the next boot take the stock path. The full checkout contains the operating guide at `HeadRush/targets/hg12/startup/README.md` and the recovery procedure at `HeadRush/targets/hg12/recovery/RESTORE.md`.
