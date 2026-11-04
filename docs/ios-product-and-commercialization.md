# DessAmp iPhone/iPad product and commercialization plan

Status: engineering and market recommendation, 2026-08-23. This is not legal
advice; release licensing, model rights, trademarks, privacy disclosures, and
creator agreements should receive a final lawyer review.

## Recommendation

Keep iOS/iPadOS in this monorepo through the first physical-amplifier release.
The mobile app should be a paid universal standalone app with its AUv3 extension,
offline import of `.dess` packages, and the exact conditioned A2 Full 48 kHz
runtime. Keep the capture/trainer workflow and package format open, and keep new
product-only iOS work private or under a separately declared proprietary license
if source secrecy matters.

Launch without a marketplace, subscription, account, or cloud dependency. Price
the founder release at EUR 19.99. Move toward EUR 29.99 only after the physical
device gates and gig-critical features such as MIDI control, robust preset/scene
switching, and host interoperability are complete. Do not charge a subscription
for offline inference. Add one only if there is durable recurring value such as
cloud training, sync, an expanding licensed catalogue, or creator services.

The positioning is not “another NAM loader.” It is:

> Your real amplifier, with its real controls, on iPhone, iPad, and AUv3 — one
> full-quality conditioned model rather than a collection of knob snapshots.

## Why the monorepo is the right boundary now

The app, AUv3, native Core, package validator, trainer, and `.dess` schema still
share a fast-changing contract. Keeping them together provides one atomic test
surface for manifest control order, hash validation, runtime compatibility, and
UI exposure. `ios/` is the mobile product entry point, not a copy of the DSP or
UI. It references the same Core and product sources used by the desktop player.

A later two-repository split is reasonable only after at least one real
amplifier ships and the `.dess` contract is versioned and stable:

- `dessamp-player`: proprietary product UI and distribution code plus the
  redistributable Core, package loader, and APP/AUv3 products;
- `dessamp-trainer`: open capture, active-learning, training, evaluation, and
  export workflow.

The Core should stay with the player because model compatibility and real-time
behavior are release-critical. A separate Core repository adds coordination
without creating a separate customer product.

## Current mobile engineering evidence

The repository now has one `iOS-APP with AUv3` workspace path, a universal
arm64 device build, file-document registration for `.dess`, and a shared
responsive editor. On iPhone, portrait shows the full head above the controls
and landscape places the same controls on the faceplate. The four phone tabs
are PRE, AMP, CAB and RIG; the last holds signal levels. The playing pages
provide a preset picker and direct save buttons. Full-size
iPad layouts keep the utility strip above the stage. The desktop, iPhone and
iPad editors share the same silver DESS AMP wordmark.

Current simulator validation covers:

- iPhone portrait and landscape rendering, Files import of a conditioned
  `.dess`, amplifier and cabinet selection, and control edits across rotation;
- iPad Pro portrait and landscape launch, native full-screen reflow without a
  compatibility letterbox, live rotation in both directions, Files import,
  loading the same byte-identical `.dess`, AMP/PRE navigation, and preservation
  of correctly aligned manifest controls across rotation;
- embedded AUv3 compilation and signing inside the standalone application.

The standalone launcher also has an explicit no-input Simulator path. A
zero-Hz input route no longer blocks the main thread during audio-session
activation: the initialized AU editor is presented without live audio, while
valid physical-device routes preserve the normal PlayAndRecord engine startup.
Main-queue editor containment/open/resize ordering and the invalid-route return
are source-regression tested. This keeps Simulator UI QA honest; it is not a
substitute for the physical-device audio matrix below.

The current Fireball R3 seed-82 package is integration evidence only. It passed
the package path but failed the frozen hidden numerical gate (mean ESR
`0.054686`, worst `0.061192`, required every setting below `0.035`). It must
remain `workflow_validation_only`; it is neither bundled release content nor a
quality claim.

The development-signed app installed and launched on a paired iPhone 16 Pro
running iOS 26.6. The standalone editor was observed on hardware. A separate
iPhone simulator host opened the AUv3 and rendered 46,080 frames at 48 kHz.
The physical audio checks below remain open: app launch and simulator playback
do not measure round-trip latency, interruptions or sustained thermal behavior.

## Full-quality device acceptance

“Same quality” means the same conditioned A2 Full equations and 48 kHz weights.
There is no silent Lite model, pruning, quantization, resampling, or snapshot
fallback. If a device cannot meet the gate, it is unsupported rather than given
a lower-quality mode.

Before TestFlight, run on the oldest intended supported iPhone and iPad:

1. Compare native current-source renders from Mac, iPhone, and iPad using the
   same `.dess`, DI, 48 kHz sample rate, and control automation. Require sample-
   aligned numerical agreement within the explicitly documented floating-point
   tolerance.
2. Measure round-trip latency at practical buffer sizes with the intended class-
   compliant interfaces. Report interface, buffer, sample rate, and measured
   milliseconds rather than only the software block size.
3. Run at least 60 minutes with continuous audio and control changes. Record
   underruns, CPU, thermal state, battery/power behavior, and route changes.
4. Exercise headphones, USB interfaces, Bluetooth rejection/warnings, calls,
   Siri, background/foreground transitions, screen lock, cable removal, and
   sample-rate changes.
5. Validate the AUv3 in representative hosts on iPad and iPhone. Confirm state
   restoration, resizing, automation, duplicate instances, and host crash
   isolation.
6. Resolve or remove the current PRE drive selector before release: ordinary
   macOS and iOS bundles deliberately omit unapproved drive-capture assets, so
   that stage must not be marketed as available until a complete rights-cleared
   bank exists.
7. Finish with loudness-matched Logic/device listening. Numerical equivalence
   and a successful build do not replace the tone gate.

## License and ownership assessment

The root MIT license permits commercial use, modification, sublicensing, and
sale provided its copyright and permission notice is included. The iPlug2
zlib-style terms also support closed-source commercial products. Eigen includes
MPL-2.0-covered files and separately listed BSD/Apache/Minpack portions; retain
the required notices and make source for any modified MPL-covered files
available as required. Keep a complete in-app acknowledgements/license screen
and a release-time source/notice archive.

The practical licensing boundary is:

| Layer | Recommended treatment | Important consequence |
| --- | --- | --- |
| Existing NAM/iPlug2/Core lineage | Retain its permissive notices | A paid closed-source binary is allowed; existing recipients keep their source rights |
| Trainer and `.dess` specification | Public MIT | Reduces capture friction and grows compatible model supply |
| New iOS product UI, commerce, catalogue, QA, and service code | Private/proprietary or dual-licensed by explicit file/repository policy | Do not publish it under the current root MIT license if exclusivity matters |
| Models, captures, IRs, icons, fonts, and artwork | File-level provenance and redistribution decision | Code licensing does not grant rights to these assets or third-party marks |
| Community packages | Contributor/creator agreement plus package metadata | Must define ownership, training consent, marks, redistribution, removal, and commercial-pack terms |

Publishing new iOS source inside this currently MIT-licensed repository makes
that published source available under MIT unless a clear, valid carve-out is
established before distribution. Changing a future license cannot revoke rights
already granted to recipients of earlier MIT versions. The durable commercial
advantage therefore should be product execution, verification, creator supply,
support, and trusted distribution—not an assumption that the inference equation
can remain secret.

Before a public build, resolve the existing notice about four PRE captures,
every bundled IR, the application icon, and the three UI artwork layers. Do not
ship the named Fireball test profile or use third-party amplifier branding until
the capture, redistribution, quality, and trademark decisions are recorded.

Sources: [MIT License](https://opensource.org/license/mit),
[MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/), and the repository's
`THIRD_PARTY_NOTICES.md`.

## App Store model

Apple requires In-App Purchase for digital features or content unlocked in the
app. Creator content also introduces moderation obligations. Subscriptions must
provide ongoing value. Accordingly:

- V1: paid universal download, all offline player/AUv3 functionality included,
  user imports via Files, no account, no analytics by default.
- V1.x: optional free, rights-cleared community index that links only to packages
  the app can safely import; no paid marketplace yet.
- V2: official or creator packs through IAP after rights, moderation, reporting,
  takedown, revenue-share, and review processes exist.
- Later: cloud training credits or a catalogue subscription through the
  appropriate Apple purchase path. Do not design a web-payment detour around
  review rules.
- Parallel: B2B/OEM licensing for pedal or interface manufacturers can be a
  separate revenue line from the consumer App Store product.

At qualifying scale, Apple's Small Business Program offers a 15% commission on
paid apps and IAP; the normal program fee and current eligibility rules must be
rechecked at enrolment. TestFlight supports up to 10,000 external testers, with
the first external build subject to review.

Sources: [App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/),
[Small Business Program](https://developer.apple.com/app-store/small-business-program/),
[Apple Developer Program](https://developer.apple.com/programs/whats-included/),
[TestFlight](https://developer.apple.com/testflight/), and
[App privacy details](https://developer.apple.com/app-store/app-privacy-details/).

## Market and pricing position

The iOS market already has inexpensive capable NAM/A2 loaders. At the time of
this review, the German App Store listed NAM Live at EUR 17.99 and NAM Loader
Pedal at EUR 5.99. TONEX and TH-U use free entry with substantial IAP catalogues.
A basic loader priced above those products would be hard to justify.

DessAmp can justify a premium only when users can hear and operate the
difference: one real amp represented across its continuous controls, the exact
Full model on every supported device, reliable AUv3/live behavior, and a clear
verified-package standard. That supports the EUR 19.99 founder price now and a
future EUR 29.99 tier after the gig-readiness gates—not an immediate EUR 39.99
price based on visual polish alone.

Current references: [NAM Live](https://apps.apple.com/de/app/nam-live-neural-amp-modeler/id6778012976),
[NAM Loader Pedal](https://apps.apple.com/de/app/nam-loader-pedal-auv3-plugin/id6775461616),
[TONEX](https://apps.apple.com/de/app/amplitube-tonex/id1613359930), and
[TH-U](https://apps.apple.com/de/app/overloud-thu/id1478394489).

## Go-to-market sequence

1. **Private hardware alpha:** 10–20 guitarists with known iPhone/iPad and audio
   interface combinations. Measure crashes, dropouts, thermals, setup failures,
   and model import—not just subjective tone.
2. **TestFlight founder beta:** 20–50 players, at least two AUv3 hosts, and a
   small set of rights-cleared, gate-passing amplifier packages. Publish the
   compatibility matrix and known limitations.
3. **Paid founder release at EUR 19.99:** universal standalone plus AUv3, offline
   Files import, no account, no subscription, and no unmoderated marketplace.
4. **Proof-led marketing:** publish matched Mac/iOS renders, measured latency,
   60-minute thermal/dropout results, interface compatibility, and short videos
   showing one conditioned profile moving through real knob positions.
5. **Raise price only with value:** MIDI/footswitch control, robust presets and
   scenes, validated host state restoration, more gate-passing creator supply,
   and a support process justify EUR 29.99.
6. **Ecosystem expansion:** open trainer documentation, verified-package badge,
   creator agreements, then IAP packs or recurring cloud services.

## Release decision gates

Do not submit until all are true:

- a quality-gate-passing, rights-cleared release package exists;
- current-source Mac/iPhone/iPad render equivalence is recorded;
- physical iPhone and iPad latency, dropout, route, interruption, and thermal
  tests pass;
- standalone and exact embedded AUv3 are validated in current builds and hosts;
- every interactive control is exercised at phone and tablet breakpoints;
- landscape launch, safe areas, VoiceOver labels, Dynamic Type exceptions,
  contrast, and 44-point touch targets are checked;
- license acknowledgements, privacy disclosure, support URL, EULA, model rights,
  and asset provenance are complete;
- final listening approval is recorded separately from build and numerical gates.
