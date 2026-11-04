# DessAmp App Store preflight

Checked 4 November 2026 for the iPhone and iPad standalone app and its
embedded Audio Unit v3. The local Release archive succeeded. An App Store
Connect build, Apple’s server-side validation and App Review are still pending.

## What passed locally

- The shared `iOS-APP with AUv3` scheme produced an unsigned device archive
  from the clean source tree with Xcode 26.6 and the iOS 26.5 SDK. Its
  application path is `Applications/DessAmp.app`. Its app, AUv3 and framework
  use matching version and build strings of `0.2.0`. Apple requires Xcode 26
  or later with the iOS 26 SDK or later for new uploads. [SDK requirements](https://developer.apple.com/news/upcoming-requirements/)
- The host app is `com.dessdynamics.dessamp`. It targets iPhone and iPad and
  contains `com.dessdynamics.dessamp.auv3` with the Audio Unit UI extension
  point. The archive contains app, extension and framework dSYMs, an icon
  catalogue, an explicit microphone purpose string and the audio background
  mode. The 1024 px store icon source has no alpha channel.
- The normal archive contains the four byte-verified Version C cabinets, zero
  bundled `.dess` amplifiers and zero pedal `.nam` models. The full-bank
  Release archive built separately with development signing includes these sounds:

  | Amplifiers | Pedals | Cabinets |
  | --- | --- | --- |
  | FIRE, FORGE, REDLINE US, REDLINE BRIT | Oxide 08, Amber 01, Verdant 09, Nocturne | DZL, SHFLD Mix, Gleichrichter, Green |

  The four amp packages, four pedal models, four cabinet WAVs and factory
  settings all matched the tested iPhone build by SHA-256. The archive contains
  the standalone app and AUv3, and the bundle passed strict code-signature
  verification. The development profiles are tied to registered devices.
  To reproduce this local archive, use the Release configuration of
  `iOS-APP with AUv3` with destination `generic/platform=iOS`. Pass the
  same external amp and drive directories used by the tested build via
  `DESSAMP_PRIVATE_AMP_DIR` and `DESSAMP_PRIVATE_DRIVE_DIR`, with
  `DESSAMP_PRIVATE_AMP_VALIDATION=YES` and `DESSAMP_PRIVATE_DRIVE_VALIDATION=YES`.
  The private checkout supplies the pinned cabinet bank and factory settings.
  Compare the 13 files in the archived app against the tested build before
  selecting an archive for distribution.
- The audio framework includes its file-timestamp privacy reasons, declares
  no tracking or collected data, and imports `stat`. The app and extension
  do not directly import that API. The privacy manifest and third-party
  notices are present in the archive. Review the privacy report again when
  final dependencies and features are frozen.
- The owner heard live guitar and YouTube Music together through an iRig HD 2
  on iPhone 16 Pro. The AUv3 rendered in an independent simulator host.
  Physical iPad audio, route changes, interruptions and sustained performance
  still need checking on the build delivered through TestFlight.

## Before the first TestFlight upload

1. Use the full bank above as the target for first launch. FIRE opens with
   SHFLD Mix, Lead Gain 6.1, Presence 3.9 and drive off. Keep a release record
   for the exact assets going to Apple, including the origin and distribution
   rights for the cabinet IRs, amp and pedal models, font, icon and artwork.
   For an IR made using a song, record whether the song served as a tonal
   reference or supplied audio retained in the IR, and the applicable rights.
   Retain the code and asset notices. [App Review Guidelines, sections 2.2 and 5.2](https://developer.apple.com/app-store/review/guidelines/)
2. Confirm active Apple Developer Program membership, the App Store Connect
   app record for the bundle ID, both registered identifiers and signing roles.
   The local keychain had a development identity and no Apple Distribution
   identity at preflight. Use Xcode’s managed distribution signing or an
   appropriate distribution identity and profiles. Assign a build number
   greater than the last upload for this release, archive the chosen payload,
   validate with Organizer, and upload to TestFlight. Account records and
   prior uploads were not verified. [TestFlight](https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/)
3. Publish working support and privacy-policy URLs and add an easily accessible
   policy link inside DessAmp. The local technical site and this build did not
   expose those release links at preflight. Explain live audio, local files and
   presets, deletions, and any actual sharing. On-device audio processing alone
   does not count as collection for Apple’s label. Verify every SDK and support
   flow before answering the App Privacy questionnaire. [App privacy details](https://developer.apple.com/app-store/app-privacy-details/) · [Guideline 5.1.1](https://developer.apple.com/app-store/review/guidelines/)
4. Prepare the Music listing, supported iPhone/iPad screenshots, description,
   age rating, review contact, content-rights and encryption answers. In Review
   Notes explain how to load a cabinet and amp, grant audio input, test TUNE,
   and open the AUv3 in an iOS host. Supply a usable test tone/DI, sample
   package or demonstration video when the reviewer lacks a guitar interface.
   Complete pricing and territories. For a paid app, complete the Paid Apps
   Agreement, tax and banking details. For EU distribution, declare trader
   status and verify the applicable public contact information.
   [Submit an app](https://developer.apple.com/help/app-store-connect/manage-submissions-to-app-review/submit-an-app/) · [Agreements](https://developer.apple.com/help/app-store-connect/manage-agreements/sign-and-update-agreements/) · [EU trader requirements](https://developer.apple.com/help/app-store-connect/manage-compliance-information/manage-european-union-digital-services-act-trader-requirements/)
5. Test the exact distribution build through TestFlight, including standalone
   playing, permission denial/recovery, preset recall, tuner, cabinet changes,
   AUv3 discovery and session restore in supported hosts. Submit that tested
   build for review with the same advertised capabilities.

## Warranty

Apple applies its Standard EULA when the developer supplies no custom EULA.
Section (e) provides an “as is” warranty disclaimer to the extent permitted by
applicable law. Applicable mandatory consumer rights still apply. The default
EULA is an appropriate starting point for DessAmp. A special disclaimer screen
is unnecessary solely to restate that clause. A custom EULA or a stronger
Germany/EU warranty claim warrants qualified legal review. Keep separate
open-source and asset notices with the app. [Standard EULA](https://www.apple.com/legal/internet-services/itunes/dev/stdeula/) · [Apple’s custom EULA guidance](https://developer.apple.com/help/app-store-connect/manage-app-information/provide-a-custom-license-agreement/)
