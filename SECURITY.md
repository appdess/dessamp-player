# Security policy

## Supported versions

DessAmp has not published a supported public release yet. Security fixes are
developed against the current source and will be assigned supported-version
windows when the first release is published.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or include secrets,
private model packages, capture audio, cloud credentials, signing material, or
personal paths in a report. Until a public repository enables private
vulnerability reporting, contact the repository owner through an established
private channel. After publication, prefer the repository host's private
security-reporting feature.

Include the affected component and version, reproduction prerequisites,
security impact, and the smallest non-sensitive proof needed to confirm the
issue. The maintainer will acknowledge a reproducible report, coordinate a
fix and disclosure window, and credit the reporter when requested and safe.

## Security boundaries

- Model/package parsing, hash validation, preset/state migration, and model
  replacement must stay off the real-time audio thread.
- `.dess` packages and `.nam` files are untrusted input until current Core
  validates their structure and declared hashes.
- RunPod and Google Cloud credentials belong only in the OS credential store
  or owner-controlled files outside Git. A reusable Google credential must
  remain on the operator workstation; non-Google workers receive only exact,
  expiring, method-specific GCS capabilities. Never place credentials or
  unexpired capabilities in fixtures, logs, ledgers, screenshots, issue
  reports, or Codex memory.
- Capture databases, hidden audit audio, checkpoints, commercial-amplifier
  models, signing identities, and notarization credentials are private
  evidence and are outside the public source boundary.
- A successful build, numerical audio score, or cloud job state is not by
  itself a security or release approval.
