# DessAmp conditioned capture and training runbook

This is the evergreen, context-independent entry point for a new amplifier.
Changing job IDs, hashes, prices, checkpoint epochs, and measured results belong
in `docs/experiments/`, not here.

Use the repository-local `$dessamp-conditioned-training` skill before operating
audio hardware, creating a cloud job, evaluating a model, or packaging a
`.dess` file. The Fortin/Nameless scripts are regression evidence only; a new
physical amplifier uses the generic commands below.

## 1. Choose the repository and experiment boundary

Run every command from the root of the independent DessAmp repository.

Keep the player, Core runtime, capture companion, trainer, evaluator, and
package code together until the `.dess` and conditioned-runtime APIs have
passed a physical-amplifier release. See [architecture.md](architecture.md).

Before capture, record:

- head-only boundary, fixed lead channel, cab/time effects disabled;
- learned controls in exact model-vector order;
- no more than eight learned controls for a directly loadable current DessAmp
  artifact, even though trainer experiments may declare up to twelve;
- normalized model range, native display range, default, step, unit, and taper;
- every frozen switch/knob, including load/cabinet and master-volume policy;
- standardized stimulus and acquisition/evaluation DI paths and SHA-256;
- 48 kHz routing, calibration, polarity, measured delay, and one safety trim;
- disjoint training, acquisition, frozen audit, and repeatability evidence;
- authorized cloud project, region, image digest, timeout, and total EUR cap.

For the first real-amplifier proof, learn one or two controls. Expand only after
the full capture → A2 Full → hidden audit → `.dess` → exact AU path passes.
Choose the owner's lowest and highest useful positions for each learned axis.
Keep model values normalized and put the usable physical endpoints in
`display_minimum` and `display_maximum`. Generate a finite candidate set
inside those endpoints and reserve separate audit positions. The
[physical guide](physical-amp-capture.md#1-define-what-you-will-capture)
shows a restricted Gain example and the actual five-axis Fireball ranges.

## 2. Create the local environment and Core tools

Use Python 3.11 or 3.12. Python 3.9 cannot install the pinned NAM 0.13 A2
package. Do not add `trainer_source` itself to `PYTHONPATH` for A2; it would
shadow the pinned upstream `nam` package with retained PANAMA sources.

```bash
export DESSAMP_PYTHON=/absolute/path/to/python3.12
export DESSAMP_LOCAL_TEST_ENV=/absolute/path/outside/the/repository/conditioned-a2-py312
UV_PROJECT_ENVIRONMENT="$DESSAMP_LOCAL_TEST_ENV" \
uv sync --project trainer_source/cloud/local_test_env --locked \
  --python "$DESSAMP_PYTHON"
export DESSAMP_LOCAL_PYTHON="$DESSAMP_LOCAL_TEST_ENV/bin/python"
uv pip check --python "$DESSAMP_LOCAL_PYTHON"
```

`local_test_env/uv.lock` is the reproducible macOS Python 3.12 test toolchain
for the official NAM 0.13 conditioned-A2 code, cloud helpers, capture companion,
and their tests. Keep its materialized environment outside the repository. It
deliberately excludes retained-trainer-only packages such as `auraloss` and
`torchaudio`, which previously leaked into the in-tree environment.

This macOS toolchain is not the production Linux/CUDA image lock. The
production image uses the reviewed
`trainer_source/cloud/requirements-conditioned-a2-image.constraints.txt` and
hash-locked `trainer_source/cloud/requirements-conditioned-a2.lock.txt`; never install that CUDA-specific lock
as a substitute for the macOS environment.
The retained PANAMA trainer under `trainer_source/nam` is a different package
with the same top-level import name. Reproduce it from its lock in a separate
external environment; never install both into one environment:

```bash
export DESSAMP_TRAINER_ENV=/tmp/dessamp-trainer-venv
UV_PROJECT_ENVIRONMENT="$DESSAMP_TRAINER_ENV" \
uv sync --project trainer_source --locked --extra test \
  --python "$DESSAMP_PYTHON"
```

The trainer lock uses the adjacent `capture_companion` through a relative
editable source and contains no machine-specific absolute path. Keep both
environments outside a future public history even though their manifests and
locks are source inputs.

```bash
cmake -S NeuralAmpModelerCore \
  -B /tmp/dessamp-core-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/dessamp-core-build \
  --target loadmodel loadmodel_manifest render_model run_tests \
  -j 8
(cd NeuralAmpModelerCore && /tmp/dessamp-core-build/tools/run_tests)
```

The Core test fixtures use paths relative to `NeuralAmpModelerCore`; keep the
subshell working directory exactly as shown.

## 3. Initialize a persistent capture project

Do not use an automatically expanded Cartesian UI grid for six or seven
controls: it can exceed the 4,096-point safety limit. Start from an explicit,
reviewable candidate set.

Copy a reviewed physical-lead template outside Git, replace every placeholder
with measured evidence, and choose an external durable data directory. For
Gain alone, use the [bounded one-control project](../trainer_source/examples/gain-3-to-7-project.json)
and [matching reservation spec](../trainer_source/examples/gain-3-to-7-evaluation.json)
as the template variables in sections 3 and 5. The two-control template below
adds Bright. Use the three-control template
when Lead Gain, Bright, and Bottom are all learned from the first run; it
contains a reviewed 24-point acquisition grid across all four switch modes.

```bash
export DESSAMP_DATA_DIR='/absolute/path/to/amp-capture/trainer-data'
export DESSAMP_PROJECT_CONFIG='/absolute/path/to/amp-capture/project.json'
export DESSAMP_STIMULUS='/absolute/path/to/verified/NAM-v3/input.wav'
export DESSAMP_PROJECT_TEMPLATE='docs/examples/physical-lead-two-control-project.json'

cp "$DESSAMP_PROJECT_TEMPLATE" \
  "$DESSAMP_PROJECT_CONFIG"
```

For the three-control switch proof, set
`DESSAMP_PROJECT_TEMPLATE=docs/examples/physical-lead-three-control-project.json`
before copying. Do not add its hidden 0.1/0.9 audit points to the project grid.

For a new physical capture, obtain the full NAM v3.0.0 reamp and verify its
SHA-256 before initialization. The initializer checks format and records the
hash of the supplied bytes. Use the owner guide for the reference hash, file
duration and download. [Training sweep](physical-amp-capture.md#2-get-the-training-sweep-and-check-the-signal-path)

The model values must stay inside `[0,1]`. Put amp-face values such as 0–10 or
0–100 in `display_minimum`/`display_maximum`. The initializer rejects
unmeasured calibration placeholders, a missing frozen-control record, a
mismatched vector order, and input other than 48 kHz mono PCM24. Enter the measured input reference in `metadata.calibration.input_level_dbu` and the measured signed alignment in `metadata.training_delay_samples`.

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" trainer_source/init_conditioned_project.py \
  --data-dir="$DESSAMP_DATA_DIR" \
  --project-config="$DESSAMP_PROJECT_CONFIG" \
  --input="$DESSAMP_STIMULUS"
```

Persist the returned project ID:

```bash
export DESSAMP_PROJECT_ID='paste-returned-project-id'
```

Open the Trainer against that same directory:

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m dessmetal_capture \
  --data-dir="$DESSAMP_DATA_DIR" \
  --loader=/tmp/dessamp-core-build/tools/loadmodel \
  --trainer-root=trainer_source
```

`dessmetal_capture` is the retained Python namespace; the product and new
documentation are DessAmp. The SQLite database, not the browser session, is the
durable project.

## 4. Prove capture integrity before training

For a tube amplifier, never connect a speaker output to an interface input. The
amplifier must see its cabinet or a correctly rated reactive load. Follow the
skill's complete capture protocol.

Before accepting a starter corpus:

1. prove loopback/routing, polarity, delay, and the loudest safe setting;
2. capture the centre twice as separate repeatability evidence outside the
   training set;
3. require repeatability ESR `<= 0.01`;
4. capture a low/high pair for every learned control while everything else is
   frozen;
5. require finite, non-identical output with a non-trivial delta;
6. use one safety trim and never peak-normalize individual reamps;
7. import only accepted training captures through pending Trainer
   recommendations.

The pair validator does not auto-align or optimize delay. Supply the same
recorded alignment trims to both WAVs:

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/validate_capture_pair.py \
  --mode=repeatability \
  --reference=/absolute/path/to/centre-repeat-a.wav \
  --candidate=/absolute/path/to/centre-repeat-b.wav \
  --maximum-esr=0.01 \
  --output=/absolute/path/to/evidence/repeatability.json

"$DESSAMP_LOCAL_PYTHON" trainer_source/validate_capture_pair.py \
  --mode=sensitivity \
  --reference=/absolute/path/to/gain-low.wav \
  --candidate=/absolute/path/to/gain-high.wav \
  --minimum-delta-esr=0.000001 \
  --output=/absolute/path/to/evidence/gain-axis.json
```

A duplicate low/high result is a hard stop: check host parameter semantics,
inactive modes, routing, or duplicated files before cloud work.

### Manual batch throughput without acceptance shortcuts

Once the physical path and repeatability gate are established, use the
two-stage operator pipeline in the skill's capture protocol. After each bounded
Logic recording, first run the fast gate (fresh routing/device state, stable
file, exact format and sufficient duration, finite/non-silent and unclipped
audio, exact standardized PCM prefix, and hashes). On success, immediately
show exactly one next queued physical setting so the owner can set the knobs
while official NAM validation runs.

The full input-version, length, latency, data-check, PyTorch-loader, and
repeatability validation remains mandatory before the recorded take is marked
complete. Back up SQLite, import the capture, persist evidence, and activate the
next recommendation transactionally. Knob setup may overlap that work; the
next Logic recording may not start until the previous take is accepted and
persisted. A deep-gate failure quarantines the take and stops queue advancement,
even if the provisional next setting was already displayed.

## 5. Freeze the hidden audit before training

Capture hidden settings on a separate, unseen DI. Never import these WAVs into
the training project or expose their target audio to disagreement scoring.

Copy and fill the reservation spec outside Git:

```bash
export DESSAMP_RESERVATION_SPEC='/absolute/path/to/evidence/evaluation-spec.json'
export DESSAMP_RESERVATION='/absolute/path/to/evidence/hidden-evaluation-reservation.json'
export DESSAMP_RESERVATION_TEMPLATE='docs/examples/evaluation-reservation-spec.json'

cp "$DESSAMP_RESERVATION_TEMPLATE" \
  "$DESSAMP_RESERVATION_SPEC"

PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" trainer_source/create_evaluation_reservation.py \
  --spec="$DESSAMP_RESERVATION_SPEC" \
  --output="$DESSAMP_RESERVATION"
```

The output hash-binds the evaluation DI, every reference WAV, every hidden
vector, and exact control order while declaring
`target_audio_available_to_acquisition=false`. Hidden vectors must also be
absent from the project's finite `candidate_points`; this prevents the local
Trainer from ever recommending them before ensemble scoring begins.

For Lead Gain, Bright, and Bottom, use
`docs/examples/evaluation-reservation-spec-lead-three-control.json`. Its eight
hidden references cover low/high gain in each switch mode and remain disjoint
from the reviewed 24-point acquisition grid.

## 6. Materialize the authoritative training corpus

Choose one accepted capture ID as internal validation. Use the Trainer API or
its project detail view to read stable capture IDs; do not select by a filename.

```bash
curl -s "http://127.0.0.1:8765/api/projects/$DESSAMP_PROJECT_ID" \
  > /absolute/path/to/evidence/project-snapshot.json
export DESSAMP_VALIDATION_CAPTURE_ID='paste-one-accepted-capture-id'
export DESSAMP_DATA_CONFIG='/absolute/path/to/evidence/data-config.json'
export DESSAMP_DATA_ROOT='/absolute/path/to/evidence'
```

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" trainer_source/materialize_conditioned_data_config.py \
  --data-dir="$DESSAMP_DATA_DIR" \
  --project-id="$DESSAMP_PROJECT_ID" \
  --validation-capture-id="$DESSAMP_VALIDATION_CAPTURE_ID" \
  --output="$DESSAMP_DATA_CONFIG"
```

The materializer rejects failed captures, warning captures unless explicitly
admitted for a workflow proof, missing/hashless paths, duplicate vectors,
ambiguous validation, non-normalized model ranges, and vector-order drift.

### Existing-corpus efficiency POC before a new capture

Use `materialize_conditioned_efficiency_poc.py` only for an explicitly scoped
`workflow_validation_only` experiment. Predeclare a multi-setting comparison
panel and move those captures out of training before either arm runs. Keep the
release-hidden reservation sealed; it is not a development score or stopping
signal. The materializer records the source-config hash, exact holdout
basenames/vectors, 40/7-style split, matched-seed arms, and adoption gates.

Compare a full-exposure arm with a contract-bound reduced arm under one
immutable input bundle. Give both arms the same fresh initialization seed and
distinct `--member-variant` values. For the reduced arm,
`--train-chunks-per-output` selects deterministic temporal strata across the
complete stimulus. It must never mean “take the first N seconds.” The trainer
records the selected-index hash, selected fraction, seconds per capture,
examples per epoch, and optimizer steps per epoch in the learning config,
model metadata, and training manifest.

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/cloud/conditioned_a2_runpod_job.py submit \
  ... \
  --seed=110 \
  --member-variant=stratified95 \
  --train-chunks-per-output=556 \
  --artifact-purpose=workflow_validation_only
```

Omit `--train-chunks-per-output` for the matched full-exposure baseline. A
matched pair is one controlled comparison, not two independent ensemble
members. Require a fresh second seed before changing the default capture or
training policy. Compare external per-setting current-Core ESR, measured cost,
and matched listening; an internal aggregate validation loss alone is not an
efficiency result. Never package a POC artifact as a release candidate.

## 7. Prepare one immutable A2 bundle

RunPod is the default GPU compute plane. Google Cloud Storage remains the
durable evidence plane because a RunPod Pod is replaceable and its local disk
is not the member record. The operator workstation therefore needs:

- the Google Cloud CLI authenticated to a dedicated non-production project;
- a GCS bucket for immutable inputs, contracts, epoch checkpoints, and finals;
- a digest-pinned trainer image in Artifact Registry plus RunPod registry auth;
- a least-privilege Google service-account JSON signer outside Git with mode
  `0600`, scoped to the training bucket and kept on the operator workstation;
  and
- a funded RunPod account whose API key is stored in the OS credential store.

Resolve the image to an immutable digest, verify the exact project and bucket,
and keep a running provider-cost ledger. The same explicit project is required
for preparation, submission, GCS access, and the bucket-scoped service-account
credential. A balance alert is not a kill switch. Every submission therefore
requires the owner-approved `--timeout-minutes`, currently verified
`--hourly-rate-usd`, and owner-approved `--max-cost-usd`; no default silently
expands cost authority.

### GPU economics policy

Choose the lowest measured USD per completed validation epoch that satisfies
the model's memory and durable-checkpoint requirements. Longer wall time is an
acceptable trade for lower total cost unless the owner sets a completion
deadline. For conditioned A2 Full, secure on-demand A40 is the current proven
default; benchmark RTX A4000 as the lower-price challenger when capacity is
available. Treat RTX 4090 and other options as candidates only after a
representative same-corpus benchmark. H100 is not the default: use it only
when measured wall-time savings reduce total USD, or when the owner explicitly
accepts the premium for a deadline.

Before a full ensemble, record the live rate and stock, measured or historical
seconds per epoch on a representative member, projected wall time, projected
USD per member, aggregate hourly rate, funded balance, and hard rate-by-time
ceiling. Include input download, validation, and checkpoint cadence in the
measurement; GPU utilization or VRAM allocation alone does not establish
economic value. Delete verified stale/terminal Pods after their GCS handoff is
stable because stopped or exited Pods may retain billable disk.

GPU type is part of the immutable member contract. A continuation must use the
same GPU. To move to a cheaper GPU, stop and preserve the prior run as evidence,
then create a new run ID and fresh independent seeds against the identical
frozen corpus; never reinterpret the old checkpoints as portable.

Review the strict conditioned-image source context without submitting a
billable build. Remove `--dry-run` only after the exact project, repository,
tag, and build are authorized, then retain the printed immutable digest URI:

```bash
trainer_source/cloud/build_conditioned_a2_image.sh \
  --project="$DESSAMP_GCP_PROJECT" \
  --confirm-project="$DESSAMP_GCP_PROJECT" \
  --region="$DESSAMP_REGION" \
  --repository="$DESSAMP_ARTIFACT_REPOSITORY" \
  --image-name="$DESSAMP_IMAGE_NAME" \
  --image-tag="$DESSAMP_IMAGE_TAG" \
  --dry-run
```

The helper supplies `conditioned-a2.gcloudignore` explicitly. That allowlist
must contain only the license and the exact Docker, direct requirements,
observed-image constraints, hash lock, contract, entrypoint, Cloud Build, and
conditioned-A2 implementation files. The lock targets CPython 3.11, Linux
x86-64, and CUDA 12.4; its header is the authoritative `uv pip compile`
regeneration command. Require a byte-identical repeat generation, review every
dependency/hash change, and retain the lock SHA-256 with the build evidence.
The Dockerfile must install it with `--require-hashes` and assert the
digest-pinned base's Torch/CUDA versions. Never upload a local environment,
cache, checkpoint, retained PANAMA package, test corpus, or unrelated trainer
source merely because it exists under `trainer_source/`.

```bash
export DESSAMP_CONTROL_ORDER='lead_gain,bright,bottom'
export DESSAMP_CAPTURE_COUNT='paste-train_count-from-the-materializer-output'
export DESSAMP_RUN_ID='owner-lead-a2-r01'

"$DESSAMP_LOCAL_PYTHON" trainer_source/cloud/conditioned_a2_runpod_job.py prepare \
  --project="$DESSAMP_GCP_PROJECT" \
  --confirm-project="$DESSAMP_GCP_PROJECT" \
  --bucket="$DESSAMP_GCS_BUCKET" \
  --run-id="$DESSAMP_RUN_ID" \
  --data-config="$DESSAMP_DATA_CONFIG" \
  --data-root="$DESSAMP_DATA_ROOT" \
  --expected-train-captures="$DESSAMP_CAPTURE_COUNT" \
  --control-order="$DESSAMP_CONTROL_ORDER"
```

`prepare` accepts only regular non-symlink WAV inputs below the explicit data
root, with per-file, count, and aggregate limits, then uploads every byte with
`--if-generation-match=0`. A config cannot expand that local authority. Never
reuse a run ID with changed inputs.

## 8. Submit independent A2 Full members

Use one on-demand RunPod Pod per independent seed. Parallel members are
encouraged, but each seed must have exactly one writer. The contract-aware
launcher rejects interruptible jobs because it binds the provider-side
`terminateAfter` deletion deadline through `podFindAndDeployOnDemand`. The Pod
receives the same digest-pinned GAR image and GCS member prefix regardless of
location. Never put the RunPod API key or Google service-account contents in a
ledger, log, or memory note.

An optional hosted RunPod OAuth MCP may provide read-only control-plane
visibility after a no-create smoke test. Configure its default tool approval
mode as `prompt`, approve each read separately, and never use generic MCP
`create-pod` in place of the contract-aware helper below. Pod-list responses
may contain unredacted launch arguments or environment values; disable that
tool for credential-bearing Pods and monitor through field-allowlisted direct
API output plus GCS instead. The immutable bundle, per-seed member contract,
one-writer proof, and `--resume-existing-member` recovery flow remain
authoritative.

The helper opens the project-validated signer only on the workstation and
mints V4 signed URLs for the exact input-manifest objects, immutable member
contract, that member's best/resume checkpoints, and fixed result filenames.
GET and PUT are separate and the URLs expire shortly after `terminateAfter`.
Every write signs `x-goog-content-length-range: 1,<object-limit>` into its V4
PUT, so GCS enforces the maximum even if a hostile worker ignores the local
guard. Capability schema 2 is required; rebuild and digest-pin the conditioned
image before the next run because schema-1 images reject the new bundle. Only
the encoded temporary capability bundle enters the Pod; the entrypoint consumes it from the environment before parsing
and clears its in-memory allowlist before quiescence. A Google private key,
access token, or application-default credential is never sent to RunPod.
The base64 transport uses a bounded zlib envelope: both compressed and
decompressed sizes are capped, trailing streams and expansion bombs are
rejected, and legacy raw-JSON decoding remains available for exact-contract
recovery. This keeps larger immutable capture sets below the provider's
GraphQL request-size boundary without broadening object scope.
When `--runpod-quiesce-on-success` identifies a RunPod execution, a missing
capability bundle aborts before training; it cannot fall back to ambient
Application Default Credentials. The no-capability path remains available
only to the separately authenticated Vertex compatibility execution.
Provider-side Pod-list history may expose the unexpired bearer URLs, so treat
unallowlisted responses as secret-bearing and persist only the redacted scope
summary. See Google Cloud Storage's official
[signed-URL overview](https://docs.cloud.google.com/storage/docs/access-control/signed-urls)
and [V4 GET/PUT examples](https://docs.cloud.google.com/storage/docs/access-control/signing-urls-with-helpers).

Every authenticated RunPod request rejects redirects before either the bearer
or the temporary capability body can be forwarded. Each submission attempt
uses a cryptographically random 128-bit nonce in the Pod name, returned
evidence, and a per-member GCS writer lease. That lease is created atomically
with generation zero before Pod creation. An unexpired lease blocks a second
writer; an expired, exact-contract lease can be reclaimed only with a
generation compare-and-swap after a bounded generation-stable read. Do not
release or overwrite a live lease merely because a monitor cannot currently
see a Pod.

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/cloud/conditioned_a2_runpod_job.py submit \
  --project="$DESSAMP_GCP_PROJECT" \
  --confirm-project="$DESSAMP_GCP_PROJECT" \
  --bucket="$DESSAMP_GCS_BUCKET" \
  --run-id="$DESSAMP_RUN_ID" \
  --image-uri="$DESSAMP_A2_IMAGE_DIGEST_URI" \
  --registry-auth-id="$DESSAMP_RUNPOD_REGISTRY_AUTH_ID" \
  --gcs-signing-credentials="$DESSAMP_RUNPOD_GCP_CREDENTIAL_FILE" \
  --runpod-keychain-service=com.runpod.api \
  --runpod-keychain-account=codex \
  --child=full \
  --seed="$DESSAMP_SEED" \
  --control-order="$DESSAMP_CONTROL_ORDER" \
  --expected-train-captures="$DESSAMP_CAPTURE_COUNT" \
  --gpu-type='NVIDIA A40' \
  --cloud-type=SECURE \
  --no-interruptible \
  --timeout-minutes="$DESSAMP_RUNPOD_TIMEOUT_MINUTES" \
  --hourly-rate-usd="$DESSAMP_RUNPOD_HOURLY_RATE_USD" \
  --max-cost-usd="$DESSAMP_RUNPOD_MAX_COST_USD" \
  --epochs=60 \
  --batch-size=16 \
  --ny=8192 \
  --validation-ny=262144 \
  --num-workers=4
```

The three cost inputs are independent and mandatory. The launcher checks the
funded balance, current spend, hourly spend limit, and the declared
rate-by-time ceiling (plus a small reserve) before creation. It passes the
hourly rate as GraphQL `deployCost`, sets the absolute `terminateAfter`
timestamp from the timeout, and validates returned `costPerHr`. A missing,
invalid, non-finite, or higher provider rate causes a Pod deletion request and
a failed preflight result. Timeout must be from 1 through 1,440 minutes; USD
inputs must be finite and positive. The repository hard limit is USD 50 per
member even if a larger value is supplied.

A successful mutation is still provisional. The launcher immediately reads
`/pods/{id}?includeMachine=true` and accepts launch evidence only if that fresh
object proves the exact ID, nonce-bound name, REST-native digest image,
registry identity, desired state, current rate, interruptibility, GPU display
name/count, machine GPU type, and secure/community tier. Conflicting image
aliases or any missing/mismatched field fail closed. Runtime mismatch after
exact identity proof deletes only that exact Pod and withholds success
evidence; an identity ambiguity is reported for manual review instead of
guessing a destructive target.

`terminateAfter` deletes the Pod; it is deliberately not `stopAfter`, which
only stops compute and can leave disk or volume billing. The deadline is a
provider-side last resort, not a substitute for the monitor's prompt deletion
after successful evidence validation.

The launch contract is checked against RunPod's current official
[GraphQL Pod guide](https://docs.runpod.io/sdks/graphql/manage-pods),
[GraphQL schema](https://graphql-spec.dev.runpod.io/),
[native Pulumi provider source](https://github.com/runpod/pulumi-runpod-native/blob/main/provider/pod.go),
and [runpodctl lifecycle guidance](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpodctl/SKILL.md).
Those sources confirm the on-demand mutation, creation-field names, returned
`costPerHr`, provider `terminateAfter`, and delete-versus-stop distinction.

When a Pod disappears, reaches its deadline, or becomes terminal, the recovery
monitor must prove that no Pod is still writing the seed, then reuse the exact
on-demand command with `--resume-existing-member`. It must not create a second
concurrent writer or infer completion from Pod state alone. Preserve the seed,
contract, image, input manifest, GPU/cloud type, optimizer, scheduler, loader
arguments, and absolute epoch target. Never recover through generic MCP Pod
creation.

Use independent seeds for disagreement members. Each seed owns one immutable
member contract and one contract-hash GCS checkpoint namespace. After every
completed validation epoch, A2 mirrors `durable/resume.ckpt` with model,
optimizer, scheduler, epoch, and global-step state; `durable/best.ckpt`
separately preserves the best internal-validation state. A timeout or
preemption therefore loses at most the unfinished work since the latest
durable mirror, not the completed seed history.

Do not add a mid-epoch SIGTERM checkpoint with the current shuffled,
non-stateful data loader: Lightning cannot guarantee an exact data sequence
when resuming such a checkpoint. The completed validation epoch is the
authoritative safe boundary, and the external monitor is responsible for
starting a replacement Pod.

Before a continuation, discover all RunPod Pods and prove that no writer for
that seed is pending or running. Then resubmit the same command with the
additional `--resume-existing-member` flag. The helper compares the immutable
member contract before submission; seed, data
manifest and hashes, image digest, control order, architecture,
optimizer/chunk/loader arguments, machine shape, and absolute epoch target must
all match. Do not rerun `prepare`, change the target epoch count, add legacy
checkpoint flags, resume across seeds, or run two jobs against one member
prefix.

Monitor the RunPod API and GCS together. A checkpoint proves recoverability,
not model acceptance. A member is complete only after its Pod has stopped or
been intentionally quiesced, no writer exists, the final NAM and manifest have
stable generations, their hashes agree, and current Core loads the model.
Neither a stopped Pod nor final-looking filenames alone prove completion.
The launcher holds a successful container quiescent after final publication so
RunPod cannot restart it and republish the same member repeatedly while the
Pod's desired state remains `RUNNING`. The temporary capability allowlist is
cleared first, then the entrypoint prints
`DESSAMP_TRAINING_COMPLETE_WAITING_FOR_POD_TERMINATION` and sleeps without
performing more training or uploads. The monitor must verify stable final GCS
generations, matching hashes, no competing writer, and current-Core loading,
then delete the Pod promptly. If the monitor fails, provider `terminateAfter`
deletes it at the authorized deadline. Quiescence is not completion evidence
and does not authorize leaving billable compute running.

```bash
export DESSAMP_MEMBER_DIR='/absolute/path/to/evidence/full-seed-output'
mkdir -p "$DESSAMP_MEMBER_DIR"

gcloud storage cp \
  "gs://$DESSAMP_GCS_BUCKET/jobs/$DESSAMP_RUN_ID/results/full-seed-$DESSAMP_SEED/conditioned-a2-full.nam" \
  "$DESSAMP_MEMBER_DIR/conditioned-a2-full.nam" \
  --project="$DESSAMP_GCP_PROJECT"

gcloud storage cp \
  "gs://$DESSAMP_GCS_BUCKET/jobs/$DESSAMP_RUN_ID/results/full-seed-$DESSAMP_SEED/training-manifest.json" \
  "$DESSAMP_MEMBER_DIR/training-manifest.json" \
  --project="$DESSAMP_GCP_PROJECT"

/tmp/dessamp-core-build/tools/loadmodel "$DESSAMP_MEMBER_DIR/conditioned-a2-full.nam"
shasum -a 256 "$DESSAMP_MEMBER_DIR/conditioned-a2-full.nam" \
  "$DESSAMP_MEMBER_DIR/training-manifest.json"
```

Keep the model at the exact relative path declared by the downloaded member
manifest; do not rename it during download. Before opening the
frozen external audit, select one of at least two independent members using
only their disjoint internal-validation manifests:

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/select_conditioned_member_by_internal_validation.py \
  --member=82=/absolute/path/to/full-seed-82 \
  --member=83=/absolute/path/to/full-seed-83 \
  --output=/absolute/path/to/evidence/internal-member-selection.json
```

For a terminal teacher-distillation audit, render the frozen unseen-DI teacher
targets outside this helper and only after the internal-only selection report
above exists. Then materialize a hash-bound external spec and pass that spec to
the normal reservation creator:

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" \
  trainer_source/materialize_teacher_distillation_terminal_audit_spec.py \
  --source-spec=/absolute/path/to/internal-validation-reservation-spec.json \
  --selection-report=/absolute/path/to/evidence/internal-member-selection.json \
  --training-input=/absolute/path/to/frozen-training-input.wav \
  --evaluation-input=/absolute/path/to/frozen-unseen-di.wav \
  --renderer=/tmp/dessamp-core-build/tools/render_model \
  --target-dir=/absolute/path/to/terminal-teacher-targets \
  --target-basename=audit-setting-01.wav \
  --target-basename=audit-setting-02.wav \
  --evaluation-scope-json='{"role":"external-known-settings-on-unseen-di","release_gate":true}' \
  --provenance-json='{"purpose":"terminal-teacher-distillation-audit"}' \
  --output=/absolute/path/to/evidence/terminal-evaluation-spec.json

PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" trainer_source/create_evaluation_reservation.py \
  --spec=/absolute/path/to/evidence/terminal-evaluation-spec.json \
  --output=/absolute/path/to/evidence/terminal-evaluation-reservation.json
```

Supply one ordered `--target-basename` per source reference, or replace the
directory/basename arguments with one
`--target='SETTING_ID=/absolute/path/to/target.wav'` mapping per setting. The
materializer never invokes the renderer, opens a candidate model, or changes
the selected member. It requires 48 kHz mono PCM24 audio, proves the external
DI and targets are byte-disjoint from the training/internal inputs, preserves
the exact source control/vector/setting/teacher order, hash-binds the selection
report and renderer, keeps audit targets unavailable to acquisition, and
refuses to overwrite an existing spec.

New production runs use only the contract-aware RunPod helper with the GCS
evidence plane. The former generic Vertex `prepare` and `submit` commands fail
closed before any cloud call, and the mutable image-build path was removed;
do not recreate their submission procedure in a new run ledger.

## 9. Evaluate the frozen hidden settings

The current A2 Full/Lite receptive field is 6,347 samples. New exports record it
in model metadata; pass it explicitly for older A2 exports. Never use the legacy
WaveNet value 4,093 for an A2 audit.

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/evaluate_conditioned_exports.py \
  --evaluation-reservation="$DESSAMP_RESERVATION" \
  --renderer=/tmp/dessamp-core-build/tools/render_model \
  --member=82=/absolute/path/to/seed-82/conditioned-a2-full.nam \
  --receptive-field=6347 \
  --output=/absolute/path/to/evidence/held-out-seed-82.json \
  --render-dir=/absolute/path/to/evidence/held-out-seed-82-renders
```

One deployable member can be release-gated alone. Multiple independent members
can additionally be evaluated as an arithmetic ensemble, but the player still
ships one selected child.

Acceptance:

- every hidden setting `< 0.035` (normal NAM gate);
- project target: mean `<= 0.01`, p95/worst `<= 0.02`;
- preferred: mean `<= 0.005` without a worse corner;
- repeatability ESR `<= 0.01`;
- current-source runtime and human listening still required.

Internal validation cannot release a model.

## 10. Select and queue the next active captures

Materialize candidates directly from current SQLite state and the frozen
reservation. The resulting file contains reserved vectors but no target paths:

```bash
export DESSAMP_CANDIDATE_CONFIG='/absolute/path/to/evidence/candidate-config-r02.json'

PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" trainer_source/materialize_conditioned_candidate_config.py \
  --data-dir="$DESSAMP_DATA_DIR" \
  --project-id="$DESSAMP_PROJECT_ID" \
  --evaluation-reservation="$DESSAMP_RESERVATION" \
  --output="$DESSAMP_CANDIDATE_CONFIG"
```

Score only with at least two independently seeded members and a separate
acquisition DI:

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/score_conditioned_exports.py \
  --candidate-config="$DESSAMP_CANDIDATE_CONFIG" \
  --renderer=/tmp/dessamp-core-build/tools/render_model \
  --member=82=/absolute/path/to/seed-82/conditioned-a2-full.nam \
  --member=83=/absolute/path/to/seed-83/conditioned-a2-full.nam \
  --input=/absolute/path/to/acquisition-di.wav \
  --receptive-field=6347 \
  --output=/absolute/path/to/evidence/disagreement-r02.json
```

Queue a spatially separated batch transactionally. The importer verifies the
candidate/reservation hashes and current capture set, creates an integrity-
checked SQLite backup, and refuses stale or leaked evidence:

```bash
PYTHONPATH=capture_companion/src:trainer_source \
"$DESSAMP_LOCAL_PYTHON" trainer_source/import_conditioned_suggestions.py \
  --data-dir="$DESSAMP_DATA_DIR" \
  --project-id="$DESSAMP_PROJECT_ID" \
  --selection-report=/absolute/path/to/evidence/disagreement-r02.json \
  --round-index=2 \
  --maximum=8 \
  --minimum-distance=0.35
```

Return to the ordinary Trainer queue and capture those settings. Do not use
`trainer_source/run_fortin_conditioned_active_learning.py` for a physical amp; it is an owned
Nameless regression harness with fixed host automation.

## 11. Package one selected player artifact

Never package the arithmetic ensemble. A failed numerical gate may only create
an explicitly labelled listening artifact using `--allow-failed-gate`.

```bash
PYTHONPATH=capture_companion/src:trainer_source \
"$DESSAMP_LOCAL_PYTHON" trainer_source/package_conditioned_export.py \
  --data-dir="$DESSAMP_DATA_DIR" \
  --project-id="$DESSAMP_PROJECT_ID" \
  --model=/absolute/path/to/selected/conditioned-a2-full.nam \
  --evaluation-report=/absolute/path/to/evidence/held-out-selected.json \
  --member=82 \
  --name='Owner Lead A2 Full' \
  --package-filename='Owner-Lead-A2-Full.dess' \
  --cloud-run-id="$DESSAMP_RUN_ID" \
  --physical-amplifier \
  --loader=/tmp/dessamp-core-build/tools/loadmodel \
  --manifest-loader=/tmp/dessamp-core-build/tools/loadmodel_manifest \
  --renderer=/tmp/dessamp-core-build/tools/render_model \
  --output=/absolute/path/to/evidence/package-report.json
```

The report's `package.path` is the friendly `.dess` file. Open it with **CUSTOM** in the player.
Check the imported name, control defaults and sweep, then save and reopen a host session.
See [the walkthrough](physical-amp-capture.md#6-open-the-package-in-the-player).

If a persistent capture label is deliberately acquisition-oriented, add a
repeatable presentation-only override such as
`--control-label='gain_index=Gain'`. This changes neither the manifest control
ID nor its model-vector index, range, default, or hash-bound training order.

A single conditioned model whose declared control order already contains
`morph` keeps that control inside the primary `controls` vector. Do not add the
top-level `morph` object: that object is reserved for the historical
`dual_model_linear_v1` runtime and appends a separate host slot. For the DESS
REDLINE shell, current player source binds the existing continuous `[0, 1]`
`morph` condition to the code-owned `US` -> `BRIT` presentation without
changing parameter order or count.

Use `--amp-simulator` instead of `--physical-amplifier` for Nameless or another
plug-in source. The command rejects provenance conflicting with project
metadata and backs up SQLite before model import.

For the final private owner-listening catalogue, do not hand-copy the four
packages into a build directory. After the physical Fire package report has
status `accepted`, materialize a new external catalogue with the fail-closed
assembler:

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" tools/assemble_private_listening_catalogue.py \
  --fire=/absolute/path/to/DESS-FIRE.dess \
  --fire-sha256="$DESSAMP_FIRE_SHA256" \
  --forge=/absolute/path/to/DESS-FORGE-private-workflow-validation.dess \
  --forge-sha256="$DESSAMP_FORGE_SHA256" \
  --redline-us=/absolute/path/to/01-Dess-Redline-US.dess \
  --redline-us-sha256="$DESSAMP_REDLINE_US_SHA256" \
  --redline-brit=/absolute/path/to/02-Dess-Redline-BRIT.dess \
  --redline-brit-sha256="$DESSAMP_REDLINE_BRIT_SHA256" \
  --fire-package-report=/absolute/path/to/fire-package-report.json \
  --fire-package-report-sha256="$DESSAMP_FIRE_PACKAGE_REPORT_SHA256" \
  --core-loader=/absolute/fresh/current-core/loadmodel_manifest \
  --output-dir=/absolute/new/path/FIRE-FORGE-REDLINE-private-listening
```

The output path must not already exist and must remain outside the repository.
The helper validates the accepted Fire report, all package bytes, manifest
identity/control/fixed-control contracts, three-product/four-slot ordering,
Redline variant grouping, and current-Core smoke loads before an atomic rename.
Its generated catalogue lock deliberately keeps owner listening, signing,
installation, and redistribution gates false.

## 12. Build and prove the exact current plugin

Build from current source without installing or signing:

```bash
export DESSAMP_BUILD_ROOT="$(mktemp -d /tmp/dessamp-release.XXXXXX)"

for target_name in APP AU VST3; do
  xcodebuild -project NeuralAmpModeler/projects/DessAmp-macOS.xcodeproj \
    -target "$target_name" -configuration Release \
    ARCHS=arm64 ONLY_ACTIVE_ARCH=YES CODE_SIGNING_ALLOWED=NO \
    CONFIGURATION_BUILD_DIR="$DESSAMP_BUILD_ROOT/products/$target_name" \
    SYMROOT="$DESSAMP_BUILD_ROOT/sym/$target_name" \
    OBJROOT="$DESSAMP_BUILD_ROOT/obj/$target_name" \
    DSTROOT="$DESSAMP_BUILD_ROOT/stage/$target_name" build
done
```

Keep all four Xcode roots outside the repository. `DSTROOT` alone stages the
bundle externally but the project otherwise writes products and intermediates
to `NeuralAmpModeler/build-mac`. A release or cleanup proof must not depend on
those in-tree leftovers.

The Apple resource stagers anchor repository-owned artwork, fonts, models,
IRs, and notices at the repository/project root and open every intervening
directory plus the source leaf with no-follow semantics. Explicitly gated
external private roots remain their own traversal anchors. Do not replace this
descriptor-based copier with `isfile()` followed by `shutil.copy*()`: a
symlink above a narrow `resources/fonts` root would reintroduce the build-time
source escape.

Locate the exact staged component and pass its executable explicitly. Omitting
`--component-binary` can accidentally exercise an older installed AU:

```bash
export DESSAMP_PACKAGE='/absolute/path/to/Owner-Lead-A2-Full.dess'
export DESSAMP_AU_BINARY="$(find "$DESSAMP_BUILD_ROOT/stage/AU" -path '*/DessAmp.component/Contents/MacOS/DessAmp' -type f -print -quit)"
test -n "$DESSAMP_AU_BINARY"

/tmp/dessamp-core-build/tools/loadmodel_manifest "$DESSAMP_PACKAGE"
bash NeuralAmpModeler/tests/run_au_player_test.sh \
  "$DESSAMP_PACKAGE" \
  --component-binary "$DESSAMP_AU_BINARY"
```

Finish with exact APP/AU/VST3 editor interaction, parameter/state recall, model
relinking, and level-matched Logic listening across centre, corners, sweeps,
transients, and sustained playing. Automated metrics do not decide tone.

## 13. Repository test gate

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m unittest discover -s capture_companion/tests -v

"$DESSAMP_LOCAL_PYTHON" -m unittest discover \
  -s trainer_source/cloud/tests -p 'test_*.py' -v

# Run the official-NAM conditioned/import/materializer tests in the root
# official-NAM environment.
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m pytest \
  trainer_source/tests/test_conditioned_a2.py \
  trainer_source/tests/test_import_teacher_distillation_plan.py \
  trainer_source/tests/test_materialize_conditioned_data_config.py -q

"$DESSAMP_LOCAL_PYTHON" -m pytest \
  trainer_source/fixed_a2/test_train_fixed_nam_a2.py -q

# Run the retained PANAMA trainer suite only in its separate locked environment.
"$DESSAMP_TRAINER_ENV/bin/python" -m pytest trainer_source/tests -q \
  --ignore=trainer_source/tests/test_conditioned_a2.py \
  --ignore=trainer_source/tests/test_import_teacher_distillation_plan.py \
  --ignore=trainer_source/tests/test_materialize_conditioned_data_config.py

# Native model/parser fixtures must run from NeuralAmpModelerCore because the
# test binary resolves its checked-in example models relative to that cwd.
(cd NeuralAmpModelerCore && /tmp/dessamp-core-build/tools/run_tests)

PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m unittest discover \
  -s NeuralAmpModeler/tests -p 'test_*.py' -v

bash NeuralAmpModeler/tests/run_unserialization_layout_tests.sh
bash NeuralAmpModeler/tests/run_desspreset_tests.sh
bash NeuralAmpModeler/tests/run_auto_input_match_tests.sh
bash NeuralAmpModeler/tests/run_player_shell_identity_tests.sh
bash NeuralAmpModeler/tests/run_player_morph_presentation_tests.sh
bash NeuralAmpModeler/tests/run_au_state_migration_test.sh \
  --component-binary "$DESSAMP_AU_BINARY"
```

Do not merge the two `trainer_source/tests` pytest commands into one Python
environment or process. The repository's retained `trainer_source/nam`
package and the official NAM 0.13 dependency both own the top-level `nam`
module name. Combining them can shadow the conditioned dependency, silently
select the wrong implementation, or produce collection-order failures.
Separate locked environments and invocations are the verified gate; this
isolation does not change the production dependency.

The state-layout, preset-envelope, and AU migration gates complement the exact
conditioned-player render in section 12: they protect the original 29 host
parameter IDs and the four appended pedal levels while the learned control schema
changes between amplifiers.

See [cloud/README.md](../trainer_source/cloud/README.md) for transport, quota,
region, timeout, cost, checkpoint, and result-prefix details. Changing run IDs
and measured status belong in internal `docs/experiments/` evidence, which is
deliberately excluded from the staged public tree.

## Stop gates

Do not claim completion at submission, checkpoint creation, internal validation,
build success, or package creation. The sequence is:

1. capture integrity and repeatability;
2. immutable cloud bundle and terminal job state;
3. downloaded hashes and current-Core load;
4. external hidden audit;
5. selected single member and `.dess` corruption checks;
6. exact staged AU render/state test;
7. exact APP/AU/VST3 UI interaction;
8. level-matched Logic listening;
9. only then any model replacement, signing, installation, or publication.
