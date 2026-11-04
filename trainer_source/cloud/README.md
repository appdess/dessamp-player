# DessAmp conditioned training on RunPod with GCS durability

RunPod is DessAmp's default GPU compute plane. Google Cloud Storage is the
durable evidence plane: immutable inputs, member contracts, validation-epoch
checkpoints, manifests, and final models survive Pod eviction and replacement.
Local Pod disks are disposable and never define completion.

The standard production shape is one **on-demand** Pod per independent seed.
Run seeds in parallel when capacity permits, but preserve exactly one writer
for each seed and GCS result prefix. The contract-aware launcher rejects
interruptible submission: its cost guard depends on binding RunPod's provider
`terminateAfter` deletion deadline during `podFindAndDeployOnDemand` creation.

## Safety and identity

- Use a dedicated non-production GCP project and bucket.
- Confirm the exact project twice before preparation or submission.
- Pin the trainer image by Artifact Registry `sha256` digest.
- Use a least-privilege Google service account scoped to the training bucket
  only as a local V4 signed-URL signer.
- Keep its JSON file outside Git with mode `0600`; never send it to RunPod.
- Store the RunPod API key in the OS credential store; never write it to a
  ledger, command transcript, or memory note.
- Keep one immutable member contract and one GCS namespace per seed.
- A continuation preserves the input manifest, seed, architecture, optimizer,
  scheduler, loader, image, hardware contract, and absolute epoch target.
- Never prepare over an existing run ID or resume one seed into another.
- Record the funded RunPod balance, current rate, owner-approved ceiling,
  timeout, Pod ID, cloud type, on-demand status, image digest, and contract
  hash in the run ledger without recording credentials.
- Require all three independent cost inputs on every submission:
  `--timeout-minutes`, `--hourly-rate-usd`, and `--max-cost-usd`. The helper
  accepts a timeout only from 1 through 1,440 minutes, requires finite positive
  USD values, rejects a worst-case rate-by-time ceiling above the authorized
  maximum, and enforces a repository hard maximum of USD 50 per member.

## GPU economics policy

Optimize for measured USD per completed validation epoch, not peak GPU class,
advertised throughput, or unused VRAM. Longer wall time is acceptable unless
the owner sets a deadline. Secure on-demand A40 is the current proven default
for conditioned A2 Full; RTX A4000 is the lower-price challenger when stock is
available. Benchmark other economical options against the same corpus and
training contract. H100 requires either evidence that its wall-time reduction
lowers total cost or explicit owner approval to pay a deadline premium.

Record live stock/rate, measured or historical seconds per epoch, projected
wall time and total cost, aggregate hourly rate, funded balance, and the hard
per-member/ensemble ceilings before submission. Include data transfer,
validation, and checkpoint cadence in the comparison. A low VRAM percentage
or a high utilization sample is diagnostic evidence, not a cost decision by
itself.

Hardware is hash-bound to each member contract. Never resume a checkpoint on
a different GPU type. An economic hardware change requires preserving the old
run as evidence and creating a new run ID with fresh independent seeds against
the identical frozen corpus. After GCS handoff is stable, delete exact verified
terminal Pods rather than leaving stopped/exited Pod disks billable.

## Required system access

The operator workstation needs:

1. Python environment and repository dependencies installed;
2. `gcloud` authenticated to the dedicated GCP project;
3. a GCS bucket for inputs, checkpoints, and results;
4. a digest-pinned trainer image in Google Artifact Registry;
5. RunPod registry authentication that can pull that private image;
6. a funded RunPod account and API key in the macOS Keychain or equivalent OS
   credential store; and
7. a bucket-scoped Google service-account JSON signer readable only by the
   local operator.

GCS remains necessary because RunPod does not provide DessAmp's immutable,
cross-Pod member/checkpoint contract natively. The local workstation uses
`gcloud` for preparation and evidence verification. At submission it keeps
the reusable Google credential local and mints V4 signed URLs for only the
exact hash-bound inputs, the member contract, that member's two durable
checkpoints, and its fixed output filenames. Each URL is method-specific and
expires shortly after the provider deadline. Every V4 PUT also signs
`x-goog-content-length-range: 1,<object-limit>`; Cloud Storage rejects a body
outside that inclusive range even if a worker bypasses the local size check.
The entrypoint consumes and removes the encoded capability bundle from its process
environment before parsing it; no Google private key, access token, or
application-default credential exists in the Pod.
The base64 transport uses a bounded zlib envelope so larger immutable corpora
remain below RunPod's GraphQL request-size limit. The worker caps both the
compressed transport and decompressed JSON before parsing, rejects trailing
compressed data and expansion bombs, and still accepts the earlier raw-JSON
transport for exact-contract recovery compatibility.
RunPod mode is identified by `--runpod-quiesce-on-success` and requires that
validated capability client before any training starts. A missing or empty
capability environment fails closed instead of silently creating a storage
client from ambient Application Default Credentials. The no-capability path
is retained only for the separately authenticated Vertex compatibility mode.

The server-bounded write contract is capability schema 2. A trainer image that
only understands schema 1 fails closed, so rebuild and digest-pin the reviewed
conditioned image before the next submission; never reuse an older image with a
schema-2 launch bundle.

Signed URLs are still temporary bearer capabilities. The launcher redacts the
bundle from returned evidence and records only expiry, counts, and a scope
fingerprint. Pod-list responses can echo environment values, so the
control-plane redaction restrictions below remain mandatory until the URLs
expire. This design follows Google Cloud Storage's documented
[signed-URL boundary](https://docs.cloud.google.com/storage/docs/access-control/signed-urls)
and [V4 GET/PUT helper contract](https://docs.cloud.google.com/storage/docs/access-control/signing-urls-with-helpers).

## Optional RunPod MCP visibility

Codex may connect to RunPod's hosted OAuth MCP as an optional control-plane
view:

```bash
codex mcp add runpod --url https://mcp.getrunpod.io/
codex mcp login runpod
codex mcp list
```

Keep `default_tools_approval_mode = "prompt"` for this server. Prove the
connection with separately approved read-only calls only, and create no Pod
during setup. The MCP is not a second launcher: every DessAmp preparation,
submission, or continuation still goes through
`conditioned_a2_runpod_job.py`, including `--resume-existing-member`.

Treat Pod-list responses as secret-bearing unless the exact server response is
known to redact launch arguments and environment values. The hosted RunPod
`list-pods` response has been observed to include those fields. Disable that
tool while a Pod carries unexpired GCS capabilities, and use a
field-allowlisted direct API query plus GCS evidence instead. GPU-catalog reads
remain suitable for MCP visibility. Never paste or persist a capability
returned by any control-plane tool.

## Files

- `conditioned_a2_runpod_job.py`: prepares the immutable GCS bundle and submits
  or resumes one exact-contract RunPod member.
- `conditioned_a2_entrypoint.py`: validates the remote contract, restores the
  durable checkpoint, trains, and publishes checkpoints/finals.
- `conditioned_a2_contract.py`: canonical member contract and hashing rules.
- `gcs_capabilities.py`: validates and executes the exact expiring signed-URL
  transfer allowlist and mandatory signed write-range headers used by
  non-Google workers.
- `build_conditioned_a2_image.sh`: builds the trainer image.
- `local_test_env/pyproject.toml` and `local_test_env/uv.lock`: reproducible
  macOS Python 3.12 toolchain for official NAM conditioned-A2, capture,
  cloud-helper, and player/resource tests. Materialize it outside the
  repository; it is deliberately not part of the Linux/CUDA image context.
- `conditioned_a2_vertex_job.py` and `conditioned_vertex_job.py`: retained
  compatibility modules that still supply shared contract/preparation helpers
  and historical-provider forensics. Generic `prepare` and `submit` mutations
  fail closed before reading arguments or contacting Google Cloud; they are not
  a production launcher.

The obsolete generic Vertex mutation surface is disabled. The mutable `:latest`
Docker context, Cloud Build file, and range-only image requirements were
removed from the source tree. They bypassed the RunPod/GCS identity, timeout,
and cost contract and must not be restored as a production shortcut.

## One-time setup

Create the locked local test environment outside the repository and
authenticate Google Cloud:

```bash
export DESSAMP_PYTHON=/absolute/path/to/python3.12
export DESSAMP_LOCAL_TEST_ENV=/absolute/path/outside/the/repository/conditioned-a2-py312
UV_PROJECT_ENVIRONMENT="$DESSAMP_LOCAL_TEST_ENV" \
uv sync --project trainer_source/cloud/local_test_env --locked \
  --python "$DESSAMP_PYTHON"
export DESSAMP_LOCAL_PYTHON="$DESSAMP_LOCAL_TEST_ENV/bin/python"
uv pip check --python "$DESSAMP_LOCAL_PYTHON"

gcloud auth login
gcloud auth application-default login
gcloud config set project "$DESSAMP_GCP_PROJECT"
gcloud storage buckets describe "gs://$DESSAMP_GCS_BUCKET"
```

Build and resolve the trainer image to an immutable digest. The helper uses
`conditioned-a2.gcloudignore` as a strict upload allowlist, confirms the GCP
project twice, and refuses a mutable implicit tag. Its `--dry-run` mode is
non-billable and is the required command review before an authorized build:

```bash
trainer_source/cloud/build_conditioned_a2_image.sh \
  --project="$DESSAMP_GCP_PROJECT" \
  --confirm-project="$DESSAMP_GCP_PROJECT" \
  --region="$DESSAMP_REGION" \
  --repository="$DESSAMP_ARTIFACT_REPOSITORY" \
  --image-name="$DESSAMP_IMAGE_NAME" \
  --image-tag="$DESSAMP_IMAGE_TAG" \
  --dry-run

# Remove --dry-run only after the exact billable build is authorized.
trainer_source/cloud/build_conditioned_a2_image.sh \
  --project="$DESSAMP_GCP_PROJECT" \
  --confirm-project="$DESSAMP_GCP_PROJECT" \
  --region="$DESSAMP_REGION" \
  --repository="$DESSAMP_ARTIFACT_REPOSITORY" \
  --image-name="$DESSAMP_IMAGE_NAME" \
  --image-tag="$DESSAMP_IMAGE_TAG"

gcloud artifacts docker images describe \
  "$DESSAMP_REGION-docker.pkg.dev/$DESSAMP_GCP_PROJECT/$DESSAMP_ARTIFACT_REPOSITORY/$DESSAMP_IMAGE_NAME:$DESSAMP_IMAGE_TAG" \
  --format='value(image_summary.digest)'
```

The upload context is intentionally limited to the root trainer license, the
conditioned-A2 implementation, and the exact Docker, dependency, entrypoint,
contract, and Cloud Build files needed for the image. Local environments,
tests, caches, checkpoints, the retained PANAMA package, and unrelated trainer
scripts must not enter the Cloud Build source archive. The cloud-helper suite
contains a semantic `gcloud meta list-files-for-upload` regression test for
that boundary.

The image dependency contract has three reviewed layers:

- `requirements-conditioned-a2.txt` declares the runtime contract plus the
  reviewed security pins that must replace vulnerable base-image packages;
- `requirements-conditioned-a2-image.constraints.txt` records the exact
  Python graph observed in the successful digest-pinned image; and
- `requirements-conditioned-a2.lock.txt` is the hash-locked CPython 3.11,
  Linux x86-64, CUDA 12.4 install graph.

The lock header contains its exact `uv pip compile` regeneration command.
Regenerate it only as an intentional dependency change, require a byte-for-byte
repeat generation during review, and review the resulting package and hash
diff. The Dockerfile installs it with `pip --require-hashes`, pins the CUDA base
image by digest, and asserts NumPy, Torch, and CUDA versions before and after
installation. The image lock is not a macOS local-environment lock; do not use
it to replace the local setup above. A real Linux/CUDA rebuild plus container
scan remains required before release.

Create RunPod registry authentication for the digest-pinned GAR image. Store
the resulting registry-auth ID in the run ledger; it is an identifier, not the
registry password. Add the RunPod API key to macOS Keychain Access as a generic
password with account `codex` and service `com.runpod.api`; enter the value
through the Keychain UI so it does not enter shell history.

```bash
chmod 600 "$DESSAMP_RUNPOD_GCP_CREDENTIAL_FILE"
```

Do not paste live credentials into documentation, Git, or shell history.

## Prepare the immutable bundle

Materialize and validate the corpus first, then upload every input with a
generation-match create guard:

```bash
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

`--data-root` is the explicit local read boundary for paths named by the data
config. The helper rejects paths outside it, symlinks, non-WAV inputs, and
oversized bundles. Never rerun `prepare` to change an existing run. A changed
corpus requires a new run ID and new immutable namespace.

### Paired data-efficiency members

An existing-corpus POC may share one immutable audio bundle while keeping a
full and reduced arm separate. Use the same fresh training seed for the two
matched arms, distinct `--member-variant` values, and
`--artifact-purpose=workflow_validation_only`. On only the reduced arm,
`--train-chunks-per-output=N` hash-binds deterministic uniform temporal strata
covering the complete input recording. The default omission remains the legacy
full sweep and preserves compatibility with existing immutable contracts.

These paired arms are an experiment, not independent ensemble members. Keep
the release-hidden reservation sealed, evaluate the predeclared POC validation
panel externally, and repeat a promising comparison with a fresh seed before
adopting it. A different chunk count, purpose, image, seed, or member variant
is a different immutable member and cannot reuse another arm's checkpoint.

## Submit independent members

Submit one on-demand Pod for each independent seed; using a shell loop is
acceptable only after checking each seed has no existing writer or member
contract. `--no-interruptible` is explicit and mandatory. Set the hourly rate
from the currently verified RunPod offer, the timeout to the owner-approved
wall-clock deadline, and the maximum to the owner-approved per-member spend.
All three cost arguments are required; no default silently expands authority.

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

Community or secure cloud may be used according to capacity. The exact GPU
type is part of the member contract, so do not change it during a continuation.
The helper submits the documented GraphQL `podFindAndDeployOnDemand` mutation
with `gpuTypeId`, `gpuCount`, `imageName`, `containerRegistryAuthId`,
`dockerArgs`, `env`, resource floors, `deployCost`, and `terminateAfter`.
`deployCost` receives the declared hourly rate. The returned `costPerHr` must
be finite and no greater than that declaration. Before any automatic
post-create deletion, the helper validates the returned Pod-ID grammar and
requested identity, then reads `/pods/{id}?includeMachine=true`. Successful
launch evidence requires the same exact ID and nonce-bound name, the
REST-native digest image with no conflicting aliases, registry identity,
desired state, current rate, interruptibility, GPU display name/count, machine
GPU type, and secure/community tier. A runtime mismatch after exact identity
proof deletes only that Pod and withholds evidence. If identity proof fails,
the helper refuses deletion and reports a potential orphan for manual review.
Authenticated requests reject redirects before the RunPod bearer or a scoped
GCS capability body can be forwarded.

Before creation, a nonce-bound per-member writer lease is atomically created
in GCS with generation zero. Its bounded expiry is tied to the provider
deadline. A live lease prevents a duplicate writer; an expired exact-contract
lease is reclaimed only by generation compare-and-swap after a bounded,
generation-stable read. The lease is intentionally not released early, so a
recovery monitor must retain global no-writer evidence and may need to wait for
expiry rather than bypassing the one-writer control.

The cost controls are intentionally layered:

1. the account preflight checks funded balance, current spend, and hourly
   spend limit;
2. `--max-cost-usd` bounds the declared rate times `--timeout-minutes` plus a
   small overhead reserve;
3. the returned provider rate is checked after creation; and
4. the absolute `terminateAfter` deadline is set by the provider at creation.

`terminateAfter` is a deletion guard, not merely a stop timer. RunPod's
official tooling distinguishes it from `stopAfter`: stopping releases the GPU
but can leave disk or volume billing, while termination deletes the Pod. This
deadline is a last-resort guard; delete a successfully quiescent Pod promptly
after stable GCS evidence is verified.

## Durable checkpoints and recovery

After each completed validation epoch the trainer mirrors:

- `durable/resume.ckpt`: model, optimizer, scheduler, epoch, and global step;
- `durable/best.ckpt`: best internal-validation state; and
- progress metadata used by the recovery monitor.

The completed validation epoch is the safe resume boundary. Mid-epoch SIGTERM
checkpointing is intentionally not used with the current shuffled,
non-stateful loader because it cannot guarantee the exact sample sequence.

When a Pod disappears, reaches its provider deadline, or becomes terminal:

1. query all RunPod Pods and prove no writer for that seed is pending/running;
2. inspect the latest GCS checkpoint generation and epoch/global step;
3. verify the immutable member contract still matches; and
4. submit the same command with `--resume-existing-member`.

```bash
"$DESSAMP_LOCAL_PYTHON" trainer_source/cloud/conditioned_a2_runpod_job.py submit \
  ...same exact arguments... \
  --resume-existing-member
```

Continue only on a fresh on-demand Pod after proving the previous writer is
absent. Use the same GPU type, cloud type, image digest, cost authority,
absolute training epoch target, and all other contract-bound inputs. A
continuation continues the same seed; it does not create a new ensemble
member. Never use generic MCP Pod creation for this recovery path.

## Completion and validation

A Pod state or final-looking filename is not acceptance. A member is complete
only when:

1. the writer has stopped or been intentionally quiesced;
2. no other Pod is writing the same seed;
3. `conditioned-a2-full.nam` and `training-manifest.json` have stable GCS
   generations;
4. the model hash matches the manifest; and
5. current DessAmp Core loads the downloaded model.

The launcher keeps a successfully completed container quiescent after the
trainer publishes its final artifacts. Before entering that state, the
entrypoint clears its already-consumed capability allowlist, then prints
`DESSAMP_TRAINING_COMPLETE_WAITING_FOR_POD_TERMINATION`. This prevents a Pod
whose desired state is still `RUNNING` from restarting the trainer and
republishing different object generations. The external monitor must verify
stable final GCS objects and then **delete**, not merely stop, the Pod promptly.
The provider `terminateAfter` deadline deletes it if that normal termination
protocol fails. Quiescence prevents duplicate publication; it does not prove
completion and is not permission to leave billable compute running.

```bash
gcloud storage cp \
  "gs://$DESSAMP_GCS_BUCKET/jobs/$DESSAMP_RUN_ID/results/full-seed-$DESSAMP_SEED/conditioned-a2-full.nam" \
  "$DESSAMP_MEMBER_DIR/conditioned-a2-full.nam"

gcloud storage cp \
  "gs://$DESSAMP_GCS_BUCKET/jobs/$DESSAMP_RUN_ID/results/full-seed-$DESSAMP_SEED/training-manifest.json" \
  "$DESSAMP_MEMBER_DIR/training-manifest.json"

shasum -a 256 "$DESSAMP_MEMBER_DIR/conditioned-a2-full.nam"
/tmp/dessamp-core-build/tools/loadmodel "$DESSAMP_MEMBER_DIR/conditioned-a2-full.nam"
```

Preserve the model's exact manifest-declared relative basename during
download. `select_conditioned_member_by_internal_validation.py` resolves the
artifact through `training-manifest.json` and deliberately rejects a renamed
or missing model.

Internal ESR is progress evidence only. After all independent members pass the
stable-final gate, run the frozen external HID evaluation through current Core
at receptive field 6,347. Only that external per-setting result decides whether
the model clears the quality gate or the next active-learning batch is needed.

## Provider contract references

The submitted field names and lifecycle semantics are checked against current
official RunPod sources:

- [RunPod GraphQL Pod management](https://docs.runpod.io/sdks/graphql/manage-pods)
  documents `podFindAndDeployOnDemand`, the GPU/resource/image/argument/env
  creation fields, and the separate stop operation.
- [RunPod Find a Pod by ID](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
  defines the fresh REST identity read used before destructive cleanup.
- [RunPod GraphQL schema](https://graphql-spec.dev.runpod.io/) declares
  `PodFindAndDeployOnDemandInput`, including `containerRegistryAuthId`,
  `deployCost`, `dockerArgs`, `env`, `terminateAfter`, and the returned
  `costPerHr` field.
- [RunPod's native Pulumi provider](https://github.com/runpod/pulumi-runpod-native/blob/main/provider/pod.go)
  sends the same on-demand mutation and `terminateAfter` field.
- [RunPod's official runpodctl skill](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpodctl/SKILL.md)
  states that `terminateAfter` deletes a Pod, whereas `stopAfter` only stops it
  and can leave disk or volume charges.

## Local verification

Run cloud-helper tests in their own process:

```bash
"$DESSAMP_LOCAL_PYTHON" -m unittest discover \
  -s trainer_source/cloud/tests -p 'test_*.py' -v

"$DESSAMP_LOCAL_PYTHON" -m py_compile \
  trainer_source/cloud/conditioned_a2_runpod_job.py \
  trainer_source/cloud/conditioned_a2_entrypoint.py
```

Run the official-NAM conditioned/import/materializer tests in a separate
pytest process from the rest of `trainer_source/tests`:

```bash
PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m pytest \
  trainer_source/tests/test_conditioned_a2.py \
  trainer_source/tests/test_import_teacher_distillation_plan.py \
  trainer_source/tests/test_materialize_conditioned_data_config.py -q

PYTHONPATH=capture_companion/src \
"$DESSAMP_LOCAL_PYTHON" -m pytest trainer_source/tests -q \
  --ignore=trainer_source/tests/test_conditioned_a2.py \
  --ignore=trainer_source/tests/test_import_teacher_distillation_plan.py \
  --ignore=trainer_source/tests/test_materialize_conditioned_data_config.py
```

Do not collapse those two pytest commands into one interpreter. A legacy
trainer test imports the repository's historical `trainer_source/nam` package
under the top-level `nam` name; import order can then shadow the official NAM
0.13 dependency used by conditioned A2 and create false failures. This test
isolation is intentional; it does not change the production dependency.

## Historical provider boundary

Historical Vertex experiment ledgers and reports retain their original
provider provenance outside the production launcher. New training always uses
the RunPod/GCS contract described above. Retained compatibility modules are not
templates for new jobs; their generic mutation commands are deliberately
disabled, and the former uncapped build context is absent from source.
