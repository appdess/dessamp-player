import importlib.util
import hashlib
import http.server
import io
import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "conditioned_a2_runpod_job.py"
SPEC = importlib.util.spec_from_file_location("conditioned_a2_runpod_job", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DIGEST_IMAGE_URI = (
    "europe-west3-docker.pkg.dev/dessamp-training-test/"
    "nam-training/conditioned-a2@sha256:" + "a" * 64
)
VALID_POD_ID = "abc123def456gh"
EXPECTED_GPU_TYPE = "NVIDIA A40"


def fresh_runtime_fields(*, rate="0.44", secure=True):
    return {
        "costPerHr": rate,
        "interruptible": False,
        "gpu": {"count": 1, "displayName": EXPECTED_GPU_TYPE},
        "machine": {
            "gpuTypeId": EXPECTED_GPU_TYPE,
            "secureCloud": secure,
        },
    }


class ConditionedA2RunPodTests(unittest.TestCase):
    def test_cross_origin_redirect_never_receives_runpod_bearers(self):
        origin_requests = []
        redirected_requests = []

        class RedirectTarget(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                redirected_requests.append(
                    (self.headers.get("Authorization"), self.rfile.read(length))
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                del args

        target = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), RedirectTarget
        )
        target_url = f"http://127.0.0.1:{target.server_port}/stolen"

        class RedirectOrigin(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                origin_requests.append(
                    (self.headers.get("Authorization"), self.rfile.read(length))
                )
                self.send_response(307)
                self.send_header("Location", target_url)
                self.end_headers()

            def log_message(self, *args):
                del args

        origin = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), RedirectOrigin
        )
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (target, origin)
        ]
        for thread in threads:
            thread.start()
        try:
            origin_url = f"http://127.0.0.1:{origin.server_port}"
            with mock.patch.object(MODULE, "RUNPOD_API", origin_url):
                with self.assertRaisesRegex(MODULE.PreflightError, r"failed \(307\)"):
                    MODULE.runpod_request(
                        api_key="runpod-bearer",
                        method="POST",
                        path="/pods",
                        payload={"secret": "temporary-capability"},
                    )
            with mock.patch.object(MODULE, "RUNPOD_GRAPHQL_API", origin_url):
                with self.assertRaisesRegex(MODULE.PreflightError, r"failed \(307\)"):
                    MODULE.runpod_graphql_request(
                        api_key="runpod-bearer",
                        query="mutation Test { createPod { id } }",
                        variables={"secret": "temporary-capability"},
                    )
        finally:
            for server in (origin, target):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)

        self.assertEqual(len(origin_requests), 2)
        self.assertTrue(
            all(value == "Bearer runpod-bearer" for value, _ in origin_requests)
        )
        self.assertEqual(redirected_requests, [])

    def test_runpod_requests_send_a_stable_non_default_user_agent(self):
        rest_response = mock.MagicMock()
        rest_response.__enter__.return_value.read.return_value = b'{}'
        graphql_response = mock.MagicMock()
        graphql_response.__enter__.return_value.read.return_value = b'{"data": {}}'
        with mock.patch.object(
            MODULE,
            "_open_runpod_url",
            side_effect=(rest_response, graphql_response),
        ) as opener:
            self.assertEqual(
                MODULE.runpod_request(api_key="not-a-live-key", method="GET", path="/pods"),
                {},
            )
            self.assertEqual(
                MODULE.runpod_graphql_request(
                    api_key="not-a-live-key",
                    query="query Test { myself { id } }",
                    variables={},
                ),
                {},
            )

        for call in opener.call_args_list:
            request = call.args[0]
            self.assertEqual(
                request.get_header("User-agent"),
                MODULE.RUNPOD_USER_AGENT,
            )

    def test_runpod_request_accepts_an_empty_successful_delete_response(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b""
        with mock.patch.object(MODULE, "_open_runpod_url", return_value=response):
            self.assertEqual(
                MODULE.runpod_request(
                    api_key="not-a-live-key",
                    method="DELETE",
                    path="/pods/pod-123",
                ),
                {},
            )

    def test_prepare_command_exposes_provider_neutral_gcs_bundle_arguments(self):
        args = MODULE.build_parser().parse_args(
            [
                "prepare",
                "--project",
                "dessamp-training",
                "--confirm-project",
                "dessamp-training",
                "--bucket",
                "dessamp-evidence",
                "--run-id",
                "amp-round-01",
                "--data-config",
                "/tmp/data.json",
                "--expected-train-captures",
                "30",
                "--control-order",
                "gain,bass",
            ]
        )
        self.assertEqual(args.command, "prepare")
        self.assertEqual(args.expected_train_captures, 30)
        self.assertEqual(args.control_order, "gain,bass")

    def test_submit_parser_requires_explicit_on_demand_opt_out(self):
        args = MODULE.build_parser().parse_args(
            [
                "submit",
                "--project",
                "dessamp-training",
                "--confirm-project",
                "dessamp-training",
                "--bucket",
                "dessamp-evidence",
                "--run-id",
                "amp-round-01",
                "--image-uri",
                "europe-west3-docker.pkg.dev/dessamp-training/nam/a2@sha256:"
                + "a" * 64,
                "--registry-auth-id",
                "registry-1",
                "--google-credentials",
                "/tmp/gcp.json",
                "--child",
                "full",
                "--seed",
                "81",
                "--control-order",
                "gain,bass",
                "--timeout-minutes",
                "600",
                "--hourly-rate-usd",
                "0.44",
                "--max-cost-usd",
                "5.00",
            ]
        )
        self.assertEqual(args.command, "submit")
        self.assertEqual(args.gpu_type, "NVIDIA RTX A5000")
        self.assertTrue(args.interruptible)
        self.assertEqual(args.gcs_signing_credentials, Path("/tmp/gcp.json"))
        self.assertEqual(args.timeout_minutes, 600)
        self.assertEqual(args.max_cost_usd, Decimal("5.00"))
        self.assertIsNone(args.member_variant)
        self.assertIsNone(args.train_chunks_per_output)
        self.assertEqual(args.artifact_purpose, "production_candidate")

    def test_efficiency_member_uses_distinct_prefix_and_contract_bound_arguments(self):
        arguments = MODULE.conditioned_container_args(
            project="dessamp-training",
            bucket="dessamp-evidence",
            run_id="amp-efficiency-poc",
            image_uri=DIGEST_IMAGE_URI,
            child="full",
            seed=110,
            member_config_sha256="b" * 64,
            control_order=("gain", "bass"),
            epochs=60,
            batch_size=16,
            ny=8192,
            validation_ny=262144,
            num_workers=4,
            expected_train_captures=40,
            machine_type="runpod-pod",
            accelerator_type="NVIDIA A40",
            result_prefix=(
                "jobs/amp-efficiency-poc/results/full-seed-110-stratified95"
            ),
            train_chunks_per_output=556,
            artifact_purpose="workflow_validation_only",
        )
        self.assertEqual(
            arguments[arguments.index("--result-prefix") + 1],
            "jobs/amp-efficiency-poc/results/full-seed-110-stratified95",
        )
        self.assertEqual(
            arguments[arguments.index("--train-chunks-per-output") + 1], "556"
        )
        self.assertEqual(
            arguments[arguments.index("--artifact-purpose") + 1],
            "workflow_validation_only",
        )

    def test_submit_rejects_interruptible_before_any_runpod_call(self):
        args = SimpleNamespace(
            project="dessamp-training",
            confirm_project="dessamp-training",
            interruptible=True,
        )
        with mock.patch.object(MODULE, "runpod_account_snapshot") as account_snapshot:
            with self.assertRaisesRegex(MODULE.PreflightError, "--no-interruptible"):
                MODULE.submit(args)
        account_snapshot.assert_not_called()

    def test_interruptible_a5000_spec_keeps_capabilities_out_of_redacted_evidence(self):
        spec = MODULE.build_pod_spec(
            name="dessamp-fireball-r4-s81",
            image_uri=(
                "europe-west3-docker.pkg.dev/dessamp-training-test/"
                "nam-training/conditioned-a2@sha256:" + "a" * 64
            ),
            registry_auth_id="registry-1",
            gcs_capabilities_b64="scoped-capability",
            gpu_type="NVIDIA RTX A5000",
            cloud_type="COMMUNITY",
            interruptible=True,
            container_args=["--seed", "81", "--epochs", "60"],
        )
        self.assertTrue(spec["interruptible"])
        self.assertEqual(spec["gpuTypeIds"], ["NVIDIA RTX A5000"])
        self.assertEqual(spec["minVCPUPerGPU"], 8)
        self.assertIn(MODULE.CAPABILITY_ENVIRONMENT_VARIABLE, spec["env"])
        start_command = spec["dockerStartCmd"][0]
        self.assertIn("conditioned_a2_entrypoint.py", start_command)
        self.assertIn("DESSAMP_TRAINING_COMPLETE_WAITING_FOR_POD_STOP", start_command)
        self.assertIn("exec tail -f /dev/null", start_command)
        self.assertIn(f"unset {MODULE.CAPABILITY_ENVIRONMENT_VARIABLE}", start_command)
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", start_command)
        self.assertNotIn("base64 -d", start_command)
        self.assertNotIn("exec python -u", start_command)
        redacted = MODULE.redacted_pod_spec(spec)
        self.assertEqual(
            redacted["env"][MODULE.CAPABILITY_ENVIRONMENT_VARIABLE], "<redacted>"
        )
        self.assertNotIn("scoped-capability", str(redacted))

    def test_mutable_image_is_rejected(self):
        with self.assertRaisesRegex(MODULE.PreflightError, "digest-pinned"):
            MODULE.build_pod_spec(
                name="dessamp-fireball-r4-s81",
                image_uri="registry.example.com/trainer:latest",
                registry_auth_id="registry-1",
                gcs_capabilities_b64="scoped-capability",
                gpu_type="NVIDIA RTX A5000",
                cloud_type="COMMUNITY",
                interruptible=True,
                container_args=["--seed", "81"],
            )

    def test_cost_guard_binds_timeout_below_authorized_member_maximum(self):
        guard = MODULE.bounded_runpod_cost(
            hourly_rate_usd=Decimal("0.44"),
            timeout_minutes=600,
            authorized_max_usd=Decimal("5.00"),
        )
        self.assertEqual(guard["compute_ceiling_usd"], "4.4000")
        self.assertEqual(guard["total_ceiling_usd"], "4.5000")
        with self.assertRaisesRegex(MODULE.PreflightError, "exceeds authorization"):
            MODULE.bounded_runpod_cost(
                hourly_rate_usd=Decimal("0.50"),
                timeout_minutes=600,
                authorized_max_usd=Decimal("5.00"),
            )

    def test_on_demand_graphql_input_has_provider_termination_and_redacts_capabilities(self):
        spec = MODULE.build_on_demand_graphql_input(
            name="dessamp-mesa-v2-s101",
            image_uri=(
                "europe-west3-docker.pkg.dev/dessamp-training-test/"
                "nam-training/conditioned-a2@sha256:" + "a" * 64
            ),
            registry_auth_id="registry-1",
            gcs_capabilities_b64="scoped-capability",
            gpu_type="NVIDIA A40",
            cloud_type="SECURE",
            container_args=["--seed", "101", "--runpod-quiesce-on-success"],
            hourly_rate_usd=Decimal("0.44"),
            terminate_after="2026-08-25T00:00:00Z",
        )
        self.assertEqual(spec["deployCost"], 0.44)
        self.assertEqual(spec["terminateAfter"], "2026-08-25T00:00:00Z")
        self.assertIn("--runpod-quiesce-on-success", spec["dockerArgs"])
        redacted = MODULE.redacted_graphql_input(spec)
        self.assertNotIn("scoped-capability", str(redacted))
        self.assertIn("<redacted>", str(redacted))

    def test_signed_url_submission_preserves_community_cloud_support(self):
        spec = MODULE.build_on_demand_graphql_input(
            name="dessamp-mesa-v2-s101",
            image_uri=DIGEST_IMAGE_URI,
            registry_auth_id="registry-1",
            gcs_capabilities_b64="scoped-capability",
            gpu_type="NVIDIA A40",
            cloud_type="COMMUNITY",
            container_args=["--seed", "101"],
            hourly_rate_usd=Decimal("0.44"),
            terminate_after="2026-08-25T00:00:00Z",
        )
        self.assertEqual(spec["cloudType"], "COMMUNITY")
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", str(spec))

    def test_provider_error_redaction_removes_echoed_capability(self):
        payload = {
            "input": {
                "env": [
                    {
                        "key": MODULE.CAPABILITY_ENVIRONMENT_VARIABLE,
                        "value": "scoped-capability",
                    }
                ]
            }
        }
        redacted = MODULE.redacted_provider_detail(
            "provider echoed scoped-capability in diagnostics", payload
        )
        self.assertNotIn("scoped-capability", redacted)
        self.assertNotIn("provider echoed", redacted)
        self.assertEqual(redacted, MODULE.RUNPOD_PROVIDER_DETAIL_REDACTED)

    def test_provider_http_error_bodies_are_bounded_and_never_rendered(self):
        api_key = "runpod-api-secret"
        capability = "temporary-gcs-capability"
        service_account_fragment = "synthetic-service-account-secret"
        provider_body = (
            f"echo {api_key} {capability} {service_account_fragment}".encode()
            + b"x" * (MODULE.MAX_RUNPOD_ERROR_BYTES * 2)
        )

        class RecordingHTTPError(MODULE.urllib.error.HTTPError):
            def __init__(self):
                super().__init__(
                    MODULE.RUNPOD_API,
                    400,
                    f"echo {api_key}",
                    hdrs=None,
                    fp=io.BytesIO(provider_body),
                )
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return self.fp.read(size)

        for request_kind in ("rest", "graphql"):
            with self.subTest(request_kind=request_kind):
                error = RecordingHTTPError()
                with mock.patch.object(
                    MODULE, "_open_runpod_url", side_effect=error
                ):
                    with self.assertRaises(MODULE.PreflightError) as raised:
                        if request_kind == "rest":
                            MODULE.runpod_request(
                                api_key=api_key,
                                method="POST",
                                path="/pods",
                                payload={
                                    MODULE.CAPABILITY_ENVIRONMENT_VARIABLE: capability
                                },
                            )
                        else:
                            MODULE.runpod_graphql_request(
                                api_key=api_key,
                                query="mutation Test { createPod { id } }",
                                variables={
                                    "input": {
                                        "env": [
                                            {
                                                "key": MODULE.CAPABILITY_ENVIRONMENT_VARIABLE,
                                                "value": capability,
                                            }
                                        ]
                                    }
                                },
                            )
                rendered = str(raised.exception)
                self.assertEqual(
                    error.read_sizes, [MODULE.MAX_RUNPOD_ERROR_BYTES + 1]
                )
                self.assertIn(MODULE.RUNPOD_PROVIDER_DETAIL_REDACTED, rendered)
                self.assertNotIn(api_key, rendered)
                self.assertNotIn(capability, rendered)
                self.assertNotIn(service_account_fragment, rendered)

    def test_graphql_error_messages_and_malformed_json_never_echo_secrets(self):
        secret = "provider-echoed-api-or-capability-secret"
        responses = (
            json.dumps({"errors": [{"message": secret}]}).encode(),
            f'{{"secret":"{secret}"'.encode(),
        )
        expected_messages = (
            MODULE.RUNPOD_PROVIDER_DETAIL_REDACTED,
            "returned invalid JSON",
        )
        for raw, expected in zip(responses, expected_messages):
            with self.subTest(expected=expected):
                response = mock.MagicMock()
                response.__enter__.return_value.read.return_value = raw
                with mock.patch.object(
                    MODULE, "_open_runpod_url", return_value=response
                ):
                    with self.assertRaises(MODULE.PreflightError) as raised:
                        MODULE.runpod_graphql_request(
                            api_key=secret,
                            query="query Test { myself { id } }",
                            variables={},
                        )
                rendered = str(raised.exception)
                self.assertIn(expected, rendered)
                self.assertNotIn(secret, rendered)
                response.__enter__.return_value.read.assert_called_once_with(
                    MODULE.MAX_RUNPOD_RESPONSE_BYTES + 1
                )

    def test_failed_submission_lease_release_requires_exact_pod_name_absence(self):
        expected_name = "dessamp-fireball-r4-s81-" + "1" * 32
        with mock.patch.object(
            MODULE,
            "runpod_graphql_request",
            return_value={
                "myself": {
                    "pods": [
                        {
                            "id": VALID_POD_ID,
                            "name": "unrelated-pod",
                            "desiredStatus": "RUNNING",
                        }
                    ]
                }
            },
        ), mock.patch.object(
            MODULE,
            "release_member_writer_lease",
            return_value={"state": "failed-submission-lease-released"},
        ) as release:
            result = MODULE.release_failed_submission_writer_lease(
                api_key="not-a-live-key",
                project="dessamp-training",
                bucket="dessamp-evidence",
                result_prefix="jobs/fireball/results/full-seed-81",
                member_contract_sha256="a" * 64,
                attempt_nonce="1" * 32,
                pod_name=expected_name,
            )
        self.assertEqual(result["state"], "failed-submission-lease-released")
        release.assert_called_once()

        with mock.patch.object(
            MODULE,
            "runpod_graphql_request",
            return_value={
                "myself": {
                    "pods": [
                        {
                            "id": VALID_POD_ID,
                            "name": expected_name,
                            "desiredStatus": "RUNNING",
                        }
                    ]
                }
            },
        ), mock.patch.object(MODULE, "release_member_writer_lease") as release:
            with self.assertRaisesRegex(MODULE.PreflightError, "exact attempted Pod"):
                MODULE.release_failed_submission_writer_lease(
                    api_key="not-a-live-key",
                    project="dessamp-training",
                    bucket="dessamp-evidence",
                    result_prefix="jobs/fireball/results/full-seed-81",
                    member_contract_sha256="a" * 64,
                    attempt_nonce="1" * 32,
                    pod_name=expected_name,
                )
        release.assert_not_called()

    def test_provider_success_responses_are_size_bounded(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            b"x" * (MODULE.MAX_RUNPOD_RESPONSE_BYTES + 1)
        )
        with mock.patch.object(
            MODULE, "_open_runpod_url", return_value=response
        ):
            with self.assertRaisesRegex(MODULE.PreflightError, "size limit"):
                MODULE.runpod_request(
                    api_key="not-a-live-key", method="GET", path="/pods"
                )
        response.__enter__.return_value.read.assert_called_once_with(
            MODULE.MAX_RUNPOD_RESPONSE_BYTES + 1
        )

    def test_local_signing_credential_requires_private_regular_file(self):
        payload = {
            "type": "service_account",
            "project_id": "dessamp-training",
            "client_email": "trainer@dessamp-training.iam.gserviceaccount.com",
        }
        loaded = SimpleNamespace(service_account_email=payload["client_email"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "signer.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            path.chmod(0o600)
            with mock.patch(
                "google.oauth2.service_account.Credentials.from_service_account_info",
                return_value=loaded,
            ):
                self.assertIs(
                    MODULE.load_gcs_signing_credentials(
                        path, project="dessamp-training"
                    ),
                    loaded,
                )

            path.chmod(0o644)
            with self.assertRaisesRegex(MODULE.PreflightError, "group/world"):
                MODULE.load_gcs_signing_credentials(path, project="dessamp-training")

            path.chmod(0o600)
            link = Path(temporary_directory) / "signer-link.json"
            link.symlink_to(path)
            with self.assertRaises(MODULE.PreflightError):
                MODULE.load_gcs_signing_credentials(link, project="dessamp-training")

    def test_capability_generator_scopes_only_bound_inputs_checkpoints_and_outputs(self):
        run_id = "amp-round-01"
        bucket = "dessamp-evidence"
        result_prefix = f"jobs/{run_id}/results/full-seed-81"
        input_name = f"jobs/{run_id}/data/capture.wav"
        input_payload = b"RIFFdataWAVE"
        input_manifest = {
            "objects": [
                {
                    "name": input_name,
                    "size_bytes": len(input_payload),
                    "sha256": hashlib.sha256(input_payload).hexdigest(),
                }
            ]
        }
        member_contract = {"schema_version": 1, "identity": {"seed": 81}}
        member_sha256 = MODULE.canonical_json_sha256(member_contract)
        signing_calls = []

        class FakeBlob:
            def __init__(self, name):
                self.name = name

            def generate_signed_url(self, **kwargs):
                signing_calls.append((self.name, kwargs))
                signed_headers = (
                    "content-type%3Bhost%3Bx-goog-content-length-range"
                    if kwargs["method"] == "PUT"
                    else "host"
                )
                return (
                    f"https://storage.googleapis.com/{bucket}/{self.name}"
                    "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
                    f"&X-Goog-SignedHeaders={signed_headers}"
                    "&X-Goog-Signature=abc"
                )

        class FakeBucket:
            def blob(self, name):
                return FakeBlob(name)

        client = mock.MagicMock()
        client.bucket.return_value = FakeBucket()
        credentials = SimpleNamespace(service_account_email="trainer@example.test")
        with mock.patch.object(
            MODULE, "load_gcs_signing_credentials", return_value=credentials
        ), mock.patch("google.cloud.storage.Client", return_value=client):
            encoded, summary = MODULE.generate_gcs_capability_bundle(
                project="dessamp-training",
                bucket=bucket,
                run_id=run_id,
                result_prefix=result_prefix,
                child="full",
                member_config_sha256=member_sha256,
                member_contract=member_contract,
                input_manifest=input_manifest,
                signing_credentials_path=Path("/not/read.json"),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )

        from trainer_source.cloud.gcs_capabilities import SignedUrlStorageClient

        scoped = SignedUrlStorageClient.from_encoded(encoded)
        read_names = set(scoped._read)
        write_names = set(scoped._write)
        self.assertIn(input_name, read_names)
        self.assertIn(
            f"{result_prefix}/durable/{member_sha256}/resume.ckpt", read_names
        )
        self.assertIn(f"{result_prefix}/conditioned-a2-full.nam", write_names)
        self.assertNotIn(f"jobs/{run_id}/private/hidden.wav", read_names)
        self.assertEqual(summary["read_object_count"], len(read_names))
        self.assertEqual(summary["write_object_count"], len(write_names))
        self.assertNotIn("https://", str(summary))
        self.assertTrue(any(kwargs["method"] == "GET" for _, kwargs in signing_calls))
        put_calls = [kwargs for _, kwargs in signing_calls if kwargs["method"] == "PUT"]
        self.assertEqual(len(put_calls), len(write_names))
        for kwargs in put_calls:
            maximum = int(kwargs["headers"]["x-goog-content-length-range"].split(",")[1])
            self.assertGreater(maximum, 0)
            self.assertEqual(kwargs["content_type"], "application/octet-stream")
        for entry in scoped._write.values():
            self.assertEqual(
                entry["headers"]["x-goog-content-length-range"],
                f"1,{entry['max_bytes']}",
            )

    def test_production_sources_do_not_send_a_reusable_google_key_to_runpod(self):
        for path in (SCRIPT_PATH, SCRIPT_PATH.with_name("conditioned_a2_entrypoint.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("GOOGLE_SERVICE_ACCOUNT_B64", text)
            self.assertNotIn("/tmp/dessamp-gcp.json", text)

    def test_runpod_cost_parser_rejects_missing_invalid_and_nonfinite_values(self):
        for value in (None, "", "not-a-number", "NaN", "Infinity", "-0.01"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(MODULE.PreflightError, "invalid costPerHr"):
                    MODULE.runpod_billing_decimal(value, field="costPerHr")
        self.assertEqual(
            MODULE.runpod_billing_decimal("0.44", field="costPerHr"),
            Decimal("0.44"),
        )

    def test_pod_id_validation_accepts_only_documented_opaque_grammar(self):
        self.assertEqual(
            MODULE.validated_runpod_pod_id(VALID_POD_ID), VALID_POD_ID
        )
        invalid_values = (
            None,
            "abc123def456g",
            "abc123def456ghi",
            "ABC123def456gh",
            "abc123-def456g",
            "abc123_def456g",
            "abc123/def456g",
            "abc123def456g\n",
            "abc123def456gé",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    MODULE.PreflightError, "exactly 14 lowercase ASCII"
                ):
                    MODULE.validated_runpod_pod_id(value)

    def test_invalid_created_pod_cost_reidentifies_then_terminates_exact_pod(self):
        created_pod = {
            "id": VALID_POD_ID,
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": None,
        }
        observed_pod = {
            **created_pod,
            "image": created_pod["imageName"],
            "containerRegistryAuthId": "registry-1",
        }
        with mock.patch.object(
            MODULE, "runpod_request", side_effect=(observed_pod, {})
        ) as request:
            with self.assertRaisesRegex(
                MODULE.PreflightError,
                f"invalid costPerHr; verified RunPod Pod {VALID_POD_ID} DELETE requested",
            ):
                MODULE.validate_created_pod_cost(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name="dessamp-fireball-r4-s81",
                    expected_image_uri=created_pod["imageName"],
                    expected_registry_auth_id="registry-1",
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        self.assertEqual(
            request.call_args_list,
            [
                mock.call(
                    api_key="not-a-live-key",
                    method="GET",
                    path=f"/pods/{VALID_POD_ID}?includeMachine=true",
                ),
                mock.call(
                    api_key="not-a-live-key",
                    method="DELETE",
                    path=f"/pods/{VALID_POD_ID}",
                ),
            ],
        )

    def test_valid_created_pod_identity_and_cost_preserve_normal_submission(self):
        created_pod = {
            "id": VALID_POD_ID,
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": "0.44",
        }
        with mock.patch.object(MODULE, "runpod_request") as request:
            self.assertEqual(
                MODULE.validate_created_pod_cost(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name="dessamp-fireball-r4-s81",
                    expected_image_uri=created_pod["imageName"],
                    expected_registry_auth_id="registry-1",
                    declared_hourly_rate_usd=Decimal("0.44"),
                ),
                Decimal("0.44"),
            )
        request.assert_not_called()

    def test_above_ceiling_cost_reidentifies_then_terminates_exact_pod(self):
        created_pod = {
            "id": VALID_POD_ID,
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": "0.45",
        }
        observed_pod = {
            "id": VALID_POD_ID,
            "name": created_pod["name"],
            "image": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "containerRegistryAuthId": "registry-1",
        }
        with mock.patch.object(
            MODULE, "runpod_request", side_effect=(observed_pod, {})
        ) as request:
            with self.assertRaisesRegex(
                MODULE.PreflightError,
                f"above the declared.*verified RunPod Pod {VALID_POD_ID} DELETE requested",
            ):
                MODULE.validate_created_pod_cost(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name=created_pod["name"],
                    expected_image_uri=DIGEST_IMAGE_URI,
                    expected_registry_auth_id="registry-1",
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        self.assertEqual(
            [call.kwargs["method"] for call in request.call_args_list],
            ["GET", "DELETE"],
        )

    def test_invalid_created_pod_identity_never_reaches_delete(self):
        malicious = {
            "id": "victim/../../pod",
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": None,
        }
        with mock.patch.object(MODULE, "runpod_request") as request:
            with self.assertRaisesRegex(
                MODULE.PreflightError, "invalid Pod identity"
            ):
                MODULE.validate_created_pod_cost(
                    api_key="not-a-live-key",
                    created_pod=malicious,
                    expected_name="dessamp-fireball-r4-s81",
                    expected_image_uri=malicious["imageName"],
                    expected_registry_auth_id="registry-1",
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        request.assert_not_called()

    def test_every_fresh_identity_mismatch_reports_orphan_without_delete(self):
        created_pod = {
            "id": VALID_POD_ID,
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": None,
        }
        observed_pod = {
            **created_pod,
            "image": DIGEST_IMAGE_URI,
            "containerRegistryAuthId": "registry-1",
        }
        mismatches = {
            "id": "abc123def456gi",
            "name": "unrelated-training-pod",
            "imageName": DIGEST_IMAGE_URI.replace("a" * 64, "b" * 64),
            "desiredStatus": "STOPPED",
            "containerRegistryAuthId": "other-registry",
        }
        for field, mismatch in mismatches.items():
            with self.subTest(field=field):
                observed = {**observed_pod, field: mismatch}
                with mock.patch.object(
                    MODULE, "runpod_request", return_value=observed
                ) as request:
                    with self.assertRaisesRegex(
                        MODULE.PreflightError,
                        "possible orphaned RunPod Pod.*automatic DELETE was not issued",
                    ):
                        MODULE.validate_created_pod_cost(
                            api_key="not-a-live-key",
                            created_pod=created_pod,
                            expected_name=created_pod["name"],
                            expected_image_uri=DIGEST_IMAGE_URI,
                            expected_registry_auth_id="registry-1",
                            declared_hourly_rate_usd=Decimal("0.44"),
                        )
                request.assert_called_once_with(
                    api_key="not-a-live-key",
                    method="GET",
                    path=f"/pods/{VALID_POD_ID}?includeMachine=true",
                )

    def test_delete_failure_after_identity_proof_reports_possible_orphan(self):
        provider_secret = "provider-echoed-secret"
        created_pod = {
            "id": VALID_POD_ID,
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": None,
        }
        observed_pod = {
            **created_pod,
            "image": DIGEST_IMAGE_URI,
            "containerRegistryAuthId": "registry-1",
        }
        with mock.patch.object(
            MODULE,
            "runpod_request",
            side_effect=(
                observed_pod,
                MODULE.PreflightError(provider_secret),
            ),
        ) as request:
            with self.assertRaises(MODULE.PreflightError) as raised:
                MODULE.validate_created_pod_cost(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name=created_pod["name"],
                    expected_image_uri=DIGEST_IMAGE_URI,
                    expected_registry_auth_id="registry-1",
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        rendered = str(raised.exception)
        self.assertIn("may be orphaned because DELETE failed", rendered)
        self.assertNotIn(provider_secret, rendered)
        self.assertEqual(
            request.call_args_list,
            [
                mock.call(
                    api_key="not-a-live-key",
                    method="GET",
                    path=f"/pods/{VALID_POD_ID}?includeMachine=true",
                ),
                mock.call(
                    api_key="not-a-live-key",
                    method="DELETE",
                    path=f"/pods/{VALID_POD_ID}",
                ),
            ],
        )

    def test_created_pod_evidence_is_field_allowlisted(self):
        provider_secret = "provider-extra-secret"
        attempt_nonce = "1" * 32
        created_pod = {
            "id": VALID_POD_ID,
            "name": f"dessamp-fireball-r4-s81-{attempt_nonce}",
            "imageName": DIGEST_IMAGE_URI,
            "image": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "containerRegistryAuthId": "registry-1",
            **fresh_runtime_fields(),
            "lastStatusChange": provider_secret,
            "env": [{"value": provider_secret}],
        }
        evidence = MODULE.allowlisted_created_pod_evidence(
            created_pod,
            expected_name=created_pod["name"],
            expected_image_uri=DIGEST_IMAGE_URI,
            expected_registry_auth_id="registry-1",
            attempt_nonce=attempt_nonce,
            expected_gpu_type=EXPECTED_GPU_TYPE,
            expected_gpu_count=1,
            expected_cloud_type="SECURE",
            expected_interruptible=False,
            declared_hourly_rate_usd=Decimal("0.44"),
        )
        self.assertEqual(
            set(evidence),
            {
                "id",
                "name",
                "desiredStatus",
                "costPerHr",
                "imageName",
                "containerRegistryAuthId",
                "attemptNonce",
                "gpuType",
                "gpuCount",
                "cloudType",
                "interruptible",
            },
        )
        self.assertNotIn(provider_secret, str(evidence))

    def test_normal_success_freshly_proves_exact_pod_before_evidence(self):
        attempt_nonce = "2" * 32
        pod_name = f"dessamp-fireball-r4-s81-{attempt_nonce}"
        created_pod = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": "0.44",
        }
        observed_rest_pod = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "containerRegistryAuthId": "registry-1",
            "costPerHr": "0.40",
            "gpuCount": 1,
            "machine": {
                "gpuTypeId": EXPECTED_GPU_TYPE,
                "secureCloud": True,
            },
        }
        observed_graphql_pod = {
            **observed_rest_pod,
            "podType": "RESERVED",
        }
        with mock.patch.object(
            MODULE, "runpod_request", return_value=observed_rest_pod
        ) as request, mock.patch.object(
            MODULE,
            "runpod_graphql_request",
            return_value={"pod": observed_graphql_pod},
        ) as graphql_request:
            evidence = MODULE.verified_created_pod_evidence(
                api_key="not-a-live-key",
                created_pod=created_pod,
                expected_name=pod_name,
                expected_image_uri=DIGEST_IMAGE_URI,
                expected_registry_auth_id="registry-1",
                attempt_nonce=attempt_nonce,
                expected_gpu_type=EXPECTED_GPU_TYPE,
                expected_gpu_count=1,
                expected_cloud_type="SECURE",
                expected_interruptible=False,
                declared_hourly_rate_usd=Decimal("0.44"),
            )
        request.assert_called_once_with(
            api_key="not-a-live-key",
            method="GET",
            path=f"/pods/{VALID_POD_ID}?includeMachine=true",
        )
        graphql_request.assert_called_once_with(
            api_key="not-a-live-key",
            query=MODULE.FRESH_POD_RUNTIME_QUERY,
            variables={"input": {"podId": VALID_POD_ID}},
        )
        self.assertEqual(evidence["attemptNonce"], attempt_nonce)
        self.assertEqual(evidence["costPerHr"], "0.40")

    def test_fresh_identity_rejects_conflicting_image_aliases_without_delete(self):
        attempt_nonce = "4" * 32
        pod_name = f"dessamp-fireball-r4-s81-{attempt_nonce}"
        created_pod = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": "0.44",
        }
        observed_pod = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "image": DIGEST_IMAGE_URI,
            "imageName": DIGEST_IMAGE_URI.replace("a" * 64, "b" * 64),
            "desiredStatus": "RUNNING",
            "containerRegistryAuthId": "registry-1",
            **fresh_runtime_fields(),
        }
        with mock.patch.object(
            MODULE, "runpod_request", return_value=observed_pod
        ) as request:
            with self.assertRaisesRegex(
                MODULE.PreflightError,
                "possible orphan requiring manual review.*evidence was not persisted",
            ):
                MODULE.verified_created_pod_evidence(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name=pod_name,
                    expected_image_uri=DIGEST_IMAGE_URI,
                    expected_registry_auth_id="registry-1",
                    attempt_nonce=attempt_nonce,
                    expected_gpu_type=EXPECTED_GPU_TYPE,
                    expected_gpu_count=1,
                    expected_cloud_type="SECURE",
                    expected_interruptible=False,
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        request.assert_called_once_with(
            api_key="not-a-live-key",
            method="GET",
            path=f"/pods/{VALID_POD_ID}?includeMachine=true",
        )

    def test_fresh_runtime_mismatches_delete_exact_pod_and_withhold_evidence(self):
        attempt_nonce = "5" * 32
        pod_name = f"dessamp-fireball-r4-s81-{attempt_nonce}"
        created_pod = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": "0.44",
        }
        base_observed = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "image": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "containerRegistryAuthId": "registry-1",
            **fresh_runtime_fields(),
        }
        mutations = {
            "live-rate": lambda pod: pod.__setitem__("costPerHr", "999"),
            "interruptible": lambda pod: pod.__setitem__("interruptible", True),
            "gpu-count": lambda pod: pod["gpu"].__setitem__("count", 8),
            "gpu-display": lambda pod: pod["gpu"].__setitem__("displayName", "NVIDIA L4"),
            "machine-gpu": lambda pod: pod["machine"].__setitem__("gpuTypeId", "NVIDIA L4"),
            "cloud-tier": lambda pod: pod["machine"].__setitem__("secureCloud", False),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                observed = json.loads(json.dumps(base_observed))
                observed["imageName"] = observed.pop("image")
                mutate(observed)
                with mock.patch.object(
                    MODULE, "runpod_request", side_effect=(base_observed, {})
                ) as request, mock.patch.object(
                    MODULE,
                    "runpod_graphql_request",
                    return_value={"pod": observed},
                ) as graphql_request:
                    with self.assertRaisesRegex(
                        MODULE.PreflightError,
                        f"Fresh RunPod Pod {VALID_POD_ID} runtime proof failed; exact Pod DELETE requested",
                    ):
                        MODULE.verified_created_pod_evidence(
                            api_key="not-a-live-key",
                            created_pod=created_pod,
                            expected_name=pod_name,
                            expected_image_uri=DIGEST_IMAGE_URI,
                            expected_registry_auth_id="registry-1",
                            attempt_nonce=attempt_nonce,
                            expected_gpu_type=EXPECTED_GPU_TYPE,
                            expected_gpu_count=1,
                            expected_cloud_type="SECURE",
                            expected_interruptible=False,
                            declared_hourly_rate_usd=Decimal("0.44"),
                        )
                self.assertEqual(
                    request.call_args_list,
                    [
                        mock.call(
                            api_key="not-a-live-key",
                            method="GET",
                            path=f"/pods/{VALID_POD_ID}?includeMachine=true",
                        ),
                        mock.call(
                            api_key="not-a-live-key",
                            method="DELETE",
                            path=f"/pods/{VALID_POD_ID}",
                        ),
                    ],
                )
                graphql_request.assert_called_once_with(
                    api_key="not-a-live-key",
                    query=MODULE.FRESH_POD_RUNTIME_QUERY,
                    variables={"input": {"podId": VALID_POD_ID}},
                )

    def test_normal_success_proof_failure_reports_orphan_and_withholds_evidence(self):
        attempt_nonce = "3" * 32
        pod_name = f"dessamp-fireball-r4-s81-{attempt_nonce}"
        created_pod = {
            "id": VALID_POD_ID,
            "name": pod_name,
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": "0.44",
        }
        observed_other = {
            **created_pod,
            "name": "unrelated-pod",
            "containerRegistryAuthId": "registry-1",
        }
        with mock.patch.object(
            MODULE, "runpod_request", return_value=observed_other
        ) as request:
            with self.assertRaisesRegex(
                MODULE.PreflightError,
                "possible orphan requiring manual review.*evidence was not persisted",
            ):
                MODULE.verified_created_pod_evidence(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name=pod_name,
                    expected_image_uri=DIGEST_IMAGE_URI,
                    expected_registry_auth_id="registry-1",
                    attempt_nonce=attempt_nonce,
                    expected_gpu_type=EXPECTED_GPU_TYPE,
                    expected_gpu_count=1,
                    expected_cloud_type="SECURE",
                    expected_interruptible=False,
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        self.assertEqual(len(request.call_args_list), 1)

    def test_identity_mismatch_refuses_automatic_termination(self):
        created_pod = {
            "id": VALID_POD_ID,
            "name": "dessamp-fireball-r4-s81",
            "imageName": DIGEST_IMAGE_URI,
            "desiredStatus": "RUNNING",
            "costPerHr": None,
        }
        observed_other_pod = {
            **created_pod,
            "name": "unrelated-training-pod",
            "image": created_pod["imageName"],
            "containerRegistryAuthId": "registry-1",
        }
        with mock.patch.object(
            MODULE, "runpod_request", return_value=observed_other_pod
        ) as request:
            with self.assertRaisesRegex(
                MODULE.PreflightError,
                "possible orphaned RunPod Pod.*automatic DELETE was not issued",
            ):
                MODULE.validate_created_pod_cost(
                    api_key="not-a-live-key",
                    created_pod=created_pod,
                    expected_name="dessamp-fireball-r4-s81",
                    expected_image_uri=created_pod["imageName"],
                    expected_registry_auth_id="registry-1",
                    declared_hourly_rate_usd=Decimal("0.44"),
                )
        request.assert_called_once_with(
            api_key="not-a-live-key",
            method="GET",
            path=f"/pods/{VALID_POD_ID}?includeMachine=true",
        )


if __name__ == "__main__":
    unittest.main()
