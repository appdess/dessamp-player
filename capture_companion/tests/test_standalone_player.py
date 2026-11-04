from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dessmetal_capture.domain import DomainError
from dessmetal_capture.manifests import import_model_artifact
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import write_pcm24_mono
from dessmetal_player.api import create_app
from dessmetal_player.artifact import PlayerArtifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class StandalonePlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = CaptureStore(self.root / "trainer-data")
        self.project = self.store.create_project(
            name="Conditioned Gain model",
            amp_name="DessBlock Green",
            controls=[
                {
                    "id": "gain",
                    "label": "Gain",
                    "minimum": 0.2,
                    "maximum": 0.8,
                    "step": 0.1,
                    "display_minimum": 2.0,
                    "display_maximum": 8.0,
                }
            ],
        )
        source_fixture = (
            REPOSITORY_ROOT / "NeuralAmpModelerCore/example_models/wavenet.nam"
        )
        conditioned_fixture = self.root / "conditioned-test-fixture.nam"
        conditioned_model = json.loads(source_fixture.read_text(encoding="utf-8"))
        conditioned_model["config"]["global_condition_size"] = 1
        conditioned_fixture.write_text(
            json.dumps(conditioned_model),
            encoding="utf-8",
        )
        artifact = import_model_artifact(
            self.store,
            project_id=self.project["id"],
            source_model_path=conditioned_fixture,
            name="Conditioned test fixture",
            provenance={"training_backend": "test-fixture"},
            loader_path=None,
        )
        self.manifest_path = Path(artifact["manifest_path"])
        self.package_path = Path(artifact["package_path"])
        self.renderer_path = self.root / "fake-renderer"
        self.renderer_path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, shutil, sys\n"
            "shutil.copyfile(sys.argv[2], sys.argv[3])\n"
            "print(json.dumps({'status':'rendered','parameters':[float(sys.argv[5])]}))\n",
            encoding="utf-8",
        )
        self.renderer_path.chmod(0o700)
        self.input_path = self.root / "input.wav"
        write_pcm24_mono(
            self.input_path,
            np.sin(np.linspace(0.0, 20.0, 4_800, dtype=np.float32)) * 0.1,
            48_000,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_manifest_drives_dynamic_controls_and_render(self) -> None:
        artifact = PlayerArtifact(self.manifest_path, self.renderer_path)
        description = artifact.describe()
        self.assertEqual(description["model"]["global_condition_size"], 1)
        self.assertEqual(description["controls"][0]["id"], "gain")
        self.assertEqual(description["controls"][0]["model_parameter_index"], 0)
        self.assertEqual(description["presentation"]["head_shell"], "studio-black-v1")
        output = self.root / "output.wav"
        evidence = artifact.render(
            input_path=self.input_path,
            output_path=output,
            settings={"gain": 0.7},
        )
        self.assertEqual(evidence["settings"], {"gain": 0.7})
        self.assertEqual(evidence["renderer"]["parameters"], [0.7])
        self.assertTrue(output.is_file())
        with self.assertRaises(DomainError):
            artifact.render(
                input_path=self.input_path,
                output_path=self.root / "rejected.wav",
                settings={"gain": 1.2},
            )

    def test_manifest_accepts_only_safe_compiled_shell_tokens(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["presentation"] = {
            "head_shell": "dual-red-modern-v1",
            "product_family": "redline",
            "product_variant": "us",
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        artifact = PlayerArtifact(self.manifest_path, self.renderer_path)
        presentation = artifact.describe()["presentation"]
        self.assertEqual(presentation["head_shell"], "dual-red-modern-v1")
        self.assertEqual(presentation["product_family"], "redline")
        self.assertEqual(presentation["product_variant"], "us")

        manifest["presentation"] = {"head_shell": "../untrusted.png"}
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(DomainError, "head_shell token"):
            PlayerArtifact(self.manifest_path, self.renderer_path)

        manifest["presentation"] = {
            "head_shell": "dual-red-modern-v1",
            "product_family": "../redline",
            "product_variant": "us",
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(DomainError, "product_family token"):
            PlayerArtifact(self.manifest_path, self.renderer_path)

        manifest["presentation"] = {
            "head_shell": "dual-red-modern-v1",
            "product_variant": "brit",
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(DomainError, "requires product_family"):
            PlayerArtifact(self.manifest_path, self.renderer_path)

    def test_self_contained_dess_package_survives_external_model_removal(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        (self.manifest_path.parent / manifest["model"]["path"]).unlink()
        artifact = PlayerArtifact(self.package_path, self.renderer_path)
        self.assertEqual(artifact.describe()["artifact_format"], "dess")
        self.assertEqual(artifact.describe()["controls"][0]["id"], "gain")
        output = self.root / "dess-output.wav"
        artifact.render(input_path=self.input_path, output_path=output, settings={"gain": 0.6})
        self.assertTrue(output.is_file())

    def test_dess_package_detects_payload_corruption(self) -> None:
        damaged = self.root / "damaged.dess"
        payload = bytearray(self.package_path.read_bytes())
        payload[-1] ^= 0x01
        damaged.write_bytes(payload)
        with self.assertRaisesRegex(DomainError, "model hash"):
            PlayerArtifact(damaged, self.renderer_path)

    def test_player_api_reaches_audio_artifact(self) -> None:
        app = create_app(
            manifest_path=self.manifest_path,
            renderer_path=self.renderer_path,
            data_dir=self.root / "player-data",
            default_input_path=self.input_path,
            access_token="test-token",
        )
        session_status, session_headers, session_body = _asgi_request(
            app, "GET", "/api/session", headers={"x-player-token": "test-token"}
        )
        self.assertEqual(session_status, 200, session_body)
        session = json.loads(session_body)
        input_id = session["input"]["id"]
        render_payload = json.dumps(
            {"input_id": input_id, "settings": {"gain": 0.3}, "quality": None}
        ).encode("utf-8")
        rendered_status, rendered_headers, rendered_body = _asgi_request(
            app,
            "POST",
            "/api/render",
            headers={"x-player-token": "test-token", "content-type": "application/json"},
            body=render_payload,
        )
        self.assertEqual(rendered_status, 200, rendered_body)
        rendered = json.loads(rendered_body)
        audio_status, audio_headers, audio_body = _asgi_request(app, "GET", rendered["audio_url"])
        self.assertEqual(audio_status, 200)
        self.assertEqual(audio_body[:4], b"RIFF")

    def test_player_api_rejects_dns_rebinding_host_and_foreign_origin(self) -> None:
        app = create_app(
            manifest_path=self.manifest_path,
            renderer_path=self.renderer_path,
            data_dir=self.root / "player-security-data",
            access_token="test-token",
        )

        hostile_status, _, hostile_body = _asgi_request(
            app, "GET", "/", headers={"host": "attacker.example:8775"}
        )
        self.assertEqual(hostile_status, 400, hostile_body)
        self.assertNotIn(b"test-token", hostile_body)

        wrong_port_status, _, wrong_port_body = _asgi_request(
            app, "GET", "/", headers={"host": "127.0.0.1:8776"}
        )
        self.assertEqual(wrong_port_status, 400, wrong_port_body)
        self.assertNotIn(b"test-token", wrong_port_body)

        local_root_status, _, local_root_body = _asgi_request(
            app, "GET", "/", headers={"host": "localhost:8775"}
        )
        self.assertEqual(local_root_status, 200, local_root_body)
        self.assertIn(b"test-token", local_root_body)

        origin_status, _, origin_body = _asgi_request(
            app,
            "POST",
            "/api/render",
            headers={
                "host": "127.0.0.1:8775",
                "origin": "https://attacker.example",
                "x-player-token": "test-token",
                "content-type": "application/json",
            },
            body=b"{}",
        )
        self.assertEqual(origin_status, 403, origin_body)

        wrong_origin_port_status, _, wrong_origin_port_body = _asgi_request(
            app,
            "POST",
            "/api/render",
            headers={
                "host": "127.0.0.1:8775",
                "origin": "http://127.0.0.1:8776",
                "x-player-token": "test-token",
                "content-type": "application/json",
            },
            body=b"{}",
        )
        self.assertEqual(wrong_origin_port_status, 403, wrong_origin_port_body)

        for prefixed_path in ("/api/session", "/prefix/api/session"):
            with self.subTest(prefixed_path=prefixed_path):
                unauthenticated_status, _, unauthenticated_body = _asgi_request(
                    app,
                    "GET",
                    prefixed_path,
                    root_path="/prefix",
                )
                self.assertEqual(
                    unauthenticated_status,
                    403,
                    unauthenticated_body,
                )

                prefixed_foreign_status, _, prefixed_foreign_body = _asgi_request(
                    app,
                    "POST",
                    prefixed_path.replace("session", "render"),
                    headers={
                        "origin": "https://attacker.example",
                        "x-player-token": "test-token",
                        "content-type": "application/json",
                    },
                    body=b"{}",
                    root_path="/prefix",
                )
                self.assertEqual(
                    prefixed_foreign_status,
                    403,
                    prefixed_foreign_body,
                )

                authenticated_status, _, authenticated_body = _asgi_request(
                    app,
                    "GET",
                    prefixed_path,
                    headers={"x-player-token": "test-token"},
                    root_path="/prefix",
                )
                self.assertEqual(authenticated_status, 200, authenticated_body)

        trusted_status, _, trusted_body = _asgi_request(
            app,
            "GET",
            "/api/session",
            headers={
                "host": "127.0.0.1:8775",
                "origin": "http://127.0.0.1:8775",
                "x-player-token": "test-token",
            },
        )
        self.assertEqual(trusted_status, 200, trusted_body)

    def test_model_hash_is_enforced_before_player_start(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        model_path = self.manifest_path.parent / manifest["model"]["path"]
        with model_path.open("ab") as model:
            model.write(b"\n")
        with self.assertRaisesRegex(DomainError, "SHA-256"):
            PlayerArtifact(self.manifest_path, self.renderer_path)

    def test_player_knobs_interpolate_continuously_between_capture_steps(self) -> None:
        javascript = (
            REPOSITORY_ROOT / "capture_companion/src/dessmetal_player/static/app.js"
        ).read_text(encoding="utf-8")
        self.assertIn('step="any"', javascript)
        self.assertIn('data-capture-step="${escapeHtml(control.step)}"', javascript)
        self.assertIn("state.renderPending = true", javascript)
        self.assertIn("if (state.renderPending)", javascript)


if __name__ == "__main__":
    unittest.main()


def _asgi_request(
    app,
    method: str,
    path: str,
    *,
    headers=None,
    body: bytes = b"",
    root_path: str = "",
):
    async def run():
        messages = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if request_sent:
                return {"type": "http.disconnect"}
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(message)

        request_headers = {"host": "127.0.0.1:8775"}
        request_headers.update(headers or {})
        encoded_headers = [
            (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
            for key, value in request_headers.items()
        ]
        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": encoded_headers,
                "client": ("127.0.0.1", 50000),
                "server": ("127.0.0.1", 8775),
                "root_path": root_path,
            },
            receive,
            send,
        )
        start = next(message for message in messages if message["type"] == "http.response.start")
        response_headers = {
            key.decode("latin-1"): value.decode("latin-1") for key, value in start["headers"]
        }
        response_body = b"".join(
            message.get("body", b"") for message in messages if message["type"] == "http.response.body"
        )
        return start["status"], response_headers, response_body

    return asyncio.run(run())
