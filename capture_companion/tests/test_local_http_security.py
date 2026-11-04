from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from starlette.requests import Request

from dessmetal_capture.api import create_app
from dessmetal_capture.local_http import (
    normalized_local_request_path,
    reject_untrusted_browser_boundary,
)


def _request(
    *,
    host: str | None,
    origin: str | None = None,
    method: str = "POST",
    path: str = "/api/test",
    scheme: str = "http",
    server_port: int = 8765,
    root_path: str = "",
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> Request:
    headers = []
    if host is not None:
        headers.append((b"host", host.encode("ascii")))
    if origin is not None:
        headers.append((b"origin", origin.encode("ascii")))
    headers.extend(
        (name.lower().encode("ascii"), value.encode("ascii"))
        for name, value in extra_headers
    )
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": scheme,
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", server_port),
            "root_path": root_path,
        }
    )


def _asgi_request(
    app,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
    server_port: int = 8765,
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

        request_headers = {"host": f"127.0.0.1:{server_port}"}
        request_headers.update(headers or {})
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
                "headers": [
                    (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
                    for key, value in request_headers.items()
                ],
                "client": ("127.0.0.1", 50000),
                "server": ("127.0.0.1", server_port),
                "root_path": root_path,
            },
            receive,
            send,
        )
        start = next(message for message in messages if message["type"] == "http.response.start")
        response_body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        return start["status"], response_body

    return asyncio.run(run())


class LocalHttpSecurityTests(unittest.TestCase):
    def test_accepts_supported_loopback_authorities(self) -> None:
        for host, origin in (
            ("127.0.0.1:8765", "http://127.0.0.1:8765"),
            ("localhost:8765", "http://localhost:8765"),
            ("[::1]:8765", "http://[::1]:8765"),
        ):
            with self.subTest(host=host):
                self.assertIsNone(
                    reject_untrusted_browser_boundary(
                        _request(host=host, origin=origin)
                    )
                )

        self.assertIsNone(
            reject_untrusted_browser_boundary(
                _request(
                    host="127.0.0.1",
                    origin="http://127.0.0.1",
                    server_port=80,
                )
            )
        )

    def test_rejects_loopback_authority_on_the_wrong_selected_port(self) -> None:
        for host in (
            "127.0.0.1",
            "127.0.0.1:8766",
            "localhost:8766",
            "[::1]:8766",
        ):
            with self.subTest(host=host):
                response = reject_untrusted_browser_boundary(_request(host=host))
                self.assertIsNotNone(response)
                self.assertEqual(response.status_code, 400)

    def test_rejects_host_header_aliases_and_malformed_authorities(self) -> None:
        for host in (
            "attacker.example",
            "127.0.0.1.attacker.example",
            "127.0.0.1@attacker.example",
            "127.0.0.1/path",
            " 127.0.0.1:8765",
            "",
        ):
            with self.subTest(host=host):
                response = reject_untrusted_browser_boundary(_request(host=host))
                self.assertIsNotNone(response)
                self.assertEqual(response.status_code, 400)

        duplicate = reject_untrusted_browser_boundary(
            _request(
                host="127.0.0.1:8765",
                extra_headers=(("host", "localhost:8765"),),
            )
        )
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate.status_code, 400)

    def test_rejects_foreign_and_opaque_origins(self) -> None:
        for origin in (
            "https://attacker.example",
            "null",
            "http://127.0.0.1@attacker.example",
            "http://127.0.0.1:8766",
            "https://127.0.0.1:8765",
            "http://127.0.0.1:8765/",
        ):
            with self.subTest(origin=origin):
                response = reject_untrusted_browser_boundary(
                    _request(host="127.0.0.1:8765", origin=origin)
                )
                self.assertIsNotNone(response)
                self.assertEqual(response.status_code, 403)

        duplicate = reject_untrusted_browser_boundary(
            _request(
                host="127.0.0.1:8765",
                origin="http://127.0.0.1:8765",
                extra_headers=(("origin", "http://localhost:8765"),),
            )
        )
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate.status_code, 403)

    def test_origin_gate_is_limited_to_state_changing_api_requests(self) -> None:
        for method, path in (("GET", "/api/test"), ("POST", "/")):
            with self.subTest(method=method, path=path):
                self.assertIsNone(
                    reject_untrusted_browser_boundary(
                        _request(
                            host="127.0.0.1:8765",
                            origin="https://attacker.example",
                            method=method,
                            path=path,
                        )
                    )
                )
        self.assertIsNone(
            reject_untrusted_browser_boundary(
                _request(host="127.0.0.1:8765", origin=None)
            )
        )

    def test_normalizes_both_asgi_root_path_conventions(self) -> None:
        for path in ("/api/test", "/prefix/api/test"):
            with self.subTest(path=path):
                request = _request(
                    host="127.0.0.1:8765",
                    origin="https://attacker.example",
                    path=path,
                    root_path="/prefix",
                )
                self.assertEqual(normalized_local_request_path(request), "/api/test")
                response = reject_untrusted_browser_boundary(request)
                self.assertIsNotNone(response)
                self.assertEqual(response.status_code, 403)

    def test_rejects_malformed_asgi_paths(self) -> None:
        malformed = (
            ("api/test", ""),
            ("//api/test", ""),
            ("/api/../test", ""),
            ("/api/test", "prefix"),
            ("/prefix/api/test", "/prefix/"),
        )
        for path, root_path in malformed:
            with self.subTest(path=path, root_path=root_path):
                response = reject_untrusted_browser_boundary(
                    _request(
                        host="127.0.0.1:8765",
                        path=path,
                        root_path=root_path,
                    )
                )
                self.assertIsNotNone(response)
                self.assertEqual(response.status_code, 400)

    def test_capture_api_blocks_rebinding_before_token_and_preserves_local_ui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(
                data_dir=Path(directory) / "capture-data",
                access_token="capture-test-token",
            )

            hostile_status, hostile_body = _asgi_request(
                app,
                "GET",
                "/",
                headers={"host": "attacker.example:8765"},
            )
            self.assertEqual(hostile_status, 400, hostile_body)
            self.assertNotIn(b"capture-test-token", hostile_body)

            wrong_port_status, wrong_port_body = _asgi_request(
                app,
                "GET",
                "/",
                headers={"host": "127.0.0.1:8766"},
            )
            self.assertEqual(wrong_port_status, 400, wrong_port_body)
            self.assertNotIn(b"capture-test-token", wrong_port_body)

            local_root_status, local_root_body = _asgi_request(
                app,
                "GET",
                "/",
                headers={"host": "localhost:8765"},
            )
            self.assertEqual(local_root_status, 200, local_root_body)
            self.assertIn(b"capture-test-token", local_root_body)

            foreign_status, foreign_body = _asgi_request(
                app,
                "POST",
                "/api/projects",
                headers={
                    "origin": "https://attacker.example",
                    "x-capture-token": "capture-test-token",
                    "content-type": "application/json",
                },
                body=b"{}",
            )
            self.assertEqual(foreign_status, 403, foreign_body)

            for prefixed_path in ("/api/config", "/prefix/api/config"):
                with self.subTest(prefixed_path=prefixed_path):
                    unauthenticated_status, unauthenticated_body = _asgi_request(
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

                    prefixed_foreign_status, prefixed_foreign_body = _asgi_request(
                        app,
                        "POST",
                        prefixed_path.replace("config", "projects"),
                        headers={
                            "origin": "https://attacker.example",
                            "x-capture-token": "capture-test-token",
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

                    authenticated_status, authenticated_body = _asgi_request(
                        app,
                        "GET",
                        prefixed_path,
                        headers={"x-capture-token": "capture-test-token"},
                        root_path="/prefix",
                    )
                    self.assertEqual(authenticated_status, 200, authenticated_body)

            project_payload = json.dumps(
                {
                    "name": "Local project",
                    "amp_name": "Local amp",
                    "controls": [
                        {
                            "id": "gain",
                            "label": "Gain",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "step": 0.1,
                        }
                    ],
                }
            ).encode("utf-8")
            local_status, local_body = _asgi_request(
                app,
                "POST",
                "/api/projects",
                headers={
                    "host": "localhost:8765",
                    "origin": "http://localhost:8765",
                    "x-capture-token": "capture-test-token",
                    "content-type": "application/json",
                },
                body=project_payload,
            )
            self.assertEqual(local_status, 200, local_body)


if __name__ == "__main__":
    unittest.main()
