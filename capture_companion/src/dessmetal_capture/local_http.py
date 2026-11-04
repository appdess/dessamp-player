"""Shared browser-boundary checks for DessAmp's loopback-only web tools."""

from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _well_formed_asgi_path(
    value: object,
    *,
    allow_empty: bool,
    allow_trailing_slash: bool,
) -> bool:
    if not isinstance(value, str):
        return False
    if allow_empty and not value:
        return True
    if not value.startswith("/"):
        return False
    if (
        "//" in value
        or "\\" in value
        or "?" in value
        or "#" in value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        or any(segment in {".", ".."} for segment in value.split("/"))
    ):
        return False
    return allow_trailing_slash or value == "/" or not value.endswith("/")


def normalized_local_request_path(request: Request) -> str:
    """Return the application-relative route path for both common ASGI conventions."""
    path = request.scope.get("path")
    root_path = request.scope.get("root_path", "")
    if not _well_formed_asgi_path(
        path,
        allow_empty=False,
        allow_trailing_slash=True,
    ) or not _well_formed_asgi_path(
        root_path,
        allow_empty=True,
        allow_trailing_slash=False,
    ):
        raise ValueError("Malformed ASGI request path")

    assert isinstance(path, str)
    assert isinstance(root_path, str)
    if root_path and root_path != "/":
        if path == root_path:
            return "/"
        if path.startswith(root_path + "/"):
            return path[len(root_path) :]
    return path


def api_path_requires_token(path: str) -> bool:
    return (path == "/api" or path.startswith("/api/")) and path != "/api/health"


def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


def _trusted_authority(
    value: str,
    *,
    expected_scheme: str,
    expected_port: int,
) -> bool:
    if (
        not value
        or value != value.strip()
        or any(character in value for character in ("/", "?", "#", "@"))
    ):
        return False
    try:
        parsed = urlsplit(f"//{value}")
        parsed_port = parsed.port
    except ValueError:
        return False
    effective_port = (
        parsed_port
        if parsed_port is not None
        else _DEFAULT_PORTS.get(expected_scheme)
    )
    return (
        parsed.hostname in _LOOPBACK_HOSTS
        and effective_port == expected_port
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def _trusted_origin(
    value: str,
    *,
    expected_scheme: str,
    expected_port: int,
) -> bool:
    if not value or value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
        parsed_port = parsed.port
    except ValueError:
        return False
    effective_port = (
        parsed_port
        if parsed_port is not None
        else _DEFAULT_PORTS.get(parsed.scheme)
    )
    return (
        parsed.scheme == expected_scheme
        and parsed.hostname in _LOOPBACK_HOSTS
        and effective_port == expected_port
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def reject_untrusted_browser_boundary(request: Request) -> JSONResponse | None:
    """Reject DNS-rebinding hosts and cross-origin browser requests."""
    server = request.scope.get("server")
    expected_scheme = str(request.scope.get("scheme", "")).lower()
    if (
        not isinstance(server, (tuple, list))
        or len(server) != 2
        or isinstance(server[1], bool)
        or not isinstance(server[1], int)
        or not 1 <= server[1] <= 65535
        or expected_scheme not in _DEFAULT_PORTS
    ):
        return JSONResponse(status_code=400, content={"detail": "Invalid local server authority"})

    host_values = request.headers.getlist("host")
    if len(host_values) != 1 or not _trusted_authority(
        host_values[0],
        expected_scheme=expected_scheme,
        expected_port=server[1],
    ):
        return JSONResponse(status_code=400, content={"detail": "Invalid local Host"})

    try:
        request_path = normalized_local_request_path(request)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid local request path"},
        )

    if (
        _is_api_path(request_path)
        and request.method.upper() in _STATE_CHANGING_METHODS
    ):
        origin_values = request.headers.getlist("origin")
        if len(origin_values) > 1 or (
            origin_values
            and not _trusted_origin(
                origin_values[0],
                expected_scheme=expected_scheme,
                expected_port=server[1],
            )
        ):
            return JSONResponse(status_code=403, content={"detail": "Invalid local Origin"})
    return None
