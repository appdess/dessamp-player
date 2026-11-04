"""Time-limited, object-scoped GCS access for non-Google training workers."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit


CAPABILITY_ENVIRONMENT_VARIABLE = "DESSAMP_GCS_CAPABILITIES_B64"
MAX_CAPABILITY_BYTES = 512 * 1024
MAX_CAPABILITY_TRANSPORT_BYTES = 512 * 1024
MAX_CAPABILITY_OBJECTS = 4096
MAX_CAPABILITY_LIFETIME = timedelta(days=7)
_CAPABILITY_ZLIB_MAGIC = b"DESSAMP_GCS_CAPABILITIES_ZLIB_V1\0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]\Z")
_WRITE_HEADER_NAMES = frozenset(
    {"content-type", "x-goog-content-length-range"}
)
_SIGNED_WRITE_HEADER_NAMES = frozenset(
    {"content-type", "host", "x-goog-content-length-range"}
)


class CapabilityError(ValueError):
    pass


class SignedUrlStorageClient:
    """Storage adapter backed by exact V4 URLs with server-bounded writes."""

    def __init__(self, payload: Mapping[str, Any]):
        if payload.get("schema_version") != 2:
            raise CapabilityError("Unsupported GCS capability schema")
        bucket = payload.get("bucket")
        if not isinstance(bucket, str) or _BUCKET_RE.fullmatch(bucket) is None:
            raise CapabilityError("GCS capability bucket is invalid")
        self.bucket_name = bucket
        self.expires_at = _parse_expiration(payload.get("expires_at"))
        self._read = _entry_map(
            payload.get("read_objects"), operation="read", bucket=bucket
        )
        self._write = _entry_map(
            payload.get("write_objects"), operation="write", bucket=bucket
        )
        if len(self._read) + len(self._write) > MAX_CAPABILITY_OBJECTS:
            raise CapabilityError("GCS capability object count exceeds its limit")
        if not self._read or not self._write:
            raise CapabilityError("GCS capability bundle requires read and write scopes")

    @classmethod
    def from_encoded(cls, encoded: str) -> "SignedUrlStorageClient":
        try:
            transport = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise CapabilityError("GCS capability environment is not valid base64") from error
        if not 0 < len(transport) <= MAX_CAPABILITY_TRANSPORT_BYTES:
            raise CapabilityError("GCS capability bundle is empty or oversized")
        raw = _decode_capability_transport(transport)
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CapabilityError("GCS capability bundle is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise CapabilityError("GCS capability root must be an object")
        return cls(payload)

    @classmethod
    def from_environment(cls) -> "SignedUrlStorageClient | None":
        encoded = os.environ.pop(CAPABILITY_ENVIRONMENT_VARIABLE, "").strip()
        return cls.from_encoded(encoded) if encoded else None

    def bucket(self, bucket_name: str) -> "SignedUrlBucket":
        if bucket_name != self.bucket_name:
            raise CapabilityError("Requested GCS bucket is outside the capability bundle")
        return SignedUrlBucket(self)

    def close(self) -> None:
        self._read.clear()
        self._write.clear()


class SignedUrlBucket:
    def __init__(self, client: SignedUrlStorageClient):
        self.client = client

    def blob(self, name: str) -> "SignedUrlBlob":
        _validate_object_name(name)
        read = self.client._read.get(name)
        write = self.client._write.get(name)
        if read is None and write is None:
            raise CapabilityError(f"GCS object is outside the capability allowlist: {name!r}")
        return SignedUrlBlob(name=name, read=read, write=write)


class SignedUrlBlob:
    def __init__(
        self,
        *,
        name: str,
        read: Mapping[str, Any] | None,
        write: Mapping[str, Any] | None,
    ):
        self.name = name
        self._read = read
        self._write = write
        self.size = read.get("size_bytes") if read is not None else None
        self.declared_max_bytes = (
            read.get("max_bytes") if read is not None else write.get("max_bytes")
        )

    def reload(self) -> None:
        if self._read is None:
            raise CapabilityError(f"GCS object has no read capability: {self.name!r}")

    def download_to_filename(self, filename: str) -> None:
        if not self._download(Path(filename), optional=False):
            raise FileNotFoundError(self.name)

    def download_if_present(self, filename: str) -> bool:
        if self._read is None or not self._read.get("optional", False):
            raise CapabilityError(f"GCS object is not an optional read: {self.name!r}")
        return self._download(Path(filename), optional=True)

    def upload_from_filename(self, filename: str) -> None:
        if self._write is None:
            raise CapabilityError(f"GCS object has no write capability: {self.name!r}")
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            raise CapabilityError(f"Upload source is not a regular file: {path}")
        size = path.stat().st_size
        maximum = self._write["max_bytes"]
        if not 0 < size <= maximum:
            raise CapabilityError(f"Upload source exceeds its object capability: {self.name!r}")
        parsed = _validated_signed_url(self._write["url"])
        headers = self._write["headers"]
        connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=300)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        try:
            connection.putrequest("PUT", target)
            connection.putheader("Content-Type", headers["content-type"])
            connection.putheader(
                "x-goog-content-length-range",
                headers["x-goog-content-length-range"],
            )
            connection.putheader("Content-Length", str(size))
            connection.putheader("User-Agent", "DessAmp-GCS-Capability/1.0")
            connection.endheaders()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    connection.send(block)
            response = connection.getresponse()
            response.read(2048)
            if not 200 <= response.status < 300:
                raise CapabilityError(
                    f"Signed GCS PUT failed for {self.name!r} with HTTP {response.status}"
                )
        finally:
            connection.close()

    def _download(self, destination: Path, *, optional: bool) -> bool:
        if self._read is None:
            raise CapabilityError(f"GCS object has no read capability: {self.name!r}")
        parsed = _validated_signed_url(self._read["url"])
        connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=300)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        try:
            connection.request(
                "GET",
                target,
                headers={"User-Agent": "DessAmp-GCS-Capability/1.0"},
            )
            response = connection.getresponse()
            if response.status == 404 and optional:
                response.read(2048)
                return False
            if not 200 <= response.status < 300:
                response.read(2048)
                raise CapabilityError(
                    f"Signed GCS GET failed for {self.name!r} with HTTP {response.status}"
                )
            maximum = self._read["max_bytes"]
            digest = hashlib.sha256()
            total = 0
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > maximum:
                        raise CapabilityError(
                            f"Signed GCS GET exceeded its object capability: {self.name!r}"
                        )
                    output.write(block)
                    digest.update(block)
            expected_size = self._read.get("size_bytes")
            if expected_size is not None and total != expected_size:
                raise CapabilityError(f"Signed GCS GET size mismatch: {self.name!r}")
            expected_sha256 = self._read.get("sha256")
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise CapabilityError(f"Signed GCS GET hash mismatch: {self.name!r}")
            return True
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            connection.close()


def encode_capability_bundle(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if not 0 < len(raw) <= MAX_CAPABILITY_BYTES:
        raise CapabilityError("GCS capability bundle is empty or oversized")
    # Validate before returning a bearer capability to a provider control plane.
    SignedUrlStorageClient(payload).close()
    transport = _CAPABILITY_ZLIB_MAGIC + zlib.compress(raw, level=9)
    if len(transport) > MAX_CAPABILITY_TRANSPORT_BYTES:
        raise CapabilityError("Compressed GCS capability bundle is oversized")
    return base64.b64encode(transport).decode("ascii")


def _decode_capability_transport(transport: bytes) -> bytes:
    """Decode the bounded transport envelope while accepting legacy raw JSON."""

    if not transport.startswith(_CAPABILITY_ZLIB_MAGIC):
        if len(transport) > MAX_CAPABILITY_BYTES:
            raise CapabilityError("GCS capability bundle is oversized")
        return transport

    compressed = transport[len(_CAPABILITY_ZLIB_MAGIC) :]
    if not compressed:
        raise CapabilityError("Compressed GCS capability bundle is empty")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(compressed, MAX_CAPABILITY_BYTES + 1)
        if decompressor.unconsumed_tail or len(raw) > MAX_CAPABILITY_BYTES:
            raise CapabilityError("GCS capability bundle is oversized")
        remaining = MAX_CAPABILITY_BYTES + 1 - len(raw)
        raw += decompressor.flush(remaining)
    except zlib.error as error:
        raise CapabilityError("GCS capability bundle compression is invalid") from error
    if (
        not decompressor.eof
        or decompressor.unused_data
        or len(raw) > MAX_CAPABILITY_BYTES
    ):
        raise CapabilityError("GCS capability bundle compression is invalid or oversized")
    return raw


def _entry_map(
    value: Any, *, operation: str, bucket: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise CapabilityError(f"GCS {operation} capabilities must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise CapabilityError(f"GCS {operation} capability {index} is malformed")
        name = entry.get("name")
        url = entry.get("url")
        max_bytes = entry.get("max_bytes")
        if not isinstance(name, str):
            raise CapabilityError(f"GCS {operation} capability {index} has no object name")
        _validate_object_name(name)
        if name in result:
            raise CapabilityError(f"Duplicate GCS {operation} capability: {name!r}")
        if not isinstance(url, str):
            raise CapabilityError(f"GCS {operation} capability {name!r} has no signed URL")
        parsed_url = _validated_signed_url(url, bucket=bucket, object_name=name)
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 0 < max_bytes <= 512 * 1024 * 1024
        ):
            raise CapabilityError(f"GCS {operation} capability {name!r} has an invalid byte limit")
        size = entry.get("size_bytes")
        if size is not None and (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= max_bytes
        ):
            raise CapabilityError(f"GCS {operation} capability {name!r} has an invalid size")
        sha256 = entry.get("sha256")
        if sha256 is not None and (not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None):
            raise CapabilityError(f"GCS {operation} capability {name!r} has an invalid SHA-256")
        optional = entry.get("optional", False)
        if not isinstance(optional, bool) or (operation != "read" and optional):
            raise CapabilityError(f"GCS {operation} capability {name!r} has invalid optional state")
        if operation == "write":
            _validate_write_headers(entry, parsed_url, max_bytes=max_bytes, name=name)
        elif "headers" in entry:
            raise CapabilityError(
                f"GCS read capability {name!r} must not declare write headers"
            )
        result[name] = dict(entry)
    return result


def _validate_write_headers(
    entry: Mapping[str, Any], parsed_url, *, max_bytes: int, name: str
) -> None:
    headers = entry.get("headers")
    if not isinstance(headers, Mapping) or set(headers) != _WRITE_HEADER_NAMES:
        raise CapabilityError(
            f"GCS write capability {name!r} lacks exact server-enforced headers"
        )
    if headers.get("content-type") != "application/octet-stream":
        raise CapabilityError(
            f"GCS write capability {name!r} has an invalid signed content type"
        )
    expected_range = f"1,{max_bytes}"
    if headers.get("x-goog-content-length-range") != expected_range:
        raise CapabilityError(
            f"GCS write capability {name!r} has an invalid signed byte range"
        )
    signed_values = parse_qs(
        parsed_url.query, keep_blank_values=True, strict_parsing=False
    ).get("X-Goog-SignedHeaders")
    if not isinstance(signed_values, list) or len(signed_values) != 1:
        raise CapabilityError(
            f"GCS write capability {name!r} has no exact signed-header declaration"
        )
    signed_names = signed_values[0].split(";")
    if (
        any(not value or value != value.lower() for value in signed_names)
        or len(signed_names) != len(set(signed_names))
        or frozenset(signed_names) != _SIGNED_WRITE_HEADER_NAMES
    ):
        raise CapabilityError(
            f"GCS write capability {name!r} does not sign its exact byte-range header"
        )


def _parse_expiration(value: Any) -> datetime:
    if not isinstance(value, str):
        raise CapabilityError("GCS capability expiration is missing")
    try:
        expiration = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CapabilityError("GCS capability expiration is invalid") from error
    if expiration.tzinfo is None:
        raise CapabilityError("GCS capability expiration must include a timezone")
    now = datetime.now(timezone.utc)
    if expiration <= now or expiration - now > MAX_CAPABILITY_LIFETIME:
        raise CapabilityError("GCS capability expiration is outside the allowed lifetime")
    return expiration


def _validate_object_name(name: str) -> None:
    if (
        not name
        or len(name.encode("utf-8")) > 1024
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise CapabilityError(f"Unsafe GCS capability object name: {name!r}")


def _validated_signed_url(
    value: str, *, bucket: str | None = None, object_name: str | None = None
):
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or (
            hostname != "storage.googleapis.com"
            and not hostname.endswith(".storage.googleapis.com")
        )
        or not parsed.path.startswith("/")
        or not parsed.query
        or "X-Goog-Signature=" not in parsed.query
        or parsed.fragment
    ):
        raise CapabilityError("GCS capability URL is not a valid signed HTTPS storage URL")
    if bucket is not None and object_name is not None:
        decoded_path = unquote(parsed.path)
        path_style = hostname == "storage.googleapis.com" and decoded_path == (
            f"/{bucket}/{object_name}"
        )
        virtual_hosted = hostname == f"{bucket}.storage.googleapis.com" and decoded_path == (
            f"/{object_name}"
        )
        if not path_style and not virtual_hosted:
            raise CapabilityError(
                "GCS capability URL does not match its declared bucket and object"
            )
    return parsed
