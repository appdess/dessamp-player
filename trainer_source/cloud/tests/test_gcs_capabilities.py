import base64
import hashlib
import io
import json
import os
import tempfile
import unittest
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from trainer_source.cloud import gcs_capabilities as MODULE


def signed_url(
    bucket: str,
    name: str,
    *,
    host: str = "storage.googleapis.com",
    write: bool = False,
) -> str:
    path = f"/{bucket}/{quote(name, safe='/')}" if host == "storage.googleapis.com" else f"/{quote(name, safe='/')}"
    signed_headers = (
        "content-type%3Bhost%3Bx-goog-content-length-range" if write else "host"
    )
    return (
        f"https://{host}{path}?X-Goog-Algorithm=GOOG4-RSA-SHA256"
        "&X-Goog-Credential=test&X-Goog-Date=20260825T000000Z"
        f"&X-Goog-Expires=3600&X-Goog-SignedHeaders={signed_headers}"
        "&X-Goog-Signature=abc123"
    )


def valid_payload(*, read_payload: bytes = b"trusted") -> dict:
    bucket = "dessamp-evidence"
    read_name = "jobs/run/configs/data.json"
    write_name = "jobs/run/results/full-seed-81/resume.ckpt"
    return {
        "schema_version": 2,
        "bucket": bucket,
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "read_objects": [
            {
                "name": read_name,
                "url": signed_url(bucket, read_name),
                "max_bytes": 1024,
                "size_bytes": len(read_payload),
                "sha256": hashlib.sha256(read_payload).hexdigest(),
            }
        ],
        "write_objects": [
            {
                "name": write_name,
                "url": signed_url(bucket, write_name, write=True),
                "max_bytes": 1024,
                "headers": {
                    "content-type": "application/octet-stream",
                    "x-goog-content-length-range": "1,1024",
                },
            }
        ],
    }


class SignedUrlStorageClientTests(unittest.TestCase):
    def test_compressed_transport_round_trips_and_legacy_json_remains_readable(self):
        payload = valid_payload()
        encoded = MODULE.encode_capability_bundle(payload)
        transport = base64.b64decode(encoded, validate=True)
        self.assertTrue(transport.startswith(MODULE._CAPABILITY_ZLIB_MAGIC))

        compressed_client = MODULE.SignedUrlStorageClient.from_encoded(encoded)
        self.assertEqual(compressed_client.bucket_name, payload["bucket"])
        compressed_client.close()

        legacy = base64.b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).decode("ascii")
        legacy_client = MODULE.SignedUrlStorageClient.from_encoded(legacy)
        self.assertEqual(legacy_client.bucket_name, payload["bucket"])
        legacy_client.close()

    def test_compressed_transport_rejects_expansion_bombs_and_trailing_data(self):
        oversized = b"x" * (MODULE.MAX_CAPABILITY_BYTES + 1)
        bomb = base64.b64encode(
            MODULE._CAPABILITY_ZLIB_MAGIC + zlib.compress(oversized, level=9)
        ).decode("ascii")
        with self.assertRaises(MODULE.CapabilityError):
            MODULE.SignedUrlStorageClient.from_encoded(bomb)

        payload = json.dumps(
            valid_payload(), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        trailing = base64.b64encode(
            MODULE._CAPABILITY_ZLIB_MAGIC + zlib.compress(payload) + b"trailing"
        ).decode("ascii")
        with self.assertRaises(MODULE.CapabilityError):
            MODULE.SignedUrlStorageClient.from_encoded(trailing)

    def test_environment_is_consumed_and_exact_bucket_object_scope_is_enforced(self):
        payload = valid_payload()
        encoded = MODULE.encode_capability_bundle(payload)
        with mock.patch.dict(
            os.environ,
            {MODULE.CAPABILITY_ENVIRONMENT_VARIABLE: encoded},
            clear=True,
        ):
            client = MODULE.SignedUrlStorageClient.from_environment()
            self.assertNotIn(MODULE.CAPABILITY_ENVIRONMENT_VARIABLE, os.environ)
        self.assertIsNotNone(client)
        self.assertEqual(client.bucket_name, "dessamp-evidence")
        with self.assertRaisesRegex(MODULE.CapabilityError, "outside"):
            client.bucket("other-bucket")
        with self.assertRaisesRegex(MODULE.CapabilityError, "outside"):
            client.bucket("dessamp-evidence").blob("jobs/run/data/unlisted.wav")
        client.close()

    def test_host_bucket_object_traversal_and_expiration_attacks_fail_closed(self):
        mutations = []

        hostile_host = valid_payload()
        hostile_host["read_objects"][0]["url"] = signed_url(
            hostile_host["bucket"],
            hostile_host["read_objects"][0]["name"],
            host="storage.googleapis.com.attacker.test",
        )
        mutations.append(hostile_host)

        mismatched_object = valid_payload()
        mismatched_object["read_objects"][0]["url"] = signed_url(
            mismatched_object["bucket"], "jobs/run/configs/other.json"
        )
        mutations.append(mismatched_object)

        traversal = valid_payload()
        traversal["read_objects"][0]["name"] = "jobs/run/../private.json"
        mutations.append(traversal)

        expired = valid_payload()
        expired["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat().replace("+00:00", "Z")
        mutations.append(expired)

        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(MODULE.CapabilityError):
                    MODULE.encode_capability_bundle(payload)

    def test_signed_get_streams_and_verifies_size_and_hash(self):
        payload_bytes = b"trusted"
        payload = valid_payload(read_payload=payload_bytes)
        client = MODULE.SignedUrlStorageClient(payload)
        blob = client.bucket(payload["bucket"]).blob(payload["read_objects"][0]["name"])

        response = mock.MagicMock(status=200)
        stream = io.BytesIO(payload_bytes)
        response.read.side_effect = stream.read
        connection = mock.MagicMock()
        connection.getresponse.return_value = response
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            MODULE.http.client,
            "HTTPSConnection",
            return_value=connection,
        ):
            destination = Path(temporary_directory) / "data.json"
            blob.download_to_filename(str(destination))
            self.assertEqual(destination.read_bytes(), payload_bytes)

        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_signed_get_hash_mismatch_removes_partial_destination(self):
        payload = valid_payload(read_payload=b"expected")
        client = MODULE.SignedUrlStorageClient(payload)
        blob = client.bucket(payload["bucket"]).blob(payload["read_objects"][0]["name"])
        response = mock.MagicMock(status=200)
        stream = io.BytesIO(b"tampered")
        response.read.side_effect = stream.read
        connection = mock.MagicMock()
        connection.getresponse.return_value = response
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            MODULE.http.client,
            "HTTPSConnection",
            return_value=connection,
        ):
            destination = Path(temporary_directory) / "data.json"
            with self.assertRaisesRegex(MODULE.CapabilityError, "mismatch"):
                blob.download_to_filename(str(destination))
            self.assertFalse(destination.exists())

    def test_signed_put_uses_the_server_enforced_range_and_signed_content_type(self):
        payload = valid_payload()
        client = MODULE.SignedUrlStorageClient(payload)
        blob = client.bucket(payload["bucket"]).blob(payload["write_objects"][0]["name"])
        response = mock.MagicMock(status=200)
        response.read.return_value = b""
        connection = mock.MagicMock()
        connection.getresponse.return_value = response
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            MODULE.http.client,
            "HTTPSConnection",
            return_value=connection,
        ):
            source = Path(temporary_directory) / "resume.ckpt"
            source.write_bytes(b"checkpoint")
            blob.upload_from_filename(str(source))

        connection.putheader.assert_any_call("Content-Type", "application/octet-stream")
        connection.putheader.assert_any_call(
            "x-goog-content-length-range", "1,1024"
        )
        self.assertEqual(b"".join(call.args[0] for call in connection.send.call_args_list), b"checkpoint")
        connection.close.assert_called_once()

    def test_write_capability_rejects_missing_mangled_or_duplicate_signed_range(self):
        mutations = []

        missing_signed_header = valid_payload()
        write = missing_signed_header["write_objects"][0]
        write["url"] = signed_url(
            missing_signed_header["bucket"], write["name"], write=False
        )
        mutations.append(missing_signed_header)

        mismatched_range = valid_payload()
        mismatched_range["write_objects"][0]["headers"][
            "x-goog-content-length-range"
        ] = "1,2048"
        mutations.append(mismatched_range)

        duplicate_signed_header = valid_payload()
        duplicate_signed_header["write_objects"][0]["url"] = duplicate_signed_header[
            "write_objects"
        ][0]["url"].replace(
            "x-goog-content-length-range&",
            "x-goog-content-length-range%3Bx-goog-content-length-range&",
        )
        mutations.append(duplicate_signed_header)

        unsigned_extra_header = valid_payload()
        unsigned_extra_header["write_objects"][0]["headers"]["x-extra"] = "value"
        mutations.append(unsigned_extra_header)

        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(MODULE.CapabilityError):
                    MODULE.encode_capability_bundle(payload)

    def test_local_oversize_check_rejects_before_opening_a_connection(self):
        payload = valid_payload()
        payload["write_objects"][0]["max_bytes"] = 4
        payload["write_objects"][0]["headers"][
            "x-goog-content-length-range"
        ] = "1,4"
        client = MODULE.SignedUrlStorageClient(payload)
        blob = client.bucket(payload["bucket"]).blob(payload["write_objects"][0]["name"])
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            MODULE.http.client, "HTTPSConnection"
        ) as connection:
            source = Path(temporary_directory) / "resume.ckpt"
            source.write_bytes(b"oversize")
            with self.assertRaisesRegex(MODULE.CapabilityError, "exceeds"):
                blob.upload_from_filename(str(source))
        connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
