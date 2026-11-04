import argparse
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ENTRYPOINT_PATH = Path(__file__).resolve().parents[1] / "cloud_train_entrypoint.py"
SPEC = importlib.util.spec_from_file_location("cloud_train_entrypoint", ENTRYPOINT_PATH)
ENTRYPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRYPOINT)


class SafeGCSDownloadPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "downloads"

    def test_maps_a_nested_object_below_the_exact_prefix(self):
        result = ENTRYPOINT.safe_gcs_download_path(
            "configs/default_config_files/learning/default.json",
            "configs",
            self.root,
        )

        self.assertEqual(
            result,
            self.root.resolve() / "default_config_files" / "learning" / "default.json",
        )
        self.assertFalse(result.parent.exists())

    def test_skips_prefix_and_directory_marker_objects(self):
        self.assertIsNone(
            ENTRYPOINT.safe_gcs_download_path("configs", "configs", self.root)
        )
        self.assertIsNone(
            ENTRYPOINT.safe_gcs_download_path("configs/nested/", "configs", self.root)
        )

    def test_rejects_prefix_confusion_and_path_traversal(self):
        unsafe_names = (
            "configs-archive/default.json",
            "configs/../../trainer_source/custom_train_full.py",
            "configs/./default.json",
            "configs//tmp/payload",
            "configs/..\\..\\payload",
            "configs/control\nname",
            "configs/../../trainer_source/",
        )

        for blob_name in unsafe_names:
            with self.subTest(blob_name=blob_name):
                with self.assertRaises(ValueError):
                    ENTRYPOINT.safe_gcs_download_path(blob_name, "configs", self.root)

        with self.assertRaises(ValueError):
            ENTRYPOINT.safe_gcs_download_path(
                "training-data/DessBlock-green/../../../trainer_source/custom_train_full.py",
                "training-data/DessBlock-green",
                self.root,
            )

    def test_rejects_a_destination_symlink(self):
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / "escape").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ValueError):
            ENTRYPOINT.safe_gcs_download_path(
                "configs/escape/new-directory/payload.json",
                "configs",
                self.root,
            )
        self.assertFalse((outside / "new-directory").exists())

    def test_rejects_a_symlinked_destination_root(self):
        outside = Path(self.temporary_directory.name) / "outside-root"
        outside.mkdir()
        self.root.symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ValueError):
            ENTRYPOINT.safe_gcs_download_path(
                "configs/payload.json",
                "configs",
                self.root,
            )

    def test_rejects_an_unsafe_source_prefix(self):
        for prefix in ("../configs", "/configs", "configs\\nested", "configs\narchive"):
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    ENTRYPOINT.safe_gcs_download_path(
                        "configs/payload.json",
                        prefix,
                        self.root,
                    )


class FakeBlob:
    def __init__(self, name, payload=None):
        self.name = name
        self.payload = payload if payload is not None else name.encode("utf-8")
        self.size = len(self.payload)
        self.downloads = []
        self.reloads = 0

    def reload(self):
        self.reloads += 1

    def download_to_filename(self, filename):
        self.downloads.append(filename)
        Path(filename).write_bytes(self.payload)


class FakeBucket:
    def __init__(self, blobs):
        self.blobs = {blob.name: blob for blob in blobs}
        self.prefixes = []

    def list_blobs(self, prefix):
        self.prefixes.append(prefix)
        raise AssertionError("bounded downloads must never list a GCS prefix")

    def blob(self, name):
        return self.blobs[name]


class FakeStorageClient:
    def __init__(self, blobs):
        self.fake_bucket = FakeBucket(blobs)

    def bucket(self, _bucket_name):
        return self.fake_bucket


class DownloadPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "downloads"

    def test_queries_the_exact_prefix_and_downloads_valid_objects(self):
        blob = FakeBlob("configs/nested/default.json")
        client = FakeStorageClient([blob])

        ENTRYPOINT.download_from_gcs(
            "bucket",
            "configs",
            self.root,
            object_specs=[
                {
                    "name": blob.name,
                    "size_bytes": blob.size,
                    "sha256": hashlib.sha256(blob.payload).hexdigest(),
                }
            ],
            storage_client=client,
        )

        self.assertEqual(client.fake_bucket.prefixes, [])
        self.assertEqual(blob.reloads, 1)
        self.assertEqual(len(blob.downloads), 1)
        self.assertEqual(
            (self.root / "nested" / "default.json").read_bytes(),
            blob.payload,
        )

    def test_unlisted_remote_object_is_never_discovered_or_downloaded(self):
        valid = FakeBlob("configs/default.json")
        malicious = FakeBlob("configs/../../trainer_source/custom_train_full.py")
        client = FakeStorageClient([valid, malicious])

        ENTRYPOINT.download_from_gcs(
            "bucket",
            "configs",
            self.root,
            object_specs=[{"name": valid.name}],
            storage_client=client,
        )

        self.assertEqual(len(valid.downloads), 1)
        self.assertEqual(malicious.downloads, [])
        self.assertEqual(client.fake_bucket.prefixes, [])

    def test_malicious_allowlist_object_causes_zero_downloads(self):
        valid = FakeBlob("configs/default.json")
        malicious = FakeBlob("configs/../../trainer_source/custom_train_full.py")
        client = FakeStorageClient([valid, malicious])

        with self.assertRaises(ValueError):
            ENTRYPOINT.download_from_gcs(
                "bucket",
                "configs",
                self.root,
                object_specs=[{"name": valid.name}, {"name": malicious.name}],
                storage_client=client,
            )

        self.assertEqual(valid.downloads, [])
        self.assertEqual(malicious.downloads, [])

    def test_file_directory_collisions_fail_before_download_in_either_order(self):
        for names in (
            ("configs/a", "configs/a/b"),
            ("configs/a/b", "configs/a"),
        ):
            with self.subTest(names=names):
                blobs = [FakeBlob(name) for name in names]
                client = FakeStorageClient(blobs)
                with self.assertRaises(ValueError):
                    ENTRYPOINT.download_from_gcs(
                        "bucket",
                        "configs",
                        self.root,
                        object_specs=[{"name": blob.name} for blob in blobs],
                        storage_client=client,
                    )
                self.assertTrue(all(blob.downloads == [] for blob in blobs))

    def test_duplicate_targets_fail_before_download(self):
        blobs = [FakeBlob("configs/default.json"), FakeBlob("configs/default.json")]
        client = FakeStorageClient(blobs)

        with self.assertRaises(ValueError):
            ENTRYPOINT.download_from_gcs(
                "bucket",
                "configs",
                self.root,
                object_specs=[{"name": blob.name} for blob in blobs],
                storage_client=client,
            )

        self.assertTrue(all(blob.downloads == [] for blob in blobs))

    def test_size_or_hash_mismatch_fails_closed(self):
        blob = FakeBlob("configs/default.json", b"trusted")
        client = FakeStorageClient([blob])
        with self.assertRaisesRegex(ValueError, "size differs"):
            ENTRYPOINT.download_from_gcs(
                "bucket",
                "configs",
                self.root,
                object_specs=[{"name": blob.name, "size_bytes": blob.size + 1}],
                storage_client=client,
            )
        self.assertEqual(blob.downloads, [])

        with self.assertRaisesRegex(ValueError, "hash differs"):
            ENTRYPOINT.download_from_gcs(
                "bucket",
                "configs",
                self.root,
                object_specs=[{"name": blob.name, "sha256": "0" * 64}],
                storage_client=client,
            )
        self.assertFalse((self.root / "default.json").exists())

    def test_expiring_capability_without_remote_metadata_uses_its_declared_bound(self):
        blob = FakeBlob("configs/input-manifest.json", b"{}")
        blob.size = None
        blob.declared_max_bytes = 2 * 1024 * 1024
        client = FakeStorageClient([blob])

        ENTRYPOINT.download_from_gcs(
            "bucket",
            "configs",
            self.root,
            object_specs=[{"name": blob.name}],
            max_object_bytes=2 * 1024 * 1024,
            max_total_bytes=2 * 1024 * 1024,
            storage_client=client,
        )

        self.assertEqual((self.root / "input-manifest.json").read_bytes(), b"{}")


class UploadAllowlistTests(unittest.TestCase):
    def test_uploads_only_exact_relative_paths_and_ignores_unlisted_files(self):
        uploads = {}

        class UploadBlob:
            def __init__(self, name):
                self.name = name

            def upload_from_filename(self, filename):
                uploads[self.name] = Path(filename).read_bytes()

        bucket = type("Bucket", (), {"blob": lambda _self, name: UploadBlob(name)})()
        client = type("Client", (), {"bucket": lambda _self, _name: bucket})()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "training-manifest.json").write_text("{}", encoding="utf-8")
            (root / "unlisted-private.txt").write_text("private", encoding="utf-8")
            ENTRYPOINT.upload_to_gcs(
                "bucket",
                root,
                "jobs/run/results/member",
                storage_client=client,
                relative_paths=("training-manifest.json",),
            )

        self.assertEqual(
            uploads,
            {"jobs/run/results/member/training-manifest.json": b"{}"},
        )

    def test_rejects_upload_escape_and_missing_allowlisted_file(self):
        client = type(
            "Client",
            (),
            {"bucket": lambda _self, _name: type("Bucket", (), {})()},
        )()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                ENTRYPOINT.upload_to_gcs(
                    "bucket",
                    root,
                    "jobs/run/results/member",
                    storage_client=client,
                    relative_paths=("../private",),
                )
            with self.assertRaises(FileNotFoundError):
                ENTRYPOINT.upload_to_gcs(
                    "bucket",
                    root,
                    "jobs/run/results/member",
                    storage_client=client,
                    relative_paths=("missing.json",),
                )

            temporary_path = Path(temporary_directory)
            outside = temporary_path.with_name(
                temporary_path.name + "-outside-upload.txt"
            )
            outside.write_text("outside", encoding="utf-8")
            self.addCleanup(outside.unlink, missing_ok=True)
            (root / "linked.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                ENTRYPOINT.upload_to_gcs(
                    "bucket",
                    root,
                    "jobs/run/results/member",
                    storage_client=client,
                    relative_paths=("linked.json",),
                )


class ModelNameTests(unittest.TestCase):
    def test_accepts_current_model_names(self):
        for model_name in (
            "DessBlock-green",
            "DessBlock-red",
            "SickDess",
            "DessTortion-blue",
            "DessTortion-red",
            "TS9",
            "aesahaettr",
        ):
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    ENTRYPOINT.validated_model_name(model_name),
                    model_name,
                )

    def test_rejects_model_names_that_can_change_paths(self):
        for model_name in ("../SickDess", "Dess/Sick", ".", "name with spaces"):
            with self.subTest(model_name=model_name):
                with self.assertRaises(argparse.ArgumentTypeError):
                    ENTRYPOINT.validated_model_name(model_name)


class PortableBundleTests(unittest.TestCase):
    def test_rewrites_only_basenames_into_downloaded_data_directory(self):
        config = {
            "common": {"x_path": "/capture/source/input.wav"},
            "train": {
                "x_path": "/capture/source/train-input.wav",
                "outputs": [
                    {"y_path": "/capture/outputs/a.wav", "g_vector": [0.2, 0.8]}
                ]
            },
            "validation": {
                "x_path": "/another/machine/validation-input.wav",
                "outputs": [
                    {"y_path": "/another/machine/b.wav", "g_vector": [0.5, 0.5]}
                ]
            },
        }
        rewritten = ENTRYPOINT.rewrite_data_config_paths(config, Path("/app/work/data"))
        self.assertEqual(rewritten["common"]["x_path"], "/app/work/data/input.wav")
        self.assertEqual(
            rewritten["train"]["x_path"], "/app/work/data/train-input.wav"
        )
        self.assertEqual(
            rewritten["validation"]["x_path"],
            "/app/work/data/validation-input.wav",
        )
        self.assertEqual(
            rewritten["train"]["outputs"][0]["y_path"],
            "/app/work/data/a.wav",
        )
        self.assertEqual(
            rewritten["validation"]["outputs"][0]["y_path"],
            "/app/work/data/b.wav",
        )

    def test_rejects_config_paths_as_names(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            ENTRYPOINT.validated_config_name("../data.json")

    def test_accepts_one_safe_checkpoint_filename(self):
        self.assertEqual(
            ENTRYPOINT.validated_config_name("member-21.ckpt"),
            "member-21.ckpt",
        )


if __name__ == "__main__":
    unittest.main()
