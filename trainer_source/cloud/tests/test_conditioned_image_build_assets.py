import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


_CLOUD_DIR = pathlib.Path(__file__).resolve().parents[1]
_TRAINER_ROOT = _CLOUD_DIR.parent
_BUILD_SCRIPT = _CLOUD_DIR / "build_conditioned_a2_image.sh"
_CONSTRAINTS_FILE = (
    _CLOUD_DIR / "requirements-conditioned-a2-image.constraints.txt"
)
_DIRECT_REQUIREMENTS_FILE = _CLOUD_DIR / "requirements-conditioned-a2.txt"
_DOCKERFILE = _CLOUD_DIR / "Dockerfile.conditioned-a2"
_IGNORE_FILE = _CLOUD_DIR / "conditioned-a2.gcloudignore"
_LOCK_FILE = _CLOUD_DIR / "requirements-conditioned-a2.lock.txt"


def _normalized_package_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _exact_pins(path):
    pins = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if separator != "==" or not name or not version:
            raise AssertionError(f"Non-exact pin in {path}: {raw_line}")
        pins[_normalized_package_name(name)] = version
    return pins


def _locked_requirement_blocks(text):
    starts = list(
        re.finditer(
            r"(?m)^([A-Za-z0-9_.-]+)==([^ \\\n]+)(?: \\)?$",
            text,
        )
    )
    blocks = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        blocks[_normalized_package_name(match.group(1))] = (
            match.group(2),
            text[match.start() : end],
        )
    return blocks


class ConditionedImageBuildAssetsTest(unittest.TestCase):
    def test_upload_allowlist_contains_only_required_roots(self):
        text = _IGNORE_FILE.read_text()

        self.assertTrue(text.lstrip().startswith("#"))
        self.assertIn("\n**\n", text)
        self.assertIn("!LICENSE", text)
        self.assertIn("!cloud/Dockerfile.conditioned-a2", text)
        self.assertIn("!cloud/cloudbuild-conditioned-a2.yaml", text)
        self.assertIn("!cloud/requirements-conditioned-a2.txt", text)
        self.assertIn(
            "!cloud/requirements-conditioned-a2-image.constraints.txt", text
        )
        self.assertIn("!cloud/requirements-conditioned-a2.lock.txt", text)
        self.assertIn("!cloud/conditioned_a2_entrypoint.py", text)
        self.assertIn("!conditioned_a2/*.py", text)
        self.assertNotIn("!.venv", text)
        self.assertNotIn("!tests", text)
        self.assertNotIn("!nam/", text)

    def test_build_script_dry_run_is_bounded_and_non_billable(self):
        result = subprocess.run(
            [
                str(_BUILD_SCRIPT),
                "--project",
                "dessamp-training-12345",
                "--confirm-project",
                "dessamp-training-12345",
                "--region",
                "europe-west3",
                "--repository",
                "nam-training",
                "--image-name",
                "conditioned-a2",
                "--image-tag",
                "dry-run-test",
                "--dry-run",
            ],
            cwd=_TRAINER_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn(
            "image=europe-west3-docker.pkg.dev/dessamp-training-12345/"
            "nam-training/conditioned-a2:dry-run-test",
            result.stdout,
        )
        self.assertIn("gcloud builds submit", result.stdout)
        self.assertIn("--ignore-file=", result.stdout)
        self.assertIn("conditioned-a2.gcloudignore", result.stdout)

    @unittest.skipUnless(shutil.which("gcloud"), "gcloud is not installed")
    def test_gcloud_upload_listing_matches_the_strict_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            staged_root = pathlib.Path(temp_dir) / "trainer_source"
            shutil.copytree(_TRAINER_ROOT, staged_root)
            shutil.copy2(_IGNORE_FILE, staged_root / ".gcloudignore")

            result = subprocess.run(
                ["gcloud", "meta", "list-files-for-upload", str(staged_root)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            set(result.stdout.splitlines()),
            {
                "LICENSE",
                "cloud/Dockerfile.conditioned-a2",
                "cloud/cloud_train_entrypoint.py",
                "cloud/cloudbuild-conditioned-a2.yaml",
                "cloud/conditioned-a2.gcloudignore",
                "cloud/conditioned_a2_contract.py",
                "cloud/conditioned_a2_entrypoint.py",
                "cloud/gcs_capabilities.py",
                "cloud/requirements-conditioned-a2-image.constraints.txt",
                "cloud/requirements-conditioned-a2.lock.txt",
                "cloud/requirements-conditioned-a2.txt",
                "conditioned_a2/__init__.py",
                "conditioned_a2/checkpoint.py",
                "conditioned_a2/dataset.py",
                "conditioned_a2/model.py",
                "conditioned_a2/train.py",
            },
        )

    def test_hash_lock_matches_observed_image_constraints(self):
        constraints = _exact_pins(_CONSTRAINTS_FILE)
        direct_requirements = _exact_pins(_DIRECT_REQUIREMENTS_FILE)
        lock_text = _LOCK_FILE.read_text()
        locked = _locked_requirement_blocks(lock_text)

        self.assertNotIn("/dev/fd/", lock_text)
        self.assertGreaterEqual(len(locked), 100)
        self.assertLessEqual(len(locked), 125)
        self.assertTrue(set(direct_requirements).issubset(locked))
        self.assertTrue(set(constraints).issubset(locked))

        for name, version in constraints.items():
            self.assertEqual(locked[name][0], version, name)
        for name, (_, block) in locked.items():
            self.assertRegex(block, r"--hash=sha256:[0-9a-f]{64}", name)

    def test_dockerfile_uses_the_hash_lock_and_asserts_cuda_base(self):
        text = _DOCKERFILE.read_text()

        self.assertIn(
            "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:"
            "77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee",
            text,
        )
        self.assertIn("requirements-conditioned-a2-image.constraints.txt", text)
        self.assertIn("requirements-conditioned-a2.lock.txt", text)
        self.assertIn("--require-hashes", text)
        self.assertIn("pip check", text)
        self.assertIn('torch.__version__ == "2.6.0+cu124"', text)
        self.assertIn('torch.version.cuda == "12.4"', text)
        self.assertIn('numpy.__version__ == "2.2.2"', text)

    def test_build_script_rejects_project_confirmation_mismatch(self):
        result = subprocess.run(
            [
                str(_BUILD_SCRIPT),
                "--project",
                "dessamp-training-12345",
                "--confirm-project",
                "different-training-12345",
                "--region",
                "europe-west3",
                "--repository",
                "nam-training",
                "--image-name",
                "conditioned-a2",
                "--image-tag",
                "dry-run-test",
                "--dry-run",
            ],
            cwd=_TRAINER_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Project confirmation mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
