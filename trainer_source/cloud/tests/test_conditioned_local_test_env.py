import pathlib
import tomllib
import unittest


_ENV_DIR = pathlib.Path(__file__).resolve().parents[1] / "local_test_env"
_LOCK_FILE = _ENV_DIR / "uv.lock"
_PROJECT_FILE = _ENV_DIR / "pyproject.toml"


class ConditionedLocalTestEnvironmentTests(unittest.TestCase):
    def test_project_declares_only_the_intended_local_test_contract(self):
        project = tomllib.loads(_PROJECT_FILE.read_text())

        self.assertEqual(project["project"]["requires-python"], ">=3.12,<3.13")
        self.assertEqual(
            set(project["project"]["dependencies"]),
            {
                "dessmetal-capture",
                "google-cloud-storage==3.4.1",
                "neural-amp-modeler==0.13.0",
                "pytest==9.0.3",
            },
        )
        self.assertFalse(project["tool"]["uv"]["package"])
        self.assertEqual(
            project["tool"]["uv"]["sources"]["dessmetal-capture"],
            {"path": "../../../capture_companion", "editable": True},
        )

    def test_lock_is_portable_and_excludes_retained_trainer_dependencies(self):
        lock_text = _LOCK_FILE.read_text()
        lock = tomllib.loads(lock_text)
        packages = {item["name"]: item for item in lock["package"]}

        self.assertEqual(lock["requires-python"], "==3.12.*")
        self.assertNotIn("/Users/", lock_text)
        self.assertNotIn("file://", lock_text)
        self.assertEqual(packages["dessmetal-capture"]["version"], "0.1.0")
        self.assertEqual(packages["google-cloud-storage"]["version"], "3.4.1")
        self.assertEqual(packages["neural-amp-modeler"]["version"], "0.13.0")
        self.assertEqual(packages["pytest"]["version"], "9.0.3")
        self.assertNotIn("auraloss", packages)
        self.assertNotIn("torchaudio", packages)


if __name__ == "__main__":
    unittest.main()
