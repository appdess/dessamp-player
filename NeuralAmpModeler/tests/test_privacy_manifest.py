import pathlib
import plistlib
import re
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "NeuralAmpModeler" / "resources" / "PrivacyInfo.xcprivacy"
IOS_PROJECT = (
    REPOSITORY_ROOT
    / "NeuralAmpModeler"
    / "projects"
    / "NeuralAmpModeler-iOS.xcodeproj"
    / "project.pbxproj"
)


class PrivacyManifestTests(unittest.TestCase):
    def test_manifest_declares_only_observed_file_metadata_use(self):
        with MANIFEST.open("rb") as source:
            manifest = plistlib.load(source)

        self.assertIs(manifest["NSPrivacyTracking"], False)
        self.assertEqual(manifest["NSPrivacyTrackingDomains"], [])
        self.assertEqual(manifest["NSPrivacyCollectedDataTypes"], [])
        self.assertEqual(
            manifest["NSPrivacyAccessedAPITypes"],
            [
                {
                    "NSPrivacyAccessedAPIType": "NSPrivacyAccessedAPICategoryFileTimestamp",
                    "NSPrivacyAccessedAPITypeReasons": ["C617.1", "3B52.1"],
                }
            ],
        )

    def test_manifest_is_a_resource_of_the_framework_that_uses_stat(self):
        project = IOS_PROJECT.read_text(encoding="utf-8")
        framework_resources = re.search(
            r"4FC6982D293BA47F0076EC33 /\* Resources \*/ = \{.*?\n\t\t\};",
            project,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(framework_resources)
        assert framework_resources is not None
        self.assertIn("PrivacyInfo.xcprivacy in Resources", framework_resources.group(0))
        self.assertEqual(project.count("PrivacyInfo.xcprivacy in Resources"), 2)


if __name__ == "__main__":
    unittest.main()
