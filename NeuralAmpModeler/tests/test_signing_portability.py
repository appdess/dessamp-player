import plistlib
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHARED_SIGNING_CONFIGS = (
    REPOSITORY_ROOT / "common-ios.xcconfig",
    REPOSITORY_ROOT / "common-mac.xcconfig",
)
ACTIVE_XCODE_PROJECT_FILES = (
    REPOSITORY_ROOT
    / "NeuralAmpModeler"
    / "projects"
    / "NeuralAmpModeler-iOS.xcodeproj"
    / "project.pbxproj",
    REPOSITORY_ROOT
    / "NeuralAmpModeler"
    / "projects"
    / "DessAmp-macOS.xcodeproj"
    / "project.pbxproj",
)
TEAM_ASSIGNMENT = re.compile(
    r"(?:DEVELOPMENT_TEAM(?:\[[^\]]+\])?|DevelopmentTeam)\s*=\s*([^;\n]*);?"
)
MACOS_PROJECT_FILE = ACTIVE_XCODE_PROJECT_FILES[1]
MACOS_RELEASE_APP_CONFIGURATION_ID = "4FD16A0D13B63082001D0217"
MACOS_RELEASE_ENTITLEMENTS = (
    REPOSITORY_ROOT
    / "NeuralAmpModeler"
    / "resources"
    / "DessAmp-macOS-release.entitlements"
)


def _configuration_block(source: str, configuration_id: str) -> str:
    marker = f"\t\t{configuration_id} /*"
    start = source.find(marker)
    if start < 0:
        raise AssertionError(f"missing Xcode configuration {configuration_id}")
    end = source.find("\n\t\t};", start)
    if end < 0:
        raise AssertionError(f"unterminated Xcode configuration {configuration_id}")
    return source[start : end + len("\n\t\t};")]


class SigningPortabilityTests(unittest.TestCase):
    def test_active_projects_do_not_embed_personal_team_ids(self):
        for source_path in SHARED_SIGNING_CONFIGS + ACTIVE_XCODE_PROJECT_FILES:
            source = source_path.read_text(encoding="utf-8")
            with self.subTest(path=source_path.relative_to(REPOSITORY_ROOT)):
                for assignment in TEAM_ASSIGNMENT.finditer(source):
                    value = assignment.group(1).strip().strip('"')
                    self.assertFalse(
                        re.fullmatch(r"[A-Z0-9]{10}", value),
                        f"hard-coded Apple team in {source_path}: {value}",
                    )

    def test_xcode_targets_do_not_override_the_shared_team_configuration(self):
        for source_path in ACTIVE_XCODE_PROJECT_FILES:
            source = source_path.read_text(encoding="utf-8")
            with self.subTest(path=source_path.relative_to(REPOSITORY_ROOT)):
                self.assertIsNone(
                    TEAM_ASSIGNMENT.search(source),
                    f"target-level team override bypasses shared config: {source_path}",
                )

    def test_local_signing_file_is_ignored_and_has_a_public_template(self):
        ignore_text = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/dessamp-signing.local.xcconfig", ignore_text.splitlines())

        template = (
            REPOSITORY_ROOT / "dessamp-signing.local.xcconfig.example"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "DESSAMP_IOS_DEVELOPMENT_TEAM = REPLACE_WITH_APPLE_TEAM_ID", template
        )
        self.assertIn(
            "DESSAMP_MAC_DEVELOPMENT_TEAM = REPLACE_WITH_APPLE_TEAM_ID", template
        )
        self.assertNotRegex(template, r"\b[A-Z0-9]{10}\b")

    def test_macos_release_app_uses_the_minimal_release_entitlements(self):
        project_source = MACOS_PROJECT_FILE.read_text(encoding="utf-8")
        release_app = _configuration_block(
            project_source, MACOS_RELEASE_APP_CONFIGURATION_ID
        )
        self.assertIn("/* Release */", release_app)
        self.assertIn(
            'CODE_SIGN_ENTITLEMENTS = "../resources/DessAmp-macOS-release.entitlements";',
            release_app,
        )
        self.assertNotIn('CODE_SIGN_ENTITLEMENTS = "DessAmp-macOS.entitlements";', release_app)
        self.assertIn("ENABLE_HARDENED_RUNTIME = YES;", release_app)

        with MACOS_RELEASE_ENTITLEMENTS.open("rb") as handle:
            entitlements = plistlib.load(handle)
        self.assertEqual(
            entitlements,
            {"com.apple.security.device.audio-input": True},
        )


if __name__ == "__main__":
    unittest.main()
