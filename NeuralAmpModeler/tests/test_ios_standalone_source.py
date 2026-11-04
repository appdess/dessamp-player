import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
APP_VIEW_CONTROLLER = (
    REPO_ROOT / "iPlug2" / "IPlug" / "AUv3" / "iOSApp" / "AppViewController.mm"
)
AUDIO_UNIT_PLAYER = (
    REPO_ROOT / "iPlug2" / "IPlug" / "AUv3" / "iOSApp" / "IPlugAUPlayer.mm"
)
APPLE_PATHS = REPO_ROOT / "iPlug2" / "IPlug" / "IPlugPaths.mm"


class IOSStandaloneSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_VIEW_CONTROLLER.read_text(encoding="utf-8")
        cls.player_source = AUDIO_UNIT_PLAYER.read_text(encoding="utf-8")
        cls.ios_paths = APPLE_PATHS.read_text(encoding="utf-8").split(
            "#pragma mark - iOS", 1
        )[1]

    def test_custom_amp_cache_uses_the_ios_application_support_directory(self):
        support = self.ios_paths.split("void AppSupportPath(", 1)[1].split("\n}", 1)[0]
        self.assertIn("NSApplicationSupportDirectory", support)
        self.assertIn("NSUserDomainMask", support)
        self.assertIn("fileSystemRepresentation", support)

    def test_extension_resources_use_the_declared_audio_unit_extension_point(self):
        extension = self.ios_paths.split("bool IsOOPAuv3AppExtension()", 1)[1]
        self.assertIn("NSExtensionPointIdentifier", extension)
        self.assertIn("com.apple.AudioUnit", extension)

    def test_async_audio_unit_completion_finishes_ui_on_main_queue(self):
        completion = self.source.index(
            "[player loadAudioUnitWithComponentDescription:desc completion:^"
        )
        main_queue = self.source.index(
            "dispatch_async(dispatch_get_main_queue()", completion
        )
        embed = self.source.index("[self embedPlugInView]", main_queue)
        open_window = self.source.index("[audioUnit openWindow:", embed)
        host_resize = self.source.index("[audioUnit hostResized:", open_window)

        self.assertLess(completion, main_queue)
        self.assertLess(main_queue, embed)
        self.assertLess(embed, open_window)
        self.assertLess(open_window, host_resize)

    def test_child_view_controller_containment_is_completed(self):
        add_child = self.source.index("[self addChildViewController:pluginVC]")
        did_move = self.source.index("[pluginVC didMoveToParentViewController:self]")

        self.assertLess(add_child, did_move)

    def test_invalid_input_route_exposes_ui_without_blocking_session_activation(self):
        invalid_route = self.player_source.index("if (!hasValidInputRoute)")
        completion = self.player_source.index("completionBlock();", invalid_route)
        early_return = self.player_source.index("return;", completion)
        session_activation = self.player_source.index(
            "setActive:TRUE", early_return
        )

        self.assertLess(invalid_route, completion)
        self.assertLess(completion, early_return)
        self.assertLess(early_return, session_activation)


if __name__ == "__main__":
    unittest.main()
