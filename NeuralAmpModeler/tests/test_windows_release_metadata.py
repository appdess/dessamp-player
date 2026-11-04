import importlib.util
import hashlib
import json
import pathlib
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT / "NeuralAmpModeler"
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_resources-win.py"
SPEC = importlib.util.spec_from_file_location("dessamp_prepare_resources_win", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class WindowsReleaseMetadataTests(unittest.TestCase):
    def test_x64_projects_compile_the_current_core_sources(self):
        core = REPOSITORY_ROOT / "NeuralAmpModelerCore/NAM"
        for target in ("app", "vst3"):
            project = PROJECT_ROOT / "projects" / f"NeuralAmpModeler-{target}.vcxproj"
            sources = {
                (project.parent / node.get("Include").replace("\\", "/")).resolve()
                for node in ET.parse(project).findall(".//{*}ClCompile[@Include]")
            }
            self.assertTrue(set(core.glob("*.cpp")).issubset(sources))

    def test_private_input_mismatch_cannot_leave_a_compilable_include(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            build = root / "build"
            build.mkdir()
            inputs = [f"Amplifiers/amp-{i:02d}.dess" for i in range(1, 5)]
            inputs += [f"Drives/{name}.nam" for name in ("OD808", "SD1", "TS9", "aesahaettr")]
            inputs += [f"IRs/{name}" for name in MODULE.CABINET_HASHES]
            inputs += ["Defaults/FactoryDefaults.json"]
            entries = {}
            for name in inputs:
                file = root / name
                file.parent.mkdir(exist_ok=True)
                if name.startswith("IRs/"):
                    if not (PROJECT_ROOT / "resources/FactoryBank").is_dir():
                        self.skipTest("Private Version C bank omitted from this source export")
                    file.write_bytes(((PROJECT_ROOT / "resources/FactoryBank") / file.name).read_bytes())
                elif name == "Defaults/FactoryDefaults.json":
                    file.write_bytes(((PROJECT_ROOT / "resources/FactoryBank") / "FactoryDefaults.json").read_bytes())
                else:
                    file.write_bytes(name.encode())
                entries[name] = {"bytes": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            ledger = root / "ledger.json"
            ledger.write_text(json.dumps(entries))
            digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
            MODULE.prepare_private_listening(root, ledger, digest, build)
            include = build / "listening-files.iss"
            self.assertEqual(include.read_text().count("Source:"), 26)
            (root / inputs[0]).write_bytes(b"modified")
            with self.assertRaises(ValueError):
                MODULE.prepare_private_listening(root, ledger, digest, build)
            self.assertFalse(include.exists())

    def test_windows_product_metadata_matches_config(self):
        config = MODULE.parse_config(str(PROJECT_ROOT))
        source = (PROJECT_ROOT / "resources" / "main.rc").read_text(encoding="utf-8")
        numeric = ",".join(
            (config["MAJOR_STR"], config["MINOR_STR"], config["BUGFIX_STR"], "0")
        )
        self.assertRegex(source, rf"(?m)^\s*FILEVERSION\s+{re.escape(numeric)}$")
        self.assertRegex(source, rf"(?m)^\s*PRODUCTVERSION\s+{re.escape(numeric)}$")
        for key, value in (
            ("FileVersion", config["FULL_VER_STR"]),
            ("ProductVersion", config["FULL_VER_STR"]),
            ("FileDescription", config["PLUG_NAME"]),
            ("InternalName", config["PLUG_NAME"]),
            ("ProductName", config["PLUG_NAME"]),
            ("CompanyName", config["PLUG_MFR"]),
            ("LegalCopyright", config["PLUG_COPYRIGHT_STR"]),
        ):
            self.assertIn(f'VALUE "{key}", "{value}"', source)

        props = (PROJECT_ROOT / "config" / "NeuralAmpModeler-win.props").read_text(
            encoding="utf-8"
        )
        self.assertIn("<BINARY_NAME>DessAmp</BINARY_NAME>", props)
        app_project = (
            PROJECT_ROOT / "projects" / "NeuralAmpModeler-app.vcxproj"
        ).read_text(encoding="utf-8")
        app_filters = (
            PROJECT_ROOT / "projects" / "NeuralAmpModeler-app.vcxproj.filters"
        ).read_text(encoding="utf-8-sig")
        self.assertNotIn("NeuralAmpModeler.ico", source + app_project + app_filters)

    def test_windows_packaging_scripts_use_current_product_paths(self):
        packaging = "\n".join(
            (PROJECT_ROOT / path).read_text(encoding="utf-8")
            for path in (
                "scripts/makedist-win.bat",
                "scripts/makezip-win.py",
                "scripts/update_installer-win.py",
                "installer/DessAmp.iss",
            )
        )
        self.assertNotIn("DessMetal", packaging)
        self.assertNotIn("NeuralAmpModeler Installer", packaging)
        self.assertIn("DessAmp Installer", packaging)
        self.assertNotIn("NeuralAmpModeler manual.pdf", packaging)
        self.assertNotIn("changelog.txt", packaging)

    def test_resource_updater_is_idempotent(self):
        config = dict(MODULE.parse_config(str(PROJECT_ROOT)))
        config.update(
            FULL_VER_STR="9.8.7", MAJOR_STR="9", MINOR_STR="8", BUGFIX_STR="7"
        )
        source = (PROJECT_ROOT / "resources" / "main.rc").read_text(encoding="utf-8")
        first = MODULE.update_resource_text(source, config)
        second = MODULE.update_resource_text(first, config)
        self.assertEqual(second, first)
        self.assertIn("FILEVERSION 9,8,7,0", first)
        self.assertIn('VALUE "FileVersion", "9.8.7"', first)


if __name__ == "__main__":
    unittest.main()
