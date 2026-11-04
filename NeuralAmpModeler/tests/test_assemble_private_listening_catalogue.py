from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "tools" / "assemble_private_listening_catalogue.py"
SPEC = importlib.util.spec_from_file_location("dessamp_private_catalogue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from dessmetal_capture.dess_package import create_dess_package  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssemblePrivateListeningCatalogueTests(unittest.TestCase):
    def _control(self, control_id: str, index: int) -> dict:
        return {
            "id": control_id,
            "label": control_id.replace("_", " ").title(),
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.1,
            "unit": "",
            "taper": "linear",
            "default_value": 0.5,
            "model_parameter_index": index,
        }

    def _model(self, path: str, digest: str, condition_count: int) -> dict:
        return {
            "path": path,
            "format": "nam",
            "sha256": digest,
            "format_version": "0.7.0",
            "architecture": "WaveNet",
            "sample_rate": 48000,
            "parameter_mode": "global_condition" if condition_count else "snapshot",
            "global_condition_size": condition_count,
        }

    def _package(
        self,
        root: pathlib.Path,
        *,
        key: str,
        name: str,
        controls: tuple[str, ...],
        head_shell: str,
        product_family: str = "",
        product_variant: str = "",
        morph: bool = False,
        fire_fixed: bool = False,
    ) -> pathlib.Path:
        package_root = root / key
        package_root.mkdir()
        model_path = package_root / "model.nam"
        model_path.write_bytes((key + "-primary-model").encode("utf-8"))
        manifest = {
            "schema_version": 1,
            "artifact_id": key,
            "name": name,
            "model": self._model("model.nam", _sha256(model_path), len(controls)),
            "controls": [self._control(control_id, index) for index, control_id in enumerate(controls)],
            "presentation": {"head_shell": head_shell},
            "fixed_controls": [],
        }
        if product_family:
            manifest["presentation"]["product_family"] = product_family
        if product_variant:
            manifest["presentation"]["product_variant"] = product_variant
        if key in {"fire", "forge"}:
            manifest["fixed_controls"] = [
                {"id": "bright", "label": "Bright", "taper": "switch", "value": 1.0},
                {"id": "bottom", "label": "Bottom", "taper": "switch", "value": 0.0},
                {"id": "mid_boost", "label": "Mid Boost", "taper": "switch", "value": 0.0},
            ]
        if fire_fixed:
            manifest["fixed_controls"].append(
                {
                    "id": "lead_volume",
                    "label": "Lead Volume",
                    "taper": "linear",
                    "value": 3.5,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "unit": "",
                }
            )
        if morph:
            morph_model = package_root / "morph-model.nam"
            morph_model.write_bytes((key + "-morph-model").encode("utf-8"))
            manifest["morph"] = {
                "mode": "dual_model_linear_v1",
                "from_label": "GER",
                "to_label": "BRIT",
                "model": self._model("morph-model.nam", _sha256(morph_model), 0),
                "alignment": {
                    "delay_samples": 0,
                    "polarity": 1,
                    "level_scale": 1.0,
                    "reference_sample_rate": 48000,
                },
                "blend": {"law": "linear", "center_gain": 1.0, "smoothing_ms": 30.0},
                "control": self._control("morph", 0) | {"model_parameter_index": None},
            }
        manifest_path = package_root / "player-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return pathlib.Path(create_dess_package(manifest_path, root / f"{key}.dess")["path"])

    def _loader(self, root: pathlib.Path, *, fail: bool = False) -> pathlib.Path:
        loader = root / "loadmodel_manifest"
        if fail:
            body = "#!/bin/sh\necho rejected >&2\nexit 1\n"
        else:
            body = """#!/usr/bin/env python3
import json
import pathlib
import sys
name = pathlib.Path(sys.argv[1]).name
is_redline = "redline" in name
print(json.dumps({
    "status": "valid",
    "parameter_count": 1 if is_redline else 6 if "forge" in name else 5,
    "morph": "forge" in name,
    "slimmable": False,
    "selected_size": 1.0,
    "size_breakpoints": [],
}))
"""
        loader.write_text(body, encoding="utf-8")
        loader.chmod(0o755)
        return loader

    def _inputs(
        self,
        root: pathlib.Path,
        *,
        brit_variant: str = "brit",
        fire_status: str = "accepted",
        core_fail: bool = False,
    ) -> dict:
        packages = {
            "fire": self._package(
                root,
                key="fire",
                name="DESS FIRE",
                controls=MODULE.FIRE_CONTROLS,
                head_shell="fire-chrome-v1",
                fire_fixed=True,
            ),
            "forge": self._package(
                root,
                key="forge",
                name="DESS FORGE private proof",
                controls=MODULE.FIRE_CONTROLS,
                head_shell="studio-black-v1",
                morph=True,
            ),
            "redline_us": self._package(
                root,
                key="redline-us",
                name="DESS REDLINE / US",
                controls=("gain_index",),
                head_shell="dual-red-modern-v1",
                product_family="redline",
                product_variant="us",
            ),
            "redline_brit": self._package(
                root,
                key="redline-brit",
                name="DESS REDLINE / BRIT",
                controls=("gain_index",),
                head_shell="dual-red-modern-v1",
                product_family="redline",
                product_variant=brit_variant,
            ),
        }
        hashes = {key: _sha256(path) for key, path in packages.items()}
        report = root / "fire-package-report.json"
        report.write_text(
            json.dumps(
                {
                    "status": fire_status,
                    "selected_member": "seed-84",
                    "package": {"sha256": hashes["fire"]},
                    "presentation": {"head_shell": "fire-chrome-v1"},
                    "external_evidence": {"passed": fire_status == "accepted", "quality_tier": "project-target"},
                }
            ),
            encoding="utf-8",
        )
        return {
            "packages": packages,
            "package_hashes": hashes,
            "fire_report": report,
            "fire_report_sha256": _sha256(report),
            "core_loader": self._loader(root, fail=core_fail),
        }

    def test_assembles_exact_catalogue_without_persisting_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = self._inputs(root)
            output = root / "final-private-catalogue"

            result = MODULE.assemble_catalogue(output_dir=output, **inputs)

            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "01-fire.dess",
                    "02-forge.dess",
                    "03-redline-us.dess",
                    "04-redline-brit.dess",
                    "catalogue-lock.json",
                ],
            )
            for contract in MODULE.SLOT_CONTRACTS:
                self.assertEqual(
                    _sha256(output / contract.filename),
                    inputs["package_hashes"][contract.key],
                )
            lock_text = (output / "catalogue-lock.json").read_text(encoding="utf-8")
            lock = json.loads(lock_text)
            self.assertNotIn(str(root), lock_text)
            self.assertEqual(lock["catalogue_product_order"], ["fire", "forge", "redline"])
            self.assertEqual(lock["catalogue_product_count"], 3)
            self.assertEqual(lock["slot_count"], 4)
            self.assertTrue(lock["redline_variants_share_product_root"])
            self.assertFalse(lock["gates"]["owner_listening_passed"])
            self.assertEqual(result["catalogue_lock_sha256"], _sha256(output / "catalogue-lock.json"))

    def test_calibration_report_bridges_accepted_fire_to_metadata_only_repackage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = self._inputs(root)
            original_fire = inputs["packages"]["fire"]
            accepted_fire_sha256 = inputs["package_hashes"]["fire"]
            calibrated_fire = root / "calibrated-fire.dess"
            calibrated_fire.write_bytes(original_fire.read_bytes() + b"calibrated")
            inputs["packages"]["fire"] = calibrated_fire
            inputs["package_hashes"]["fire"] = _sha256(calibrated_fire)
            calibration_report = root / "fire-calibration-report.json"
            calibration_report.write_text(
                json.dumps(
                    {
                        "status": "metadata-only-input-calibration-repackage",
                        "input_level_dbu": -0.54,
                        "source_package": {"sha256": accepted_fire_sha256},
                        "calibrated_package": {"sha256": _sha256(calibrated_fire)},
                        "network_payload_sha256": "1" * 64,
                        "network_payload_unchanged": True,
                        "current_core": {"status": "valid"},
                    }
                ),
                encoding="utf-8",
            )
            inputs["fire_calibration_report"] = calibration_report
            inputs["fire_calibration_report_sha256"] = _sha256(calibration_report)

            original_reader = MODULE.read_dess_package
            MODULE.read_dess_package = lambda path, cache_dir: original_reader(
                original_fire
                if pathlib.Path(path).resolve() == calibrated_fire.resolve()
                else path,
                cache_dir=cache_dir,
            )
            try:
                result = MODULE.assemble_catalogue(
                    output_dir=root / "catalogue", **inputs
                )
            finally:
                MODULE.read_dess_package = original_reader

            lock = json.loads(
                pathlib.Path(result["catalogue_lock"]).read_text(encoding="utf-8")
            )
            self.assertEqual(lock["fire_acceptance"]["status"], "accepted")
            self.assertEqual(lock["fire_calibration"]["input_level_dbu"], -0.54)

    def test_rejects_a_package_hash_mismatch_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = self._inputs(root)
            inputs["package_hashes"]["forge"] = "0" * 64
            output = root / "catalogue"

            with self.assertRaisesRegex(RuntimeError, "forge package SHA-256"):
                MODULE.assemble_catalogue(output_dir=output, **inputs)

            self.assertFalse(output.exists())

    def test_rejects_a_wrong_redline_variant_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = self._inputs(root, brit_variant="us")

            with self.assertRaisesRegex(RuntimeError, "redline_brit product variant"):
                MODULE.assemble_catalogue(output_dir=root / "catalogue", **inputs)

    def test_requires_an_accepted_fire_package_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = self._inputs(root, fire_status="workflow-validation-only")

            with self.assertRaisesRegex(RuntimeError, "not externally accepted"):
                MODULE.assemble_catalogue(output_dir=root / "catalogue", **inputs)

    def test_rejects_a_current_core_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = self._inputs(root, core_fail=True)

            with self.assertRaisesRegex(RuntimeError, "current-Core validation failed"):
                MODULE.assemble_catalogue(output_dir=root / "catalogue", **inputs)

    def test_refuses_private_output_inside_the_repository(self):
        forbidden = REPOSITORY_ROOT / "never-create-private-catalogue-here"
        self.assertFalse(forbidden.exists())
        with self.assertRaisesRegex(RuntimeError, "outside the repository"):
            MODULE._checked_output_directory(forbidden)


if __name__ == "__main__":
    unittest.main()
