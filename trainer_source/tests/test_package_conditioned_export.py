import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "package_conditioned_export.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("package_conditioned_export_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PackageMemberSelectionTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "models": {"41": {"sha256": "a" * 64, "path": "/tmp/model.nam"}},
            "individual": {
                "41": {
                    "summary": {
                        "min_esr": 0.001,
                        "mean_esr": 0.009,
                        "median_esr": 0.008,
                        "p95_esr": 0.018,
                        "max_esr": 0.019,
                    }
                }
            },
            "acceptance": {
                "required_mean_esr_lte": 0.035,
                "required_max_esr_lte": 0.035,
            },
            "quality_target": {
                "required_mean_esr_lte": 0.01,
                "required_max_esr_lte": 0.02,
            },
            "preferred": {
                "required_mean_esr_lte": 0.005,
                "required_max_esr_lte": 0.02,
            },
        }

    def test_hash_bound_member_passes_its_external_gate(self):
        evidence = MODULE.select_member_evidence(
            self.report,
            member="41",
            model_sha256="a" * 64,
        )
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["summary"]["mean_esr"], 0.009)
        self.assertEqual(evidence["quality_tier"], "project-target")

    def test_member_gate_is_independent_of_ensemble_status(self):
        self.report["acceptance"]["passed"] = False
        self.report["individual"]["41"]["summary"]["max_esr"] = 0.036
        evidence = MODULE.select_member_evidence(
            self.report,
            member="41",
            model_sha256="a" * 64,
        )
        self.assertFalse(evidence["passed"])

    def test_nam_normal_member_is_accepted_without_claiming_target(self):
        self.report["individual"]["41"]["summary"]["mean_esr"] = 0.024256
        self.report["individual"]["41"]["summary"]["max_esr"] = 0.033319
        evidence = MODULE.select_member_evidence(
            self.report,
            member="41",
            model_sha256="a" * 64,
        )
        self.assertTrue(evidence["passed"])
        self.assertFalse(evidence["quality_target"]["passed"])
        self.assertEqual(evidence["quality_tier"], "nam-normal")

    def test_model_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MODULE.select_member_evidence(
                self.report,
                member="41",
                model_sha256="b" * 64,
            )

    def test_teacher_distillation_remains_private_validation_only(self):
        fields = MODULE.project_provenance_fields(
            {
                "source_kind": "teacher_model_distillation",
                "workflow_validation_only": True,
                "source_provenance": {
                    "platform": "TONE3000",
                    "tone_id": 70207,
                    "redistribution_decision": "private POC only",
                },
            },
            physical_amplifier=False,
        )
        self.assertEqual(
            fields["purpose"], "teacher-model-distillation-private-validation"
        )
        self.assertEqual(fields["source_kind"], "teacher_model_distillation")
        self.assertTrue(fields["project_workflow_validation_only"])
        self.assertEqual(fields["source_provenance"]["tone_id"], 70207)

    def test_non_teacher_project_keeps_legacy_purpose(self):
        fields = MODULE.project_provenance_fields(
            {"physical_amplifier": True}, physical_amplifier=True
        )
        self.assertEqual(fields["purpose"], "conditioned-cloud-quality-validation")
        self.assertEqual(fields["source_kind"], "physical_amplifier")
        self.assertFalse(fields["project_workflow_validation_only"])

    def test_persistent_project_source_object_is_preserved(self):
        fields = MODULE.project_provenance_fields(
            {
                "source_kind": "teacher_model_distillation",
                "source": {"platform": "TONE3000", "tone_id": 70207},
            },
            physical_amplifier=False,
        )
        self.assertEqual(fields["source_provenance"]["platform"], "TONE3000")
        self.assertEqual(fields["source_provenance"]["tone_id"], 70207)

    def test_malformed_project_release_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "workflow_validation_only"):
            MODULE.project_provenance_fields(
                {"workflow_validation_only": "false"}, physical_amplifier=False
            )

    def test_head_shell_style_is_a_bounded_presentation_token(self):
        for style in (
            "dual-red-modern-v1",
            "forge-streamlined-v1",
            "fire-chrome-v1",
        ):
            with self.subTest(style=style):
                self.assertEqual(MODULE.checked_head_shell_style(style), style)
        with self.assertRaisesRegex(ValueError, "presentation token"):
            MODULE.checked_head_shell_style("../private-art.png")

    def test_product_tokens_are_optional_bounded_presentation_tokens(self):
        self.assertEqual(MODULE.checked_product_token("redline", "product family"), "redline")
        self.assertEqual(MODULE.checked_product_token("brit", "product variant"), "brit")
        self.assertEqual(MODULE.checked_product_token("", "product variant"), "")
        with self.assertRaisesRegex(ValueError, "presentation token"):
            MODULE.checked_product_token("../brit", "product variant")

    def test_control_label_override_changes_only_presentation_text(self):
        self.assertEqual(
            MODULE.checked_control_label_overrides(["gain_index=Gain"]),
            {"gain_index": "Gain"},
        )

    def test_control_label_override_rejects_paths_duplicates_and_empty_labels(self):
        for values in (
            ["../gain=Gain"],
            ["gain_index="],
            ["gain_index=Gain", "gain_index=Drive"],
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MODULE.checked_control_label_overrides(values)


if __name__ == "__main__":
    unittest.main()
