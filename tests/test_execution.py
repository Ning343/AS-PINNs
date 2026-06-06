import json
import tempfile
import unittest
from pathlib import Path

from as_pinns.cases import CASE_IDS
from as_pinns.execution import build_run_manifest, check_dependencies, write_run_manifest
from as_pinns.training_plan import build_training_plan


class ExecutionManifestTests(unittest.TestCase):
    def test_dependency_status_matches_training_plan(self):
        for case_id in CASE_IDS:
            with self.subTest(case_id=case_id):
                plan = build_training_plan(case_id)
                statuses = check_dependencies(plan)
                self.assertEqual([status.package for status in statuses], list(plan.execute_requires))
                self.assertTrue(all(isinstance(status.available, bool) for status in statuses))

    def test_run_manifest_contains_reproducibility_context(self):
        manifest = build_run_manifest("force_discontinuity", output_directory="outputs/_intermediate")
        payload = manifest.to_dict()
        self.assertEqual(payload["plan"]["case_id"], "force_discontinuity")
        self.assertIn("python_version", payload)
        self.assertIn("platform", payload)
        self.assertIn("dependency_status", payload)
        self.assertIn("notes", payload)

    def test_write_run_manifest(self):
        manifest = build_run_manifest("function_fitting")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run_manifest(manifest, output_directory=Path(tmp))
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["plan"]["case_id"], "function_fitting")
            self.assertEqual(path.name, "function_fitting_run_manifest.json")


if __name__ == "__main__":
    unittest.main()
