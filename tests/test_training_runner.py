import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from as_pinns.training.runner import (
    TrainingDependencyError,
    prepare_training_run,
)


class TrainingRunnerTests(unittest.TestCase):
    def test_prepare_training_run_writes_traceable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_training_run(
                "function_fitting",
                output_directory=Path(tmp),
                run_name="unit-run",
                require_dependencies=False,
            )
            self.assertEqual(prepared.case_id, "function_fitting")
            self.assertTrue(prepared.manifest_path.exists())
            self.assertTrue(prepared.copied_script_path.exists())
            self.assertTrue(prepared.copied_solution_path.exists())
            self.assertTrue((prepared.output_directory / "prepared_run.json").exists())
            self.assertEqual(prepared.work_directory.name, "work")
            self.assertIn("as_pinns_ex1.py", str(prepared.copied_script_path))

            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["plan"]["case_id"], "function_fitting")
            self.assertEqual(Path(manifest["plan"]["output_directory"]), prepared.output_directory)

    def test_prepare_training_run_reports_missing_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("as_pinns.training.runner.check_dependencies") as patched:
                patched.return_value = [
                    mock.Mock(package="deepxde", available=False),
                    mock.Mock(package="tensorflow", available=False),
                ]
                with self.assertRaises(TrainingDependencyError) as raised:
                    prepare_training_run(
                        "force_discontinuity",
                        output_directory=Path(tmp),
                        run_name="missing-deps",
                    )
            self.assertIn("deepxde", str(raised.exception))
            self.assertIn("tensorflow", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
