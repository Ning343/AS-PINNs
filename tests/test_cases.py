import unittest

from as_pinns.cases import CASE_IDS, get_case, list_cases
from as_pinns.training_plan import build_training_plan


class CaseRegistryTests(unittest.TestCase):
    def test_case_registry_contains_three_benchmarks(self):
        self.assertEqual(CASE_IDS, ("function_fitting", "force_discontinuity", "force_material_discontinuity"))
        self.assertEqual(len(list_cases()), 3)

    def test_case_paths_are_relative_and_stable(self):
        for case in list_cases():
            with self.subTest(case_id=case.case_id):
                self.assertTrue(case.notebook_path.exists())
                self.assertTrue(case.solution_notebook_path.exists())
                self.assertTrue(case.port_path.exists())
                self.assertTrue(case.solution_port_path.exists())
                self.assertNotIn(" ", case.primary_notebook)
                self.assertLess(case.domain[0], case.domain[1])

    def test_training_plans_are_dry_run_by_default(self):
        for case_id in CASE_IDS:
            with self.subTest(case_id=case_id):
                plan = build_training_plan(case_id)
                self.assertEqual(plan.default_mode, "dry-run")
                self.assertEqual(plan.case_id, get_case(case_id).case_id)
                self.assertTrue(plan.iterations)


if __name__ == "__main__":
    unittest.main()
