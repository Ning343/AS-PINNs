import unittest

from as_pinns.cases import CASE_IDS
from as_pinns.config import iter_case_config_paths, load_case_config


class CaseConfigTests(unittest.TestCase):
    def test_all_case_configs_load(self):
        paths = iter_case_config_paths()
        self.assertEqual(len(paths), 3)
        loaded_ids = {load_case_config(path.stem)["case_id"] for path in paths}
        self.assertEqual(loaded_ids, set(CASE_IDS))

    def test_config_references_have_lightweight_checks(self):
        for case_id in CASE_IDS:
            with self.subTest(case_id=case_id):
                config = load_case_config(case_id)
                self.assertTrue(config["lightweight_checks"])
                self.assertTrue(config["training"]["execute_requires"])


if __name__ == "__main__":
    unittest.main()
