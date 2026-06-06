import unittest

import numpy as np

from as_pinns.reference_solutions import (
    force_material_profiles,
    force_profile,
    function_fitting_reference,
    sample_reference,
)


class ReferenceSolutionTests(unittest.TestCase):
    def test_function_fitting_reference_piecewise_values(self):
        profile = function_fitting_reference([[0.2], [0.5], [1.0], [2.5]])
        self.assertTrue(np.allclose(profile.values["du"].ravel(), [4.0, 3.8, 3.0, 2.0]))
        self.assertTrue(np.allclose(profile.values["d2u"].ravel(), [0.0, -2.0, 0.0, 0.0]))

    def test_force_profile_segments(self):
        values = force_profile([[0.0], [1.0 / 3.0], [0.5], [0.8]]).ravel()
        self.assertTrue(np.allclose(values, [30.0, 20.0, 20.0, 0.0]))

    def test_force_material_profiles_segments(self):
        profile = force_material_profiles([[0.2], [0.4], [0.6], [0.8]])
        self.assertTrue(np.allclose(profile.values["q"].ravel(), [0.0, 20.0, 20.0, 0.0]))
        self.assertTrue(np.allclose(profile.values["ei"].ravel(), [10.0, 10.0, 5.0, 5.0]))

    def test_sample_reference_supports_all_cases(self):
        for case_id in ("function_fitting", "force_discontinuity", "force_material_discontinuity"):
            with self.subTest(case_id=case_id):
                profile = sample_reference(case_id, points=11)
                self.assertEqual(profile.x.shape, (11, 1))
                self.assertTrue(profile.values)


if __name__ == "__main__":
    unittest.main()
