import unittest

import numpy as np

from pressure_model import (
    ReservoirInputs,
    invert_pressure_gaver_stehfest,
    late_time_pressure_approximation,
    simulate_drawdown,
    stehfest_weights,
)


class PressureModelTests(unittest.TestCase):
    def setUp(self):
        self.inputs = ReservoirInputs(
            permeability_md=75.0,
            porosity=0.18,
            total_compressibility_psi_inv=1.5e-5,
            viscosity_cp=1.2,
            formation_volume_factor_rb_stb=1.15,
            thickness_ft=60.0,
            wellbore_radius_ft=0.328,
            rate_stb_day=450.0,
            initial_pressure_psi=3500.0,
        )

    def test_stehfest_weights_are_finite_and_alternating(self):
        weights = stehfest_weights(12)
        self.assertEqual(len(weights), 12)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights[:-1] * weights[1:] < 0))

    def test_late_time_matches_radial_flow_approximation(self):
        td = np.array([1e3, 1e4, 1e5])
        numerical = invert_pressure_gaver_stehfest(td, n_terms=12)
        reference = late_time_pressure_approximation(td)
        np.testing.assert_allclose(numerical, reference, rtol=0.012, atol=0.02)

    def test_drawdown_is_finite_monotonic_and_storage_is_zero(self):
        time = np.geomspace(0.001, 240.0, 180)
        result = simulate_drawdown(time, self.inputs)
        for key in ("dimensionless_pressure", "pressure_drop_psi", "flowing_pressure_psi", "pressure_derivative_psi"):
            self.assertTrue(np.all(np.isfinite(result[key])), key)
        self.assertTrue(np.all(np.diff(result["pressure_drop_psi"]) > 0))
        self.assertTrue(np.all(np.diff(result["flowing_pressure_psi"]) < 0))
        self.assertEqual(result["dimensionless_storage"], 0.0)

    def test_dimensionless_derivative_reaches_half_plateau(self):
        td = np.geomspace(10.0, 1e6, 180)
        pd = invert_pressure_gaver_stehfest(td)
        derivative = np.gradient(pd, np.log(td), edge_order=2)
        self.assertAlmostEqual(float(np.median(derivative[-30:])), 0.5, delta=0.015)


if __name__ == "__main__":
    unittest.main()
