import unittest
from argparse import Namespace

import numpy as np
import pandas as pd

from scripts.calibrate_heston import build_config
from stabilvol.heston import CalibrationConfig, HestonCalibrator, HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.calibration import moments_frame
from stabilvol.utility.classes.stability_analysis import StabilVolter


class HestonSimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_shaped(self):
        params = HestonParams()
        config = SimulationConfig(n_paths=4, n_steps=12, seed=42)
        first = simulate_modified_heston(params, config)
        second = simulate_modified_heston(params, config)

        self.assertEqual(first.returns.shape, (12, 4))
        self.assertEqual(first.variance.shape, (12, 4))
        self.assertEqual(first.x.shape, (12, 4))
        np.testing.assert_allclose(first.returns.to_numpy(), second.returns.to_numpy())
        self.assertTrue((first.variance >= 0).all())

    def test_correlated_noise_is_available(self):
        params = HestonParams(cc=0.01, rho=0.75)
        config = SimulationConfig(n_paths=2000, n_steps=3, seed=7, correlated_noise=True)
        result = simulate_modified_heston(params, config, keep_shocks=True)
        corr = np.corrcoef(result.price_shocks.ravel(), result.variance_shocks.ravel())[0, 1]
        self.assertGreater(corr, 0.6)


class HestonCountingTests(unittest.TestCase):
    def test_simulated_returns_work_with_stabilvolter(self):
        result = simulate_modified_heston(
            HestonParams(),
            SimulationConfig(n_paths=4, n_steps=200, seed=3, store_state=False),
        )
        calibrator = HestonCalibrator(CalibrationConfig(root=".", threshold_pairs=((-0.5, -1.5),)))
        stabilvol = calibrator.count_simulated_events(result.returns, (-0.5, -1.5), "SIM")
        self.assertIn("Volatility", stabilvol.columns)
        self.assertIn("FHT", stabilvol.columns)
        self.assertTrue((stabilvol["FHT"] >= 2).all())

    def test_quiet_stabilvol_handles_dataframe_apply_result(self):
        index = pd.date_range("2000-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "a": [0.0, -0.6, -0.7, -1.6, 0.0],
                "b": [0.0, -0.8, -0.9, -1.7, 0.0],
            },
            index=index,
        )
        calibrator = HestonCalibrator(CalibrationConfig(root=".", threshold_pairs=((-0.5, -1.5),)))
        stabilvol = calibrator.count_simulated_events(data, (-0.5, -1.5), "SIM")
        self.assertEqual(len(stabilvol), 2)
        self.assertTrue((stabilvol["FHT"] == 3).all())

    def test_threshold_signs_use_existing_stabilvolter_behavior(self):
        index = pd.date_range("2000-01-01", periods=5, freq="D")
        series = pd.Series([0.0, -0.6, -0.7, -1.6, 0.0], index=index)

        negative = StabilVolter(start_level=-0.5, end_level=-1.5, std_normalization=False, tau_min=2, tau_max=30)
        negative_result = negative.count_stock_fht(series)
        self.assertEqual(int(negative_result[1, 0]), 3)

        positive = StabilVolter(start_level=0.5, end_level=1.5, std_normalization=False, tau_min=2, tau_max=30)
        positive_result = positive.count_stock_fht(-series)
        self.assertEqual(int(positive_result[1, 0]), 3)


class HestonCalibrationTests(unittest.TestCase):
    def test_empirical_loader_finds_default_tables(self):
        calibrator = HestonCalibrator(CalibrationConfig(root="."))
        grid = calibrator.load_empirical_grid("UN")
        self.assertEqual(len(grid), 4)
        self.assertTrue(all(not frame.empty for frame in grid.values()))

    def test_loss_is_finite_for_valid_small_simulation(self):
        config = CalibrationConfig(root=".", threshold_pairs=((-0.5, -1.5),), pilot_max_paths=4, pilot_n_steps=200, n_vol_bins=5)
        calibrator = HestonCalibrator(config)
        empirical = calibrator.load_empirical_grid("UN")
        simulation = simulate_modified_heston(
            HestonParams(),
            SimulationConfig(n_paths=4, n_steps=200, seed=5, store_state=False, correlated_noise=True),
        )
        loss = calibrator.grid_loss(empirical, simulation.returns, "UN")
        self.assertTrue(np.isfinite(loss))

    def test_distribution_loss_uses_fht_not_volatility(self):
        calibrator = HestonCalibrator(CalibrationConfig(root=".", tau_min=2, tau_max=5))
        empirical = pd.DataFrame({"Volatility": [0.1, 0.2, 0.3, 0.4], "FHT": [2, 3, 4, 5]})
        simulated = pd.DataFrame({"Volatility": [10.0, 20.0, 30.0, 40.0], "FHT": [2, 3, 4, 5]})
        self.assertEqual(calibrator.distribution_loss(empirical, simulated), 0.0)

    def test_return_moments_ignore_missing_values(self):
        moments = moments_frame(pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]}))
        self.assertAlmostEqual(moments["mean"], 2.0)
        self.assertAlmostEqual(moments["variance"], 2.0 / 3.0)
        self.assertAlmostEqual(moments["skewness"], 0.0)
        self.assertAlmostEqual(moments["kurtosis"], 1.5)

    def test_partial_config_bounds_are_merged_with_defaults(self):
        args = Namespace(
            database=None,
            threshold_pair=None,
            markets=None,
            start_date=None,
            end_date=None,
            vol_limit=None,
            tau_min=None,
            tau_max=None,
            pilot_paths=None,
            pilot_steps=None,
            full_steps=None,
            vol_bins=None,
            seed=None,
        )
        config = build_config(args, {"bounds": {"aa": [0.005, 8.0]}})
        calibrator = HestonCalibrator(config)
        self.assertEqual(config.bounds["aa"], (0.005, 8.0))
        self.assertEqual(len(calibrator.bounds_sequence()), 6)

    def test_correlated_noise_mode_keeps_rho_as_parameter(self):
        args = Namespace(
            database=None,
            threshold_pair=None,
            markets=None,
            start_date=None,
            end_date=None,
            vol_limit=None,
            tau_min=None,
            tau_max=None,
            pilot_paths=None,
            pilot_steps=None,
            full_steps=None,
            vol_bins=None,
            seed=None,
        )
        config = build_config(args, {"correlated_noise": True})
        calibrator = HestonCalibrator(config)
        self.assertIn("rho", calibrator.parameter_names())
        self.assertEqual(len(calibrator.bounds_sequence()), 7)


if __name__ == "__main__":
    unittest.main()
