import unittest
from argparse import Namespace
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.calibrate_heston import build_config, plot_market_comparisons
from stabilvol.heston import CalibrationConfig, HestonCalibrator, HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.calibration import _heston_worker_objective, moments_frame, table_name_for_thresholds
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

    def test_simulated_count_uses_configured_std_normalization(self):
        data = pd.DataFrame({"a": [0.0, -0.6, -1.6]}, index=pd.date_range("2000-01-01", periods=3))
        config = CalibrationConfig(root=".", threshold_pairs=((-0.5, -1.5),), count_method="pandas", std_normalization=False)
        calibrator = HestonCalibrator(config)

        with patch("stabilvol.heston.calibration.StabilVolter") as stabilvolter:
            stabilvolter.return_value.get_stabilvol.return_value = pd.DataFrame()
            calibrator.count_simulated_events(data, (-0.5, -1.5), "SIM")

        self.assertFalse(stabilvolter.call_args.kwargs["std_normalization"])

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
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "fht.sqlite"
            config = CalibrationConfig(root=directory, database_path=database)
            events = pd.DataFrame({
                "Volatility": [0.1], "FHT": [3],
                "start": ["2001-01-01"], "end": ["2001-01-04"], "Market": ["UN"],
            })
            with closing(sqlite3.connect(database)) as connection:
                for pair in config.threshold_pairs:
                    events.to_sql(table_name_for_thresholds(*pair), connection, index=False)
            grid = HestonCalibrator(config).load_empirical_grid("UN")
            self.assertEqual(len(grid), 4)
            self.assertTrue(all(len(frame) == 1 for frame in grid.values()))

    def test_missing_database_is_reported_without_creating_one(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.sqlite"
            calibrator = HestonCalibrator(CalibrationConfig(root=directory, database_path=database))
            with self.assertRaisesRegex(FileNotFoundError, "empirical FHT database not found"):
                calibrator.load_empirical_grid("UN")
            self.assertFalse(database.exists())

    def test_loss_is_finite_for_valid_small_simulation(self):
        config = CalibrationConfig(
            root=".", threshold_pairs=((-0.5, -1.5),), pilot_max_paths=4,
            pilot_n_steps=200, n_vol_bins=5,
            min_empirical_bin_events=1, min_simulated_bin_events=1,
        )
        calibrator = HestonCalibrator(config)
        simulation = simulate_modified_heston(
            HestonParams(),
            SimulationConfig(n_paths=4, n_steps=200, seed=5, store_state=False, correlated_noise=True),
        )
        empirical = {(-0.5, -1.5): calibrator.count_simulated_events(simulation.returns, (-0.5, -1.5), "UN")}
        loss = calibrator.grid_loss(empirical, simulation.returns, "UN")
        self.assertEqual(loss, 0.0)

    def test_distribution_loss_uses_fht_not_volatility(self):
        calibrator = HestonCalibrator(CalibrationConfig(root=".", tau_min=2, tau_max=5))
        empirical = pd.DataFrame({"Volatility": [0.1, 0.2, 0.3, 0.4], "FHT": [2, 3, 4, 5]})
        simulated = pd.DataFrame({"Volatility": [10.0, 20.0, 30.0, 40.0], "FHT": [2, 3, 4, 5]})
        self.assertEqual(calibrator.distribution_loss(empirical, simulated), 0.0)

    def test_mfht_objective_detects_reversed_curve_with_same_fht_distribution(self):
        config = CalibrationConfig(
            root=".", n_vol_bins=2, tau_min=2, tau_max=10,
            min_empirical_bin_events=1, min_simulated_bin_events=1,
            event_count_weight=0.0,
        )
        calibrator = HestonCalibrator(config)
        empirical = pd.DataFrame({"Volatility": [0.1, 0.12, 0.3, 0.32], "FHT": [2, 2, 8, 8]})
        reversed_curve = pd.DataFrame({"Volatility": [0.1, 0.12, 0.3, 0.32], "FHT": [8, 8, 2, 2]})
        target = calibrator.prepare_curve_target(empirical)
        self.assertEqual(calibrator.curve_loss(target, empirical), 0.0)
        self.assertGreater(calibrator.curve_loss(target, reversed_curve), 0.0)
        self.assertEqual(calibrator.distribution_loss(empirical, reversed_curve), 0.0)

    def test_missing_simulated_volatility_bin_is_penalized(self):
        config = CalibrationConfig(
            root=".", n_vol_bins=2, tau_min=2, tau_max=10,
            min_empirical_bin_events=1, min_simulated_bin_events=1,
            event_count_weight=0.0,
        )
        calibrator = HestonCalibrator(config)
        empirical = pd.DataFrame({"Volatility": [0.1, 0.12, 0.3, 0.32], "FHT": [2, 2, 8, 8]})
        simulated = pd.DataFrame({"Volatility": [0.1, 0.12], "FHT": [2, 2]})
        target = calibrator.prepare_curve_target(empirical)
        self.assertGreater(calibrator.curve_loss(target, simulated), 0.0)

    def test_mfht_calibration_reads_sqlite_and_matches_identical_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            params = HestonParams()
            seed = 5
            n_paths, n_steps = 16, 300
            simulation = simulate_modified_heston(
                params, SimulationConfig(n_paths=n_paths, n_steps=n_steps, seed=seed, store_state=False),
            )
            simulation.returns.to_pickle(root / "data" / "interim" / "UN.pickle")
            pair = (-0.5, -1.5)
            config = CalibrationConfig(
                root=root, database_path=Path("events.sqlite"), markets=("UN",),
                threshold_pairs=(pair,), n_vol_bins=4,
                pilot_max_paths=n_paths, pilot_n_steps=n_steps,
                min_empirical_bin_events=1, min_simulated_bin_events=1,
            )
            calibrator = HestonCalibrator(config)
            events = calibrator.count_simulated_events(simulation.returns, pair, "UN")
            self.assertGreater(len(events), 0)
            with closing(sqlite3.connect(root / "events.sqlite")) as connection:
                events.to_sql(table_name_for_thresholds(*pair), connection, index=False)
            loss, _ = calibrator.evaluate_params("UN", params, n_paths=n_paths, n_steps=n_steps, seed=seed)
            self.assertEqual(loss, 0.0)
            figure_dir = root / "figures"
            plot_market_comparisons(
                calibrator, "UN", params, figure_dir,
                n_paths=n_paths, n_steps=n_steps, seed=seed,
            )
            curve_file = figure_dir / "UN_m0p5_m1p5_mfht.csv"
            self.assertTrue(curve_file.is_file())
            comparison = pd.read_csv(curve_file)
            self.assertIn("simulated_mfht", comparison)
            self.assertIn("empirical_mfht", comparison)
            pilot = calibrator.calibrate_market(
                "UN", maxiter=0, popsize=1, workers=1,
                validate_full=False, progress=False,
            )
            self.assertTrue(np.isfinite(pilot.pilot_loss))

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
            std_normalization=None,
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
            std_normalization=None,
        )
        config = build_config(args, {"correlated_noise": True})
        calibrator = HestonCalibrator(config)
        self.assertIn("rho", calibrator.parameter_names())
        self.assertEqual(len(calibrator.bounds_sequence()), 7)

    def test_initial_params_are_loaded_for_optimizer_start(self):
        args = Namespace(
            database=None, threshold_pair=None, markets=None, start_date=None, end_date=None,
            vol_limit=None, tau_min=None, tau_max=None, pilot_paths=None, pilot_steps=None,
            full_steps=None, vol_bins=None, seed=None, std_normalization=None,
            correlated_noise=None,
        )
        config = build_config(args, {"initial_params": {"aa": 0.12}})
        self.assertAlmostEqual(config.initial_params.aa, 0.12)
        self.assertEqual(config.initial_params.b, config.base_params.b)

        calibrator = HestonCalibrator(config)
        with patch.object(calibrator, "load_empirical_grid", return_value={}), \
             patch.object(calibrator, "market_shape", return_value=(10, 4)), \
             patch("stabilvol.heston.calibration.differential_evolution") as optimize:
            optimize.return_value = SimpleNamespace(
                x=calibrator.vector_from_params(config.initial_params),
                fun=0.2, message="done", success=True,
            )
            calibrator.calibrate_market("UN", validate_full=False, progress=False)
            np.testing.assert_allclose(
                optimize.call_args.kwargs["x0"],
                calibrator.vector_from_params(config.initial_params),
            )

    def test_parallel_optimizer_passes_only_vectors_to_workers(self):
        calibrator = HestonCalibrator(CalibrationConfig(root="."))
        with patch.object(calibrator, "load_empirical_grid", return_value={}), \
             patch.object(calibrator, "market_shape", return_value=(10, 4)), \
             patch("stabilvol.heston.calibration.mp.Pool") as pool_factory, \
             patch("stabilvol.heston.calibration.differential_evolution") as optimize:
            optimize.return_value = SimpleNamespace(
                x=calibrator.vector_from_params(HestonParams()),
                fun=0.2, message="done", success=True,
            )
            calibrator.calibrate_market("UN", workers=2, validate_full=False, progress=False)
            self.assertIs(optimize.call_args.args[0], _heston_worker_objective)
            self.assertIs(optimize.call_args.kwargs["workers"], pool_factory.return_value.__enter__.return_value.map)
            self.assertEqual(pool_factory.call_args.args[0], 2)

    def test_config_controls_std_normalization(self):
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
            std_normalization=None,
        )
        config = build_config(args, {"std_normalization": False})
        self.assertFalse(config.std_normalization)


if __name__ == "__main__":
    unittest.main()
