import unittest
from argparse import Namespace
from contextlib import closing
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import calibrate_heston
from scripts.calibrate_heston import build_config, plot_market_comparisons
from stabilvol.heston import CalibrationConfig, HestonCalibrator, HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.calibration import _heston_worker_objective, moments_frame, table_name_for_thresholds
from stabilvol.heston.paper_reproduction import count_fortran_hitting_events
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

    def test_daily_returns_are_log_state_increments_before_resets(self):
        params = HestonParams(reset_threshold=-1e6)
        result = simulate_modified_heston(
            params, SimulationConfig(n_paths=4, n_steps=30, seed=42),
        )
        previous_x = np.vstack((np.full((1, 4), params.start), result.x[:-1]))
        np.testing.assert_allclose(result.returns.to_numpy(), result.x - previous_x)

    def test_crossing_increment_is_recorded_before_state_reset(self):
        result = simulate_modified_heston(
            HestonParams(reset_threshold=0.0),
            SimulationConfig(n_paths=16, n_steps=2, seed=42),
        )
        crossed = result.returns.iloc[0].to_numpy() < 0
        self.assertTrue(crossed.any())
        np.testing.assert_allclose(result.x[0, crossed], 0.0)

    def test_correlated_noise_is_available(self):
        params = HestonParams(cc=0.01, rho=0.75)
        config = SimulationConfig(n_paths=2000, n_steps=3, seed=7, correlated_noise=True)
        result = simulate_modified_heston(params, config, keep_shocks=True)
        corr = np.corrcoef(result.price_shocks.ravel(), result.variance_shocks.ravel())[0, 1]
        self.assertGreater(corr, 0.6)

    def test_sampled_returns_use_endpoint_states_after_burn_in(self):
        params = HestonParams(reset_threshold=-1e6)
        fine = simulate_modified_heston(
            params, SimulationConfig(n_paths=4, n_steps=23, seed=17),
        )
        sampled = simulate_modified_heston(
            params, SimulationConfig(n_paths=4, n_steps=5, seed=17,
                                     sample_every=4, burn_in_steps=3,
                                     return_mode="sampled_x"),
        )
        observed = fine.x[6::4]
        np.testing.assert_allclose(sampled.x, observed)
        np.testing.assert_allclose(
            sampled.returns.to_numpy(),
            np.diff(np.vstack((fine.x[2:3], observed)), axis=0),
        )

    def test_sampled_returns_include_reset_jump(self):
        params = HestonParams(reset_threshold=0.0)
        fine = simulate_modified_heston(
            params, SimulationConfig(n_paths=16, n_steps=6, seed=42),
        )
        sampled = simulate_modified_heston(
            params, SimulationConfig(n_paths=16, n_steps=3, seed=42,
                                     sample_every=2, return_mode="sampled_x"),
        )
        np.testing.assert_allclose(
            sampled.returns.to_numpy(),
            np.diff(np.vstack((np.zeros((1, 16)), fine.x[1::2])), axis=0),
        )


class HestonCountingTests(unittest.TestCase):
    def test_fortran_counter_includes_crossing_return_and_breaks_on_missing_day(self):
        values = np.array([[0.0], [-0.1], [0.2], [-2.0]])
        events = count_fortran_hitting_events(values, 1.0, tau_max=30)
        self.assertEqual(int(events.iloc[0].FHT), 3)
        self.assertEqual(int(events.iloc[0].volatility_observations), 3)
        self.assertGreater(float(events.iloc[0].Volatility), 0.9)
        values[2, 0] = np.nan
        self.assertTrue(count_fortran_hitting_events(
            values, 1.0, tau_max=30, missing_policy="break").empty)

    def test_clean_fortran_counter_resets_after_discarded_crossing(self):
        values = np.array([[0.0], [0.0], [0.0], [0.0], [-2.0],
                           [0.0], [0.0], [0.0], [-2.0]])
        faithful = count_fortran_hitting_events(values, 1.0, tau_max=3)
        clean = count_fortran_hitting_events(values, 1.0, tau_max=3, reset_on_discard=True)
        self.assertTrue(faithful.empty)
        self.assertEqual(clean.FHT.tolist(), [3])

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
    def test_simple_return_transform_preserves_index_and_values(self):
        frame = pd.DataFrame({"a": [0.0, np.log(1.1)]},
                             index=pd.date_range("2000-01-01", periods=2))
        calibrator = HestonCalibrator(CalibrationConfig(synthetic_return_transform="simple"))
        transformed = calibrator.observed_simulated_returns(frame)
        pd.testing.assert_index_equal(transformed.index, frame.index)
        np.testing.assert_allclose(transformed.a, [0.0, 0.1])

    def test_conditional_fht_objective_distinguishes_conditional_durations(self):
        config = CalibrationConfig(
            root=".", loss_metric="conditional_fht", tau_min=2, tau_max=10,
            conditional_vol_quantiles=(0.0, 0.5, 1.0),
            min_empirical_bin_events=1, min_simulated_bin_events=1,
            event_count_weight=0.0,
        )
        calibrator = HestonCalibrator(config)
        empirical = pd.DataFrame({"Volatility": [0.1, 0.2, 0.3, 0.4], "FHT": [2, 2, 8, 8]})
        reversed_durations = pd.DataFrame({"Volatility": [0.1, 0.2, 0.3, 0.4], "FHT": [8, 8, 2, 2]})
        target = calibrator.prepare_conditional_target(empirical)
        self.assertAlmostEqual(calibrator.conditional_loss(target, empirical), 0.0)
        self.assertGreater(calibrator.conditional_loss(target, reversed_durations), 0.0)
        self.assertGreater(calibrator.conditional_loss(target, empirical.iloc[:2]), 0.0)

    def test_nonzero_return_distribution_excludes_exact_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            returns = pd.DataFrame({"a": [0.0, -0.1, 0.1, 0.2]},
                                   index=pd.date_range("2000-01-01", periods=4))
            returns.to_pickle(root / "data" / "interim" / "UN.pickle")
            calibrator = HestonCalibrator(CalibrationConfig(root=root))
            target = calibrator.empirical_return_distribution_target("UN")
            self.assertAlmostEqual(target.exact_zero_fraction, 0.25)
            self.assertAlmostEqual(calibrator.return_distribution_loss(returns, target), 0.0)
            self.assertAlmostEqual(calibrator.return_distribution_loss(
                pd.DataFrame({"a": [-0.1, 0.1, 0.2]}), target), 0.0)
            self.assertGreater(calibrator.return_distribution_loss(returns * 2, target), 0.0)

    def test_return_objective_uses_configured_sampling_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            params = HestonParams(reset_threshold=-1e6)
            simulation = simulate_modified_heston(
                params, SimulationConfig(n_paths=8, n_steps=40, seed=17,
                                         sample_every=3, burn_in_steps=5,
                                         return_mode="sampled_x", column_prefix="UN"),
            )
            simulation.returns.to_pickle(root / "data" / "interim" / "UN.pickle")
            config = CalibrationConfig(root=root, markets=("UN",), loss_metric="return_moments",
                                       sampling_interval_steps=3, sampling_burn_in_steps=5,
                                       sampling_return_mode="sampled_x")
            loss, simulated = HestonCalibrator(config).evaluate_params(
                "UN", params, n_paths=8, n_steps=40, seed=17,
            )
            self.assertAlmostEqual(loss, 0.0)
            pd.testing.assert_frame_equal(simulated, simulation.returns)

    def test_sampled_fit_validates_with_independent_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            params = HestonParams(reset_threshold=-1e6)
            empirical = simulate_modified_heston(
                params, SimulationConfig(n_paths=8, n_steps=40, seed=3,
                                         sample_every=2, return_mode="sampled_x"),
            ).returns
            empirical.to_pickle(root / "data" / "interim" / "UN.pickle")
            config = CalibrationConfig(root=root, markets=("UN",), loss_metric="return_moments",
                                       pilot_max_paths=8, pilot_n_steps=20, full_n_steps=40,
                                       sampling_interval_steps=2,
                                       sampling_return_mode="sampled_x")
            calibrator = HestonCalibrator(config)
            result = calibrator.calibrate_market("UN", maxiter=0, popsize=1,
                                                workers=1, validate_full=True,
                                                progress=False)
            seed = config.seed + sum((index + 1) * ord(char) for index, char in enumerate("UN"))
            expected, _ = calibrator.evaluate_params(
                "UN", result.params, n_paths=8, n_steps=40, seed=seed + 100_000,
            )
            self.assertAlmostEqual(result.validation_loss, expected)

    def test_staged_run_saves_both_fits_on_small_synthetic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            returns = simulate_modified_heston(
                HestonParams(), SimulationConfig(n_paths=16, n_steps=300, seed=5, store_state=False),
            ).returns
            returns.to_pickle(root / "data" / "interim" / "UN.pickle")
            config_path = root / "staged.json"
            config_path.write_text(json.dumps({
                "markets": ["UN"], "threshold_pairs": [[-0.1, -1.5]],
                "empirical_source": "returns", "loss_metric": "mfht_curve",
                "return_loss_weight": 2.0, "pilot_max_paths": 16,
                "pilot_n_steps": 300, "full_n_steps": 300, "n_vol_bins": 4,
                "min_empirical_bin_events": 1, "min_simulated_bin_events": 1,
                "run": {"staged": True, "return_maxiter": 0, "return_popsize": 1,
                        "maxiter": 0, "popsize": 1, "workers": 1,
                        "skip_full_validation": True, "output_dir": "results", "figure_dir": "figures"},
            }), encoding="utf-8")
            with patch.object(calibrate_heston, "PROJECT_ROOT", root), \
                 patch.object(sys, "argv", ["calibrate_heston.py", "--config-json", str(config_path)]), \
                 redirect_stdout(io.StringIO()):
                calibrate_heston.main()
            stage1 = pd.read_csv(root / "results" / "heston_calibration_stage1_parameters.csv")
            stage2 = pd.read_csv(root / "results" / "heston_calibration_parameters.csv")
            self.assertTrue(np.isfinite(stage1.loc[0, "pilot_loss"]))
            self.assertTrue(np.isfinite(stage2.loc[0, "pilot_loss"]))
            self.assertAlmostEqual(stage2.loc[0, "threshold_sigma"],
                                   np.std(returns.to_numpy(), ddof=0))
            self.assertTrue((root / "figures" / "UN" / "UN_stage1_returns_pdf.png").is_file())
            self.assertTrue((root / "figures" / "UN" / "UN_m0p1_m1p5_mfht.csv").is_file())

    def test_return_target_and_loss_use_average_per_series_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            returns = pd.DataFrame({
                "a": [0.0, 2.0, 4.0],
                "b": [1.0, 1.0, 1.0],
                "short": [1.0, np.nan, np.nan],
            }, index=pd.date_range("2000-01-01", periods=3))
            returns.to_pickle(root / "data" / "interim" / "UN.pickle")
            calibrator = HestonCalibrator(CalibrationConfig(
                root=root, loss_metric="return_moments", min_empirical_observations=3,
            ))
            target = calibrator.empirical_return_target("UN")
            self.assertEqual(calibrator.market_shape("UN"), (3, 2))
            self.assertAlmostEqual(target.mean, 2.0)
            self.assertAlmostEqual(target.std, 2.0)
            self.assertAlmostEqual(calibrator.empirical_collective_sigma("UN"),
                                   np.std(returns[["a", "b"]].to_numpy(), ddof=0))
            self.assertEqual(target.n_series, 1)
            self.assertAlmostEqual(calibrator.return_moment_loss(returns[["a"]], target), 0.0)
            self.assertGreater(calibrator.return_moment_loss(returns[["a"]] * 2, target), 0.0)

    def test_fixed_empirical_sigma_is_used_for_both_event_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "interim").mkdir(parents=True)
            returns = pd.DataFrame(
                {"a": [0.0, -0.6, -0.7, -1.6, 0.0]},
                index=pd.date_range("2000-01-01", periods=5),
            )
            returns.to_pickle(root / "data" / "interim" / "UN.pickle")
            config = CalibrationConfig(
                root=root, empirical_source="returns", threshold_sigma=1.0,
                threshold_pairs=((-0.5, -1.5),), std_normalization=True,
            )
            calibrator = HestonCalibrator(config)
            empirical = calibrator.load_empirical_events("UN", (-0.5, -1.5))
            simulated = calibrator.count_simulated_events(returns, (-0.5, -1.5), "UN")
            pd.testing.assert_frame_equal(empirical, simulated)
            self.assertEqual(int(empirical.iloc[0]["FHT"]), 3)
            calibrator.config = replace(config, threshold_sigma=0.025)
            with patch("stabilvol.heston.calibration.StabilVolter") as counter:
                with patch.object(calibrator, "_quiet_stabilvol", return_value=empirical):
                    calibrator.count_simulated_events(returns, (-0.1, -1.5), "UN")
                self.assertAlmostEqual(counter.call_args.kwargs["start_level"], -0.0025)
                self.assertAlmostEqual(counter.call_args.kwargs["end_level"], -0.0375)
                self.assertFalse(counter.call_args.kwargs["std_normalization"])

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
