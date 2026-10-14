import json
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from stabilvol.heston.simulation import (HestonParams, SimulationConfig, simulate_modified_heston,
                                        cir_exact_step, daily_parameters)
from stabilvol.heston.corrected import (CorrectedConfig, count_episodes, make_target, bin_events,
    score_curve, make_constraints, return_features, score_constraints)
from stabilvol.heston.paper_reproduction import count_fortran_hitting_events
from stabilvol.heston.run_io import immutable_manifest, verify_artifacts, file_digest, digest
from stabilvol.heston.search import differential_search, rank
from stabilvol.heston.workflow import (stock_split, comparison_gate, EvaluationContext,
    prepare, context, initial_vectors, export_evaluation, load_verified_comparison)
from stabilvol.heston.run_io import atomic_json

TEST_TEMP = Path(__file__).resolve().parents[1] / "tmp"
TEST_TEMP.mkdir(exist_ok=True)


class ToyObjective:
    def __call__(self, vector):
        loss = float(np.sum(np.asarray(vector)**2))
        violation = max(0.0, float(vector[0]) - .2)
        return {"valid": True, "feasible": violation == 0, "violation": violation, "mfht_loss": loss}


class SlowObjective(ToyObjective):
    def __call__(self, vector):
        time.sleep(.005)
        return super().__call__(vector)


class CorrectedSimulationTests(unittest.TestCase):
    def test_accumulation_excludes_resets_and_preserves_crash(self):
        params = HestonParams(reset_threshold=-.001, cc=.2)
        cfg = SimulationConfig(n_paths=12, n_steps=20, sample_every=4, seed=44,
            return_mode="accumulated_pre_reset", separate_streams=True, burn_in_steps=8)
        coarse = simulate_modified_heston(params, cfg)
        fine = simulate_modified_heston(params, replace(cfg, n_steps=80, sample_every=1, return_mode="pre_reset_step"))
        np.testing.assert_allclose(coarse.returns, fine.returns.to_numpy().reshape(20, 4, 12).sum(1), atol=1e-14)
        np.testing.assert_array_equal(coarse.reset_counts, fine.reset_counts.reshape(20, 4, 12).sum(1))
        self.assertGreater(coarse.reset_counts.sum(), 0)
        np.testing.assert_allclose(coarse.log_prices, np.log(100) + coarse.returns.cumsum())
        endpoint = simulate_modified_heston(params, replace(cfg, return_mode="sampled_x"))
        self.assertGreater(np.max(endpoint.returns.to_numpy() - coarse.returns.to_numpy()), .001)

    def test_exact_cir_conditional_moments(self):
        for v0 in (0., .01, .3):
            k, theta, xi, dt, n = 1.5, .012, .8, .02, 150000
            sample = cir_exact_step(np.full(n, v0), k, theta, xi, dt, np.random.default_rng(13))
            decay = np.exp(-k * dt)
            expected = theta + (v0 - theta) * decay
            variance = v0 * xi**2 * decay * (1-decay)/k + theta * xi**2 * (1-decay)**2/(2*k)
            self.assertTrue(np.all(sample >= 0))
            self.assertLess(abs(sample.mean()-expected), 6*np.sqrt(variance/n))
            self.assertAlmostEqual(sample.var()/variance, 1, delta=.07)

    def test_stationary_variance_and_separate_price_stream(self):
        cfg = SimulationConfig(n_paths=30000, n_steps=1, seed=18, variance_scheme="cir_exact", stationary_variance=True)
        first = simulate_modified_heston(HestonParams(aa=2, bb=.01, cc=.2), cfg, keep_shocks=True)
        second = simulate_modified_heston(HestonParams(aa=2, bb=.02, cc=.9), cfg, keep_shocks=True)
        self.assertAlmostEqual(first.variance.mean(), .01, delta=.0003)
        self.assertAlmostEqual(first.variance.var(), .01*.2**2/4, delta=.00001)
        np.testing.assert_array_equal(first.price_shocks, second.price_shocks)
        self.assertEqual(set(first.stream_seeds), {"price", "variance", "observation", "initial"})
        with self.assertRaises(ValueError):
            simulate_modified_heston(HestonParams(), replace(cfg, correlated_noise=True))

    def test_daily_conversion_and_deterministic_variance(self):
        old = HestonParams()
        new = daily_parameters(old, .05, 20)
        for name in ("a", "b", "aa", "bb", "cc", "vstart"):
            self.assertAlmostEqual(getattr(new, name), .05*getattr(old, name))
        self.assertEqual(new.dt, .05)
        actual = cir_exact_step(np.array([0., .1]), 2., .02, 0., .1, np.random.default_rng(1))
        np.testing.assert_allclose(actual, .02+(np.array([0., .1])-.02)*np.exp(-.2))

    def test_activity_accumulates_unquoted_returns(self):
        cfg = SimulationConfig(n_paths=10, n_steps=100, seed=77, variance_scheme="cir_exact",
            return_mode="accumulated_pre_reset", stationary_variance=True)
        full = simulate_modified_heston(HestonParams(cc=.2), cfg)
        stale = simulate_modified_heston(HestonParams(cc=.2), replace(cfg, activity_transitions=(1., 1.)))
        x, y = full.returns.to_numpy(), stale.returns.to_numpy()
        self.assertGreater(np.mean(y == 0), .4)
        for path in range(10):
            active = np.flatnonzero(y[:, path] != 0)
            np.testing.assert_allclose(np.cumsum(x[:, path])[active], np.cumsum(y[:, path])[active], atol=1e-12)


class EpisodeTests(unittest.TestCase):
    def test_overlong_episode_does_not_disable_path(self):
        values = np.array([0.]*302 + [-.1] + [0., 0., -.1]*5)[:, None]
        result = count_episodes(values, CorrectedConfig(.02))
        self.assertEqual(len(result.events), 5)
        self.assertEqual((result.episodes.status == "overlong").sum(), 1)
        np.testing.assert_array_equal(result.events.FHT, result.events.end_step-result.events.start_step)

    def test_short_missing_unfinished_and_local_sd(self):
        values = np.array([0., -.1, 0., -.01, -.1, 0., np.nan, 0., -.01])[:, None]
        result = count_episodes(values, CorrectedConfig(.02))
        self.assertEqual(result.episodes.status.tolist(), ["too_short", "accepted", "missing", "unfinished"])
        self.assertAlmostEqual(result.events.Volatility.iloc[0], np.std([-.01, -.1]))

    def test_agrees_with_clean_fortran_for_ordinary_returns(self):
        values = np.random.default_rng(4).normal(0, .02, (300, 8))
        values[33, 2] = np.nan
        config = CorrectedConfig(.02)
        new = count_episodes(values, config).events.reset_index(drop=True)
        old = count_fortran_hitting_events(values, .02, reset_on_discard=True, missing_policy="break")
        for key in ("FHT", "path", "start_step", "end_step", "Volatility"):
            np.testing.assert_allclose(new[key], old[key], atol=1e-12)


class TargetTests(unittest.TestCase):
    def setUp(self):
        paths = np.repeat(np.arange(30), 30)
        self.events = pd.DataFrame({"path": paths, "Volatility": np.tile(np.linspace(.001, .039, 30), 30),
            "FHT": np.tile(np.r_[np.arange(2, 17), np.arange(16, 1, -1)], 30)})
        self.config = CorrectedConfig(.02, min_events=40, min_stocks=10, bootstraps=30)

    def test_frozen_edges_bootstrap_and_coverage_penalty(self):
        target = make_target(self.events, 30, self.config)
        self.assertTrue(np.all(target.counts >= 40))
        self.assertAlmostEqual(target.weights.sum(), 1)
        for region in np.unique(target.regions):
            self.assertAlmostEqual(target.weights[target.regions == region].sum(), 1/3)
        n, t = bin_events(self.events, target.edges)
        score = score_curve(target, n, t)
        self.assertEqual(score["mfht_loss"], 0)
        self.assertTrue(score["mfht_pass"])
        n[0], t[0] = 0, 0
        bad = score_curve(target, n, t)
        self.assertGreater(bad["mfht_loss"], 0)
        self.assertFalse(bad["mfht_pass"])
        np.testing.assert_array_equal(target.edges, make_target(self.events, 30, self.config).edges)

    def test_return_constraints_and_no_zero_atom_constraint(self):
        values = np.random.default_rng(3).normal(0, .02, (500, 30))
        values[:5, 0] = np.nan
        constraints = make_constraints(values, self.config)
        features, _, _ = return_features(values, constraints.tail_thresholds)
        self.assertTrue(score_constraints(features, constraints)["feasible"])
        self.assertNotIn("zero_fraction", constraints.tolerances)
        features["mean"] += .1
        self.assertFalse(score_constraints(features, constraints)["feasible"])

    def test_split_stable_under_reordering(self):
        names = [f"stock-{i}" for i in range(100)]
        a, b = stock_split(names), stock_split(names[::-1])
        for split in a:
            if split != "full":
                self.assertEqual({names[i] for i in a[split]}, {names[::-1][i] for i in b[split]})
        self.assertEqual([len(a[x]) for x in ("train", "selection", "heldout")], [70, 15, 15])


class RecoveryTests(unittest.TestCase):
    def test_manifest_rejects_stale_artifacts(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            path = Path(tmp)
            (path / "curve.csv").write_text("a,b\n1,2\n")
            spec = {"artifacts": {"curve.csv": file_digest(path / "curve.csv")}}
            immutable_manifest(path / "manifest.json", spec)
            verify_artifacts(path)
            with self.assertRaises(ValueError):
                immutable_manifest(path / "manifest.json", {"different": True})
            (path / "curve.csv").write_text("stale")
            with self.assertRaises(ValueError):
                verify_artifacts(path)

    def test_feasibility_precedes_fit(self):
        self.assertLess(rank({"feasible": True, "mfht_loss": 99}),
                        rank({"feasible": False, "mfht_loss": .001, "violation": .001}))

    def test_mid_generation_resume_and_parallel_are_identical(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            base = dict(bounds=[(-1., 1.), (-1., 1.)], fingerprint="test", seed=17, maxiter=3, popsize=2)
            serial = differential_search(ToyObjective(), output=Path(tmp)/"serial", **base)
            partial = differential_search(SlowObjective(), output=Path(tmp)/"resume", time_limit=.001, **base)
            self.assertFalse(partial["complete"])
            resumed = differential_search(SlowObjective(), output=Path(tmp)/"resume", resume=True, **base)
            parallel = differential_search(ToyObjective(), output=Path(tmp)/"parallel", workers=2, **base)
            for field in ("population", "results", "rng", "evaluations", "archive"):
                self.assertEqual(digest(serial[field]), digest(resumed[field]))
                self.assertEqual(digest(serial[field]), digest(parallel[field]))
            with self.assertRaises(ValueError):
                differential_search(ToyObjective(), output=Path(tmp)/"resume", resume=True,
                    **{**base, "fingerprint": "changed"})

    def test_real_simulation_objective_parallel_and_masks(self):
        from concurrent.futures import ProcessPoolExecutor
        values = np.random.default_rng(81).normal(0, .02, (200, 24))
        values[:20, 0] = np.nan
        values[80:90, 1] = np.nan
        config = CorrectedConfig(.02, burn_in=5, bootstraps=20, min_events=10, min_stocks=2, seeds=(1001,))
        events = count_episodes(values, config).events
        target = make_target(events, 24, config)
        constraints = make_constraints(values, config)
        params = daily_parameters(HestonParams(cc=.2), .05)
        names = ("a", "b", "aa", "bb", "cc")
        ctx = EvaluationContext(values, target, constraints, config, params, names)
        vector = np.log([getattr(params, name) for name in names])
        details, observed = ctx.realization(vector, 1001)
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(values))
        expected = ctx(vector)
        with ProcessPoolExecutor(2) as pool:
            parallel = list(pool.map(ctx, [vector, vector]))
        self.assertEqual(digest(expected), digest(parallel[0]))
        self.assertEqual(digest(expected), digest(parallel[1]))

    def test_numerical_gate_rejects_material_grid_changes(self):
        def sample(mean):
            return {"valid": True, "realizations": [
                {"counts": [50]*3, "totals": [50*mean]*3,
                 "features": {"stock_sd": .02, "iqr": .02}} for _ in range(5)]}
        self.assertTrue(comparison_gate(sample(10), sample(10.1), 30)["pass"])
        self.assertFalse(comparison_gate(sample(10), sample(20), 30)["pass"])

    def test_prepare_context_export_and_selected_parameter_guard(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            root = Path(temporary)
            (root / "data/interim").mkdir(parents=True)
            source = root / "source/UN"
            source.mkdir(parents=True)
            values = np.random.default_rng(2).normal(0, .02, (3100, 30))
            frame = pd.DataFrame(values, index=pd.bdate_range("2000-01-01", periods=3100))
            frame.to_pickle(root / "data/interim/UN.pickle")
            params = HestonParams(cc=.2)
            pd.DataFrame([{**params.to_dict(), "market": "UN"}]).to_csv(source / "heston_calibration_parameters.csv", index=False)
            atomic_json(source / "trial_config.json", {"threshold_sigma": .02, "sampling_interval_steps": 1})
            directory = prepare(root, root / "runs", "UN", root / "source", {"bootstraps": 20, "min_stocks": 2, "min_events": 300})
            ctx, doc = context(root, directory, seeds=(1001,))
            ctx = replace(ctx, values=ctx.values[:150], paths=5, config=replace(ctx.config, burn_in=5))
            vector = initial_vectors(ctx, doc["specification"])[0]
            folder = directory / "baseline/heldout"
            atomic_json(folder.parent / "validation.json", {"selected": {"vector": vector}})
            export_evaluation(ctx, vector, folder, doc["fingerprint"], root, (1001,))
            comparison = load_verified_comparison(folder, directory, root)
            self.assertEqual(len(comparison), len(ctx.target.means))
            atomic_json(folder.parent / "validation.json", {"selected": {"vector": vector + .1}})
            with self.assertRaises(ValueError):
                load_verified_comparison(folder, directory, root)


if __name__ == "__main__":
    unittest.main()
