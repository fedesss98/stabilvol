#!/usr/bin/env python3
"""Audit fit provenance and run controlled Heston measurement ablations.

This does not refit parameters. All ablations retain the fitted empirical
thresholds; random zero insertion is a sensitivity test, not a market model.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from stabilvol.heston import CalibrationConfig, HestonCalibrator, HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.paper_reproduction import count_fortran_hitting_events


def load_fit(folder, market):
    setup = json.loads((folder / market / "trial_config.json").read_text())
    allowed = {f.name for f in fields(CalibrationConfig)}
    config = {k: v for k, v in setup.items() if k in allowed}
    for key in ("base_params", "initial_params"):
        if isinstance(config.get(key), dict):
            config[key] = HestonParams(**config[key])
    config["root"] = ROOT
    row = pd.read_csv(folder / market / "heston_calibration_parameters.csv").query("market == @market").iloc[-1]
    params = HestonParams(**{f.name: float(row[f.name]) for f in fields(HestonParams)})
    return setup, HestonCalibrator(CalibrationConfig(**config)), params, row


def binned(events, edges):
    counts, means = HestonCalibrator._binned_fht(events.Volatility.to_numpy(), events.FHT.to_numpy(), edges)
    return pd.DataFrame({"volatility": (edges[1:] + edges[:-1]) / 2, "events": counts, "mfht": means})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=ROOT / "data/processed/heston_calibration/conditional_mfht_joint_explore")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/heston_diagnosis")
    parser.add_argument("--markets", nargs="+", default=["UN", "LN"])
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3030)
    parser.add_argument("--seeds", nargs="+", type=int, default=[81001, 81002, 81003])
    args = parser.parse_args()
    if args.paths < 1 or args.steps < 2:
        parser.error("paths must be positive and steps must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    all_metrics = []
    common_support_metrics = []
    edges = np.linspace(0, 0.04, 81)
    fig, axes = plt.subplots(len(args.markets), 2, figsize=(13, 4 * len(args.markets)), squeeze=False, layout="constrained")
    for row_index, market in enumerate(args.markets):
        print(f"{market}: loading empirical data and fit", flush=True)
        setup, cal, params, fit = load_fit(args.fit_dir, market)
        out = args.output_dir / market
        out.mkdir(exist_ok=True)
        empirical = cal.load_empirical_returns(market)
        empirical_events = cal.load_empirical_events(market, (-0.1, -1.5))
        target = cal.prepare_curve_target(empirical_events)
        conditional = cal.prepare_conditional_target(empirical_events)
        return_target = cal.empirical_return_distribution_target(market)
        target_frame = cal.curve_comparison(target, empirical_events)
        target_frame.to_csv(out / "objective_bins.csv", index=False)
        actual = binned(empirical_events, edges)
        actual.to_csv(out / "empirical_curve.csv", index=False)
        info = {
            "fit": fit.to_dict(), "parameters": params.to_dict(),
            "empirical_shape": list(empirical.shape),
            "empirical_zero_fraction": return_target.exact_zero_fraction,
            "objective_bin_width": float(np.diff(target.edges)[0]),
            "objective_upper_volatility": float(target.edges[-1]),
            "objective_bins_below_0p04": int(np.sum(target.edges[:-1] < 0.04)),
            "conditional_region_edges": conditional.edges.tolist(),
            "feller_ratio": 2 * params.aa * params.bb / params.cc**2,
            "nominal_initial_variance_memory_after_burnin": float(np.exp(-params.aa * params.dt * setup["sampling_burn_in_steps"])),
            "carryover_event_fraction": float(np.mean(empirical_events.FHT != empirical_events.end_step - empirical_events.start_step)),
        }
        recount_file = args.fit_dir / "mfht_recount" / market / "summary.json"
        if recount_file.exists():
            recount = json.loads(recount_file.read_text())
            info["recount_parameter_mismatches"] = {
                key: {"fit": value, "recount": recount["parameters"].get(key)}
                for key, value in params.to_dict().items()
                if not np.isclose(value, recount["parameters"].get(key, np.nan), rtol=1e-10, atol=1e-14)
            }
            info["recount_source"] = recount["fit_parameters"]
        curves = {}

        def measure(label, returns, seed, variance=None):
            events = cal.count_simulated_events(returns, (-0.1, -1.5), market)
            curve = binned(events, edges)
            curve.to_csv(out / f"{label}_{seed}_curve.csv", index=False)
            curves.setdefault(label, []).append(curve)
            compare = binned(empirical_events, edges)
            valid = (compare.events >= 100) & (curve.events >= 30)
            vals = returns.to_numpy()
            finite = vals[np.isfinite(vals)]
            item = {
                "market": market, "experiment": label, "seed": seed,
                "conditional_loss": cal.conditional_loss(conditional, events),
                "mfht_loss": cal.curve_loss(target, events),
                "return_loss": cal.return_distribution_loss(returns, return_target),
                "events": len(events), "mean_return": float(finite.mean()),
                "mean_stock_std": float(returns.std(ddof=0).mean()),
                "fraction_return_gt_1": float(np.mean(finite > 1)),
                "fraction_return_lt_minus_0p5": float(np.mean(finite < -0.5)),
                "max_return": float(finite.max()),
                "physical_curve_rmse": float(np.sqrt(np.mean((curve.mfht[valid] - compare.mfht[valid])**2))),
                "covered_physical_bins": int(valid.sum()),
            }
            if variance is not None:
                v = variance.ravel()
                mean = v + params.aa * (params.bb - v) * params.dt
                sd = params.cc * np.sqrt(v * params.dt)
                item["variance_mean"] = float(v.mean())
                item["mean_first_proposal_negative_probability"] = float(np.mean(ndtr(-mean / np.maximum(sd, 1e-300))))
            all_metrics.append(item)
            return events

        # Isolate the old counter's carry-over using the identical empirical data.
        clean_events = count_fortran_hitting_events(empirical.to_numpy(), cal.config.threshold_sigma, missing_policy="break", reset_on_discard=True, tau_min=cal.config.tau_min, tau_max=cal.config.tau_max)
        binned(clean_events, edges).to_csv(out / "empirical_clean_counter_curve.csv", index=False)
        info["empirical_clean_counter_events"] = len(clean_events)
        for seed in args.seeds:
            print(f"{market}: seed {seed}, baseline and observation ablations", flush=True)
            sim_config = SimulationConfig(n_paths=args.paths, n_steps=args.steps, sample_every=setup["sampling_interval_steps"], burn_in_steps=setup["sampling_burn_in_steps"], return_mode=setup["sampling_return_mode"], seed=seed, store_state=True, correlated_noise=cal.config.correlated_noise)
            sim = simulate_modified_heston(params, sim_config)
            returns = cal.observed_simulated_returns(sim.returns)
            measure("fitted", returns, seed, sim.variance)
            if seed == args.seeds[0]:
                # Identical internal shocks: isolate artificial resets from
                # increments by recording every pre-reset Euler increment.
                internal = simulate_modified_heston(params, replace(
                    sim_config, n_steps=args.steps * sim_config.sample_every,
                    sample_every=1, return_mode="pre_reset_step", store_state=False))
                sums = internal.returns.to_numpy().reshape(args.steps, sim_config.sample_every, args.paths).sum(axis=1)
                reset_jump = sim.returns.to_numpy() - sums
                touched = np.abs(reset_jump) > 1e-8
                info["reset_affected_observation_fraction"] = float(touched.mean())
                info["positive_returns_over_100pct_at_reset_fraction"] = float(
                    np.mean(touched[returns.to_numpy() > 1])) if np.any(returns.to_numpy() > 1) else None
                without_jump = cal.observed_simulated_returns(pd.DataFrame(sums, index=returns.index, columns=returns.columns))
                measure("pre_reset_sum", without_jump, seed)
            # Independent zeros test sensitivity only; does not reproduce zero runs.
            rng = np.random.default_rng(seed + 1000000)
            with_zeros = returns.mask(rng.random(returns.shape) < return_target.exact_zero_fraction, 0.0)
            measure("random_zeros", with_zeros, seed)
            # Permute within each path, preserving its marginal distribution.
            shuffled = pd.DataFrame(rng.permuted(returns.to_numpy(), axis=0), index=returns.index, columns=returns.columns)
            measure("shuffled", shuffled, seed)
            # Hold observation duration and burn-in duration fixed when halving dt.
            fine = simulate_modified_heston(replace(params, dt=params.dt / 2), replace(sim_config, sample_every=2 * sim_config.sample_every, burn_in_steps=2 * sim_config.burn_in_steps))
            measure("half_dt", cal.observed_simulated_returns(fine.returns), seed)
        pooled_curves = {}
        for label in ("fitted", "half_dt", "random_zeros", "shuffled"):
            counts = np.stack([c.events for c in curves[label]])
            means = np.stack([c.mfht for c in curves[label]])
            total = counts.sum(axis=0)
            pooled = np.divide(np.nansum(counts * means, axis=0), total, out=np.full(len(total), np.nan), where=total > 0)
            pooled_curves[label] = (total, pooled)
        common = (actual.events.to_numpy() >= 100) & np.logical_and.reduce([
            n >= 30 * len(args.seeds) for n, _ in pooled_curves.values()])
        for label, (_, means) in pooled_curves.items():
            common_support_metrics.append({"market": market, "experiment": label,
                "common_bins": int(common.sum()),
                "rmse": float(np.sqrt(np.mean((means[common] - actual.mfht.to_numpy()[common])**2)))})
        for col in range(2):
            ax = axes[row_index, col]
            shown = actual.events >= 100
            ax.plot(actual.volatility[shown], actual.mfht[shown], color="black", label="Empirical", linewidth=2)
            labels = ("fitted", "random_zeros") if col == 0 else ("fitted", "shuffled", "half_dt")
            for label in labels:
                frames = curves[label]
                counts = np.stack([c.events for c in frames])
                means = np.stack([c.mfht for c in frames])
                total = counts.sum(axis=0)
                pooled = np.divide(np.nansum(counts * means, axis=0), total, out=np.full(len(total), np.nan), where=total > 0)
                shown = total >= 30 * len(args.seeds)
                ax.plot(actual.volatility[shown], pooled[shown], label=label.replace("_", " "))
            ax.set(title=f"{market}: {'observation sensitivity' if col == 0 else 'time order and numerical sensitivity'}", xlabel="Physical event volatility", ylabel="MFHT (observations)", ylim=(0, 110))
            ax.grid(alpha=0.2)
            ax.legend()
        report[market] = info
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        pd.DataFrame(all_metrics).to_csv(args.output_dir / "experiments.csv", index=False)
        pd.DataFrame(common_support_metrics).to_csv(args.output_dir / "common_support_rmse.csv", index=False)
    fig.savefig(args.output_dir / "diagnostics.png", dpi=160)
    plt.close(fig)
    print(f"Diagnostics saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
