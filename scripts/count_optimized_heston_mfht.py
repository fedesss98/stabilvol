#!/usr/bin/env python3
"""Recount fitted Heston and real returns with one Fortran-style MFHT rule."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stabilvol.heston import HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.paper_reproduction import count_fortran_hitting_events, fortran_mfht_curve


def fortran_scale(values: np.ndarray, anomaly_limit: float) -> float:
    """Mean per-column population SD, clipping raw anomalies as calmG.f does."""
    clipped = np.where(np.abs(values) > anomaly_limit, 0.0, values)
    finite_counts = np.isfinite(clipped).sum(axis=0)
    if not np.any(finite_counts >= 2):
        raise ValueError("need at least one return series with two observations")
    std = np.nanstd(clipped[:, finite_counts >= 2], axis=0, ddof=0)
    sigma = float(np.mean(std))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid Fortran return scale")
    return sigma


def physical_x(curve: pd.DataFrame) -> pd.Series:
    return (curve["physical_volatility_lower"] + curve["physical_volatility_upper"]) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=Path("data/processed/heston_calibration/conditional_fortran_300"))
    parser.add_argument("--markets", nargs="+", choices=("UN", "UW", "LN", "JT"), default=["UN", "UW", "LN", "JT"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--paths", type=int, default=1071, help="Simulated paths per market")
    parser.add_argument("--steps", type=int, default=3030, help="Recorded returns per simulated path")
    parser.add_argument("--seed-offset", type=int, default=500000, help="Independent diagnostic seed offset")
    parser.add_argument("--sigma-mode", choices=("fitted", "fortran"), default="fitted",
                        help="fitted matches the optimizer's fixed threshold; fortran recomputes each data set's scale")
    args = parser.parse_args()
    if args.paths < 1 or args.steps < 2:
        parser.error("require paths >= 1 and steps >= 2")
    fit_dir = args.fit_dir if args.fit_dir.is_absolute() else PROJECT_ROOT / args.fit_dir
    output_dir = args.output_dir or fit_dir / "mfht_recount"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    reference = json.loads((PROJECT_ROOT / "configs/heston_fortran_reference.json").read_text(encoding="utf-8"))
    counter_defaults = reference["counter"]
    binning = reference["binning"]

    for market in args.markets:
        market_fit = fit_dir / market
        params_path = market_fit / "heston_calibration_parameters.csv"
        config_path = market_fit / "trial_config.json"
        if not params_path.is_file() or not config_path.is_file():
            parser.error(f"missing fitted parameters or trial config for {market} in {market_fit}; run optimization first")
        fit = pd.read_csv(params_path)
        selected = fit.loc[fit["market"] == market]
        if selected.empty:
            parser.error(f"no fitted row for {market} in {params_path}")
        row = selected.iloc[-1]
        params = HestonParams(**{field.name: float(row[field.name]) for field in fields(HestonParams)})
        setup = json.loads(config_path.read_text(encoding="utf-8"))
        if setup["count_method"] not in ("fortran", "fortran_clean"):
            parser.error(f"{market} fit must use a Fortran counter")
        if setup["threshold_pairs"] != [[-0.1, -1.5]]:
            parser.error(f"{market} fit must use the -0.1 to -1.5 crash thresholds")
        market_dir = output_dir / market
        market_dir.mkdir(parents=True, exist_ok=True)

        empirical_path = PROJECT_ROOT / "data/interim" / f"{market}.pickle"
        empirical = pd.read_pickle(empirical_path).loc[
            setup.get("start_date", "1980-01-01"):setup.get("end_date", "2022-07-01")
        ]
        eligible = empirical.notna().sum(axis=0) >= max(2, setup.get("min_empirical_observations", 0))
        empirical = empirical.loc[:, eligible]
        if empirical.empty or empirical.shape[1] == 0:
            raise ValueError(f"no eligible empirical returns for {market}")
        empirical_values = empirical.to_numpy(dtype=float, copy=False)
        simulation = simulate_modified_heston(
            params,
            SimulationConfig(
                n_paths=args.paths, n_steps=args.steps,
                sample_every=setup["sampling_interval_steps"],
                burn_in_steps=setup["sampling_burn_in_steps"],
                return_mode=setup["sampling_return_mode"],
                correlated_noise=setup.get("correlated_noise", False),
                seed=int(setup["seed"]) + args.seed_offset,
                store_state=False,
            ),
        )
        synthetic_values = simulation.returns.to_numpy(dtype=float, copy=False)
        if setup["synthetic_return_transform"] == "simple":
            synthetic_values = np.expm1(synthetic_values)
        elif setup["synthetic_return_transform"] != "log":
            parser.error(f"unknown return transform for {market}")

        anomaly_limit = counter_defaults["anomaly_limit"]
        if args.sigma_mode == "fitted":
            real_sigma = synthetic_sigma = float(setup["threshold_sigma"])
        else:
            real_sigma = fortran_scale(empirical_values, anomaly_limit)
            synthetic_sigma = fortran_scale(synthetic_values, anomaly_limit)
        counter = {
            "start_sigma": counter_defaults["start_sigma"],
            "start_upper_sigma": counter_defaults["start_upper_sigma"],
            "end_sigma": counter_defaults["end_sigma"],
            "anomaly_limit": anomaly_limit,
            "tau_min": int(setup.get("tau_min", 2)),
            "tau_max": int(setup["tau_max"]),
            "reset_on_discard": setup["count_method"] == "fortran_clean",
        }
        summaries = {}
        curves = {}
        for label, values, sigma in (("empirical", empirical_values, real_sigma),
                                     ("synthetic", synthetic_values, synthetic_sigma)):
            events = count_fortran_hitting_events(
                values, sigma, direction="crash",
                missing_policy="break" if label == "empirical" else "error",
                **counter,
            )
            curve = fortran_mfht_curve(events, **binning)
            events.to_csv(market_dir / f"{label}_crash_events.csv.gz", index=False)
            curve.to_csv(market_dir / f"{label}_crash_mfht.csv", index=False)
            summaries[label] = {
                "paths": values.shape[1], "steps": values.shape[0],
                "sigma_bar": sigma, "events": len(events),
                "events_in_mfht_bins": int(curve["events"].sum()),
            }
            curves[label] = curve

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for ax, axis_name in zip(axes, ("reported", "physical")):
            for label, curve in curves.items():
                shown = curve.loc[curve["events"] >= 10]
                x = shown["volatility"] if axis_name == "reported" else physical_x(shown)
                ax.scatter(x, shown["mfht"], s=9, label=label.capitalize())
            ax.set(xlabel=f"{axis_name.capitalize()} local volatility", ylabel="MFHT (steps)",
                   title=f"{market}: {setup['sampling_interval_steps']}dt, tau <= {counter['tau_max']}")
            ax.grid(alpha=0.2)
            ax.legend()
        fig.savefig(market_dir / "empirical_vs_synthetic_mfht.png", dpi=180)
        plt.close(fig)
        summary = {
            "market": market, "fit_parameters": str(params_path), "fit_config": str(config_path),
            "parameters": asdict(params), "sigma_mode": args.sigma_mode,
            "counter": counter, "binning": binning,
            "synthetic_observation": {
                "sample_every": setup["sampling_interval_steps"],
                "burn_in_steps": setup["sampling_burn_in_steps"],
                "return_mode": setup["sampling_return_mode"],
                "return_transform": setup["synthetic_return_transform"],
                "seed": int(setup["seed"]) + args.seed_offset,
            },
            "missing_empirical_observations": int(np.isnan(empirical_values).sum()),
            "empirical_missing_policy": "break",
            "datasets": summaries,
        }
        (market_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{market}: empirical {summaries['empirical']['events']}, "
              f"synthetic {summaries['synthetic']['events']} events -> {market_dir}", flush=True)


if __name__ == "__main__":
    main()
