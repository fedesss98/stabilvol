#!/usr/bin/env python3
"""Run the supplied heston.f and calmG.f conventions from an explicit JSON config.

This reproduces the numerical rules, not the original ran2/Box-Muller random stream.
The supplied calmG.f counts crashes only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from stabilvol.heston.paper_reproduction import (
    count_fortran_hitting_events, fortran_mfht_curve, market_return_scale,
)


def count_empirical_market(
    market: str, empirical: dict, counter: dict, binning: dict, output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    returns_dir = Path(empirical["returns_dir"])
    if not returns_dir.is_absolute():
        returns_dir = PROJECT_ROOT / returns_dir
    returns_path = returns_dir / f"{market}.pickle"
    returns = pd.read_pickle(returns_path).loc[empirical["start_date"]:empirical["end_date"]]
    if returns.empty:
        raise ValueError(f"no returns for {market} in the requested date range")
    observed = returns.notna().sum(axis=0)
    returns = returns.loc[:, observed >= empirical["min_observations"]]
    if returns.shape[1] == 0:
        raise ValueError(f"no eligible return series for {market}")
    values = returns.to_numpy(dtype=float, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"infinite return in {returns_path}")
    # calmG.f clips anomalies only for threshold normalization and crossing tests.
    clipped = np.where(np.abs(values) > counter["anomaly_limit"], 0.0, values)
    path_sigma = np.nanstd(clipped, axis=0, ddof=0)
    path_sigma = path_sigma[np.isfinite(path_sigma)]
    sigma_bar = float(path_sigma.mean())
    if not np.isfinite(sigma_bar) or sigma_bar <= 0:
        raise ValueError(f"invalid pooled Fortran scale for {market}")
    events = count_fortran_hitting_events(
        values, sigma_bar, direction="crash", missing_policy=empirical["missing_policy"], **counter,
    )
    curve = fortran_mfht_curve(events, **binning)
    events.to_csv(output_dir / f"{market}_crash_events.csv.gz", index=False)
    curve.to_csv(output_dir / f"{market}_crash_mfht.csv", index=False)
    summary = {
        "returns_path": str(returns_path),
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "paths": int(returns.shape[1]),
        "days": int(returns.shape[0]),
        "missing_observations": int(np.isnan(values).sum()),
        "missing_policy": empirical["missing_policy"],
        "sigma_bar": sigma_bar,
        "events": len(events),
        "events_in_mfht_bins": int(curve["events"].sum()),
    }
    return curve, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/heston_fortran_reference.json")
    parser.add_argument("--paths", type=int, help="Override paths for a quick trial")
    parser.add_argument("--steps", type=int, help="Override recorded steps for a quick trial")
    parser.add_argument("--seed", type=int, help="Override NumPy seed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--markets", nargs="+", help="Empirical markets; default from config")
    parser.add_argument("--synthetic-only", action="store_true", help="Skip empirical return counting")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    params = HestonParams(**config["parameters"])
    simulation_options = dict(config["simulation"])
    zero_first_return = simulation_options.pop("zero_first_return")
    for option, value in (("n_paths", args.paths), ("n_steps", args.steps), ("seed", args.seed)):
        if value is not None:
            simulation_options[option] = value
    # The supplied Fortran writes one pre-reset Euler increment per row.
    if simulation_options["sample_every"] != 1 or simulation_options["burn_in_steps"] != 0:
        parser.error("the reference run requires sample_every=1 and burn_in_steps=0")
    if simulation_options["return_mode"] != "pre_reset_step" or not zero_first_return:
        parser.error("the reference run requires pre_reset_step and zero_first_return=true")
    if simulation_options["n_paths"] < 1 or simulation_options["n_steps"] < 2:
        parser.error("require at least one path and two recorded steps")
    simulation = simulate_modified_heston(
        params, SimulationConfig(**simulation_options, store_state=False),
    )
    returns = simulation.returns.to_numpy(copy=True)
    returns[0, :] = 0.0  # heston.f overwrites the first saved increment of each path.
    counter = config["counter"]
    if counter["anomaly_limit"] <= 0:
        parser.error("anomaly_limit must be positive")
    # calmG.f applies rit_anom during its first, per-path population-SD pass.
    scale_returns = np.where(np.abs(returns) > counter["anomaly_limit"], 0.0, returns)
    sigma_bar = market_return_scale(scale_returns, ddof=0)
    if counter["tau_min"] < 1 or counter["tau_max"] < counter["tau_min"]:
        parser.error("require 1 <= tau_min <= tau_max")
    events = count_fortran_hitting_events(
        returns, sigma_bar, direction="crash", **counter,
    )
    curve = fortran_mfht_curve(events, **config["binning"])

    output_dir = args.output_dir or Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "crash_events.csv.gz", index=False)
    curve.to_csv(output_dir / "crash_mfht.csv", index=False)

    empirical = config["empirical"]
    if empirical["missing_policy"] != "break":
        parser.error("real returns require missing_policy='break' to preserve calendar gaps")
    market_curves = {}
    market_summaries = {}
    if not args.synthetic_only:
        for market in (args.markets or empirical["markets"]):
            market_curves[market], market_summaries[market] = count_empirical_market(
                market, empirical, counter, config["binning"], output_dir,
            )
            print(f"{market}: {market_summaries[market]['events']} real crash events", flush=True)

    shown = curve.loc[curve["events"] >= 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    physical = (shown["physical_volatility_lower"] + shown["physical_volatility_upper"]) / 2
    for ax, x, label in zip(
        axes, (shown["volatility"], physical),
        ("Fortran reported volatility", "Physical local volatility"),
    ):
        ax.scatter(x, shown["mfht"], s=10)
        ax.set(xlabel=label, ylabel="MFHT (steps)")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "crash_mfht_axes.png", dpi=180)
    plt.close(fig)

    if market_curves:
        fig, axes = plt.subplots(len(market_curves), 2, figsize=(11, 3.5 * len(market_curves)),
                                 squeeze=False, layout="constrained")
        for row, (market, real_curve) in enumerate(market_curves.items()):
            real_shown = real_curve.loc[real_curve["events"] >= 10]
            for col, x_col in enumerate(("volatility", "physical")):
                ax = axes[row, col]
                for frame, label in ((shown, "Synthetic"), (real_shown, market)):
                    x = (frame["volatility"] if x_col == "volatility" else
                         (frame["physical_volatility_lower"] + frame["physical_volatility_upper"]) / 2)
                    ax.scatter(x, frame["mfht"], s=9, label=label)
                ax.set(xlabel="Fortran reported volatility" if col == 0 else "Physical local volatility",
                       ylabel="MFHT (steps)", title=f"{market}: same counter and binning")
                ax.legend()
                ax.grid(alpha=0.2)
        fig.savefig(output_dir / "synthetic_vs_empirical_crash_mfht.png", dpi=180)
        plt.close(fig)

    summary = {
        "source": ["old_code/heston.f", "old_code/calmG.f", "old_code/parm.dat", "old_code/parmG.dat"],
        "config": str(args.config.resolve()),
        "parameters": asdict(params),
        "simulation": {**simulation_options, "zero_first_return": zero_first_return},
        "counter": counter,
        "binning": config["binning"],
        "sigma_bar": sigma_bar,
        "events": len(events),
        "events_in_mfht_bins": int(curve["events"].sum()),
        "empirical": market_summaries,
        "note": "NumPy RNG differs from Fortran; original calmG.f counts crashes only. Empirical missing days break episodes; original Fortran input was dense. Empirical returns are simple price changes; synthetic returns are model log increments.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"sigma_bar={sigma_bar:.6g}; crash events={len(events)}; output={output_dir}")


if __name__ == "__main__":
    main()
