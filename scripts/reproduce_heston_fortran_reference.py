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

from stabilvol.heston import HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.paper_reproduction import (
    count_fortran_hitting_events, fortran_mfht_curve, market_return_scale,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/heston_fortran_reference.json")
    parser.add_argument("--paths", type=int, help="Override paths for a quick trial")
    parser.add_argument("--steps", type=int, help="Override recorded steps for a quick trial")
    parser.add_argument("--seed", type=int, help="Override NumPy seed")
    parser.add_argument("--output-dir", type=Path)
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
        "note": "NumPy RNG differs from Fortran; original calmG.f counts crashes only.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"sigma_bar={sigma_bar:.6g}; crash events={len(events)}; output={output_dir}")


if __name__ == "__main__":
    main()
