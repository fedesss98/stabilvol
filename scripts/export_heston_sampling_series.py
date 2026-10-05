"""Recreate and save synthetic return series from a completed sampling-fit sweep."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stabilvol.heston import HestonParams, SimulationConfig, simulate_modified_heston


def fitted_parameters(path: Path, market: str) -> tuple[HestonParams, pd.Series]:
    table = pd.read_csv(path)
    if "market" in table:
        table = table.loc[table["market"] == market]
    if table.empty:
        raise ValueError(f"no fitted parameters for {market} in {path}")
    row = table.iloc[-1]
    values = {
        field.name: float(row[field.name])
        for field in fields(HestonParams)
        if field.name in row and pd.notna(row[field.name])
    }
    return HestonParams(**values), row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path,
                        default=Path("data/processed/heston_calibration/sampling_fit_sweep"))
    parser.add_argument("--intervals", type=int, nargs="+",
                        help="Intervals to export; default is every completed interval")
    parser.add_argument("--market", default="UN")
    parser.add_argument("--kind", choices=("plot", "validation", "both"), default="plot",
                        help="Plot seed/length, independent full-validation seed/length, or both")
    parser.add_argument("--states", action="store_true",
                        help="Also save sampled x and variance arrays; requires more memory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir if args.sweep_dir.is_absolute() else PROJECT_ROOT / args.sweep_dir
    if not sweep_dir.is_dir():
        parser.error(f"sweep folder not found: {sweep_dir}")
    if args.intervals is None:
        intervals = sorted(
            int(match.group(1))
            for path in sweep_dir.glob("config_interval_*.json")
            if (match := re.fullmatch(r"config_interval_(\d+)\.json", path.name))
            and (sweep_dir / f"interval_{match.group(1)}" / "heston_calibration_parameters.csv").is_file()
        )
    else:
        intervals = sorted(set(args.intervals))
    if not intervals or any(interval < 1 for interval in intervals):
        parser.error("no completed positive sampling intervals found")

    kinds = ("plot", "validation") if args.kind == "both" else (args.kind,)
    for interval in intervals:
        config_path = sweep_dir / f"config_interval_{interval}.json"
        interval_dir = sweep_dir / f"interval_{interval}"
        parameters_path = interval_dir / "heston_calibration_parameters.csv"
        if not config_path.is_file() or not parameters_path.is_file():
            raise FileNotFoundError(f"missing config or parameters for interval {interval}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        params, row = fitted_parameters(parameters_path, args.market)
        if int(data["sampling_interval_steps"]) != interval:
            raise ValueError(f"config interval disagrees with folder name: {config_path}")

        for kind in kinds:
            returns_path = interval_dir / f"{args.market}_synthetic_returns_{kind}.pkl"
            states_path = interval_dir / f"{args.market}_synthetic_states_{kind}.npz"
            metadata_path = interval_dir / f"{args.market}_synthetic_returns_{kind}_metadata.json"
            if returns_path.exists() and not args.overwrite and (not args.states or states_path.exists()):
                print(f"Already present: {returns_path}")
                continue
            if kind == "plot":
                full_plot = bool(data.get("run", {}).get("plot_full", False))
                n_paths = int(row["n_paths_full"] if full_plot else row["n_paths_pilot"])
                n_steps = int(row["n_steps_full"] if full_plot else row["n_steps_pilot"])
                seed = int(data["seed"]) + 200_000
            else:
                n_paths = int(row["n_paths_full"])
                n_steps = int(row["n_steps_full"])
                market_offset = sum((index + 1) * ord(char) for index, char in enumerate(args.market))
                seed = int(data["seed"]) + market_offset + 100_000

            print(f"Regenerating {interval}dt {kind}: {n_paths} paths x {n_steps} observations "
                  f"({int(data.get('sampling_burn_in_steps', 0)) + interval * n_steps} Euler steps)",
                  flush=True)
            result = simulate_modified_heston(
                params,
                SimulationConfig(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    sample_every=interval,
                    burn_in_steps=int(data.get("sampling_burn_in_steps", 0)),
                    return_mode=data.get("sampling_return_mode", "sampled_x"),
                    seed=seed,
                    correlated_noise=bool(data.get("correlated_noise", False)),
                    store_state=args.states,
                    column_prefix=args.market,
                ),
            )
            if kind == "plot" and "synthetic_return_variance" in row:
                values = result.returns.to_numpy(dtype=float, copy=False)
                saved_mean = float(row["synthetic_return_mean"])
                saved_variance = float(row["synthetic_return_variance"])
                if not (np.isclose(values.mean(), saved_mean, atol=1e-9)
                        and np.isclose(values.var(), saved_variance, atol=1e-9)):
                    raise RuntimeError(
                        f"regenerated series does not reproduce saved plot moments for {interval}dt"
                    )
            result.returns.to_pickle(returns_path)
            if args.states:
                np.savez_compressed(states_path, x=result.x, variance=result.variance)
            metadata_path.write_text(json.dumps({
                "market": args.market,
                "interval_steps": interval,
                "kind": kind,
                "seed": seed,
                "n_paths": n_paths,
                "n_observations": n_steps,
                "dt": params.dt,
                "burn_in_steps": int(data.get("sampling_burn_in_steps", 0)),
                "return_mode": data.get("sampling_return_mode", "sampled_x"),
                "parameters_csv": str(parameters_path),
            }, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {returns_path}", flush=True)


if __name__ == "__main__":
    main()
