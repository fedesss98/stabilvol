"""Compare observation intervals on one fixed modified-Heston trajectory."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_heston import plot_returns_pdf
from stabilvol.heston import CalibrationConfig, HestonCalibrator, HestonParams, SimulationConfig, simulate_modified_heston


def load_parameters(path: Path, market: str) -> HestonParams:
    table = pd.read_csv(path)
    if "market" in table:
        table = table.loc[table["market"] == market]
    if table.empty:
        raise ValueError(f"no parameters for {market} in {path}")
    row = table.iloc[-1]
    values = {
        field.name: float(row[field.name])
        for field in fields(HestonParams)
        if field.name in row and pd.notna(row[field.name])
    }
    return HestonParams(**values)


def load_calibration_config(path: Path) -> CalibrationConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    included = {field.name for field in fields(CalibrationConfig)} - {"root", "base_params", "initial_params", "bounds"}
    values = {name: data[name] for name in included if name in data}
    return CalibrationConfig(root=PROJECT_ROOT, **values)


def sampled_returns(x: np.ndarray, start: float, *, interval: int, burn_in: int, days: int) -> np.ndarray:
    """Endpoint differences of x at equally spaced observation times."""
    last_step = burn_in + days * interval
    if last_step > len(x):
        raise ValueError("trajectory is too short for the requested observation interval")
    first_state = np.full((1, x.shape[1]), start) if burn_in == 0 else x[burn_in - 1 : burn_in]
    observed = x[burn_in + interval - 1 : last_step : interval]
    return np.diff(np.concatenate((first_state, observed), axis=0), axis=0)


def near_zero_fraction(values: np.ndarray, width: float, *, exclude_exact_zero: bool) -> float:
    finite = values[np.isfinite(values)]
    if exclude_exact_zero:
        finite = finite[finite != 0]
    return float(np.mean(np.abs(finite) <= width)) if finite.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True, help="Calibration parameters CSV to evaluate")
    parser.add_argument("--config-json", type=Path, default=Path("configs/heston_un_full_ensemble_refine.json"))
    parser.add_argument("--market", default="UN")
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--days", type=int, default=3030)
    parser.add_argument("--burn-in", type=int, default=1000, help="Euler steps discarded before observation")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--windows", type=float, nargs="+", default=[0.001, 0.0025, 0.005])
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/heston_calibration/sampling_sweep"))
    args = parser.parse_args()

    if args.paths < 1 or args.days < 2 or args.burn_in < 0 or any(k < 1 for k in args.intervals):
        parser.error("paths and intervals must be positive, days must be at least 2, and burn-in must be nonnegative")
    if any(width <= 0 for width in args.windows):
        parser.error("near-zero windows must be positive")

    config = load_calibration_config(args.config_json)
    if config.empirical_source != "returns" or config.loss_metric != "mfht_curve":
        parser.error("the sweep requires empirical_source=returns and loss_metric=mfht_curve")
    params = load_parameters(args.parameters, args.market)
    calibrator = HestonCalibrator(config)
    empirical_returns = calibrator.load_empirical_returns(args.market)
    return_target = calibrator.empirical_return_target(args.market)
    curve_targets = {
        pair: calibrator.prepare_curve_target(calibrator.load_empirical_events(args.market, pair))
        for pair in config.threshold_pairs
    }
    empirical_values = empirical_returns.to_numpy(dtype=float, copy=False)
    intervals = sorted(set(args.intervals))
    n_internal = args.burn_in + args.days * max(intervals)
    print(f"Simulating {args.paths} paths for {n_internal} Euler steps at dt={params.dt:g}", flush=True)
    simulation = simulate_modified_heston(
        params,
        SimulationConfig(n_paths=args.paths, n_steps=n_internal, seed=args.seed,
                         correlated_noise=config.correlated_noise, store_state=True,
                         column_prefix=args.market),
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_steps = int(np.count_nonzero(simulation.x == params.start))
    print(f"Post-step resets: {reset_steps}; endpoint returns include any reset jump", flush=True)

    rows = []
    index = pd.date_range("1980-01-01", periods=args.days, freq="B")
    for interval in intervals:
        values = sampled_returns(simulation.x, params.start, interval=interval,
                                 burn_in=args.burn_in, days=args.days)
        returns = pd.DataFrame(values, index=index,
                               columns=[f"{args.market}_{i:05d}" for i in range(args.paths)])
        mfht_losses = [
            calibrator.curve_loss(target, calibrator.count_simulated_events(returns, pair, args.market))
            for pair, target in curve_targets.items()
        ]
        mfht_loss = float(np.mean(mfht_losses))
        return_loss = calibrator.return_moment_loss(returns, return_target)
        row = {
            "interval_steps": interval,
            "observation_time": interval * params.dt,
            "paths": args.paths,
            "days": args.days,
            "synthetic_mean": float(returns.mean(axis=0).mean()),
            "synthetic_average_std": float(returns.std(axis=0).mean()),
            "empirical_average_mean": return_target.mean,
            "empirical_average_std": return_target.std,
            "return_loss": return_loss,
            "mfht_loss": mfht_loss,
            "combined_loss": mfht_loss + config.return_loss_weight * return_loss,
        }
        for width in args.windows:
            suffix = f"{width:g}"
            row[f"empirical_mass_abs_le_{suffix}"] = near_zero_fraction(empirical_values, width, exclude_exact_zero=False)
            row[f"synthetic_mass_abs_le_{suffix}"] = near_zero_fraction(values, width, exclude_exact_zero=False)
            row[f"empirical_nonzero_mass_abs_le_{suffix}"] = near_zero_fraction(
                empirical_values, width, exclude_exact_zero=True,
            )
            row[f"synthetic_nonzero_mass_abs_le_{suffix}"] = near_zero_fraction(
                values, width, exclude_exact_zero=True,
            )
        rows.append(row)
        plot_returns_pdf(empirical_returns, returns,
                         output_dir / f"{args.market}_returns_interval_{interval}.png",
                         market=f"{args.market}, sample every {interval}dt")
        print(f"{interval:>2}dt: average std={row['synthetic_average_std']:.5g}, "
              f"MFHT loss={mfht_loss:.5g}, return loss={return_loss:.5g}", flush=True)

    summary_path = output_dir / f"{args.market}_sampling_sweep.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
