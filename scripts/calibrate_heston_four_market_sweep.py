"""Run one sampling-interval calibration sweep per market with local sigma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def empirical_selection(market: str, config: dict) -> tuple[int, float, tuple[float, float]]:
    returns = pd.read_pickle(PROJECT_ROOT / "data" / "interim" / f"{market}.pickle")
    dated = returns.loc[config.get("start_date", "1980-01-01"):config.get("end_date", "2022-07-01")]
    selected = dated.loc[:, dated.notna().sum() >= int(config["min_empirical_observations"])]
    values = selected.to_numpy(dtype=float, copy=False)
    finite = values[np.isfinite(values)]
    if selected.empty or finite.size < 2:
        raise ValueError(f"no eligible returns for {market}")
    low, high = np.quantile(finite, [0.001, 0.999])
    sigma = float(np.clip(finite, low, high).std(ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"invalid clipped threshold sigma for {market}")
    return selected.shape[1], sigma, (float(low), float(high))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path,
                        default=Path("configs/heston_un_full_ensemble_refine.json"))
    parser.add_argument("--markets", nargs="+", default=["UN", "UW", "LN", "JT"])
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--pilot-paths", type=int, help="Optional smoke-run override")
    parser.add_argument("--pilot-steps", type=int, help="Optional smoke-run override")
    parser.add_argument("--full-steps", type=int, help="Optional smoke-run override")
    parser.add_argument("--skip-full-validation", action="store_true", help="Pilot-only smoke run")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/heston_calibration/four_market_sampling_sweep"))
    parser.add_argument("--figure-dir", type=Path,
                        default=Path("visualization/heston_calibration/four_market_sampling_sweep"))
    args = parser.parse_args()
    if args.workers < 1 or any(k < 1 for k in args.intervals):
        parser.error("workers and intervals must be positive")
    source = args.config_json.resolve()
    base = json.loads(source.read_text(encoding="utf-8"))
    if base.get("empirical_source") != "returns" or base.get("loss_metric") != "mfht_curve":
        parser.error("template must use empirical returns and MFHT-curve loss")
    if base.get("min_empirical_observations") != 3030:
        parser.error("template must select stocks with at least 3030 observations")
    output_root = args.output_dir.resolve()
    figure_root = args.figure_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    for market in args.markets:
        count, sigma, clip_bounds = empirical_selection(market, base)
        config = json.loads(json.dumps(base))
        config["markets"] = [market]
        config["description"] = (f"{market} single-threshold sampling-interval calibration; "
                                 f"{count} eligible stocks, clipped market collective sigma.")
        config["threshold_pairs"] = [[-0.1, -1.5]]
        config["threshold_sigma"] = sigma
        config["threshold_sigma_method"] = "pooled returns clipped to 0.1% and 99.9% empirical quantiles"
        config["threshold_sigma_clip_bounds"] = list(clip_bounds)
        config["pilot_max_paths"] = count
        config["full_n_steps"] = 11089 if args.full_steps is None else args.full_steps
        config["run"]["workers"] = args.workers
        if market != "UN":
            config.pop("initial_params", None)
        market_root = output_root / market
        market_root.mkdir(parents=True, exist_ok=True)
        config_path = market_root / "base_config.json"
        if args.skip_existing and config_path.is_file():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise ValueError(f"cannot --skip-existing after changing the base config: {config_path}")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "calibrate_heston_sampling_sweep.py"),
                   "--config-json", str(config_path), "--intervals", *map(str, args.intervals),
                   "--workers", str(args.workers),
                   "--output-dir", str(market_root), "--figure-dir", str(figure_root / market)]
        for option in ("maxiter", "popsize", "pilot_paths", "pilot_steps", "full_steps"):
            value = getattr(args, option)
            if value is not None:
                command.extend(["--" + option.replace("_", "-"), str(value)])
        if args.skip_full_validation:
            command.append("--skip-full-validation")
        if args.skip_existing:
            command.append("--skip-existing")
        print(f"[{market}] {count} stocks, clipped collective sigma={sigma:.9g}; "
              f"intervals={args.intervals}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
