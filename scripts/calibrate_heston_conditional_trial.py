#!/usr/bin/env python3
"""Refit the selected four-market sampling intervals with conditional FHT loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_heston_sampling_sweep import parameters_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path(
        "data/processed/heston_calibration/four_market_sampling_sweep/final_selection.json"))
    parser.add_argument("--markets", nargs="+", choices=("UN", "UW", "LN", "JT"))
    parser.add_argument("--counter", choices=("fortran_clean", "fortran", "quiet_pandas"),
                        default="fortran_clean")
    parser.add_argument("--return-transform", choices=("simple", "log"), default="simple")
    parser.add_argument("--tau-max", type=int, default=30,
                        help="30 compares with prior fits; 300 matches calmG.f")
    parser.add_argument("--return-weight", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, help="Override the optimizer and simulation seed")
    parser.add_argument("--pilot-paths", type=int)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--skip-full-validation", action="store_true")
    parser.add_argument("--plot-full", action="store_true",
                        help="Generate diagnostic plots from full path count and length instead of the pilot sample")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/processed/heston_calibration/conditional_trial"))
    parser.add_argument("--figure-dir", type=Path, default=Path(
        "visualization/heston_calibration/conditional_trial"))
    args = parser.parse_args()
    if args.tau_max < 2 or args.return_weight < 0 or args.maxiter < 0 or args.popsize < 1 or args.workers < 1:
        parser.error("invalid tau-max, return-weight, maxiter, popsize, or workers")
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    markets = args.markets or list(selection)
    output_root = args.output_dir.resolve()
    figure_root = args.figure_dir.resolve()
    for market in markets:
        interval = int(selection[market])
        source = selection_path.parent / market
        base = json.loads((source / f"config_interval_{interval}.json").read_text(encoding="utf-8"))
        parameters = source / f"interval_{interval}" / "heston_calibration_parameters.csv"
        base["initial_params"] = parameters_from_csv(parameters, market)
        base["loss_metric"] = "conditional_fht"
        base["count_method"] = args.counter
        base["synthetic_return_transform"] = args.return_transform
        base["tau_max"] = args.tau_max
        base["return_loss_weight"] = args.return_weight
        base["conditional_vol_quantiles"] = [0.0, 0.5, 0.85, 0.97, 0.995]
        if args.seed is not None:
            base["seed"] = args.seed
        base["description"] = (
            f"{market} {interval}dt conditional FHT and nonzero-return trial, "
            f"seeded from the selected MFHT fit; {args.counter} event counter."
        )
        if args.pilot_paths is not None:
            base["pilot_max_paths"] = args.pilot_paths
        if args.pilot_steps is not None:
            base["pilot_n_steps"] = args.pilot_steps
        run = base.setdefault("run", {})
        run.update(maxiter=args.maxiter, popsize=args.popsize, workers=args.workers,
                   staged=False, output_dir=str(output_root / market),
                   figure_dir=str(figure_root))
        if args.skip_full_validation:
            run["skip_full_validation"] = True
        if args.plot_full:
            run["plot_full"] = True
        config_path = output_root / market / "trial_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(f"{market}: {interval}dt -> {config_path}", flush=True)
        if not args.prepare_only:
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "calibrate_heston.py"),
                            "--config-json", str(config_path)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
