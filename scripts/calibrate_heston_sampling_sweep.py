"""Fit modified-Heston parameters separately at several observation intervals."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stabilvol.heston import HestonParams


def parameters_from_csv(path: Path, market: str) -> dict[str, float]:
    table = pd.read_csv(path)
    if "market" in table:
        table = table.loc[table["market"] == market]
    if table.empty:
        raise ValueError(f"no parameters for {market} in {path}")
    row = table.iloc[-1]
    return {
        field.name: float(row[field.name])
        for field in fields(HestonParams)
        if field.name in row and pd.notna(row[field.name])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=Path("configs/heston_un_full_ensemble_refine.json"))
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--burn-in", type=int, default=1000, help="Euler steps before the first observed return")
    parser.add_argument("--initial-parameters", type=Path, help="Optional parameter CSV used to seed each fit")
    parser.add_argument("--pilot-paths", type=int, help="Override pilot path count for a cheaper first sweep")
    parser.add_argument("--pilot-steps", type=int, help="Override observed pilot returns per path")
    parser.add_argument("--full-steps", type=int, help="Override observed validation returns per path")
    parser.add_argument("--maxiter", type=int, help="Override optimizer generations")
    parser.add_argument("--popsize", type=int, help="Override optimizer population multiplier")
    parser.add_argument("--workers", type=int, help="Override optimizer workers")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/heston_calibration/sampling_fit_sweep"))
    parser.add_argument("--figure-dir", type=Path,
                        default=Path("visualization/heston_calibration/sampling_fit_sweep"))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed interval parameter CSVs")
    parser.add_argument("--skip-full-validation", action="store_true", help="Run a quick pilot-only sweep")
    args = parser.parse_args()

    if any(interval < 1 for interval in args.intervals) or args.burn_in < 0:
        parser.error("intervals must be positive and burn-in nonnegative")
    if args.pilot_paths is not None and args.pilot_paths < 1:
        parser.error("pilot-paths must be positive")
    if args.pilot_steps is not None and args.pilot_steps < 1:
        parser.error("pilot-steps must be positive")
    if args.full_steps is not None and args.full_steps < 1:
        parser.error("full-steps must be positive")

    base = json.loads(args.config_json.read_text(encoding="utf-8"))
    if base.get("empirical_source") != "returns" or base.get("loss_metric") != "mfht_curve":
        parser.error("base config must use empirical_source=returns and loss_metric=mfht_curve")
    markets = base.get("markets", [])
    if len(markets) != 1:
        parser.error("the sweep requires exactly one market")
    if base.get("run", {}).get("staged", False):
        parser.error("use a single-stage base config so each interval gets one comparable fit")
    market = markets[0]

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    figure_root = args.figure_dir if args.figure_dir.is_absolute() else PROJECT_ROOT / args.figure_dir
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    if args.initial_parameters is not None:
        base["initial_params"] = parameters_from_csv(args.initial_parameters, market)
    if args.pilot_paths is not None:
        base["pilot_max_paths"] = args.pilot_paths
    if args.pilot_steps is not None:
        base["pilot_n_steps"] = args.pilot_steps
    if args.full_steps is not None:
        base["full_n_steps"] = args.full_steps

    summary_path = output_root / f"{market}_sampling_fit_sweep.csv"
    summary = []
    for interval in sorted(set(args.intervals)):
        data = json.loads(json.dumps(base))
        data["sampling_interval_steps"] = interval
        data["sampling_burn_in_steps"] = args.burn_in
        data["sampling_return_mode"] = "sampled_x"
        run = data.setdefault("run", {})
        run["output_dir"] = str(output_root / f"interval_{interval}")
        run["figure_dir"] = str(figure_root / f"interval_{interval}")
        for name in ("maxiter", "popsize", "workers"):
            value = getattr(args, name)
            if value is not None:
                run[name] = value
        if args.skip_full_validation:
            run["skip_full_validation"] = True
        config_path = output_root / f"config_interval_{interval}.json"
        config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        parameters_path = Path(run["output_dir"]) / "heston_calibration_parameters.csv"
        if not args.skip_existing or not parameters_path.is_file():
            print(f"\nFitting {market} at {interval}dt; {data['pilot_n_steps']} observed pilot returns "
                  f"per path ({data['pilot_n_steps'] * interval + args.burn_in} Euler steps)", flush=True)
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "calibrate_heston.py"),
                 "--config-json", str(config_path)],
                cwd=PROJECT_ROOT,
                check=True,
            )
        result = pd.read_csv(parameters_path).iloc[-1].to_dict()
        summary.append({
            "interval_steps": interval,
            "observation_time": interval * float(result["dt"]),
            "parameters_csv": str(parameters_path),
            "figure_dir": run["figure_dir"],
            **result,
        })
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"Saved sweep progress to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
