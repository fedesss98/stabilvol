#!/usr/bin/env python3
"""Calibrate modified Heston parameters against empirical MFHT curves."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stabilvol.heston import CalibrationConfig, HestonCalibrator, HestonParams, SimulationConfig, simulate_modified_heston
from stabilvol.heston.calibration import config_to_record, moments_frame
from stabilvol.heston.simulation import SimulationError


def parse_threshold_pair(value: str) -> tuple[float, float]:
    try:
        start, end = value.split(",", maxsplit=1)
        return float(start), float(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold pairs must look like START,END") from exc


def load_json_config(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def default_run_options() -> dict[str, object]:
    return {
        "maxiter": 20,
        "popsize": 8,
        "workers": 1,
        "polish": False,
        "refine": False,
        "quiet_progress": False,
        "skip_full_validation": False,
        "evaluate_default": False,
        "plot_full": False,
        "staged": False,
        "return_maxiter": 20,
        "return_popsize": 8,
        "output_dir": PROJECT_ROOT / "data/processed/heston_calibration",
        "figure_dir": PROJECT_ROOT / "visualization/heston_calibration",
    }


def config_value(config_data: dict, key: str, default: object) -> object:
    run_data = config_data.get("run", {})
    if key in run_data:
        return run_data[key]
    return config_data.get(key, default)


def resolve_path(value: object) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def format_threshold_pairs(pairs: tuple[tuple[float, float], ...]) -> str:
    return ", ".join(f"{start:g}->{end:g}" for start, end in pairs)


def print_result(record: dict[str, object]) -> None:
    print("\nResult")
    print(f"  market: {record['market']}")
    print(f"  pilot_loss: {record['pilot_loss']}")
    print(f"  validation_loss: {record['validation_loss']}")
    print(f"  success: {record['optimization_success']}")
    print(f"  message: {record['optimization_message']}")
    print(
        "  params: "
        f"a={record['a']:.6g}, b={record['b']:.6g}, aa={record['aa']:.6g}, "
        f"bb={record['bb']:.6g}, cc={record['cc']:.6g}, "
        f"rho={record['rho']:.6g}, vstart={record['vstart']:.6g}"
    )
    if "synthetic_return_mean" in record:
        print(
            "  synthetic return moments: "
            f"mean={record['synthetic_return_mean']:.6g}, "
            f"variance={record['synthetic_return_variance']:.6g}, "
            f"skewness={record['synthetic_return_skewness']:.6g}, "
            f"kurtosis={record['synthetic_return_kurtosis']:.6g}"
        )
    if "empirical_return_mean" in record:
        print(
            "  empirical return moments: "
            f"mean={record['empirical_return_mean']:.6g}, "
            f"variance={record['empirical_return_variance']:.6g}, "
            f"skewness={record['empirical_return_skewness']:.6g}, "
            f"kurtosis={record['empirical_return_kurtosis']:.6g}"
        )
    if "empirical_average_std" in record:
        print(
            "  average per-series returns: "
            f"empirical mean={record['empirical_average_mean']:.6g}, "
            f"empirical std={record['empirical_average_std']:.6g}, "
            f"synthetic mean={record['synthetic_average_mean']:.6g}, "
            f"synthetic std={record['synthetic_average_std']:.6g}"
        )
    if "plot_mfht_loss" in record and "plot_return_loss" in record:
        print(f"  plot losses: MFHT={record['plot_mfht_loss']:.6g}, returns={record['plot_return_loss']:.6g}")
    print(f"  elapsed_seconds: {record['elapsed_seconds']:.3f}")


def prefixed_moments(prefix: str, returns: pd.DataFrame) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in moments_frame(returns).items()}


def estimated_differential_evolution_evaluations(maxiter: int, popsize: int, n_params: int = 7) -> int:
    return (maxiter + 1) * popsize * n_params


def build_config(args: argparse.Namespace, config_data: dict) -> CalibrationConfig:
    defaults = CalibrationConfig()
    base_params = HestonParams(**config_data.get("base_params", {}))
    initial_params = (
        base_params.update(**config_data["initial_params"])
        if "initial_params" in config_data else None
    )
    bounds = dict(defaults.bounds)
    for key, value in config_data.get("bounds", {}).items():
        bounds[key] = tuple(value)

    threshold_pairs = tuple(args.threshold_pair) if args.threshold_pair else tuple(
        tuple(pair) for pair in config_data.get("threshold_pairs", defaults.threshold_pairs)
    )
    markets = tuple(args.markets or config_data.get("markets", defaults.markets))

    return CalibrationConfig(
        root=PROJECT_ROOT,
        database_path=Path(args.database if args.database is not None else config_data.get("database_path", defaults.database_path)),
        markets=markets,
        threshold_pairs=threshold_pairs,
        start_date=args.start_date if args.start_date is not None else config_data.get("start_date", defaults.start_date),
        end_date=args.end_date if args.end_date is not None else config_data.get("end_date", defaults.end_date),
        min_empirical_observations=config_data.get("min_empirical_observations", defaults.min_empirical_observations),
        vol_limit=args.vol_limit if args.vol_limit is not None else config_data.get("vol_limit", defaults.vol_limit),
        tau_min=args.tau_min if args.tau_min is not None else config_data.get("tau_min", defaults.tau_min),
        tau_max=args.tau_max if args.tau_max is not None else config_data.get("tau_max", defaults.tau_max),
        pilot_max_paths=args.pilot_paths if args.pilot_paths is not None else config_data.get("pilot_max_paths", defaults.pilot_max_paths),
        pilot_n_steps=args.pilot_steps if args.pilot_steps is not None else config_data.get("pilot_n_steps", defaults.pilot_n_steps),
        full_n_steps=args.full_steps if args.full_steps is not None else config_data.get("full_n_steps", defaults.full_n_steps),
        sampling_interval_steps=config_data.get("sampling_interval_steps", defaults.sampling_interval_steps),
        sampling_burn_in_steps=config_data.get("sampling_burn_in_steps", defaults.sampling_burn_in_steps),
        sampling_return_mode=config_data.get("sampling_return_mode", defaults.sampling_return_mode),
        n_vol_bins=args.vol_bins if args.vol_bins is not None else config_data.get("n_vol_bins", defaults.n_vol_bins),
        event_count_weight=config_data.get("event_count_weight", defaults.event_count_weight),
        empty_penalty=config_data.get("empty_penalty", defaults.empty_penalty),
        min_empirical_bin_events=config_data.get("min_empirical_bin_events", defaults.min_empirical_bin_events),
        min_simulated_bin_events=config_data.get("min_simulated_bin_events", defaults.min_simulated_bin_events),
        loss_metric=args.loss_metric if getattr(args, "loss_metric", None) is not None else config_data.get("loss_metric", defaults.loss_metric),
        empirical_source=config_data.get("empirical_source", defaults.empirical_source),
        threshold_sigma=config_data.get("threshold_sigma", defaults.threshold_sigma),
        return_loss_weight=config_data.get("return_loss_weight", defaults.return_loss_weight),
        seed=args.seed if args.seed is not None else config_data.get("seed", defaults.seed),
        bounds=bounds,
        base_params=base_params,
        initial_params=initial_params,
        count_method=config_data.get("count_method", defaults.count_method),
        std_normalization=(
            args.std_normalization
            if getattr(args, "std_normalization", None) is not None
            else config_data.get("std_normalization", defaults.std_normalization)
        ),
        correlated_noise=(
            args.correlated_noise
            if getattr(args, "correlated_noise", None) is not None
            else config_data.get("correlated_noise", defaults.correlated_noise)
        ),
    )


def build_run_options(args: argparse.Namespace, config_data: dict) -> argparse.Namespace:
    options = default_run_options()
    for key in tuple(options):
        value = getattr(args, key, None)
        if value is None:
            value = config_value(config_data, key, options[key])
        if key in {"output_dir", "figure_dir"}:
            value = resolve_path(value)
        options[key] = value
    return argparse.Namespace(**options)


def bin_mfht(events: pd.DataFrame, vol_edges: np.ndarray) -> pd.DataFrame:
    frame = events[["Volatility", "FHT"]].copy()
    frame["Bin"] = pd.cut(frame["Volatility"], bins=vol_edges, include_lowest=True)
    grouped = frame.groupby("Bin", observed=True)["FHT"].agg(["mean", "size"])
    grouped["left"] = [interval.left for interval in grouped.index]
    return grouped


def finite_values(frame: pd.DataFrame, columns: int | None = None) -> np.ndarray:
    if columns is not None:
        frame = frame.iloc[:, :columns]
    values = frame.to_numpy(dtype=float, copy=False).ravel()
    return values[np.isfinite(values)]


def near_zero_masses(frame: pd.DataFrame, width: float = 0.0025) -> tuple[float, float]:
    values = finite_values(frame)
    if values.size == 0:
        return float("nan"), float("nan")
    nonzero = values[values != 0]
    return (float(np.mean(np.abs(values) <= width)),
            float(np.mean(np.abs(nonzero) <= width)) if nonzero.size else float("nan"))


def shared_edges(first: np.ndarray, second: np.ndarray, *, bins: int, lower_q: float, upper_q: float) -> np.ndarray:
    combined = np.concatenate([first, second])
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return np.linspace(-1.0, 1.0, bins + 1)

    low, high = np.quantile(combined, [lower_q, upper_q])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        center = float(np.nanmean(combined)) if combined.size else 0.0
        width = max(float(np.nanstd(combined)), np.finfo(float).eps)
        low, high = center - width, center + width
    return np.linspace(float(low), float(high), bins + 1)


def plot_returns_pdf(
    empirical_returns: pd.DataFrame,
    simulated_returns: pd.DataFrame,
    output_path: Path,
    *,
    market: str,
    bins: int = 240,
) -> None:
    empirical = finite_values(empirical_returns)
    simulated = finite_values(simulated_returns)
    if empirical.size == 0 or simulated.size == 0:
        return

    edges = shared_edges(empirical, simulated, bins=bins, lower_q=0.001, upper_q=0.999)
    empirical = empirical[(empirical >= edges[0]) & (empirical <= edges[-1])]
    simulated = simulated[(simulated >= edges[0]) & (simulated <= edges[-1])]

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.hist(empirical, bins=edges, density=True, histtype="step", linewidth=1.8, label=f"Empirical ({empirical_returns.shape[1]} stocks)")
    ax.hist(simulated, bins=edges, density=True, histtype="step", linewidth=1.8, label=f"Synthetic ({simulated_returns.shape[1]} paths)")
    ax.set_title(f"{market} return PDF ({bins} bins)")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fht_probability(events: pd.DataFrame, tau_min: int, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(events["FHT"], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[(values >= tau_min) & (values <= tau_max)]
    taus = np.arange(tau_min, tau_max + 1)
    if values.size == 0:
        return taus, np.zeros_like(taus, dtype=float)
    counts, _ = np.histogram(values, bins=np.arange(tau_min - 0.5, tau_max + 1.5, 1.0))
    return taus, counts / counts.sum()


def plot_fht_pdf(
    event_pairs: dict[tuple[float, float], tuple[pd.DataFrame, pd.DataFrame]],
    output_path: Path,
    *,
    market: str,
    tau_min: int,
    tau_max: int,
) -> None:
    if not event_pairs:
        return

    n_pairs = len(event_pairs)
    n_cols = min(2, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.7 * n_rows), squeeze=False, layout="constrained")

    for ax, (pair, (empirical, simulated)) in zip(axs.ravel(), event_pairs.items()):
        taus, empirical_prob = fht_probability(empirical, tau_min, tau_max)
        _, simulated_prob = fht_probability(simulated, tau_min, tau_max)
        ax.step(taus, empirical_prob, where="mid", linewidth=1.8, label=f"Empirical ({len(empirical)} events)")
        ax.step(taus, simulated_prob, where="mid", linewidth=1.8, label=f"Synthetic ({len(simulated)} events)")
        ax.set_title(f"{pair[0]:g} -> {pair[1]:g}")
        ax.set_xlabel("FHT")
        ax.set_ylabel("Probability")
        ax.legend()

    for ax in axs.ravel()[n_pairs:]:
        ax.set_visible(False)

    fig.suptitle(f"{market} FHT PDF")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_market_comparisons(
    calibrator: HestonCalibrator,
    market: str,
    params: HestonParams,
    output_dir: Path,
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulation = simulate_modified_heston(
        params,
        SimulationConfig(
            n_paths=n_paths,
            n_steps=n_steps,
            sample_every=calibrator.config.sampling_interval_steps,
            burn_in_steps=calibrator.config.sampling_burn_in_steps,
            return_mode=calibrator.config.sampling_return_mode,
            seed=seed,
            correlated_noise=calibrator.config.correlated_noise,
            store_state=False,
            column_prefix=market,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    empirical_returns = calibrator.load_empirical_returns(market)
    plot_returns_pdf(
        empirical_returns,
        simulation.returns,
        output_dir / f"{market}_returns_pdf.png",
        market=market,
    )
    if calibrator.config.loss_metric == "return_moments":
        return empirical_returns, simulation.returns

    event_pairs = {}
    for pair in calibrator.config.threshold_pairs:
        empirical = calibrator.load_empirical_events(market, pair)
        simulated = calibrator.count_simulated_events(simulation.returns, pair, market)
        event_pairs[pair] = (empirical, simulated)
        start = str(pair[0]).replace("-", "m").replace(".", "p")
        end = str(pair[1]).replace("-", "m").replace(".", "p")
        if calibrator.config.loss_metric == "mfht_curve":
            target = calibrator.prepare_curve_target(empirical)
            comparison = calibrator.curve_comparison(target, simulated)
            comparison.to_csv(output_dir / f"{market}_{start}_{end}_mfht.csv", index=False)
            vol_edges = target.edges
            x = comparison["volatility_midpoint"]
            empirical_x, empirical_y = x[comparison["fit_bin"]], comparison.loc[comparison["fit_bin"], "empirical_mfht"]
            simulated_x, simulated_y = x[comparison["simulated_bin_covered"]], comparison.loc[comparison["simulated_bin_covered"], "simulated_mfht"]
        else:
            if simulated.empty:
                continue
            vol_cap = max(
                float(empirical["Volatility"].quantile(0.995)),
                float(simulated["Volatility"].quantile(0.995)),
                np.finfo(float).eps,
            )
            vol_edges = np.linspace(0.0, vol_cap, calibrator.config.n_vol_bins + 1)
            empirical_mfht = bin_mfht(empirical, vol_edges)
            simulated_mfht = bin_mfht(simulated, vol_edges)
            empirical_x, empirical_y = empirical_mfht["left"], empirical_mfht["mean"]
            simulated_x, simulated_y = simulated_mfht["left"], simulated_mfht["mean"]

        fht_edges = np.arange(calibrator.config.tau_min, calibrator.config.tau_max + 2)
        empirical_hist, _, _ = np.histogram2d(empirical["Volatility"], empirical["FHT"], bins=[vol_edges, fht_edges])
        simulated_hist, _, _ = np.histogram2d(simulated["Volatility"], simulated["FHT"], bins=[vol_edges, fht_edges])
        empirical_hist /= max(empirical_hist.sum(), 1)
        simulated_hist /= max(simulated_hist.sum(), 1)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        vmax = max(empirical_hist.max(), simulated_hist.max(), np.finfo(float).eps)
        axs[0].imshow(empirical_hist.T, aspect="auto", origin="lower", vmax=vmax)
        axs[0].set_title("Empirical event fraction")
        axs[1].imshow(simulated_hist.T, aspect="auto", origin="lower", vmax=vmax)
        axs[1].set_title("Simulated event fraction")
        axs[2].plot(empirical_x, empirical_y, label="Empirical")
        axs[2].plot(simulated_x, simulated_y, label="Simulated")
        axs[2].set_title("MFHT projection")
        axs[2].set_xlabel("Volatility")
        axs[2].set_ylabel("FHT")
        axs[2].legend()
        fig.suptitle(f"{market} thresholds {pair[0]} -> {pair[1]}")

        fig.savefig(output_dir / f"{market}_{start}_{end}.png", dpi=180)
        plt.close(fig)

    plot_fht_pdf(
        event_pairs,
        output_dir / f"{market}_fht_pdf.png",
        market=market,
        tau_min=calibrator.config.tau_min,
        tau_max=calibrator.config.tau_max,
    )
    return empirical_returns, simulation.returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", choices=["UN", "UW", "LN", "JT"], help="Markets to calibrate")
    parser.add_argument("--threshold-pair", action="append", type=parse_threshold_pair, help="Threshold pair START,END; can be repeated")
    parser.add_argument("--database", help="Empirical SQLite database path")
    parser.add_argument("--config-json", type=Path, help="Optional JSON config for calibration and run options")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--vol-limit", type=float)
    parser.add_argument("--tau-min", type=int)
    parser.add_argument("--tau-max", type=int)
    parser.add_argument("--pilot-paths", type=int)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--full-steps", type=int)
    parser.add_argument("--vol-bins", type=int)
    parser.add_argument("--loss-metric", choices=("mfht_curve", "fht_distribution", "return_moments"),
                        help="Objective to optimize; default is the MFHT-versus-volatility curve")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--std-normalization", action="store_true", default=None, help="Scale simulated FHT thresholds by the simulated return standard deviation")
    parser.add_argument("--no-std-normalization", dest="std_normalization", action="store_false", help="Use raw simulated FHT thresholds")
    parser.add_argument("--correlated-noise", action="store_true", default=None)
    parser.add_argument("--uncorrelated-noise", dest="correlated_noise", action="store_false")
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--workers", type=int, help="Parallel optimizer workers. Use -1 for all available cores.")
    parser.add_argument("--polish", action="store_true", default=None)
    parser.add_argument("--no-polish", dest="polish", action="store_false")
    parser.add_argument("--refine", action="store_true", default=None)
    parser.add_argument("--no-refine", dest="refine", action="store_false")
    parser.add_argument("--quiet-progress", action="store_true", default=None, help="Do not print per-generation optimizer progress")
    parser.add_argument("--show-progress", dest="quiet_progress", action="store_false")
    parser.add_argument("--skip-full-validation", action="store_true", default=None)
    parser.add_argument("--full-validation", dest="skip_full_validation", action="store_false")
    parser.add_argument("--evaluate-default", action="store_true", default=None, help="Skip optimization and evaluate parm.dat defaults")
    parser.add_argument("--optimize", dest="evaluate_default", action="store_false")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--plot-full", action="store_true", default=None, help="Use full path count and length for plots")
    parser.add_argument("--plot-pilot", dest="plot_full", action="store_false")
    parser.add_argument("--staged", action="store_true", default=None,
                        help="Fit return mean/std first, then one MFHT curve with a return penalty")
    parser.add_argument("--return-maxiter", type=int, help="First-stage optimizer generations")
    parser.add_argument("--return-popsize", type=int, help="First-stage population multiplier")
    args = parser.parse_args()

    config_data = load_json_config(args.config_json) if args.config_json else {}
    run_options = build_run_options(args, config_data)
    config = build_config(args, config_data)
    if config.empirical_source == "database" and config.loss_metric != "return_moments" and not config.absolute_database_path.is_file():
        parser.error(f"empirical FHT database not found: {config.absolute_database_path}")
    if run_options.staged:
        if run_options.evaluate_default or config.loss_metric != "mfht_curve" or len(config.threshold_pairs) != 1:
            parser.error("staged mode requires optimization of exactly one MFHT threshold pair")
        if config.empirical_source != "returns" or config.return_loss_weight <= 0:
            parser.error("staged mode requires empirical_source=returns and positive return_loss_weight")
    for market in config.markets:
        returns_path = config.root / "data" / "interim" / f"{market}.pickle"
        if not returns_path.is_file():
            parser.error(f"market returns file not found: {returns_path}")
    calibrator = HestonCalibrator(config)
    run_options.output_dir.mkdir(parents=True, exist_ok=True)
    run_options.figure_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = run_options.output_dir / f"heston_calibration_config_{run_id}.json"
    manifest_record = {"calibration": config_to_record(config), "run": vars(run_options)}
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_record, handle, indent=2, default=str)
    run_start = time.perf_counter()
    mode = "staged" if run_options.staged else ("evaluate-default" if run_options.evaluate_default else "optimize")

    print("Modified Heston calibration")
    print(f"  run_id: {run_id}")
    if args.config_json:
        print(f"  config_json: {args.config_json}")
    if config_data.get("description"):
        print(f"  config_description: {config_data['description']}")
    print(f"  mode: {mode}")
    print(f"  markets: {', '.join(config.markets)}")
    print(f"  thresholds: {format_threshold_pairs(config.threshold_pairs)}")
    print(f"  std_normalization: {config.std_normalization}")
    print(f"  empirical_source: {config.empirical_source}")
    print(f"  threshold_sigma: {config.threshold_sigma if config.threshold_sigma is not None else 'automatic'}")
    print(f"  return_loss_weight: {config.return_loss_weight}")
    print(f"  loss_metric: {config.loss_metric}")
    print(f"  noise: {'correlated' if config.correlated_noise else 'uncorrelated'}")
    print(f"  optimized_parameters: {', '.join(calibrator.parameter_names())}")
    if config.empirical_source == "database":
        print(f"  database: {config.absolute_database_path}")
    print(f"  output_dir: {run_options.output_dir}")
    print(f"  effective_config: {manifest_path}")
    print(f"  figure_dir: {run_options.figure_dir}")
    print(
        "  pilot: "
        f"max_paths={config.pilot_max_paths}, steps={config.pilot_n_steps}, "
        f"vol_bins={config.n_vol_bins}"
    )
    print(f"  sampling: every={config.sampling_interval_steps}dt, "
          f"burn_in_steps={config.sampling_burn_in_steps}, mode={config.sampling_return_mode}")
    if not run_options.evaluate_default:
        estimated_evals = estimated_differential_evolution_evaluations(
            run_options.maxiter,
            run_options.popsize,
            n_params=len(calibrator.parameter_names()),
        )
        print(
            "  optimizer: "
            f"maxiter={run_options.maxiter}, popsize={run_options.popsize}, workers={run_options.workers}, "
            f"full_validation={not run_options.skip_full_validation}"
        )
        print(
            "  estimated objective evaluations: "
            f"{estimated_evals} before optional polish/refine "
            "(first progress line appears after the initial population is evaluated)"
        )

    results = []
    stage1_results = []
    loss_records = []
    for market in config.markets:
        market_start = time.perf_counter()
        _, n_market_stocks = calibrator.market_shape(market)
        print(f"\n[{market}] starting with {n_market_stocks} empirical stocks")
        market_config = config
        if run_options.staged:
            first_config = replace(
                config, loss_metric="return_moments", threshold_pairs=(), return_loss_weight=0.0,
                initial_params=config.initial_params or config.base_params,
            )
            first_calibrator = HestonCalibrator(first_config)
            target = first_calibrator.empirical_return_target(market)
            collective_sigma = first_calibrator.empirical_collective_sigma(market)
            print(f"[{market}] empirical average mean={target.mean:.8g}, average std={target.std:.8g} "
                  f"across {target.n_series} valid stocks")
            print(f"[{market}] stage 1: fitting return mean and std")
            first_result = first_calibrator.calibrate_market(
                market,
                maxiter=run_options.return_maxiter,
                popsize=run_options.return_popsize,
                workers=run_options.workers,
                polish=run_options.polish,
                refine=run_options.refine,
                validate_full=False,
                progress=not run_options.quiet_progress,
            )
            first_plot = None
            if np.isfinite(first_result.pilot_loss):
                try:
                    first_plot = simulate_modified_heston(
                        first_result.params,
                        SimulationConfig(
                            n_paths=min(config.pilot_max_paths, n_market_stocks),
                            n_steps=config.pilot_n_steps,
                            sample_every=config.sampling_interval_steps,
                            burn_in_steps=config.sampling_burn_in_steps,
                            return_mode=config.sampling_return_mode,
                            seed=config.seed + 200_000,
                            correlated_noise=config.correlated_noise,
                            store_state=False,
                            column_prefix=market,
                        ),
                    ).returns
                    first_figure_dir = run_options.figure_dir / market
                    first_figure_dir.mkdir(parents=True, exist_ok=True)
                    plot_returns_pdf(
                        first_calibrator.load_empirical_returns(market), first_plot,
                        first_figure_dir / f"{market}_stage1_returns_pdf.png", market=market,
                    )
                except (SimulationError, FloatingPointError, OverflowError, ValueError) as exc:
                    print(f"[{market}] stage 1 independent-seed return plot unavailable: {exc}")
            stage1_results.append({
                **first_result.to_record(),
                "run_id": run_id,
                "loss_metric": "return_moments",
                "empirical_average_mean": target.mean,
                "empirical_average_std": target.std,
                "empirical_target_stocks": target.n_series,
                "plot_return_loss": first_calibrator.return_moment_loss(first_plot, target)
                if first_plot is not None else None,
                "synthetic_average_mean": float(first_plot.mean(axis=0).mean())
                if first_plot is not None else None,
                "synthetic_average_std": float(first_plot.std(axis=0).mean())
                if first_plot is not None else None,
            })
            stage1_latest = run_options.output_dir / "heston_calibration_stage1_parameters.csv"
            stage1_run = run_options.output_dir / f"heston_calibration_stage1_parameters_{run_id}.csv"
            pd.DataFrame(stage1_results).to_csv(stage1_latest, index=False)
            pd.DataFrame(stage1_results).to_csv(stage1_run, index=False)
            if not np.isfinite(first_result.pilot_loss):
                raise RuntimeError(f"[{market}] stage 1 found no finite return fit; saved {stage1_run}")
            print(f"[{market}] stage 1 loss={first_result.pilot_loss:.6g}; parameters={first_result.params.to_dict()}")
            market_config = replace(config, threshold_sigma=collective_sigma, initial_params=first_result.params)
            manifest_record.setdefault("resolved_threshold_sigma_by_market", {})[market] = collective_sigma
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest_record, handle, indent=2, default=str)
            del first_calibrator
            calibrator = HestonCalibrator(market_config)
            print(f"[{market}] stage 2: MFHT {format_threshold_pairs(config.threshold_pairs)} "
                  f"with fixed collective empirical sigma={collective_sigma:.8g}")
        if run_options.evaluate_default:
            n_paths = min(config.pilot_max_paths, n_market_stocks)
            print(f"[{market}] evaluating default parameters on {n_paths} paths x {config.pilot_n_steps} steps")
            loss, _ = calibrator.evaluate_params(
                market,
                market_config.base_params,
                n_paths=n_paths,
                n_steps=config.pilot_n_steps,
                seed=config.seed,
            )
            result = {
                "market": market,
                "pilot_loss": loss,
                "validation_loss": None,
                "optimization_success": True,
                "optimization_message": "evaluated default parameters only",
                "n_paths_pilot": n_paths,
                "n_paths_full": n_market_stocks,
                "n_steps_pilot": config.pilot_n_steps,
                "n_steps_full": config.full_n_steps,
                **market_config.base_params.to_dict(),
            }
            result_record = result
            fitted_params = market_config.base_params
        else:
            print(f"[{market}] optimizing parameters")
            calibration_result = calibrator.calibrate_market(
                market,
                maxiter=run_options.maxiter,
                popsize=run_options.popsize,
                workers=run_options.workers,
                polish=run_options.polish,
                refine=run_options.refine,
                validate_full=not run_options.skip_full_validation,
                progress=not run_options.quiet_progress,
            )
            result_record = calibration_result.to_record()
            fitted_params = calibration_result.params

        plot_n_paths = n_market_stocks if run_options.plot_full else min(config.pilot_max_paths, n_market_stocks)
        plot_n_steps = config.full_n_steps if run_options.plot_full else config.pilot_n_steps
        print(f"[{market}] writing plots from {plot_n_paths} paths x {plot_n_steps} steps")
        empirical_returns, synthetic_returns = plot_market_comparisons(
            calibrator,
            market,
            fitted_params,
            run_options.figure_dir / market,
            n_paths=plot_n_paths,
            n_steps=plot_n_steps,
            seed=config.seed + 200_000,
        )
        moment_record = {
            **prefixed_moments("empirical_return", empirical_returns),
            **prefixed_moments("synthetic_return", synthetic_returns),
        }
        empirical_near_zero, empirical_nonzero_near_zero = near_zero_masses(empirical_returns)
        synthetic_near_zero, synthetic_nonzero_near_zero = near_zero_masses(synthetic_returns)
        return_target = calibrator.empirical_return_target(market)
        market_elapsed = time.perf_counter() - market_start
        result_record.update(
            {
                "run_id": run_id,
                "mode": mode,
                "loss_metric": market_config.loss_metric,
                "empirical_source": market_config.empirical_source,
                "threshold_sigma": market_config.threshold_sigma,
                "return_loss_weight": market_config.return_loss_weight,
                "sampling_interval_steps": market_config.sampling_interval_steps,
                "sampling_burn_in_steps": market_config.sampling_burn_in_steps,
                "sampling_return_mode": market_config.sampling_return_mode,
                "correlated_noise": config.correlated_noise,
                "optimized_parameters": ",".join(calibrator.parameter_names()),
                "config_json": str(args.config_json) if args.config_json else None,
                "config_description": config_data.get("description"),
                "threshold_pairs": format_threshold_pairs(config.threshold_pairs),
                "maxiter": run_options.maxiter,
                "popsize": run_options.popsize,
                "workers": run_options.workers,
                "polish": run_options.polish,
                "refine": run_options.refine,
                "skip_full_validation": run_options.skip_full_validation,
                "plot_full": run_options.plot_full,
                "elapsed_seconds": market_elapsed,
                "empirical_average_mean": return_target.mean,
                "empirical_average_std": return_target.std,
                "synthetic_average_mean": float(synthetic_returns.mean(axis=0).mean()),
                "synthetic_average_std": float(synthetic_returns.std(axis=0).mean()),
                "empirical_mass_abs_le_0p0025": empirical_near_zero,
                "synthetic_mass_abs_le_0p0025": synthetic_near_zero,
                "empirical_nonzero_mass_abs_le_0p0025": empirical_nonzero_near_zero,
                "synthetic_nonzero_mass_abs_le_0p0025": synthetic_nonzero_near_zero,
                **moment_record,
            }
        )
        if run_options.staged:
            result_record["plot_return_loss"] = calibrator.return_moment_loss(synthetic_returns, target)
            result_record["plot_mfht_loss"] = calibrator.grid_loss(
                calibrator.load_empirical_grid(market), synthetic_returns, market)
            result_record["empirical_average_mean"] = target.mean
            result_record["empirical_average_std"] = target.std
            result_record["empirical_target_stocks"] = target.n_series
            result_record["synthetic_average_mean"] = float(synthetic_returns.mean(axis=0).mean())
            result_record["synthetic_average_std"] = float(synthetic_returns.std(axis=0).mean())
        results.append(result_record)
        print_result(result_record)
        loss_records.append(
            {
                "run_id": run_id,
                "market": market,
                "loss_metric": market_config.loss_metric,
                "empirical_source": market_config.empirical_source,
                "threshold_sigma": market_config.threshold_sigma,
                "return_loss_weight": market_config.return_loss_weight,
                "sampling_interval_steps": market_config.sampling_interval_steps,
                "sampling_burn_in_steps": market_config.sampling_burn_in_steps,
                "sampling_return_mode": market_config.sampling_return_mode,
                "correlated_noise": config.correlated_noise,
                "optimized_parameters": ",".join(calibrator.parameter_names()),
                "config_json": str(args.config_json) if args.config_json else None,
                "config_description": config_data.get("description"),
                "mode": mode,
                "threshold_pairs": format_threshold_pairs(config.threshold_pairs),
                "maxiter": run_options.maxiter,
                "popsize": run_options.popsize,
                "workers": run_options.workers,
                "plots_n_paths": plot_n_paths,
                "plots_n_steps": plot_n_steps,
                "elapsed_seconds": market_elapsed,
                **moment_record,
                **({
                    "plot_return_loss": result_record["plot_return_loss"],
                    "plot_mfht_loss": result_record["plot_mfht_loss"],
                    "empirical_average_mean": result_record["empirical_average_mean"],
                    "empirical_average_std": result_record["empirical_average_std"],
                    "empirical_target_stocks": result_record["empirical_target_stocks"],
                    "synthetic_average_mean": result_record["synthetic_average_mean"],
                    "synthetic_average_std": result_record["synthetic_average_std"],
                } if run_options.staged else {}),
            }
        )

    total_elapsed = time.perf_counter() - run_start
    parameters_latest = run_options.output_dir / "heston_calibration_parameters.csv"
    parameters_run = run_options.output_dir / f"heston_calibration_parameters_{run_id}.csv"
    outputs_latest = run_options.output_dir / "heston_calibration_outputs.csv"
    outputs_run = run_options.output_dir / f"heston_calibration_outputs_{run_id}.csv"
    pd.DataFrame(results).to_csv(parameters_latest, index=False)
    pd.DataFrame(results).to_csv(parameters_run, index=False)
    pd.DataFrame(loss_records).to_csv(outputs_latest, index=False)
    pd.DataFrame(loss_records).to_csv(outputs_run, index=False)
    if stage1_results:
        print(f"  saved stage 1 parameters: {stage1_run}")
    print("\nRun complete")
    print(f"  total_elapsed_seconds: {total_elapsed:.3f}")
    print(f"  saved latest parameters: {parameters_latest}")
    print(f"  saved run parameters: {parameters_run}")
    print(f"  saved latest outputs: {outputs_latest}")
    print(f"  saved run outputs: {outputs_run}")
    print(f"Saved comparison plots to {run_options.figure_dir}")


if __name__ == "__main__":
    main()
