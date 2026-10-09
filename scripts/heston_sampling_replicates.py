"""Run independent ensembles for selected Heston sampling fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.export_heston_sampling_series import fitted_parameters
from scripts.sweep_heston_sampling import load_calibration_config
from stabilvol.heston import HestonCalibrator, SimulationConfig, simulate_modified_heston


def band(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    count = np.isfinite(array).sum(axis=0)
    mean = np.divide(np.nansum(array, axis=0), count,
                     out=np.full(array.shape[1], np.nan), where=count > 0)
    squared = np.nansum((array - mean) ** 2, axis=0)
    std = np.sqrt(np.divide(squared, count - 1,
                            out=np.full(array.shape[1], np.nan), where=count > 1))
    return mean, std, count


def band_details(values: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Summarize every bin across realizations, ignoring uncovered MFHT bins."""
    array = np.asarray(values, dtype=float)
    mean, std, count = band(values)
    median = np.full(array.shape[1], np.nan)
    minimum = median.copy()
    maximum = median.copy()
    for index in range(array.shape[1]):
        observed = array[np.isfinite(array[:, index]), index]
        if observed.size:
            median[index] = np.median(observed)
            minimum[index] = observed.min()
            maximum[index] = observed.max()
    return {
        "synthetic_mean": mean, "synthetic_std": std,
        "synthetic_median": median, "synthetic_min": minimum,
        "synthetic_max": maximum, "replicates_covered": count,
    }


def probabilities(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inside = np.histogram(values, bins=edges)[0]
    all_counts = np.r_[np.count_nonzero(values < edges[0]), inside,
                       np.count_nonzero(values > edges[-1])]
    return inside, all_counts / all_counts.sum()


def plot_band(frame: pd.DataFrame, x: str, empirical: str, path: Path,
              xlabel: str, ylabel: str, xlim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xx = frame[x].to_numpy(dtype=float)
    ax.plot(xx, frame[empirical], label="Empirical", color="C0")
    if {"synthetic_median", "synthetic_min", "synthetic_max"}.issubset(frame.columns):
        ax.plot(xx, frame.synthetic_median, label="Synthetic median", color="C1")
        ax.fill_between(xx, frame.synthetic_min.to_numpy(dtype=float),
                        frame.synthetic_max.to_numpy(dtype=float), color="C1", alpha=0.25,
                        label="Synthetic min–max")
    else:
        yy = frame.synthetic_mean.to_numpy(dtype=float)
        sd = frame.synthetic_std.to_numpy(dtype=float)
        ax.plot(xx, yy, label="Synthetic mean", color="C1")
        ax.fill_between(xx, yy - sd, yy + sd, color="C1", alpha=0.25, label="Synthetic ±1 SD")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def completed_intervals(sweep: Path) -> list[int]:
    return sorted(int(match.group(1)) for path in sweep.glob("config_interval_*.json")
                  if (match := re.fullmatch(r"config_interval_(\d+)\.json", path.name))
                  and (sweep / f"interval_{match.group(1)}" / "heston_calibration_parameters.csv").is_file())


def run_fit(args: argparse.Namespace, market: str, interval: int, sweep: Path, output: Path) -> list[dict]:
    config_path = sweep / f"config_interval_{interval}.json"
    parameters_path = sweep / f"interval_{interval}" / "heston_calibration_parameters.csv"
    if not config_path.is_file() or not parameters_path.is_file():
        raise FileNotFoundError(f"missing config or parameters for {market} at {interval}dt")
    config = load_calibration_config(config_path)
    if config.sampling_interval_steps != interval or market not in config.markets:
        raise ValueError(f"market or interval disagrees with {config_path}")
    if len(config.threshold_pairs) != 1:
        raise ValueError("replicate analysis expects exactly one threshold pair")
    params, _ = fitted_parameters(parameters_path, market)
    calibrator = HestonCalibrator(config)
    empirical = calibrator.load_empirical_returns(market)
    n_paths = args.paths or empirical.shape[1]
    n_days = args.days or config.pilot_n_steps
    target = calibrator.empirical_return_target(market)
    empirical_values = empirical.to_numpy(dtype=float, copy=False)
    empirical_values = empirical_values[np.isfinite(empirical_values)]
    edges = np.linspace(*np.quantile(empirical_values, [0.001, 0.999]), args.bins + 1)
    widths = np.diff(edges)
    empirical_inside, empirical_prob = probabilities(empirical_values, edges)
    empirical_density = empirical_inside / (empirical_inside.sum() * widths)
    empirical_nonzero = empirical_values[empirical_values != 0]
    zero_bins = args.bins + args.bins % 2
    span = float(max(abs(np.quantile(empirical_nonzero, 0.001)),
                     abs(np.quantile(empirical_nonzero, 0.999))))
    zero_edges = np.linspace(-span, span, zero_bins + 1)
    zero_edges[zero_bins // 2] = 0.0
    zero_widths = np.diff(zero_edges)
    empirical_zero_inside, empirical_zero_prob = probabilities(empirical_nonzero, zero_edges)
    empirical_zero_density = empirical_zero_inside / (empirical_zero_inside.sum() * zero_widths)
    pair = config.threshold_pairs[0]
    empirical_events = calibrator.load_empirical_events(market, pair)
    curve_target = calibrator.prepare_curve_target(empirical_events)
    fht_days = np.arange(config.tau_min, config.tau_max + 1)
    fht_edges = np.arange(config.tau_min - 0.5, config.tau_max + 1.5)
    empirical_fht = np.histogram(empirical_events.FHT, bins=fht_edges)[0]
    empirical_fht = empirical_fht / empirical_fht.sum()
    market_offset = int.from_bytes(market.encode("ascii"), "big")
    empirical_rng = np.random.default_rng(config.seed + market_offset + 900_000)
    empirical_sample = empirical_rng.choice(empirical_values, size=min(args.wasserstein_sample, len(empirical_values)), replace=False)
    pdfs, nonzero_pdfs, curves, fht_pdfs, rows = [], [], [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for replicate in range(1, args.replicates + 1):
        seed = config.seed + market_offset + 300_000 + replicate
        print(f"{market} {interval}dt replicate {replicate}/{args.replicates}: "
              f"{n_paths} paths x {n_days} observations", flush=True)
        returns = simulate_modified_heston(params, SimulationConfig(
            n_paths=n_paths, n_steps=n_days, sample_every=interval,
            burn_in_steps=config.sampling_burn_in_steps, return_mode=config.sampling_return_mode,
            seed=seed, correlated_noise=config.correlated_noise, store_state=False,
            column_prefix=market,
        )).returns
        if args.save_series:
            returns.to_pickle(output / f"{market}_returns_replicate_{replicate:03d}.pkl")
        values = returns.to_numpy(dtype=float, copy=False)
        finite = values[np.isfinite(values)]
        counts, model_prob = probabilities(finite, edges)
        pdfs.append(counts / (counts.sum() * widths) if counts.sum() else np.full(args.bins, np.nan))
        synthetic_nonzero = finite[finite != 0]
        zero_counts, zero_prob = probabilities(synthetic_nonzero, zero_edges)
        nonzero_pdfs.append(zero_counts / (zero_counts.sum() * zero_widths)
                            if zero_counts.sum() else np.full(zero_bins, np.nan))
        synthetic_rng = np.random.default_rng(seed + 600_000)
        synthetic_sample = synthetic_rng.choice(finite, size=min(args.wasserstein_sample, len(finite)), replace=False)
        events = calibrator.count_simulated_events(returns, pair, market)
        comparison = calibrator.curve_comparison(curve_target, events)
        curves.append(np.where(comparison.simulated_bin_covered, comparison.simulated_mfht, np.nan))
        fht_counts = np.histogram(events.FHT, bins=fht_edges)[0]
        fht_pdfs.append(fht_counts / fht_counts.sum() if fht_counts.sum() else np.full(len(fht_days), np.nan))
        return_loss = calibrator.return_moment_loss(returns, target)
        mfht_loss = calibrator.curve_loss(curve_target, events)
        rows.append({
            "market": market, "interval_steps": interval, "replicate": replicate, "seed": seed,
            "n_paths": n_paths, "n_days": n_days,
            "average_stock_mean": float(values.mean(axis=0).mean()),
            "average_stock_std": float(values.std(axis=0, ddof=1).mean()),
            "pooled_mean": float(finite.mean()), "pooled_std": float(finite.std(ddof=1)),
            "near_zero_fraction_0p0025": float((np.abs(finite) <= 0.0025).mean()),
            "exact_zero_fraction": float(np.mean(finite == 0)),
            "return_pdf_total_variation": float(0.5 * np.abs(model_prob - empirical_prob).sum()),
            "return_pdf_nonzero_total_variation": float(0.5 * np.abs(zero_prob - empirical_zero_prob).sum()),
            "return_wasserstein_sampled": float(wasserstein_distance(empirical_sample, synthetic_sample)),
            "n_events": len(events), "return_loss": return_loss, "mfht_loss": mfht_loss,
            "combined_loss": mfht_loss + config.return_loss_weight * return_loss,
        })
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    return_frame = pd.DataFrame({
        "return_lower": edges[:-1], "return_upper": edges[1:],
        "return_midpoint": (edges[:-1] + edges[1:]) / 2,
        "empirical_density": empirical_density, **band_details(pdfs),
    })
    return_frame.to_csv(output / "return_pdf_band.csv", index=False)
    np.savez_compressed(output / "return_pdf_replicates.npz", values=np.asarray(pdfs),
                        edges=edges, replicate=np.arange(1, args.replicates + 1), seed=seeds)
    plot_band(return_frame, "return_midpoint", "empirical_density", output / "return_pdf_band.png",
              "Daily return", "Density", (-0.05, 0.05))
    nonzero_frame = pd.DataFrame({
        "return_lower": zero_edges[:-1], "return_upper": zero_edges[1:],
        "return_midpoint": (zero_edges[:-1] + zero_edges[1:]) / 2,
        "empirical_density": empirical_zero_density, **band_details(nonzero_pdfs),
    })
    nonzero_frame.to_csv(output / "return_pdf_nonzero_band.csv", index=False)
    np.savez_compressed(output / "return_pdf_nonzero_replicates.npz", values=np.asarray(nonzero_pdfs),
                        edges=zero_edges, replicate=np.arange(1, args.replicates + 1), seed=seeds)
    plot_band(nonzero_frame, "return_midpoint", "empirical_density",
              output / "return_pdf_nonzero_band.png", "Nonzero daily return",
              "Conditional density", (-0.05, 0.05))
    curve_frame = pd.DataFrame({
        "volatility_lower": curve_target.edges[:-1], "volatility_upper": curve_target.edges[1:],
        "volatility_midpoint": (curve_target.edges[:-1] + curve_target.edges[1:]) / 2,
        "empirical_events": curve_target.counts, "empirical_mfht": curve_target.mfht,
        "fit_bin": curve_target.eligible, **band_details(curves),
    })
    curve_frame.to_csv(output / "mfht_band.csv", index=False)
    np.savez_compressed(output / "mfht_replicates.npz", values=np.asarray(curves),
                        edges=curve_target.edges, replicate=np.arange(1, args.replicates + 1), seed=seeds)
    plot_band(curve_frame, "volatility_midpoint", "empirical_mfht", output / "mfht_band.png",
              "Local volatility", "MFHT")
    fht_frame = pd.DataFrame({"fht": fht_days, "empirical_probability": empirical_fht,
                              **band_details(fht_pdfs)})
    fht_frame.to_csv(output / "fht_pdf_band.csv", index=False)
    np.savez_compressed(output / "fht_pdf_replicates.npz", values=np.asarray(fht_pdfs),
                        fht=fht_days, replicate=np.arange(1, args.replicates + 1), seed=seeds)
    plot_band(fht_frame, "fht", "empirical_probability", output / "fht_pdf_band.png",
              "FHT (days)", "Probability")
    (output / "metadata.json").write_text(json.dumps({
        "market": market, "interval_steps": interval, "replicates": args.replicates,
        "n_paths": n_paths, "n_days": n_days, "parameters_csv": str(parameters_path),
        "return_pdf_edges": "empirical 0.1% to 99.9% quantiles; displayed density conditional on this range",
        "return_pdf_nonzero_edges": "symmetric empirical nonzero 0.1% to 99.9% span; zero is a bin boundary; density conditional on nonzero returns in range",
        "total_variation": "all returns, including below-range and above-range probability bins",
        "nonzero_total_variation": "conditional on nonzero returns; includes below-range and above-range probability bins",
        "empirical_exact_zero_fraction": float(np.mean(empirical_values == 0)),
        "seed_rule": "config.seed + int.from_bytes(market.encode('ascii'), 'big') + 300000 + replicate",
        "wasserstein": f"exact W1 on deterministic samples of at most {args.wasserstein_sample} returns per distribution; raw daily-return units",
        "band": "sample standard deviation (ddof=1), median, and exact min/max across independent simulations; MFHT bins use covered replicates only",
        "uncertainty": "Monte Carlo variation at fixed fitted parameters",
    }, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path,
                        default=Path("data/processed/heston_calibration/four_market_sampling_sweep"))
    parser.add_argument("--selection", type=Path,
                        help='JSON mapping market to chosen interval, e.g. {"UN": 2, "UW": 5}')
    parser.add_argument("--intervals", type=int, nargs="+", help="Single-market mode: completed intervals by default")
    parser.add_argument("--market", default="UN", help="Market for single-market mode")
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--paths", type=int, help="Optional smoke-run override; default is selected empirical stocks")
    parser.add_argument("--days", type=int, help="Default is pilot_n_steps; pass 11089 for full runs")
    parser.add_argument("--bins", type=int, default=240)
    parser.add_argument("--wasserstein-sample", type=int, default=200_000)
    parser.add_argument("--output-dir", type=Path, help="Default: SWEEP_DIR/replicate_variation")
    parser.add_argument("--save-series", action="store_true")
    args = parser.parse_args()
    if (args.replicates < 2 or args.bins < 2 or args.wasserstein_sample < 2
            or (args.paths is not None and args.paths < 1)
            or (args.days is not None and args.days < 2)):
        parser.error("replicates, bins, days, and Wasserstein sample must be >= 2; paths >= 1")
    sweep = args.sweep_dir.resolve()
    if not sweep.is_dir():
        parser.error(f"sweep directory not found: {sweep}")
    output = (args.output_dir or sweep / "replicate_variation").resolve()
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if not isinstance(selection, dict) or set(selection) != {"UN", "UW", "LN", "JT"}:
            parser.error("selection must map exactly UN, UW, LN, and JT to intervals")
        if any(type(k) is not int or k < 1 for k in selection.values()):
            parser.error("selected intervals must be positive integers")
        jobs = [(market, interval, sweep / market, output / market / f"interval_{interval}")
                for market, interval in selection.items()]
    else:
        intervals = args.intervals or completed_intervals(sweep)
        if not intervals or any(k < 1 for k in intervals):
            parser.error("no completed positive intervals found")
        jobs = [(args.market, interval, sweep, output / f"interval_{interval}") for interval in intervals]
    rows = []
    for market, interval, market_sweep, job_output in jobs:
        rows.extend(run_fit(args, market, interval, market_sweep, job_output))
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "replicate_metrics.csv", index=False)
    columns = ["average_stock_mean", "average_stock_std", "pooled_mean", "pooled_std",
               "near_zero_fraction_0p0025", "exact_zero_fraction", "return_pdf_total_variation",
               "return_pdf_nonzero_total_variation",
               "return_wasserstein_sampled", "n_events", "return_loss", "mfht_loss", "combined_loss"]
    summary = metrics.groupby(["market", "interval_steps"], as_index=False)[columns].agg(["mean", "std"])
    summary.columns = [name if not stat else f"{name}_{stat}" for name, stat in summary.columns]
    summary.to_csv(output / "replicate_summary.csv", index=False)
    print(f"Saved replicate results to {output}")


if __name__ == "__main__":
    main()
