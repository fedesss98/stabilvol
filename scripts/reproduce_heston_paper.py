#!/usr/bin/env python3
"""Reproduce the fixed-parameter theoretical diagnostics of PRE 97, 062307."""

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
from stabilvol.heston.calibration import moments_frame
from stabilvol.heston.paper_reproduction import (
    count_fortran_hitting_events, count_hitting_events, fortran_mfht_curve,
    market_return_scale, mfht_curve,
)


def resolved_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def autocorrelation(values: np.ndarray, max_lag: int, *, absolute: bool = False) -> np.ndarray:
    data = np.abs(values) if absolute else values
    centered = data - data.mean(axis=0, keepdims=True)
    denominator = np.mean(centered * centered)
    if denominator == 0:
        return np.full(max_lag + 1, np.nan)
    result = np.empty(max_lag + 1)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = np.mean(centered[:-lag] * centered[lag:]) / denominator
    return result


def save_figures(
    output_dir: Path,
    returns: np.ndarray,
    events: dict[str, pd.DataFrame],
    curves: dict[str, pd.DataFrame],
    *,
    vol_max: float,
    tau_max: int,
    min_bin_events: int,
    method: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, name, label in zip(axes, ("crash", "rally"), ("Fig. 2(a): crashes", "Fig. 3(a): rallies")):
        curve = curves[name]
        shown = curve.loc[curve["events"] >= min_bin_events]
        ax.scatter(shown["volatility"], shown["mfht"], marker="^", s=9, color="firebrick")
        xlabel = "Fortran reported volatility (half bin position)" if method == "fortran" else "local return volatility"
        ax.set(title=label, xlabel=xlabel, xlim=(0, vol_max), ylim=(0, None))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean first hitting time (steps)")
    fig.savefig(output_dir / "fig2a_fig3a_theoretical_mfht.png", dpi=180)
    plt.close(fig)

    if method == "fortran":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
        for ax, name in zip(axes, ("crash", "rally")):
            shown = curves[name].loc[curves[name]["events"] >= min_bin_events]
            physical = (shown["physical_volatility_lower"] + shown["physical_volatility_upper"]) / 2
            ax.scatter(physical, shown["mfht"], marker="^", s=9, color="firebrick")
            ax.set(title=name.capitalize(), xlabel="physical local return volatility",
                   xlim=(0, 2 * vol_max), ylim=(0, None))
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("mean first hitting time (steps)")
        fig.savefig(output_dir / "mfht_corrected_volatility_axis.png", dpi=180)
        plt.close(fig)

    crash_fht = events["crash"]["FHT"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    if crash_fht.size:
        counts, edges = np.histogram(crash_fht, bins=np.arange(0.5, tau_max + 1.5))
        ax.scatter((edges[:-1] + edges[1:]) / 2, counts / counts.sum(), s=8, color="firebrick")
    ax.set(xlabel="first hitting time (steps)", ylabel="probability", yscale="log", xlim=(0, tau_max),
           title="Fig. 4(b): theoretical crash FHT distribution")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "fig4b_theoretical_fht_pdf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.hist(returns.ravel(), bins=np.linspace(-0.5, 0.5, 201), density=True, histtype="step", color="firebrick")
    ax.set(xlabel="daily model return", ylabel="density", yscale="log", xlim=(-0.5, 0.5),
           title="Fig. 5: theoretical return distribution")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "fig5_theoretical_return_pdf.png", dpi=180)
    plt.close(fig)

    local_vol = events["crash"]["Volatility"].to_numpy(dtype=float)
    local_vol = local_vol[np.isfinite(local_vol) & (local_vol > 0)]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    if local_vol.size:
        ax.hist(local_vol, bins=np.linspace(0, 0.2, 101), density=True, histtype="step", color="firebrick")
    ax.set(xlabel="local return volatility", ylabel="density", xlim=(0, 0.2),
           title="Fig. 6(a): theoretical event-volatility distribution")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "fig6a_theoretical_volatility_pdf.png", dpi=180)
    plt.close(fig)

    max_lag = min(200, returns.shape[0] - 1)
    lags = np.arange(max_lag + 1)
    plain = autocorrelation(returns, max_lag)
    absolute = autocorrelation(returns, max_lag, absolute=True)
    pd.DataFrame({"lag": lags, "return_acf": plain, "absolute_return_acf": absolute}).to_csv(
        output_dir / "autocorrelation.csv", index=False,
    )
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(lags, plain, marker="^", ms=2, lw=0.5, color="firebrick")
    ax.set(xlabel="lag (steps)", ylabel="return correlation", title="Fig. 6(b): theoretical autocorrelation")
    ax.grid(alpha=0.2)
    inset = ax.inset_axes((0.48, 0.48, 0.48, 0.42))
    inset.plot(lags, absolute, lw=0.8, color="firebrick")
    inset.set(title="absolute returns", xlabel="lag", ylim=(0, 1))
    inset.tick_params(labelsize=7)
    fig.savefig(output_dir / "fig6b_theoretical_autocorrelation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1071)
    parser.add_argument("--steps", type=int, default=3030)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tau-min", type=int, default=2)
    parser.add_argument("--tau-max", type=int, default=300)
    parser.add_argument("--bins", type=int, default=500, help="Number of bins for --method clean only")
    parser.add_argument("--vol-max", type=float, default=0.02,
                        help="MFHT plot x limit; also the bin maximum for --method clean")
    parser.add_argument("--min-bin-events", type=int, default=10)
    parser.add_argument("--method", choices=("fortran", "clean"), default="fortran",
                        help="Fortran reproduces calmG.f; clean uses the repository's earlier counter")
    parser.add_argument("--include-end-in-volatility", action="store_true",
                        help="Endpoint sensitivity for --method clean only")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/heston_paper"))
    args = parser.parse_args()
    if args.paths < 1 or args.steps < 2 or args.tau_min < 1 or args.tau_max < args.tau_min:
        parser.error("require paths >= 1, steps >= 2, and 1 <= tau-min <= tau-max")
    if args.bins < 1 or args.vol_max <= 0 or args.min_bin_events < 1:
        parser.error("bins, vol-max, and min-bin-events must be positive")
    if args.method == "fortran" and args.include_end_in_volatility:
        parser.error("the Fortran method always includes the crossing return")

    output_dir = resolved_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = HestonParams()  # U(x)=2x^3+3x^2; CIR a=2, b=.01, c=.83, vstart=8.62e-5.
    simulation = simulate_modified_heston(
        params,
        SimulationConfig(n_paths=args.paths, n_steps=args.steps, seed=args.seed, store_state=False),
    )
    returns = simulation.returns.to_numpy(copy=False)
    if args.method == "fortran":
        # heston.f overwrites the first return of every path before writing its output.
        returns[0, :] = 0.0
    sigma_bar = market_return_scale(returns, ddof=0 if args.method == "fortran" else 1)
    pairs = {"crash": (-0.1, -1.5), "rally": (0.1, 1.5)}
    events: dict[str, pd.DataFrame] = {}
    curves: dict[str, pd.DataFrame] = {}
    event_summary = {}
    for name, pair in pairs.items():
        if args.method == "fortran":
            event_frame = count_fortran_hitting_events(
                returns, sigma_bar, direction=name, tau_min=args.tau_min, tau_max=args.tau_max,
            )
            curve = fortran_mfht_curve(event_frame)
        else:
            event_frame = count_hitting_events(
                returns, pair[0] * sigma_bar, pair[1] * sigma_bar,
                tau_min=args.tau_min, tau_max=args.tau_max,
                include_end_in_volatility=args.include_end_in_volatility,
            )
            curve = mfht_curve(event_frame, bins=args.bins, vol_max=args.vol_max)
        event_frame.to_csv(output_dir / f"{name}_events.csv.gz", index=False)
        curve.to_csv(output_dir / f"{name}_mfht.csv", index=False)
        events[name] = event_frame
        curves[name] = curve
        eligible = curve.loc[curve["events"] >= args.min_bin_events]
        peak = eligible.loc[eligible["mfht"].idxmax()] if not eligible.empty else None
        event_summary[name] = {
            "thresholds_in_sigma_bar_units": pair,
            "events": len(event_frame),
            "events_with_local_volatility": int(event_frame["Volatility"].notna().sum()),
            "mean_fht": float(event_frame["FHT"].mean()) if not event_frame.empty else None,
            "peak_volatility": float(peak["volatility"]) if peak is not None else None,
            "peak_physical_volatility": float((peak["physical_volatility_lower"] + peak["physical_volatility_upper"]) / 2)
            if peak is not None and args.method == "fortran" else None,
            "peak_mfht": float(peak["mfht"]) if peak is not None else None,
        }
        print(f"{name}: {len(event_frame)} FHT events; peak at {event_summary[name]['peak_volatility']}", flush=True)

    save_figures(
        output_dir, returns, events, curves,
        vol_max=args.vol_max, tau_max=args.tau_max, min_bin_events=args.min_bin_events,
        method=args.method,
    )
    summary = {
        "paper_doi": "10.1103/PhysRevE.97.062307",
        "paper_reference_sigma_bar": 0.02383,
        "model_sigma_bar": sigma_bar,
        "model_return_std": float(np.std(returns)),
        "return_moments": moments_frame(simulation.returns),
        "parameters": asdict(params),
        "fortran_binning": {"sigma_max": 0.2, "num_bin": 5000, "max_bin": 1000}
        if args.method == "fortran" else None,
        "run": {**vars(args), "output_dir": str(output_dir)},
        "events": event_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"sigma_bar={sigma_bar:.6g}; results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
