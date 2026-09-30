#!/usr/bin/env python3
"""Summarize independent seed runs of reproduce_heston_paper.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing seed-*/summary.json")
    parser.add_argument("--min-bin-events", type=int, help="Minimum events in each seed/bin to include its MFHT")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summaries = sorted(run_dir.glob("seed-*/summary.json"))
    if not summaries:
        parser.error(f"no seed-*/summary.json files found under {run_dir}")
    if args.min_bin_events is not None and args.min_bin_events < 1:
        parser.error("min-bin-events must be positive")

    rows = []
    curves = {"crash": [], "rally": []}
    minimum = args.min_bin_events
    reference = None
    seen_seeds = set()
    for summary_path in summaries:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        seed = int(summary["run"]["seed"])
        if seed in seen_seeds:
            parser.error(f"duplicate seed {seed} under {run_dir}")
        seen_seeds.add(seed)
        settings = (
            summary["parameters"],
            *(summary["run"][key] for key in (
                "paths", "steps", "tau_min", "tau_max", "bins", "vol_max",
                "include_end_in_volatility", "method",
            )),
            summary.get("fortran_binning"),
        )
        if reference is None:
            reference = settings
        elif settings != reference:
            parser.error(f"incompatible model or measurement settings in {summary_path}")
        if minimum is None:
            minimum = int(summary["run"]["min_bin_events"])
        rows.append({
            "seed": seed,
            "sigma_bar": summary["model_sigma_bar"],
            "return_std": summary["model_return_std"],
            "return_skewness": summary["return_moments"]["skewness"],
            "return_kurtosis": summary["return_moments"]["kurtosis"],
            "crash_peak_volatility": summary["events"]["crash"]["peak_volatility"],
            "crash_peak_physical_volatility": summary["events"]["crash"].get("peak_physical_volatility"),
            "crash_peak_mfht": summary["events"]["crash"]["peak_mfht"],
            "rally_peak_volatility": summary["events"]["rally"]["peak_volatility"],
            "rally_peak_physical_volatility": summary["events"]["rally"].get("peak_physical_volatility"),
            "rally_peak_mfht": summary["events"]["rally"]["peak_mfht"],
        })
        for name in curves:
            frame = pd.read_csv(summary_path.parent / f"{name}_mfht.csv")
            frame.loc[frame["events"] < minimum, "mfht"] = float("nan")
            frame["seed"] = seed
            curves[name].append(frame)

    pd.DataFrame(rows).sort_values("seed").to_csv(run_dir / "seed_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, name in zip(axes, curves):
        pooled = pd.concat(curves[name], ignore_index=True)
        grouped = pooled.groupby("volatility", sort=True).agg(
            mean_mfht=("mfht", "mean"),
            sd_mfht=("mfht", "std"),
            contributing_seeds=("mfht", "count"),
            total_events=("events", "sum"),
        ).reset_index()
        grouped.to_csv(run_dir / f"{name}_mfht_across_seeds.csv", index=False)
        shown = grouped[grouped["contributing_seeds"] > 0]
        x = shown["volatility"].to_numpy()
        y = shown["mean_mfht"].to_numpy()
        sd = shown["sd_mfht"].fillna(0).to_numpy()
        ax.plot(x, y, color="firebrick", linewidth=1.2)
        if len(summaries) > 1:
            ax.fill_between(x, (y - sd).clip(min=0), y + sd, color="firebrick", alpha=0.18)
        xlabel = "Fortran reported volatility (half bin position)" if summary["run"]["method"] == "fortran" else "local return volatility"
        ax.set(title=name.capitalize(), xlabel=xlabel, xlim=(0, 0.02), ylim=(0, None))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean first hitting time (steps)")
    fig.savefig(run_dir / "mfht_across_seeds.png", dpi=180)
    plt.close(fig)
    print(f"Summarized {len(summaries)} seeds in {run_dir}")


if __name__ == "__main__":
    main()
