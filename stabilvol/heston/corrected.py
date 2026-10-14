"""Corrected episode measurements and resolved, constrained MFHT targets.

Separate from the historical Fortran statistic. Volatility is physical return
SD; event distributions are conditional on completed durations in [2, 300].
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

COUNTER_VERSION = "independent-episodes-v1"
PARAMETER_NAMES = ("a", "b", "aa", "bb", "cc")


@dataclass(frozen=True)
class CorrectedConfig:
    sigma: float
    substeps: int = 10
    burn_in: int = 250  # Observations, not internal steps.
    tau_min: int = 2
    tau_max: int = 300
    start: float = -0.1
    end: float = -1.5
    bin_width: float = 0.0005
    vol_max: float = 0.04
    min_events: int = 200
    min_stocks: int = 20
    min_synthetic_events: int = 30
    bootstraps: int = 300
    bootstrap_seed: int = 19771
    seeds: tuple[int, ...] = (1001, 1002)
    fixed_parameters: dict[str, float] = field(default_factory=lambda: {"start": 0.0, "rho": 0.0})
    mean_tolerance: float = 0.05
    scale_tolerance: float = 0.15
    quantile_tolerance: float = 0.15
    tail_relative_tolerance: float = 0.25
    acf_tolerance: float = 0.05

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if self.substeps < 1 or self.burn_in < 0 or not 1 <= self.tau_min <= self.tau_max:
            raise ValueError("invalid simulation/counter settings")
        if self.start == self.end or self.bin_width <= 0 or self.vol_max <= 0:
            raise ValueError("invalid thresholds/bins")
        if min(self.min_events, self.min_stocks, self.min_synthetic_events) < 1 or self.bootstraps < 2:
            raise ValueError("invalid minimum counts/bootstrap count")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("simulation seeds must be nonempty and unique")
        if any(v < 0 or not np.isfinite(v) for v in (self.mean_tolerance, self.scale_tolerance,
               self.quantile_tolerance, self.tail_relative_tolerance, self.acf_tolerance)):
            raise ValueError("constraint tolerances must be finite and nonnegative")
        if set(self.fixed_parameters) - {"a", "b", "aa", "bb", "cc", "vstart", "start", "rho", "reset_threshold"}:
            raise ValueError("unknown fixed parameter")
        if self.fixed_parameters.get("rho", 0) != 0 or self.fixed_parameters.get("start", 0) != 0:
            raise ValueError("corrected baseline requires rho=0 and start=0")


@dataclass
class EpisodeResult:
    events: pd.DataFrame
    episodes: pd.DataFrame


def count_episodes(values: np.ndarray, config: CorrectedConfig) -> EpisodeResult:
    """Disjoint crossings with explicit rejection and censoring records.

    No clipping or synthetic anomaly substitution. A completed overlong event
    is cleared at its terminal crossing, not prematurely restarted at day 300.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError("returns must be a two-dimensional array without infinities")
    n = values.shape[1]
    active = np.zeros(n, bool)
    start = np.zeros(n, int)
    counts = np.zeros(n, int)
    means = np.zeros(n)
    m2 = np.zeros(n)
    parts = []
    sign = 1 if config.end < config.start else -1

    def record(mask, step, reason):
        paths = np.flatnonzero(mask)
        if not len(paths):
            return
        duration = counts[paths].copy()
        status = np.full(len(paths), reason, dtype=object)
        if reason == "completed":
            status[:] = "accepted"
            status[duration < config.tau_min] = "too_short"
            status[duration > config.tau_max] = "overlong"
        variance = np.divide(m2[paths], duration, out=np.zeros(len(paths)), where=duration > 0)
        parts.append(pd.DataFrame({"path": paths, "start_step": start[paths],
            "end_step": step, "FHT": duration, "Volatility": np.sqrt(np.maximum(variance, 0)),
            "status": status}))
        active[paths] = False
        counts[paths] = 0
        means[paths] = m2[paths] = 0

    for step, row in enumerate(values):
        missing = np.isnan(row)
        record(active & missing, step, "missing")
        on = active.copy()
        counts[on] += 1
        delta = row[on] - means[on]
        means[on] += delta / counts[on]
        m2[on] += delta * (row[on] - means[on])
        finish = on & (sign * row <= sign * config.end * config.sigma)
        record(finish, step, "completed")
        begin = ~active & ~finish & ~missing & (sign * row > sign * config.start * config.sigma)
        active[begin] = True
        start[begin] = step
    record(active, len(values), "unfinished")
    columns = ["path", "start_step", "end_step", "FHT", "Volatility", "status"]
    episodes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    events = episodes.loc[episodes.status == "accepted"].copy()
    return EpisodeResult(events, episodes)


def event_matrices(events, edges, n_paths):
    count = np.zeros((n_paths, len(edges) - 1))
    total = np.zeros_like(count)
    x = events.Volatility.to_numpy(float)
    bins = np.searchsorted(edges, x, side="right") - 1
    bins[x == edges[-1]] = len(edges) - 2
    valid = np.isfinite(x) & (bins >= 0) & (bins < len(edges) - 1)
    paths = events.path.to_numpy(int)[valid]
    np.add.at(count, (paths, bins[valid]), 1)
    np.add.at(total, (paths, bins[valid]), events.FHT.to_numpy(float)[valid])
    return count, total


def bin_events(events, edges):
    counts = np.histogram(events.Volatility, edges)[0]
    totals = np.histogram(events.Volatility, edges, weights=events.FHT)[0]
    return counts, totals


@dataclass
class ResolvedTarget:
    edges: np.ndarray
    counts: np.ndarray
    means: np.ndarray
    log_se: np.ndarray
    weights: np.ndarray
    regions: np.ndarray
    mean_ci: np.ndarray
    peak_height_ci: np.ndarray
    peak_interval: np.ndarray

    def frame(self):
        return pd.DataFrame({"lower": self.edges[:-1], "upper": self.edges[1:],
            "events": self.counts, "mfht": self.means, "log_se": self.log_se,
            "weight": self.weights, "region": self.regions,
            "ci_lower": self.mean_ci[0], "ci_upper": self.mean_ci[1]})


def make_target(events, n_paths, config, edges=None):
    if edges is None:
        fine_edges = np.linspace(0, config.vol_max, int(np.ceil(config.vol_max / config.bin_width)) + 1)
        count, total = event_matrices(events, fine_edges, n_paths)
        edges = [0.0]
        cursor = 0
        for stop in range(1, len(fine_edges)):
            pooled = count[:, cursor:stop].sum(axis=1)
            if pooled.sum() >= config.min_events and np.count_nonzero(pooled) >= config.min_stocks:
                edges.append(float(fine_edges[stop]))
                cursor = stop
        if len(edges) < 2:
            raise ValueError("not enough events/stocks to create a resolved target")
        if edges[-1] != config.vol_max:
            edges[-1] = config.vol_max  # Merge unsupported final remainder backwards.
    edges = np.asarray(edges, float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("target edges must increase")
    count, total = event_matrices(events, edges, n_paths)
    counts, totals = count.sum(0), total.sum(0)
    if np.any(counts == 0):
        raise ValueError("held-out target has empty fixed bins; report unsupported target")
    means = totals / counts
    rng = np.random.default_rng(config.bootstrap_seed)
    multiplicity = rng.multinomial(n_paths, np.full(n_paths, 1 / n_paths), size=config.bootstraps)
    boot_counts = multiplicity @ count
    boot_total = multiplicity @ total
    boot = np.divide(boot_total, boot_counts, out=np.full_like(boot_total, np.nan), where=boot_counts > 0)
    log_se = np.nanstd(np.log(boot), axis=0, ddof=1)
    peak_bins = np.flatnonzero(means >= 0.9 * means.max())
    regions = np.where(np.arange(len(means)) < peak_bins[0], 0,
                       np.where(np.arange(len(means)) > peak_bins[-1], 2, 1))
    weights = np.zeros(len(means))
    for region in np.unique(regions):
        mask = regions == region
        weights[mask] = 1 / (len(np.unique(regions)) * mask.sum())
    boot_max = np.nanmax(boot, axis=1)
    left, right = [], []
    for sample, maximum in zip(boot, boot_max):
        peaks = np.flatnonzero(sample >= 0.9 * maximum)
        left.append(edges[peaks[0]])
        right.append(edges[peaks[-1] + 1])
    return ResolvedTarget(edges, counts, means, log_se, weights, regions,
        np.nanquantile(boot, [0.025, 0.975], axis=0), np.quantile(boot_max, [0.025, 0.975]),
        np.array([np.quantile(left, 0.025), np.quantile(right, 0.975)]))


def score_curve(target, counts, totals, minimum=30):
    means = np.divide(totals, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    covered = counts >= minimum
    log_error = np.zeros(len(counts))
    log_error[covered] = np.log(means[covered] / target.means[covered])
    residual = 10 + np.maximum(0, minimum - counts) / minimum
    residual[covered] = log_error[covered] / np.maximum(0.10, target.log_se[covered])
    shape = float(np.sum(target.weights * residual**2))
    # Unsupported bins remain a failure in both the loss and acceptance metric.
    raw = log_error.copy()
    raw[~covered] = 1.0 + np.maximum(0, minimum - counts[~covered]) / minimum
    finite_means = np.where(covered, means, -np.inf)
    maximum = float(finite_means.max())
    peak = np.flatnonzero(covered & (means >= 0.9 * maximum))
    height_pass = (abs(maximum / target.means.max() - 1) <= 0.1
                   or target.peak_height_ci[0] <= maximum <= target.peak_height_ci[1])
    overlap = bool(len(peak) and target.edges[peak[-1] + 1] >= target.peak_interval[0]
                   and target.edges[peak[0]] <= target.peak_interval[1])
    rms = float(np.sqrt(np.sum(target.weights * raw**2)))
    coverage = float(covered.mean())
    return {"mfht_loss": shape, "rms_log_mfht": rms, "coverage": coverage,
        "peak_height": maximum if np.isfinite(maximum) else None,
        "peak_height_pass": bool(height_pass), "peak_overlap_pass": overlap,
        "mfht_pass": bool(rms <= 0.10 and coverage >= 0.95 and height_pass and overlap),
        "means": means.tolist(), "counts": counts.tolist()}


def stock_acf(values, lag, absolute=False):
    """Average stock-specific Pearson correlation on finite calendar pairs."""
    if values.shape[1] > 64:
        return np.concatenate([stock_acf(values[:, start:start + 64], lag, absolute)
                               for start in range(0, values.shape[1], 64)])
    x = np.abs(values) if absolute else values
    a, b = x[:-lag], x[lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(0)
    av = np.where(ok, a, 0)
    bv = np.where(ok, b, 0)
    sa, sb = av.sum(0), bv.sum(0)
    den_n = np.maximum(n, 1)
    covariance = (av * bv).sum(0) - sa * sb / den_n
    variance_a = (av * av).sum(0) - sa**2 / den_n
    variance_b = (bv * bv).sum(0) - sb**2 / den_n
    denominator = np.sqrt(np.maximum(variance_a * variance_b, 0))
    return np.divide(covariance, denominator, out=np.full(x.shape[1], np.nan), where=(n >= 3) & (denominator > 0))


def return_features(values, tail_thresholds=None):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        raise ValueError("too few finite returns")
    thresholds = np.quantile(finite, [0.001, 0.999]) if tail_thresholds is None else np.asarray(tail_thresholds)
    features = {"mean": float(finite.mean()), "sd": float(finite.std()),
        "stock_sd": float(np.nanmedian(np.nanstd(values, axis=0))),
        "iqr": float(np.diff(np.quantile(finite, [0.25, 0.75]))[0]),
        "zero_fraction": float(np.mean(finite == 0)),
        "tail_low": float(np.mean(finite < thresholds[0])),
        "tail_high": float(np.mean(finite > thresholds[1]))}
    for q, value in zip((1, 5, 50, 95, 99), np.quantile(finite, [.01, .05, .5, .95, .99])):
        features[f"q{q}"] = float(value)
    stock = {}
    for absolute in (False, True):
        for lag in (1, 5, 20):
            key = f"{'abs_' if absolute else ''}acf{lag}"
            stock[key] = stock_acf(values, lag, absolute)
            features[key] = float(np.nanmean(stock[key]))
    return features, thresholds, stock


@dataclass
class ReturnConstraints:
    empirical: dict
    tolerances: dict
    tail_thresholds: np.ndarray
    activity_transitions: tuple[float, float]


def make_constraints(values, config):
    empirical, thresholds, acf = return_features(values)
    n_paths = values.shape[1]
    rng = np.random.default_rng(config.bootstrap_seed + 1)
    multiplicity = rng.multinomial(n_paths, np.full(n_paths, 1 / n_paths), size=config.bootstraps)
    tolerances = {"mean": config.mean_tolerance * empirical["sd"],
        "stock_sd": config.scale_tolerance * empirical["stock_sd"],
        "iqr": config.scale_tolerance * empirical["iqr"]}
    tolerances.update({f"q{q}": config.quantile_tolerance * empirical["sd"] for q in (1, 5, 50, 95, 99)})
    finite_counts = np.isfinite(values).sum(0)
    for key, mask in (("tail_low", values < thresholds[0]), ("tail_high", values > thresholds[1])):
        samples = (multiplicity @ mask.sum(0)) / np.maximum(multiplicity @ finite_counts, 1)
        tolerances[key] = max(3 * samples.std(ddof=1), config.tail_relative_tolerance * empirical[key], 1e-12)
    for key, per_stock in acf.items():
        valid = np.isfinite(per_stock)
        samples = (multiplicity @ np.nan_to_num(per_stock)) / np.maximum(multiplicity @ valid.astype(float), 1)
        tolerances[key] = max(3 * samples.std(ddof=1), config.acf_tolerance)
    valid_pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    active = values != 0
    from_active = valid_pairs & active[:-1]
    from_stale = valid_pairs & ~active[:-1]
    to_stale = float(np.sum(from_active & ~active[1:]) / max(from_active.sum(), 1))
    to_active = float(np.sum(from_stale & active[1:]) / max(from_stale.sum(), 1))
    if to_stale + to_active == 0:
        to_active = 1.0
    return ReturnConstraints(empirical, tolerances, thresholds, (to_stale, to_active))


def score_constraints(features, target):
    rows = {}
    for key, tolerance in target.tolerances.items():
        value = features.get(key, np.nan)
        error = abs(value - target.empirical[key])
        ratio = error / max(tolerance, 1e-12) if np.isfinite(error) else float("inf")
        rows[key] = {"empirical": target.empirical[key], "synthetic": value,
                     "tolerance": tolerance, "normalized_error": ratio, "pass": bool(ratio <= 1)}
    violation = float(sum(max(0, r["normalized_error"] - 1) for r in rows.values()))
    return {"feasible": bool(violation == 0), "violation": violation, "constraints": rows}


def zero_run_summary(values):
    lengths = []
    for column in np.asarray(values).T:
        zeros = np.r_[False, column == 0, False]
        starts = np.flatnonzero(~zeros[:-1] & zeros[1:])
        stops = np.flatnonzero(zeros[:-1] & ~zeros[1:])
        lengths.extend(stops - starts)
    return {"runs": len(lengths), "quantiles": np.quantile(lengths, [.5, .9, .99]).tolist() if lengths else [0, 0, 0]}
