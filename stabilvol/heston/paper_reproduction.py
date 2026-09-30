"""Measurements used for the fixed-parameter theoretical results in PRE 97, 062307."""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_return_scale(returns: np.ndarray, *, ddof: int = 1) -> float:
    """Mean per-path standard deviation; the Fortran counter uses ``ddof=0``."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("returns must have at least two steps and one path")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    if ddof < 0 or ddof >= values.shape[0]:
        raise ValueError("ddof must be non-negative and smaller than the path length")
    return float(np.std(values, axis=0, ddof=ddof).mean())


def count_fortran_hitting_events(
    returns: np.ndarray,
    sigma_bar: float,
    *,
    direction: str = "crash",
    start_sigma: float = -0.1,
    start_upper_sigma: float = 100.0,
    end_sigma: float = -1.5,
    anomaly_limit: float = 100.0,
    tau_min: int = 2,
    tau_max: int = 300,
) -> pd.DataFrame:
    """Port the state transitions and local volatility of ``old_code/calmG.f``.

    The local window excludes the start return and includes the crossing return.
    The original leaves sums/counters intact after an ineligible crossing; that
    behavior is preserved. Rally counting uses sign-reversed returns because the
    supplied Fortran program and parameter file implement crashes only.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite two-dimensional array")
    if direction not in ("crash", "rally"):
        raise ValueError("direction must be 'crash' or 'rally'")
    if sigma_bar <= 0 or tau_min < 1 or tau_max < tau_min:
        raise ValueError("require sigma_bar > 0 and 1 <= tau_min <= tau_max")
    if not end_sigma < start_sigma < start_upper_sigma:
        raise ValueError("require end_sigma < start_sigma < start_upper_sigma")

    n_paths = values.shape[1]
    sign = 1.0 if direction == "crash" else -1.0
    active = np.zeros(n_paths, dtype=bool)
    calm = np.zeros(n_paths, dtype=np.int32)
    counts = np.zeros(n_paths, dtype=np.int32)
    sums = np.zeros(n_paths, dtype=float)
    squares = np.zeros(n_paths, dtype=float)
    starts = np.zeros(n_paths, dtype=np.int32)
    event_parts: list[pd.DataFrame] = []

    for step in range(values.shape[0]):
        raw = values[step]
        # In calmG.f the sums are updated before anomaly clipping and before
        # testing the terminal crossing.
        sums[active] += raw[active]
        squares[active] += raw[active] ** 2
        counts[active] += 1
        row = sign * np.where(np.abs(raw) > anomaly_limit, 0.0, raw)
        begin = ~active & (row > start_sigma * sigma_bar) & (row < start_upper_sigma * sigma_bar)
        starts[begin] = step
        active |= begin
        finish = active & (row <= end_sigma * sigma_bar)
        valid = finish & (calm >= tau_min) & (calm <= tau_max)
        if np.any(valid):
            paths = np.flatnonzero(valid)
            means = sums[paths] / counts[paths]
            volatility = np.sqrt(np.maximum(squares[paths] / counts[paths] - means**2, 0.0))
            event_parts.append(pd.DataFrame({
                "Volatility": volatility,
                "FHT": calm[paths].copy(),
                "path": paths,
                "start_step": starts[paths].copy(),
                "end_step": np.full(paths.size, step, dtype=np.int32),
                "volatility_observations": counts[paths].copy(),
            }))
            counts[valid] = 0
            sums[valid] = 0.0
            squares[valid] = 0.0
            calm[valid] = 0
        active[finish] = False
        calm[active] += 1

    columns = ["Volatility", "FHT", "path", "start_step", "end_step", "volatility_observations"]
    return pd.concat(event_parts, ignore_index=True)[columns] if event_parts else pd.DataFrame(columns=columns)


def fortran_mfht_curve(
    events: pd.DataFrame, *, sigma_max: float = 0.2, num_bin: int = 5000, max_bin: int = 1000,
) -> pd.DataFrame:
    """Recreate ``calmG.f`` bins and expose its halved printed coordinate.

    The printed x is ``i*delta/2`` whereas bin ``i`` actually contains local
    volatility from ``(i-1)*delta`` to ``i*delta``.
    """
    if sigma_max <= 0 or num_bin < 1 or max_bin < 1:
        raise ValueError("sigma_max, num_bin and max_bin must be positive")
    delta = sigma_max / num_bin
    volatility = events["Volatility"].to_numpy(dtype=float)
    fht = events["FHT"].to_numpy(dtype=float)
    finite = np.isfinite(volatility) & np.isfinite(fht)
    indices = np.full(volatility.shape, -1, dtype=int)
    indices[finite] = np.floor(volatility[finite] / delta).astype(int)
    valid = finite & (indices >= 0) & (indices < max_bin)
    counts = np.bincount(indices[valid], minlength=max_bin)
    totals = np.bincount(indices[valid], weights=fht[valid], minlength=max_bin)
    means = np.divide(totals, counts, out=np.full(max_bin, np.nan), where=counts > 0)
    bin_number = np.arange(1, max_bin + 1)
    return pd.DataFrame({
        "volatility": bin_number * delta / 2,  # As printed by the original Fortran.
        "physical_volatility_lower": (bin_number - 1) * delta,
        "physical_volatility_upper": bin_number * delta,
        "mfht": means,
        "events": counts,
    })


def count_hitting_events(
    returns: np.ndarray,
    start_level: float,
    end_level: float,
    *,
    tau_min: int = 2,
    tau_max: int = 300,
    include_end_in_volatility: bool = False,
) -> pd.DataFrame:
    """Count disjoint first crossings in each path and measure local volatility.

    The default volatility window is [start, end), matching the repository's
    existing StabilVolter counter. The paper does not specify endpoint inclusion.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite two-dimensional array")
    if tau_min < 1 or tau_max < tau_min:
        raise ValueError("require 1 <= tau_min <= tau_max")
    if start_level == end_level:
        raise ValueError("start and end thresholds must differ")

    # The existing counter reverses rallies and then applies the crash rule.
    sign = -1.0 if end_level > start_level else 1.0
    start_threshold = sign * start_level
    end_threshold = sign * end_level
    n_steps, n_paths = values.shape
    active = np.zeros(n_paths, dtype=bool)
    starts = np.zeros(n_paths, dtype=np.int32)
    path_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []

    for step in range(n_steps):
        row = sign * values[step]
        begin = ~active & (row > start_threshold)
        starts[begin] = step
        active[begin] = True
        finish = active & (row < end_threshold)
        if np.any(finish):
            paths = np.flatnonzero(finish)
            durations = step - starts[paths]
            keep = (durations >= tau_min) & (durations <= tau_max)
            if np.any(keep):
                selected = paths[keep]
                path_parts.append(selected)
                start_parts.append(starts[selected].copy())
                end_parts.append(np.full(selected.size, step, dtype=np.int32))
            active[paths] = False

    columns = ["Volatility", "FHT", "path", "start_step", "end_step"]
    if not path_parts:
        return pd.DataFrame(columns=columns)

    paths = np.concatenate(path_parts)
    event_starts = np.concatenate(start_parts)
    event_ends = np.concatenate(end_parts)
    fht = event_ends - event_starts
    stop = event_ends + int(include_end_in_volatility)
    sample_sizes = stop - event_starts

    # Prefix sums avoid repeated slicing of every event window.
    sums = np.empty((n_steps + 1, n_paths), dtype=float)
    sums[0] = 0.0
    np.cumsum(values, axis=0, out=sums[1:])
    window_sum = sums[stop, paths] - sums[event_starts, paths]
    np.square(values, out=sums[1:])
    np.cumsum(sums[1:], axis=0, out=sums[1:])
    window_squares = sums[stop, paths] - sums[event_starts, paths]
    variance = np.full(sample_sizes.shape, np.nan, dtype=float)
    valid = sample_sizes >= 2
    variance[valid] = np.maximum(
        (window_squares[valid] - window_sum[valid] ** 2 / sample_sizes[valid])
        / (sample_sizes[valid] - 1),
        0.0,
    )
    return pd.DataFrame({
        "Volatility": np.sqrt(variance),
        "FHT": fht,
        "path": paths,
        "start_step": event_starts,
        "end_step": event_ends,
    })[columns]


def mfht_curve(events: pd.DataFrame, *, bins: int = 500, vol_max: float = 0.02) -> pd.DataFrame:
    """Equal-width local-volatility bins with event counts and mean FHT."""
    if bins < 1 or vol_max <= 0:
        raise ValueError("bins and vol_max must be positive")
    edges = np.linspace(0.0, vol_max, bins + 1)
    volatility = events["Volatility"].to_numpy(dtype=float)
    fht = events["FHT"].to_numpy(dtype=float)
    valid = np.isfinite(volatility) & np.isfinite(fht)
    counts, _ = np.histogram(volatility[valid], bins=edges)
    totals, _ = np.histogram(volatility[valid], bins=edges, weights=fht[valid])
    means = np.divide(totals, counts, out=np.full(bins, np.nan), where=counts > 0)
    return pd.DataFrame({
        "volatility": (edges[:-1] + edges[1:]) / 2,
        "mfht": means,
        "events": counts,
    })
