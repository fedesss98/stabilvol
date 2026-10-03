"""Calibration helpers for fitting modified Heston simulations to empirical FHT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import ExitStack, closing
from functools import partial
import multiprocessing as mp
from pathlib import Path
import sqlite3
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import ks_2samp

from stabilvol.utility.classes.stability_analysis import StabilVolter

from .simulation import HestonParams, SimulationConfig, SimulationError, simulate_modified_heston


DEFAULT_MARKETS = ("UN", "UW", "LN", "JT")
DEFAULT_THRESHOLD_PAIRS = ((-0.5, -1.5), (-1.0, -2.0), (0.5, 1.5), (1.0, 2.0))
UNCORRELATED_PARAMETER_NAMES = ("a", "b", "aa", "bb", "cc", "vstart")
CORRELATED_PARAMETER_NAMES = ("a", "b", "aa", "bb", "cc", "rho", "vstart")
PARAMETER_NAMES = CORRELATED_PARAMETER_NAMES
DEFAULT_BOUNDS = {
    "a": (0.2, 8.0),
    "b": (0.1, 10.0),
    "aa": (0.05, 8.0),
    "bb": (1e-5, 0.2),
    "cc": (0.01, 3.0),
    "rho": (-0.95, 0.95),
    "vstart": (1e-6, 0.05),
}


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for Heston-to-empirical MFHT calibration."""

    root: Path = Path(".")
    database_path: Path = Path("data/processed/trapezoidal_selection/stabilvol_filtered.sqlite")
    markets: tuple[str, ...] = DEFAULT_MARKETS
    threshold_pairs: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLD_PAIRS
    start_date: str = "1980-01-01"
    end_date: str = "2022-07-01"
    min_empirical_observations: int = 0
    vol_limit: float = 100.0
    tau_min: int = 2
    tau_max: int = 30
    pilot_max_paths: int = 512
    pilot_n_steps: int = 3030
    full_n_steps: int = 11089
    sampling_interval_steps: int = 1
    sampling_burn_in_steps: int = 0
    sampling_return_mode: str = "pre_reset_step"
    n_vol_bins: int = 40
    event_count_weight: float = 0.1
    empty_penalty: float = 10.0
    min_empirical_bin_events: int = 20
    min_simulated_bin_events: int = 5
    loss_metric: str = "mfht_curve"
    empirical_source: str = "database"
    threshold_sigma: Optional[float] = None
    return_loss_weight: float = 0.0
    seed: int = 12345
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    base_params: HestonParams = field(default_factory=HestonParams)
    initial_params: Optional[HestonParams] = None
    count_method: str = "quiet_pandas"
    std_normalization: bool = True
    correlated_noise: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "markets", tuple(self.markets))
        object.__setattr__(self, "threshold_pairs", tuple(tuple(pair) for pair in self.threshold_pairs))
        object.__setattr__(self, "bounds", {key: tuple(value) for key, value in self.bounds.items()})
        if self.loss_metric not in ("mfht_curve", "fht_distribution", "return_moments"):
            raise ValueError("loss_metric must be mfht_curve, fht_distribution, or return_moments")
        if self.empirical_source not in ("database", "returns"):
            raise ValueError("empirical_source must be database or returns")
        if self.threshold_sigma is not None and self.threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")
        if self.return_loss_weight < 0:
            raise ValueError("return_loss_weight must be nonnegative")
        if self.n_vol_bins < 1 or self.min_empirical_bin_events < 1 or self.min_simulated_bin_events < 1:
            raise ValueError("bin counts and minimum events per bin must be positive")
        if self.min_empirical_observations < 0:
            raise ValueError("min_empirical_observations must be nonnegative")
        if self.sampling_interval_steps < 1 or self.sampling_burn_in_steps < 0:
            raise ValueError("sampling interval must be positive and burn-in nonnegative")
        if self.sampling_return_mode not in ("pre_reset_step", "sampled_x"):
            raise ValueError("sampling_return_mode must be pre_reset_step or sampled_x")
        if self.sampling_interval_steps > 1 and self.sampling_return_mode != "sampled_x":
            raise ValueError("sampling intervals above one require sampled_x returns")
        if self.event_count_weight < 0 or self.empty_penalty <= 0:
            raise ValueError("event_count_weight must be nonnegative and empty_penalty positive")

    @property
    def absolute_database_path(self) -> Path:
        return self.database_path if self.database_path.is_absolute() else self.root / self.database_path


@dataclass
class CalibrationResult:
    market: str
    params: HestonParams
    pilot_loss: float
    validation_loss: Optional[float]
    optimization_success: bool
    optimization_message: str
    n_paths_pilot: int
    n_paths_full: int
    n_steps_pilot: int
    n_steps_full: int

    def to_record(self) -> dict[str, object]:
        record = {
            "market": self.market,
            "pilot_loss": self.pilot_loss,
            "validation_loss": self.validation_loss,
            "optimization_success": self.optimization_success,
            "optimization_message": self.optimization_message,
            "n_paths_pilot": self.n_paths_pilot,
            "n_paths_full": self.n_paths_full,
            "n_steps_pilot": self.n_steps_pilot,
            "n_steps_full": self.n_steps_full,
        }
        record.update(self.params.to_dict())
        return record


@dataclass(frozen=True)
class ObjectivePayload:
    config: CalibrationConfig
    market: str
    empirical_grid: dict[tuple[float, float], pd.DataFrame | CurveTarget]
    n_paths: int
    n_steps: int
    seed: int
    return_target: Optional[ReturnTarget] = None


@dataclass(frozen=True)
class CurveTarget:
    """Fixed empirical volatility bins reused for every candidate and validation."""

    edges: np.ndarray
    counts: np.ndarray
    mfht: np.ndarray
    eligible: np.ndarray
    fht_scale: float


@dataclass(frozen=True)
class ReturnTarget:
    """Across-stock averages of daily-return mean and standard deviation."""

    mean: float
    std: float
    n_series: int


def stringify_threshold(value: float) -> str:
    return str(round(float(value), 2)).replace("-", "m").replace(".", "p")


def table_name_for_thresholds(start_level: float, end_level: float) -> str:
    return f"stabilvol_{stringify_threshold(start_level)}_{stringify_threshold(end_level)}"


def moments_frame(returns: pd.DataFrame) -> dict[str, float]:
    values = returns.to_numpy(dtype=float, copy=False).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "variance": np.nan, "skewness": np.nan, "kurtosis": np.nan}
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=0))
    sigma = float(np.sqrt(variance))
    if sigma == 0:
        return {"mean": mean, "variance": variance, "skewness": 0.0, "kurtosis": 0.0}
    centered = values - mean
    skewness = float(np.mean(centered**3) / sigma**3)
    kurtosis = float(np.mean(centered**4) / sigma**4)
    return {"mean": mean, "variance": variance, "skewness": skewness, "kurtosis": kurtosis}


def heston_objective(vector: np.ndarray, payload: ObjectivePayload) -> float:
    """Pickleable objective function for scipy multiprocessing workers."""

    calibrator = HestonCalibrator(payload.config)
    return calibrator.objective(
        vector,
        market=payload.market,
        empirical_grid=payload.empirical_grid,
        n_paths=payload.n_paths,
        n_steps=payload.n_steps,
        seed=payload.seed,
        return_target=payload.return_target,
    )


_WORKER_PAYLOAD: Optional[ObjectivePayload] = None


def _init_heston_worker(payload: ObjectivePayload) -> None:
    global _WORKER_PAYLOAD
    _WORKER_PAYLOAD = payload


def _heston_worker_objective(vector: np.ndarray) -> float:
    if _WORKER_PAYLOAD is None:
        raise RuntimeError("Heston optimizer worker was not initialized")
    return heston_objective(vector, _WORKER_PAYLOAD)


class HestonCalibrator:
    """Fit modified Heston parameters against empirical StabilVol FHT data."""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self._empirical_cache: dict[tuple[str, tuple[float, float]], pd.DataFrame] = {}
        self._returns_cache: dict[str, pd.DataFrame] = {}

    def market_shape(self, market: str) -> tuple[int, int]:
        return self.load_empirical_returns(market).shape

    def load_empirical_returns(self, market: str) -> pd.DataFrame:
        if market in self._returns_cache:
            return self._returns_cache[market]
        path = self.config.root / "data" / "interim" / f"{market}.pickle"
        data = pd.read_pickle(path)
        dated = data.loc[self.config.start_date : self.config.end_date]
        keep = dated.notna().sum(axis=0) >= self.config.min_empirical_observations
        self._returns_cache[market] = dated.loc[:, keep]
        return self._returns_cache[market]

    def empirical_return_target(self, market: str) -> ReturnTarget:
        returns = self.load_empirical_returns(market)
        means = returns.mean(axis=0).to_numpy(dtype=float)
        stds = returns.std(axis=0, ddof=1).to_numpy(dtype=float)
        valid = np.isfinite(means) & np.isfinite(stds) & (stds > 0)
        if not np.any(valid):
            raise ValueError(f"no valid empirical return series for {market}")
        return ReturnTarget(float(means[valid].mean()), float(stds[valid].mean()), int(valid.sum()))

    def empirical_collective_sigma(self, market: str) -> float:
        """Standard deviation of all finite returns in the selected ensemble."""
        values = self.load_empirical_returns(market).to_numpy(dtype=float, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            raise ValueError(f"no valid collective return standard deviation for {market}")
        sigma = float(finite.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"no valid collective return standard deviation for {market}")
        return sigma

    @staticmethod
    def return_moment_loss(returns: pd.DataFrame, target: ReturnTarget) -> float:
        values = returns.to_numpy(dtype=float, copy=False)
        model_mean = float(values.mean(axis=0).mean())
        model_std = float(values.std(axis=0, ddof=1).mean())
        if not np.isfinite(model_mean) or not np.isfinite(model_std) or model_std <= 0:
            return float("inf")
        # The mean is small in daily units; express its error in tenths of
        # the empirical daily standard deviation. Compare widths by ratio.
        return float(((model_mean - target.mean) / (0.1 * target.std)) ** 2
                     + np.log(model_std / target.std) ** 2)

    def load_empirical_events(self, market: str, threshold_pair: tuple[float, float]) -> pd.DataFrame:
        cache_key = (market, threshold_pair)
        if cache_key in self._empirical_cache:
            return self._empirical_cache[cache_key]

        if self.config.empirical_source == "returns":
            frame = self.count_simulated_events(self.load_empirical_returns(market), threshold_pair, market)
            if frame.empty:
                raise ValueError(f"no empirical events for {market} at thresholds {threshold_pair}")
            self._empirical_cache[cache_key] = frame
            return frame

        table = table_name_for_thresholds(*threshold_pair)
        query = f"""
        SELECT Volatility, FHT, start, end, Market
        FROM {table}
        WHERE Market = ?
          AND start >= ?
          AND end <= ?
          AND FHT >= ?
          AND FHT <= ?
          AND Volatility < ?
        """
        database = self.config.absolute_database_path
        if not database.is_file():
            raise FileNotFoundError(
                f"empirical FHT database not found: {database}. "
                "Provide the processed database with --database before calibrating."
            )
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            exists = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()[0]
            if not exists:
                raise ValueError(f"missing empirical table {table} in {database}")
            frame = pd.read_sql_query(
                query,
                conn,
                params=(
                    market,
                    self.config.start_date,
                    self.config.end_date,
                    self.config.tau_min,
                    self.config.tau_max,
                    self.config.vol_limit,
                ),
            )
        if frame.empty:
            raise ValueError(f"no empirical events for {market} at thresholds {threshold_pair}")
        frame["Volatility"] = pd.to_numeric(frame["Volatility"], errors="coerce")
        frame["FHT"] = pd.to_numeric(frame["FHT"], errors="coerce")
        frame = frame.dropna(subset=["Volatility", "FHT"])
        self._empirical_cache[cache_key] = frame
        return frame

    def load_empirical_grid(self, market: str) -> dict[tuple[float, float], pd.DataFrame]:
        return {pair: self.load_empirical_events(market, pair) for pair in self.config.threshold_pairs}

    def count_simulated_events(
        self,
        returns: pd.DataFrame,
        threshold_pair: tuple[float, float],
        market: str,
    ) -> pd.DataFrame:
        fixed_sigma = self.config.threshold_sigma
        analyst = StabilVolter(
            start_level=threshold_pair[0] * fixed_sigma if fixed_sigma is not None else threshold_pair[0],
            end_level=threshold_pair[1] * fixed_sigma if fixed_sigma is not None else threshold_pair[1],
            std_normalization=False if fixed_sigma is not None else self.config.std_normalization,
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
        )
        if self.config.count_method == "quiet_pandas":
            return self._quiet_stabilvol(analyst, returns, Market=market)
        return analyst.get_stabilvol(returns, method=self.config.count_method, Market=market)

    @staticmethod
    def _quiet_stabilvol(analyst: StabilVolter, returns: pd.DataFrame, **frame_info: object) -> pd.DataFrame:
        analyst.data = returns
        # The original property recomputes the full frame's standard deviation
        # for every path. Resolve thresholds once per simulated realization.
        start_level = analyst.threshold_start
        end_level = analyst.threshold_end
        divergence_limit = analyst.divergence_limit
        applied = returns.apply(
            analyst.count_stock_fht,
            threshold_start=start_level,
            threshold_end=end_level,
            divergence_limit=divergence_limit,
            squeeze=True,
        )
        result_matrices = []
        if isinstance(applied, pd.DataFrame):
            values = applied.to_numpy()
            if values.size > 0:
                result_matrices.append(values.reshape(4, -1))
        else:
            result_matrices = [series.reshape(4, -1) for series in applied if len(series) > 0]
        if not result_matrices:
            return pd.DataFrame(columns=["Volatility", "FHT", "start", "end", *frame_info.keys()])
        stabilvol = pd.DataFrame(
            np.concatenate(result_matrices, axis=1).T,
            columns=["Volatility", "FHT", "start", "end"],
        )
        stabilvol["Volatility"] = pd.to_numeric(stabilvol["Volatility"], errors="coerce")
        stabilvol["FHT"] = pd.to_numeric(stabilvol["FHT"], errors="coerce")
        for key, value in frame_info.items():
            stabilvol[key] = value
        return stabilvol.dropna(subset=["Volatility", "FHT"])

    def distribution_loss(self, empirical: pd.DataFrame, simulated: pd.DataFrame) -> float:
        if simulated.empty:
            return self.config.empty_penalty

        empirical_fht = pd.to_numeric(empirical["FHT"], errors="coerce").dropna().to_numpy(dtype=float)
        simulated_fht = pd.to_numeric(simulated["FHT"], errors="coerce").dropna().to_numpy(dtype=float)
        empirical_fht = empirical_fht[(empirical_fht >= self.config.tau_min) & (empirical_fht <= self.config.tau_max)]
        simulated_fht = simulated_fht[(simulated_fht >= self.config.tau_min) & (simulated_fht <= self.config.tau_max)]
        if empirical_fht.size == 0 or simulated_fht.size == 0:
            return self.config.empty_penalty

        return float(ks_2samp(empirical_fht, simulated_fht).statistic)

    def _event_arrays(self, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        volatility = pd.to_numeric(events["Volatility"], errors="coerce").to_numpy(dtype=float)
        fht = pd.to_numeric(events["FHT"], errors="coerce").to_numpy(dtype=float)
        valid = (
            np.isfinite(volatility) & np.isfinite(fht) & (volatility >= 0)
            & (volatility < self.config.vol_limit)
            & (fht >= self.config.tau_min) & (fht <= self.config.tau_max)
        )
        return volatility[valid], fht[valid]

    @staticmethod
    def _binned_fht(volatility: np.ndarray, fht: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts, _ = np.histogram(volatility, bins=edges)
        totals, _ = np.histogram(volatility, bins=edges, weights=fht)
        means = np.divide(totals, counts, out=np.full(len(counts), np.nan), where=counts > 0)
        return counts, means

    def prepare_curve_target(self, empirical: pd.DataFrame) -> CurveTarget:
        volatility, fht = self._event_arrays(empirical)
        if len(volatility) == 0:
            raise ValueError("no finite empirical FHT events for MFHT curve")
        upper = float(np.quantile(volatility, 0.995))
        if upper <= 0:
            upper = max(float(np.max(volatility)), np.finfo(float).eps)
        edges = np.linspace(0.0, upper, self.config.n_vol_bins + 1)
        counts, means = self._binned_fht(volatility, fht, edges)
        eligible = counts >= self.config.min_empirical_bin_events
        if not np.any(eligible):
            raise ValueError(
                "no empirical MFHT bins meet min_empirical_bin_events; "
                "reduce n_vol_bins or min_empirical_bin_events"
            )
        return CurveTarget(
            edges=edges,
            counts=counts,
            mfht=means,
            eligible=eligible,
            fht_scale=max(float(np.nanmax(means[eligible])), 1.0),
        )

    def curve_comparison(self, target: CurveTarget, simulated: pd.DataFrame) -> pd.DataFrame:
        volatility, fht = self._event_arrays(simulated)
        counts, means = self._binned_fht(volatility, fht, target.edges)
        eligible_simulated = counts >= self.config.min_simulated_bin_events
        return pd.DataFrame({
            "volatility_lower": target.edges[:-1],
            "volatility_upper": target.edges[1:],
            "volatility_midpoint": (target.edges[:-1] + target.edges[1:]) / 2,
            "empirical_events": target.counts,
            "empirical_mfht": target.mfht,
            "simulated_events": counts,
            "simulated_mfht": means,
            "fit_bin": target.eligible,
            "simulated_bin_covered": eligible_simulated,
        })

    def curve_loss(self, target: CurveTarget, simulated: pd.DataFrame) -> float:
        if simulated.empty:
            return self.config.empty_penalty
        comparison = self.curve_comparison(target, simulated)
        empirical_mask = target.eligible
        simulated_counts = comparison["simulated_events"].to_numpy(dtype=int)
        simulated_means = comparison["simulated_mfht"].to_numpy(dtype=float)
        covered = empirical_mask & (simulated_counts >= self.config.min_simulated_bin_events)
        errors = np.ones(int(empirical_mask.sum()), dtype=float)
        errors[covered[empirical_mask]] = (
            (simulated_means[covered] - target.mfht[covered]) / target.fht_scale
        ) ** 2
        shape_loss = float(np.sqrt(np.mean(errors)))

        empirical_counts = target.counts[empirical_mask].astype(float)
        model_counts = simulated_counts[empirical_mask].astype(float)
        if model_counts.sum() == 0:
            return self.config.empty_penalty
        empirical_weights = empirical_counts / empirical_counts.sum()
        model_weights = model_counts / model_counts.sum()
        event_distribution_loss = 0.5 * float(np.abs(empirical_weights - model_weights).sum())
        return shape_loss + self.config.event_count_weight * event_distribution_loss

    def grid_loss(self, empirical_grid: dict[tuple[float, float], pd.DataFrame | CurveTarget], simulated_returns: pd.DataFrame, market: str) -> float:
        losses = []
        for pair, empirical in empirical_grid.items():
            simulated = self.count_simulated_events(simulated_returns, pair, market)
            if self.config.loss_metric == "mfht_curve":
                target = empirical if isinstance(empirical, CurveTarget) else self.prepare_curve_target(empirical)
                losses.append(self.curve_loss(target, simulated))
            else:
                if not isinstance(empirical, pd.DataFrame):
                    raise TypeError("fht_distribution loss requires empirical events")
                losses.append(self.distribution_loss(empirical, simulated))
        return float(np.mean(losses)) if losses else float("inf")

    def parameter_names(self) -> tuple[str, ...]:
        return CORRELATED_PARAMETER_NAMES if self.config.correlated_noise else UNCORRELATED_PARAMETER_NAMES

    def params_from_vector(self, vector: Iterable[float]) -> HestonParams:
        values = dict(zip(self.parameter_names(), map(float, vector)))
        return self.config.base_params.update(**values)

    def vector_from_params(self, params: HestonParams) -> np.ndarray:
        return np.array([getattr(params, name) for name in self.parameter_names()], dtype=float)

    def bounds_sequence(self) -> list[tuple[float, float]]:
        return [self.config.bounds[name] for name in self.parameter_names()]

    def objective(
        self,
        vector: np.ndarray,
        *,
        market: str,
        empirical_grid: dict[tuple[float, float], pd.DataFrame | CurveTarget],
        n_paths: int,
        n_steps: int,
        seed: int,
        return_target: Optional[ReturnTarget] = None,
    ) -> float:
        try:
            params = self.params_from_vector(vector)
            sim_config = SimulationConfig(
                n_paths=n_paths,
                n_steps=n_steps,
                sample_every=self.config.sampling_interval_steps,
                burn_in_steps=self.config.sampling_burn_in_steps,
                return_mode=self.config.sampling_return_mode,
                seed=seed,
                correlated_noise=self.config.correlated_noise,
                store_state=False,
                column_prefix=market,
            )
            result = simulate_modified_heston(params, sim_config)
            if self.config.loss_metric == "return_moments":
                if return_target is None:
                    raise ValueError("return target required for return_moments")
                return self.return_moment_loss(result.returns, return_target)
            loss = self.grid_loss(empirical_grid, result.returns, market)
            if self.config.return_loss_weight:
                if return_target is None:
                    raise ValueError("return target required for return loss")
                loss += self.config.return_loss_weight * self.return_moment_loss(result.returns, return_target)
            return loss
        except (FloatingPointError, OverflowError, SimulationError, ValueError):
            return float("inf")

    def calibrate_market(
        self,
        market: str,
        *,
        maxiter: int = 20,
        popsize: int = 8,
        workers: int = 1,
        polish: bool = False,
        refine: bool = False,
        validate_full: bool = True,
        progress: bool = True,
    ) -> CalibrationResult:
        empirical_grid = {} if self.config.loss_metric == "return_moments" else self.load_empirical_grid(market)
        objective_grid = (
            {pair: self.prepare_curve_target(events) for pair, events in empirical_grid.items()}
            if self.config.loss_metric == "mfht_curve" else empirical_grid
        )
        _, n_market_stocks = self.market_shape(market)
        return_target = (
            self.empirical_return_target(market)
            if self.config.loss_metric == "return_moments" or self.config.return_loss_weight else None
        )
        n_paths_pilot = min(self.config.pilot_max_paths, n_market_stocks)
        seed = self.config.seed + sum((index + 1) * ord(char) for index, char in enumerate(market))
        payload = ObjectivePayload(
            config=self.config,
            market=market,
            empirical_grid=objective_grid,
            n_paths=n_paths_pilot,
            n_steps=self.config.pilot_n_steps,
            seed=seed,
            return_target=return_target,
        )
        objective = partial(heston_objective, payload=payload)
        generation_start = time.perf_counter()
        generation = 0

        def callback(intermediate_result):
            nonlocal generation
            generation += 1
            if progress:
                elapsed = time.perf_counter() - generation_start
                print(
                    f"[{market}] generation {generation}: "
                    f"best_loss={intermediate_result.fun:.6g}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if workers == 0 or workers < -1:
            raise ValueError("workers must be a positive integer or -1")
        with ExitStack() as stack:
            map_workers = 1
            optimizer_objective = objective
            if workers != 1:
                # SciPy may evaluate the objective in the parent during polishing.
                _init_heston_worker(payload)
                worker_count = mp.cpu_count() if workers == -1 else workers
                pool = stack.enter_context(
                    mp.Pool(worker_count, initializer=_init_heston_worker, initargs=(payload,))
                )
                map_workers = pool.map
                optimizer_objective = _heston_worker_objective
            result = differential_evolution(
                optimizer_objective,
                bounds=self.bounds_sequence(),
                x0=self.vector_from_params(self.config.initial_params) if self.config.initial_params is not None else None,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
                workers=map_workers,
                polish=polish,
                updating="immediate" if workers == 1 else "deferred",
                callback=callback,
            )
        best_vector = result.x
        best_loss = float(result.fun)
        message = str(result.message)

        if refine and np.isfinite(best_loss):
            local = minimize(objective, best_vector, method="Nelder-Mead", options={"maxiter": 200})
            if local.fun < best_loss:
                best_vector = local.x
                best_loss = float(local.fun)
                message = f"{message}; refined: {local.message}"

        best_params = self.params_from_vector(best_vector)
        validation_loss = None
        if validate_full and np.isfinite(best_loss):
            validation_seed = seed + 100_000
            try:
                validation_sim = simulate_modified_heston(
                    best_params,
                    SimulationConfig(
                        n_paths=n_market_stocks,
                        n_steps=self.config.full_n_steps,
                        sample_every=self.config.sampling_interval_steps,
                        burn_in_steps=self.config.sampling_burn_in_steps,
                        return_mode=self.config.sampling_return_mode,
                        seed=validation_seed,
                        correlated_noise=self.config.correlated_noise,
                        store_state=False,
                        column_prefix=market,
                    ),
                )
                if self.config.loss_metric == "return_moments":
                    validation_loss = self.return_moment_loss(validation_sim.returns, return_target)
                else:
                    validation_loss = self.grid_loss(objective_grid, validation_sim.returns, market)
                    validation_loss += self.config.return_loss_weight * self.return_moment_loss(
                        validation_sim.returns, return_target)
            except (SimulationError, ValueError, FloatingPointError, OverflowError):
                validation_loss = float("inf")

        return CalibrationResult(
            market=market,
            params=best_params,
            pilot_loss=best_loss,
            validation_loss=validation_loss,
            optimization_success=bool(result.success),
            optimization_message=message,
            n_paths_pilot=n_paths_pilot,
            n_paths_full=n_market_stocks,
            n_steps_pilot=self.config.pilot_n_steps,
            n_steps_full=self.config.full_n_steps,
        )

    def evaluate_params(
        self,
        market: str,
        params: HestonParams,
        *,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[float, pd.DataFrame]:
        empirical_grid = {} if self.config.loss_metric == "return_moments" else self.load_empirical_grid(market)
        objective_grid = (
            {pair: self.prepare_curve_target(events) for pair, events in empirical_grid.items()}
            if self.config.loss_metric == "mfht_curve" else empirical_grid
        )
        _, n_market_stocks = self.market_shape(market)
        n_paths = n_market_stocks if n_paths is None else n_paths
        n_steps = self.config.full_n_steps if n_steps is None else n_steps
        seed = self.config.seed if seed is None else seed
        simulation = simulate_modified_heston(
            params,
            SimulationConfig(
                n_paths=n_paths,
                n_steps=n_steps,
                sample_every=self.config.sampling_interval_steps,
                burn_in_steps=self.config.sampling_burn_in_steps,
                return_mode=self.config.sampling_return_mode,
                seed=seed,
                correlated_noise=self.config.correlated_noise,
                store_state=False,
                column_prefix=market,
            ),
        )
        if self.config.loss_metric == "return_moments":
            loss = self.return_moment_loss(simulation.returns, self.empirical_return_target(market))
        else:
            loss = self.grid_loss(objective_grid, simulation.returns, market)
            if self.config.return_loss_weight:
                loss += self.config.return_loss_weight * self.return_moment_loss(
                    simulation.returns, self.empirical_return_target(market))
        return loss, simulation.returns

    def result_frame(self, results: Iterable[CalibrationResult]) -> pd.DataFrame:
        return pd.DataFrame([result.to_record() for result in results])


def config_to_record(config: CalibrationConfig) -> dict[str, object]:
    record = asdict(config)
    record["root"] = str(config.root)
    record["database_path"] = str(config.database_path)
    record["threshold_pairs"] = list(config.threshold_pairs)
    record["markets"] = list(config.markets)
    record["base_params"] = config.base_params.to_dict()
    return record
