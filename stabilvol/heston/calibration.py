"""Calibration helpers for fitting modified Heston simulations to empirical FHT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
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
    """Configuration for Heston-to-FHT calibration."""

    root: Path = Path(".")
    database_path: Path = Path("data/processed/trapezoidal_selection/stabilvol_filtered.sqlite")
    markets: tuple[str, ...] = DEFAULT_MARKETS
    threshold_pairs: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLD_PAIRS
    start_date: str = "1980-01-01"
    end_date: str = "2022-07-01"
    vol_limit: float = 100.0
    tau_min: int = 2
    tau_max: int = 30
    pilot_max_paths: int = 512
    pilot_n_steps: int = 3030
    full_n_steps: int = 11089
    n_vol_bins: int = 40
    event_count_weight: float = 0.1
    empty_penalty: float = 10.0
    seed: int = 12345
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    base_params: HestonParams = field(default_factory=HestonParams)
    count_method: str = "quiet_pandas"
    correlated_noise: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "markets", tuple(self.markets))
        object.__setattr__(self, "threshold_pairs", tuple(tuple(pair) for pair in self.threshold_pairs))
        object.__setattr__(self, "bounds", {key: tuple(value) for key, value in self.bounds.items()})

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
    empirical_grid: dict[tuple[float, float], pd.DataFrame]
    n_paths: int
    n_steps: int
    seed: int


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
    )


class HestonCalibrator:
    """Fit modified Heston parameters against empirical StabilVol FHT data."""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self._empirical_cache: dict[tuple[str, tuple[float, float]], pd.DataFrame] = {}

    def market_shape(self, market: str) -> tuple[int, int]:
        path = self.config.root / "data" / "interim" / f"{market}.pickle"
        data = pd.read_pickle(path)
        return data.shape

    def load_empirical_returns(self, market: str) -> pd.DataFrame:
        path = self.config.root / "data" / "interim" / f"{market}.pickle"
        data = pd.read_pickle(path)
        return data.loc[self.config.start_date : self.config.end_date]

    def load_empirical_events(self, market: str, threshold_pair: tuple[float, float]) -> pd.DataFrame:
        cache_key = (market, threshold_pair)
        if cache_key in self._empirical_cache:
            return self._empirical_cache[cache_key]

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
        with sqlite3.connect(database) as conn:
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
        analyst = StabilVolter(
            start_level=threshold_pair[0],
            end_level=threshold_pair[1],
            std_normalization=True,
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
        )
        if self.config.count_method == "quiet_pandas":
            return self._quiet_stabilvol(analyst, returns, Market=market)
        return analyst.get_stabilvol(returns, method=self.config.count_method, Market=market)

    @staticmethod
    def _quiet_stabilvol(analyst: StabilVolter, returns: pd.DataFrame, **frame_info: object) -> pd.DataFrame:
        analyst.data = returns
        applied = returns.apply(analyst.count_stock_fht, squeeze=True)
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

    def grid_loss(self, empirical_grid: dict[tuple[float, float], pd.DataFrame], simulated_returns: pd.DataFrame, market: str) -> float:
        losses = []
        for pair, empirical in empirical_grid.items():
            simulated = self.count_simulated_events(simulated_returns, pair, market)
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
        empirical_grid: dict[tuple[float, float], pd.DataFrame],
        n_paths: int,
        n_steps: int,
        seed: int,
    ) -> float:
        try:
            params = self.params_from_vector(vector)
            sim_config = SimulationConfig(
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                correlated_noise=self.config.correlated_noise,
                store_state=False,
                column_prefix=market,
            )
            result = simulate_modified_heston(params, sim_config)
            return self.grid_loss(empirical_grid, result.returns, market)
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
        empirical_grid = self.load_empirical_grid(market)
        _, n_market_stocks = self.market_shape(market)
        n_paths_pilot = min(self.config.pilot_max_paths, n_market_stocks)
        seed = self.config.seed + sum((index + 1) * ord(char) for index, char in enumerate(market))
        payload = ObjectivePayload(
            config=self.config,
            market=market,
            empirical_grid=empirical_grid,
            n_paths=n_paths_pilot,
            n_steps=self.config.pilot_n_steps,
            seed=seed,
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

        result = differential_evolution(
            objective,
            bounds=self.bounds_sequence(),
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            workers=workers,
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
                        seed=validation_seed,
                        correlated_noise=self.config.correlated_noise,
                        store_state=False,
                        column_prefix=market,
                    ),
                )
                validation_loss = self.grid_loss(empirical_grid, validation_sim.returns, market)
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
        empirical_grid = self.load_empirical_grid(market)
        _, n_market_stocks = self.market_shape(market)
        n_paths = n_market_stocks if n_paths is None else n_paths
        n_steps = self.config.full_n_steps if n_steps is None else n_steps
        seed = self.config.seed if seed is None else seed
        simulation = simulate_modified_heston(
            params,
            SimulationConfig(
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                correlated_noise=self.config.correlated_noise,
                store_state=False,
                column_prefix=market,
            ),
        )
        return self.grid_loss(empirical_grid, simulation.returns, market), simulation.returns

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
