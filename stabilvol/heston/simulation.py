"""Implementation of the modified Heston simulator by Spagnolo-Valenti."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HestonParams:
    """Parameters of the modified Heston model.

    Defaults mirror ``simulation_tau_vs_noise/parm.dat``.
    """

    a: float = 2.0
    b: float = 3.0
    aa: float = 2.0
    bb: float = 0.01
    cc: float = 0.83
    rho: float = 0.0
    vstart: float = 8.62e-5
    dt: float = 0.01
    start: float = 0.0
    reset_threshold: float = -6.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def update(self, **values: float) -> "HestonParams":
        return replace(self, **values)


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime options for the simulator."""

    n_paths: int
    n_steps: int
    seed: Optional[int] = None
    variance_scheme: str = "redraw"
    correlated_noise: bool = False
    max_redraws: int = 500
    start_date: str = "1980-01-01"
    frequency: str = "B"
    column_prefix: str = "sim"
    store_state: bool = True
    dtype: str = "float64"


@dataclass
class SimulationResult:
    """Container returned by :func:`simulate_modified_heston`."""

    returns: pd.DataFrame
    variance: Optional[np.ndarray]
    x: Optional[np.ndarray]
    params: HestonParams
    config: SimulationConfig
    price_shocks: Optional[np.ndarray] = None
    variance_shocks: Optional[np.ndarray] = None

    @property
    def metadata(self) -> dict[str, object]:
        data = self.params.to_dict()
        data.update(
            {
                "n_paths": self.config.n_paths,
                "n_steps": self.config.n_steps,
                "seed": self.config.seed,
                "variance_scheme": self.config.variance_scheme,
                "correlated_noise": self.config.correlated_noise,
            }
        )
        return data


class SimulationError(RuntimeError):
    """Raised when a parameter set cannot produce a valid simulation."""


def dU_dx(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Derivative of the cubic potential used by the old Fortran code."""

    return 3.0 * a * x**2 + 2.0 * b * x


def _validate_inputs(params: HestonParams, config: SimulationConfig) -> None:
    if config.n_paths <= 0 or config.n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if config.variance_scheme != "redraw":
        raise ValueError("Only variance_scheme='redraw' is implemented")
    if params.dt <= 0:
        raise ValueError("dt must be positive")
    if params.vstart < 0 or params.bb <= 0:
        raise ValueError("vstart must be non-negative and bb must be positive")
    if params.cc < 0 or params.aa <= 0:
        raise ValueError("aa must be positive and cc must be non-negative")
    if not -1.0 <= params.rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")


def _variance_step_redraw(
    variance: np.ndarray,
    params: HestonParams,
    rng: np.random.Generator,
    z_price: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance variance, redrawing negative proposals like ``heston.f``."""

    if config.correlated_noise:
        z_independent = rng.standard_normal(config.n_paths)
        z_vol = params.rho * z_price + np.sqrt(max(0.0, 1.0 - params.rho**2)) * z_independent
    else:
        z_vol = rng.standard_normal(config.n_paths)

    sqrt_term = np.sqrt(np.maximum(variance, 0.0) * params.dt)
    proposal = variance + params.aa * (params.bb - variance) * params.dt + params.cc * sqrt_term * z_vol
    bad = proposal < 0
    redraws = 0

    while np.any(bad):
        redraws += 1
        if redraws > config.max_redraws:
            raise SimulationError("negative variance redraw limit exceeded")
        if config.correlated_noise:
            z_independent = rng.standard_normal(int(bad.sum()))
            z_vol_bad = params.rho * z_price[bad] + np.sqrt(max(0.0, 1.0 - params.rho**2)) * z_independent
        else:
            z_vol_bad = rng.standard_normal(int(bad.sum()))
        proposal[bad] = (
            variance[bad]
            + params.aa * (params.bb - variance[bad]) * params.dt
            + params.cc * sqrt_term[bad] * z_vol_bad
        )
        z_vol[bad] = z_vol_bad
        bad = proposal < 0

    return proposal, z_vol


def simulate_modified_heston(
    params: HestonParams,
    config: SimulationConfig,
    *,
    keep_shocks: bool = False,
) -> SimulationResult:
    """Simulate the modified Heston model and return returns in StabilVol format."""

    _validate_inputs(params, config)
    rng = np.random.default_rng(config.seed)
    dtype = np.dtype(config.dtype)

    current_x = np.full(config.n_paths, params.start, dtype=dtype)
    current_variance = np.full(config.n_paths, params.vstart, dtype=dtype)
    returns = np.empty((config.n_steps, config.n_paths), dtype=dtype)
    x_values = np.empty_like(returns) if config.store_state else None
    variance_values = np.empty_like(returns) if config.store_state else None
    price_shocks = np.empty_like(returns) if keep_shocks else None
    variance_shocks = np.empty_like(returns) if keep_shocks else None

    for step in range(config.n_steps):
        z_price = rng.standard_normal(config.n_paths)
        sqrt_variance_dt = np.sqrt(np.maximum(current_variance, 0.0) * params.dt)
        next_x = (
            current_x
            - dU_dx(params.a, params.b, current_x) * params.dt
            - 0.5 * current_variance * params.dt
            + sqrt_variance_dt * z_price
        )
        step_returns = next_x - current_x

        next_variance, z_vol = _variance_step_redraw(current_variance, params, rng, z_price, config)

        next_x = np.where(next_x < params.reset_threshold, params.start, next_x)
        if not np.all(np.isfinite(step_returns)) or not np.all(np.isfinite(next_x)) or not np.all(np.isfinite(next_variance)):
            raise SimulationError("simulation produced non-finite values")

        returns[step] = step_returns
        if config.store_state:
            x_values[step] = next_x
            variance_values[step] = next_variance
        if keep_shocks:
            price_shocks[step] = z_price
            variance_shocks[step] = z_vol

        current_x = next_x
        current_variance = next_variance

    index = pd.date_range(config.start_date, periods=config.n_steps, freq=config.frequency)
    columns = [f"{config.column_prefix}_{i:05d}" for i in range(config.n_paths)]
    returns_frame = pd.DataFrame(returns, index=index, columns=columns)

    return SimulationResult(
        returns=returns_frame,
        variance=variance_values,
        x=x_values,
        params=params,
        config=config,
        price_shocks=price_shocks,
        variance_shocks=variance_shocks,
    )
