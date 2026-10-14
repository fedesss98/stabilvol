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
    sample_every: int = 1
    burn_in_steps: int = 0
    return_mode: str = "pre_reset_step"
    seed: Optional[int] = None
    variance_scheme: str = "redraw"
    correlated_noise: bool = False
    max_redraws: int = 500
    start_date: str = "1980-01-01"
    frequency: str = "B"
    column_prefix: str = "sim"
    store_state: bool = True
    dtype: str = "float64"
    stationary_variance: bool = False
    separate_streams: bool = False
    variance_dispersion: float = 0.0
    activity_transitions: Optional[tuple[float, float]] = None


@dataclass
class SimulationResult:
    """Model paths and observed log-return increments.

    ``x`` stores the post-reset state at observation times. With the default
    pre_reset_step mode, ``returns`` records the final Euler increment before
    a reset, as in the supplied Fortran model. In sampled_x mode, returns are
    differences of observed post-reset states and include any reset jump.
    """

    returns: pd.DataFrame
    variance: Optional[np.ndarray]
    x: Optional[np.ndarray]
    params: HestonParams
    config: SimulationConfig
    price_shocks: Optional[np.ndarray] = None
    variance_shocks: Optional[np.ndarray] = None
    reset_counts: Optional[np.ndarray] = None
    stream_seeds: Optional[dict[str, int]] = None

    @property
    def reset_affected(self) -> Optional[np.ndarray]:
        return None if self.reset_counts is None else self.reset_counts > 0

    @property
    def log_prices(self) -> pd.DataFrame:
        """End-of-observation log prices, starting immediately before at log(100)."""
        return np.log(100.0) + self.returns.cumsum()

    @property
    def prices(self) -> pd.DataFrame:
        with np.errstate(under="ignore", over="ignore"):
            return np.exp(self.log_prices)

    @property
    def metadata(self) -> dict[str, object]:
        data = self.params.to_dict()
        data.update(
            {
                "n_paths": self.config.n_paths,
                "n_steps": self.config.n_steps,
                "sample_every": self.config.sample_every,
                "burn_in_steps": self.config.burn_in_steps,
                "return_mode": self.config.return_mode,
                "seed": self.config.seed,
                "variance_scheme": self.config.variance_scheme,
                "correlated_noise": self.config.correlated_noise,
                "stationary_variance": self.config.stationary_variance,
                "stream_seeds": self.stream_seeds,
                "variance_dispersion": self.config.variance_dispersion,
                "activity_transitions": self.config.activity_transitions,
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
    if config.sample_every <= 0 or config.burn_in_steps < 0:
        raise ValueError("sample_every must be positive and burn_in_steps nonnegative")
    if config.return_mode not in ("pre_reset_step", "sampled_x", "accumulated_pre_reset"):
        raise ValueError("invalid return_mode")
    if config.sample_every != 1 and config.return_mode == "pre_reset_step":
        raise ValueError("sample_every > 1 requires an aggregated return mode")
    if config.variance_scheme not in ("redraw", "cir_exact"):
        raise ValueError("invalid variance_scheme")
    if config.variance_scheme == "cir_exact" and config.correlated_noise:
        raise ValueError("cir_exact currently supports independent shocks only")
    if config.variance_dispersion < 0 or not np.isfinite(config.variance_dispersion):
        raise ValueError("variance_dispersion must be finite and nonnegative")
    if config.variance_dispersion and config.variance_scheme != "cir_exact":
        raise ValueError("variance dispersion requires cir_exact")
    if config.activity_transitions is not None:
        if len(config.activity_transitions) != 2 or any(not 0 <= p <= 1 for p in config.activity_transitions):
            raise ValueError("activity transitions must be two probabilities")
        if sum(config.activity_transitions) == 0:
            raise ValueError("activity chain must have a nonzero transition probability")
    if params.dt <= 0:
        raise ValueError("dt must be positive")
    if params.vstart < 0 or params.bb <= 0:
        raise ValueError("vstart must be non-negative and bb must be positive")
    if params.cc < 0 or params.aa <= 0:
        raise ValueError("aa must be positive and cc must be non-negative")
    if not -1.0 <= params.rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if not all(np.isfinite(v) for v in params.to_dict().values()):
        raise ValueError("parameters must be finite")


def daily_parameters(params: HestonParams, observation_duration: float, substeps: int = 10) -> HestonParams:
    """Change time units and variance units, keeping one observation one day."""
    if observation_duration <= 0 or substeps < 1:
        raise ValueError("positive duration and substeps required")
    names = ("a", "b", "aa", "bb", "cc", "vstart")
    return params.update(**{key: getattr(params, key) * observation_duration for key in names}, dt=1 / substeps)


def cir_exact_step(variance, kappa, theta, xi, dt, rng):
    """Exact nonnegative CIR transition, including deterministic xi=0."""
    decay = np.exp(-kappa * dt)
    if np.all(np.asarray(xi) == 0):
        return theta + (variance - theta) * decay
    scale = np.asarray(xi)**2 * (-np.expm1(-kappa * dt)) / (4 * kappa)
    degrees = 4 * kappa * theta / np.asarray(xi)**2
    noncentrality = np.maximum(variance, 0) * decay / scale
    return scale * rng.noncentral_chisquare(degrees, noncentrality)


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
    """Simulate independent log-price paths and their observed increments.

    Each column is one path of x(t) = log[p(t)/p(0)]. By default, one Euler
    step produces one pre-reset return. With sampled_x mode, each return is
    the endpoint difference across sample_every Euler steps.
    """

    _validate_inputs(params, config)
    rng = np.random.default_rng(config.seed)
    stream_seeds = None
    price_rng = variance_rng = observation_rng = initial_rng = rng
    if config.separate_streams or config.variance_scheme == "cir_exact":
        children = np.random.SeedSequence(config.seed).spawn(4)
        stream_seeds = {name: int(child.generate_state(1, dtype=np.uint64)[0])
                        for name, child in zip(("price", "variance", "observation", "initial"), children)}
        price_rng, variance_rng, observation_rng, initial_rng = [np.random.default_rng(s) for s in stream_seeds.values()]
    dtype = np.dtype(config.dtype)

    current_x = np.full(config.n_paths, params.start, dtype=dtype)
    current_variance = np.full(config.n_paths, params.vstart, dtype=dtype)
    multiplier = np.ones(config.n_paths)
    if config.variance_dispersion:
        spread = config.variance_dispersion
        multiplier = np.exp(spread * initial_rng.standard_normal(config.n_paths) - spread**2 / 2)
    theta = params.bb * multiplier
    xi = params.cc * np.sqrt(multiplier)
    if config.stationary_variance:
        current_variance = (initial_rng.gamma(2 * params.aa * theta / xi**2, xi**2 / (2 * params.aa))
                            if params.cc else theta.copy())
    returns = np.empty((config.n_steps, config.n_paths), dtype=dtype)
    x_values = np.empty_like(returns) if config.store_state else None
    variance_values = np.empty_like(returns) if config.store_state else None
    price_shocks = np.empty_like(returns) if keep_shocks else None
    variance_shocks = np.empty_like(returns) if keep_shocks else None
    reset_counts = np.zeros(returns.shape, dtype=np.int32)
    accumulated = np.zeros(config.n_paths)
    interval_resets = np.zeros(config.n_paths, dtype=np.int32)
    pending_quote = np.zeros(config.n_paths)
    active_quote = np.ones(config.n_paths, dtype=bool)
    if config.activity_transitions is not None:
        to_stale, to_active = config.activity_transitions
        active_quote = observation_rng.random(config.n_paths) < to_active / (to_stale + to_active)

    last_sampled_x = current_x.copy()
    observed_step = 0
    total_internal_steps = config.burn_in_steps + config.n_steps * config.sample_every
    for internal_step in range(total_internal_steps):
        z_price = price_rng.standard_normal(config.n_paths)
        sqrt_variance_dt = np.sqrt(np.maximum(current_variance, 0.0) * params.dt)
        next_x = (
            current_x
            - dU_dx(params.a, params.b, current_x) * params.dt
            - 0.5 * current_variance * params.dt
            + sqrt_variance_dt * z_price
        )
        step_returns = next_x - current_x

        if config.variance_scheme == "cir_exact":
            next_variance = cir_exact_step(current_variance, params.aa, theta, xi, params.dt, variance_rng)
            z_vol = np.full(config.n_paths, np.nan)  # An exact transition has no Euler Gaussian shock.
        else:
            next_variance, z_vol = _variance_step_redraw(current_variance, params, variance_rng, z_price, config)

        crossed = next_x < params.reset_threshold
        next_x = np.where(crossed, params.start, next_x)
        if not np.all(np.isfinite(step_returns)) or not np.all(np.isfinite(next_x)) or not np.all(np.isfinite(next_variance)):
            raise SimulationError("simulation produced non-finite values")

        current_x = next_x
        current_variance = next_variance
        if internal_step < config.burn_in_steps:
            if internal_step == config.burn_in_steps - 1:
                last_sampled_x = current_x.copy()
            continue
        accumulated += step_returns
        interval_resets += crossed
        if (internal_step - config.burn_in_steps + 1) % config.sample_every:
            continue

        observed = (accumulated.copy() if config.return_mode == "accumulated_pre_reset" else
                    step_returns if config.return_mode == "pre_reset_step" else current_x - last_sampled_x)
        if config.activity_transitions is not None:
            pending_quote += observed
            u = observation_rng.random(config.n_paths)
            active_quote = np.where(active_quote, u >= to_stale, u < to_active)
            observed = np.where(active_quote, pending_quote, 0.0)
            pending_quote[active_quote] = 0.0
        returns[observed_step] = observed
        reset_counts[observed_step] = interval_resets
        accumulated.fill(0)
        interval_resets.fill(0)
        if config.store_state:
            x_values[observed_step] = current_x
            variance_values[observed_step] = current_variance
        if keep_shocks:
            price_shocks[observed_step] = z_price
            variance_shocks[observed_step] = z_vol
        last_sampled_x = current_x.copy()
        observed_step += 1

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
        reset_counts=reset_counts,
        stream_seeds=stream_seeds,
    )
