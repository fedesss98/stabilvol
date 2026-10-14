"""Staged corrected-Heston experiment, numerical gates, and held-out reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .simulation import HestonParams, SimulationConfig, SimulationError, daily_parameters, simulate_modified_heston
from .corrected import (COUNTER_VERSION, PARAMETER_NAMES, CorrectedConfig, ResolvedTarget,
    ReturnConstraints, count_episodes, make_target, bin_events, score_curve,
    make_constraints, return_features, score_constraints, zero_run_summary)
from .run_io import atomic_json, digest, file_digest, immutable_manifest, source_fingerprint, verify_artifacts, read_manifest
from .search import differential_search, rank

TRAIN_SEEDS = (1001, 1002, 1003, 1004)
VALIDATION_SEEDS = (2001, 2002, 2003, 2004, 2005)
FINAL_SEEDS = tuple(range(3001, 3011))
VARIANTS = ("baseline", "heterogeneity", "activity", "combined")


def stock_split(columns, seed=41981):
    """Stable under input-column ordering; no use of held-out stock outcomes."""
    ordered = sorted(range(len(columns)), key=lambda i: digest([seed, str(columns[i])]))
    first, second = int(.70 * len(ordered)), int(.85 * len(ordered))
    if first < 1 or second == first or second == len(ordered):
        raise ValueError("at least seven stocks required for a 70/15/15 split")
    return {"train": ordered[:first], "selection": ordered[first:second], "heldout": ordered[second:],
            "full": list(range(len(columns)))}


def save_bundle(directory, frame, episodes, target, constraints, config, parent, root):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "data.npz", values=frame.to_numpy(float),
                        dates=frame.index.to_numpy(dtype="datetime64[ns]"), columns=np.asarray(frame.columns, str))
    episodes.episodes.to_csv(directory / "episodes.csv.gz", index=False)
    target.frame().to_csv(directory / "target.csv", index=False)
    atomic_json(directory / "target.json", target)
    atomic_json(directory / "constraints.json", constraints)
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "observation.json", {"zero_runs": zero_run_summary(frame.to_numpy()),
        "zero_fraction": constraints.empirical["zero_fraction"],
        "episode_counts": episodes.episodes.status.value_counts().to_dict(),
        "outer_volatility_events": int((episodes.events.Volatility > config.vol_max).sum()),
        "finite_observations": int(np.isfinite(frame.to_numpy()).sum()),
        "returns_below_minus_one": int((frame.to_numpy() < -1).sum()),
        "returns_above_100": int((frame.to_numpy() > 100).sum())})
    blocks = []
    years = pd.DatetimeIndex(frame.index).year.to_numpy()
    for first_year in range(int(years.min()) // 5 * 5, int(years.max()) + 1, 5):
        indices = np.flatnonzero((years >= first_year) & (years < first_year + 5))
        if len(indices) < 2:
            continue
        # Chronological slices are views, avoiding another full pandas block copy.
        block = frame.to_numpy(copy=False)[indices[0]:indices[-1] + 1]
        result = count_episodes(block, config)
        count, total = bin_events(result.events, target.edges)
        for i, (n, s) in enumerate(zip(count, total)):
            blocks.append({"start_year": first_year, "end_year": first_year + 4, "bin": i,
                "events": int(n), "mfht": s / n if n else None})
    pd.DataFrame(blocks).to_csv(directory / "calendar_blocks.csv", index=False)
    files = [p for p in directory.iterdir() if p.is_file() and p.name != "manifest.json"]
    immutable_manifest(directory / "manifest.json", {"parent": parent, "source": source_fingerprint(root),
        "counter": COUNTER_VERSION, "artifacts": {p.name: file_digest(p) for p in files}})


def prepare(root, run_dir, market, source_fit, overrides=None):
    """Create immutable empirical targets; no outlier clipping or imputation."""
    from .calibration import DEFAULT_BOUNDS
    root, run_dir, source_fit = Path(root), Path(run_dir), Path(source_fit)
    directory = run_dir / market
    if (directory / "manifest.json").exists():
        existing = verify_prepared(root, directory)
        if overrides and any(existing["specification"]["config"].get(k) != v for k, v in overrides.items()):
            raise ValueError("configuration differs from prepared run")
        if all((directory / split_name / "manifest.json").exists() or (directory / f"{split_name}_unsupported.json").exists()
               for split_name in ("train", "selection", "heldout", "full")):
            return directory
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("partial preparation exists; choose a new run directory")
    source = json.loads((source_fit / market / "trial_config.json").read_text())
    frame = pd.read_pickle(root / "data/interim" / f"{market}.pickle")
    frame = frame.loc[source.get("start_date", "1980-01-01"):source.get("end_date", "2022-07-01")]
    frame = frame.loc[:, frame.notna().sum(0) >= 3030]
    frame = frame.loc[frame.notna().any(axis=1)]
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("empirical infinite returns require an explicit upstream data correction")
    setup = {"sigma": source["threshold_sigma"], **(overrides or {})}
    config = CorrectedConfig(**setup)
    fits = pd.read_csv(source_fit / market / "heston_calibration_parameters.csv")
    row = fits.loc[fits.market == market].iloc[-1]
    previous = HestonParams(**{f.name: float(row[f.name]) for f in fields(HestonParams)})
    duration = previous.dt * source["sampling_interval_steps"]
    initial = daily_parameters(previous, duration)
    paper = daily_parameters(HestonParams(), .01)
    names = [name for name in PARAMETER_NAMES if name not in config.fixed_parameters]
    bounds = {name: [DEFAULT_BOUNDS[name][0] * duration, DEFAULT_BOUNDS[name][1] * duration] for name in names}
    # Both converted seeds must be in bounds and reevaluated, never clipped into a different seed silently.
    for name in names:
        bounds[name][0] = min(bounds[name][0], getattr(paper, name), getattr(initial, name))
        bounds[name][1] = max(bounds[name][1], getattr(paper, name), getattr(initial, name))
    split = stock_split(frame.columns)
    specification = {"version": 1, "market": market, "source": source_fingerprint(root),
        "input_sha256": file_digest(root / "data/interim" / f"{market}.pickle"),
        "source_fit_sha256": file_digest(source_fit / market / "heston_calibration_parameters.csv"),
        "source_config_sha256": file_digest(source_fit / market / "trial_config.json"),
        "counter": COUNTER_VERSION, "config": asdict(config), "splits": {
            key: [str(frame.columns[i]) for i in indices] for key, indices in split.items()},
        "initial": initial.to_dict(), "paper": paper.to_dict(), "bounds": bounds,
        "historical": {"parameters": previous.to_dict(), "setup": source},
        "seed_sets": {"train": TRAIN_SEEDS, "selection": VALIDATION_SEEDS, "final": FINAL_SEEDS}}
    directory.mkdir(parents=True, exist_ok=True)
    parent = immutable_manifest(directory / "manifest.json", specification)
    target = None
    for name, indices in split.items():
        print(f"{market}: preparing {name} ({len(indices)} stocks)", flush=True)
        if (directory / name / "manifest.json").exists():
            verify_artifacts(directory / name, root)
            if name == "train":
                target = ResolvedTarget(**{k: np.asarray(v) for k, v in json.loads((directory / name / "target.json").read_text()).items()})
            continue
        selected = frame.iloc[:, indices]
        episodes = count_episodes(selected.to_numpy(), config)
        try:
            current = make_target(episodes.events, len(indices), config, None if target is None else target.edges)
        except ValueError as error:
            # Keep an explicit failure; never silently change held-out bins.
            atomic_json(directory / f"{name}_unsupported.json", {"reason": str(error), "parent": parent})
            if name == "train":
                raise
            continue
        if target is None:
            target = current
        constraints = make_constraints(selected.to_numpy(), config)
        save_bundle(directory / name, selected, episodes, current, constraints, config, parent, root)
    return directory


def verify_prepared(root, directory):
    doc = verify_artifacts(directory, root)
    market = doc["specification"]["market"]
    if file_digest(Path(root) / "data/interim" / f"{market}.pickle") != doc["specification"]["input_sha256"]:
        raise ValueError("empirical input changed since preparation")
    return doc


@dataclass
class EvaluationContext:
    values: np.ndarray
    target: ResolvedTarget
    constraints: ReturnConstraints
    config: CorrectedConfig
    base_params: HestonParams
    names: tuple[str, ...]
    variant: str = "baseline"
    paths: int | None = None

    def parameters(self, vector):
        values = dict(zip(self.names, np.exp(vector)))
        dispersion = values.pop("dispersion", 0.0)
        return self.base_params.update(**values, **self.config.fixed_parameters, dt=1 / self.config.substeps), dispersion

    def realization(self, vector, seed, export=False):
        params, dispersion = self.parameters(vector)
        n = self.values.shape[1] if self.paths is None else min(self.paths, self.values.shape[1])
        mask_rng = np.random.default_rng(np.random.SeedSequence([seed, 881]))
        selection = mask_rng.choice(self.values.shape[1], n, replace=False) if n < self.values.shape[1] else np.arange(n)
        simulation = simulate_modified_heston(params, SimulationConfig(n_paths=n, n_steps=len(self.values),
            sample_every=self.config.substeps, burn_in_steps=self.config.burn_in * self.config.substeps,
            return_mode="accumulated_pre_reset", variance_scheme="cir_exact", stationary_variance=True,
            separate_streams=True, seed=seed, store_state=False, variance_dispersion=dispersion,
            activity_transitions=self.constraints.activity_transitions if self.variant in ("activity", "combined") else None))
        with np.errstate(over="raise", invalid="raise"):
            observed = np.expm1(simulation.returns.to_numpy())
        observed[~np.isfinite(self.values[:, selection])] = np.nan
        episodes = count_episodes(observed, self.config)
        features, _, _ = return_features(observed, self.constraints.tail_thresholds)
        counts, totals = bin_events(episodes.events, self.target.edges)
        details = {"seed": seed, "features": features, "counts": counts, "totals": totals,
            "reset_observations": int(simulation.reset_affected.sum()),
            "reset_count": int(simulation.reset_counts.sum()), "streams": simulation.stream_seeds,
            "episode_counts": episodes.episodes.status.value_counts().to_dict(),
            "outer_volatility_events": int((episodes.events.Volatility > self.config.vol_max).sum())}
        if export:
            details.update(simulation=simulation, observed=observed, episodes=episodes, selected_masks=selection)
        return details, observed

    def evaluate(self, vector, seeds=None):
        seeds = self.config.seeds if seeds is None else seeds
        arrays, realizations = [], []
        try:
            for seed in seeds:
                detail, observed = self.realization(vector, seed)
                arrays.append(observed)
                realizations.append(detail)
            features, _, _ = return_features(np.concatenate(arrays, axis=1), self.constraints.tail_thresholds)
            count = np.sum([d["counts"] for d in realizations], axis=0)
            total = np.sum([d["totals"] for d in realizations], axis=0)
            score = score_curve(self.target, count, total, self.config.min_synthetic_events)
            score.update(score_constraints(features, self.constraints))
            score.update(valid=True, features=features, seeds=list(seeds),
                resets=sum(d["reset_count"] for d in realizations), realizations=realizations)
            return score
        except (ValueError, SimulationError, FloatingPointError, OverflowError) as error:
            return {"valid": False, "feasible": False, "violation": None, "mfht_loss": None, "reason": str(error)}

    def __call__(self, vector):
        result = self.evaluate(vector)
        result.pop("realizations", None)  # Keep search checkpoints compact; validation retains seed-level data.
        return result


def context(root, directory, split="train", variant="baseline", paths=None, seeds=TRAIN_SEEDS[:2], numerical=None):
    doc = verify_prepared(root, directory)
    folder = Path(directory) / split
    bundle = verify_artifacts(folder, root)
    if bundle["specification"]["parent"] != doc["fingerprint"]:
        raise ValueError("empirical bundle belongs to a different run")
    with np.load(folder / "data.npz", allow_pickle=False) as data:
        values = data["values"]
    config = CorrectedConfig(**json.loads((folder / "config.json").read_text()))
    config = replace(config, seeds=tuple(seeds), **(numerical or {}))
    target = ResolvedTarget(**{k: np.asarray(v) for k, v in json.loads((folder / "target.json").read_text()).items()})
    constraint_data = json.loads((folder / "constraints.json").read_text())
    constraints = ReturnConstraints(**constraint_data)
    # Observation model is estimated exclusively from training stocks, even in held-out contexts.
    training_constraints = json.loads((Path(directory) / "train/constraints.json").read_text())
    constraints.activity_transitions = tuple(training_constraints["activity_transitions"])
    names = tuple(doc["specification"]["bounds"])
    if variant in ("heterogeneity", "combined"):
        names += ("dispersion",)
    return EvaluationContext(values, target, constraints, config,
        HestonParams(**doc["specification"]["initial"]), names, variant, paths), doc


def initial_vectors(ctx, spec):
    return [np.log([(.25 if name == "dispersion" else p[name]) for name in ctx.names])
            for p in (spec["initial"], spec["paper"])]


def comparison_gate(left, right, minimum):
    """Require 5% agreement allowing two Monte Carlo standard errors."""
    if any(not x.get("valid", True) for x in (left, right)):
        return {"pass": False, "reason": "invalid simulation"}
    a, b = left["realizations"], right["realizations"]
    ca, cb = np.array([r["counts"] for r in a]), np.array([r["counts"] for r in b])
    ta, tb = np.array([r["totals"] for r in a]), np.array([r["totals"] for r in b])
    ma = np.divide(ta, ca, out=np.full_like(ta, np.nan, dtype=float), where=ca > 0)
    mb = np.divide(tb, cb, out=np.full_like(tb, np.nan, dtype=float), where=cb > 0)
    supported_a, supported_b = ca.sum(0) >= minimum, cb.sum(0) >= minimum
    supported = supported_a & supported_b & np.all(ca > 0, axis=0) & np.all(cb > 0, axis=0)
    union = supported_a | supported_b
    enough = supported.sum() >= 3 and supported.sum() >= .95 * max(union.sum(), 1)
    errors = []
    for i in np.flatnonzero(supported):
        x, y = np.log(ma[:, i]), np.log(mb[:, i])
        errors.append(abs(x.mean() - y.mean()) <= np.log1p(.05) + 2 * np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
    scales = {}
    for key in ("stock_sd", "iqr"):
        x = np.array([r["features"][key] for r in a])
        y = np.array([r["features"][key] for r in b])
        tolerance = .05 * max(abs(x.mean()), abs(y.mean())) + 2 * np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        scales[key] = {"left": x.mean(), "right": y.mean(), "pass": bool(abs(x.mean() - y.mean()) <= tolerance)}
    return {"pass": bool(enough and all(errors) and all(s["pass"] for s in scales.values())),
        "supported_bins": int(supported.sum()), "union_bins": int(union.sum()),
        "bin_checks": errors, "scales": scales}


def numerical_gate(ctx, vector, output, parent, *, paths=256, quick=False):
    output = Path(output)
    identity = digest({"parent": parent, "vector": vector, "variant": ctx.variant, "quick": quick,
                       "paths": paths, "config": asdict(ctx.config)})
    cache = {}
    cache_path = output.with_suffix(".cache.json")
    if cache_path.exists():
        old = json.loads(cache_path.read_text())
        if old["identity"] != identity:
            raise ValueError("numerical cache mismatch")
        cache = old["evaluations"]
    gate_seeds = (4001, 4002) if quick else tuple(range(4001, 4006))

    def evaluate(substeps, burn):
        key = f"{substeps}:{burn}"
        if key not in cache:
            print(f"numerical check substeps={substeps} burn={burn}", flush=True)
            candidate = replace(ctx, paths=min(paths, ctx.values.shape[1]),
                config=replace(ctx.config, substeps=substeps, burn_in=burn, seeds=gate_seeds))
            cache[key] = candidate.evaluate(vector)
            atomic_json(cache_path, {"identity": identity, "evaluations": cache})
        return cache[key]

    resolutions = [10, 20] if quick else [10, 20, 40, 80, 160]
    burnins = [25, 50] if quick else [250, 500, 1000, 2000]
    chosen = None
    checks = []
    for low, high in zip(resolutions, resolutions[1:]):
        result = comparison_gate(evaluate(low, burnins[-1]), evaluate(high, burnins[-1]), ctx.config.min_synthetic_events)
        checks.append({"kind": "substeps", "left": low, "right": high, **result})
        if result["pass"] and (quick or high >= 40):
            chosen = high
            break
    selected_burn = None
    if chosen is not None:
        for low, high in zip(burnins, burnins[1:]):
            result = comparison_gate(evaluate(chosen, low), evaluate(chosen, high), ctx.config.min_synthetic_events)
            checks.append({"kind": "burn_in", "left": low, "right": high, **result})
            if result["pass"]:
                selected_burn = low
                break
    result = {"identity": identity, "parent": parent, "quick": quick, "vector": vector,
        "pass": bool(chosen is not None and selected_burn is not None and not quick),
        "substeps": chosen, "burn_in": selected_burn, "checks": checks}
    atomic_json(output, result)
    return result


def benchmark(ctx, vector, directory, parent):
    started = time.perf_counter()
    probe = replace(ctx, values=ctx.values[:min(500, len(ctx.values))], paths=min(32, ctx.values.shape[1]),
                    config=replace(ctx.config, seeds=(1001,), burn_in=25))
    result = probe.evaluate(vector)
    report = {"parent": parent, "seconds": time.perf_counter() - started, "paths": probe.paths,
        "observations": len(probe.values), "substeps": probe.config.substeps,
        "valid": result.get("valid", False), "not_a_calibration": True, "result": result}
    atomic_json(Path(directory) / "benchmark.json", report)
    return report


def stage_search(root, directory, stage, variant="baseline", restart=0, workers=1,
                 resume=False, time_limit=23 * 3600, smoke=False):
    started = time.monotonic()
    directory = Path(directory)
    gate_path = directory / variant / "numerical.json"
    ctx, doc = context(root, directory, variant=variant)
    spec = doc["specification"]
    seeds = initial_vectors(ctx, spec)
    if not gate_path.exists():
        raise ValueError("run the numerical gate before calibration")
    gate = json.loads(gate_path.read_text())
    if gate.get("parent") != doc["fingerprint"] or not gate.get("pass"):
        raise ValueError("numerical gate did not pass; calibration is blocked")
    numerical = {"substeps": gate["substeps"], "burn_in": gate["burn_in"]}
    require_extension(directory, variant)
    ctx, doc = context(root, directory, variant=variant, paths=256 if stage == "screen" else None,
                       seeds=TRAIN_SEEDS[:2] if stage == "screen" else TRAIN_SEEDS, numerical=numerical)
    bounds = np.log([spec["bounds"].get(name, [0.01, 2.0]) for name in ctx.names])
    if stage == "refine":
        finalists = collect_finalists(directory / variant / "screen", require_complete=True)
        if restart >= len(finalists):
            raise ValueError("refinement index exceeds distinct finalists")
        seeds = [np.array(finalists[restart]["vector"])] + seeds
    folder = directory / variant / stage / f"restart-{restart}"
    setup = {"parent": doc["fingerprint"], "numerical_gate": digest(gate), "config": asdict(ctx.config),
             "variant": variant, "paths": ctx.paths, "stage": stage, "restart": restart,
             "names": ctx.names, "bounds": bounds, "smoke": smoke}
    identity = immutable_manifest(folder / "manifest.json", setup)
    state = differential_search(ctx, bounds, folder, fingerprint=identity, seed=91001 + restart,
        maxiter=0 if smoke else (150 if stage == "screen" else 100),
        popsize=1 if smoke else (12 if stage == "screen" else 8), workers=workers,
        initial=seeds, time_limit=time_limit, resume=resume)
    if stage == "refine" and state["complete"] and state["archive"] and not smoke:
        winner = min(state["archive"], key=lambda e: rank(e["result"]))
        vector = np.asarray(winner["vector"])
        near_low = (vector - bounds[:, 0]) / np.diff(bounds, axis=1).ravel() < .02
        near_high = (bounds[:, 1] - vector) / np.diff(bounds, axis=1).ravel() < .02
        if np.any(near_low | near_high):
            expanded = bounds.copy()
            expanded[near_low, 0] -= np.log(2)
            expanded[near_high, 1] += np.log(2)
            expanded_folder = folder / "expanded"
            expanded_id = immutable_manifest(expanded_folder / "manifest.json", {**setup, "bounds": expanded, "expanded_once": True})
            extra = differential_search(ctx, expanded, expanded_folder, fingerprint=expanded_id, seed=92001 + restart,
                maxiter=100, popsize=8, workers=workers, initial=[vector], time_limit=max(.001, time_limit - (time.monotonic() - started)),
                resume=resume and (expanded_folder / "checkpoint.json").exists())
            remaining = None
            if extra["archive"]:
                best = np.array(min(extra["archive"], key=lambda e: rank(e["result"]))["vector"])
                remaining = np.minimum(best - expanded[:, 0], expanded[:, 1] - best) / np.diff(expanded, axis=1).ravel() < .02
            atomic_json(folder / "boundary.json", {"expanded": True, "remaining_pressure": remaining, "complete": extra["complete"]})
            if not extra["complete"]:
                return {**state, "complete": False, "expansion_pending": True}
    return state


def collect_finalists(folder, limit=4, require_complete=False):
    candidates = []
    checkpoints = sorted(Path(folder).glob("restart-*/checkpoint.json"))
    if require_complete and not checkpoints:
        raise ValueError("no completed search checkpoints")
    if require_complete and Path(folder).name == "screen" and len(checkpoints) != 3:
        raise ValueError("all three screening restarts must complete")
    if require_complete and Path(folder).name == "refine":
        expected = len(collect_finalists(Path(folder).parent / "screen", require_complete=True))
        actual = {int(p.parent.name.split("-")[-1]) for p in checkpoints}
        if actual != set(range(expected)):
            raise ValueError("every selected screening finalist must finish refinement")
    for file in checkpoints:
        checkpoint = json.loads(file.read_text())
        if require_complete and not checkpoint["complete"]:
            raise ValueError(f"search incomplete: {file}")
        manifest = read_manifest(file.parent / "manifest.json")
        if checkpoint.get("search_spec", {}).get("parent") != manifest["fingerprint"] or checkpoint["fingerprint"] != digest(checkpoint.get("search_spec")):
            raise ValueError("checkpoint and search manifest disagree")
        if manifest["specification"].get("smoke"):
            raise ValueError("smoke searches cannot feed scientific validation")
        expanded = file.parent / "expanded/checkpoint.json"
        if expanded.exists():
            extra = json.loads(expanded.read_text())
            expanded_manifest = read_manifest(expanded.parent / "manifest.json")
            if extra.get("search_spec", {}).get("parent") != expanded_manifest["fingerprint"] or extra["fingerprint"] != digest(extra.get("search_spec")):
                raise ValueError("expanded checkpoint and manifest disagree")
            if require_complete and not extra["complete"]:
                raise ValueError("expanded search incomplete")
            candidates.extend(extra["archive"])
        candidates.extend(checkpoint["archive"])
    candidates.sort(key=lambda e: rank(e["result"]))
    selected = []
    for candidate in candidates:
        if all(np.linalg.norm(np.asarray(candidate["vector"]) - e["vector"]) > .05 for e in selected):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def require_extension(directory, variant):
    if variant == "baseline":
        return
    baseline = Path(directory) / "baseline/validation.json"
    if not baseline.exists():
        raise ValueError("extensions require completed baseline validation")
    if json.loads(baseline.read_text()).get("selection_pass"):
        raise ValueError("baseline passed; extensions are not warranted")
    if variant == "combined":
        for name in ("heterogeneity", "activity"):
            path = Path(directory) / name / "validation.json"
            if not path.exists() or json.loads(path.read_text()).get("selection_pass"):
                raise ValueError("combined extension requires both single extensions to fail")


def validate(root, directory, variant="baseline", export=True):
    directory = Path(directory)
    require_extension(directory, variant)
    ctx, doc = context(root, directory, split="selection", variant=variant, seeds=VALIDATION_SEEDS)
    reference_gate = json.loads((directory / variant / "numerical.json").read_text())
    ctx.config = replace(ctx.config, substeps=reference_gate["substeps"], burn_in=reference_gate["burn_in"])
    candidates = collect_finalists(directory / variant / "refine", require_complete=True)
    if not candidates:
        raise ValueError("no valid refinement candidates")
    results = []
    for i, candidate in enumerate(candidates):
        print(f"validating {variant} candidate {i}", flush=True)
        gate_ctx, _ = context(root, directory, variant=variant, numerical={"substeps": ctx.config.substeps, "burn_in": ctx.config.burn_in})
        gate = numerical_gate(gate_ctx, candidate["vector"], directory / variant / f"candidate-{i}-numerical.json", doc["fingerprint"])
        result = {"valid": False, "feasible": False, "mfht_loss": None, "reason": "candidate numerical gate failed"}
        if gate["pass"]:
            evaluation = replace(ctx, config=replace(ctx.config, substeps=gate["substeps"], burn_in=gate["burn_in"]))
            result = evaluation.evaluate(candidate["vector"])
        results.append({"vector": candidate["vector"], "numerical": gate, "result": result})
        atomic_json(directory / variant / "validation_progress.json", results)
    winner = min(results, key=lambda e: rank(e["result"]))
    final = None
    if winner["numerical"]["pass"]:
        test, _ = context(root, directory, split="heldout", variant=variant, seeds=FINAL_SEEDS,
            numerical={"substeps": winner["numerical"]["substeps"], "burn_in": winner["numerical"]["burn_in"]})
        final = test.evaluate(winner["vector"])
    accepted = bool(final and final.get("feasible") and final.get("mfht_pass") and
                    winner["result"].get("feasible") and winner["result"].get("mfht_pass"))
    report = {"parent": doc["fingerprint"], "variant": variant, "candidates": results,
        "selected": winner, "heldout": final, "accepted": accepted,
        "selection_pass": bool(winner["numerical"]["pass"] and winner["result"].get("feasible") and winner["result"].get("mfht_pass")),
        "definition": "held-out stocks; no full-ensemble refit used in this decision"}
    atomic_json(directory / variant / "validation.json", report)
    if export and final:
        export_evaluation(test, winner["vector"], directory / variant / "heldout", doc["fingerprint"], root, FINAL_SEEDS)
    return report


def select_variant(directory):
    """Select on selection stocks only; held-out outcomes never rank variants."""
    candidates = []
    for variant in VARIANTS:
        path = Path(directory) / variant / "validation.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        selected = report["selected"]
        if not selected["result"].get("valid"):
            continue
        candidates.append({"variant": variant, **selected})
    if not candidates:
        raise ValueError("no valid validated model variants")
    best = min(candidates, key=lambda c: rank(c["result"]))
    if best["result"]["feasible"]:
        target_loss = best["result"]["mfht_loss"]
        # Sampling variation in pooled loss, measured by leave-one-seed-out jackknife.
        realizations = best["result"]["realizations"]
        losses = []
        target_data = json.loads((Path(directory) / "selection/target.json").read_text())
        target = ResolvedTarget(**{k: np.asarray(v) for k, v in target_data.items()})
        for omit in range(len(realizations)):
            subset = [r for i, r in enumerate(realizations) if i != omit]
            losses.append(score_curve(target, np.sum([r["counts"] for r in subset], 0),
                np.sum([r["totals"] for r in subset], 0))["mfht_loss"])
        se = float(np.sqrt((len(losses) - 1) / len(losses) * np.sum((np.asarray(losses) - np.mean(losses))**2)))
        eligible = [c for c in candidates if c["result"]["feasible"] and c["result"]["mfht_loss"] <= target_loss + se]
        best = min(eligible, key=lambda c: (VARIANTS.index(c["variant"]), c["result"]["mfht_loss"]))
        best["one_standard_error_tolerance"] = se
    return best


def refit_full(root, directory, workers=1, resume=False, time_limit=23 * 3600):
    directory = Path(directory)
    chosen = select_variant(directory)
    variant = chosen["variant"]
    numerical = {key: chosen["numerical"][key] for key in ("substeps", "burn_in")}
    ctx, doc = context(root, directory, split="full", variant=variant, seeds=TRAIN_SEEDS, numerical=numerical)
    spec = doc["specification"]
    folder = directory / "full_refit"
    atomic_json(directory / "selected_model.json", {"parent": doc["fingerprint"], **chosen,
        "selection_uses": "selection stocks only; held-out evaluation retained separately"})
    bounds = np.log([spec["bounds"].get(name, [.01, 2.0]) for name in ctx.names])
    seed = np.asarray(chosen["vector"])
    bounds[:, 0] = np.minimum(bounds[:, 0], seed - 1e-8)
    bounds[:, 1] = np.maximum(bounds[:, 1], seed + 1e-8)
    identity = immutable_manifest(folder / "manifest.json", {"parent": doc["fingerprint"],
        "variant": variant, "config": asdict(ctx.config), "seed": seed, "bounds": bounds,
        "selection_fingerprint": digest(chosen), "not_held_out": True})
    state = differential_search(ctx, bounds, folder, fingerprint=identity, seed=93001,
        maxiter=100, popsize=8, workers=workers, initial=[seed], time_limit=time_limit, resume=resume)
    if not state["complete"] or not state["archive"]:
        return state
    best = min(state["archive"], key=lambda e: rank(e["result"]))
    gate = numerical_gate(ctx, best["vector"], folder / "numerical.json", doc["fingerprint"])
    report = {"parent": doc["fingerprint"], "variant": variant, "selected": best,
              "numerical": gate, "accepted": False, "not_held_out": True}
    if gate["pass"]:
        ctx.config = replace(ctx.config, substeps=gate["substeps"], burn_in=gate["burn_in"], seeds=FINAL_SEEDS)
        evaluation = ctx.evaluate(best["vector"])
        report["evaluation"] = evaluation
        report["accepted"] = bool(evaluation.get("feasible") and evaluation.get("mfht_pass"))
        if evaluation.get("valid"):
            export_evaluation(ctx, best["vector"], folder / "export", identity, root, FINAL_SEEDS)
    atomic_json(folder / "result.json", report)
    return state


def summarize(root, directory):
    directory = Path(directory)
    doc = verify_prepared(root, directory)
    reports = {}
    for variant in VARIANTS:
        path = directory / variant / "validation.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report["parent"] != doc["fingerprint"]:
                raise ValueError("stale validation report")
            reports[variant] = report
    full_path = directory / "full_refit/result.json"
    full = json.loads(full_path.read_text()) if full_path.exists() else None
    selected = select_variant(directory) if any(r["selected"]["result"].get("valid") for r in reports.values()) else None
    lines = [f"# Corrected Heston: {doc['specification']['market']}", "",
        "Completed, retained episodes (2–300 observations); physical volatility. Historical fits are a separate statistic.", "",
        "| Model | Selection feasible | Selection RMS log MFHT | Held-out passed |",
        "|---|---:|---:|---:|"]
    for name, report in reports.items():
        score = report["selected"]["result"]
        lines.append(f"| {name} | {score.get('feasible', False)} | {score.get('rms_log_mfht')} | {report['accepted']} |")
    lines += ["", f"Selected specification: {selected['variant'] if selected else 'none validated'}.",
        f"Full-data refit passed: {full.get('accepted', False) if full else 'not run'}.",
        "", "The full-data refit is not held-out performance. Failed constraints are retained in evaluation JSONs.",
        "Empirical bands are stock-bootstrap intervals; synthetic bands show independent-realization spread.",
        "Five-year calendar blocks and zero runs are reported in each empirical bundle."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    historical = directory / "historical"
    if (historical / "manifest.json").exists():
        old = verify_artifacts(historical, root)
        if old["specification"]["parent"] != doc["fingerprint"]:
            raise ValueError("stale historical comparison")
        for label in ("empirical", "synthetic"):
            frame = pd.read_csv(historical / f"{label}.csv")
            axes[0].plot((frame.lower + frame.upper) / 2, frame.mfht, label=label)
    axes[0].set_title("Original fitting pipeline / legacy statistic")
    for ax, folder, title in ((axes[1], directory / "baseline/heldout", "Corrected baseline / held-out stocks"),
                              (axes[2], directory / "full_refit/export", "Selected specification / full-data refit")):
        ax.set_title(title)
        if (folder / "manifest.json").exists():
            frame = load_verified_comparison(folder, directory, root)
            x = (frame.lower + frame.upper) / 2
            ax.plot(x, frame.mfht, label="empirical")
            ax.fill_between(x, frame.ci_lower, frame.ci_upper, alpha=.2)
            ax.plot(x, frame.synthetic_mfht, label="synthetic")
            ax.fill_between(x, frame.simulation_lower, frame.simulation_upper, alpha=.2)
        else:
            ax.text(.5, .5, "Not yet available", ha="center", transform=ax.transAxes)
    for ax in axes:
        ax.set(xlabel="Physical event volatility", ylabel="MFHT (observations)")
        if ax.lines:
            ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(directory / "model_comparison.png", dpi=160)
    plt.close(fig)
    atomic_json(directory / "summary.json", {"parent": doc["fingerprint"], "selected": selected,
        "variants": reports, "full_refit": full, "successful": bool(full and full["accepted"])})
    return directory / "report.md"


def historical_reference(root, directory):
    """Recompute the original fitting pipeline; never trust copied old recounts."""
    from .paper_reproduction import count_fortran_hitting_events
    directory = Path(directory)
    ctx, doc = context(root, directory, split="full")
    folder = directory / "historical"
    if (folder / "manifest.json").exists():
        old = verify_artifacts(folder, root)
        if old["specification"]["parent"] != doc["fingerprint"]:
            raise ValueError("historical reference belongs to another run")
        return
    old = doc["specification"]["historical"]
    setup = old["setup"]
    params = HestonParams(**old["parameters"])
    simulation = simulate_modified_heston(params, SimulationConfig(n_paths=min(256, ctx.values.shape[1]),
        n_steps=3030, sample_every=setup["sampling_interval_steps"],
        burn_in_steps=setup["sampling_burn_in_steps"], return_mode=setup["sampling_return_mode"],
        correlated_noise=setup.get("correlated_noise", False), seed=500001, store_state=False))
    returns = simulation.returns.to_numpy()
    if setup["synthetic_return_transform"] == "simple":
        returns = np.expm1(returns)
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in (("empirical", ctx.values), ("synthetic", returns)):
        events = count_fortran_hitting_events(values, ctx.config.sigma, missing_policy="break",
            reset_on_discard=setup["count_method"] == "fortran_clean", tau_max=setup["tau_max"])
        count, total = bin_events(events, ctx.target.edges)
        pd.DataFrame({"lower": ctx.target.edges[:-1], "upper": ctx.target.edges[1:], "events": count,
            "mfht": np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)}).to_csv(folder / f"{name}.csv", index=False)
    immutable_manifest(folder / "manifest.json", {"parent": doc["fingerprint"], "source": source_fingerprint(root),
        "label": "original fitting pipeline, not the corrected FHT statistic", "settings": old, "seed": 500001,
        "artifacts": {p.name: file_digest(p) for p in folder.glob("*.csv")}})


def load_verified_comparison(export_directory, run_directory, root):
    """Notebook entry point: verify current fit identity as well as file hashes."""
    export_directory, run_directory = Path(export_directory), Path(run_directory)
    parent = verify_prepared(root, run_directory)
    export = verify_artifacts(export_directory, root)
    full_refit = export_directory.parent.name == "full_refit"
    if full_refit:
        intermediate = read_manifest(export_directory.parent / "manifest.json")
        if intermediate["specification"]["parent"] != parent["fingerprint"]:
            raise ValueError("refit belongs to a different empirical run")
        expected = intermediate["fingerprint"]
        record = json.loads((export_directory.parent / "result.json").read_text())
    else:
        expected = parent["fingerprint"]
        record = json.loads((export_directory.parent / "validation.json").read_text())
    if export["specification"]["parent"] != expected or export["specification"]["parameters"] != digest(record["selected"]["vector"]):
        raise ValueError("recount does not match the currently selected parameters")
    return pd.read_csv(export_directory / "comparison.csv")


def export_evaluation(ctx, vector, folder, parent, root, seeds):
    folder = Path(folder)
    if (folder / "manifest.json").exists():
        manifest = verify_artifacts(folder, root)
        if manifest["specification"]["parent"] != parent or manifest["specification"]["parameters"] != digest(vector):
            raise ValueError("incompatible existing export")
        return
    folder.mkdir(parents=True, exist_ok=True)
    records, counts, totals, all_features = [], [], [], []
    cross = []
    for seed in seeds:
        detail, observed = ctx.realization(vector, seed, export=True)
        simulation = detail.pop("simulation")
        episodes = detail.pop("episodes")
        detail.pop("observed")
        selection = detail.pop("selected_masks")
        mask = np.isfinite(observed)
        # Full latent log prices retained, alongside the observed availability mask.
        np.savez_compressed(folder / f"series-{seed}.npz", log_returns=simulation.returns.to_numpy(),
            log_prices=simulation.log_prices.to_numpy(), prices=simulation.prices.to_numpy(),
            observed_returns=observed, available=mask, selected_masks=selection,
            reset_counts=simulation.reset_counts)
        episodes.episodes.to_csv(folder / f"episodes-{seed}.csv.gz", index=False)
        detail["zero_runs"] = zero_run_summary(observed)
        records.append(detail)
        counts.append(detail["counts"])
        totals.append(detail["totals"])
        all_features.append(observed)
        for pair in ((.1, 1.5), (-.5, -1.5), (-1., -2.)):
            alternate = replace(ctx.config, start=pair[0], end=pair[1])
            event = count_episodes(observed, alternate).events
            count, total = bin_events(event, ctx.target.edges)
            for i, (n, t) in enumerate(zip(count, total)):
                cross.append({"seed": seed, "start": pair[0], "end": pair[1], "bin": i,
                              "events": int(n), "total_fht": float(t)})
    features, _, _ = return_features(np.concatenate(all_features, axis=1), ctx.constraints.tail_thresholds)
    count, total = np.sum(counts, 0), np.sum(totals, 0)
    score = score_curve(ctx.target, count, total, ctx.config.min_synthetic_events)
    score.update(score_constraints(features, ctx.constraints))
    comparison = ctx.target.frame()
    comparison["synthetic_events"] = count
    comparison["synthetic_mfht"] = np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)
    seed_means = np.divide(totals, counts, out=np.full_like(np.array(totals), np.nan, dtype=float), where=np.array(counts) > 0)
    supported = np.any(np.isfinite(seed_means), axis=0)
    interval = np.full((2, len(count)), np.nan)
    interval[:, supported] = np.nanquantile(seed_means[:, supported], [.025, .975], axis=0)
    comparison["simulation_lower"] = interval[0]
    comparison["simulation_upper"] = interval[1]
    comparison.to_csv(folder / "comparison.csv", index=False)
    pd.DataFrame(cross).to_csv(folder / "threshold_sensitivity_synthetic.csv", index=False)
    empirical_cross = []
    for pair in ((.1, 1.5), (-.5, -1.5), (-1., -2.)):
        event = count_episodes(ctx.values, replace(ctx.config, start=pair[0], end=pair[1])).events
        c, t = bin_events(event, ctx.target.edges)
        for i, (n, total_fht) in enumerate(zip(c, t)):
            empirical_cross.append({"start": pair[0], "end": pair[1], "bin": i, "events": int(n), "total_fht": float(total_fht)})
    pd.DataFrame(empirical_cross).to_csv(folder / "threshold_sensitivity_empirical.csv", index=False)
    atomic_json(folder / "evaluation.json", {"score": score, "seeds": records, "features": features,
        "config": asdict(ctx.config), "variant": ctx.variant, "parameters": ctx.parameters(vector)[0].to_dict()})
    plot_comparison(comparison, folder / "comparison.png")
    artifacts = {p.name: file_digest(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}
    immutable_manifest(folder / "manifest.json", {"parent": parent, "parameters": digest(vector),
        "source": source_fingerprint(root), "counter": COUNTER_VERSION, "artifacts": artifacts,
        "config": asdict(ctx.config), "seeds": seeds, "variant": ctx.variant})


def plot_comparison(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = (frame.lower + frame.upper) / 2
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(x, frame.mfht, label="Empirical corrected FHT")
    ax.fill_between(x, frame.ci_lower, frame.ci_upper, alpha=.2)
    ax.plot(x, frame.synthetic_mfht, label="Synthetic corrected FHT")
    ax.fill_between(x, frame.simulation_lower, frame.simulation_upper, alpha=.2)
    ax.set(xlabel="Physical event volatility", ylabel="MFHT (observations)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
