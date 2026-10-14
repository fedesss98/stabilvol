"""Checkpointed rand/1/bin DE with explicit feasibility-first selection.

Uses a public, self-contained generation loop so RNG and pending trial batches
can be restored exactly; no dependence on SciPy optimizer private state.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from .run_io import atomic_json, digest, plain

_EVALUATOR = None


def _initialize(evaluator):
    global _EVALUATOR
    _EVALUATOR = evaluator


def _evaluate(vector):
    return _EVALUATOR(vector)


def rank(result):
    if not result.get("valid", True) or result.get("mfht_loss") is None:
        return (2, float("inf"), float("inf"))
    if result["feasible"]:
        return (0, result["mfht_loss"], 0)
    violation = result.get("violation")
    return (1, violation if violation is not None else float("inf"), result["mfht_loss"])


def retain(archive, vector, result):
    if not result.get("valid", True) or result.get("mfht_loss") is None:
        return archive
    key = tuple(np.round(vector, 10))
    entries = {tuple(np.round(e["vector"], 10)): e for e in archive}
    entries[key] = {"vector": list(vector), "result": result}
    all_entries = list(entries.values())
    # Retain both useful feasible candidates and unconstrained shape optima.
    winners = sorted(all_entries, key=lambda e: rank(e["result"]))[:20]
    winners += sorted(all_entries, key=lambda e: e["result"]["mfht_loss"])[:20]
    return list({tuple(e["vector"]): e for e in winners}.values())


def write_best(output, state, identity):
    valid = state["archive"]
    feasible = [entry for entry in valid if entry["result"]["feasible"]]
    atomic_json(Path(output) / "best.json", {"complete": state["complete"], "fingerprint": identity,
        "best_feasible": min(feasible, key=lambda e: e["result"]["mfht_loss"]) if feasible else None,
        "best_unconstrained": min(valid, key=lambda e: e["result"]["mfht_loss"]) if valid else None,
        "best_ranked": min(valid, key=lambda e: rank(e["result"])) if valid else None})


def differential_search(evaluator, bounds, output, *, fingerprint, seed=91001,
                        maxiter=150, popsize=12, workers=1, initial=(),
                        time_limit=23 * 3600, resume=False):
    """Atomic checkpoints include partially evaluated generations and RNG state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "checkpoint.json"
    bounds = np.asarray(bounds, float)
    lower, upper = bounds.T
    if len(lower) < 1 or np.any(~np.isfinite(bounds)) or np.any(upper <= lower):
        raise ValueError("search requires positive-width finite transformed bounds")
    if workers < 1 or popsize < 1 or maxiter < 0 or time_limit <= 0:
        raise ValueError("invalid search budget")
    search_spec = {"parent": fingerprint, "bounds": bounds, "seed": seed,
        "maxiter": maxiter, "popsize": popsize, "strategy": "rand1bin-v1"}
    identity = digest(search_spec)
    rng = np.random.default_rng(seed)
    if path.exists():
        if not resume:
            raise ValueError("checkpoint exists: use resume or a new directory")
        state = json.loads(path.read_text())
        if state["fingerprint"] != identity:
            raise ValueError("incompatible checkpoint")
        if state.get("complete"):
            write_best(output, state, identity)
            return state
        rng.bit_generator.state = state["rng"]
    else:
        size = max(4, popsize * len(lower))
        population = rng.uniform(lower, upper, (size, len(lower)))
        for index, vector in enumerate(initial):
            if index >= size:
                break
            population[index] = np.clip(vector, lower, upper)
        state = {"fingerprint": identity, "search_spec": plain(search_spec), "generation": 0, "evaluations": 0,
            "population": population.tolist(), "results": [None] * size,
            "pending": population.tolist(), "pending_results": [], "archive": [],
            "history": [], "complete": False, "rng": plain(rng.bit_generator.state)}
    started = time.monotonic()
    executor = ProcessPoolExecutor(workers, initializer=_initialize, initargs=(evaluator,)) if workers > 1 else None
    try:
        while not state["complete"]:
            pending = state["pending"]
            done = len(state["pending_results"])
            if done < len(pending):
                if time.monotonic() - started >= time_limit:
                    break
                batch = pending[done:done + workers]
                results = list(executor.map(_evaluate, batch)) if executor else [evaluator(v) for v in batch]
                for vector, result in zip(batch, results):
                    result = plain(result)
                    state["pending_results"].append(result)
                    state["archive"] = retain(state["archive"], vector, result)
                    state["evaluations"] += 1
                state["rng"] = plain(rng.bit_generator.state)
                atomic_json(path, state)
                continue
            for i, result in enumerate(state["pending_results"]):
                if state["results"][i] is None or rank(result) < rank(state["results"][i]):
                    state["population"][i] = pending[i]
                    state["results"][i] = result
            best = min(state["results"], key=rank)
            state["history"].append({"generation": state["generation"], "evaluations": state["evaluations"],
                "best_rank": rank(best), "best_mfht": best.get("mfht_loss"),
                "feasible_population": sum(r.get("feasible", False) for r in state["results"])})
            print(f"generation={state['generation']} evaluations={state['evaluations']} best={rank(best)}", flush=True)
            if state["generation"] >= maxiter:
                state["complete"] = True
            else:
                population = np.asarray(state["population"])
                trials = []
                mutation = rng.uniform(0.5, 1.0)
                for i, parent in enumerate(population):
                    indices = np.delete(np.arange(len(population)), i)
                    a, b, c = rng.choice(indices, 3, replace=False)
                    mutant = population[a] + mutation * (population[b] - population[c])
                    outside = (mutant < lower) | (mutant > upper)
                    mutant[outside] = rng.uniform(lower[outside], upper[outside])
                    cross = rng.random(len(lower)) < 0.7
                    cross[rng.integers(len(lower))] = True
                    trials.append(np.where(cross, mutant, parent).tolist())
                state["generation"] += 1
                state["pending"] = trials
                state["pending_results"] = []
            state["rng"] = plain(rng.bit_generator.state)
            atomic_json(path, state)
    finally:
        if executor:
            executor.shutdown()
    state["rng"] = plain(rng.bit_generator.state)
    atomic_json(path, state)
    write_best(output, state, identity)
    return state
