#!/usr/bin/env python3
"""Prepare, gate, fit, resume, and validate corrected MFHT experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from stabilvol.heston.corrected import CorrectedConfig, count_episodes, make_target, make_constraints
from stabilvol.heston.simulation import HestonParams, daily_parameters
from stabilvol.heston.run_io import atomic_json, digest, source_fingerprint, immutable_manifest, verify_artifacts
from stabilvol.heston.search import differential_search
from stabilvol.heston.workflow import (VARIANTS, EvaluationContext, prepare, context, initial_vectors,
    numerical_gate, benchmark, stage_search, validate, summarize, refit_full, require_extension, historical_reference,
    export_evaluation)


def smoke(root, run_dir, market, source_fit):
    """Small real-data integration check, permanently marked non-scientific."""
    source = json.loads((source_fit / market / "trial_config.json").read_text())
    frame = pd.read_pickle(root / "data/interim" / f"{market}.pickle")
    frame = frame.loc[:, frame.notna().sum() >= 3030].iloc[-512:]
    frame = frame.loc[:, frame.notna().sum() >= 400].iloc[:, :32]
    if frame.shape[1] < 7:
        raise ValueError("not enough recent real stock histories for smoke check")
    config = CorrectedConfig(sigma=source["threshold_sigma"], burn_in=25,
        min_events=20, min_stocks=2, bootstraps=20, seeds=(1001,))
    episodes = count_episodes(frame.to_numpy(), config)
    target = make_target(episodes.events, frame.shape[1], config)
    constraints = make_constraints(frame.to_numpy(), config)
    params = daily_parameters(HestonParams(), .01)
    names = ("a", "b", "aa", "bb", "cc")
    ctx = EvaluationContext(frame.to_numpy(), target, constraints, config, params, names, paths=16)
    vector = np.log([getattr(params, key) for key in names])
    folder = run_dir / "smoke" / market
    fingerprint = immutable_manifest(folder / "manifest.json", {"source": source_fingerprint(root),
        "market": market, "config": asdict(config), "data": digest(frame.to_numpy()), "not_a_calibration": True})
    report = benchmark(ctx, vector, folder, fingerprint)
    if not report["valid"]:
        raise ValueError("smoke benchmark simulation failed")
    bounds = np.column_stack((vector - .25, vector + .25))
    state = differential_search(ctx, bounds, folder / "search", fingerprint=fingerprint,
        maxiter=0, popsize=1, initial=[vector], time_limit=120, resume=(folder / "search/checkpoint.json").exists())
    atomic_json(folder / "smoke.json", {"benchmark": report, "complete": state["complete"],
        "evaluations": state["evaluations"], "not_a_calibration": True})
    export_evaluation(ctx, vector, folder / "diagnostic_export", fingerprint, root, (1001, 1002))
    print(f"Smoke result: {folder}; this is not a calibrated fit", flush=True)


def write_jobs(run_dir, source_fit, markets, variant, root, config_file):
    """Generate portable Slurm files only after a recorded local benchmark."""
    for market in markets:
        candidates = [run_dir / "smoke" / market / "benchmark.json", run_dir / market / "benchmark.json"]
        valid = False
        for path in candidates:
            if path.exists() and json.loads(path.read_text()).get("valid"):
                verify_artifacts(path.parent, root)
                valid = True
                break
        if not valid:
            raise ValueError(f"benchmark {market} locally before generating jobs")
    import shlex
    try:
        relative_run = run_dir.relative_to(root).as_posix()
        relative_source = source_fit.relative_to(root).as_posix()
        relative_config = config_file.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("portable Slurm generation requires run/source directories within the repository") from error
    folder = root / "scripts" / "heston_corrected" / run_dir.name / variant
    folder.mkdir(parents=True, exist_ok=True)
    stages = {"prepare": 1, "screen": 3, "refine": 4, "validate": 1, "refit": 1}
    for stage, slots in stages.items():
        header = f"""#!/usr/bin/env bash
#SBATCH --job-name=hmfht-{stage}
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --array=0-{len(markets) * slots - 1}%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
repo_root="${{HESTON_REPO_ROOT:-/data/qmla/famato/stabilvol}}"
cd "$repo_root"
markets=({' '.join(shlex.quote(m) for m in markets)})
market="${{markets[$((SLURM_ARRAY_TASK_ID / {slots}))]}}"
restart="$((SLURM_ARRAY_TASK_ID % {slots}))"
common=(--run-dir {shlex.quote(relative_run)} --source-fit {shlex.quote(relative_source)} --config {shlex.quote(relative_config)} --markets "$market" --variant {shlex.quote(variant)} --workers "$SLURM_CPUS_PER_TASK")
"""
        if stage == "prepare":
            body = "uv run python scripts/calibrate_heston_corrected.py prepare \"${common[@]}\"\nuv run python scripts/calibrate_heston_corrected.py gate \"${common[@]}\"\n"
        elif stage in ("screen", "refine", "refit"):
            body = f"uv run python scripts/calibrate_heston_corrected.py resume --stage {stage} --restart \"$restart\" \"${{common[@]}}\"\n"
        else:
            body = "uv run python scripts/calibrate_heston_corrected.py validate \"${common[@]}\"\n"
        (folder / f"{stage}.sbatch").write_text(header + body, encoding="utf-8", newline="\n")
    submit = """#!/usr/bin/env bash
set -euo pipefail
cd "${HESTON_REPO_ROOT:-/data/qmla/famato/stabilvol}"
mkdir -p logs
"""
    relative_jobs = folder.relative_to(root).as_posix()
    previous = ""
    # Full refit is deliberately separate: extensions must be compared first if baseline fails.
    for stage in ("prepare", "screen", "refine", "validate"):
        dependency = f'--dependency=afterok:"${previous}" ' if previous else ""
        submit += f'{stage}_job=$(sbatch --parsable {dependency}{shlex.quote(relative_jobs + "/" + stage + ".sbatch")})\n'
        previous = f"{stage}_job"
    submit += 'echo "Validation job: $validate_job. Inspect summarize before extensions or full-data refit."\n'
    (folder / "submit.sh").write_text(submit, encoding="utf-8", newline="\n")
    print(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "benchmark", "gate", "screen", "refine", "validate", "refit", "summarize", "resume", "jobs", "smoke"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/processed/heston_corrected/v1"))
    parser.add_argument("--source-fit", type=Path, default=Path("data/processed/heston_calibration/conditional_mfht_joint_explore"))
    parser.add_argument("--config", type=Path, default=Path("configs/heston_corrected.json"))
    parser.add_argument("--markets", nargs="+", choices=("UN", "LN", "UW", "JT"), default=["UN", "LN"])
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hours", type=float, default=23)
    parser.add_argument("--stage", choices=("screen", "refine", "refit"), default="screen")
    args = parser.parse_args()
    if args.workers < 1 or args.hours <= 0 or args.restart < 0:
        parser.error("invalid workers, restart, or time limit")
    run_dir = args.run_dir.resolve()
    source_fit = args.source_fit.resolve()
    operation = args.stage if args.operation == "resume" else args.operation
    overrides = json.loads(args.config.read_text())
    if operation == "jobs":
        write_jobs(run_dir, source_fit, args.markets, args.variant, ROOT, args.config.resolve())
        return
    for market in args.markets:
        directory = run_dir / market
        if operation == "smoke":
            smoke(ROOT, run_dir, market, source_fit)
            continue
        if market in ("UW", "JT"):
            for initial in ("UN", "LN"):
                result = run_dir / initial / "full_refit/result.json"
                if not result.exists() or not json.loads(result.read_text()).get("accepted"):
                    raise ValueError("UN and LN must pass before extending to UW/JT")
        if operation == "prepare":
            prepare(ROOT, run_dir, market, source_fit, overrides)
            historical_reference(ROOT, directory)
        elif operation in ("gate", "benchmark"):
            require_extension(directory, args.variant)
            ctx, doc = context(ROOT, directory, variant=args.variant)
            vector = initial_vectors(ctx, doc["specification"])[0]
            if operation == "benchmark":
                print(benchmark(ctx, vector, directory, doc["fingerprint"])["seconds"])
            else:
                for index, vector in enumerate(initial_vectors(ctx, doc["specification"])):
                    gate = numerical_gate(ctx, vector, directory / args.variant / f"initial-{index}-numerical.json", doc["fingerprint"])
                    if gate["pass"]:
                        break
                atomic_json(directory / args.variant / "numerical.json", gate)
                if not gate["pass"]:
                    raise SystemExit("Numerical gate failed; inspect its report before calibration")
        elif operation in ("screen", "refine"):
            resume = args.operation == "resume" and (directory / args.variant / operation / f"restart-{args.restart}/checkpoint.json").exists()
            if operation == "refine":
                from stabilvol.heston.workflow import collect_finalists
                candidates = collect_finalists(directory / args.variant / "screen", require_complete=True)
                if args.restart >= len(candidates):
                    print(f"{market}: no distinct finalist for slot {args.restart}; skipped")
                    continue
            state = stage_search(ROOT, directory, operation, args.variant, args.restart, args.workers,
                resume=resume, time_limit=args.hours * 3600)
            if not state["complete"]:
                raise SystemExit(75)  # Prevent afterok dependencies from consuming a partial stage.
        elif operation == "validate":
            validate(ROOT, directory, args.variant)
            print(summarize(ROOT, directory))
        elif operation == "refit":
            state = refit_full(ROOT, directory, args.workers,
                resume=args.operation == "resume" and (directory / "full_refit/checkpoint.json").exists(), time_limit=args.hours * 3600)
            print(summarize(ROOT, directory))
            if not state["complete"]:
                raise SystemExit(75)
        elif operation == "summarize":
            print(summarize(ROOT, directory))


if __name__ == "__main__":
    main()
