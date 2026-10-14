#!/usr/bin/env bash
#SBATCH --job-name=hmfht-prepare
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
repo_root="${HESTON_REPO_ROOT:-/data/qmla/famato/stabilvol}"
cd "$repo_root"
markets=(UN LN)
market="${markets[$((SLURM_ARRAY_TASK_ID / 1))]}"
restart="$((SLURM_ARRAY_TASK_ID % 1))"
common=(--run-dir data/processed/heston_corrected/ready_v2 --source-fit data/processed/heston_calibration/conditional_mfht_joint_explore --config configs/heston_corrected.json --markets "$market" --variant baseline --workers "$SLURM_CPUS_PER_TASK")
uv run python scripts/calibrate_heston_corrected.py prepare "${common[@]}"
uv run python scripts/calibrate_heston_corrected.py gate "${common[@]}"
