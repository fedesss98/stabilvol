#!/usr/bin/env bash
#SBATCH --job-name=heston-calibration
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

repo_root="/data/qmla/famato/stabilvol"
cd "$repo_root"

markets=(UN UW LN JT)
market="${markets[$SLURM_ARRAY_TASK_ID]}"
job_id="$SLURM_ARRAY_JOB_ID"
workers="$SLURM_CPUS_PER_TASK"

HESTON_CONFIG="$repo_root/configs/heston_default_grid.json"
HESTON_DATABASE="$repo_root/data/processed/trapezoidal_selection/stabilvol_filtered.sqlite"
HESTON_OUTPUT_ROOT="$repo_root/data/processed/heston_calibration"
HESTON_FIGURE_ROOT="$repo_root/visualization/heston_calibration"

echo "Job $job_id: market=$market workers=$workers config=$HESTON_CONFIG"
uv run python scripts/calibrate_heston.py \
    --config-json "$HESTON_CONFIG" \
    --database "$HESTON_DATABASE" \
    --markets "$market" \
    --workers "$workers" \
    --output-dir "$HESTON_OUTPUT_ROOT/$job_id/$market" \
    --figure-dir "$HESTON_FIGURE_ROOT/$job_id"
