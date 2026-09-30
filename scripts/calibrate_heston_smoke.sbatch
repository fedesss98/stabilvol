#!/usr/bin/env bash
#SBATCH --job-name=heston-smoke
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

repo_root="/data/qmla/famato/stabilvol"
cd "$repo_root"
job_id="$SLURM_JOB_ID"

HESTON_CONFIG="$repo_root/configs/heston_quick.json"
HESTON_DATABASE="$repo_root/data/processed/trapezoidal_selection/stabilvol_filtered.sqlite"
HESTON_OUTPUT_ROOT="$repo_root/data/processed/heston_calibration"
HESTON_FIGURE_ROOT="$repo_root/visualization/heston_calibration"

echo "Heston smoke test: job=$job_id database=$HESTON_DATABASE"
uv run python scripts/calibrate_heston.py \
    --config-json "$HESTON_CONFIG" \
    --database "$HESTON_DATABASE" \
    --output-dir "$HESTON_OUTPUT_ROOT/smoke-$job_id" \
    --figure-dir "$HESTON_FIGURE_ROOT/smoke-$job_id"
