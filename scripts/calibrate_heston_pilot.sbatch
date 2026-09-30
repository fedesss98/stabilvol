#!/usr/bin/env bash
#SBATCH --job-name=heston-mfht-pilot
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

repo_root="/data/qmla/famato/stabilvol"
cd "$repo_root"
job_id="$SLURM_ARRAY_JOB_ID"

HESTON_CONFIG="$repo_root/configs/heston_mfht_pilot.json"
HESTON_DATABASE="$repo_root/data/processed/trapezoidal_selection/stabilvol_filtered.sqlite"
HESTON_OUTPUT_ROOT="$repo_root/data/processed/heston_calibration"
HESTON_FIGURE_ROOT="$repo_root/visualization/heston_calibration"

echo "Heston MFHT pilot: job=$job_id config=$HESTON_CONFIG"
uv run python scripts/calibrate_heston.py \
    --config-json "$HESTON_CONFIG" \
    --database "$HESTON_DATABASE" \
    --markets UN \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$HESTON_OUTPUT_ROOT/$job_id/UN" \
    --figure-dir "$HESTON_FIGURE_ROOT/$job_id"
