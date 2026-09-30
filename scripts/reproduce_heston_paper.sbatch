#!/usr/bin/env bash
#SBATCH --job-name=heston-paper
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=heston-paper-%A_%a.out
#SBATCH --error=heston-paper-%A_%a.err

set -euo pipefail

repo_root="/data/qmla/famato/stabilvol"
cd "$repo_root"
job_id="$SLURM_ARRAY_JOB_ID"
task_id="$SLURM_ARRAY_TASK_ID"
seed="$((12345 + task_id))"
output_root="$repo_root/data/processed/heston_paper"

echo "Paper reproduction: job=$job_id task=$task_id seed=$seed"
uv run python scripts/reproduce_heston_paper.py \
    --seed "$seed" \
    --output-dir "$output_root/$job_id/seed-$seed"
