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

repo_root="${HESTON_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"
if [[ ! -f scripts/reproduce_heston_paper.py ]]; then
    echo "Submit from the repository root or set HESTON_PROJECT_ROOT." >&2
    exit 2
fi

python_bin="${HESTON_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable unavailable: $python_bin (set HESTON_PYTHON)." >&2
    exit 2
fi

job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
seed="$(( ${HESTON_PAPER_BASE_SEED:-12345} + task_id ))"
output_root="${HESTON_PAPER_OUTPUT_ROOT:-data/processed/heston_paper}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/heston-paper-mpl-${job_id}-${task_id}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

echo "Paper reproduction: job=$job_id task=$task_id seed=$seed python=$python_bin"
"$python_bin" -u scripts/reproduce_heston_paper.py \
    --seed "$seed" \
    --output-dir "$output_root/$job_id/seed-$seed"
