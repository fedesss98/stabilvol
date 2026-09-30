#!/usr/bin/env bash
#SBATCH --job-name=heston-smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=heston-smoke-%j.out
#SBATCH --error=heston-smoke-%j.err

set -euo pipefail

repo_root="${HESTON_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"
python_bin="${HESTON_PYTHON:-python3}"
database="${HESTON_DATABASE:-data/processed/trapezoidal_selection/stabilvol_filtered.sqlite}"
job_id="${SLURM_JOB_ID:-manual}"

if [[ ! -f scripts/calibrate_heston.py || ! -f configs/heston_quick.json ]]; then
    echo "Submit from the repository root or set HESTON_PROJECT_ROOT." >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable unavailable: $python_bin (set HESTON_PYTHON)." >&2
    exit 2
fi
if [[ ! -r "$database" || ! -r data/interim/UN.pickle ]]; then
    echo "Missing/read-protected database or data/interim/UN.pickle." >&2
    echo "Database: $database" >&2
    exit 2
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/heston-smoke-mpl-${job_id}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

echo "Heston smoke test: job=$job_id database=$database python=$python_bin"
"$python_bin" -u scripts/calibrate_heston.py \
    --config-json configs/heston_quick.json \
    --database "$database" \
    --output-dir "data/processed/heston_calibration/smoke-$job_id" \
    --figure-dir "visualization/heston_calibration/smoke-$job_id"
