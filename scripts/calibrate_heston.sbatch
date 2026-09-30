#!/usr/bin/env bash
#SBATCH --job-name=heston-calibration
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=heston-%A_%a.out
#SBATCH --error=heston-%A_%a.err

set -euo pipefail

repo_root="${HESTON_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"
if [[ ! -f scripts/calibrate_heston.py ]]; then
    echo "Submit from the repository root or set HESTON_PROJECT_ROOT." >&2
    exit 2
fi

markets=(UN UW LN JT)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$task_id" =~ ^[0-3]$ ]]; then
    echo "Unsupported array index $task_id; use --array=0-3 (UN, UW, LN, JT)." >&2
    exit 2
fi
market="${markets[$task_id]}"
job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
workers="${SLURM_CPUS_PER_TASK:-1}"

python_bin="${HESTON_PYTHON:-python3}"
config="${HESTON_CONFIG:-configs/heston_default_grid.json}"
database="${HESTON_DATABASE:-data/processed/trapezoidal_selection/stabilvol_filtered.sqlite}"
output_root="${HESTON_OUTPUT_ROOT:-data/processed/heston_calibration}"
figure_root="${HESTON_FIGURE_ROOT:-visualization/heston_calibration}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable unavailable: $python_bin (set HESTON_PYTHON)." >&2
    exit 2
fi
if [[ ! -f "$config" || ! -f "$database" || ! -f "data/interim/${market}.pickle" ]]; then
    echo "Missing config, empirical FHT database, or data/interim/${market}.pickle." >&2
    echo "Config: $config; database: $database" >&2
    exit 2
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/heston-mpl-${job_id}-${market}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

echo "Job $job_id task $task_id: market=$market workers=$workers python=$python_bin"
"$python_bin" -u scripts/calibrate_heston.py \
    --config-json "$config" \
    --database "$database" \
    --markets "$market" \
    --workers "$workers" \
    --output-dir "$output_root/$job_id/$market" \
    --figure-dir "$figure_root/$job_id"
