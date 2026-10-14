#!/usr/bin/env bash
set -euo pipefail
cd "${HESTON_REPO_ROOT:-/data/qmla/famato/stabilvol}"
mkdir -p logs
prepare_job=$(sbatch --parsable scripts/heston_corrected/ready_v2/baseline/prepare.sbatch)
screen_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" scripts/heston_corrected/ready_v2/baseline/screen.sbatch)
refine_job=$(sbatch --parsable --dependency=afterok:"$screen_job" scripts/heston_corrected/ready_v2/baseline/refine.sbatch)
validate_job=$(sbatch --parsable --dependency=afterok:"$refine_job" scripts/heston_corrected/ready_v2/baseline/validate.sbatch)
echo "Validation job: $validate_job. Inspect summarize before extensions or full-data refit."
