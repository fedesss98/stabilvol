# Corrected pipeline verification — 2026-10-14

The implementation is tested, but no completed, scientifically accepted new
calibration is claimed. Full differential-evolution searches, finalist numerical
gates, independent-seed validation, and model selection require the cluster run.

## Completed checks

- All 60 Heston tests pass, including the historical reproduction suite and new
  counter, CIR, reset-return, target, provenance, checkpoint recovery, and
  serial/parallel regression checks.
- UN and LN real-data smoke runs completed with the current source fingerprint.
  Both exported synthetic returns, authoritative log prices, availability masks,
  episode counts, constraint reports, and diagnostic comparison figures.
- Smoke/export manifests were reread and verified against the current source.
  Smoke results are labeled `not_a_calibration` and fail scientific acceptance;
  these small probes are evidence of execution, not fitted-model quality.
- All six generated Bash/Slurm files pass `bash -n`.
- The new notebook's code parses, and `git diff --check` passes.
- Existing edits in notebook 25 were preserved; notebook 26 is the new guarded
  visualization entry point.
- Full empirical preparation and historical recounts completed for both markets,
  including 300 stock-bootstrap resamples and five-year block diagnostics.
  UN has 1,294 eligible stocks, 67 frozen target bins, and 244,499 retained events;
  LN has 1,189 eligible stocks, 66 frozen bins, and 159,540 retained events.
  These event totals include separately reported outer-volatility tails.
  No empirical outliers or zero returns were removed. LN preparation reports
  one simple return above 100; its influence is preserved and visible.

The completed preparation described above is archived in `data/processed/heston_corrected/ready_v1`.
Earlier development directories are retained and may have incompatible source
fingerprints; they must not be reused as current results.

The launcher-location update changes the recorded source fingerprint. New runs
use `ready_v2`; the earlier `ready_v1` manifests remain unchanged. Preparation
for the new run is performed by its first Slurm stage. New launchers live under
`scripts/heston_corrected/ready_v2/baseline/`, outside the data directory.

## Cluster launch

Copy the repository, input data, and generated jobs to the configured cluster
checkout, preserving paths and the source revision, then run:

```bash
export HESTON_REPO_ROOT=/data/qmla/famato/stabilvol
bash scripts/heston_corrected/ready_v2/baseline/submit.sh
```

There is no local Slurm scheduler, and no cluster jobs were submitted from this
session. Preparation is resumable. The submission chain performs numerical
gates before screening and stops when a stage fails or remains partial. Inspect
the validation report before choosing the predefined extensions or full-data
refit. See [the workflow guide](heston-corrected-workflow.md) for stage commands,
resume behavior, scientific acceptance criteria, and provenance rules.
