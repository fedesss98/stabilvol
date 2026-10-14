# Corrected MFHT calibration

The corrected workflow is additive: the paper/Fortran reproduction and legacy
calibration defaults are unchanged. The new entry point is
`scripts/calibrate_heston_corrected.py`; its settings are in
`configs/heston_corrected.json`.

## Measurement and simulation

`CorrectedConfig` and the resolved target types are exported alongside the old
calibration API. Corrected episodes clear all state after every completed
crossing, including rejected crossings. They record `accepted`, `too_short`,
`overlong`, `missing`, and `unfinished` separately. Only accepted completed
durations 2–300 enter MFHT. Local population SD excludes the initial observation
and includes the terminal one. An overlong active episode waits for its actual
terminal crossing before a new episode may begin. Missing observations break it.

`SimulationConfig` now supports `return_mode="accumulated_pre_reset"`,
`variance_scheme="cir_exact"`, stationary variance initialization, independent
random streams, variance dispersion, and active/stale quotation transitions.
The corrected workflow always uses independent shocks. Exact CIR transitions do
not make the coupled price integration exact; a numerical gate is mandatory.

Observed log returns sum internal pre-reset increments, retaining real crash
increments but excluding numerical restart jumps. `SimulationResult` exposes
`log_prices`, `prices`, `reset_counts`, `reset_affected`, and `stream_seeds`.
Prices start at 100 immediately before observation zero. Log prices are the
authoritative continuous history; exponentiated prices can underflow on long
crashing paths. Empirical availability masks are applied after simulation.
Exported NPZs retain both the continuous latent history and the availability mask.

The initial comparison keeps each prior fit's empirical sigma. No return is
silently clipped or imputed. Preparation reports extreme values and preserves
zeros. Empirical infinities fail preparation; any upstream data correction must
be explicit and create a new data fingerprint.

## Run locally, then on the cluster

Choose a fresh run directory when changing scientific settings or source code:

```powershell
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py smoke --markets UN LN --run-dir data/processed/heston_corrected/ready_v2
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py jobs --markets UN LN --run-dir data/processed/heston_corrected/ready_v2
```

The smoke command runs a small real-data objective and population evaluation and
exports diagnostic returns/prices. It is permanently labeled **not a calibration**.
Its timings measure the small probe, not a promised full-run duration. The jobs
command requires a successful benchmark from the current source version.

Generated launchers are in `scripts/heston_corrected/ready_v2/baseline/`.
Run directories should have distinct names so their launchers remain separate.
Results, manifests, and checkpoints stay in `data/processed/heston_corrected/ready_v2/`. Transfer the generated
files with the repository/data, preserving relative paths, then on the cluster:

```bash
export HESTON_REPO_ROOT=/data/qmla/famato/stabilvol
bash scripts/heston_corrected/ready_v2/baseline/submit.sh
```

The submission script chains preparation/numerical gates, three screening
restarts per market, up to four refinements, and validation. Arrays run at most
two tasks concurrently; each task requests eight CPUs, 64 GB, and 24 hours.
Search stops after 23 hours and returns exit code 75 if incomplete, preventing
dependent stages from consuming partial populations. Resubmit the chain to
resume. Completed compatible searches are reused. A batch already executing
finishes before the deadline check; the one-hour scheduling margin is deliberate.

For direct stage control:

```powershell
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py prepare --markets UN LN --run-dir data/processed/heston_corrected/ready_v2
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py gate --markets UN LN --run-dir data/processed/heston_corrected/ready_v2
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py screen --markets UN --restart 0 --workers 8 --run-dir data/processed/heston_corrected/ready_v2
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py resume --stage screen --markets UN --restart 0 --workers 8 --run-dir data/processed/heston_corrected/ready_v2
.venv/Scripts/python.exe scripts/calibrate_heston_corrected.py summarize --markets UN LN --run-dir data/processed/heston_corrected/ready_v2
```

Use screening restart indices 0–2. Refinement indices 0–3 correspond to distinct
screening finalists; unused slots exit successfully. Validation requires all
selected finalists to finish. `--source-fit` selects another prior-fit directory;
each market must contain its parameter CSV and trial config. `--config` selects
explicit corrected settings and is propagated into generated jobs.

## Numerical and scientific gates

Numerical checks use seeds 4001–4005, up to 256 sampled stock masks, fixed daily
observation duration, and 10, 20, 40, 80, 160 internal substeps. A comparison must
cover at least three common bins and 95% of the union of supported bins. Every
common bin and the stock-SD/IQR diagnostics must agree to 5% plus two Monte Carlo
standard errors. Selection never stops below 40 substeps. Burn-in comparisons
are 250/500, 500/1000, and 1000/2000 observations. Both converted prior and paper
parameters are available as initial gate references; fitted finalists are gated
again individually. Insufficient support is reported as a failed gate rather
than evidence of convergence. Cached gate evaluations can be reused unchanged.

Empirical stocks split deterministically 70/15/15. Target edges and sigma remain
fixed across training, selection, and held-out stocks. All selected stocks have
at least 3,030 observations. Missing fixed bins in a held-out target are reported
explicitly rather than silently rebinned. Stock-bootstrap uncertainty uses 300
resamples; five-year calendar-block comparisons are saved separately.

The primary objective is the specified region-balanced standardized log-MFHT
loss. Sparse synthetic bins receive a residual of 10 plus their proportional
count deficit. Feasibility precedes MFHT loss in the DE selection rule. Mean,
scale, quantile, tail, and autocorrelation constraints are reported separately;
the exact-zero atom is a diagnostic, not an impossible baseline constraint.

`best.json` retains the best feasible candidate, best unconstrained MFHT
candidate, and best feasibility-ranked candidate. Checkpoints also include the
entire population, result records, pending trial evaluations, RNG state,
evaluation count, and generation history. Serial and parallel evaluation order
does not alter the algorithm's generation updates.

Training uses seeds 1001–1004, selection 2001–2005, held-out evaluation 3001–3010;
optimizer seeds are separate. Screening uses two training seeds and 256 masks;
refinement uses four seeds and every training-stock mask. A near-boundary
refinement winner triggers one factor-two bound expansion. Remaining pressure
is saved. No optimizer convergence flag is treated as statistical acceptance.

Acceptance requires numerical stability, all return constraints, RMS log-MFHT
error <=0.10, >=95% synthetic bin coverage, acceptable peak height, and peak-region
overlap. Each failed criterion remains in JSON. Reports also include the three
specified unfitted crash/rally threshold pairs and zero-run summaries.

## Extensions and final refit

If the baseline fails **selection-stock** criteria, run the `heterogeneity` and
`activity` variants separately by adding `--variant` to jobs or stage commands.
This trigger deliberately does not use held-out stocks to choose model structure.
Heterogeneity adds one lognormal variance-dispersion parameter. Activity uses
training-stock zero/nonzero transition estimates and accumulates latent returns
until a quote becomes active. The combined variant is allowed only after both
individual extensions fail selection criteria.

After completing the warranted comparisons, run `refit` (or `resume --stage
refit`). It selects the feasible specification with lowest selection loss,
preferring the simpler model within one jackknife simulation standard error,
then refits the full ensemble. If none is feasible it preserves and labels the
best infeasible tradeoff. The full-data fit is gated and independently simulated
again; its result is never labeled held-out performance. Full refit is not
automatically submitted before checking the extension decision.

UW/JT preparation is blocked until UN/LN full-data results pass. Their source
fits must be explicitly available, for example through the earlier four-market
paper-sigma fit directory.

## Artifacts and provenance

Each market directory contains immutable preparation manifests, split bundles,
physical target bins, episode categories, calendar-block diagnostics, numerical
checks, stage checkpoints, and candidate/constraint reports. `summarize` writes
`report.md`, `summary.json`, and `model_comparison.png`, with separate panels for
the old fitting pipeline, corrected held-out baseline, and full-data selected fit.
The historical panel is recomputed using its recorded parameters, so copied stale
recounts are not reused. Original and corrected panels measure different statistics.

Exported `series-<seed>.npz` contains log returns, log prices, prices, masked
simple returns, availability, sampled stock-mask indices, and reset counts.
Empirical bands are stock-bootstrap intervals; synthetic bands show realization
spread, not confidence intervals for an infinite-simulation mean. Unsupported
synthetic bins remain visibly absent and still incur objective penalties.

Use `load_verified_comparison(export_dir, market_run_dir, repository_root)` in
notebooks. It checks source/data fingerprints, artifact hashes, and the currently
selected parameters. The new notebook 26 uses this guard; notebook 25 remains a
historical exploration workbook. Manually copying CSVs between runs cannot make
them current validated outputs.

Run the regression suite with:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_heston*.py"
```

Large cluster fits are still scientific experiments. Successful software tests
and smoke execution do not imply that Heston can meet every acceptance target.
