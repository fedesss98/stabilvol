# StabilVol

StabilVol is a research codebase for studying stabilizing effects of volatility in financial markets through first hitting times (FHT) and mean first hitting time (MFHT) curves.

The project works mostly with Bloomberg market return data stored as pandas pickles, counts FHT events for threshold pairs, stores those events in SQLite databases, and then bins/plots MFHT as a function of volatility.

## Repository Map

- `stabilvol/utility/classes/stability_analysis.py`: core FHT logic. `StabilVolter` counts threshold-crossing events and returns rows with `Volatility`, `FHT`, `start`, `end`, plus optional metadata such as `Market`.
- `stabilvol/utility/classes/data_extraction.py`: date, stock-coverage, and volatility filtering for market return DataFrames.
- `stabilvol/general_stabilvol_counter.py`: legacy script for one threshold pair.
- `stabilvol/general_counter_iterator.py`: legacy script for sweeping threshold pairs and writing one SQLite table per pair.
- `stabilvol/bin_mfht.py`: bins FHT rows from SQLite into MFHT pickle files.
- `stabilvol/heston/`: Python port of the modified Heston simulator and calibration tools.
- `scripts/reproduce_heston_paper.py`: fixed-parameter theoretical simulation and diagnostics for Valenti et al. (2018), without an empirical database.
- `scripts/reproduce_heston_paper.sbatch`: Slurm job for that theoretical simulation; array indices can run independent seeds.
- `scripts/summarize_heston_paper.py`: combine completed seed-array runs into mean MFHT curves and a seed summary.
- `scripts/calibrate_heston.py`: CLI for fitting Heston parameters to empirical FHT distributions.
- `notebooks/11-mfht-grid.ipynb`: exploratory MFHT grid notebook. It loads/bins FHT data, caches MFHT pickles, and builds grid, peak, and resistance-band plots.
- `notebooks/20-filter-fht.ipynb`: filters the main FHT database into `stabilvol_filtered.sqlite`, which notebook 11 currently uses.

## Data Flow

1. Market return pickles live in `data/interim/`, for example `UN.pickle`, `UW.pickle`, `LN.pickle`, `JT.pickle`, and log-return variants such as `UN_log.pickle`.
2. FHT counting writes SQLite tables named `stabilvol_<theta_i>_<theta_f>`, where negative signs become `m` and decimal points become `p`. Example: `stabilvol_m2p0_0p0`.
3. Main FHT databases are stored under `data/processed/<selection>_selection/`, especially:
   - `stabilvol.sqlite`: standard FHT counts.
   - `stabilvol_logs.sqlite`: log-return FHT counts.
   - `stabilvol_filtered.sqlite`: filtered FHT counts, used by notebook 11 and `bin_mfht.py`.
4. MFHT pickle caches are stored under paths produced by `format_mfht_directory`, for example `data/processed/trapezoidal_selection/vol100/`.
5. Figures are written under `visualization/mfhts/`.

## Threshold Counters

### `general_stabilvol_counter.py`

This is intended for a single threshold pair over one or more markets.

Defaults in the current file:

- markets: `["UN"]`
- date range: `1980-01-01` to `2022-07-01`
- stock selection: `percentage` with value `0.05`, therefore `trapezoidal_selection`
- input returns: `data/interim/<market>.pickle`
- thresholds: `START_LEVEL = -2.0`, `END_LEVEL = 0.0`
- tau range: `tau_min = 2`, `tau_max = 100`
- output: `data/processed/trapezoidal_selection/stabilvol.sqlite`

Conceptually, this is the quick counter for "count this one threshold pair now".

### `general_counter_iterator.py`

This is intended for threshold-grid runs. It loops over `LEVELS`, skips threshold tables already present in the output database, counts all selected markets for each remaining pair, and saves each pair to its own table.

Defaults in the current file:

- markets: `["UN", "UW", "LN", "JT"]`
- date range: `1980-01-01` to `2022-07-01`
- stock selection: `percentage` with value `0.05`, therefore `trapezoidal_selection`
- counting method: `multi`
- input returns: `data/interim/<market>_log.pickle`
- standard-deviation normalization: `False`
- tau range: nominally `tau_max = 30`
- output: `data/processed/trapezoidal_selection/stabilvol_logs.sqlite`

Conceptually, this is the batch counter for "populate the threshold database".

Current important detail: the active default `START_LEVELS = [-0.001, -0.002, 0.001, 0.002]` is rounded to two decimals when `LEVELS` is built, so the default set collapses to `{(-0.0, 0.0)}`. If you want a real grid, change `START_LEVELS`/`DELTAS` or rebuild `LEVELS` before running.

## Notebook 11: MFHT Grid

`notebooks/11-mfht-grid.ipynb` is the current workbook for threshold-grid MFHT inspection.

It currently:

- reads `../data/processed/trapezoidal_selection/stabilvol_filtered.sqlite`;
- lists available threshold tables with `list_database_thresholds`;
- queries FHT rows by market, threshold pair, date range, `VOL_LIMIT`, and `TAU_MAX`;
- bins FHT data into MFHT curves using `query_binned_data` from `stabilvol.utility.functions` or the notebook-local `optimized_binning`;
- caches files named `mfht_<market>_<theta_i>_<theta_f>.pkl`;
- plots MFHT grids for crash/rally threshold families;
- plots peak-MFHT comparison heatmaps and resistance-band figures.

The notebook is exploratory and contains some stale cells/output. In the current saved state, one cell errors because table `stabilvol_0p5_0p4` is missing, and another refers to undefined `log_binning`; the maintained notebook-local binning helper is `optimized_binning`.

## Reproducing the Paper's Theoretical Model

Valenti, Fazio, and Spagnolo, [*Stabilizing effect of volatility in financial markets*](https://doi.org/10.1103/PhysRevE.97.062307), compare empirical curves with a **fixed-parameter simulation** of their nonlinear Heston model. That is a separate task from fitting model parameters to this repository's four-market SQLite database.

The paper uses Eqs. (5) and (6), with `U(x) = 2x^3 + 3x^2`, independent price/variance shocks, variance parameters `a=2`, `b=0.01`, `c=0.83`, `vstart=8.62e-5`, and `x0=0`. It generates 1071 return series of 3030 steps. Crash thresholds are `(-0.1, -1.5)` and rally thresholds `(0.1, 1.5)`, each multiplied by the mean of the per-series return standard deviations. The key outputs are theoretical MFHT-versus-local-volatility curves (Figs. 2(a) and 3(a)), followed by FHT, return, event-volatility, and autocorrelation diagnostics (Figs. 4(b), 5, and 6).

Run this without any SQLite database:

```bash
uv run python scripts/reproduce_heston_paper.py
```

Results go to `data/processed/heston_paper/`: figure PNGs, binned MFHT CSVs, event CSVs, autocorrelation CSV, and `summary.json`. This runs the paper's **fixed parameters**, without optimization or an empirical SQLite database. The default `--method fortran` ports the uploaded `old_code/heston.f`, `old_code/calmG.f`, `parm.dat`, and `parmG.dat` numerical conventions: `dt=0.01`, reset below `x=-6`, redraw negative variance proposals, zero the first saved return, use population standard deviations for normalization and local volatility, count FHTs from 2 through 300 steps, and compute event volatility from returns after the start through the terminal crossing. The Fortran counter carries its sums and duration across crossings discarded by the FHT range filter; the port preserves that behavior. The supplied `calmG.f` handles crashes; rally counting applies the same rules to sign-reversed returns, an inference because no rally-specific original counter was supplied.

**Volatility-axis issue in the original Fortran:** `parmG.dat` gives `delta = sigma_max/num_bin = 0.2/5000 = 0.00004`; `calmG.f` assigns events to bin `i` using this width but writes `i*delta/2` as the x coordinate. Thus the published-style MFHT axis is approximately half the physical event volatility. The default script saves both `fig2a_fig3a_theoretical_mfht.png` on the original reported axis and `mfht_corrected_volatility_axis.png` on a physical axis. Each MFHT CSV contains the reported coordinate and the true lower/upper bin limits. `--method clean` uses the earlier repository counter and a physically labeled axis; `--bins` and `--include-end-in-volatility` apply only to that method.

For Slurm on the configured cluster, run `uv sync` from `/data/qmla/famato/stabilvol` before submitting:

```bash
sbatch scripts/reproduce_heston_paper.sbatch
```

The job defaults to one 1071-by-3030 realization and writes under `data/processed/heston_paper/<job ID>/seed-<seed>/`. To assess variation across seeds, submit `sbatch --array=0-9%5 scripts/reproduce_heston_paper.sbatch`; the seed is 12345 plus the array index. After all tasks finish, run `uv run python scripts/summarize_heston_paper.py data/processed/heston_paper/<job ID>` to save cross-seed MFHT curves and parameter summaries. Each task requests one CPU, 8 GB, and one hour. Its repository and output paths are set in the batch file; Slurm output goes to `logs/heston-paper_<job ID>_<array index>.out/.err`.

**Current comparison:** seed 12345 gives mean per-series return standard deviation `0.023805`, close to the paper's `0.02383`. The Fortran-style crash and rally MFHT peaks are at reported coordinates `0.00584` and `0.00638`, with heights `94.2` and `92.4` steps, close to the peaks shown around `0.006` and 90–100 steps in Figs. 2(a) and 3(a). Their physical bin centers are `0.01166` and `0.01274`. The pooled return standard deviation is `0.02406` versus the paper's `0.024`, while skewness and kurtosis for this seed are `-1.06` and `67.2` versus the paper's `-1.96` and `105`; those higher moments remain sensitive to random realization and the original custom Fortran RNG. The Python and Fortran generators are therefore not expected to produce identical paths. The MFHT peak agreement is a comparison of the analysis convention, not a bit-for-bit replay.

## Modified Heston Calibration

The original Fortran sources and parameter files supplied by the user are in `old_code/`. The modified Heston simulator is in `stabilvol/heston/`; the paper-specific Fortran event counter is ported in `stabilvol/heston/paper_reproduction.py`.

The simulator implements:

```text
dU_dx = 3*a*x**2 + 2*b*x
x[t+1] = x[t] - dU_dx*dt - 0.5*V[t]*dt + sqrt(V[t]*dt)*Z_price
V[t+1] = V[t] + aa*(bb - V[t])*dt + cc*sqrt(V[t]*dt)*Z_vol
```

By default, parameters mirror `simulation_tau_vs_noise/parm.dat`. Negative variance proposals are redrawn up to 500 times, matching the active Fortran behavior. The calibration workflow uses uncorrelated price/variance shocks by default, matching the active old Fortran lines; in this mode `rho` is fixed from `base_params` and is not optimized.

Run these commands from the repository root with `uv`. The old `MPLCONFIGDIR=/tmp` prefix only selected Matplotlib's writable cache directory; it did not change the working directory, and `/tmp` is not required. The examples below use Bash line continuations; in PowerShell, put each command on one line. First provide `data/processed/trapezoidal_selection/stabilvol_filtered.sqlite` (or pass `--database /path/to/your.sqlite`). This processed database is not included in the repository.

Run a cheap pipeline smoke test with the default Fortran parameters:

```bash
uv run python scripts/calibrate_heston.py \
  --config-json configs/heston_quick.json
```

Run a first parallel calibration on `UN`:

```bash
uv run python scripts/calibrate_heston.py \
  --config-json configs/heston_mfht_pilot.json
```

Run calibration over the default four-market, four-threshold grid:

```bash
uv run python scripts/calibrate_heston.py \
  --config-json configs/heston_default_grid.json
```

The JSON files under `configs/` are the recommended way to run the workflow. Command-line flags override the config for one run, so this is valid:

```bash
uv run python scripts/calibrate_heston.py \
  --config-json configs/heston_mfht_pilot.json \
  --workers 8 \
  --maxiter 20
```

For optimizer multiprocessing, set `run.workers` in the config or pass `--workers N`; `--workers -1` uses all available cores. Empirical FHT data is loaded once per worker rather than sent with every candidate, but each worker still needs memory for its own simulation and data. Start conservatively.

`base_params` sets fixed simulation parameters and the evaluate-default point. If `initial_params` is supplied, its optimized coordinates replace one member of the differential-evolution starting population; it does not constrain the fit.

### Staged return and single-threshold MFHT calibration

The model evolves one log-price displacement path per simulated stock,
`x(t) = log[p(t)/p(0)]`, and records its daily log increment
`x(t) - x(t-1)`. This approximates the simple daily returns in
`data/interim/UN.pickle` when price changes are small. The original model
restarts a path when it crosses `reset_threshold`; the crossing increment is
recorded before the restart, while the optional stored `x` is the state after
the restart.

For the staged UN experiment, run:

```bash
uv run python scripts/calibrate_heston.py --config-json configs/heston_staged_un.json
```

The first optimizer fits the average of the per-stock daily means and the
average of the per-stock daily standard deviations. Its loss is the squared
mean error in units of `0.1 * empirical_average_std`, plus the squared log
ratio of synthetic to empirical average standard deviations. The default
model parameters are included in the initial population. Six parameters
cannot be uniquely identified by two return statistics, so this is a scale
and location fit, not a complete return-PDF fit.

The staged config retains stocks with at least 3030 finite daily returns,
matching the pilot simulation length as a minimum observation requirement.
It does not recreate the paper's exact 1987-1998 stock sample. Set
`min_empirical_observations` to zero to use every stock with a valid mean and
standard deviation.

The second optimizer starts with the first-stage parameters and fits only
the crash MFHT curve for `(-0.1, -1.5)`. It adds the configured
`return_loss_weight` times the return loss to keep the daily returns close
to their target. The collective empirical scale is the standard deviation
of all finite returns from the selected stocks and dates. Both empirical and
simulated events use the same fixed thresholds, `-0.1 * collective_sigma` and
`-1.5 * collective_sigma`. Empirical events are counted directly from
`UN.pickle`, so this run does not need the previously processed SQLite
database. The original paper computed a separate collective scale for its
simulated paths; this experiment deliberately fixes the empirical scale on
both sides.

Stage-one parameters and losses are saved in
`data/processed/heston_calibration/staged_un/heston_calibration_stage1_parameters*.csv`.
Its independent-seed return comparison is
`visualization/heston_calibration/staged_un/UN/UN_stage1_returns_pdf.png`.
Return-PDF comparison plots use 240 equal-width bins by default; this affects
only the display and does not change the calibration loss.
The usual parameter and output CSVs contain the second-stage fit, its fixed
threshold scale, and return and MFHT diagnostic losses from an independent
plotting seed. The plots and MFHT comparison CSV go to
`visualization/heston_calibration/staged_un/UN/`. Adjust `run.return_maxiter`
and `run.maxiter` separately in the JSON or with `--return-maxiter` and
`--maxiter`.

To refine the saved UN parameters with all 1294 eligible stocks in each
pilot simulation, use `configs/heston_un_full_ensemble_refine.json`. It seeds
the optimizer with the saved staged fit, uses the pooled empirical standard
deviation for the fixed thresholds, and writes to separate output folders:

```bash
uv run python scripts/calibrate_heston.py --config-json configs/heston_un_full_ensemble_refine.json
```

The earlier `20261001_134240` staged run used the average per-stock standard
deviation as its threshold scale. Its saved losses therefore are not directly
comparable to losses from the collective-scale refinement.

### Sampling interval sweep

Use `scripts/sweep_heston_sampling.py` to keep the fitted parameters and Euler
step `dt` fixed while observing each simulated `x` path every 1, 2, 5, or 10
internal steps. For a first check with the saved staged parameters:

```bash
uv run python scripts/sweep_heston_sampling.py --parameters data/processed/heston_calibration/staged_un/heston_calibration_parameters.csv --paths 64 --days 3030 --intervals 1 2 5 10
```

For the full-ensemble fit, point `--parameters` to its parameter CSV on the
workstation. Each interval produces 3030 observed returns per path, using the
same random trajectory and a 1000-step burn-in. The CSV reports return moments,
the fraction of returns inside fixed windows around zero (also conditional on
nonzero returns), MFHT loss, and the existing combined loss. Return-PDF plots
are saved alongside the CSV in `data/processed/heston_calibration/sampling_sweep/`.
This is a diagnostic sweep, not a parameter refit. It uses differences of the
sampled post-reset `x` states; any interval containing a reset includes its
jump. The script prints the reset count so this convention is visible.

To **refit the model separately** for each observation interval, run:

```bash
uv run python scripts/calibrate_heston_sampling_sweep.py --config-json configs/heston_un_full_ensemble_refine.json --intervals 1 2 5 10
```

The runner writes four interval-specific configs and calls the normal
calibrator once per interval. In each objective evaluation, it evolves `x`
for `k` Euler steps per observed day, takes endpoint differences of the
sampled `x`, and computes the same return-moment and MFHT losses. It uses a
1000-step burn-in for every interval, the same empirical targets and parameter
bounds, and an independent full-length validation seed. The return-PDF plots
and parameter CSVs are separated under `interval_1`, `interval_2`, `interval_5`,
and `interval_10`. The aggregate parameter/loss table is
`data/processed/heston_calibration/sampling_fit_sweep/UN_sampling_fit_sweep.csv`.
Its columns include the empirical and simulated fractions of daily returns
within `±0.0025`, with an additional fraction conditional on nonzero returns.
This near-zero fraction is a diagnostic and is not part of the fit loss.

For a quick pipeline check before the costly full sweep, run:

```bash
uv run python scripts/calibrate_heston_sampling_sweep.py --intervals 1 --pilot-paths 8 --pilot-steps 80 --maxiter 0 --popsize 1 --workers 1 --skip-full-validation
```

To seed every interval from a newer fit, pass
`--initial-parameters PATH_TO_PARAMETERS_CSV`. The complete
four-interval sweep can take many hours, especially at `10dt`, because each
candidate then uses about ten times as many Euler steps as the `1dt` fit.

To save the synthetic time series from completed sweep fits without repeating
optimization, run:

```bash
uv run python scripts/export_heston_sampling_series.py --kind plot
```

This reproduces the independently seeded series used for each return-PDF plot
and writes `UN_synthetic_returns_plot.pkl` inside each interval's processed
output folder. Use `--kind validation` for the longer full-validation series,
or `--states` to also save the sampled log-price state `x` and variance arrays.
The exporter verifies reproduced plot return moments against the saved result.
Open the pickles with `pandas.read_pickle`; the calibration's original
empirical returns remain in `data/interim/UN.pickle`. The analysis notebook can
redraw the return PDFs with a different bin count after the pickles are copied.

### Four-market sampling fits and final ensembles

On the workstation, fit each of `UN`, `UW`, `LN`, and `JT` separately at four
sampling intervals:

```bash
uv run python scripts/calibrate_heston_four_market_sweep.py --markets UN UW LN JT --intervals 1 2 5 10 --workers 8
```

The driver derives one threshold sigma and one full selected-stock path count
per market from `data/interim/<MARKET>.pickle` using the 3030-observation rule.
To prevent rare extreme returns from making the thresholds unusable, it clips
the pooled selected returns to their market-specific 0.1% and 99.9% quantiles
when calculating sigma. The original returns remain unchanged for fitting,
event counting, and PDF comparison. The clipping bounds and sigma are saved in
each market config.
It applies the same `-0.1 sigma -> -1.5 sigma` threshold pair, retains the
template's optimizer settings and 11089-day full validation, and writes each
market's configs and parameter CSVs under
`data/processed/heston_calibration/four_market_sampling_sweep/<MARKET>/`.
Its MFHT and return-PDF figures go under
`visualization/heston_calibration/four_market_sampling_sweep/<MARKET>/`.
The other markets do not inherit UN's starting parameter vector. The fits run
sequentially and may take many hours, particularly at `10dt`.

For a cheap pipeline check, run the same driver with `--intervals 1 --workers 1
--maxiter 0 --popsize 1 --pilot-paths 8 --pilot-steps 80 --full-steps 80
--skip-full-validation` and separate `--output-dir` / `--figure-dir` paths.

Copy the two `four_market_sampling_sweep` directories to the same locations
locally, then open `notebooks/25-heston-sampling-calibration.ipynb`. Review
validation losses, return PDFs, and MFHT bin coverage. In its selection cell,
enter one sampling interval for each market. The notebook writes
`data/processed/heston_calibration/four_market_sampling_sweep/final_selection.json`;
copy that small file back to the same workstation directory.

On the workstation, run 10 independent full-length ensembles for the four
selected fits and retain all return matrices:

```bash
uv run python scripts/heston_sampling_replicates.py --selection data/processed/heston_calibration/four_market_sampling_sweep/final_selection.json --replicates 10 --days 11089 --save-series
```

Before that full run, use `--replicates 2 --days 80 --paths 8 --output-dir
data/processed/heston_calibration/four_market_replicate_smoke` to test the
four-market data flow. Final outputs are under `replicate_variation`: one
`replicate_metrics.csv` row per simulation, a `replicate_summary.csv` with
across-run mean and sample standard deviation, and per-market PDF/FHT/MFHT
bands as CSV and PNG. The return-PDF total-variation score includes tail bins;
the Wasserstein score is calculated on reproducible samples of up to 200000
raw returns per distribution to bound memory and runtime. Both scores are
lower when distributions agree more closely. The plotted PDF range is fixed
from the empirical 0.1% to 99.9% quantiles and normalized within that range.
Copy the summary CSVs and band CSVs/PNGs locally for the notebook. Copy the
large `*_returns_replicate_*.pkl` files only when rebinned analysis is needed.

The `plot` series uses a separate seed and the plot path/day counts (usually
the pilot counts); it is the realization shown in the calibration figures.
The `validation` series uses another independent seed, the full selected-stock
count, and `full_n_steps`; it is the realization scored by `validation_loss`.
Both use the fitted parameters. A single plot or validation run gives no
estimate of simulation-to-simulation variation.

To measure that variation at fixed fitted parameters, simulate `M` independent
ensembles for each completed interval:

```bash
uv run python scripts/heston_sampling_replicates.py --sweep-dir data/processed/heston_calibration/sampling_fit_sweep --market UN --replicates 10
```

By default, each ensemble has one path per selected empirical stock (1294 for
the current UN selection) and `pilot_n_steps` observed days (3030 in the full
config). Use `--days 11089` for validation-length runs. The output under
`data/processed/heston_calibration/sampling_fit_sweep/replicate_variation/`
contains per-run metrics, their across-run mean and sample standard deviation,
and CSV/PNG mean ± standard-deviation bands for the return PDF, FHT PDF, and
MFHT curve. The return PDF uses fixed empirical bin edges across runs, so its
band is comparable. Add `--save-series` to retain each simulated return matrix
for further binning experiments. These bands describe Monte Carlo variation
with the fitted parameters held fixed; they do not include fit or empirical
sampling uncertainty. For each interval `k`, runtime grows approximately with
`M × N × days × k`.

### Slurm cluster run

The cluster batch files set `repo_root=/data/qmla/famato/stabilvol` and use fixed paths below it. Stage this working tree, the SQLite file at `data/processed/trapezoidal_selection/stabilvol_filtered.sqlite`, and the market pickles under `data/interim/`. The smoke and pilot jobs need `UN.pickle`; the full four-market job also needs `UW.pickle`, `LN.pickle`, and `JT.pickle`. Data and the newly added batch files are not automatically available from an older Git checkout.

Dependencies are declared in `pyproject.toml`. On the cluster, prepare the project environment once; subsequent jobs use `uv run python`:

```bash
cd /data/qmla/famato/stabilvol
uv sync
sbatch scripts/calibrate_heston_smoke.sbatch
```

The one-CPU smoke job evaluates the default parameters on a small UN sample, checks that its empirical table is readable, and writes under `data/processed/heston_calibration/smoke-<job ID>/`. A successful smoke test checks the pipeline, not the quality of a fitted model.

For the first **MFHT-curve optimization**, submit the separate UN pilot after the smoke job succeeds:

```bash
sbatch scripts/calibrate_heston_pilot.sbatch
```

The pilot uses 128 simulated paths, 3030 steps, one threshold pair, five differential-evolution generations, and eight workers. It skips full validation. Review `visualization/heston_calibration/<job ID>/UN/UN_m0p5_m1p5_mfht.csv`, the corresponding plot, pilot loss, and event counts before running the full grid.

For the four-market, four-threshold run, submit the [full array job](scripts/calibrate_heston.sbatch):

```bash
sbatch scripts/calibrate_heston.sbatch
```

Array tasks `0=UN`, `1=UW`, `2=LN`, `3=JT` each use eight CPUs and 64 GB for up to 24 hours, with at most two tasks concurrent. The batch files target the `master` partition on `treachery`, request one task per job, and use the tracked `logs/` directory for Slurm output. Submit from the repository root so these relative log paths resolve there. The batch files set the config, database, output, and figure paths directly and run the optimizer with `$SLURM_CPUS_PER_TASK` workers. They do not require MPI or a GPU. The calibration loader opens SQLite read-only and reports missing database files or tables rather than creating them. Full-job logs are `logs/heston-calibration_<job ID>_<array index>.out/.err`.

### Interpretation

The default optimizer now fits the **MFHT-versus-local-volatility curves** directly. For each market and threshold pair, it fixes 40 equally spaced volatility bins from zero through the empirical 99.5th percentile, then compares mean FHT in bins with enough empirical events. The loss is the root mean squared curve difference, scaled by the empirical MFHT peak; missing simulated bins receive a penalty, and `event_count_weight` adds a small penalty for a mismatch in the volatility-bin event distribution. The bin edges are derived once from empirical data and reused for every candidate and validation run. `min_empirical_bin_events` (default 20) and `min_simulated_bin_events` (default 5) control which bins are trusted. Each fitted comparison saves a `*_mfht.csv` with bin edges, MFHTs, counts, and coverage flags alongside its plot.

The former KS FHT-distribution objective is still available with `--loss-metric fht_distribution` or `"loss_metric": "fht_distribution"` in a config. Neither objective alone establishes that a fit generalizes; compare independent-seed curves and event counts. Keep threshold normalization consistent with the empirical database: the quick/default-grid configs use `std_normalization=true`, while `heston_un_parallel.json` uses `false`.

The script writes the effective run configuration, fitted parameters, and loss summaries to `data/processed/heston_calibration/`, and comparison plots to `visualization/heston_calibration/`. For each market it saves the diagnostic 2D empirical-vs-synthetic histogram/MFHT plots, a return-density overlay named `<MARKET>_returns_pdf.png`, and an FHT-density overlay named `<MARKET>_fht_pdf.png`. The CSV outputs also include empirical and synthetic return moments: mean, variance, skewness, and kurtosis.

Default calibration choices:

- empirical target: `data/processed/trapezoidal_selection/stabilvol_filtered.sqlite`;
- markets: `UN`, `UW`, `LN`, `JT`;
- thresholds: `(-0.5, -1.5)`, `(-1.0, -2.0)`, `(0.5, 1.5)`, `(1.0, 2.0)`;
- simulated FHT threshold normalization: `std_normalization = true`; set `"std_normalization": false` in the JSON config or pass `--no-std-normalization` to use raw thresholds;
- loss target: MFHT-versus-volatility curve discrepancy, averaged across configured threshold pairs, with an event-distribution penalty; the former FHT KS score remains optional;
- optimized parameters by default: `a`, `b`, `aa`, `bb`, `cc`, and `vstart`; `rho` is optimized only if `correlated_noise` is explicitly set to `true`;
- pilot optimization: `min(512, n_market_stocks)` paths and 3030 steps;
- full validation: market-sized path count and 11089 steps.

The old `calmG.f` counter is not the primary v1 API. It differs from the current calibration path in important ways: it counts with a global `sigma_tot`, uses start/end inequalities from `parmG.dat`, bins manually with `sigma_max / num_bin`, and accepts tau up to 300. The Python calibration instead reuses `StabilVolter`, applies its local event volatility, and uses `tau_max=30` to match the filtered empirical database and notebook 11 workflow.

## Setup

The Heston dependencies are declared in `pyproject.toml`. From the project root, run:

```bash
uv sync
uv run python scripts/calibrate_heston.py --help
```

To import or update dependencies from the portable requirements file in the future, use:

```bash
uv add -r requirements-heston.txt
```

`uv add -r` updates `pyproject.toml` and the lockfile; it is not needed for this checkout because those six dependencies are already present in `pyproject.toml`. `requirements.txt` is a Windows environment snapshot containing `pywin32` and is unsuitable for the Linux cluster.

## Current Caveats

The codebase still has a few legacy-script rough edges:

- `general_stabilvol_counter.py` currently fails at import time because it imports `ROOT` from `utility.definitions`, where `ROOT` is not defined. It also imports `datetime` as a module but calls `datetime.now()`, and `save_to_database` names tables using module constants instead of parsed threshold arguments.
- `general_counter_iterator.py` is import-path sensitive. In this workspace, `.venv/bin/python stabilvol/general_counter_iterator.py --help` works, while `python3 stabilvol/general_counter_iterator.py --help` and `python3 -m stabilvol.general_counter_iterator --help` fail for different import-path reasons.
- `general_counter_iterator.py --levels` is parsed as a flat list of floats, but `main()` expects an iterable of `(start_level, end_level)` pairs.
- Successful market counts in `general_counter_iterator.py` are appended twice to `stabilvols`.
- `stabilvol/__init__.py` computes `ROOT` from the current working directory, so path behavior can change depending on where a script is launched.

These notes reflect the current repository state and should be revisited when the scripts are cleaned up.
