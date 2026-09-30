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
