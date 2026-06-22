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

## Modified Heston Calibration

The old Fortran code uploaded in `simulation_tau_vs_noise/` has been ported into `stabilvol/heston/`.

The simulator implements:

```text
dU_dx = 3*a*x**2 + 2*b*x
x[t+1] = x[t] - dU_dx*dt - 0.5*V[t]*dt + sqrt(V[t]*dt)*Z_price
V[t+1] = V[t] + aa*(bb - V[t])*dt + cc*sqrt(V[t]*dt)*Z_vol
```

By default, parameters mirror `simulation_tau_vs_noise/parm.dat`. Negative variance proposals are redrawn up to 500 times, matching the active Fortran behavior. The calibration workflow uses uncorrelated price/variance shocks by default, matching the active old Fortran lines; in this mode `rho` is fixed from `base_params` and is not optimized.

Run a cheap pipeline smoke test with the default Fortran parameters:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/calibrate_heston.py \
  --config-json configs/heston_quick.json
```

Run a first parallel calibration on `UN`:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/calibrate_heston.py \
  --config-json configs/heston_un_parallel.json
```

Run calibration over the default four-market, four-threshold grid:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/calibrate_heston.py \
  --config-json configs/heston_default_grid.json
```

The JSON files under `configs/` are the recommended way to run the workflow. Command-line flags override the config for one run, so this is valid:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/calibrate_heston.py \
  --config-json configs/heston_un_parallel.json \
  --workers 8 \
  --maxiter 20
```

For optimizer multiprocessing, set `run.workers` in the config or pass `--workers N`; `--workers -1` uses all available cores. Start conservatively, because each worker holds simulated returns and empirical FHT data in memory.

The script writes fitted parameters and loss summaries to `data/processed/heston_calibration/`, and comparison plots to `visualization/heston_calibration/`. For each market it saves the diagnostic 2D empirical-vs-synthetic histogram/MFHT plots, a return-density overlay named `<MARKET>_returns_pdf.png`, and an FHT-density overlay named `<MARKET>_fht_pdf.png`. The CSV outputs also include empirical and synthetic return moments: mean, variance, skewness, and kurtosis.

Default calibration choices:

- empirical target: `data/processed/trapezoidal_selection/stabilvol_filtered.sqlite`;
- markets: `UN`, `UW`, `LN`, `JT`;
- thresholds: `(-0.5, -1.5)`, `(-1.0, -2.0)`, `(0.5, 1.5)`, `(1.0, 2.0)`;
- loss target: mean two-sample Kolmogorov-Smirnov statistic between empirical and simulated FHT distributions, averaged across configured threshold pairs and ignoring volatility;
- optimized parameters by default: `a`, `b`, `aa`, `bb`, `cc`, and `vstart`; `rho` is optimized only if `correlated_noise` is explicitly set to `true`;
- pilot optimization: `min(512, n_market_stocks)` paths and 3030 steps;
- full validation: market-sized path count and 11089 steps.

The old `calmG.f` counter is not the primary v1 API. It differs from the current calibration path in important ways: it counts with a global `sigma_tot`, uses start/end inequalities from `parmG.dat`, bins manually with `sigma_max / num_bin`, and accepts tau up to 300. The Python calibration instead reuses `StabilVolter`, applies its local event volatility, and uses `tau_max=30` to match the filtered empirical database and notebook 11 workflow.

## Setup

Use the project virtual environment if it already exists:

```bash
.venv/bin/python -m pip install -e .
```

Or create one:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

`requirements.txt` is an environment snapshot. On non-Windows systems, `pywin32` may need to be removed or skipped if installation fails.

Matplotlib may try to write cache files outside the sandbox/user-writable area. If needed, run commands with:

```bash
MPLCONFIGDIR=/tmp
```

## Current Caveats

The codebase still has a few legacy-script rough edges:

- `general_stabilvol_counter.py` currently fails at import time because it imports `ROOT` from `utility.definitions`, where `ROOT` is not defined. It also imports `datetime` as a module but calls `datetime.now()`, and `save_to_database` names tables using module constants instead of parsed threshold arguments.
- `general_counter_iterator.py` is import-path sensitive. In this workspace, `.venv/bin/python stabilvol/general_counter_iterator.py --help` works, while `python3 stabilvol/general_counter_iterator.py --help` and `python3 -m stabilvol.general_counter_iterator --help` fail for different import-path reasons.
- `general_counter_iterator.py --levels` is parsed as a flat list of floats, but `main()` expects an iterable of `(start_level, end_level)` pairs.
- Successful market counts in `general_counter_iterator.py` are appended twice to `stabilvols`.
- `stabilvol/__init__.py` computes `ROOT` from the current working directory, so path behavior can change depending on where a script is launched.

These notes reflect the current repository state and should be revisited when the scripts are cleaned up.
