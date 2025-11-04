"""
Experiment 1:
START_LEVELS = [1.8, 1.4, 1., 0.6]
DELTAS = [-0.2, -0.4, -0.8]
DAYS_APART = 27
______________________________
Experiment 2:
START_LEVELS = [-1.8, -1.4, -1., -0.6]
DELTAS = [0.2, 0.4, 0.8]
DAYS_APART = 27
______________________________
Experiment 3:
START_LEVELS = [1.8, 1.4, 1., 0.6]
DELTAS = [-0.2, -0.4, -0.8]
DAYS_APART = 90
______________________________
Experiment 4:
START_LEVELS = [-1.8, -1.4, -1., -0.6]
DELTAS = [0.2, 0.4, 0.8]
DAYS_APART = 90
______________________________
Experiment 5:
START_LEVELS = [1.8, 1.4, 1., 0.6]
DELTAS = [-0.2, -0.4, -0.8]
DAYS_APART = 120
______________________________
Experiment 6:
START_LEVELS = [-1.8, -1.4, -1., -0.6]
DELTAS = [0.2, 0.4, 0.8]
DAYS_APART = 120
______________________________
Experiment 7:
START_LEVELS = [1.8, 1.4, 1., 0.6]
DELTAS = [-0.2, -0.4, -0.8]
DAYS_APART = 10
______________________________
Experiment 8:
START_LEVELS = [-1.8, -1.4, -1., -0.6]
DELTAS = [0.2, 0.4, 0.8]
DAYS_APART = 10
______________________________
"""

import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
import sqlite3
from scipy.stats import ks_2samp
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

from stabilvol.utility import functions as f

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATABASE = ROOT_DIR / 'data/processed/trapezoidal_selection/stabilvol_filtered.sqlite'

MARKETS = ["UN", "UW", "LN"]

EXPERIMENT = 3
START_LEVELS = [1.8, 1.4, 1., 0.6]
DELTAS = [-0.2, -0.4, -0.8]
DAYS_APART = 90
LEVELS = {
    (round(start, 2), round(start+delta, 2)) for start in START_LEVELS for delta in DELTAS
}
LEVELS = sorted(LEVELS)
WINDOWS_DURATION = 90  # days

VOL_LIMIT= 100  # Change this will change all the pickle files, remember to re-generate them
TAU_MAX = 30

MIN_BINS = -1  # This will return FHT values without binning if < 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--days-apart', type=int, default=DAYS_APART, help="Next window distance to compare PDF of each rolling window")
    return parser.parse_args()


# Optimization: Use a larger cache size and WAL mode for better performance
# Suggested by Claude AI
def setup_optimized_connection(database_path):
    """Setup an optimized SQLite connection"""
    conn = sqlite3.connect(database_path)
    # SQLite optimizations
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL") 
    conn.execute("PRAGMA cache_size=100000")  # Increase cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def make_pdf(data):
    pass


def test_ks(df1, df2):
    test_results = ks_2samp(df1, df2)
    return test_results.pvalue, test_results.statistic


def main():
    args = parse_arguments()

    days_apart = args.days_apart if DAYS_APART is None else DAYS_APART

    print(os.getcwd())
    if not os.path.exists(DATABASE):
        raise FileNotFoundError(f"Database not found at {DATABASE}")
    
    # Connect to the SQLite database
    conn = setup_optimized_connection(DATABASE)

    # Generate rolling windows
    windows = f.roll_windows(duration=WINDOWS_DURATION, start_date=datetime.date(1985, 1, 1), end_date=datetime.date(2022, 12, 31))
    print(f"Selected {len(windows)} rolling windows with {days_apart}-days shifts for the KS comparison.")
    
    # Allocate lists of pvalues and statistic values for every market
    pvalues = [np.zeros((len(LEVELS), len(windows))) for _ in range(len(MARKETS))]
    statistics = [np.zeros((len(LEVELS), len(windows))) for _ in range(len(MARKETS))]
    outcasts = [np.zeros((len(LEVELS), len(windows))) for _ in range(len(MARKETS))]
    coefficients = [(f.stringify_threshold(t1), f.stringify_threshold(t2)) for t1, t2 in LEVELS]
    for i, market in enumerate(MARKETS):
        for j, (t1, t2) in enumerate(tqdm(coefficients, leave=True, desc=f"Processing market {market}")):
            for w, window in enumerate(tqdm(windows, desc=f" - thresholds {t1}-{t2}", leave=False)):
                start, end = window
                next_start, next_end = start + pd.to_timedelta(days_apart, 'D'), end + pd.to_timedelta(days_apart, 'D')
                try:
                    fht, _ = f.query_binned_data(
                        market, start, end, VOL_LIMIT, TAU_MAX, t1, t2, conn=conn,
                        min_bins=MIN_BINS
                    )

                    fht_next, _ = f.query_binned_data(
                        market, next_start, next_end, VOL_LIMIT, TAU_MAX, t1, t2, conn=conn,
                        min_bins=MIN_BINS
                    )
                except ValueError:
                    outcasts[i][j, w] = 1
                else:
                        # Perform the Kolmogorov-Smirnov Test and take pvalue and statistic results
                    if not fht.empty and not fht_next.empty:
                        pvalue, statistic = test_ks(fht["FHT"], fht_next["FHT"])
                        pvalues[i][j, w] = pvalue
                        statistics[i][j, w] = statistic
                    else:
                        outcasts[i][j, w] = 1
        np.save(ROOT_DIR / f'data/processed/dynamics/{market}_rolling_{WINDOWS_DURATION}d_KS_pvalue_{EXPERIMENT}.npy', pvalues[i])
        np.save(ROOT_DIR / f'data/processed/dynamics/{market}_rolling_{WINDOWS_DURATION}d_KS_statistic_{EXPERIMENT}.npy', statistics[i])
        np.save(ROOT_DIR / f'data/processed/dynamics/{market}_rolling_{WINDOWS_DURATION}d_KS_outcasts_{EXPERIMENT}.npy', outcasts[i])
        print(f"There are {np.sum(outcasts[i] == 1)} outcasts for market {market}")
    
    conn.close()



if __name__ == "__main__":
    main()
