import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
import sqlite3
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

from stabilvol.utility import functions as f

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATABASE = ROOT_DIR / 'data/processed/trapezoidal_selection/stabilvol_filtered.sqlite'

MARKETS = ["UN", "UW", "LN", "JT"]

START_LEVELS = [3.0, 2.6, 2.2, 1.8, 1.4, 1., 0.6, 0.2]
DELTAS = [-0.2, -0.4, -0.8, -1.0]
LEVELS = {
    (round(start, 2), round(start+delta, 2)) for start in START_LEVELS for delta in DELTAS
}
LEVELS = sorted(LEVELS)
WINDOWS_DURATION = 90  # days

VOL_LIMIT= 100  # Change this will change all the pickle files, remember to re-generate them
TAU_MAX = 30

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


def main():
    print(os.getcwd())
    if not os.path.exists(DATABASE):
        raise FileNotFoundError(f"Database not found at {DATABASE}")
    
    # Connect to the SQLite database
    conn = setup_optimized_connection(DATABASE)

    # Generate rolling windows
    windows = f.roll_windows(duration=WINDOWS_DURATION, start_date=datetime.date(1985, 1, 1), end_date=datetime.date(2022, 12, 31))
    print(f"Selected {len(windows)} rolling windows for the analysis.")
    
    # Start taking maximum values of MFHT in each window
    max_values = np.zeros((len(MARKETS), len(LEVELS), len(windows)))
    outcasts = np.zeros((len(MARKETS), len(LEVELS), len(windows)))
    coefficients = [(f.stringify_threshold(t1), f.stringify_threshold(t2)) for t1, t2 in LEVELS]
    for i, market in enumerate(MARKETS):
        for j, (t1, t2) in enumerate(tqdm(coefficients, leave=True, desc=f"Processing market {market}")):
            for w, window in enumerate(tqdm(windows, desc=f" - thresholds {t1}-{t2}", leave=False)):
                start, end = window
                try:
                    mfht, nbins = f.query_binned_data(
                        market, start, end, VOL_LIMIT, TAU_MAX, t1, t2, conn=conn
                    )
                except ValueError:
                    outcasts[i, j, w] = 1
                else:
                    if not mfht.empty:
                        # Take the maximum MeanFHT in this window with this thresholds
                        max_values[i, j, w] = mfht['mean'].max()
                    else:
                        outcasts[i, j, w] = 1
        print(f"There are {np.sum(outcasts[i, j, :] == 1)} outcasts for market {market}")
    
    conn.close()
    np.save(ROOT_DIR / f'data/processed/dynamics/rolling_{WINDOWS_DURATION}d_MFHT_peaks.npy', max_values)



if __name__ == "__main__":
    main()