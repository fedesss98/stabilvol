"""
This script is used to bin the FHT data from the database and save it as a pickle file.
Binned FHT represents our MFHT vs Volatility data.
The number of observations in each bin is saved in a dictionary.
"""

import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite3
import pickle

from pathlib import Path
from utility.functions import stringify_threshold, numerify_threshold, format_mfht_directory


ROOT = Path('./')
SELECTION_CRITERION = 'trapezoidal'
DATABASE = f'data/processed/{SELECTION_CRITERION}_selection/stabilvol_filtered.sqlite'

MARKETS = ["UN", "UW", "LN", "JT"]
START_LEVELS = [1.8, 1.6, 1.4, 1.2, 1.0, -1.0, -1.2, -1.4, -1.6, -1.8]
DELTAS = [1.0, 0.8, 0.6, 0.4, 0.2, -0.2, -0.4, -0.6, -0.8, -1.0]

VOL_LIMIT= 100  # Change this will change all the pickle files, remember to re-generate them
TAU_MAX = 30

ENOUGH_IN_BIN = 1000  # Maximum number of observations in a bin


def parse_arguments():
    parser = argparse.ArgumentParser(description="MFHT Binning Script")
    parser.add_argument("-m", "--markets", type=str, nargs="+", default=MARKETS, help="Markets to analyze")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate all pickle files")
    parser.add_argument("-s", "--start-levels", type=float, nargs="+", default=START_LEVELS, help="Levels to analyze")
    parser.add_argument("-d", "--deltas", type=float, nargs="+", default=DELTAS, help="Deltas to analyze")
    parser.add_argument("-t", "--tau-max", type=int, default=TAU_MAX, help="Maximum tau value")
    parser.add_argument("-v", "--vol-limit", type=float, default=VOL_LIMIT, help="Volatility limit")

    return parser.parse_args()


def select_bins(df, max_n=ENOUGH_IN_BIN):
    # Function to calculate the standard error of MFHT in each bin
    def error_on_the_mean(values):
        return np.std(values)/np.sqrt(len(values))
    
    nbins = 50
    
    while True:
        # Use qcut to bin 'Volatility' values
        df['Bins'] = pd.qcut(df['Volatility'], nbins, duplicates='drop')
        bin_count = df.groupby('Bins', observed=False)['FHT'].count()
        bin_count.index = [(i.left, i.right) for i in bin_count.index]
        # Group by the bins and calculate the mean and standard error of 'value'
        grouped = df.groupby('Bins', observed=False)['FHT'].agg(['mean', error_on_the_mean, 'size'])
        count = grouped['size'].min()
        
        if count < max_n or nbins > 1000:
            break
        else:
            nbins += 50
    return grouped, bin_count.to_dict()


def query_binned_data(conn, market: str, t1_string: str, t2_string: str, filters: list):
    # Unpack the filters
    vol_limit, tau_max = filters
    grouped_data = None
    nbins = 0
    bin_count = 0
    try:            
        # Write the SQL query
        query = f'''
        SELECT *
        FROM stabilvol_{t1_string}_{t2_string}
        WHERE Volatility < {vol_limit} 
        AND Market = "{market}"
        AND FHT < "{tau_max}"
        '''
        # Load the FHT data from the database
        df = pd.read_sql_query(query, conn)
    except pd.errors.DatabaseError:
        raise ValueError(f'No data for market {market} with thresholds {t1_string}-{t2_string}')
    else:
        grouped_data, bin_count = select_bins(df)
    return grouped_data, bin_count


def bin_fht(bin_counts, market, levels, filters, file_dir, conn, regenerate=False):

    if not regenerate:
        levels = [level_tuple for level_tuple in levels if level_tuple not in bin_counts.keys()]

    pbar = tqdm(levels, desc=f'Processing {market}')
    for t1, t2 in pbar:
        # Create the strings for the threshold values
        t1_string = stringify_threshold(t1)
        t2_string = stringify_threshold(t2)
        # Filename for the MFHT data
        filename = file_dir / f'mfht_{market}_{t1_string}_{t2_string}.pkl'

        # Load the FHT dataframe from the database and bin it
        try:
            grouped_data, bin_count = query_binned_data(conn, market, t1_string, t2_string, filters)
        except ValueError as e:
            logging.warning(f'Error in query: {e}')
            continue
        else:
            grouped_data.to_pickle(filename)
            pbar.set_postfix_str(f'Saved to {filename}')
        
        bin_counts[(t1, t2)] = bin_count


def main():
    args = parse_arguments()
    
    levels = {
        (start, round(start+delta, 2)) for start in args.start_levels for delta in args.deltas
    }

    file_dir = ROOT / format_mfht_directory(SELECTION_CRITERION, args.vol_limit)
    output_file = file_dir / "bin_counts.pkl"
    if output_file.exists():
        with open(output_file, "rb") as output:
            counts = pickle.load(output)
    else:
        counts = {}

    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE)

    filter_args = (args.vol_limit, args.tau_max)
    for market in args.markets:
        if counts.get(market) is None:
            counts[market] = {}
        
        bin_fht(counts[market], market, levels, filter_args, file_dir, conn, regenerate=args.regenerate)

    # Save the bin counts to JSON file
    with open(output_file, "wb") as output:
        pickle.dump(counts, output)


if __name__ == "__main__":
    main()
