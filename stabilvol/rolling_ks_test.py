"""

"""
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm
import datetime

from utility import functions as f

DATABASE = '../data/processed/trapezoidal_selection/stabilvol.sqlite'

MARKETS = ["UN", "UW", "LN", "JT"]
START_DATE = "1980-01-01"
END_DATE = "2022-07-01"

START_LEVELS = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]
DELTAS = [2.0, 1.0, 0.5, 0.2, 0.1, -0.1, -0.2, -0.5, -1.0, -2.0]
LEVELS = {
    (start, round(start + delta, 2)) for start in START_LEVELS for delta in DELTAS
}
LEVELS = sorted(LEVELS)

VOL_LIMIT = 0.5  # Change this will change all the pickle files, remember to re-generate them


def select_bins(df, startingbins=100, maxbins=1000):
    nbins = startingbins

    while True:
        # Use qcut to bin 'Volatility' values
        df['Bins'] = pd.qcut(df['Volatility'], nbins, duplicates='drop')

        # Group by the bins and calculate the mean of 'FHT'
        # and the number of observations in each bin ('size')
        grouped = df.groupby('Bins')['FHT'].agg(['mean', 'size'])
        # Take the lowest count of observations in the bins
        count = grouped['size'].min()

        if count < maxbins or nbins > 1000:
            break
        else:
            nbins += 20
    return grouped, nbins


def query_binned_data(market: str, start_date: str, end_date: str = None,
                      vol_limit: float = VOL_LIMIT,
                      t1_string: str = "m0p5", t2_string: str = "m1p5", nbins: int = 200):
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE)
    # cur = conn.cursor()
    grouped_data = None
    end_date = '2023-01-01' if end_date is None else end_date
    try:
        # Write the SQL query
        query = f'''
        SELECT *
        FROM stabilvol_{t1_string}_{t2_string}
        WHERE Volatility < {vol_limit} 
        AND Market = "{market}"
        AND start >= "{start_date}"
        AND end <= "{end_date}"    
        '''
        # Load the FHT data from the database
        df = pd.read_sql_query(query, conn)
    except pd.errors.DatabaseError as e:
        raise ValueError(f'No data for market {market} with thresholds {t1_string}-{t2_string}') from e
    else:
        # Data are taken correctly from database,
        # but we need at least one data point for every bin, on average
        if len(df) > nbins:
            return select_bins(df, startingbins=nbins)
        else:
            raise ValueError(
                f'Not enough data for market {market} with thresholds {t1_string}-{t2_string} from {start_date} to {end_date}')


def create_dataset(market, windows, **kwargs):
    outcasts = {market: []}
    df_list = []
    for start_date, end_date in tqdm(windows, desc=market):
        try:
            mfht, nbins = query_binned_data(market, start_date, end_date, **kwargs)
        except ValueError:
            outcasts[market].append((start_date, end_date))
        else:
            mfht['start'] = start_date
            mfht['end'] = end_date
            mfht['market'] = market
            df_list.append(mfht.reset_index())

    return pd.concat(df_list), outcasts


def test_mfhts(market, windows, coefficients, nbins, regenerate=False):
    outcasts = {coeff: [] for coeff in coefficients}
    p_values = np.zeros((len(coefficients), len(windows))) - 1
    stats_values = np.zeros((len(coefficients), len(windows))) - 1
    for i, (t1_string, t2_string) in enumerate(coefficients):
        filename = f'{market}_rolling_MFHT_peaks_{t1_string}_{t2_string}_{nbins}bins_{VOL_LIMIT}.pickle'
        df = pd.read_pickle(f'../data/processed/dynamics/{filename}')
        # Create the first mfht to test with
        mfht_previous = df[(df['start'] == windows[0][0]) & (df['end'] == windows[0][1])]['mean']
        for j, (start_date, end_date) in tqdm(enumerate(windows[1:]), desc=market):
            mfht = df[(df['start'] == start_date) & (df['end'] == end_date)]['mean']
            if not mfht.empty and not mfht_previous.empty:
                p_values[i, j] = ks_2samp(mfht_previous, mfht).pvalue
                stats_values[i, j] = ks_2samp(mfht_previous, mfht).statistic
            else:
                outcasts[(t1_string, t2_string)].append((start_date, end_date))
            mfht_previous = mfht
    np.save(f'../data/processed/kstest/{market}_rolling_MFHT_stats_variousthresholds_{VOL_LIMIT}.npy', stats_values)
    np.save(f'../data/processed/kstest/{market}_rolling_MFHT_p_variousthresholds_{VOL_LIMIT}.npy', p_values)
    return p_values, stats_values, outcasts


def roll_windows(duration=250, start_date=None, end_date=None):
    # Define the start and end dates
    start_date = datetime.date(1980, 1, 1) if start_date is None else start_date
    end_date = datetime.date(2022, 7, 1) if end_date is None else end_date

    start = start_date + pd.to_timedelta(duration / 2, 'D')
    end = end_date - pd.to_timedelta(duration / 2, 'D')
    return [(mid - pd.to_timedelta(duration // 2, 'D'), mid + pd.to_timedelta(duration // 2, 'D')) for mid in
            pd.date_range(start, end, freq='D')]




def main(
        market,
        coefficients,
        nbins,
        force = False,
        ):
    # We can take 250 the approximate number of business days in a year
    windows_duration = 250
    windows = roll_windows(windows_duration, start_date=datetime.date(1980, 1, 1), end_date=datetime.date(2022, 7, 1))
    print(f'===============================\n'
          f'Starting KS-testing market {market} with {nbins} bins in '
          f'{windows_duration}-days long windows.\n'
          f'===============================')

    for t1_string, t2_string in coefficients:
        filename = f'{market}_rolling_MFHT_peaks_{t1_string}_{t2_string}_{nbins}bins_{VOL_LIMIT}.pickle'
        if not os.path.exists(f'../data/processed/dynamics/{filename}') or force:
            print(f"Generating {market} MFHT {nbins} bins with thresholds {t1_string}-{t2_string}")
            # Data must be regenerate
            df, outcasts = create_dataset(market, windows,
                                          t1_string=t1_string, t2_string=t2_string)
            print(f"There are {len(outcasts[market])} outcasts")
            df.to_pickle(f'../data/processed/dynamics/{filename}')
        else:
            print(f'File "{filename}" already exists')


if __name__ == '__main__':
    import os

    print(os.getcwd())
    os.path.exists(DATABASE)

    coefficients = [
        ("m0p5", "m1p5"),
        # ("0p5", "m1p5"),
        ("0p5", "1p5"),
        # ("1p0", "3p0"),
    ]
    main(market='UN', coefficients=coefficients, nbins=200, force=False)
