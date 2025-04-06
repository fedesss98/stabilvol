"""
Perform a Kolmogorov-Smirnov test on two MFHTs distributions n-days apart
"""
import datetime

import click
from multiprocessing import Pool
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
import os

try:
    from data.create_mfht_dataset import roll_windows
except ModuleNotFoundError:
    from create_mfht_dataset import roll_windows

VOL_LIMIT = 0.5


def create_window_from_center(center):
    return pd.Timestamp(center) - pd.Timedelta(days=365 // 2), pd.Timestamp(center) + pd.Timedelta(days=365 // 2)


def test_mfht(args):
    df, window_ref, window_shifted = args
    start_date1, end_date1 = window_ref
    start_date2, end_date2 = window_shifted
    # Take the MFHT at the current window
    try:
        mfht = df[start_date1, end_date1]
        mfht_shifted = df[start_date2, end_date2]
    except KeyError:
        # There are no data for this time window
        return -2, -2, (start_date1, end_date1)

    # Perform the KS-test
    if not mfht.empty and not mfht_shifted.empty:
        p_value = ks_2samp(mfht, mfht_shifted).pvalue
        stats_value = ks_2samp(mfht, mfht_shifted).statistic
        outcast_window = []
    else:
        p_value = -1
        stats_value = -1
        outcast_window = (start_date1, end_date1)
    return p_value, stats_value, outcast_window


@click.command()
@click.option('-m', '--market', prompt='Market', help='The market to test.',
              required=True, type=click.Choice(['UN', 'UW', 'LN', 'JT']))
@click.option('--ndays', help='The number of days to test.',
              default=28, type=int)
@click.option('-ref', '--ref-date', help='Center of fixed window to compare other to.',
              default=None, type=str)
@click.option('-t', '--thresholds', help='Thresholds to use.',
              type=str, nargs=2, default=["0p5", "1p5"])
def test_mfhts(market, ndays, ref_date, thresholds):
    main(market, ndays, ref_date, thresholds)


def main(market, ndays, ref_date, thresholds):
    # Parse the thresholds
    t1_string, t2_string = thresholds
    # Read the MFHTs data
    try:
        df = pd.read_pickle(f'../../data/processed/dynamics/{market}_rolling_MFHT_peaks_{t1_string}_{t2_string}_200bins_{VOL_LIMIT}.pickle')
    except FileNotFoundError:
        print(f"File not found, you are in: {os.getcwd()}")
        return
    # Retain only the MFHT series with start-date/end-date indexes
    df = df.set_index(['start', 'end'])['mean']  # This will return a Series
    # Create rolling windows of a given duration
    windows = roll_windows(365, start_date=datetime.date(1980, 1, 1), end_date=datetime.date(2022, 7, 1))
    # Create the two shifted windows series
    windows2 = windows[ndays:] if ref_date is None else windows
    fixed_window = create_window_from_center(ref_date) if ref_date is not None else None
    windows1 = windows[:-ndays] if fixed_window is None else [fixed_window for _ in windows2]
    print(f"Testing {market} MFHTs distributions with thresholds {t1_string}-{t2_string} {ndays} days apart")
    if ref_date is not None:
        print(f"Reference date: {ref_date}")

    # Alllocate space for the p-values and statistics values of the KS-test
    # Default it to -1 to identify the windows with processing errors
    p_values = np.zeros(len(windows1)) - 1
    stats_values = np.zeros(len(windows1)) - 1
    outcasts = []

    common_args = (df, )
    with Pool() as pool:
        args = [common_args + (window_ref, window_shift)
                for window_ref, window_shift in zip(windows1, windows2)]
        results = pool.map(test_mfht, args)

    for i, result in enumerate(results):
        p_value, stats_value, outcast = result
        p_values[i] = p_value
        stats_values[i] = stats_value
        if outcast:
            outcasts.append(outcast)

    print(f"\nProcess finished with {len(outcasts)} outcasts found among {len(results)} windows.")
    plt.plot(p_values)
    plt.plot(0.05 * np.ones(len(p_values)), 'r--')
    plt.show()
    # Save the p-values and statistics values of the KS-tests
    days_or_win = ndays if ref_date is None else ref_date.replace('-', '_')
    print(f"Saving p-values and stats values to: "
          f'../../data/processed/kstest/{market}_rolling_{days_or_win}_MFHT_p_{t1_string}_{t2_string}_{VOL_LIMIT}.npy')
    np.save(
        f'../../data/processed/kstest/{market}_rolling_{days_or_win}_MFHT_stats_{t1_string}_{t2_string}_{VOL_LIMIT}.npy', stats_values)
    np.save(
        f'../../data/processed/kstest/{market}_rolling_{days_or_win}_MFHT_p_{t1_string}_{t2_string}_{VOL_LIMIT}.npy', p_values)
    return p_values, stats_values, outcasts


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    df_maxs = pd.read_csv(
        f'../../data/processed/dynamics/extremevalues_rolling_MFHT_peaks_variousthresholds_{VOL_LIMIT}.csv',
        parse_dates=True)
    for i in df_maxs.index:
        row = df_maxs.loc[i]
        thresholds = row['thresholds'].split('_')
        ref_date = str(row['center_crysis']).split(' ')[0]
        main(row['market'], 28, ref_date, thresholds)
    end_time = datetime.datetime.now()
    print(f"Finished in {end_time - start_time}")