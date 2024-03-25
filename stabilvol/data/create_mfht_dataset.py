import pandas as pd
import datetime
import sqlite3
from multiprocessing import Pool, current_process
import os
import click

DATABASE = 'data/processed/trapezoidal_selection/stabilvol.sqlite'
# CONN = sqlite3.connect(DATABASE)

VOL_LIMIT = 0.5


def select_bins(df, max_n=1000):
    nbins = 200
    while True:
        df['Bins'] = pd.qcut(df['Volatility'], nbins, duplicates='drop')
        grouped = df.groupby('Bins')['FHT'].agg(['mean', 'size'])
        count = grouped['size'].min()
        if count < max_n or nbins > 1000:
            break
        else:
            nbins += 100
    return grouped, nbins


def query_binned_data(market: str, start_date:str, end_date:str = None, vol_limit:float = 0.5, t1_string:str = "m0p5", t2_string:str = "m1p5", db=DATABASE):
    conn = sqlite3.connect(db)
    end_date = '2023-01-01' if end_date is None else end_date
    try:
        query = f'''
        SELECT *
        FROM stabilvol_{t1_string}_{t2_string}
        WHERE Volatility < {vol_limit}
        AND Market = "{market}"
        AND start >= "{start_date}"
        AND end <= "{end_date}"
        '''
        df = pd.read_sql_query(query, conn)
    except pd.errors.DatabaseError:
        print(f'No data for market {market} with thresholds {t1_string}-{t2_string}')
        nbins = 0
    else:
        if len(df) > 50:
            return  select_bins(df)
        else:
            raise ValueError(f'Not enough data for market {market} with thresholds {t1_string}-{t2_string} from {start_date} to {end_date}')


def process_market_window(args):
    market, t1_string, t2_string, db, window = args
    try:
        mfht, nbins = query_binned_data(
            market, *window, vol_limit=0.5, t1_string=t1_string, t2_string=t2_string,
            db=db)
    except ValueError as e:
        # file.write(f'ERROR: {e}\n')
        return window
    else:
        # file.write(f'Processed {market} with thresholds {t1_string}-{t2_string} from {window[0]} to {window[1]}\n')
        mfht['start'] = window[0]
        mfht['end'] = window[1]
        mfht['market'] = market
        return mfht.reset_index()


def create_dataset(market, windows, t1_string, t2_string, vol_limit=0.5):
    with open(f'data/processed/dynamics/{market}_{t1_string}_{t2_string}.log', 'w') as f:
        f.write(f"Processing {market} with {len(windows)} windows")
        print(f"Processing {market} with {len(windows)} windows")
        outcasts = []
        df_list = []
        common_args = (market, t1_string, t2_string, DATABASE)
        with Pool() as pool:
            args = [common_args + (window,) for window in windows]
            results = pool.map(process_market_window, args)

        for result in results:
            if isinstance(result, pd.DataFrame):
                f.write(f"Processed {market} with thresholds {t1_string}-{t2_string} from {result['start'].min()} to {result['end'].max()}\n")
                df_list.append(result)
            else:
                f.write(f"ERROR: in window {result}\n")
                window = result
                outcasts.append(window)

    return pd.concat(df_list), outcasts


def roll_windows(duration=250, start_date=None, end_date=None):
    """
    Create rolling windows of a given duration.
    The windows are shifted by one day.
    """
    # Define the start and end dates
    start_date = datetime.date(1980, 1, 1) if start_date is None else start_date
    end_date = datetime.date(2022, 7, 1) if end_date is None else end_date

    start = start_date + pd.to_timedelta(duration / 2, 'D')
    end = end_date - pd.to_timedelta(duration / 2, 'D')
    return [(mid - pd.to_timedelta(duration // 2, 'D'), mid + pd.to_timedelta(duration // 2, 'D')) for mid in
            pd.date_range(start, end, freq='D')]


@click.command()
@click.option('-m', '--market', required=True, help='Market to process')
@click.option('-t', '--thresholds', help='Thresholds to use', type=str, nargs=2, default=["0p5", "1p5"])
def main(market, thresholds):
    windows = roll_windows(365, start_date=datetime.date(1980, 1, 1), end_date=datetime.date(2022, 7, 1))
    t1_string, t2_string = thresholds
    df, outcasts = create_dataset(market, windows, t1_string, t2_string)
    df.to_pickle(f'data/processed/dynamics/{market}_rolling_MFHT_peaks_{t1_string}_{t2_string}_200bins_{VOL_LIMIT}.pickle')
    return None


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f"Processing took {end_time - start_time}")