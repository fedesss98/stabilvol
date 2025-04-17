"""
Iterate in the threshold grid and save to database

THRESHOLD GRID
______________________________________________________________________
theta i |                    theta f
-2.0    |  -4.0  -3.0  -2.5  -2.2  -2.1  -1.9  -1.8  -1.5  -1.0   0.0 \n
-1.0    |  -3.0  -2.0  -1.5  -1.2  -1.1  -0.9  -0.8  -0.5   0.0   1.0 \n
-0.5    |  -2.5  -1.5  -1.0  -0.7  -0.6  -0.4  -0.3   0.0   0.5   1.5 \n
-0.2    |  -2.2  -1.2  -0.7  -0.4  -0.3  -0.1   0.0   0.3   0.8   1.8 \n
-0.1    |  -2.1  -1.1  -0.6  -0.3  -0.2   0.0   0.1   0.4   0.9   1.9 \n
 0.1    |  -1.9  -0.9  -0.4  -0.1   0.0   0.2   0.3   0.6   1.1   2.1 \n
 0.2    |  -1.8  -0.8  -0.3   0.0   0.1   0.3   0.4   0.7   1.2   2.2 \n
 0.5    |  -1.5  -0.5   0.0   0.3   0.4   0.6   0.7   1.0   1.5   2.5 \n
 1.0    |  -1.0   0.0   0.5   0.8   0.9   1.1   1.2   1.5   2.0   3.0 \n
 2.0    |   0.0   1.0   1.5   1.8   1.9   2.1   2.2   2.5   3.0   4.0 \n
______________________________________________________________________
"""

from utility.definitions import ROOT
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter
from single_stabilvol import print_indicators_table, get_stabilvol
from utility.functions import list_database_thresholds

import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
import argparse
import requests
import os


MARKETS = ['UN', 'UW', 'LN', 'JT']
START_DATE = '1980-01-01'
END_DATE = '2022-07-01'
CRITERION = 'percentage'
VALUE = 0.05
COUNTING_METHOD = 'multi'

START_LEVELS = [1.2, 1.4, 1.0, 0.8, 0.6, 0.4, 0.2, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4]
DELTAS = [0.2, -0.2]
LEVELS = {
    (round(start, 2), round(start+delta, 2)) for start in START_LEVELS for delta in DELTAS
}
TAU_MAX = 30

DATABASE = ROOT / 'data/interim'
PLOT_FHT = False


def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stabilizing Volatility Analysis')
    
    # Add arguments
    parser.add_argument('-m', '--markets', type=str, nargs='+', default=MARKETS, help='Markets to analyze')
    parser.add_argument('-l', '--levels', type=float, nargs='+', default=LEVELS, help='Levels to analyze')
    parser.add_argument('--method', type=str, default=COUNTING_METHOD, choices=['pandas', 'multi', 'numpy'], 
                        help='Method to use for calculation (default: multi)')
    parser.add_argument('--tau-min', type=int, default=2, 
                        help='Minimum tau value (default: 2)')
    parser.add_argument('--tau-max', type=int, default=TAU_MAX, 
                        help='Maximum tau value')
    
    # Parse arguments
    return parser.parse_args()


def save_to_database(database_dir, stabilvol: pd.DataFrame, start_level: float, end_level: float):
    # SAVE TO DATABASE
    engine = sqlalchemy.create_engine(f'sqlite:///{database_dir}')
    start_threshold_string = str(start_level).replace('-', 'm').replace('.', 'p')
    end_threshold_string = str(end_level).replace('-', 'm').replace('.', 'p')
    table_name = f"stabilvol_{start_threshold_string}_{end_threshold_string}"
    stabilvol.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return None


def main():
    args = parse_arguments()

    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(1e-5, 1e5)
    )
    # Make global variables local
    levels = args.levels
    markets = args.markets
    print(f"Iterating in levels: {levels}")
    
    selection_type = 'trapezoidal_selection' if CRITERION == 'percentage' else 'rectangular_selection'

    database_dir = ROOT / f'data/processed/{selection_type}/stabilvol_filtered.sqlite'
    saved_levels = list_database_thresholds(database_dir).values.tolist()
    for i, (start_level, end_level) in enumerate(levels):
        if (start_level, end_level) in saved_levels:
            print(f"Skipping {start_level, end_level}...")
            continue
        print(f"- {i/len(levels)}% ({start_level, end_level}) ")
        stabilvols = []
        analyst = StabilVolter(
            start_level=start_level,
            end_level=end_level,
            tau_max=TAU_MAX)

        for market in markets:
            print(f"\n{'-'*25}\nCounting {market} stabilvol starting at {datetime.now()}...")
            # GET STABILVOL
            start_time = datetime.now()
            data = accountant.extract_data(DATABASE / f'{market}.pickle')

            analysis_info = {'Market': market}  # Info column to add to result DataFrame
            try:
                stabilvol = analyst.get_stabilvol(data, method=args.method, **analysis_info)
            except ValueError as e:
                print(f"Error in counting stabilvol: {e}")
                analyst.data = None
                stabilvol = pd.DataFrame()
            else:
                stabilvols.append(stabilvol)
                # STATISTICS
                print_indicators_table('FHT Indicators'.upper(), analyst.get_indicators(stabilvol))
                if PLOT_FHT:
                    analyst.plot_fht(title=f"{market} FHT")
                    plt.show()

            finally:
                end_time = datetime.now()
                print(f"Stabilvol calculated in {end_time - start_time} seconds\n")

            stabilvols.append(stabilvol)

        stabilvols = pd.concat(stabilvols, axis=0)
        save_to_database(database_dir, stabilvols, start_level, end_level)

    return None


def send_notification(start, end, run_error=None):
    token = os.getenv('PYTHONNOTIFIER_TOKEN')
    account_id = os.getenv('TELEGRAM_ID')
    if token is None or account_id is None:
        print("\nUnable to send notification, I lack env variables.")
        return None
    
    url = f"https://api.telegram.org/bot{token}"
    message = f"""Your code finished running! 
    It started at {start} and ended at {end}, taking {end-start} seconds.
    It raised {run_error if run_error is not None else 'no errors'}."""
    params = {
        "chat_id": account_id, 
        "text": message}
    try:
        r = requests.get(url + '/sendMessage', params=params)
    except Exception as e:
        print(f"Unable to send notification: {e}")
    else:
        print(f"Request sent with status code: {r.status_code}")
    return None


if __name__ == '__main__':
    from datetime import datetime

    start_time = datetime.now()
    run_error = None
    try:
        main()
    except Exception as e:
        print(f"Error while processing: {e}")
        run_error = e
    end_time = datetime.now()
    print(f"\n{'_'*20}\nTotal Elapsed time: {end_time - start_time} seconds\n\n")
    send_notification(start_time, end_time, run_error=run_error)
