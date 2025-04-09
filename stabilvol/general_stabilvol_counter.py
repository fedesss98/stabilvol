"""
Created by Federico Amato
2023 - 05 - 22

Count the stabilvol of all markets in the whole time interval.
Save it to SQLite database.

RECTANGULAR SELECTION is obtained using 'startend' criterion
to select only stocks that starts and ends in the selected interval.
TRAPEZOIDAL SELECTION is obtained using 'percentage' criterion to
select only stocks that have a percentage of data in the selected
interval, including in this way even those that starts after the
start date or ends before the end date.
"""
from utility.definitions import ROOT
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter
from single_stabilvol import print_indicators_table

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine

MARKETS = ['UN_little']
START_DATE = '1980-01-01'
END_DATE = '2022-07-01'
CRITERION = 'percentage'
VALUE = 0.05
COUNTING_METHOD = 'multi'  # This uses multiprocessing

START_LEVEL = -2.0
START_LEVELS = [0.1, 0.2, 0.5, 1.0, 2.0]
DELTAS = [0.1, 0.2, 0.5, 1.0, 2.0]
END_LEVEL = 0.0
LEVELS = {
    (start, start+delta) for start in START_LEVELS for delta in DELTAS
}
TAU_MAX = 1000000

DATABASE = ROOT / 'data/interim'


def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stabilizing Volatility Analysis')
    
    # Add arguments
    parser.add_argument('-m', '--markets', type=str, nargs='+', default=None, help='Markets to analyze')
    parser.add_argument('--method', type=str, default='multi', choices=['pandas', 'multi', 'numpy'], 
                        help='Method to use for calculation (default: multi)')
    parser.add_argument('--threshold-start', type=float, default=START_LEVEL, 
                        help='Starting threshold value')
    parser.add_argument('--threshold-end', type=float, default=END_LEVEL, 
                        help='Ending threshold value')
    parser.add_argument('--tau-min', type=int, default=2, 
                        help='Minimum tau value (default: 2)')
    parser.add_argument('--tau-max', type=int, default=TAU_MAX, 
                        help='Maximum tau value')
    
    # Parse arguments
    return parser.parse_args()


def save_to_database(stabilvol: pd.DataFrame):
    selection_type = 'trapezoidal_selection' if CRITERION == 'percentage' else 'rectangular_selection'
    database_dir = ROOT / f'data/processed/{selection_type}/stabilvol.sqlite'
    # SAVE TO DATABASE
    engine = create_engine(f'sqlite:///{database_dir}')
    start_threshold_string = str(START_LEVEL).replace('-', 'm').replace('.', 'p')
    end_threshold_string = str(END_LEVEL).replace('-', 'm').replace('.', 'p')
    table_name = f"stabilvol_{start_threshold_string}_{end_threshold_string}"
    stabilvol.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return None


def main():
    args = parse_arguments()
    stabilvols = []

    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(1e-5, 1e5)
    )
    analyst = StabilVolter(
        start_level=args.threshold_start,
        end_level=args.threshold_end,
        tau_max=args.tau_max,
        tau_min=args.tau_min)

    markets = MARKETS if args.markets is None else args.markets
    for market in markets:
        print(f"\n{'-'*25}\nCounting {market} stabilvol starting at {datetime.now()}...")
        # GET STABILVOL
        start_time = datetime.now()
        data = accountant.extract_data(DATABASE / f'{market}.pickle')

        analysis_info = {'Market': market}  # Info column to add to result DataFrame
        stabilvol = analyst.get_stabilvol(data, method=COUNTING_METHOD, **analysis_info)
        end_time = datetime.now()
        print(f"Stabilvol calculated in {end_time - start_time} seconds\n")

        # STATISTICS
        print_indicators_table('FHT Indicators'.upper(), analyst.get_indicators(stabilvol))
        analyst.plot_fht(title=f"{market} FHT")
        plt.show()

        stabilvols.append(stabilvol)

    stabilvols = pd.concat(stabilvols, axis=0)
    #save_to_database(stabilvols)
    return stabilvols


if __name__ == '__main__':
    from datetime import datetime

    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"\n{'_'*20}\nTotal Elapsed time: {end_time - start_time} seconds\n\n")
