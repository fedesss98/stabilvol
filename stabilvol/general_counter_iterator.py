"""
Iterate in the threshold grid and save to database

THRESHOLD GRID
______________________________________________________________________
theta i |                    theta f
-0.1    |   0.0   0.1   0.4   0.9   1.9  -0.2  -0.3  -0.6  -1.1  -2.1 \n
-0.2    |  -0.1   0.0   0.3   0.8   1.8  -0.3  -0.4  -0.7  -1.2  -2.2 \n
-0.5    |  -0.4  -0.3  0.0   0.5   1.5  -0.6  -0.7  -1.0  -1.5  -2.5  \n
-1.0    |  -0.9  -0.8  -0.5  0.0   1.0  -1.1  -1.2  -1.5  -2.0  -3.0  \n
-2.0    |  -1.9  -1.8  -1.5  -1.0  0.0  -2.1  -2.2  -2.5  -3.0  -4.0  \n
 0.1    |   0.2   0.3   0.6   1.1   2.1   0.0  -0.1  -0.4  -0.9  -1.9 \n
 0.2    |   0.3   0.4   0.7   1.2   2.2   0.1   0.0  -0.3  -0.8  -1.8 \n
 0.5    |   0.6   0.7   1.0   1.5   2.5   0.4   0.3   0.0  -0.5  -1.5 \n
 1.0    |   1.1   1.2   1.5   2.0   3.0   0.9   0.8   0.5   0.0  -1.0 \n
 2.0    |   2.1   2.2   2.5   3.0   4.0   1.9   1.8   1.5   1.0  0.0  \n
______________________________________________________________________
"""

from utility.definitions import ROOT
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter
from single_stabilvol import print_indicators_table, get_stabilvol

import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy


MARKETS = ['UN', 'UW', 'LN', 'JT']
START_DATE = '1980-01-01'
END_DATE = '2022-07-01'
CRITERION = 'percentage'
VALUE = 0.05

START_LEVELS = [-0.1, -0.2, -0.5, -1.0, -2.0]
DELTAS = [0.1, 0.2, 0.5, 1.0, 2.0]
LEVELS = {
    (start, start+delta) for start in START_LEVELS for delta in DELTAS
}
TAU_MAX = 1000000

DATABASE = ROOT / 'data/interim'


def save_to_database(stabilvol: pd.DataFrame, start_level: float, end_level: float):
    selection_type = 'trapezoidal_selection' if CRITERION == 'percentage' else 'rectangular_selection'
    database_dir = ROOT / f'data/processed/{selection_type}/stabilvol.sqlite'
    # SAVE TO DATABASE
    engine = sqlalchemy.create_engine(f'sqlite:///{database_dir}')
    start_threshold_string = str(start_level).replace('-', 'm').replace('.', 'p')
    end_threshold_string = str(end_level).replace('-', 'm').replace('.', 'p')
    table_name = f"stabilvol_{start_threshold_string}_{end_threshold_string}"
    stabilvol.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return None


def main():
    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(1e-5, 1e5)
    )
    print(LEVELS)
    """
    {(-0.2, 0.0), (-0.5, 0.0), (-0.5, 0.5), (-0.1, 1.9), (-1.0, -0.8), 
    (-0.1, 0.1), (-0.2, 1.8), (-0.2, -0.1), (-2.0, -1.0), (-1.0, 1.0), 
    (-0.1, 0.0), (-1.0, -0.9), (-0.2, 0.3), (-0.2, 0.8), (-2.0, -1.8), 
    (-0.1, 0.4), (-0.1, 0.9), (-1.0, -0.5), (-1.0, 0.0), (-2.0, 0.0 ), 
    (-0.5, -0.3), (-0.5, 1.5), (-2.0, -1.9), (-0.5, -0.4), (-2.0, -1.5)}
    """
    for start_level, end_level in LEVELS:
        stabilvols = []
        analyst = StabilVolter(
            start_level=start_level,
            end_level=end_level,
            tau_max=TAU_MAX)

        for market in MARKETS:
            print(f"\n{'-'*25}\nCounting {market} stabilvol with thresholds {start_level, end_level}...")
            # GET STABILVOL
            start_time = datetime.now()
            stabilvol = get_stabilvol(market, accountant, analyst)
            end_time = datetime.now()
            print(f"Stabilvol calculated in {end_time - start_time} seconds\n")

            # STATISTICS
            print_indicators_table('FHT Indicators'.upper(), analyst.get_indicators(stabilvol))
            analyst.plot_fht(title=f"{market} FHT")
            plt.show()

            stabilvols.append(stabilvol)

        stabilvols = pd.concat(stabilvols, axis=0)
        save_to_database(stabilvols, start_level, end_level)
    return None


if __name__ == '__main__':
    from datetime import datetime

    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"\n{'_'*20}\nTotal Elapsed time: {end_time - start_time} seconds\n\n")