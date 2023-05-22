"""
Extract data and count FHT
"""
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes

import matplotlib.pyplot as plt
from pathlib import Path

DATABASE = Path('../data/interim')

MARKET = 'UN'
START_DATE = '2010-01-01'
END_DATE = '2022-07-01'
CRITERION = 'percentage'
VALUE = 0.05

START_LEVEL = -0.1
END_LEVEL = -1.5
TAU_MAX = 1e6
NBINS = 2000


def print_indicators_table(header, indicators):
    print(header)
    keys = list(indicators.keys())
    values = list(indicators.values())
    keys_width = max(len(key) for key in keys)
    values_width = max(len(str(value)) for value in values)
    table_width = keys_width + values_width + 5
    print(f"{'-' * table_width}")
    print(f"|{'Key':^{keys_width}}|{'Value':^{values_width}}|")
    print(f"{'-' * table_width}")
    for key, value in indicators.items():
        value_str = str(value) if not isinstance(value, tuple) else ', '.join(map(str, value))
        print(f"|{key:<{keys_width}}|{value_str:^{values_width}}|")
    print(f"{'-' * table_width}")


def get_stabilvol(accountant, analyst):
    data = accountant.extract_data(DATABASE / f'{MARKET}.pickle')

    analysis_info = {
        'Market': MARKET,
        # 'Start date': accountant.start_date.strftime("%Y-%m-%d"),
        # 'Window length': int(accountant.window.length.days / 365.2425)
    }
    stabilvol = analyst.get_stabilvol(data, 'multi', **analysis_info)
    return stabilvol


if __name__ == '__main__':
    from datetime import datetime

    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(1e-5, 1e5)
    )
    analyst = StabilVolter(
        start_level=START_LEVEL,
        end_level=END_LEVEL,
        tau_max=TAU_MAX)

    # GET STABILVOL
    start_time = datetime.now()
    stabilvol = get_stabilvol(accountant, analyst)
    end_time = datetime.now()
    print(f"\n\nStabilvol calculated in {end_time-start_time} seconds\n\n")

    # STATISTICS
    print_indicators_table('FHT Indicators'.upper(), analyst.get_indicators(stabilvol))
    analyst.plot_fht()
    plt.show()

    # GET MFHT
    mfht = MeanFirstHittingTimes(stabilvol, nbins=NBINS, max_volatility=0.2)
    # STATISTICS
    baricenters = mfht.baricenters
    print_indicators_table('MFHT Indicators'.upper(), mfht.indicators)
    mfht.plot()
