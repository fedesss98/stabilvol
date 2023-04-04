"""
Extract data and count FHT
"""
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes

import matplotlib.pyplot as plt
from time import time
from pathlib import Path

DATABASE = Path('../data/interim')

MARKET = 'UN'
START_DATE = '1986'
END_DATE = '1998'
CRITERION = 'percentage'
VALUE = 0

START_LEVEL = -0.1
END_LEVEL = -1.5
TAU_MAX = 1e6
NBINS = 6000


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
        print(f"|{key:<{keys_width}}|{value:^{values_width}}|")
    print(f"{'-' * table_width}")


if __name__ == '__main__':
    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(1e-5, 1e5)
    )
    data = accountant.extract_data(DATABASE / f'{MARKET}.pickle')

    analyst = StabilVolter(
        start_level=START_LEVEL,
        end_level=END_LEVEL,
        tau_max=TAU_MAX)
    analysis_info = {
        'Market': MARKET,
        'Start date': accountant.start_date.strftime("%Y-%m-%d"),
        'Window length': int(accountant.window.length.days / 365.2425)
    }
    start_time = time()
    stabilvol = analyst.get_stabilvol(data, 'multi', **analysis_info)
    end_time = time()
    print(f"\n\nStabilvol calculated in {end_time-start_time} seconds\n\n")
    indicators = analyst.get_indicators(stabilvol)
    print_indicators_table('FHT Indicators'.upper(), indicators)
    analyst.plot_fht()
    plt.show()
    mfht = MeanFirstHittingTimes(stabilvol, nbins=6000, max_volatility=0.2)
    baricenters = mfht.baricenters
    mfht_indicators = mfht.indicators
    mfht.plot()
    print_indicators_table('MFHT Indicators'.upper(), mfht_indicators)
    mfht2 = MeanFirstHittingTimes(mfht.values, nbins=500)
    mfht2.plot()
    mfht_indicators = mfht2.indicators
    print_indicators_table('MFHT2 Indicators'.upper(), mfht_indicators)
