"""
Extract data and count FHT
"""
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter

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
TAU_MAX = 300
NBINS = 6000

if __name__ == '__main__':
    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(1e-5, 1e5)
    )
    data = accountant.extract_data(DATABASE / f'{MARKET}.pickle')
    # accountant.plot_selection()

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
    analyst.get_stabilvol(data, 'multi', **analysis_info)
    end_time = time()
    print(f"\n\nStabilvol calculated in {end_time-start_time} seconds\n\n")
    analyst.plot_fht()
    plt.show()
    mfht = analyst.get_average_stabilvol(nbins=NBINS)
    fht_ax = analyst.plot_mfht(x_range=(0, 0.05), edit=True)
    plt.show()
