"""
Extract data and count FHT
"""
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter

from pathlib import Path

DATABASE = Path('../data/interim')

MARKET = 'UW'
START_DATE = None
END_DATE = None
CRITERION = 'percentage'
VALUE = 0

NBINS = 3000

if __name__ == '__main__':
    accountant = DataExtractor()
    analyst = StabilVolter()

    accountant = DataExtractor(
        start_date=START_DATE,
        end_date=END_DATE,
        criterion=CRITERION,
        criterion_value=VALUE,
        sigma_range=(0.001, 10000)
    )
    data = accountant.extract_data(DATABASE / f'{MARKET}.pickle')
    # accountant.plot_selection()

    analysis_info = {
        'Market': MARKET,
        'Start date': data.index[0],
        'Window length': accountant.window.length
    }
    analyst.get_stabilvol(data, **analysis_info)
    mfht = analyst.get_average_stabilvol(nbins=NBINS)
    # analyst.plot_mfht()
