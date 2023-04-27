"""
Created by Federico Amato
2023 - 04 - 25

ROADMAP -- EXPERIMENTS TO TRY

1) 1990 to 2020 every 4 month, 2-years windows,
    thresholds [-0.1; -2.0],
    2000 bins

2) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -2.0],
    2000 bins

3) 1990 to 2020 every 4 month, 1-year windows,
    thresholds [-0.1; -2.0],
    2000 bins

4) 1990 to 2020 every 4 month, 2-years windows,
    thresholds [-0.1; -1.5],
    2000 bins

5) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -1.5],
    2000 bins

6) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -1.5],
    5000 bins

7) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -1.5],
    500 bins

8) 1990 to 2020 every 4 month, 2-years windows,
    thresholds [-0.1; -1.5],
    500 bins
"""
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes
from log.logger import Logger

import logging
from joblib import Parallel, delayed
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

from pathlib import Path
DATABASE = Path(__file__).parent.parent / 'data'

logging.basicConfig(level=logging.WARNING)

START_DATE = '1990-01-01'
END_DATE = '2020-01-01'
FREQ = '4M'

EXPERIMENT = 'dynamics_1'

COMMON_EXTRACTOR_DICT = {
    'duration': 2,
    'criterion': 'startend',
    'criterion_value': '20d',
}

COMMON_ANALYSIS_DICT = {
    'start_level': -0.1,
    'end_level': -2.0,
    'tau_max': 1e6,
    'nbins': 2000,
}


# def retrieve_data(start_date, market):
#     with DataExtractor(
#         start_date=start_date,
#         duration=6,
#         criterion='startend',
#         criterion_value='20d',
#     ) as accountant:
#         data = accountant.extract_data(DATABASE / f'interim/{market}.pickle')
#     # del accountant
#     return data
# def get_stabilvol(data):
#     with StabilVolter(end_level=-2.0, tau_max=1e6) as analyst:
#         stabilvol = analyst.get_stabilvol(data, 'multi')
#         if len(stabilvol) < 4000:
#             raise ValueError(f"Too little stabilvol data: {len(stabilvol)} rows")
#         indicators = analyst.get_indicators(stabilvol)
#     return stabilvol, indicators


def retrieve_data(start_date, market):
    accountant = DataExtractor(
        start_date=start_date,
        duration=4,
        criterion='startend',
        criterion_value='20d',
    )
    data = accountant.extract_data(DATABASE / f'interim/{market}.pickle')
    del accountant
    return data


def get_stabilvol(data):
    analyst = StabilVolter(end_level=-2.0, tau_max=int(1e6))
    stabilvol = analyst.get_stabilvol(data, 'multi')
    if len(stabilvol) < 4000:
        raise ValueError(f"Too little stabilvol data: {len(stabilvol)} rows")
    indicators = analyst.get_indicators(stabilvol)
    del analyst
    return stabilvol, indicators


def set_ref_max_volatility(market):
    data = retrieve_data('2004-01-01', market)
    _, indicators = get_stabilvol(data)
    max_volatility = indicators['Peak'] + 4 * indicators['FWHM']
    return max_volatility


def analyze_data(data, max_volatility=None):
    stabilvol, indicators = get_stabilvol(data)
    volatilty_cut = indicators['Peak'] + 4 * indicators['FWHM']
    max_volatility = max_volatility if max_volatility is not None else volatilty_cut
    mfht = MeanFirstHittingTimes(stabilvol, nbins=2000, max_volatility=max_volatility)
    return mfht


def take_indicators(args):
    start_date, market, max_volatility = args
    data = retrieve_data(start_date, market)
    if data.empty:
        logging.warning(f"Empty data in {start_date}")
        return None
    try:
        mfht = analyze_data(data, max_volatility)
    except ValueError as e:
        logging.warning(f"Error in getting stabilvol in {start_date}: {e}")
        return None
    indicators = mfht.indicators
    indicators['Stocks'] = len(data.columns)
    return start_date, indicators


def format_and_save(results, logger, market):
    market_results = {}  # dict of results that will be converted to DataFrame
    for result in results:
        if result is not None:
            start_date, indicators = result
            market_results[start_date] = indicators
    market_results = pd.DataFrame.from_dict(market_results, orient='index')
    # market_results['Max'].plot()
    # plt.show()
    logger.save_log(experiment=EXPERIMENT,
                    market=market,
                    **COMMON_EXTRACTOR_DICT,
                    **COMMON_ANALYSIS_DICT)
    market_results.to_pickle(DATABASE / f'processed/dynamics/{logger.id}.pickle')
    print(f"Succesfully saved {market} results with id {logger.id}")
    return None


def main():
    logger = Logger()
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQ)

    for market in ['UN', 'LN', 'UW', 'JT']:
        # Create new Logger ID for this iteration
        logger.update()
        # Set reference maximum volatility to cut the MFHT
        max_volatility = set_ref_max_volatility(market)
        market_results = {}  # dict of results that will be converted to DataFrame
        # Arguments for parallel computation
        args = [(start_date, market, max_volatility) for start_date in dates]
        # Make analysis in parallel leaving 4 CPUs free
        results = Parallel(n_jobs=7)(delayed(take_indicators)(arg) for arg in args)
        format_and_save(results, logger, market)
    return None


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
