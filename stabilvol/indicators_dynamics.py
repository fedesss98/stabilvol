"""
Created by Federico Amato
2023 - 04 - 25

ROADMAP -- EXPERIMENTS TO TRY

1) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -2.0],
    2000 bins
    LOG IDS:
        UN: 003865Y
        LN: 653851X
        UW: 643630M
        JT: 102939K
Elapsed time: 00:46:39.99
2) 1990 to 2020 every 4 month, 2-years windows,
    thresholds [-0.1; -2.0],
    2000 bins
    LOG IDS:
        UN: 365374G
        LN: 655972O
        UW: 941047N
        JT: 019647Q
Elapsed time: 00:35:39.72
3) 1990 to 2020 every 4 month, 1-year windows,
    thresholds [-0.1; -2.0],
    2000 bins
    LOG IDS:
        UN: 644354L
        LN: 485604S
        UW: 256890I
        JT: 560436R
    and
        UN: 928387S
        LN: 774892E
        UW: 960223U
        JT: 897643T
Elapsed time: 00:28:42.0 and 00:39:45.7
4) 1990 to 2020 every 4 month, 1-year windows,
    thresholds [-0.1; -2.0],
    1000 bins
    LOG IDS:
        UN: 242769F
        LN: 963882T
        UW: 088988V
        JT: 711534E
Elapsed time: 00:48:09.12
5) 1990 to 2020 every 4 month, 1-year windows,
    thresholds [-0.1; -2.0],
    500 bins
    LOG IDS:
        UN: 565988I
        LN: 899299O
        UW: 368353O
        JT: 668904O
Elapsed time: 0:28:12.86
6) 1990 to 2020 every 4 month, 2-years windows,
    thresholds [-0.1; -1.5],
    2000 bins
    LOG IDS:
        UN: 852031O
        LN: 605850U
        UW: 021340F
        JT: 482288K
Elapsed time: 0:35:48.537231
7) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -1.5],
    2000 bins
    LOG IDS:
        UN: 242914S
        LN: 565750N
        UW: 429252I
        JT: 943908F
Elapsed time: 0:46:54.86
8) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -1.5],
    5000 bins
    LOG IDS:
        UN: 709338D
        LN: 927491K
        UW: 289657I
        JT: 902600X
Elapsed time:
9) 1990 to 2020 every 4 month, 4-years windows,
    thresholds [-0.1; -1.5],
    500 bins
    LOG IDS:
        UN: 042109Z
        LN: 715743X
        UW: 062485K
        JT: 927218C
Elapsed time: 0:47:25.958985
10) 1990 to 2020 every 4 month, 2-years windows,
    thresholds [-0.1; -1.5],
    500 bins
    LOG IDS:
        UN: 855945R
        LN: 877537K
        UW: 775312L
        JT: 637990M
Elapsed time: 0:39:08.42
11) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -2.0],
    2000 bins
    LOG IDS:
        UN: 720192X
        LN: 811006M
        UW: 739420C
        JT: 223400A
Elapsed time: 1:11:22.762211
12) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [0.0; -1.0],
    1000 bins
    LOG IDS:
        UN: 329900I
        LN: 971711C
        UW: 129109Y
        JT: 864193Z
Elapsed time: 1:11:35.33
13) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -2.0],
    1000 bins
    LOG IDS:
        UN: 331719B
        LN: 927049F
        UW: 222677X
        JT: 737397X
Elapsed time: 1:12:36.52
14) 1990 to 2020 every 2 months, 1-years windows,
    thresholds [0.0; -1.0],
    1000 bins
    LOG IDS:
        UN: 992991E
        LN: 883178B
        UW: 225521B
        JT: 091567J
Elapsed time: 0:58:21.67
15) 1990 to 2020 every 2 months, 1-years windows,
    thresholds [-0.1; -2.5],
    1000 bins
    LOG IDS:
        UN: 107340Q
        LN: 258743T
        UW: 926899H
        JT: 781892G
Elapsed time: 0:58:19.79
16) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -2.0],
    500 bins
    LOG IDS:
        UN: 048676L
        LN: 897626F
        UW: 960615A
        JT: 821663I
Elapsed time: 1:16:13.7421
17) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [0.0; -1.0],
    2000 bins
    LOG IDS:
        UN: 778268L
        LN: 078472F
        UW: 635896G
        JT: 349950Z
Elapsed time: 1:09:59.51
18) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -2.5],
    2000 bins
    LOG IDS:
        UN: 792644H
        LN: 379283H
        UW: 467421T
        JT: 459347R
Elapsed time: 1:09:23.77
19) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -3.0],
    2000 bins
    LOG IDS:
        UN: 619996D
        LN: 788769K
        UW: 993583A
        JT: 388030Z
Elapsed time:  1:12:22.55
20) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -3.0],
    1000 bins
    LOG IDS:
Elapsed time:
21) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -3.0],
    500 bins
    LOG IDS:
Elapsed time:
22) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -1.0],
    2000 bins
    LOG IDS:
Elapsed time:
23) 1990 to 2020 every 2 months, 2-years windows,
    thresholds [-0.1; -1.0],
    1000 bins
    LOG IDS:
Elapsed time:
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
FREQ = '2M'

EXPERIMENT = 'dynamics_18'

COMMON_EXTRACTOR_DICT = {
    'duration': 2,
    'criterion': 'startend',
    'criterion_value': '7d',
}

COMMON_ANALYSIS_DICT = {
    'start_level': -0.1,
    'end_level': -3.0,
    'tau_max': 1e6,
}

COMMON_MFHT_DICT = {
    'nbins': 2000,
}

MINIMUM_STABILVOL_LEN = 800


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
        **COMMON_EXTRACTOR_DICT,
    )
    data = accountant.extract_data(DATABASE / f'interim/{market}.pickle')
    del accountant
    return data


def get_stabilvol(data):
    analyst = StabilVolter(**COMMON_ANALYSIS_DICT)
    stabilvol = analyst.get_stabilvol(data, 'multi')
    if len(stabilvol) < MINIMUM_STABILVOL_LEN:
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
    mfht = MeanFirstHittingTimes(stabilvol, max_volatility=max_volatility, **COMMON_MFHT_DICT)
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
                    **COMMON_ANALYSIS_DICT,
                    **COMMON_MFHT_DICT,)
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
        # Arguments for parallel computation
        args = [(start_date, market, max_volatility) for start_date in dates]
        # Make analysis in parallel leaving 4 CPUs free
        results = Parallel(n_jobs=8)(delayed(take_indicators)(arg) for arg in args)
        format_and_save(results, logger, market)
    return None


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    print(f"Starting dynamics analysis at {start_time}")
    main()
    end_time = datetime.now()
    print(f"Elapsed time: {end_time - start_time}")
