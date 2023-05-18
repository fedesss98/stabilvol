"""
Created by Federico Amato
2023 - 04 - 18

How does the MFHT indicators varies with the choice of different parameters?
"""


from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes

import logging
from itertools import product
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path
DATABASE = Path('../data')

logging.basicConfig(level=logging.WARNING)

MARKETS = [
    'UN'
]
START_DATE = '2002'
END_DATE = '2008'
SIGMA_RANGE = (1e-5, 1e5)

MAX_VOLATILITY_CUT = 4

PARAMETERS1 = {
    'nbins': [200, 500, 1000, 2000, 5000, 10000],
    'start_threshold': [-0.1, -0.4, -0.6, -0.8],
    'thresholds_spacing': [2, 1.5, 1, 0.3],
    'tau_max': [1e8],
    'criterion': ['PR09', 'PR075', 'SE7D'],
}

PARAMETERS2 = {
    'nbins': [200, 500, 1000, 2000, 5000, 10000],
    'start_threshold': [0.0, -0.5, -0.8, -0.9],
    'thresholds_spacing': [2, 1.5, 1, 0.4],
    'tau_max': [1e8, 1e6, 1e4, 1e3],
    'criterion': ['PR09', 'PR075', 'PR05'],
}

CRITERION_CONVERSION = {
    'PR09': ('percentage', 0.9),
    'PR075': ('percentage', 0.75),
    'PR05': ('percentage', 0.5),
    'SE7D': ('startend', '7d'),
}


def create_index():
    index = []
    for i, params in enumerate(product(*PARAMETERS1.values())):
        index = []


def main():
    for market in MARKETS:
        logging.info(f"Market: {market}")
        market_results = {}
        for params in tqdm(product(*PARAMETERS1.values())):
            current_params = dict(zip(PARAMETERS1.keys(), params))
            # DATA EXTRACTION
            # The selection criterion and its value are nested inside a dictionary
            criterion, criterion_value = CRITERION_CONVERSION[current_params['criterion']]
            accountant = DataExtractor(
                start_date=START_DATE,
                end_date=END_DATE,
                criterion=criterion,
                criterion_value=criterion_value,
                sigma_range=SIGMA_RANGE,
            )
            data = accountant.extract_data(DATABASE / f'interim/{market}.pickle')
            if data.empty:
                logging.warning(f"Empty data for {market} with: {current_params}")
                continue
            # FHT COUNTING
            analyst = StabilVolter(
                start_level=current_params['start_threshold'],
                end_level=current_params['start_threshold'] - current_params['thresholds_spacing'],
                tau_max=current_params['tau_max'])
            stabilvol = analyst.get_stabilvol(data, 'multi')
            if len(stabilvol) < 1000:
                raise ValueError(f"Too little stabilvol data: {len(stabilvol)} rows")
            indicators = analyst.get_indicators(stabilvol)
            # FHT AVERAGING
            nbins = current_params['nbins']
            # Set maximum volatility to cut the long mfht tail
            max_volatility = indicators['Peak'] + MAX_VOLATILITY_CUT * indicators['FWHM']
            mfht = MeanFirstHittingTimes(stabilvol, nbins=nbins, max_volatility=max_volatility)
            # mfht.plot()
            # Take MFHT indicators
            market_results[params] = mfht.indicators

        market_results = pd.DataFrame.from_dict(market_results, orient='index')
        market_results.to_pickle(DATABASE / f'processed/parameters/{market}_results.pickle')
        print(f"Run saved for {market}")


if __name__ == '__main__':
    main()