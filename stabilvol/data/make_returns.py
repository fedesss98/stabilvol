"""

Make returns from daily stock prices
"""
import logging

import pandas as pd
import numpy as np
from stabilvol.utility.definitions import MARKETS
from stabilvol import ROOT


class ReturnsMaker:
    def __init__(self, market, method='pct_change'):
        logging.info("ReturnsMaker created")
        self.market = market
        self.filename = str()
        self.price_data: pd.DataFrame = pd.read_pickle(ROOT / 'data/raw' / f'{market}.pickle')
        self.returns_data: pd.DataFrame = None
        self.method = method

    def make_returns(self):
        if self.method == 'pct_change':
            df = self.price_data.pct_change(fill_method=None)
        elif self.method == 'log':
            df = self.price_data.apply(lambda x: np.log(x) - np.log(x.shift(1)))
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self.returns_data = df
        return df

    def save(self, filename=None):
        if self.method == 'pct_change':
            self.filename = filename if filename else ROOT / 'data/interim' / f'{self.market}.pickle'
        elif self.method == 'log':
            self.filename = filename if filename else ROOT / 'data/interim' / f'{self.market}_log.pickle'
        try:
            self.returns_data.to_pickle(self.filename)
            logging.info(f"Returns data saved in {self.filename}")
        except ValueError:
            logging.error("Data cannot be saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for market in MARKETS:
        returns_maker = ReturnsMaker(market, method='log')
        returns_maker.make_returns()
        returns_maker.save()
