"""

Make returns from daily stock prices
"""
import logging

import pandas as pd
from utility.definitions import ROOT, MARKETS


class ReturnsMaker:
    def __init__(self, market):
        logging.info("ReturnsMaker created")
        self.market = market
        self.filename = str()
        self.price_data: pd.DataFrame = pd.read_pickle(ROOT / 'data/raw' / f'{market}.pickle')
        self.returns_data: pd.DataFrame = None

    def make_returns(self):
        df = self.price_data.pct_change(fill_method=None)
        self.returns_data = df
        return df

    def save(self, filename=None):
        self.filename = filename if filename else ROOT / 'data/interim' / f'{self.market}.pickle'
        try:
            self.returns_data.to_pickle(self.filename)
            logging.info(f"Returns data saved in {self.filename}")
        except ValueError:
            logging.error("Data cannot be saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for market in MARKETS:
        returns_maker = ReturnsMaker(market)
        returns_maker.make_returns()
        returns_maker.save()
