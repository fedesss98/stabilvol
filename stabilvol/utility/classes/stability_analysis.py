"""

Class to perform First Hitting Times counting
"""
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)


class StabilVolter:
    def __init__(
            self,
            start_level=-0.1,
            end_level=-1.5,
            divergence_level=100,
            std_normalization=True,
            tau_min=2,
            tau_max=700,
    ):
        """
        Creates StabilVolter object with FHT analysis parameters.

        :param float start_level:
        :param float end_level:
        :param float divergence_level:
        :param bool std_normalization:
        :param int tau_min:
        :param int tau_max:
        """
        self._start = start_level
        self._end = end_level
        self._divergence = divergence_level
        self._std_normalize = std_normalization
        # Counting times limits
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Initialize internal attributes
        self.data: pd.DataFrame = None
        self.data_states: pd.DataFrame = None
        self.stabilvol: pd.DataFrame = None
        self.nbins = None

    @property
    def threshold_start(self) -> float:
        thresh = None
        if not self._std_normalize:
            thresh = self._start
        elif not self.total_std:
            logging.info("Starting threshold cannot be defined without data.")
        else:
            thresh = self._start * self.total_std
        return thresh

    @property
    def threshold_end(self) -> float:
        thresh = None
        if not self._std_normalize:
            thresh = self._end
        elif not self.total_std:
            logging.warning("Ending threshold cannot be defined without data.")
        else:
            thresh = self._end * self.total_std
        return thresh

    @property
    def divergence_limit(self) -> float:
        limit = None
        if not self._std_normalize:
            limit = self._divergence
        elif not self.total_std:
            logging.warning("Divergence limit cannot be defined without data.")
        else:
            limit = self._divergence * self.total_std
        return limit

    @property
    def total_std(self) -> float:
        total_std = None
        if self.data is None:
            logging.warning("Total std cannot be defined without data")
        else:
            total_std = self.data.std().sum() / len(self.data.std())
        return total_std

    @staticmethod
    def _set_stock_states(series, start, end):
        """
        Return the state of one return,
        +1 if it is over the starting threshold,
        0 if it is between the thresholds,
        -1 if it is under the ending threshold

        :param x: value of a return
        :param start: starting threshold
        :param end: ending threshold
        :return: state of the value [+1, -1 or 0]
        """
        series_iterator = np.nditer(series, flags=['c_index'])
        X = np.zeros(series.shape, dtype=np.int8)
        counting = False
        for x in series_iterator:
            i = series_iterator.index
            if x > start and not counting:
                X[i] = 1
                counting = True
            elif x < end and counting:
                X[i] = -1
                counting = False
        return X

    def _make_states(self):
        """
        Transform returns DataFrame in states DataFrame,
        where every state indicate if the return is over,
        under, or between the thresholds.

        :return: None
        """
        states = self.data.apply(
            self._set_stock_states, start=self.threshold_start, end=self.threshold_end
        )
        self.data_states = states.copy()
        return None

    @staticmethod
    def _select_date_ranges(series, start, end):
        """
        Select start and end date where returns change
        from the starting threshold to the ending threshold

        :param series: Returns series
        :param start: Starting Threshold
        :param end: Ending Threshold
        :return: date_ranges: list of tuples with (start_date, end_date)
        """
        date_ranges = []
        counting = False
        for i, x in series.items():
            if x > start and not counting:
                start_index = i
                counting = True
            elif x < end and counting:
                end_index = i
                date_ranges.append((start_index, end_index))
                counting = False
        return date_ranges

    def _make_date_ranges(self):
        """
        Makes a DataFrame with date ranges for every stock.

        :return: date_ranges: DataFrame
        """
        date_ranges = self.data.apply(
            self._select_date_ranges, start=self.threshold_start, end=self.threshold_end
        )
        self.date_ranges = date_ranges.copy()
        return date_ranges


    def count_stock_fht(
            self, series, threshold_start=None, threshold_end=None, divergence_limit=None
    ):
        """
        Count First Hitting Times of one stock prices series.

        :param pd.Series series:
        :param float threshold_start:
        :param float threshold_end:
        :param float divergence_limit:
        :return:
        """
        start = threshold_start if threshold_start else self.threshold_start
        end = threshold_end if threshold_end else self.threshold_end
        divergence = divergence_limit if divergence_limit else self.divergence_limit
        try:
            date_ranges = self.date_ranges[series.name]
        except IndexError:
            date_ranges = self._select_date_ranges(series, start=threshold_start, end=threshold_end)
        stabilvol_list = list()
        for interval in date_ranges[series.name]:
            chunck = series.loc[interval[0]: interval[1]]
            volatility = chunck.std()
            fht = len(chunck)
            stabilvol_list.append((volatility, fht))
        self.stabilvol = pd.DataFrame.from_records(
            stabilvol_list, columns=['Volatility', 'FHT']
        )
        return self.stabilvol

    def get_stabilvol(self, data=None) -> pd.Series:
        """
        Count First Hitting Times of entire DataFrame.

        :param pd.DataFrame data: DataFrame with stocks data
        :return: Series with FHTs
        """
        if data is not None:
            self.data = data
        # Take date ranges for starting and ending counts
        date_ranges = self._make_date_ranges()
        analyzed_stocks = list()
        stabilvol_list = list()
        for stock, series in self.data.items():
            analyzed_stocks.append(stock)
            for interval in date_ranges[stock]:
                chunk = series.loc[interval[0]: interval[1]]
                volatility = chunk.std()
                fht = len(chunk)
                stabilvol_list.append((volatility, fht))
        self.stabilvol = pd.DataFrame.from_records(
            stabilvol_list, columns=['Volatility', 'FHT']
        )
        logging.info(f"{len(analyzed_stocks)} Stocks analyzed.")
        return self.stabilvol

    def get_average_stabilvol(self, nbins=50):
        nbins = self.nbins if self.nbins is not None else nbins
        volatility = self.stabilvol['Volatility']
        bins = np.linspace(0, volatility.max(), num=nbins)
        stabilvol_binned = self.stabilvol.groupby(pd.cut(volatility, bins=bins)).mean()
        self.stabilvol_binned = pd.DataFrame(stabilvol_binned)
        return stabilvol_binned

    def _plot_states(self):
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(self.data_states.T,
                    cmap='coolwarm',
                    ax=ax)
        fig.suptitle(f"Returns States",
                     fontsize=24)
        ax.set_title(f"Thresholds: [ {self.threshold_start:.4} / {self.threshold_end:.4} ]",
                     fontsize=16)
        ax.set_ylabel('Stocks', fontsize=24)
        ax.set_xlabel('Days', fontsize=24)
        return ax

    def plot_fht(self, title=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        suptitle = "First Hitting Times" if title is None else title
        fig.suptitle(suptitle, fontsize=20)
        ax.set_title(f"Thresholds: [ {self.threshold_start:.4} / {self.threshold_end:.4} ]",
                     fontsize=16)
        sns.scatterplot(self.stabilvol,
                        x='Volatility',
                        y='FHT')
        ax.set_yscale('log')
        ax.grid()
        plt.tight_layout()
        plt.show()
        return ax

    def plot_mfht(self, title=None, edit=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        suptitle = "Mean First Hitting Times" if title is None else title
        fig.suptitle(suptitle, fontsize=20)
        ax.set_title(f"Thresholds: [ {self.threshold_start:.4} / {self.threshold_end:.4} ]",
                     fontsize=16)
        sns.scatterplot(self.stabilvol_binned,
                        x='Volatility',
                        y='FHT')
        ax.set_yscale('log')
        ax.grid()
        plt.tight_layout()
        if not edit:
            plt.show()
        return ax


if __name__ == "__main__":
    from utility.classes.data_extraction import DataExtractor
    from utility.classes.stability_analysis import StabilVolter
    from utility.definitions import ROOT

    accountant = DataExtractor(start_date='2002-01-01', duration=12)
    data = accountant.extract_data(ROOT / 'data/interim/GF.pickle')
    analyst = StabilVolter(end_level=-1)
    stabilvol = analyst.get_stabilvol(data)
    analyst.plot_fht()
    mfht = analyst.get_average_stabilvol(nbins=35000)
    ax = analyst.plot_mfht(edit=True)
    ax.set_xlim(0, 0.15)
    plt.show()
