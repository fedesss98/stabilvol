"""

Extract Dataset from raw DataFrames by selecting:
- Start and End dates or Duration
- Stocks included in dates window
- Regular Stocks without too much variability
"""
import logging
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path
import seaborn as sns

try:
    from stabilvol.utility.classes.data_inspection import Window
    from stabilvol.utility.definitions import ROOT, MARKETS_STATS
except ModuleNotFoundError as e:
    logging.warning(f"Error in data_extraction: {e}")
    from data_inspection import Window
    from utility.definitions import ROOT

logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)


class DataExtractor:
    def __init__(
            self,
            start_date: str = None,
            end_date: str = None,
            duration: int = None,
            sigma_range: tuple = (0.001, 1000),
            criterion: str = 'startend',
            criterion_value: str | int | float = '6d',
    ):
        """
        Initialize Data Extractor.

        :param str start_date:
        :param str end_date:
        :param int duration: Number of years
        :param tuple sigma_range: Standard Deviation range for regular stocks
        :param str criterion: 'percentage' or 'startend'
        :param str | int | float criterion_value: threshold for the criterion
        """
        self.start_date, self.end_date, self.duration = self.check_dates(
            start_date, end_date, duration
        )
        self._sigma_range = sigma_range
        self.min_sigma = sigma_range[0]
        self.max_sigma = sigma_range[-1]
        self._criterion, self._value = self.__check_criterion(criterion, criterion_value)

        # Initialize internal attributes
        self.data: pd.DataFrame = None
        self.total_std: float = None

        # Print info
        logging.info("DataExtractor created.")

    @property
    def criterion(self):
        self._criterion, self._value = self.__check_criterion(
            self._criterion, self._value)
        return {self._criterion: self._value}

    @property
    def inputs(self) -> dict:
        inputs_dict = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'duration': self.duration,
            'min_stddev': self.min_sigma,
            'max_stddev': self.max_sigma,
            'selection_criterion': self.criterion,
        }
        return inputs_dict

    @staticmethod
    def check_dates(start, end, duration):
        if start is not None:
            start = pd.Timestamp(start)
        if duration and not end:
            duration = DateOffset(years=int(duration))
            end = start + duration
        elif end and not duration:
            end = pd.Timestamp(end)
            duration = end - start
        elif end and duration:
            print("Both end-date and duration specified:\nEnd-date will be used.")
            end = pd.Timestamp(end)
            duration = end - start
        return start, end, duration

    @staticmethod
    def __check_criterion(criterion: str, value: float | str) -> (str, str | float | int):
        if criterion == 'percentage':
            try:
                value = float(value)
            except ValueError:
                logging.warning("Criterion value's format incorrect. Falling back to default value")
                value = 0.8
        elif criterion == 'startend':
            try:
                int(value)
                # Then the value is in the shape of a number
                value = str(value) + "d"
            except ValueError:
                # Then the value contains a letter
                value = str(value)
        else:
            logging.warning("Criterion not known. Falling back to default value")
            criterion = 'percentage'
            value = 0.8
        return criterion, value

    def __filter_dates(self, df):
        if not self.start_date:
            self.start_date = df.index[0]
        if not self.end_date:
            self.end_date = df.index[-1]
        df = df.loc[self.start_date: self.end_date]
        logging.info(f" - {len(df)} days selected")
        return df

    def __pick_stocks(self, df):
        """
        Creates a Window object.
        Select stocks included in the window between
        start date and end date with chosen criterion

        :param pd.DataFrame df: DataFrame of stocks data
        :return: DataFrame with selected stocks
        """
        self.window = Window(start=self.start_date, stop=self.end_date)
        stocks_in_range = self.window.count_series(df, return_stocks=True, **self.criterion)
        df = df.loc[:, stocks_in_range]
        return df

    def __filter_variability(self, df):
        """
        Select stocks with standard deviation included in sigma-range

        :param pd.DataFrame df: DataFrame of stocks data
        :return: DataFrame with selected stocks
        """
        df = df.loc[:, df.std().between(*self._sigma_range)]
        return df

    def extract_data(self, filename: Path | str) -> pd.DataFrame:
        """
        Extract data from a pickle file

        :param Path | str filename: Path name of the pickle file
        :return: DataFrame with selected data
        """
        logging.info(f"Starting data extraction from {filename}")
        try:
            df: pd.DataFrame = pd.read_pickle(filename)
        except pickle.UnpicklingError as e:
            print("Error reading the file. Be sure that it is in .pickle format.\n", e)
            # Return a null DataFrame
            return pd.DataFrame()
        # Filter dates between start date and end date
        df = self.__filter_dates(df)
        # Select stocks in date range with chosen criterion
        df = self.__pick_stocks(df)
        # Take out stocks with too much variability
        df = self.__filter_variability(df)
        logging.info(f" - {len(df.columns)} stocks selected with criterion {self.criterion}")
        self.data = df
        return df

    def plot_selection(self, edit=False):
        avg_min = self.data.min().median()
        avg_max = self.data.max().median()
        plot_data = self.data.copy()
        plot_data.index = plot_data.index.strftime("%Y-%m-%d")
        fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
        ax.set_title("Selected stocks")
        sns.heatmap(plot_data.T,
                    vmax=avg_max,
                    vmin=avg_min,
                    ax=ax)
        if edit:
            return ax
        else:
            plt.show()

    def plot_returns(self, returns=None, ticker=None):
        if returns is None and ticker is None:
            raise ValueError("Insert something to plot or the ticker of a stock")
        elif ticker is not None:
            returns = self.data[ticker]
        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
        ax.set_title(f"{ticker} returns in window")
        sns.lineplot(returns)
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    market = 'UW'
    start = None
    end = None
    criterion = 'percentage'
    value = .0
    sigma_range = (0, 1e5)
    accountant = DataExtractor(start_date=start,
                               end_date=end,
                               criterion=criterion,
                               criterion_value=value,
                               sigma_range=sigma_range)
    data = accountant.extract_data(ROOT / f'data/interim/{market}.pickle')
    accountant.plot_selection()
    assert isinstance(data, pd.DataFrame)
    assert len(data.columns) > 0, "There are no data with selected criterion"
    assert len(data.columns) == MARKETS_STATS[market][1], "There are less data than original dataframe"
