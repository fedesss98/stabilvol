"""

Extract Dataset from raw DataFrames by selecting:
- Start and End dates or Duration
- Stocks included in dates window
- Regular Stocks without too much variability
"""
import logging
import pickle
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path

try:
    from data_inspection import Window
    from utility.definitions import ROOT
except ModuleNotFoundError:
    from utility.classes.data_inspection import Window
    from utility.definitions import ROOT

logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)


class DataExtractor:
    def __init__(
            self,
            start_date=None,
            end_date=None,
            duration=None,
            sigma_range=(0.01, 1000),
            criterion='startend',
            criterion_value='5d',
    ):
        """
        Initialize Data Extractor.

        :param str start_date:
        :param str end_date:
        :param int duration: Number of years
        :param list sigma_range: Standard Deviation range for regular stocks
        :param str criterion: 'percentage' or 'startend'
        :param str | int | float criterion_value: threshold for the criterion
        """
        self.start_date, self.end_date, self.duration = self.__check_dates(
            start_date, end_date, duration
        )
        self._sigma_range = sigma_range
        self.min_sigma = sigma_range[0]
        self.max_sigma = sigma_range[-1]
        self._criterion, self._value = self.__check_criterion(criterion, criterion_value)
        self.criterion = {self._criterion: self._value}

        # Initialize internal attributes
        self.data: pd.DataFrame = None
        self.total_std: float = None

        # Print info
        logging.info("DataExtractor created.")

    @staticmethod
    def __check_dates(start, end, duration):
        if start is not None:
            start = pd.Timestamp(start)
        if duration and not end:
            duration = DateOffset(years=duration)
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
        window = Window(start=self.start_date, stop=self.end_date)
        stocks_in_range = window.count_series(df, return_stocks=True, **self.criterion)
        logging.info(f" - {len(stocks_in_range)} stocks selected with criterion {self.criterion}")
        df = df.loc[:, stocks_in_range]
        return df

    def __filter_variability(self, df):
        """
        Select stocks with standard deviation included in sigma-range

        :param pd.DataFrame df: DataFrame of stocks data
        :return: DataFrame with selected stocks
        """
        df = df.loc[:, (df.std() > self.min_sigma) & (df.std() < self.max_sigma)]
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
        self.data = df
        return df


if __name__ == "__main__":
    market = 'JT'
    accountant = DataExtractor(start_date='2002-01-01', duration=12, criterion_value=5)
    data = accountant.extract_data(ROOT / f'data/interim/{market}.pickle')
    assert isinstance(data, pd.DataFrame)
    assert len(data.columns) > 0, "There are no data with selected criterion"
