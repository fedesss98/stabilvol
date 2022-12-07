"""
Classes to Preliminary Data Inspection.
"""
import pandas as pd
from pandas.tseries.offsets import DateOffset
import logging


class Window:
    def __init__(self, length, start):
        """
        The Window is an Interval of dates of fixed length.
        Create it specifying the length and the initial date.
        :param length: [integer] number of years.
        :param start: [string] or [datetime] initial date.
        """
        window_start = pd.Timestamp(start)
        window_stop = window_start + DateOffset(years=length)
        self._length = length
        self.interval = pd.Interval(window_start, window_stop)
        self.stocks_inside = None

    @property
    def left(self):
        return self.interval.left

    @property
    def right(self):
        return self.interval.right

    @property
    def length(self):
        return self._length

    def __repr__(self):
        return f'Window: start={self.left} - end:{self.right} (length: {self._length}))'

    def slide(self, step, limit=None):
        step = self.check_step(step, limit)
        right_end = self.right + step
        left_end = self.left + step
        # Update the interval
        self.interval = pd.Interval(left_end, right_end)

    def cut_series(self, df):
        return df.loc[self.left: self.right]

    def count_series(self, df, **method):
        series = self.cut_series(df)
        if 'percentage' in method:
            selected_stocks = self.percent_selection(
                series, method.get('percentage')
            )
        elif 'startend' in method:
            selected_stocks = self.startend_selection(
                series, self.left, self.right, method.get('startend')
            )
        self.stocks_inside = selected_stocks
        return len(selected_stocks)

    @staticmethod
    def percent_selection(df, threshold):
        # Select and count stocks with more data in slide than the threshold
        percentages_series = df.count() / len(df)
        selected_stocks = df.loc[:, percentages_series > threshold].columns
        return selected_stocks.values

    @staticmethod
    def startend_selection(df, left, right, threshold):
        try:
            left_distance = df.apply(pd.Series.first_valid_index) - left
            right_distance = right - df.apply(pd.Series.last_valid_index)
        except TypeError as e:
            logging.warning(f"Window ({left}, {right}] outside borders")
            # There can be no stocks here, return empty list
            return []
        else:
            selected_stocks = df.loc[:, (left_distance < threshold) & (right_distance < threshold)].columns
            return selected_stocks.values

    @staticmethod
    def check_count_method(method):
        if method not in ['percentage', 'startend']:
            raise Exception('Counting Method Unknown: try "percentage" or "startend"')
        else:
            return method

    def check_threshold(self, threshold):
        if self._count_method == 'percentage' and not isinstance(threshold, float):
            raise ValueError("Use a float for 'percentage' threshold")
        elif self._count_method == 'percentage' and not isinstance(threshold, str):
            raise ValueError("Use a string for 'startend' threshold")
        else:
            return threshold

    def check_step(self, step, limit):
        if not isinstance(step, pd.Timedelta):
            step = pd.Timedelta(step)
        if limit is not None:
            if self.right + step > limit:
                step = limit - self.right
        return step


if __name__ == "__main__":
    from stabilvol.utility.definitions import ROOT
    market = 'GF'
    data = pd.read_pickle(ROOT/f'data/raw/{market}.pickle')
    window = Window(start=data.first_valid_index(), length=10)
    stocks_in_window = window.count_series(data, percentage=-1)
    assert len(data.columns) == stocks_in_window
