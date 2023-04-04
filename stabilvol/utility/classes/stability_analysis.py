"""

Class to perform First Hitting Times counting
"""

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import seaborn as sns

from pathlib import Path
import logging
logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)


class StabilVolter:
    def __init__(
            self,
            start_level=-0.1,
            end_level=-1.5,
            divergence_level=100,
            std_normalization=True,
            tau_min=2,
            tau_max=1e4,
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
        # Bins for fht averaging
        self.nbins = 1000

        # Initialize internal attributes
        self.root = Path(__file__).parent.parent.parent.parent
        self.data: pd.DataFrame = None
        self.data_states: pd.DataFrame = None
        self.stabilvol: pd.DataFrame = None
        self.stabilvol_binned = None

        # Print info
        logging.info("StabilVolter created.")

    @property
    def threshold_start(self) -> float:
        thresh = None
        if not self._std_normalize:
            thresh = self._start
        elif not self.total_std:
            logging.warning("Starting threshold cannot be defined without data.")
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

    @property
    def inputs(self) -> dict:
        input_dict = {
            'threshold_start': self._start,
            'threshold_end': self._end,
            'divergence': self._divergence,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'nbins': self.nbins,
        }
        return input_dict

    def count_stock_fht(
            self, series, squeeze=False, threshold_start=None, threshold_end=None, divergence_limit=None
    ):
        """
        Count First Hitting Times of one stock returns series.

        :param pd.Series series:
        :param float threshold_start:
        :param float threshold_end:
        :param float divergence_limit:
        :return:
        """
        start_level = threshold_start if threshold_start is not None else self.threshold_start
        end_level = threshold_end if threshold_end is not None else self.threshold_end
        divergence_limit = divergence_limit if divergence_limit is not None else self.divergence_limit
        # Make series continuous
        if isinstance(series, pd.Series):
            series = series.dropna().values
        elif isinstance(series, np.ndarray):
            series = series[~np.isnan(series)]
        else:
            raise ValueError("Strange series")
        counting = False
        fht = []
        volatility = []
        # Ignore datetime indexes for iteration, use only integers
        for t, level in enumerate(series):
            if not counting and start_level <= level < divergence_limit:
                # Start counting
                counting = True
                start_t = t
            if counting and level < end_level:
                # Stop counting and take FHT
                counting = False
                end_t = t
                counting_time = end_t - start_t
                if self.tau_min <= counting_time <= self.tau_max:
                    # Append FHT and Volatility
                    local_volatility = series[start_t: end_t].std(ddof=1)
                    fht.append(counting_time)
                    volatility.append(local_volatility)
        # Gather data in a DataFrame
        stock_stabilvol = np.array([volatility, fht])
        if squeeze:
            stock_stabilvol = stock_stabilvol.flatten()
        return stock_stabilvol

    def get_stabilvol(self, data=None, method='pandas', **frame_info) -> pd.DataFrame:
        """
        Count First Hitting Times of entire DataFrame.
        Gaps in returns data are dropped.

        :param pd.DataFrame data: DataFrame with stocks data
        :param bool save: Save FHT to file
        :return: Series with FHTs
        """
        logging.info("Starting FHT counting.")
        self.data = data if data is not None else self.data
        if method == 'pandas':
            result = self.data.apply(self.count_stock_fht, squeeze=True)
            result_matrix = [series.reshape(2, -1) for series in result]
            self.stabilvol = pd.DataFrame(np.concatenate(result_matrix, axis=1).T, columns=['Volatility', 'FHT'])
        elif method == 'multi':
            # Multiprocessing method (faster)
            pool = mp.Pool(processes=mp.cpu_count()-1)
            result = pool.map(self.count_stock_fht, [self.data[col].values for col in self.data.columns])
            pool.close()
            self.stabilvol = pd.DataFrame(np.concatenate(result, axis=1).T, columns=['Volatility', 'FHT'])
        else:
            # Numpy method
            np.apply_along_axis(self.count_stock_fht, 0, self.data.values)
        # stabilvol = stabilvol_frame.melt(ignore_index=False, value_name='FHT', var_name='Stock').dropna()
        for info in frame_info.keys():
            self.stabilvol[info] = frame_info[info]
        return self.stabilvol

    def get_average_stabilvol(self, stabilvol=None, nbins=50):
        stabilvol = self.stabilvol if stabilvol is None else stabilvol
        info = [col for col in stabilvol.columns if col not in ['Volatility', 'FHT']]
        self.nbins = nbins
        stabilvol_filtered = stabilvol.loc[stabilvol['Volatility'] <= .5]
        bins = np.linspace(0, .5, num=self.nbins)
        stabilvol_binned = stabilvol_filtered.groupby(pd.cut(stabilvol_filtered['Volatility'], bins=bins)).mean()
        self.stabilvol_binned = pd.DataFrame(stabilvol_binned)
        for i in info:
            self.stabilvol_binned[i] = str(stabilvol[i].unique())
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

    def plot_fht(self, data_to_plot=None, use_ax=None, title=None, plot_indicators=True):
        data_to_plot = data_to_plot if data_to_plot is not None else self.stabilvol
        ax_title = f"Thresholds: [ {self.threshold_start:.4} / {self.threshold_end:.4} ]"
        if use_ax is not None:
            ax = use_ax
            ax.set_title(title if title is not None else ax_title, fontsize=16)
        else:
            # Create a new figure
            fig, ax = plt.subplots(figsize=(10, 6))
            suptitle = "First Hitting Times" if title is None else title
            fig.suptitle(suptitle, fontsize=20)
            ax.set_title(ax_title, fontsize=16)
        sns.scatterplot(data_to_plot,
                        x='Volatility',
                        y='FHT',
                        ax=ax)
        if plot_indicators:
            indicators = self.get_indicators(data_to_plot)
            ax.axhline(y=indicators['Max'], ls='--', c='r')
            ax.axvline(x=indicators['Peak'], ls='--', c='r')
            ax.axvspan(indicators['HM Range'][0], indicators['HM Range'][1], color='blue', alpha=0.2)
        ax.grid()
        plt.tight_layout()
        if use_ax is not None:
            return ax
        else:
            plt.show()
        return None

    def plot_mfht(self, *data_to_plot, title=None, x_range=None, edit=False, ):
        if len(data_to_plot) == 0:
            data_to_plot = [self.stabilvol_binned]
        data_to_plot = [mfht[mfht['Volatility'].between(*x_range)] for mfht in data_to_plot]
        data_to_plot = pd.concat(data_to_plot)
        # fig, ax = plt.subplots(figsize=(10, 6))
        markets = data_to_plot['Market'].unique()
        starts = data_to_plot['Start date'].unique()
        lengths = data_to_plot['Window length'].unique()
        g = sns.relplot(data=data_to_plot,
                        x='Volatility',
                        y='FHT',
                        col='Market',
                        hue='Start date',
                        style='Window length',
                        aspect=1.2,
                        facet_kws={'legend_out': True,
                                   'xlim': x_range,}
                        )
        fig = g.figure
        suptitle = f"MFHT with Thresholds: [ {self._start:.4} / {self._end:.4} ]" if title is None else title
        fig.suptitle(suptitle, fontsize=20)
        axs = g.axes[0]
        for ax, market in zip(axs, markets):
            # ax.set_yscale('log')
            ax.grid()
            max_value = data_to_plot.loc[data_to_plot['Market'] == market]['FHT'].max()
            ax.axhline(y=max_value, color='red', linestyle='--')
            ax.text(x_range[1]+0.01, max_value, f"Max: {max_value:.2f}", c='red')
        g.add_legend()
        g.tight_layout()
        # ax.set_title(f"Thresholds: [ {self.threshold_start:.4} / {self.threshold_end:.4} ]",
        #              fontsize=16)
        # plt.tight_layout()
        if not edit:
            plt.show()
        return ax

    @staticmethod
    def get_indicators(stabilvol: pd.DataFrame):
        stabilvol = stabilvol.set_index('Volatility').sort_index()
        max_value = stabilvol['FHT'].max()
        peak_position = stabilvol['FHT'].idxmax()
        half_max_values = stabilvol.loc[stabilvol['FHT'] >= max_value/2]
        fwhm = half_max_values.index[-1] - half_max_values.index[0]
        indicators = {
            'Max': max_value,
            'Peak': peak_position,
            'FWHM': fwhm,
            'HM Range': (half_max_values.index[0], half_max_values.index[-1]),
        }
        return indicators

    def save_fht(self, data_to_save=None, market=None, filename=None, format='pickle', *args):
        data_to_save = self.stabilvol if data_to_save is None else data_to_save
        if not market and not filename:
            raise ValueError("Specify a market or a filename to save the data.")
        start_level_string = str(self._start).replace('.', 'p').replace('-', 'n')
        end_level_string = str(self._end).replace('.', 'p').replace('-', 'n')
        if filename is None:
            filename = f'{market}_{start_level_string}_{end_level_string}_{"_".join(args)}'
        if format == 'pickle':
            data_to_save.to_pickle(self.root / f"data/processed/fht/{filename}.pickle")
        elif format == 'csv':
            data_to_save.to_pickle(self.root / f"data/processed/fht/{filename}.csv")
        else:
            raise ValueError("File format for saving unknown. Try 'csv' or 'pickle'")
        return None

    def save_mfht(self, data_to_save=None, market=None, filename=None, format='pickle', *args):
        data_to_save = self.stabilvol_binned if data_to_save is None else data_to_save
        if not market and not filename:
            raise ValueError("Specify a market or a filename to save the data.")
        # Remove punctuation for use in filename
        start_level_string = str(self._start).replace('.', 'p').replace('-', 'n')
        end_level_string = str(self._end).replace('.', 'p').replace('-', 'n')
        if filename is None:
            filename = f'{market}_{start_level_string}_{end_level_string}_{self.nbins}_{"_".join(args)}'
        if format == 'pickle':
            data_to_save.to_pickle(self.root / f"data/processed/mfht/{filename}.pickle")
        elif format == 'csv':
            data_to_save.to_csv(self.root / f"data/processed/mfht/{filename}.csv")
        else:
            raise ValueError("File format for saving unknown. Try 'csv' or 'pickle'")
        return None


class MeanFirstHittingTimes:
    def __init__(self, data, nbins=10001, max_volatility=0.5, **metadata):
        logging.info("Mean First Hitting times initialized")
        self.mfht = pd.Series(dtype=float)
        self.nbins = nbins
        self.max_volatility = max_volatility
        self.raw_stabilvol = data
        self.make_average_stabilvol()

    @property
    def raw_stabilvol(self):
        return self._raw_stabilvol

    @raw_stabilvol.setter
    def raw_stabilvol(self, stabilvol):
        if isinstance(stabilvol, pd.DataFrame):
            fht = stabilvol['FHT'].values
            v = stabilvol['Volatility'].values
        elif isinstance(stabilvol, pd.Series):
            fht = stabilvol.values
            v = stabilvol.index.values
        elif isinstance(stabilvol, np.ndarray):
            fht = stabilvol[:, 1]
            v = stabilvol[:, 0]
        else:
            raise ValueError("Stabilvol data has incorrect format. "
                             "Try again with a Pandas Series or DataFrame or Numpy Array")
        self._raw_stabilvol = pd.Series(data=fht, index=v)
        return pd.Series(data=fht, index=v)

    @property
    def data(self):
        return self.mfht

    @property
    def bins(self):
        return np.linspace(0, .5, num=self.nbins)

    @property
    def max_value(self):
        return self.mfht.max()

    @property
    def peak_position(self):
        return self.mfht.idxmax()

    @property
    def fwhm(self):
        half_max_values = self.mfht.loc[self.mfht >= self.max_value/2]
        return half_max_values.index[-1] - half_max_values.index[0]

    @property
    def indicators(self):
        baricenters = self.baricenters
        # Baricenters widths
        first_baricenter = baricenters[0].index[-1] - baricenters[0].index[0]
        second_baricenter = baricenters[1].index[-1] - baricenters[1].index[0]
        third_baricenter = baricenters[2].index[-1] - baricenters[2].index[0]
        # Indicators width
        indicators = {
            'Max': self.max_value,
            'Peak': self.peak_position,
            'FWHM': self.fwhm,
            'First Baricenter': first_baricenter,
            'Second Baricenter': second_baricenter,
            'Third Baricenter': third_baricenter,
        }
        return indicators

    @property
    def values(self, mode='pandas'):
        return self.mfht

    @property
    def baricenters(self):
        """
        Characterize points in the peak:
        Those are high and near the peak.
        :return:
        """
        baricenters = []
        # Consider the series without peak point
        mfht = self.mfht.drop(self.peak_position).dropna()
        # peak_index = np.where(self.mfht == self.max_value)[0][0]
        distances = np.array(abs(mfht.index - self.peak_position).values)
        weights = mfht / distances
        weights.sort_values(ascending=False, inplace=True)
        total_weight = np.sum(weights)
        for fraction in [0.5, 0.9, 0.95]:
            min_weight = fraction * total_weight
            accumulated_w = 0
            i = 0
            while accumulated_w <= min_weight:
                # Sum the biggest weights
                accumulated_w += weights.iloc[i]
                i += 1
            # Take the values of volatility until index
            baricenter = weights.iloc[: i]
            baricenters.append(mfht.loc[baricenter.index].sort_index())
        return baricenters

    @staticmethod
    def filter_stabilvol(stabilvol, max_volatility):
        """ Returns the stabilvol filtered by maximum volatility """
        volatility = stabilvol.index
        return stabilvol[volatility <= max_volatility]

    @staticmethod
    def find_outliers(series):
        outliers = []
        if len(series) > 1:
            for i, fht in series['FHT'].items():
                distance = abs(fht - series['FHT'].mean())
                if distance > 6 * series['FHT'].std():
                    outliers.append(i)
        return outliers

    @staticmethod
    def classify_inliers(series, std_range=6):
        fhts = series['FHT']
        distance = abs(fhts - fhts.mean())
        std = fhts.std() if len(fhts) > 1 else 0
        return distance <= std * std_range

    def plot_baricenter(self, baricenter):
        fig, ax = plt.subplots()
        self.mfht.reset_index().plot(ax=ax, x='Volatility', y='FHT', c='b', kind='scatter')
        self.mfht.loc[baricenter.index].reset_index().plot(ax=ax, x='Volatility', y='FHT', c='r', kind='scatter')
        plt.show()

    def smooth_stabilvol(self, stabilvol, bins):
        # Create a DataFrame from the Series with names FHT and Volatility
        stabilvol = stabilvol.to_frame(name='FHT').sort_index().reset_index(names='Volatility')
        stabilvol['ranges'] = pd.cut(stabilvol['Volatility'], bins=bins, include_lowest=True)
        # Classify inliers that are inside 6 standard deviations
        inliers = stabilvol.groupby('ranges').apply(self.classify_inliers, std_range=2)
        stabilvol_smoothed = stabilvol.loc[inliers.values]
        return stabilvol_smoothed

    def make_average_stabilvol(self):
        filtered = self.filter_stabilvol(self.raw_stabilvol, self.max_volatility)
        smoothed = self.smooth_stabilvol(filtered, self.bins)
        stabilvol_binned = smoothed.groupby('ranges').mean().set_index('Volatility')
        self.mfht = stabilvol_binned.squeeze()

    def plot(self, ax=None, edit=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle('Mean First Hitting Times')
        ax.set_title(f'{self.nbins} bins')
        self.mfht.reset_index().plot(x='Volatility', y='FHT', kind='scatter', ax=ax)
        ax.axhline(y=self.max_value, ls='--', c='r')
        ax.axvline(x=self.peak_position, ls='--', c='r')
        # Second baricenter area
        baricenters = self.baricenters
        ax.axvspan(baricenters[1].index[0], baricenters[1].index[-1], color='r', alpha=0.2)
        ax.axvspan(baricenters[1].index[0], baricenters[1].index[-1], color='g', alpha=0.2)
        ax.axvspan(baricenters[1].index[0], baricenters[1].index[-1], color='b', alpha=0.2)
        ax.scatter(x=self.peak_position, y=self.max_value, c='r')
        if not edit:
            plt.show()
            return None
        else:
            return ax


if __name__ == "__main__":
    Y1 = np.array([1, -1, 0, 1, 0, -1, 1, -1, 0, -1])
    Y2 = np.array([1, 0, 0, 0, -1, -1, 1, 0, 0, -1])
    Y3 = np.array([1, 0, 0, 0, -1, -1, 1, 0, 0, -1])
    Y4 = np.array([1, np.nan, np.nan, 0, 0, -1, 1, -1, 0, -1])
    data = pd.DataFrame(np.vstack((Y1, Y2, Y3, Y4)).T, columns=['y1', 'y2', 'y3', 'y4'])
    start_level = 0.5
    end_level = -0.5
    analyst = StabilVolter(start_level=start_level, end_level=end_level)
    analyst.data = data
    # Check last stabilvol
    # stabilvol1 = analyst.count_stock_fht(data.iloc[:, 2])
    stabilvol = analyst.get_stabilvol(data)
    given_stabilvol = np.array([
        [np.sqrt(2), 2],
        [np.sqrt(2 / 2), 3],
        [np.sqrt(2), 2],
        [np.sqrt(2 / 4), 5],
        [np.sqrt(2 / 3), 4],
        [np.sqrt(2 / 3), 4],
        [np.sqrt(2 / 1), 2]
    ])
    analyst.plot_fht()
    assert np.array_equal(stabilvol.values, given_stabilvol), "Stabilvol incorrect."
