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
            self, series, threshold_start=None, threshold_end=None, divergence_limit=None
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
            data.apply(self.count_stock_fht)
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

    def plot_fht(self, data_to_plot=None, title=None):
        data_to_plot = data_to_plot if data_to_plot is not None else self.stabilvol
        fig, ax = plt.subplots(figsize=(10, 6))
        suptitle = "First Hitting Times" if title is None else title
        fig.suptitle(suptitle, fontsize=20)
        ax.set_title(f"Thresholds: [ {self.threshold_start:.4} / {self.threshold_end:.4} ]",
                     fontsize=16)
        sns.scatterplot(data_to_plot,
                        x='Volatility',
                        y='FHT')
        ax.set_xlim(0, 0.1)
        ax.grid()
        plt.tight_layout()
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


if __name__ == "__main__":
    Y1 = np.array([1, -1, 0, 1, 0, -1, 1, -1, 0, -1])
    Y2 = np.array([1, 0, 0, 0, -1, -1, 1, 0, 0, -1])
    Y3 = np.array([1, np.nan, np.nan, 0, 0, -1, 1, -1, 0, -1])
    data = pd.DataFrame(np.vstack((Y1, Y2, Y3)).T, columns=['y1', 'y2', 'y3'])
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
