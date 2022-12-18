"""
Plot stocks count sliding the window
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from stabilvol.utility.definitions import ROOT, MARKETS, FIRST_VALID_DAY, LAST_VALID_DAY


class StocksCount:
    def __init__(self):
        self.data: pd.DataFrame = pd.read_pickle(ROOT / 'data/external/stocks_counts_5dstartend.pickle')
        self.windows_lengths = self.data.columns.unique(level="Window Length")
        self.n_windows = len(self.windows_lengths)

    @property
    def max_values(self):
        return self.data.max()

    def plot_counts_in_window(self, rows=None, cols=None):
        if rows is None:
            rows = int(self.n_windows / 2)
        if cols is None:
            cols = 2
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        for i, ax in enumerate(axs.ravel()):
            data_in_window = self.data.iloc[:, i*5: (i+1)*5].droplevel("Window Length", axis=1)
            data_in_window.dropna().plot(ax=ax,
                                         title=f"{self.windows_lengths[i]} years Window",
                                         colormap="Accent",
                                         xlabel=""
                                         )
            # Add two vertical lines indicating the window containing the WS crash in the middle
            crash_day = pd.Timestamp('2008-09-15')
            left_end = crash_day - pd.Timedelta(int(self.windows_lengths[i])*365/2, unit='D')
            right_end = crash_day + pd.Timedelta(int(self.windows_lengths[i]) * 365 / 2, unit='D')
            ax.axvline(crash_day, ls='--', c='r')
            ax.axvspan(left_end, right_end, alpha=0.1)
            ax.text(crash_day+pd.Timedelta('200d'), 1000, "Crash-centered\nwindow", ha='left', size=9, color='red', alpha=0.5)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
            ax.set_xlim(pd.Timestamp(FIRST_VALID_DAY), pd.Timestamp(LAST_VALID_DAY))
        fig.tight_layout(h_pad=1)
        plt.show()


if __name__ == "__main__":
    plotter = StocksCount()
    plotter.plot_counts_in_window()
