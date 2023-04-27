"""
Created by Federico Amato
2023 - 04 - 20

Visualize how indicators vary with the choice of different parameters
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATABASE = Path('../data/processed/parameters')
MARKET = 'UN'

INDICATORS = [
    'Max', 'Peak', 'FWHM', 'First Baricenter', 'Second Baricenter', 'Third Baricenter'
]


def print_parameters(df):
    parameters = list()
    for level in range(df.index.nlevels):
        unique_values = df.index.get_level_values(level).unique()
        parameters.append(unique_values)
        print(unique_values)
    return parameters


def plot_indicator_variability(df, indicator):
    g = sns.catplot(df,
                    x='Nbins',
                    y=indicator,
                    col='Depth',
                    row='Start',
                    hue='Criterion',
                    kind='bar',
                    palette='bright'
                    )
    g.fig.suptitle(indicator)
    g.fig.show()


def main():
    df: pd.DataFrame = pd.read_pickle(DATABASE / f'{MARKET}_results.pickle')
    idx = pd.IndexSlice

    df.index = df.index.set_names(['Nbins', 'Start', 'Depth', 'Tau', 'Criterion'])
    df = df.reset_index()

    sns.set(font_scale=1.5)
    for indicator in INDICATORS:
        plot_indicator_variability(df, indicator)

    # df_nbins = df.loc[idx[:, -0.1, 1.5, 1e8, 'SE7D'], :]
    # df_nbins.plot(rot=60, figsize=(14, 12), subplots=True)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
