"""
Created on 2022 - 11 - 10

Read CSV with pandas, process them and make a pickle file
(faster and lighter).

PROCESSING:
    - Clean stocks ticker removing 'Market Equity' string,
    - Remove holidays.
"""
import click
import pandas as pd
from stabilvol.utility.definitions import ROOT, MARKETS


class Database:
    def __init__(self, folder, market):
        df = pd.read_csv(f'{folder}/{market}.csv',
                         index_col=0,
                         skiprows=[1],
                         sep=';',
                         decimal=',',
                         parse_dates=True,
                         infer_datetime_format=True,
                         na_values=['#N/D'])
        # Rename columns and index
        df.rename(columns=lambda x: x.replace(f' {market} Equity', ''), inplace=True)
        df.index.name = 'Day'
        # Order columns by number of values
        ordered_columns_list = df.isnull().sum().sort_values().index
        df = df.loc[:, ordered_columns_list]
        self.data = df
        self.market = market

    def remove_holidays(self):
        """
        Select only business days in dataframe dates range
        """
        # Select business days in the dataframe
        business_days = pd.bdate_range(self.data.index[0], self.data.index[-1])
        self.data = self.data.loc[business_days]

    def save(self, folder):
        print(f'Saving {self.market}.csv to {folder}')
        self.data.to_pickle(f'{folder}/{self.market}.pickle')


def main():
    raw_folder = ROOT / 'data/external'
    pickle_folder = ROOT / 'data/raw'
    for market in MARKETS:
        database = Database(raw_folder, market)
        database.remove_holidays()
        database.save(pickle_folder)
    return None


@click.command()
def make_pickles():
    print(f"{'_'*(25-len('counting stocks'))}"
          f" EXTRACTING AND PROCESSING FILES "
          f"{'_'*(25-len('counting stocks'))}")
    main()


if __name__ == "__main__":
    main()
