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


def extract_raw_data(market):
    raw_data = ROOT / f'data/external/{market}.csv'
    df = pd.read_csv(raw_data,
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
    return df


def remove_holidays(df):
    """
    Select only business days in dataframe dates range
    :param df: DataFrame
    :return: df: DataFrame without holidays
    """
    # Select business days in the dataframe
    business_days = pd.bdate_range(df.index[0], df.index[-1])
    df = df.loc[business_days]
    return df


def main():
    for market in MARKETS:
        data = extract_raw_data(market)
        data = remove_holidays(data)
        pickle_file = ROOT / f'data/raw/{market}.pickle'
        print(f'Saving {market}.csv to {pickle_file}')
        data.to_pickle(pickle_file)
    return data


@click.command()
def make_pickles():
    print(f"{'_'*(25-len('counting stocks'))}"
          f" EXTRACTING AND PROCESSING FILES "
          f"{'_'*(25-len('counting stocks'))}")
    main()


if __name__ == "__main__":
    main()
