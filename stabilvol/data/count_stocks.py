"""
Created 2022-11-21

Stocks must be counted on a consistency criterion:
They cannot appear and disappear during the analysis.

To be counted in a temporal window,
a stock must follow one of those criteria:

- Percentage: the percentage of price data in the window
              must be greater than some given percentage
- StartEnd: the price data must start and end within
            a certain number of days from the window
            limits
"""
import click
import pandas as pd
from tqdm import tqdm
import sys

sys.path.extend(['G:\\UNIPA\\ECONOFISICA\\stabilizing_volatility'])

from stabilvol.utility.definitions import ROOT, MARKETS, LAST_VALID_DAY
from stabilvol.utility.classes.data_inspection import Window


def check_inputs(criterion, threshold):
    if criterion == 'percentage':
        threshold = float(threshold)
    elif criterion != 'startend':
        raise ValueError("Method not supported. Use 'percentage' or 'startend'")
    return {criterion: threshold}


def read_data(market):
    df = pd.read_pickle(ROOT / f'data/raw/{market}.pickle')
    return df


def sliding_counting(window, shift='1d', **criterion):
    shift = pd.Timedelta(shift)
    right_limit = pd.Timestamp(LAST_VALID_DAY)
    market_stocks_counts = pd.DataFrame(columns=MARKETS)
    progressbar = tqdm(desc=f'{window.length}yrs Window', total=470)
    while window.right <= right_limit:
        # Create new row of counts inside window
        counts_in_window = pd.DataFrame(index=[window.left])
        for market in MARKETS:
            data = read_data(market)
            num_stocks = window.count_series(data, **criterion)
            counts_in_window[f'{market}'] = num_stocks
        progressbar.update()
        # Update total toll for the considered window
        market_stocks_counts = pd.concat([market_stocks_counts, counts_in_window], copy=False)
        window.slide(shift)
    return market_stocks_counts


def output_results(df):
    print(df.max())
    return None


def save_results(df, method):
    method_name = list(method.keys())[0].replace('.', '')
    method_value = str(list(method.values())[0]).replace('.', '')
    df.to_pickle(ROOT / f'data/external/stocks_counts_{method_value}{method_name}.pickle')
    df.max().to_csv(ROOT / f'data/external/max_counts_{method_value}{method_name}.csv')
    return None


def main(window_lengths, **method):
    total_stocks_count = dict()
    for length in window_lengths:
        window = Window(start='1980-01-01', length=length)
        window_counts = sliding_counting(window, shift='30d', **method)
        total_stocks_count[f'{length}'] = window_counts
    total_stocks_count = pd.concat(
        total_stocks_count.values(), axis=1,
        keys=total_stocks_count.keys(), names=["Window Length", "Market"]
    )
    total_stocks_count.index.name = 'Day'
    save_results(total_stocks_count, method)
    output_results(total_stocks_count)
    return None


@click.command()
@click.argument('window-lengths', type=int, nargs=-1)
@click.option('-c', '--criterion', type=(str, str), multiple=False,
              default=('percentage', '0.6'))
def count_stocks(window_lengths, criterion):
    print(f"{'_'*(25-len('counting stocks'))} "
          f"COUNTING STOCKS "
          f"{'_'*(25-len('counting stocks'))}")
    print(f"Windows lengths: {window_lengths}")
    print(f"Criterion: {criterion[0]}")
    print(f"Threshold: {criterion[1]}")
    criterion = check_inputs(*criterion)
    main(window_lengths, **criterion)


if __name__ == "__main__":
    window_lengths = [5, 10, 15, 20, 30, 40]
    method = {'startend': '5d'}
    count_stocks(window_lengths, **method)
