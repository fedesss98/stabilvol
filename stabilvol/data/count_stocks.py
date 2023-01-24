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
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
try:
    from ..utility.classes.data_inspection import Window
    from ..utility.definitions import ROOT, MARKETS, FIRST_VALID_DAY, LAST_VALID_DAY
except ImportError:
    from stabilvol.utility.classes.data_inspection import Window
    from stabilvol.utility.definitions import ROOT, MARKETS, FIRST_VALID_DAY, LAST_VALID_DAY


def check_inputs(criterion, threshold):
    if criterion == 'percentage':
        threshold = float(threshold)
    elif criterion != 'startend':
        raise ValueError("Method not supported. Use 'percentage' or 'startend'")
    return {criterion: threshold}


def read_data(market):
    df = pd.read_pickle(ROOT / f'data/raw/{market}.pickle')
    return df


def sliding_counting(markets, window, end, shift='1d', **criterion):
    shift = pd.Timedelta(shift)
    right_limit = pd.Timestamp(end)
    market_stocks_counts = pd.DataFrame(columns=markets)
    progressbar = tqdm(desc=f'{window.length}yrs Window', total=470)
    while window.right <= right_limit:
        # Create new row of counts inside window
        counts_in_window = pd.DataFrame(index=[window.left])
        for market in markets:
            data = read_data(market)
            num_stocks = window.count_series(data, **criterion)
            counts_in_window[f'{market}'] = num_stocks
        progressbar.update()
        # Update total toll for the considered window
        market_stocks_counts = pd.concat([market_stocks_counts, counts_in_window], copy=False)
        window.slide(shift)
    return market_stocks_counts


def single_counting(markets, window, **criterion):
    market_stocks_counts = pd.DataFrame(columns=markets, index=[window.left])
    for market in markets:
        data = read_data(market)
        num_stocks = window.count_series(data, **criterion)
        market_stocks_counts[f'{market}'] = num_stocks
    return market_stocks_counts


def output_results(df, sliding):
    print(df.max())
    if sliding:
        df.plot()
        plt.show()
    return None


def save_results(df, method):
    method_name = list(method.keys())[0]
    method_value = str(list(method.values())[0]).replace('.', '')
    df.to_pickle(ROOT / f'data/external/stocks_counts_{method_value}{method_name}.pickle')
    df.max().to_csv(ROOT / f'data/external/max_counts_{method_value}{method_name}.csv')
    return None


def main(window_lengths: list,
         start: str,
         sliding: bool,
         end: str,
         save: bool,
         markets: list,
         criterion: dict) -> None:
    """
    Count stocks in window based on the criterion
    :param list window_lengths: lengths of the windows to use
    :param str start: start date for the windows
    :param bool sliding: slide the windows to the end of the data or not
    :param str end: right limit for the sliding window
    :param bool save: save results
    :param list markets: markets to analyze
    :param dict criterion: ['percentage' / 'startend'] criterion to use and threshold value
    :return: None
    """
    total_stocks_count = dict()
    windows = [Window(start=start, length=int(length)) for length in window_lengths]
    pool = mp.Pool(mp.cpu_count())
    for length in window_lengths:
        window = Window(start=start, length=int(length))
        if sliding:
            window_counts = sliding_counting(markets, window, end, shift='30d', **criterion)
        else:
            window_counts = single_counting(markets, window, **criterion)
        total_stocks_count[f'{length}'] = window_counts
    total_stocks_count = pd.concat(
        total_stocks_count.values(), axis=1,
        keys=total_stocks_count.keys(), names=["Window Length", "Market"]
    )
    total_stocks_count.index.name = 'Day'
    if save:
        save_results(total_stocks_count, criterion)
    output_results(total_stocks_count, sliding)
    return total_stocks_count


@click.command()
@click.argument('window-lengths', type=int, nargs=-1)
@click.option('-s', '--start', type=str, default='1980-01-01')
@click.option('-e', '--end', type=str, default=LAST_VALID_DAY)
@click.option('-c', '--criterion', type=(str, str), multiple=False,
              default=('startend', '6d'))
@click.option('--slide', is_flag=True)
@click.option('--save/--no-save', default=True)
@click.option('-m', '--markets', multiple=True, default=MARKETS)
def count_stocks(window_lengths, start, slide, end, save, markets, criterion):
    print(f"{'_'*(25-len('counting stocks'))} "
          f"COUNTING STOCKS"
          f"{'_'*(25-len('counting stocks'))}")
    print(f"Windows lengths: {window_lengths}")
    print(f"Criterion: {criterion[0]}")
    print(f"Threshold: {criterion[1]}")
    main(markets, window_lengths, start, slide, end, save, criterion)


if __name__ == "__main__":
    window_lengths = [12, ]
    criterion = {'startend': '6d'}
    slide = True
    start = FIRST_VALID_DAY
    end = LAST_VALID_DAY
    markets = MARKETS
    save = False
    main(window_lengths, start, slide, end, save, markets, criterion)
