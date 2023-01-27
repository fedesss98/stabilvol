"""

Count FHT
"""
import click
import matplotlib.pyplot as plt
import pandas as pd
import time

try:
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter
except ModuleNotFoundError:
    print('Not found')
    from ..utility.classes.data_extraction import DataExtractor
    from ..utility.classes.stability_analysis import StabilVolter

from utility.definitions import ROOT, MARKETS, FIRST_VALID_DAY, LAST_VALID_DAY


def main(markets, start, end, duration, nbins):
    for market in markets:
        # The duration is used by default to extract data
        if duration is None:
            # Take the duration in years from start to end
            duration = len(pd.date_range(start=start, end=end, freq='Y'))
        accountant = DataExtractor(start_date=start, duration=duration)
        data = accountant.extract_data(ROOT / f'data/interim/{market}.pickle')
        accountant.plot_selection()
        analyst = StabilVolter()
        stabilvol = analyst.get_stabilvol(data)
        mfht = analyst.get_average_stabilvol(nbins=nbins)
        # Save
        analyst.save_fht(market)
        analyst.save_mfht(market)
        # Plot
        analyst.plot_fht()
        ax = analyst.plot_mfht(edit=True)
        ax.set_xlim(0, 0.15)
        plt.show()
    return None


@click.command()
@click.option('-m', '--markets', multiple=True, default=MARKETS)
@click.option('-s', '--start', type=str, default=FIRST_VALID_DAY)
@click.option('-e', '--end', type=str, default=LAST_VALID_DAY)
@click.option('-d', '--duration', type=int, default=None)
@click.option('-n', '--nbins', type=int, default=2000)
def count_fht(markets, start, end, duration, nbins):
    main(markets, start, end, duration, nbins)
    return None


if __name__ == "__main__":
    markets = MARKETS
    start_date = '2002-01-01'
    end_date = None
    duration = 12
    nbins = 3000
    start_time = time.time()  # to track execution time
    main(markets, start_date, end_date, duration, nbins)
    end_time = time.time()
    print(f"\n\n***** Execution time for {len(markets)} markets: {end_time - start_time}s ***** ")
