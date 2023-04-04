"""

Count FHT
"""
import click
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging

try:
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes
    from stabilvol.utility.definitions import MARKETS, FIRST_VALID_DAY, LAST_VALID_DAY
except ModuleNotFoundError as e:
    logging.warning(f"Error in count_fht: {e}")
    from utility.classes.data_extraction import DataExtractor
    from utility.classes.stability_analysis import StabilVolter
    from utility.definitions import MARKETS, FIRST_VALID_DAY, LAST_VALID_DAY

from pathlib import Path
DATABASE = Path(__file__).parent.parent.parent / 'data'


def main(markets,
         start_date='2002-01-01', end_date='2014-01-01', duration=12,
         criterion='startend', criterion_value='6d',
         start_level=-0.1, end_level=-1.5, tau_min=2, tau_max=10000,
         nbins=3000, save=False):
    if duration is None:
        # Take the duration in years from start to end
        duration = len(pd.date_range(start=start_date, end=end_date, freq='Y'))
    if not isinstance(duration, list):
        duration = [duration]
    for years in duration:
        for market in markets:
            # The duration is used by default to extract data
            accountant = DataExtractor(
                start_date=start_date, duration=int(years), criterion=criterion, criterion_value=criterion_value
            )
            data = accountant.extract_data(DATABASE / f'interim/{market}.pickle')
            # accountant.plot_selection()
            analyst = StabilVolter(
                start_level=start_level, end_level=end_level, tau_min=tau_min, tau_max=tau_max
            )
            stabilvol = analyst.get_stabilvol(data)
            # mfht = analyst.get_average_stabilvol(nbins=nbins)
            mfht = MeanFirstHittingTimes(stabilvol, nbins=500)
            # Save
            if save:
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
def count_fht(markets, start, end, duration, nbins, save=True):
    main(markets, start, end, duration, nbins)
    return None


if __name__ == "__main__":
    markets: list = ['UN']
    start_date = '2002-01-01'
    end_date = None
    duration = 6
    nbins = 3000
    save = True
    start_time = time.time()  # to track execution time
    main(markets=markets, start_date=start_date, end_date=end_date, duration=duration, nbins=nbins, save=save)
    end_time = time.time()
    print(f"\n\n***** Execution time for {len(markets)} markets: {end_time - start_time}s ***** ")
