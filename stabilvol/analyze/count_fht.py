"""

Count FHT
"""
import pandas as pd
import matplotlib.pyplot as plt
from utility.classes.data_extraction import DataExtractor
from utility.classes.stability_analysis import StabilVolter
from utility.definitions import ROOT
import time


def state_of_x(x, start, end):
    if x > start:
        return 1
    elif x < end:
        return -1
    else:
        return 0


if __name__ == "__main__":
    start_time = time.time()
    accountant = DataExtractor(start_date='2002-01-01', duration=10)
    data = accountant.extract_data(ROOT / 'data/interim/GF.pickle')
    for end_level in [-1.0, ]:
        analyst = StabilVolter(end_level=end_level)
        analyst.get_stabilvol(data.iloc[:100, 0:1].reset_index(drop=True))
        plot_ax = analyst._plot_states()
        plt.show()
        plot_count_ax = analyst._plot_count_states()
        plt.show()
    print(f"\n\n***** First Execution time: {time.time() - start_time} ***** ")
