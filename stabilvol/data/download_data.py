"""
Created on 2022 - 11 - 10

Ask for market, search market data
(in ROOT_DATA directory) and copy
them in the raw_data file
"""
from pathlib import Path
import shutil

from stabilvol.utility import functions as use
from stabilvol.utility.definitions import ROOT, DATA_ROOT, MARKETS


def download_data():
    for market in MARKETS:
        data_file = Path(DATA_ROOT, market).with_suffix('.csv')
        print(f'Starting analysis of {data_file}')
        if data_file.is_file():
            output_file = ROOT / f'data/external/{market}.csv'
            print(f'Saving it in {output_file}')
            shutil.copyfile(data_file, output_file)
        else:
            raise FileNotFoundError(
                f"Cannot find file {data_file}.\n"
                f"Check DATA_ROOT definitions."
            )
    return None


if __name__ == "__main__":
    print(ROOT)
    download_data()
