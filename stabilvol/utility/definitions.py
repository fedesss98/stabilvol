from pathlib import Path

# Root of the project, it should point to the stabilizing_volatility folder
ROOT = Path(
    'G:/UNIPA/ECONOFISICA/stabilizing_volatility'
)

# External directory containing raw stock prices to analyze
DATA_ROOT = Path(
    'G:/UNIPA/ECONOFISICA/DATA/BLOOMBERG'
)

# Available markets from Bloomberg
MARKETS = {'GF', 'JT', 'LN', 'UN', 'UW'}

FIRST_VALID_DAY = '1980-01-01'
LAST_VALID_DAY = '2022-07-01'


class Project:
    def __init__(self):
        self.root = ROOT
        self.data_root = DATA_ROOT
        self.market = None


if __name__ == "__main__":
    project = Project()
    print(f'Root Directory:\n{project.root}')
    print(f'Data Directory:\n{project.data_root}')
