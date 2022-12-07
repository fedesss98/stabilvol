"""
Command Line Interface for Volatility Analysis in Financial Markets.
"""
import sys
sys.path.extend(['G:\\UNIPA\\ECONOFISICA\\stabilizing_volatility'])

import click
from data.count_stocks import count_stocks


@click.group()
def cli():
    return None

cli.add_command(count_stocks)


if __name__ == "__main__":
    cli()