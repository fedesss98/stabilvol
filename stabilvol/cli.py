"""
Command Line Interface for Volatility Analysis in Financial Markets.
"""
import sys
sys.path.extend(['G:\\UNIPA\\ECONOFISICA\\stabilizing_volatility'])

import click
from data.count_stocks import count_stocks
from analyze.count_fht import count_fht


@click.group()
def cli():
    return None

cli.add_command(count_stocks)
cli.add_command(count_fht)


if __name__ == "__main__":
    cli()