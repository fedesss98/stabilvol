"""
Created on 2022 - 11 - 10

Utility functions
"""


def check_input_market(market):
    available_markets = {'GF', 'JT', 'LN', 'UN', 'UW'}
    if market in available_markets:
        is_market = True
    else:
        is_market = False
        print("Market not available. Try again.")
    return is_market


def ask_for_market():
    is_market = False
    while not is_market:
        market = input("Which market do you want to analyze?\n")
        is_market = check_input_market(market.upper())
    return market.upper()


if __name__ == "__main__":
    ask_for_market()