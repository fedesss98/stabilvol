"""
Created on 2022 - 11 - 10

Utility functions
"""
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

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


def list_database_tables(database):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        print(table[0])


def query_data(database, query):
    engine = create_engine(f'sqlite:///{database}')
    df = pd.read_sql_query(query, con=engine)

    return df


if __name__ == "__main__":
    ask_for_market()