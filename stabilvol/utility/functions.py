"""
Created on 2022 - 11 - 10

Utility functions
"""
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import re


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
    cursor.close()
    conn.close()
    return tables


def extract_t1_t2(table_name: str) -> tuple:
    """ Extract thresholds t1 and t2 from a table name """
    t1 = table_name.split('_')[1]
    t2 = table_name.split('_')[2]
    t1 = t1.replace('m', '-').replace('p', '.')
    t2 = t2.replace('m', '-').replace('p', '.')
    return round(float(t1), 2), round(float(t2), 2)


def stringify_threshold(t):
    t = str(t).replace('-', 'm').replace('.', 'p')
    return t


def list_database_thresholds(database) -> pd.DataFrame:
    # Connect to the SQLite database
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    # Query the database to get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = cur.fetchall()

    # Extract the values of t1 and t2 from each table name
    results = []
    for table_name in table_names:
        try:
            t1, t2 = extract_t1_t2(table_name[0])
        except TypeError:
            pass
        else:
            results.append((t1, t2))

    thresholds = pd.DataFrame(results, columns=['Start', 'End'])
    # Print thresholds
    print(f"{'Start Threshold':^16}\t{'End Thresholds':^25}")
    for t1, t2_group in thresholds.groupby('Start'):
        print(f"{' '*5}{t1:>6}{' '*5}", end='\t')
        for t2 in t2_group['End'].sort_values():
            print(f"{t2:>6}", end=' ')
        print()

    # Close the database connection
    cur.close()
    conn.close()
    return thresholds


def query_data(database, query):
    engine = create_engine(f'sqlite:///{database}')
    df = pd.read_sql_query(query, con=engine)

    return df


if __name__ == "__main__":
    database = "../../data/processed/trapezoidal_selection/stabilvol.sqlite"
    list_database_thresholds(database)