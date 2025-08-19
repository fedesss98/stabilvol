"""
Created on 2022 - 11 - 10

Utility functions
"""
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
# from sqlalchemy import create_engine
from tqdm import tqdm
import datetime

STARTING_BINS = 25
VOL_LIMIT = 0.5
DATABASE = '../data/processed/trapezoidal_selection/stabilvol.sqlite'


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
    t = round(float(t), 2)
    t = str(t).replace('-', 'm').replace('.', 'p')
    return t


def numerify_threshold(t):
    t = t.replace('m', '-').replace('p', '.')
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


# def query_data(database, query):
#     engine = create_engine(f'sqlite:///{database}')
#     return pd.read_sql_query(query, con=engine)


def error_on_the_mean(values):
    return np.std(values) / np.sqrt(len(values))


def select_bins(df, max_n=1000, min_n = STARTING_BINS):
    """
    Bin FHT to make MFHT.
    Bins are chosen dynamically to have at most max_n observations in each bin and at most 1000 bins.
    If in each bin there are more than max_n observations, the number of bins is increased by 100.
    """
    nbins = min_n
    # Take the numpy arrays of the columns for faster access
    volatility = df['Volatility'].values
    fht = df['FHT'].values

    while nbins <= 1000:
        try:
            # Use qcut with numpy arrays - faster
            bins = pd.qcut(volatility, nbins, duplicates='drop')
            
            # Create DataFrame only once
            temp_df = pd.DataFrame({'Bins': bins, 'FHT': fht})
            
            # Group and aggregate
            grouped = temp_df.groupby('Bins', observed=True)['FHT'].agg([
                'mean', error_on_the_mean, 'size'
            ])
            
            count = grouped['size'].min()
            
            if count < max_n:
                return grouped, nbins
                
        except ValueError:  # Handle edge cases in qcut
            pass
            
        nbins += 20
    
    # Fallback if we exceed 1000 bins
    return grouped, nbins


def query_binned_data(
        market:str, 
        start_date:str, 
        end_date:str = None, 
        vol_limit:float = 0.5,
        tau_max:int = 30,
        t1_string:str = "m0p5", 
        t2_string:str = "m1p5", 
        conn=None,
        raise_error:bool = True):
    grouped_data = None
    conn = sqlite3.connect(DATABASE) if conn is None else conn
    end_date = '2023-01-01' if end_date is None else end_date
    # If your dates are datetime objects, convert them:
    start_date = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_date = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

    try:            
        # Write the SQL query
        query = f'''
        SELECT Volatility, FHT
        FROM stabilvol_{t1_string}_{t2_string}
        WHERE Volatility < ? 
        AND Market = ?
        AND start >= ?
        AND end <= ?
        AND FHT <= ?
        '''
        # Load the FHT data from the database
        df = pd.read_sql_query(query, conn, params=(vol_limit, market, start_date, end_date, tau_max))
        
    except pd.errors.DatabaseError as e:
        if raise_error:
            raise ValueError(f'No data for market {market} with thresholds {t1_string}-{t2_string} from {start_date} to {end_date}')
        else:
            print(f'No data for market {market} with thresholds {t1_string}-{t2_string} from {start_date} to {end_date}')
            return pd.DataFrame(), 0
    else:
        if len(df) > 50:
            return select_bins(df)
        else:
            raise ValueError(f'Not enough data for market {market} with thresholds {t1_string}-{t2_string} from {start_date} to {end_date}')


def create_dataset(markets, windows, t1_string, t2_string, vol_limit=VOL_LIMIT):
    outcasts = {market: [] for market in markets}
    df_list = []
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE)
    for market in markets:
        for start_date, end_date in tqdm(windows, desc=market):
            try:
                mfht, nbins = query_binned_data(
                    market, start_date, end_date, vol_limit, t1_string=t1_string, t2_string=t2_string, conn=conn)
            except ValueError:
                outcasts[market].append((start_date, end_date))
            else:
                mfht['start'] = start_date
                mfht['end'] = end_date
                mfht['market'] = market
                df_list.append(mfht.reset_index())

    return pd.concat(df_list), outcasts


def _add_ticks(ax, windows, coeff, outcasts, highlights=True, **kwargs):
    ax.set_title(
        ' '.join([r'$\theta_i$=', numerify_threshold(coeff[0]), r'/ $\theta_f$=', numerify_threshold(coeff[1])]),
        fontsize=12)
    # Remove yticks
    ax.yaxis.set_ticks([])

    # Set the xticks to be the start date of each window
    label_spacing = kwargs.get('label_spacing', 1)
    labels = [win[0].strftime('%Y-%b') for win in windows][::label_spacing]
    l = 1
    # Add the last date
    labels.append(windows[-1][1].strftime('%Y-%b'))
    if len(labels) != len(np.arange(0, len(windows) + l, label_spacing)):
        # Add last window end
        l += label_spacing
    ax.set_xticks(np.arange(0, len(windows) + l, label_spacing))
    ax.set_xticklabels(labels, rotation=90, va='bottom', fontsize=11, y=-0.9)
    ax.tick_params(axis='x', colors='black', direction='out', length=6, width=2)

    label_dates = [start_date for start_date, end_date in windows]
    label_dates.append(windows[-1][1])
    label_dates = pd.to_datetime(label_dates)
    outcast_dates = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in outcasts]
    for outcast in outcast_dates:
        # Find the indices of the start and end labels
        try:
            start_index = np.where(label_dates <= outcast[0])
            # Since only the end date is labeled, if the first start date is an outcast, it must be set manually
            start_index = start_index[0][-1] if len(start_index[0]) > 0 else 0
            end_index = np.where(label_dates >= outcast[1])[0][0]
        except IndexError as e:
            print(f'Cannot find end index for date {outcast[1]}')
        else:
            ax.axvspan(start_index, end_index, color='black')

    if highlights:
        try:
            # Find the indices of the start and end labels
            start_index = np.where(label_dates < pd.to_datetime('2006-12-31'))[0][-1]
            end_index = np.where(label_dates > pd.to_datetime('2008-12-31'))[0][0]
        except IndexError as e:
            print('Cannot highlight crisis: ', e)
        else:
            # Add vertical lines at the start and end of the region
            ax.axvline(start_index, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(end_index, color='k', linestyle='--', linewidth=1.5)


def plot_rolling_pmesh(coefficients, windows, values, **kwargs):
    outcasts = {(t1, t2): [] for t1, t2 in coefficients}

    if kwargs.get('latex', False):
        # Use LaTeX for text rendering
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'

    fig, axs = plt.subplots(len(coefficients), figsize=(12, 2.5), sharex=True, layout='constrained')
    flattened_axs = axs.flatten() if len(coefficients) > 1 else [axs]
    if kwargs.get('suptitle', False):
        fig.suptitle(kwargs.get('suptitle'), fontsize=16)

    # Search for outcasts
    for i, (coeff, ax) in enumerate(zip(coefficients, flattened_axs)):
        for j, (start_date, end_date) in enumerate(windows):
            # See where the max is zero and label it as outcast
            if values[i, j] < 0:
                outcasts[coeff].append((start_date, end_date))
        print(f"Outcasts for {coeff}: {len(outcasts[coeff])}")

        # Create a TwoSlopeNorm
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=1)
        pmesh = ax.pcolormesh(values[i].reshape(1, -1),
                              cmap='coolwarm', norm=norm,
                              edgecolors='w', linewidth=kwargs.get('linewidth', 0)
                              )
        # Add ticks to the plot
        _add_ticks(ax, windows, coeff, outcasts[coeff], **kwargs)
        # Set the colorbar for each plot showing only maximum and minimum values
    cbar = fig.colorbar(pmesh, ax=axs, orientation='vertical', pad=0.01, ticks=[0, 0.05, 1], aspect=10)
    # cbar.set_ticks([0.0, values[i].mean(), values[i].max()])
    cbar.ax.set_yticklabels([0, 'Accept\n' + r'$\big\uparrow$' + '\nThreshold\n' + r'$\big\downarrow$' + '\nReject', 1],
                            fontsize=11)

    plt.show()
    return fig, outcasts


def format_mfht_directory(selection_criterion, volatility):
    # Convert the number to a float to ensure proper handling
    num = float(volatility)
    
    # Extract the integer part and decimal part
    int_part = int(num)
    decimal_part = num - int_part
    
    # Format the integer part to have at least 2 digits
    int_str = f"{int_part:02d}"
    
    # For the decimal part, handle it only if it's non-zero
    if decimal_part > 0:
        # Convert to string, remove '0.' prefix, and ensure no scientific notation
        decimal_str = f"{decimal_part:.10f}".split('.')[1].rstrip('0')
        result = int_str + decimal_str
    else:
        result = int_str
    
    # Return the final string with 'vol' prefix
    return f"data/processed/{selection_criterion}_selection/vol{result}"


def roll_windows(duration=90,  start_date=None, end_date=None):
    # Define the start and end dates
    start_date = datetime.date(1980, 1, 1) if start_date is None else start_date
    end_date = datetime.date(2022, 7, 1) if end_date is None else end_date
    
    half_win_len = pd.to_timedelta(duration//2, 'D')
    start = start_date + half_win_len
    end = end_date - half_win_len
    centers = pd.date_range(start, end, freq='D')
    return [(mid - half_win_len, mid + half_win_len) for mid in centers]


if __name__ == "__main__":
    database = "../../data/processed/trapezoidal_selection/stabilvol.sqlite"
    list_database_thresholds(database)