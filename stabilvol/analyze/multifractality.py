import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import argparse

from stabilvol.utility.functions import query_binned_data, stringify_threshold, setup_optimized_connection, list_database_thresholds


MARKETS = [
    "UN", 
    "UW", 
    "LN", 
    "JT"
    ]

ROOT = Path().parent.parent.parent
PRICES = ROOT / 'data/raw'
RETURNS = ROOT / 'data/interim'
DATABASE = ROOT / 'data/processed/trapezoidal_selection/stabilvol_logs.sqlite'
START = 0.3

class InverseMultifractal:
    def __init__(self, series):
        # Convert prices to Log-Prices for scaling analysis
        # X(t) = ln(P(t))
        self.series = np.array(series)
        self.n = len(self.series)
        
    def get_standard_fluctuations(self, tau_lags, q_orders):
        """
        Standard Analysis: Fix Time (tau), Measure Volatility (dX)
        S_q(tau) ~ tau^zeta(q)
        """
        results = {}
        
        for q in q_orders:
            fluctuation_moments = []
            valid_taus = []
            
            for tau in tau_lags:
                # Calculate differences for lag tau
                # dX = |X(t+tau) - X(t)|
                dX = np.abs(self.series[tau:] - self.series[:-tau])
                
                if np.any(~np.isnan(dX)):
                    # Calculate q-th moment
                    moment = np.nanmean(dX ** q)

                    if np.isfinite(moment) and moment > 0:
                        fluctuation_moments.append(moment)
                        valid_taus.append(tau)
            
            if len(fluctuation_moments) > 2:
                # Fit Log-Log to find zeta(q)
                # Use valid_taus instead of the original tau_lags
                try:
                    slope, intercept, _, _, _ = stats.linregress(
                        np.log(valid_taus), 
                        np.log(fluctuation_moments)
                    )
                    results[q] = slope
                except ValueError:
                    results[q] = np.nan
            else:
                results[q] = np.nan
            
        return results

    def get_inverse_exit_times(self, thresholds, q_orders):
        """
        Inverse Analysis: Fix Threshold (delta), Measure Time (tau)
        I_q(delta) ~ delta^chi(q)
        """
        results = {}
        
        # Pre-calculate exit times for each threshold
        # This is the computationally heavy part
        avg_exit_moments = {q: [] for q in q_orders}
        
        for delta in thresholds:
            exit_times = []
            t = 0
            while t < self.n - 1:
                # Current price reference
                ref_price = self.series[t]
                if np.isnan(ref_price):
                    t += 1
                    continue
                
                # Find the first future point where price exits [ref-delta, ref+delta]
                # Efficient search using numpy where
                future_prices = self.series[t+1:]
                
                dist = np.abs(future_prices - ref_price)
                
                # Create mask, treating NaNs as False (not a crossing)
                crossings = np.greater_equal(dist, delta, where=~np.isnan(dist))
                
                if np.any(crossings):
                    # Get the index of the first True value
                    # +1 because index 0 in future_prices is t+1
                    tau = np.argmax(crossings) + 1 
                    exit_times.append(tau)
                    
                    # Move t forward. 
                    # OPTION A: Non-overlapping (jump to exit). 
                    # OPTION B: Overlapping (t += 1). 
                    # We use Option A for distinct event analysis.
                    t += tau 
                else:
                    break # End of series reached without exit
            
            if len(exit_times) < 10:
                print(f"Warning: Not enough events for threshold {delta}")
                for q in q_orders:
                    avg_exit_moments[q].append(np.nan)
                continue

            # Calculate moments of exit times
            for q in q_orders:
                moment = np.mean(np.array(exit_times) ** q)
                avg_exit_moments[q].append(moment)

        # Fit Log-Log to find chi(q)
        # log(time_moment) = chi * log(delta) + C
        for q in q_orders:
            # Filter out NaNs
            valid_idx = np.isfinite(avg_exit_moments[q])
            if np.sum(valid_idx) > 2:
                y = np.log(np.array(avg_exit_moments[q])[valid_idx])
                x = np.log(np.array(thresholds)[valid_idx])
                slope, _, _, _, _ = stats.linregress(x, y)
                results[q] = slope
            else:
                results[q] = np.nan
                
        return results

    def load_inverse_exit_times(self, conn, market, thresholds, q_orders):
        """
        Inverse Analysis: Fix Threshold (delta), Measure Time (tau)
        I_q(delta) ~ delta^chi(q)
        """
        results = {}
        
        # Pre-calculate exit times for each threshold
        # This is the computationally heavy part
        avg_exit_moments = {q: [] for q in q_orders}
        for ts in thresholds:
            fhts = query_binned_data(
                market, "1980-01-01", "2022-31-12",
                t1_string=stringify_threshold(ts[0]), t2_string=stringify_threshold(ts[1]),
                min_bins=-1,
                conn=conn
            )[0]["FHT"].values

            # Calculate moments of exit times
            for q in q_orders:
                moment = np.mean(np.array(fhts) ** q)
                avg_exit_moments[q].append(moment)

        # Fit Log-Log to find chi(q)
        # log(time_moment) = chi * log(delta) + C
        for q in q_orders:
            # Filter out NaNs
            valid_idx = np.isfinite(avg_exit_moments[q])
            if np.sum(valid_idx) > 2:
                y = np.log(np.array(avg_exit_moments[q])[valid_idx])
                x = np.log(np.array(thresholds)[valid_idx, 1])
                slope, _, _, _, _ = stats.linregress(x, y)
                results[q] = slope
            else:
                results[q] = np.nan
                
        return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multifractality Analysis")
    parser.add_argument('-t', '--multifractality-type', type=int, choices=[1,2,3], default=1,
                        help='Type of Inverse Multifractal Analysis to perform (default: 1)')
    return parser.parse_args()


def process_single_ticker(market, prices, taus, deltas, qs, conn=None):
    """
    Worker function to calculate exponents for a single ticker.
    Returns a list of products (zeta * chi) corresponding to the q-orders.
    """
    try:
        # Initialize Analyzer with your class
        analyzer = InverseMultifractal(prices)

        # Run Calculations
        zeta_exponents = analyzer.get_standard_fluctuations(taus, qs)
        if conn is None:
            chi_exponents = analyzer.get_inverse_exit_times(deltas, qs)
        else:
            chi_exponents = analyzer.load_inverse_exit_times(conn, market, deltas, qs)
        
        return [zeta_exponents[q] * chi_exponents[q] for q in qs]
        
    except Exception as e:
        # If one ticker fails (e.g., too short, all NaNs), return NaNs
        return [np.nan] * len(qs)


def select_thresholds(df_thresholds, start):
    ends = df_thresholds[df_thresholds["Start"] == start]["End"]
    if start > 0:
        ends = ends[(ends > 0) & (ends < start)].sort_values()
    else:
        ends = ends[(ends < 0) & (ends > start)].sort_values()
    thresholds = [(start, end) for end in ends]
    return thresholds


def main():
    args = parse_arguments()
    # Setup parameters
    taus = np.unique(np.logspace(0, 2, 20).astype(int)) # Example tau
    if args.multifractality_type == 3:
        conn = setup_optimized_connection()
        df_experiments = list_database_thresholds(DATABASE)
        thresholds = select_thresholds(df_experiments, START)
    else:
        conn = None
        thresholds = np.logspace(np.log10(0.0001), np.log10(0.1), 50)

    qs = [1, 2, 3] # Example qs (ensure this is defined)
    
    results = {}

    for market in MARKETS:
        print(f"Processing Market: {market}...")
        
        if args.multifractality_type == 1:
            # Load raw prices
            df = pd.read_pickle(PRICES / f"{market}.pickle")
        else:
            # Load log-returns
            df = pd.read_pickle(RETURNS / f"{market}_log.pickle")
        
        # n_jobs=-1 uses all available CPU cores
        market_results = Parallel(n_jobs=-1)(
            delayed(process_single_ticker)(market, prices, taus, thresholds, qs, conn)
            for _, prices in tqdm(df.items(), desc=f"Tickers in {market}")
        )
        
        # Aggregate Results
        results[market] = np.nanmean(np.array(market_results), axis=0)

    # Save final results
    final_df = pd.DataFrame.from_dict(results, orient="index", columns=[f"q{q}" for q in qs])
    final_df.to_csv(
        ROOT / f"data/processed/multifractality/scaling_exponents_type{args.multifractality_type}.csv")
    
    print("Done!")


if __name__ == "__main__":
    main()
