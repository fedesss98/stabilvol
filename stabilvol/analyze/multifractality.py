import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

MARKETS = [
    "UN", 
    "UW", 
    "LN", 
    "JT"
    ]

ROOT = Path().parent.parent.parent
PRICES = ROOT / 'data/raw'

class InverseMultifractal:
    def __init__(self, price_series):
        # Convert prices to Log-Prices for scaling analysis
        # X(t) = ln(P(t))
        self.prices = np.array(price_series)
        self.log_prices = np.log(self.prices)
        self.n = len(self.log_prices)
        
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
                dX = np.abs(self.log_prices[tau:] - self.log_prices[:-tau])
                
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
                ref_price = self.log_prices[t]
                if np.isnan(ref_price):
                    t += 1
                    continue
                
                # Find the first future point where price exits [ref-delta, ref+delta]
                # Efficient search using numpy where
                future_prices = self.log_prices[t+1:]
                
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


def process_single_ticker(ticker, prices, taus, deltas, qs):
    """
    Worker function to calculate exponents for a single ticker.
    Returns a list of products (zeta * chi) corresponding to the q-orders.
    """
    try:
        # Initialize Analyzer with your class
        analyzer = InverseMultifractal(prices)

        # Run Calculations
        zeta_exponents = analyzer.get_standard_fluctuations(taus, qs)
        chi_exponents = analyzer.get_inverse_exit_times(deltas, qs)
        
        return [zeta_exponents[q] * chi_exponents[q] for q in qs]
        
    except Exception as e:
        # If one ticker fails (e.g., too short, all NaNs), return NaNs
        return [np.nan] * len(qs)


def main():
    # Setup parameters
    taus = np.unique(np.logspace(0, 2, 20).astype(int)) # Example tau
    deltas = np.logspace(np.log10(0.0001), np.log10(0.1), 50)
    qs = [1, 2, 3] # Example qs (ensure this is defined)
    
    results = {}

    for market in MARKETS:
        print(f"Processing Market: {market}...")
        
        # Load data (Sequential to save RAM)
        df = pd.read_pickle(PRICES / f"{market}.pickle")
        
        # 2. Parallelize the Inner Loop
        # n_jobs=-1 uses all available CPU cores
        market_results = Parallel(n_jobs=-1)(
            delayed(process_single_ticker)(ticker, prices, taus, deltas, qs)
            for ticker, prices in tqdm(df.items(), desc=f"Tickers in {market}")
        )
        
        # Aggregate Results
        results[market] = np.nanmean(np.array(market_results), axis=0)

    # Save final results
    final_df = pd.DataFrame.from_dict(results, orient="index", columns=[f"q{q}" for q in qs])
    final_df.to_csv(ROOT / "data/processed/multifractality/scaling_exponents.csv")
    print("Done!")
        

if __name__ == "__main__":
    main()