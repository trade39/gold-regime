import yfinance as yf
import pandas as pd
import numpy as np

def fetch_gold_data(ticker="GC=F", start_date="2000-01-01"):
    """
    Fetches daily gold prices and calculates log returns.
    
    Args:
        ticker (str): Ticker symbol (default "GC=F" for Gold Futures).
        start_date (str): Start date for data fetching.
        
    Returns:
        pd.DataFrame: DataFrame with 'Price' and 'Returns' columns.
    """
    print(f"Fetching data for {ticker} from {start_date}...")
    df = yf.download(ticker, start=start_date, progress=False)
    
    # Check if data is empty
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check your internet connection or ticker symbol.")

    # Handle MultiIndex columns if present (yfinance update for multiple tickers, 
    # but sometimes happens even with one)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Try to get the ticker level if it exists
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            # Fallback if structure is different, usually 'Close' is at level 0
            pass

    # Ensure we have the Close price
    if 'Close' not in df.columns:
         # Try using 'Adj Close' if 'Close' is missing
        if 'Adj Close' in df.columns:
             df['Close'] = df['Adj Close']
        else:
             raise KeyError(f"Column 'Close' not found in data. Columns: {df.columns}")

    df = df[['Close']].copy()
    df.columns = ['Price']
    df.dropna(inplace=True)
    
    # Calculate Log Returns * 100 for better numerical stability in optimization
    # r_t = 100 * (ln(P_t) - ln(P_{t-1}))
    df['Returns'] = np.log(df['Price'] / df['Price'].shift(1)) * 100
    df.dropna(inplace=True)
    
    return df
