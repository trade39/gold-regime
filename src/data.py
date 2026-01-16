import yfinance as yf
import pandas as pd
import numpy as np

def fetch_gold_data(ticker="GC=F", start_date=None, period="max", interval="1d"):
    """
    Fetches gold prices and calculates log returns.
    
    Args:
        ticker (str): Ticker symbol (default "GC=F" for Gold Futures).
        start_date (str): Start date for data fetching (used if interval is '1d').
        period (str): Data period to download (e.g. '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max').
        interval (str): Data interval (e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
        
    Returns:
        pd.DataFrame: DataFrame with 'Price' and 'Returns' columns.
    """
    print(f"Fetching data for {ticker} (Interval: {interval}, Period: {period})...")
    
    # If Start Date is provided and interval is daily, prefer start date
    if interval == "1d" and start_date:
        df = yf.download(ticker, start=start_date, interval=interval, progress=False)
    else:
        # For intraday, use period
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    
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
    
    # Ensure index is timezone-naive for compatibility with matplotlib/statsmodels
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df
