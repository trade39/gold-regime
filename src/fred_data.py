import pandas as pd
from fredapi import Fred
import streamlit as st

def fetch_fred_data(api_key, start_date="2000-01-01"):
    """
    Fetches key macroeconomic indicators from FRED.
    """
    if not api_key:
        return None
        
    try:
        fred = Fred(api_key=api_key)
        
        # Series IDs
        # DGS10: 10-Year Treasury Constant Maturity Rate
        # T10YIE: 10-Year Breakeven Inflation Rate
        # DTWEXBGS: Trade Weighted U.S. Dollar Index: Broad, Goods and Services
        # T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        
        # Note: Some series might have different start dates or frequencies.
        # We'll fetch them and resample/forward fill.
        
        series_ids = {
            '10Y Yield': 'DGS10',
            '10Y Breakeven Inflation': 'T10YIE', 
            'US Dollar Index': 'DTWEXBGS',
            'Yield Curve (10Y-2Y)': 'T10Y2Y'
        }
        
        data_frames = []
        for name, series_id in series_ids.items():
            try:
                # Fetch series
                series = fred.get_series(series_id, observation_start=start_date)
                series.name = name
                data_frames.append(series)
            except Exception as e:
                st.warning(f"Could not fetch {name} ({series_id}): {e}")
                
        if not data_frames:
            return None
            
        # Combine into DataFrame
        macro_df = pd.concat(data_frames, axis=1)
        
        # Forward fill to handle weekends/holidays differences if any
        macro_df = macro_df.fillna(method='ffill')
        
        return macro_df
        
    except Exception as e:
        st.error(f"Error connecting to FRED API: {e}")
        return None
