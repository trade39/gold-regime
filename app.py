import streamlit as st
import pandas as pd
from src.data import fetch_gold_data
from src.model import GoldRegimeModel
from src.plots import plot_price_and_regimes
from src.fred_data import fetch_fred_data

# Page Config
st.set_page_config(
    page_title="Gold Regime Predictor",
    page_icon="🪙",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #D4AF37;
    }
    .stButton>button {
        width: 100%;
        background-color: #D4AF37;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🪙 Gold Price Regime Prediction")
st.markdown("""
### Based on Hamilton's Markov-Switching Methodology
This application analyzes Gold Futures (`GC=F`) prices to identify hidden market regimes (e.g., Low vs. High Volatility).
It uses a **Markov-Switching AutoRegressive Model** to estimate the probability of the current regime.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("Configure the data source and model parameters.")

ticker = st.sidebar.text_input("Ticker Symbol", value="GC=F", help="Yahoo Finance Ticker")

# Timeframe Selection
interval = st.sidebar.selectbox("Timeframe", options=['1d', '90m', '60m', '30m', '15m', '5m'], index=0)

# Show start date only for daily data
start_date = None
if interval == '1d':
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2000-01-01"))
    st.sidebar.caption("For daily data, you can choose a specific start date.")
else:
    st.sidebar.caption(f"Intraday data ({interval}) assumes 'max' recent history (~60 days).")

fred_api_key = st.sidebar.text_input("FRED API Key (Optional)", type="password", help="Get one at fred.stlouisfed.org")
k_regimes = st.sidebar.slider("Number of Regimes", min_value=2, max_value=3, value=3, help="3 = Bull/Bear/Consolidating")

st.sidebar.markdown("---")
st.sidebar.info("Click **Run Analysis** to fetch data and train the model.")
run_btn = st.sidebar.button("Run Analysis")

# Main Execution Block
if run_btn:
    # 1. Fetch Data
    with st.spinner(f"Fetching data for {ticker} ({interval})..."):
        try:
            # For intraday, we pass period="max" (or "60d" for safety)
            # data.py logic handles defaults
            data = fetch_gold_data(ticker, start_date=str(start_date) if start_date else None, interval=interval, period="max")
            st.success(f"Successfully loaded {len(data)} bars.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
            
    # 2. Fetch FRED Data (Optional)
    macro_data = None
    if fred_api_key:
        # Determine start date for FRED if not set (i.e. Intraday)
        fred_start_date = start_date if start_date else (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        
        with st.spinner("Fetching Macro Data from FRED..."):
            macro_data = fetch_fred_data(fred_api_key, start_date=str(fred_start_date))
            if macro_data is not None:
                st.success(f"Loaded Macro Indicators: {', '.join(macro_data.columns)}")
            
    # 3. Fit Model
    with st.spinner(f"Fitting {k_regimes}-Regime Markov-Switching Model..."):
        try:
            model = GoldRegimeModel(k_regimes=k_regimes)
            summary = model.fit(data['Returns'])
            probs = model.predict_probs()
            stats = model.get_regime_stats()
            
            # Interpret Regimes
            regime_labels = model.interpret_regimes(stats)
        except Exception as e:
            st.error(f"Error fitting model: {e}")
            st.stop()

    # 3. Display Results
    st.divider()
    
    # Layout: Stats on left, Chart on right (or stacked)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Regime Statistics")
        st.dataframe(stats.style.format("{:.4f}"))
        
        # Display Legend for Regimes
        st.markdown("### Regime Interpretation")
        for r, label in regime_labels.items():
            st.write(f"- **{r}**: {label}")

        # Current State
        last_prob = probs.iloc[-1]
        # Find dominant regime
        current_regime_idx = last_prob.argmax()
        current_regime_name = f"Regime {current_regime_idx}"
        current_label = regime_labels.get(current_regime_name, current_regime_name)
        confidence = last_prob[current_regime_idx]
        
        st.markdown(f"### Current State ({data.index[-1].date()})")
        st.metric("Dominant Regime", current_label)
        st.metric("Probability", f"{confidence:.2%}")

        if macro_data is not None:
            st.markdown("### Macro Context")
            latest_macro = macro_data.iloc[-1]
            st.metric("10Y Yield", f"{latest_macro.get('10Y Yield', 0):.2f}%")
            st.metric("Dollar Index", f"{latest_macro.get('US Dollar Index', 0):.2f}")

    with col2:
        st.subheader("Regime Analysis Visuals")
