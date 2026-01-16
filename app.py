import streamlit as st
import pandas as pd
from src.data import fetch_gold_data
from src.model import GoldRegimeModel
from src.plots import plot_price_and_regimes

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
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2000-01-01"))

st.sidebar.markdown("---")
st.sidebar.info("Click **Run Analysis** to fetch data and train the model.")
run_btn = st.sidebar.button("Run Analysis")

# Main Execution Block
if run_btn:
    # 1. Fetch Data
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            data = fetch_gold_data(ticker, start_date=str(start_date))
            st.success(f"Successfully loaded {len(data)} trading days.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
            
    # 2. Fit Model
    with st.spinner("Fitting Markov-Switching Model (Optimizing parameters)..."):
        try:
            model = GoldRegimeModel(k_regimes=2)
            summary = model.fit(data['Returns'])
            probs = model.predict_probs()
            stats = model.get_regime_stats()
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
        
        # Interpretation Logic
        regime_0_vol = stats.loc['Volatility (Std Dev)', 'Regime 0']
        regime_1_vol = stats.loc['Volatility (Std Dev)', 'Regime 1']
        
        high_vol_regime = "Regime 0" if regime_0_vol > regime_1_vol else "Regime 1"
        low_vol_regime = "Regime 1" if high_vol_regime == "Regime 0" else "Regime 0"
        
        st.info(f"""
        **Interpretation**:
        - **{high_vol_regime}**: High Volatility
        - **{low_vol_regime}**: Low Volatility
        """)
        
        # Current State
        last_prob = probs.iloc[-1]
        current_regime = 0 if last_prob[0] > last_prob[1] else 1
        current_regime_name = f"Regime {current_regime}"
        confidence = last_prob[current_regime]
        
        st.markdown(f"### Current State ({data.index[-1].date()})")
        st.metric("Dominant Regime", current_regime_name)
        st.metric("Probability", f"{confidence:.2%}")

    with col2:
        st.subheader("Regime Analysis Visuals")
        fig = plot_price_and_regimes(data, probs)
        st.pyplot(fig)
    
    # 4. Detailed Output
    with st.expander("See Detailed Model Summary"):
        st.text(summary)
else:
    st.info("👈 Please configure settings in the sidebar and click 'Run Analysis'.")

