import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import joblib
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="BTC 5m AI Predictor", layout="wide")
st.title("🤖 Bitcoin 5-Minute AI Predictor")

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("btc_5m_xgboost.joblib")
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")
        st.stop()

model = load_model()

# --- 2. Fetch Live Data ---
@st.cache_data(ttl=60) # Cache for 60 seconds so we don't spam the API on every click
def fetch_live_data(symbol='BTC/USDT', timeframe='1m', limit=100):
    exchange = ccxt.binanceus()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

# --- 3. Feature Engineering (Must match training EXACTLY) ---
def engineer_features(df):
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    return df.dropna()

# --- Main App Execution ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("Fetching latest market data and generating predictions...")
    raw_data = fetch_live_data(limit=150) # Need extra rows to calculate MACD/EMA
    live_data = engineer_features(raw_data.copy())
    
    # --- Make Prediction on the MOST RECENT complete candle ---
    latest_features = live_data.iloc[[-1]] # Get the last row
    
    # Get the raw prediction (0 or 1) and the probability/confidence
    prediction = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    confidence = probabilities[prediction] * 100
    
    # --- Visualization ---
    st.subheader("Live Price Action & Indicators")
    fig = go.Figure(data=[go.Candlestick(x=raw_data.index,
                    open=raw_data['Open'],
                    high=raw_data['High'],
                    low=raw_data['Low'],
                    close=raw_data['Close'],
                    name="Price")])
                    
    # Add the 9 EMA line to the chart as an example indicator
    if 'EMA_9' in live_data.columns:
        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['EMA_9'], 
                                 line=dict(color='orange', width=1), name='9 EMA'))

    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("AI Signal")
    st.write(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    if prediction == 1:
        st.success("🟢 PUMP (UP)")
        st.metric("Confidence", f"{confidence:.2f}%")
        if confidence > 65: # High confidence threshold
            st.warning("🔥 High Confidence Signal: Consider entering LONG (Paper Trade)")
    else:
        st.error("🔴 DUMP (DOWN)")
        st.metric("Confidence", f"{confidence:.2f}%")
        if confidence > 65:
            st.warning("🔥 High Confidence Signal: Consider entering SHORT (Paper Trade)")
            
    st.divider()
    st.write("Current Market Stats:")
    st.write(f"**Current Price:** ${raw_data['Close'].iloc[-1]:,.2f}")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()