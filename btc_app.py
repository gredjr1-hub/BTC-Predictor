import streamlit as st
import pandas as pd
import ta
import ccxt
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- 1. Page Setup ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin 5-Minute AI Predictor")
st.markdown("This dashboard uses a trained Random Forest model to predict if BTC will close higher in the next 5 minutes.")

# --- 2. Load the Brain ---
@st.cache_resource
def load_model():
    return joblib.load("btc_5m_rf_model.joblib")

try:
    model = load_model()
    st.sidebar.success("✅ AI Model Loaded Successfully")
except Exception as e:
    st.error(f"Could not load the model. Make sure 'btc_5m_rf_model.joblib' is in the folder. Error: {e}")
    st.stop()

# --- 3. Live Data Fetcher & Feature Engineer ---
def get_live_prediction_data():
    # Fetch the last 100 minutes of data to calculate indicators
    exchange = ccxt.binanceus({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=100)
    
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    # Calculate the exact same features the model was trained on
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    
    # Clean up infinities and NaNs just like in training
    import numpy as np
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# --- 4. UI and Logic ---
if st.button("🔮 Generate Live Prediction", type="primary"):
    with st.spinner("Fetching live exchange data and consulting AI..."):
        # Get data
        live_data = get_live_prediction_data()
        
        # Grab the absolute newest row of data
        current_state = live_data.iloc[-1:]
        current_price = current_state['Close'].values[0]
        current_rsi = current_state['RSI_14'].values[0]
        
        # Ask the AI to predict
        prediction = model.predict(current_state)[0]
        probability = model.predict_proba(current_state)[0]
        
        # --- 5. Display Results ---
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current BTC Price", f"${current_price:,.2f}")
        col2.metric("Current RSI (14)", f"{current_rsi:.1f}")
        
        if prediction == 1:
            col3.success(f"📈 AI Predicts UP (Confidence: {probability[1]*100:.1f}%)")
        else:
            col3.error(f"📉 AI Predicts DOWN (Confidence: {probability[0]*100:.1f}%)")
            
        # Draw a quick chart
        st.subheader("Last 60 Minutes of Price Action")
        fig = go.Figure(data=[go.Candlestick(x=live_data.index[-60:],
                        open=live_data['Open'][-60:],
                        high=live_data['High'][-60:],
                        low=live_data['Low'][-60:],
                        close=live_data['Close'][-60:])])
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)