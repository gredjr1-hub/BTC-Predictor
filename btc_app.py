import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# --- 1. Page Setup ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin 5-Minute AI Predictor")

# --- 2. Load the Brain ---
@st.cache_resource
def load_model():
    return joblib.load("btc_5m_rf_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Error: {e}")
    st.stop()

# --- 3. Live Data Fetcher & Feature Engineer ---
def get_live_prediction_data():
    exchange = ccxt.binanceus({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=100)
    
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    # Calculate features
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# --- 4. Tracker System (Lazy Evaluator) ---
HISTORY_FILE = "trade_history.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE, parse_dates=['Prediction_Time', 'Target_Time'])
    else:
        return pd.DataFrame(columns=['Prediction_Time', 'Entry_Price', 'Prediction', 'Confidence', 'Target_Time', 'Close_Price', 'Outcome'])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

def resolve_pending_trades(live_data):
    history = load_history()
    if history.empty: return history
    
    pending_mask = history['Outcome'] == 'Pending'
    
    for idx, row in history[pending_mask].iterrows():
        target_time = row['Target_Time']
        
        # Check if the target time has passed AND is in our recently downloaded live data
        if target_time <= datetime.utcnow() and target_time in live_data.index:
            actual_close = live_data.loc[target_time, 'Close']
            entry_price = row['Entry_Price']
            prediction = row['Prediction']
            
            history.at[idx, 'Close_Price'] = actual_close
            
            # Grade the prediction
            if prediction == 'UP' and actual_close > entry_price:
                history.at[idx, 'Outcome'] = 'Win'
            elif prediction == 'DOWN' and actual_close < entry_price:
                history.at[idx, 'Outcome'] = 'Win'
            else:
                history.at[idx, 'Outcome'] = 'Loss'
                
    save_history(history)
    return history


# --- 5. UI Layout (Tabs) ---
tab1, tab2 = st.tabs(["🔮 Live Predictor", "📊 Analytics & History"])

with tab1:
    st.markdown("### Generate Next Move")
    if st.button("Generate Live Prediction", type="primary"):
        with st.spinner("Fetching live data & analyzing..."):
            live_data = get_live_prediction_data()
            
            # Auto-grade any pending trades while we have the fresh data!
            resolve_pending_trades(live_data)
            
            current_state = live_data.iloc[-1:]
            current_price = current_state['Close'].values[0]
            current_time = current_state.index[0]
            
            # Predict
            prediction_val = model.predict(current_state)[0]
            probabilities = model.predict_proba(current_state)[0]
            
            direction = "UP" if prediction_val == 1 else "DOWN"
            confidence = probabilities[1] if prediction_val == 1 else probabilities[0]
            confidence_pct = confidence * 100
            
            # Threshold Logic
            if confidence_pct < 55:
                signal_strength = "⚠️ WEAK SIGNAL (Coin Flip - Do not trade)"
                color = "orange"
            elif confidence_pct < 60:
                signal_strength = "✅ MODERATE SIGNAL (Standard Edge)"
                color = "blue"
            else:
                signal_strength = "🔥 STRONG SIGNAL (High Probability)"
                color = "green"

            # Display Results
            st.markdown(f"### Target Time: {(current_time + timedelta(minutes=5)).strftime('%H:%M:%S')} UTC")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current BTC Price", f"${current_price:,.2f}")
            col2.metric("AI Prediction", f"{direction}", delta="Long" if direction=="UP" else "-Short")
            col3.metric("AI Confidence", f"{confidence_pct:.1f}%")
            
            st.markdown(f"**Signal Strength:** :{color}[{signal_strength}]")
            
            # Log the prediction
            history = load_history()
            new_trade = pd.DataFrame([{
                'Prediction_Time': current_time,
                'Entry_Price': current_price,
                'Prediction': direction,
                'Confidence': round(confidence_pct, 2),
                'Target_Time': current_time + timedelta(minutes=5),
                'Close_Price': np.nan,
                'Outcome': 'Pending'
            }])
            history = pd.concat([history, new_trade], ignore_index=True)
            save_history(history)
            st.success("Prediction logged to history tracker.")

with tab2:
    st.markdown("### Forward Testing Analytics")
    
    # Reload and resolve any pending data just in case
    try:
        live_data = get_live_prediction_data()
        history = resolve_pending_trades(live_data)
    except:
        history = load_history()
    
    if not history.empty:
        # Metrics
        completed_trades = history[history['Outcome'].isin(['Win', 'Loss'])]
        total_completed = len(completed_trades)
        wins = len(completed_trades[completed_trades['Outcome'] == 'Win'])
        
        win_rate = (wins / total_completed * 100) if total_completed > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Completed Predictions", total_completed)
        m2.metric("Real-World Win Rate", f"{win_rate:.1f}%")
        m3.metric("Pending Predictions", len(history[history['Outcome'] == 'Pending']))
        
        # Display the Tracker
        st.markdown("#### Permanent Tracker Log")
        
        # Style the dataframe so Wins are green and Losses are red
        def highlight_outcome(val):
            if val == 'Win': return 'background-color: #004d00'
            elif val == 'Loss': return 'background-color: #4d0000'
            return ''
            
        st.dataframe(history.style.map(highlight_outcome, subset=['Outcome']), use_container_width=True)
        
        st.markdown("*Note: Pending predictions are automatically graded the next time you generate a prediction after 5 minutes have passed.*")
        
    else:
        st.info("No predictions generated yet. Go to the Live Predictor tab to start tracking!")