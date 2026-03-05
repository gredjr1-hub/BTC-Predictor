import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# --- 1. Page Setup & Auto-Pilot ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin 5-Minute AI Predictor")

st.sidebar.markdown("### ⚙️ Bot Settings")
auto_pilot = st.sidebar.toggle("Enable Auto-Pilot (5m interval)")

if auto_pilot:
    # 300,000 milliseconds = exactly 5 minutes
    st_autorefresh(interval=300000, key="data_refresh")
    st.sidebar.success("Auto-Pilot Active: Scanning every 5 minutes.")

# --- 2. Load the Brain ---
@st.cache_resource
def load_model():
    return joblib.load("btc_5m_rf_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Error: {e}")
    st.stop()

# --- 3. Data Fetchers ---
def get_live_prediction_data():
    """Fetches 1m data for the AI to calculate strict technical features."""
    exchange = ccxt.binanceus({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=100)
    
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
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

def get_24h_chart_data():
    """Fetches 5m data specifically to render a clean 24-hour visual chart."""
    exchange = ccxt.binanceus({'enableRateLimit': True})
    # 24 hours = 288 5-minute candles
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=288)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

# --- 4. Tracker System ---
HISTORY_FILE = "trade_history.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df['Prediction_Time'] = pd.to_datetime(df['Prediction_Time'])
        df['Target_Time'] = pd.to_datetime(df['Target_Time'])
        return df
    else:
        return pd.DataFrame(columns=['Prediction_Time', 'Entry_Price', 'Prediction', 'Confidence', 'Target_Time', 'Close_Price', 'Outcome'])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

def resolve_pending_trades(live_data):
    history = load_history()
    if history.empty: return history
    
    pending_mask = history['Outcome'] == 'Pending'
    latest_live_time = live_data.index[-1] 
    
    for idx, row in history[pending_mask].iterrows():
        target_time = row['Target_Time']
        if target_time <= latest_live_time:
            valid_candles = live_data[live_data.index >= target_time]
            if not valid_candles.empty:
                actual_close = valid_candles['Close'].iloc[0]
                entry_price = row['Entry_Price']
                prediction = row['Prediction']
                
                history.at[idx, 'Close_Price'] = round(actual_close, 2)
                if prediction == 'UP' and actual_close > entry_price:
                    history.at[idx, 'Outcome'] = 'Win'
                elif prediction == 'DOWN' and actual_close < entry_price:
                    history.at[idx, 'Outcome'] = 'Win'
                else:
                    history.at[idx, 'Outcome'] = 'Loss'
                    
    save_history(history)
    return history

def calculate_streak(history, hours):
    """Calculates the win rate over a specific rolling time window."""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    window_data = history[(history['Prediction_Time'] >= cutoff_time) & (history['Outcome'].isin(['Win', 'Loss']))]
    
    total = len(window_data)
    if total == 0:
        return "Not Enough Data", "gray"
        
    wins = len(window_data[window_data['Outcome'] == 'Win'])
    win_rate = (wins / total) * 100
    
    if total < 3: # Need at least 3 trades to call it a streak
        return f"{win_rate:.0f}% ({wins}/{total}) - Neutral", "gray"
    elif win_rate >= 60:
        return f"{win_rate:.0f}% ({wins}/{total}) - 🔥 HOT", "green"
    elif win_rate <= 45:
        return f"{win_rate:.0f}% ({wins}/{total}) - ❄️ COLD", "red"
    else:
        return f"{win_rate:.0f}% ({wins}/{total}) - ⚖️ CHOP", "orange"

# --- 5. UI Layout (Tabs) ---
tab1, tab2 = st.tabs(["🔮 Live Predictor", "📊 Analytics & 24h Visualizer"])

with tab1:
    st.markdown("### Generate Next Move")
    if st.button("Generate Live Prediction", type="primary") or auto_pilot:
        with st.spinner("Fetching live data & analyzing..."):
            live_data = get_live_prediction_data()
            resolve_pending_trades(live_data)
            
            current_state = live_data.iloc[-1:]
            current_price = current_state['Close'].values[0]
            current_time = current_state.index[0]
            
            prediction_val = model.predict(current_state)[0]
            probabilities = model.predict_proba(current_state)[0]
            
            direction = "UP" if prediction_val == 1 else "DOWN"
            confidence = probabilities[1] if prediction_val == 1 else probabilities[0]
            confidence_pct = confidence * 100
            
            if confidence_pct < 55:
                signal_strength = "⚠️ WEAK SIGNAL (Coin Flip - Do not trade)"
                color = "orange"
            elif confidence_pct < 60:
                signal_strength = "✅ MODERATE SIGNAL (Standard Edge)"
                color = "blue"
            else:
                signal_strength = "🔥 STRONG SIGNAL (High Probability)"
                color = "green"

            st.markdown(f"### Target Time: {(current_time + timedelta(minutes=5)).strftime('%H:%M:%S')} UTC")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current BTC Price", f"${current_price:,.2f}")
            col2.metric("AI Prediction", f"{direction}", delta="Long" if direction=="UP" else "-Short")
            col3.metric("AI Confidence", f"{confidence_pct:.1f}%")
            
            st.markdown(f"**Signal Strength:** :{color}[{signal_strength}]")
            
            history = load_history()
            # Prevent logging the exact same minute twice if a manual button mash happens during autopilot
            if history.empty or history['Prediction_Time'].iloc[-1] != current_time:
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
    st.markdown("### Model Performance Analytics")
    
    live_data = get_live_prediction_data()
    history = resolve_pending_trades(live_data)
    
    if not history.empty:
        # --- Thermal Streaks ---
        st.markdown("#### Model Temperature (Rolling Windows)")
        s1, s2, s3 = st.columns(3)
        
        rate_1h, color_1h = calculate_streak(history, 1)
        rate_12h, color_12h = calculate_streak(history, 12)
        rate_24h, color_24h = calculate_streak(history, 24)
        
        s1.markdown(f"**Last 1 Hour:** :{color_1h}[{rate_1h}]")
        s2.markdown(f"**Last 12 Hours:** :{color_12h}[{rate_12h}]")
        s3.markdown(f"**Last 24 Hours:** :{color_24h}[{rate_24h}]")
        st.divider()

        # --- The 24h Visualizer ---
        st.markdown("#### 24-Hour Trade Overlay")
        with st.spinner("Rendering 24-hour chart..."):
            chart_data = get_24h_chart_data()
            
            fig = go.Figure(data=[go.Candlestick(x=chart_data.index,
                            open=chart_data['Open'],
                            high=chart_data['High'],
                            low=chart_data['Low'],
                            close=chart_data['Close'],
                            name="BTC/USDT")])
            
            # Overlay Predictions
            recent_history = history[history['Prediction_Time'] >= chart_data.index[0]]
            
            for _, row in recent_history.iterrows():
                # Determine marker style
                symbol = 'triangle-up' if row['Prediction'] == 'UP' else 'triangle-down'
                color = 'green' if row['Outcome'] == 'Win' else 'red' if row['Outcome'] == 'Loss' else 'yellow'
                size = 14 if row['Outcome'] != 'Pending' else 10
                
                fig.add_trace(go.Scatter(
                    x=[row['Prediction_Time']],
                    y=[row['Entry_Price']],
                    mode='markers',
                    marker=dict(symbol=symbol, size=size, color=color, line=dict(width=2, color='white')),
                    name=f"{row['Outcome']} ({row['Prediction']})",
                    hoverinfo='text',
                    hovertext=f"{row['Prediction']} | Conf: {row['Confidence']}% | {row['Outcome']}",
                    showlegend=False
                ))

            fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # --- The Data Table ---
        st.markdown("#### Permanent Tracker Log")
        def highlight_outcome(val):
            if val == 'Win': return 'background-color: rgba(0, 255, 0, 0.15)'
            elif val == 'Loss': return 'background-color: rgba(255, 0, 0, 0.15)'
            return ''
            
        # Reverse dataframe to show newest at the top
        st.dataframe(history.iloc[::-1].style.map(highlight_outcome, subset=['Outcome']), use_container_width=True)
        
    else:
        st.info("No predictions generated yet. Go to the Live Predictor tab to start tracking!")