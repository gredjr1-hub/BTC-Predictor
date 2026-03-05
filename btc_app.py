import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import joblib
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# --- 1. Page Setup & Live Sync ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin AI Trading Terminal")

st.sidebar.markdown("### ⚙️ Terminal Settings")
auto_pilot = st.sidebar.toggle("Enable Live Sync (5m interval)")

if auto_pilot:
    st_autorefresh(interval=300000, key="data_refresh")
    st.sidebar.success("Live Sync Active: Fetching latest data every 5 minutes.")

# --- 2. Load the Brain ---
@st.cache_resource
def load_model():
    return joblib.load("btc_5m_rf_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Error: {e}")
    st.stop()

# --- 3. Data Fetchers (UPDATED TO KRAKEN) ---
def get_live_prediction_data():
    exchange = ccxt.kraken({'enableRateLimit': True})
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
    exchange = ccxt.kraken({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=288)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

def get_live_ticker_price():
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        ticker = exchange.fetch_ticker('BTC/USDT')
        return ticker['last']
    except:
        return None

# --- 4. Google Sheets Connection & Grader ---
def get_gspread_client():
    creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def load_history_from_sheets():
    try:
        client = get_gspread_client()
        sheet = client.open("BTC_AI_Tracker").sheet1
        records = sheet.get_all_records()
        
        if not records:
            return pd.DataFrame(), sheet
            
        df = pd.DataFrame(records)
        df['Prediction_Time'] = pd.to_datetime(df['Prediction_Time'])
        df['Target_Time'] = pd.to_datetime(df['Target_Time'])
        return df, sheet
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame(), None

def resolve_pending_trades_in_sheets(live_data, history_df, sheet):
    if history_df.empty or sheet is None: return history_df
    
    pending_mask = history_df['Outcome'] == 'Pending'
    latest_live_time = live_data.index[-1]
    
    for idx, row in history_df[pending_mask].iterrows():
        target_time = row['Target_Time']
        
        if target_time <= latest_live_time:
            valid_candles = live_data[live_data.index >= target_time]
            
            if not valid_candles.empty:
                actual_close = valid_candles['Close'].iloc[0]
                entry_price = float(row['Entry_Price'])
                prediction = row['Prediction']
                
                outcome = 'Win' if (prediction == 'UP' and actual_close > entry_price) or (prediction == 'DOWN' and actual_close < entry_price) else 'Loss'
                
                history_df.at[idx, 'Close_Price'] = round(actual_close, 2)
                history_df.at[idx, 'Outcome'] = outcome
                
                sheet.update_cell(idx + 2, 6, round(actual_close, 2))
                sheet.update_cell(idx + 2, 7, outcome)
                
    return history_df

# --- 5. UI Layout (Tabs) ---
tab1, tab2 = st.tabs(["🔮 Live Predictor", "📊 Analytics & 24h Visualizer"])

with tab1:
    st.markdown("### Generate Next Move")
    if st.button("Generate Live Prediction", type="primary"):
        with st.spinner("Fetching live data & consulting AI..."):
            live_data = get_live_prediction_data()
            history_df, sheet = load_history_from_sheets()
            
            history_df = resolve_pending_trades_in_sheets(live_data, history_df, sheet)
            
            current_state = live_data.iloc[-1:]
            current_price = float(current_state['Close'].values[0])
            current_time = current_state.index[0]
            
            if not history_df.empty and history_df['Prediction_Time'].iloc[-1] == current_time:
                st.warning("Prediction already generated for this minute. Wait for the next candle.")
            else:
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
                
                if sheet:
                    new_row = [
                        str(current_time),
                        current_price,
                        direction,
                        round(confidence_pct, 2),
                        str(current_time + timedelta(minutes=5)),
                        "",
                        "Pending"
                    ]
                    sheet.append_row(new_row)
                    st.success("✅ Prediction successfully logged to Google Sheets!")

with tab2:
    st.markdown("### Model Performance Analytics")
    
    history, _ = load_history_from_sheets()
    
    if not history.empty:
        completed_trades = history[history['Outcome'].isin(['Win', 'Loss'])]
        total_completed = len(completed_trades)
        
        # --- Thermal Streaks ---
        def calculate_streak(history_df, hours):
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            window = history_df[(history_df['Prediction_Time'] >= cutoff) & (history_df['Outcome'].isin(['Win', 'Loss']))]
            total = len(window)
            if total == 0: return "Not Enough Data", "gray"
            wins = len(window[window['Outcome'] == 'Win'])
            win_rate = (wins / total) * 100
            
            if total < 3: return f"{win_rate:.0f}% ({wins}/{total}) - Neutral", "gray"
            elif win_rate >= 60: return f"{win_rate:.0f}% ({wins}/{total}) - 🔥 HOT", "green"
            elif win_rate <= 45: return f"{win_rate:.0f}% ({wins}/{total}) - ❄️ COLD", "red"
            else: return f"{win_rate:.0f}% ({wins}/{total}) - ⚖️ CHOP", "orange"

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
            
            recent_history = history[history['Prediction_Time'] >= chart_data.index[0]]
            
            for _, row in recent_history.iterrows():
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

        st.divider()

        # --- Live Market & Advanced Stats ---
        st.markdown("#### ⚡ Live Market & Advanced Stats")
        
        @st.fragment(run_every=2)  # Refreshes every 2 seconds
    def live_price_ticker():
        live_price = get_live_ticker_price()
        if live_price:
            # We use a container to ensure the metric doesn't "jump" or duplicate
            with st.container():
                st.metric(
                    label="BTC/USDT Live Price (Kraken)",
                    value=f"${live_price:,.2f}",
                    delta=None # You can calculate a 2s change here if you want!
                )
                
                # Re-check open bets against the freshest price
                pending_trades = history[history['Outcome'] == 'Pending']
                if not pending_trades.empty:
                    st.markdown("🔍 **Real-Time PnL Tracking:**")
                    for _, row in pending_trades.iterrows():
                        entry = float(row['Entry_Price'])
                        direction = row['Prediction']
                        diff = live_price - entry if direction == 'UP' else entry - live_price
                        status_color = "green" if diff > 0 else "red"
                        
                        st.markdown(
                            f"**{direction}** from ${entry:,.2f} → "
                            f":{status_color}[${abs(diff):,.2f} {'Profit' if diff > 0 else 'Loss'}]"
                        )
        else:
            st.write("⌛ Syncing with Kraken...")

        with col_stats1:
            st.markdown("**Core Win Rates:**")
            wins = len(completed_trades[completed_trades['Outcome'] == 'Win'])
            overall_wr = (wins / total_completed * 100) if total_completed > 0 else 0
            st.write(f"- Overall Win Rate: **{overall_wr:.1f}%**")
            
            if total_completed > 0:
                p90_threshold = completed_trades['Confidence'].quantile(0.90)
                p90_trades = completed_trades[completed_trades['Confidence'] >= p90_threshold]
                p90_wins = len(p90_trades[p90_trades['Outcome'] == 'Win'])
                p90_wr = (p90_wins / len(p90_trades) * 100) if len(p90_trades) > 0 else 0
                st.write(f"- Top 10% Conf Win Rate: **{p90_wr:.1f}%** *(>={p90_threshold:.1f}% Conf)*")
            else:
                st.write("- Top 10% Conf Win Rate: **N/A**")

        with col_stats2:
            st.markdown("**Direction & Conviction:**")
            if total_completed > 0:
                up_trades = completed_trades[completed_trades['Prediction'] == 'UP']
                up_wr = (len(up_trades[up_trades['Outcome']=='Win']) / len(up_trades) * 100) if len(up_trades) > 0 else 0
                
                down_trades = completed_trades[completed_trades['Prediction'] == 'DOWN']
                down_wr = (len(down_trades[down_trades['Outcome']=='Win']) / len(down_trades) * 100) if len(down_trades) > 0 else 0
                
                st.write(f"- Long (UP) Win Rate: **{up_wr:.1f}%**")
                st.write(f"- Short (DOWN) Win Rate: **{down_wr:.1f}%**")
                
                avg_conf_win = completed_trades[completed_trades['Outcome'] == 'Win']['Confidence'].mean()
                avg_conf_loss = completed_trades[completed_trades['Outcome'] == 'Loss']['Confidence'].mean()
                
                avg_conf_win = avg_conf_win if pd.notna(avg_conf_win) else 0.0
                avg_conf_loss = avg_conf_loss if pd.notna(avg_conf_loss) else 0.0
                
                st.caption(f"Avg Conf on Wins: {avg_conf_win:.1f}% | On Losses: {avg_conf_loss:.1f}%")

        st.divider()

        # --- The Data Table ---
        st.markdown("#### Cloud Tracker Log")
        def highlight_outcome(val):
            if val == 'Win': return 'background-color: rgba(0, 255, 0, 0.15)'
            elif val == 'Loss': return 'background-color: rgba(255, 0, 0, 0.15)'
            return ''
            
        st.dataframe(history.iloc[::-1].style.map(highlight_outcome, subset=['Outcome']), use_container_width=True)
        
    else:
        st.info("No predictions found in the Google Sheet yet. Run a prediction to start tracking!")