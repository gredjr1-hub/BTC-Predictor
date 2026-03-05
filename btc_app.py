import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import ccxt

# --- 1. Page Setup & Live Sync ---
st.set_page_config(page_title="Crypto AI Predictor", layout="wide")
st.title("🤖 Bitcoin AI Trading Terminal (Live Viewer)")

st.sidebar.markdown("### ⚙️ Terminal Settings")
auto_pilot = st.sidebar.toggle("Enable Live Sync (5m interval)")

if auto_pilot:
    st_autorefresh(interval=300000, key="data_refresh")
    st.sidebar.success("Live Sync Active: Fetching latest data every 5 minutes.")

# --- 2. Google Sheets Connection ---
@st.cache_data(ttl=60) # Cache for 60 seconds so we don't spam Google's API
def load_history_from_sheets():
    try:
        # Load credentials securely from Streamlit Secrets
        creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open the sheet
        sheet = client.open("BTC_AI_Tracker").sheet1
        records = sheet.get_all_records()
        
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        df['Prediction_Time'] = pd.to_datetime(df['Prediction_Time'])
        df['Target_Time'] = pd.to_datetime(df['Target_Time'])
        return df
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return pd.DataFrame()

def get_24h_chart_data():
    """Fetches 5m data specifically to render the background chart."""
    exchange = ccxt.binanceus({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=288)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

def calculate_streak(history, hours):
    """Calculates the win rate over a specific rolling time window."""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    window_data = history[(history['Prediction_Time'] >= cutoff_time) & (history['Outcome'].isin(['Win', 'Loss']))]
    
    total = len(window_data)
    if total == 0:
        return "Not Enough Data", "gray"
        
    wins = len(window_data[window_data['Outcome'] == 'Win'])
    win_rate = (wins / total) * 100
    
    if total < 3: 
        return f"{win_rate:.0f}% ({wins}/{total}) - Neutral", "gray"
    elif win_rate >= 60:
        return f"{win_rate:.0f}% ({wins}/{total}) - 🔥 HOT", "green"
    elif win_rate <= 45:
        return f"{win_rate:.0f}% ({wins}/{total}) - ❄️ COLD", "red"
    else:
        return f"{win_rate:.0f}% ({wins}/{total}) - ⚖️ CHOP", "orange"

# --- 3. Render Dashboard ---
st.markdown("### Model Performance Analytics")

history = load_history_from_sheets()

if not history.empty:
    # --- System Status ---
    completed_trades = history[history['Outcome'].isin(['Win', 'Loss'])]
    total_completed = len(completed_trades)
    wins = len(completed_trades[completed_trades['Outcome'] == 'Win'])
    win_rate = (wins / total_completed * 100) if total_completed > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Completed Predictions", total_completed)
    m2.metric("Real-World Win Rate", f"{win_rate:.1f}%")
    m3.metric("Pending Predictions", len(history[history['Outcome'] == 'Pending']))
    
    st.divider()

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
    st.markdown("#### Cloud Tracker Log")
    def highlight_outcome(val):
        if val == 'Win': return 'background-color: rgba(0, 255, 0, 0.15)'
        elif val == 'Loss': return 'background-color: rgba(255, 0, 0, 0.15)'
        return ''
        
    st.dataframe(history.iloc[::-1].style.map(highlight_outcome, subset=['Outcome']), use_container_width=True)
    
else:
    st.info("No predictions found in the Google Sheet. Waiting for the headless bot to log its first trade...")