import pandas as pd
import numpy as np
import ta
import ccxt
import joblib
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import json
import os

# --- 1. Google Sheets Auth ---
def get_sheet():
    # Load credentials securely from GitHub Actions secret
    creds_json = os.environ.get('GCP_CREDENTIALS')
    creds_dict = json.loads(creds_json)
    
    # THE FIX: Add the Google Drive scope so it can search for the file by name
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    
    # Open your exact sheet name
    sheet = client.open("BTC_AI_Tracker").sheet1
    return sheet

# --- 2. Live Data ---
def get_data():
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

# --- 3. Grade Pending Trades ---
def grade_trades(sheet, live_data):
    records = sheet.get_all_records()
    if not records: return
    
    latest_live_time = live_data.index[-1]
    
    for i, row in enumerate(records):
        if row['Outcome'] == 'Pending':
            # gspread returns strings, so we parse the target time
            target_time = pd.to_datetime(row['Target_Time'])
            
            if target_time <= latest_live_time:
                valid_candles = live_data[live_data.index >= target_time]
                
                if not valid_candles.empty:
                    actual_close = valid_candles['Close'].iloc[0]
                    entry = float(row['Entry_Price'])
                    prediction = row['Prediction']
                    
                    outcome = 'Win' if (prediction == 'UP' and actual_close > entry) or (prediction == 'DOWN' and actual_close < entry) else 'Loss'
                    
                    # Update Google Sheet (i+2 accounts for 0-index and header row)
                    sheet.update_cell(i + 2, 6, round(actual_close, 2))
                    sheet.update_cell(i + 2, 7, outcome)

# --- 4. Main Execution ---
if __name__ == "__main__":
    print(f"[{datetime.utcnow()}] Waking up headless bot...")
    
    sheet = get_sheet()
    live_data = get_data()
    
    # 1. Grade the past
    print("Grading pending trades...")
    grade_trades(sheet, live_data)
    
    # 2. Predict the future
    print("Loading AI Brain...")
    model = joblib.load("btc_5m_rf_model.joblib")
    
    current_state = live_data.iloc[-1:]
    current_price = current_state['Close'].values[0]
    current_time = current_state.index[0]
    
    prediction_val = model.predict(current_state)[0]
    probs = model.predict_proba(current_state)[0]
    
    direction = "UP" if prediction_val == 1 else "DOWN"
    conf = probs[1] if prediction_val == 1 else probs[0]
    
    # 3. Save to Google Sheets
    new_row = [
        str(current_time),
        float(current_price),
        direction,
        round(conf * 100, 2),
        str(current_time + timedelta(minutes=5)),
        "", # Close_Price (Blank until graded)
        "Pending" # Outcome
    ]
    
    sheet.append_row(new_row)
    print(f"[{datetime.utcnow()}] Successfully appended {direction} prediction to Google Sheets. Going back to sleep.")