import ccxt
import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
import time
import os
import json
import datetime as _dt

MASTER_FILE = "BTCUSDT_1m_processed.csv"
MODEL_FILE = "btc_5m_rf_model.joblib"

def fetch_recent_kraken_data(days=7):
    print(f"📡 Fetching the last {days} days of live Kraken data...")
    exchange = ccxt.kraken({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    timeframe = '1m'
    
    target_candles = days * 24 * 60
    all_ohlcv = []
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    
    while len(all_ohlcv) < target_candles:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 60000 
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.drop_duplicates(subset=['Timestamp'], inplace=True)
    return df

def build_features(df):
    print("🧮 Calculating technical indicators for the new data...")
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    df['Price_Delta_From_Window_Start'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)

    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df['Timestamp'].dt.minute % 5 == 0]  # train only on boundary candles (:00/:05/:10...)
    return df

def stitch_and_train(new_data):
    print("🔗 Loading the Macro Master Dataset...")
    if os.path.exists(MASTER_FILE):
        master_df = pd.read_csv(MASTER_FILE)
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp'])
    else:
        print("Master file not found. Creating a new one.")
        master_df = pd.DataFrame()

    print("🧵 Stitching new data to the bottom of the master file...")
    combined_df = pd.concat([master_df, new_data])
    
    # Drop any overlapping minutes and sort from oldest to newest
    combined_df.drop_duplicates(subset=['Timestamp'], keep='last', inplace=True)
    combined_df.sort_values('Timestamp', ascending=True, inplace=True)
    
    print(f"💾 Saving expanded dataset ({len(combined_df):,} total rows)...")
    combined_df.to_csv(MASTER_FILE, index=False)
    
    print("🧠 Retraining the AI on the expanded timeline...")
    combined_df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    combined_df.dropna(inplace=True)
    combined_df.set_index('Timestamp', inplace=True)
    X = combined_df.drop(columns=['Target'])
    y = combined_df['Target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_FILE)
    print("✅ Model successfully upgraded and saved!")

    _prev_meta = {}
    try:
        with open("model_metadata.json") as _f:
            _prev_meta = json.load(_f)
    except Exception:
        pass
    _meta = {
        "retrained_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "update_brain.py",
        "total_rows": len(combined_df),
        "rows_per_horizon": None,
        "data_start": str(combined_df.index.min()),
        "data_end":   str(combined_df.index.max()),
        "new_rows_added": len(combined_df) - _prev_meta.get("total_rows", len(combined_df)),
        "previous_total_rows": _prev_meta.get("total_rows"),
        "previous_retrained_at_utc": _prev_meta.get("retrained_at_utc"),
    }
    with open("model_metadata.json", "w") as _f:
        json.dump(_meta, _f, indent=2)
    print("📋 Saved model_metadata.json")

if __name__ == "__main__":
    recent_data = fetch_recent_kraken_data(days=7)
    processed_recent_data = build_features(recent_data)
    stitch_and_train(processed_recent_data)