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

# Must match train_model.py exactly
FEATURE_COLS = [
    'RSI_14', 'MACD', 'MACD_Signal', 'EMA_9', 'EMA_21',
    'BB_Upper', 'BB_Lower', 'Volume_ROC',
    'price_change_since_window_start', 'price_change_abs'
]

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

    # Match column name used in master CSV and FEATURE_COLS
    df['price_change_since_window_start'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)

    # Window position columns (matching master CSV schema)
    df['minutes_since_window_start'] = df['Timestamp'].dt.minute % 5
    df['minutes_to_window_end'] = df['minutes_since_window_start'].map(lambda m: 5 - m if m != 0 else 5)

    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # Keep all rows (all 5 window positions) — same as master CSV
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
    combined_df = pd.concat([master_df, new_data], ignore_index=True)

    # Drop any overlapping minutes and sort from oldest to newest
    combined_df.drop_duplicates(subset=['Timestamp'], keep='last', inplace=True)
    combined_df.sort_values('Timestamp', ascending=True, inplace=True)

    print(f"💾 Saving expanded dataset ({len(combined_df):,} total rows)...")
    combined_df.to_csv(MASTER_FILE, index=False)

    print("🧠 Retraining on the expanded timeline — 5 horizon-specific models...")
    combined_df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    _dropna_cols = [c for c in FEATURE_COLS if c != 'price_change_abs'] + ['Target', 'minutes_to_window_end']
    combined_df.dropna(subset=_dropna_cols, inplace=True)
    combined_df.set_index('Timestamp', inplace=True)

    # Derived feature computed in-memory (not stored in CSV), same as train_model.py
    combined_df['price_change_abs'] = combined_df['price_change_since_window_start'].abs()

    print(f"Total rows available for training: {len(combined_df):,}")

    models = {}
    for mins_left in range(1, 6):
        subset = combined_df[combined_df['minutes_to_window_end'] == mins_left]
        X = subset[FEATURE_COLS]
        y = subset['Target']
        print(f"\nTraining model for minutes_to_window_end={mins_left} ({len(X):,} rows)...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        models[mins_left] = rf

        fi = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': rf.feature_importances_})
        print(f"  Top 3 features:\n{fi.sort_values('Importance', ascending=False).head(3).to_string(index=False)}")

    joblib.dump(models, MODEL_FILE)
    print(f"\n✅ Saved dict of 5 horizon-specific models to {MODEL_FILE}")

    # Per-horizon feature importances (matches train_model.py format)
    _feat_importance = {
        str(h): {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURE_COLS, rf.feature_importances_)
        }
        for h, rf in models.items()
    }

    _prev_meta = {}
    try:
        with open("model_metadata.json") as _f:
            _prev_meta = json.load(_f)
    except Exception:
        pass

    _model_version = _prev_meta.get("model_version", 0) + 1

    _meta = {
        "retrained_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "update_brain.py",
        "model_version": _model_version,
        "total_rows": len(combined_df),
        "rows_per_horizon": {str(h): int((combined_df["minutes_to_window_end"] == h).sum()) for h in range(1, 6)},
        "data_start": str(combined_df.index.min()),
        "data_end":   str(combined_df.index.max()),
        "new_rows_added": len(combined_df) - _prev_meta.get("total_rows", len(combined_df)),
        "previous_total_rows": _prev_meta.get("total_rows"),
        "previous_retrained_at_utc": _prev_meta.get("retrained_at_utc"),
        "feature_importance": _feat_importance,
    }
    with open("model_metadata.json", "w") as _f:
        json.dump(_meta, _f, indent=2)
    print("📋 Saved model_metadata.json")

    _hist_path = "model_history.json"
    _hist_entry = {
        "retrained_at_utc": _meta["retrained_at_utc"],
        "script": _meta["script"],
        "model_version": _model_version,
        "total_rows": _meta["total_rows"],
        "new_rows_added": _meta["new_rows_added"],
        "data_start": _meta["data_start"],
        "data_end": _meta["data_end"],
        "feature_importance": _feat_importance,
    }
    try:
        with open(_hist_path) as _f:
            _hist = json.load(_f)
    except Exception:
        _hist = []
    _hist.append(_hist_entry)
    with open(_hist_path, "w") as _f:
        json.dump(_hist, _f, indent=2)
    print("📋 Appended to model_history.json")

if __name__ == "__main__":
    recent_data = fetch_recent_kraken_data(days=7)
    processed_recent_data = build_features(recent_data)
    stitch_and_train(processed_recent_data)
