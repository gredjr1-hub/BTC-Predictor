import pandas as pd
import numpy as np
import ta

def process_macro_data(input_file, output_file):
    print(f"Loading massive macro dataset from {input_file}...")
    
    # CryptoDataDownload CSVs usually have a weird text warning on row 1, so we skip it (skiprows=1)
    try:
        df = pd.read_csv(input_file, skiprows=1)
    except Exception as e:
        print("Failed to read with skiprows=1. Trying standard read...")
        df = pd.read_csv(input_file)

    # Standardize Column Names (CryptoDataDownload uses lowercase and weird volume names)
    print("Standardizing columns...")
    rename_map = {
        'unix': 'Timestamp',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'Volume USDT': 'Volume',  # Sometimes it's Volume USDT
        'Volume USD': 'Volume',
        'volume': 'Volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Keep only the columns we need
    required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[col for col in required_cols if col in df.columns]]
    
    # Convert timestamps and SORT from oldest to newest (crucial for rolling math)
    print("Sorting timeline from oldest to newest...")
    # Some datasets are in milliseconds, some in seconds
    if df['Timestamp'].max() > 1e11:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        
    df.sort_values('Timestamp', ascending=True, inplace=True)
    df.set_index('Timestamp', inplace=True)

    print(f"Dataset contains {len(df):,} minutes of historical data!")

    print("Calculating Technical Indicators (This might take a minute)...")
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    
    # The Target: Did the price go strictly UP 5 minutes from this row?
    print("Mapping future targets...")
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    print("Cleaning up math anomalies...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"Saving final master dataset to {output_file}...")
    df.to_csv(output_file)
    print("✅ Injection Complete!")

if __name__ == "__main__":
    process_macro_data("macro_raw.csv", "BTCUSDT_1m_processed.csv")