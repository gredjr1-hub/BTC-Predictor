import pandas as pd
import ta

def prepare_data(filepath):
    print("Loading data...")
    # READ CSV INSTEAD OF PARQUET
    df = pd.read_csv(filepath)
    
    # CSVs lose their datetime index type when saved, so we must restore it
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df = df[df.index.minute % 5 == 0]  # train only on boundary candles (:00/:05/:10...)

    print("Calculating Technical Indicators (Features)...")
    
    # 1. Momentum & Trend Indicators using 'ta'
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    df['Price_Delta_From_Window_Start'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)

    print("Generating Target Variable...")
    # 2. The Target (Predicting 5 minutes into the future)
    df['Future_Close_5m'] = df['Close'].shift(-5)
    df['Target'] = (df['Future_Close_5m'] > df['Close']).astype(int)

    print("Cleaning up data...")
    # 3. Cleanup
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['Future_Close_5m'], inplace=True)
    
    print(f"Data ready! Final shape: {df.shape}")
    
    # SAVE AS CSV
    processed_filename = filepath.replace('history.csv', 'processed.csv')
    df.to_csv(processed_filename)
    print(f"Processed data saved to {processed_filename}")
    
    return df

if __name__ == "__main__":
    # Point to the CSV file you just downloaded
    processed_df = prepare_data("BTCUSDT_1m_history.csv")
    
    up_count = (processed_df['Target'] == 1).sum()
    down_count = (processed_df['Target'] == 0).sum()
    total = len(processed_df)
    
    print("\n--- Target Balance ---")
    print(f"UP (1): {up_count:,} ({up_count/total*100:.2f}%)")
    print(f"DOWN (0): {down_count:,} ({down_count/total*100:.2f}%)")