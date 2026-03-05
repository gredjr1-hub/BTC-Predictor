import pandas as pd
import ta

def prepare_data(filepath):
    print("Loading data...")
    df = pd.read_parquet(filepath)
    
    print("Calculating Technical Indicators (Features)...")
    
    # 1. Momentum & Trend Indicators using 'ta'
    # RSI
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD (We'll grab the main MACD line and the Signal line)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    
    # Moving Averages (EMA)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    
    # Bollinger Bands (Upper and Lower bands)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)

    # Volume Rate of Change
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)

    print("Generating Target Variable...")
    # 2. The Target (Predicting 5 minutes into the future)
    df['Future_Close_5m'] = df['Close'].shift(-5)
    df['Target'] = (df['Future_Close_5m'] > df['Close']).astype(int)

    print("Cleaning up data...")
    # 3. Cleanup
    df.dropna(inplace=True)
    df.drop(columns=['Future_Close_5m'], inplace=True)
    
    print(f"Data ready! Final shape: {df.shape}")
    
    processed_filename = filepath.replace('history.parquet', 'processed.parquet')
    df.to_parquet(processed_filename)
    print(f"Processed data saved to {processed_filename}")
    
    return df

if __name__ == "__main__":
    processed_df = prepare_data("BTCUSDT_1m_history.parquet")
    
    up_count = (processed_df['Target'] == 1).sum()
    down_count = (processed_df['Target'] == 0).sum()
    total = len(processed_df)
    
    print("\n--- Target Balance ---")
    print(f"UP (1): {up_count:,} ({up_count/total*100:.2f}%)")
    print(f"DOWN (0): {down_count:,} ({down_count/total*100:.2f}%)")