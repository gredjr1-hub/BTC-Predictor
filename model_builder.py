import pandas as pd
import ta

def prepare_data(filepath):
    print("Loading data...")
    # READ CSV INSTEAD OF PARQUET
    df = pd.read_csv(filepath)

    # CSVs lose their datetime index type when saved, so we must restore it
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    # NOTE: Do NOT boundary-filter here — indicators must be computed on raw 1-min data
    # so that lookback windows match live inference (which uses 1-min candles).

    print("Calculating Technical Indicators (Features)...")

    # 1. Momentum & Trend Indicators on raw 1-min data
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)

    print("Computing Window Context Features...")
    # 2. Window context features — each row knows its 5-min window boundaries
    df['window_start_time'] = df.index.floor('5min')
    df['window_start_price'] = df.groupby('window_start_time')['Close'].transform('first')
    df['window_end_time'] = df['window_start_time'] + pd.Timedelta(minutes=5)

    # window_end_price: the close of the candle at the next :00/:05 boundary
    boundary_closes = df[df.index.minute % 5 == 0][['Close']].copy()
    boundary_closes.index.name = 'window_end_time'
    df = df.join(boundary_closes.rename(columns={'Close': 'window_end_price'}), on='window_end_time')

    # Variable-horizon features (model knows how much time remains in the window)
    df['minutes_since_window_start'] = df.index.minute % 5  # 0 at :00, 1 at :01, ..., 4 at :04
    df['minutes_to_window_end'] = 5 - df['minutes_since_window_start']  # 5 at :00, 1 at :04
    df['price_change_since_window_start'] = (
        (df['Close'] - df['window_start_price']) / df['window_start_price']
    )

    print("Generating Target Variable...")
    # 3. True target: did price end higher than window_start_price at window close?
    df['Target'] = (df['window_end_price'] > df['window_start_price']).astype(int)

    print("Cleaning up data...")
    # 4. Cleanup
    df.drop(columns=['window_start_time', 'window_end_time', 'window_start_price', 'window_end_price'], inplace=True)
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)

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