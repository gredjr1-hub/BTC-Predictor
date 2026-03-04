import pandas as pd
import pandas_ta as ta

def prepare_data(filepath):
    print("Loading data...")
    # Load the compressed parquet file we generated earlier
    df = pd.read_parquet(filepath)
    
    print("Calculating Technical Indicators (Features)...")
    # 1. Momentum & Trend Indicators
    # RSI (Relative Strength Index) - 14 period is standard
    df.ta.rsi(length=14, append=True)
    
    # MACD (Moving Average Convergence Divergence)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Moving Averages (Exponential) - 9 and 21 minute EMAs
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    
    # Bollinger Bands (Volatility)
    df.ta.bbands(length=20, std=2, append=True)

    # Volume Rate of Change (Is volume spiking?)
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)

    print("Generating Target Variable...")
    # 2. The Target (What we want to predict)
    # We want to know the price 5 minutes in the future.
    # We use shift(-5) to pull the closing price from 5 rows DOWN into the current row.
    df['Future_Close_5m'] = df['Close'].shift(-5)
    
    # Create the binary classification label: 
    # 1 if Future_Close is greater than current Close (UP)
    # 0 if Future_Close is less than or equal to current Close (DOWN)
    df['Target'] = (df['Future_Close_5m'] > df['Close']).astype(int)

    print("Cleaning up data...")
    # 3. Cleanup
    # Calculating moving averages like a 26-period MACD means the first 26 rows will be NaN.
    # Shifting the target -5 means the LAST 5 rows will be NaN.
    # Machine learning models hate NaNs, so we drop them.
    df.dropna(inplace=True)
    
    # Drop the Future_Close column so our model doesn't accidentally use it to cheat
    df.drop(columns=['Future_Close_5m'], inplace=True)
    
    print(f"Data ready! Final shape: {df.shape}")
    
    # Save the processed data for our ML model and Streamlit app to use
    processed_filename = filepath.replace('history.parquet', 'processed.parquet')
    df.to_parquet(processed_filename)
    print(f"Processed data saved to {processed_filename}")
    
    return df

if __name__ == "__main__":
    # Make sure this matches the filename output by your data_fetcher script
    processed_df = prepare_data("BTCUSDT_1m_history.parquet")
    
    # Let's peek at the balance of our target variable
    up_count = (processed_df['Target'] == 1).sum()
    down_count = (processed_df['Target'] == 0).sum()
    total = len(processed_df)
    
    print("\n--- Target Balance ---")
    print(f"UP (1): {up_count:,} ({up_count/total*100:.2f}%)")
    print(f"DOWN (0): {down_count:,} ({down_count/total*100:.2f}%)")