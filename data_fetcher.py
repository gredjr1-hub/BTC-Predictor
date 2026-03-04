import ccxt
import pandas as pd
import time
from datetime import datetime

def fetch_historical_data(symbol='BTC/USDT', timeframe='1m', years=2):
    # Initialize the exchange. 
    # 'enableRateLimit' is crucial: it tells CCXT to automatically pause 
    # between requests so we don't get banned by the exchange.
    exchange = ccxt.binanceus({
        'enableRateLimit': True, 
    })

    # Calculate timestamps in milliseconds (which the API requires)
    now = exchange.milliseconds()
    two_years_ms = years * 365 * 24 * 60 * 60 * 1000
    start_time = now - two_years_ms

    all_ohlcv = []
    current_time = start_time

    print(f"Starting download for {symbol} ({timeframe} intervals)...")
    print(f"From: {datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will take a few minutes as we page through ~1 million rows.")
    print("-" * 50)

    while current_time < now:
        try:
            # Fetch a chunk of data (usually max 1000 rows per request)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_time, limit=1000)
            
            if not ohlcv:
                # If the exchange returns an empty list, we've hit the end
                break 
                
            all_ohlcv.extend(ohlcv)
            
            # Update the current_time to the last fetched timestamp + 1 minute (60,000 ms)
            # This ensures we don't pull overlapping duplicate rows
            current_time = ohlcv[-1][0] + 60000 
            
            # Print a progress update every ~100,000 rows
            if len(all_ohlcv) % 100000 < 1000: 
                current_date = datetime.fromtimestamp(current_time/1000).strftime('%Y-%m-%d')
                print(f"Progress: Downloaded {len(all_ohlcv):,} rows. Currently at {current_date}")
                
        except Exception as e:
            print(f"Network or API Error: {e}")
            print("Sleeping for 15 seconds to cool off before retrying...")
            time.sleep(15)

    print("-" * 50)
    print(f"Download complete! Total rows: {len(all_ohlcv):,}")

    # Convert the raw list of lists into a Pandas DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Convert milliseconds timestamp to a readable DatetimeIndex
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # Save to Parquet (Highly compressed, very fast to load in Streamlit)
    filename = f"{symbol.replace('/', '')}_{timeframe}_history.parquet"
    df.to_parquet(filename)
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    fetch_historical_data()