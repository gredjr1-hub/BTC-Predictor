import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

FEATURE_COLS = [
    'RSI_14', 'MACD', 'MACD_Signal', 'EMA_9', 'EMA_21',
    'BB_Upper', 'BB_Lower', 'Volume_ROC',
    'price_change_since_window_start', 'price_change_abs'
]

def train_production_model(filepath):
    print("Loading processed data (this takes a few seconds)...")
    df = pd.read_csv(filepath)
    df.set_index('Timestamp', inplace=True)

    print("Cleaning up Infinities from division-by-zero volume spikes...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Derived feature: magnitude of within-window price move (direction-agnostic)
    # Lets the RF learn that extreme moves in either direction signal reversal risk
    df['price_change_abs'] = df['price_change_since_window_start'].abs()

    print(f"Total rows available: {len(df):,}")

    models = {}
    for mins_left in range(1, 6):
        subset = df[df['minutes_to_window_end'] == mins_left]
        X = subset[FEATURE_COLS]
        y = subset['Target']

        print(f"\nTraining model for minutes_to_window_end={mins_left} ({len(X):,} rows)...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=1  # Safely single-threaded for Mac
        )
        rf.fit(X, y)
        models[mins_left] = rf

        fi = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': rf.feature_importances_})
        fi = fi.sort_values('Importance', ascending=False)
        print(f"  Top 3 features:")
        print(fi.head(3).to_string(index=False))

    model_filename = "btc_5m_rf_model.joblib"
    joblib.dump(models, model_filename)
    print(f"\n✅ Saved dict of 5 horizon-specific models to {model_filename}")
    print("  Keys: {1: 1-min-left model, 2: 2-min-left, ..., 5: 5-min-left (boundary)}")

if __name__ == "__main__":
    train_production_model("BTCUSDT_1m_processed.csv")
