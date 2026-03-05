import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib 

def train_production_model(filepath):
    print("Loading processed data (this takes a few seconds)...")
    df = pd.read_csv(filepath)
    df.set_index('Timestamp', inplace=True)

    print("Cleaning up Infinities from division-by-zero volume spikes...")
    # Replace Infinity with NaN, then drop those broken rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Separate our technical indicators (X) from our target (y)
    X = df.drop(columns=['Target'])
    y = df['Target']

    print(f"Training PRODUCTION model on ALL {len(X):,} rows...")
    print("Note: Skipping the test split. The AI is consuming 100% of the data.")

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,      
        max_depth=5,           
        random_state=42,
        n_jobs=1 # Safely single-threaded for Mac
    )

    print("Training the model (this will take 1 to 3 minutes)...")
    model.fit(X, y)

    # Save the production model to disk
    model_filename = "btc_5m_rf_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\n✅ Production model saved successfully as {model_filename}")

    # Extract Feature Importance to see what the final brain prioritizes
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Most Important Patterns/Indicators:")
    print(feature_importance.head(5))

if __name__ == "__main__":
    # Point to your processed CSV
    train_production_model("BTCUSDT_1m_processed.csv")