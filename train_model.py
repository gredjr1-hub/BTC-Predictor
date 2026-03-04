import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib 

def train_xgboost(filepath):
    print("Loading processed data...")
    df = pd.read_parquet(filepath)

    # Separate our technical indicators (X) from our target (y)
    X = df.drop(columns=['Target'])
    y = df['Target']

    print("Splitting data into training and testing sets...")
    # CRITICAL TRADING RULE: Never shuffle time-series data!
    # We must train on the past (first 80%) to predict the future (last 20%).
    # If you shuffle, the model looks into the future to predict the past.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Training on {len(X_train):,} rows, Testing on {len(X_test):,} rows.")

    # Initialize the XGBoost Classifier
    # We use conservative parameters to prevent it from memorizing market noise
    model = xgb.XGBClassifier(
        n_estimators=100,      # Number of trees
        learning_rate=0.05,    # How aggressively it learns
        max_depth=5,           # How complex each tree can get
        random_state=42,
        eval_metric='logloss'
    )

    print("Training the model (this will take a minute or two)...")
    model.fit(X_train, y_train)

    print("Evaluating the model on unseen future data...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Results ---")
    print(f"Baseline Accuracy: {accuracy * 100:.2f}%")
    
    # Precision is often more important than accuracy in trading. 
    # If it says UP, how often is it actually UP?
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Save the model to disk
    model_filename = "btc_5m_xgboost.joblib"
    joblib.dump(model, model_filename)
    print(f"\nModel saved successfully as {model_filename}")

    # Extract Feature Importance (The "Why")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Most Important Patterns/Indicators:")
    print(feature_importance.head(5))

if __name__ == "__main__":
    train_xgboost("BTCUSDT_1m_processed.parquet")