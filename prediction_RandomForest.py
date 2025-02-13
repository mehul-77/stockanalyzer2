import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def ensure_features(df, required_features):
    # Add missing features with zero values
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    # Reorder columns to match the required features order
    df = df[required_features]
    return df

# Define the required features
REQUIRED_FEATURES = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "Daily_Return", "Sentiment_Score", "Headlines_Count",
    "Next_Day_Return", "Moving_Avg", "Rolling_Std_Dev",
    "RSI", "EMA", "ROC", "Sentiment_Numeric"
]

# Load the data
df = pd.read_csv("stock_sentiment_analysis_multivariate_imputed.csv")
df = ensure_features(df, REQUIRED_FEATURES)
print("Updated Data:", df.head())

# Load the trained Random Forest model
def load_rf_model():
    try:
        with open("random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return rf_model, scaler
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")
        return None, None

# Define preprocessing function
def preprocess_input_rf(data, scaler):
    """Preprocess input data for Random Forest model with correct features."""
    processed_data = scaler.transform(data)
    return processed_data

# Interpret prediction result
def interpret_prediction(score):
    if score > 0.6:
        return "📈 Positive Sentiment"
    elif score < 0.4:
        return "📉 Negative Sentiment"
    else:
        return "⚖️ Neutral Sentiment"

# Predict stock sentiment using Random Forest
def predict_stock_sentiment_rf(data):
    rf_model, scaler = load_rf_model()
    if rf_model is None or scaler is None:
        return "Error: Random Forest Model not loaded"
    processed_data = preprocess_input_rf(data, scaler)
    prediction = rf_model.predict(processed_data)[0]
    return interpret_prediction(prediction)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("stock_sentiment_analysis_multivariate_imputed.csv")
    df = ensure_features(df, REQUIRED_FEATURES)  # Ensure dataset has the required features
    print("Updated Data:", df.head())
    try:
        result_rf = predict_stock_sentiment_rf(df)
        print("Predicted Stock Sentiment (Random Forest):", result_rf)
    except Exception as e:
        print(f"Error during prediction: {e}")
