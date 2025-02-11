import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from transformers import pipeline
from datetime import datetime, timedelta
import joblib  # For loading your trained model
import numpy as np

# Ensure page configuration is set first
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Load FinBERT model efficiently
@st.cache_resource
def load_sentiment_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

# Load your trained prediction model
@st.cache_resource  # Cache the loaded model
def load_prediction_model(model_path):  # Pass the path to your model
    try:
        model = joblib.load(model_path)  # Load using joblib
        return model
    except Exception as e:
        st.error(f"Error loading prediction model: {e}")  # Handle loading errors
        return None

# Replace 'your_model.pkl' with the actual path to your saved model file
MODEL_PATH = "your_model.pkl"  # **Important:** Set the path to your model file
prediction_model = load_prediction_model(MODEL_PATH)


# Function to fetch stock data (optimized with caching)
@st.cache_data
def fetch_stock_data(stock_ticker, start_date, end_date):
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")
        if stock_data.empty:
            return None
        stock_data = stock_data.reset_index()
        stock_data['Date'] = stock_data['Date'].astype(str)
        return stock_data
    except Exception as e:
        return None

# ... (rest of the functions: fetch_current_stock_info, fetch_news, analyze_sentiment remain the same)


# Streamlit UI
st.title("📈 Stock Market Analyzer")

stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):").upper()
date = st.date_input("Select Date for Analysis:", datetime.today())

if stock_ticker:
    start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = date.strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(stock_ticker, start_date, end_date)
    stock_info = fetch_current_stock_info(stock_ticker)
    news = fetch_news(stock_ticker)

    col1, col2 = st.columns(2)

    with col1:  # Stock Data and Chart
        if stock_data is not None:
            # ... (stock chart plotting code remains the same)

            st.subheader("📈 Current Stock Information")
            if stock_info:
                # ... (display stock info remains the same)
            else:
                st.error("❌ No market data found for this stock.")

        else:
            st.error("❌ No stock data available! Please check the ticker symbol.")


    with col2:  # News and Sentiment + Prediction
        st.subheader("📰 Latest News & Sentiment")
        if news:
            sentiments = analyze_sentiment(news)
            # ... (display news and sentiment remains the same)
        else:
            st.write("No news available.")

        # Make Prediction if model and data are available
        if prediction_model and stock_data is not None:
            try:
                # Prepare data for prediction (this will be VERY model-specific)
                # **Crucial:**  You MUST adapt this to how your model was trained.

                # Example: If your model used closing prices as input:
                last_30_days_data = stock_data['Close'].values[-30:]  # Get the last 30 days of closing prices
                if len(last_30_days_data) < 30:
                    st.warning("Not enough data for prediction. Need at least 30 days.")
                else:
                    # Reshape for sklearn if needed:
                    input_data = last_30_days_data.reshape(1, -1)  # Reshape to (1, 30) for a single prediction

                    prediction = prediction_model.predict(input_data)[0]  # Get the prediction
                    st.subheader("🔮 Stock Price Prediction")
                    st.write(f"Predicted Value: {prediction}")  # Display the prediction

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        elif prediction_model is None:
            st.warning("Prediction model not loaded. Check the file path.")

else:
    st.write("Please enter a stock ticker to begin.")
