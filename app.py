import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from transformers import pipeline
from datetime import datetime, timedelta
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import talib  # For technical indicators

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Load Sentiment Analysis Model
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

# Load Random Forest Model and Scaler
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"

# Feature Engineering Function
def engineer_features(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['Moving_Avg'] = df['Close'].rolling(window=10).mean()  # Adjust window as needed
    df['Rolling_Std_Dev'] = df['Close'].rolling(window=10).std()  # Adjust window as needed
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)  # Requires TA-Lib
    df['EMA'] = df['Close'].ewm(span=12, adjust=False).mean()  # Adjust span as needed
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10) # Rate of Change
    # ... Add other features as needed (Sentiment_Numeric, Headlines_Count, Next_Day_Return)
    # Important: Fill NaN values after feature engineering
    df.fillna(0, inplace=True) # Or another appropriate strategy
    return df
@st.cache_resource(show_spinner=False)
def load_prediction_model_and_scaler(model_path, scaler_path):
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Model and Scaler loaded successfully!")
            return model, scaler
        else:
            st.warning(f"⚠️ Model or Scaler file not found at: {model_path} or {scaler_path}")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return None, None

prediction_model, scaler = load_prediction_model_and_scaler(MODEL_PATH, SCALER_PATH)

# Ensure Required Features
REQUIRED_FEATURES = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "Daily_Return", "Sentiment_Score", "Headlines_Count",
    "Next_Day_Return", "Moving_Avg", "Rolling_Std_Dev",
    "RSI", "EMA", "ROC", "Sentiment_Numeric"
]

def ensure_features(df, required_features):
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    df = df[required_features]
    return df

# Fetch Stock Data
@st.cache_data(show_spinner=False)
def fetch_stock_data(stock_ticker, start_date, end_date):
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")
        if stock_data.empty:
            return None
        stock_data = stock_data.reset_index()
        stock_data['Date'] = stock_data['Date'].astype(str)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Fetch Current Stock Info
@st.cache_data(show_spinner=False)
def fetch_current_stock_info(stock_ticker):
    try:
        stock = yf.Ticker(stock_ticker)
        stock_info = stock.history(period="1d")
        if stock_info.empty:
            return None
        return {
            "current_price": stock_info["Close"].iloc[-1],
            "previous_close": stock_info["Close"].iloc[-2] if len(stock_info) > 1 else None,
            "open": stock_info["Open"].iloc[-1],
            "day_high": stock_info["High"].iloc[-1],
            "day_low": stock_info["Low"].iloc[-1],
            "volume": stock_info["Volume"].iloc[-1]
        }
    except Exception as e:
        st.error(f"Error fetching current stock info: {e}")
        return None

# Fetch News Data
@st.cache_data(show_spinner=False)
def fetch_news(stock_ticker):
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Perform Sentiment Analysis with Score Mapping
def analyze_sentiment(news_articles):
    if news_articles:
        try:
            sentiments = sentiment_pipeline(news_articles)
            for sentiment in sentiments:
                if sentiment["label"] == "positive":
                    sentiment["score"] = 1.0
                elif sentiment["label"] == "neutral":
                    sentiment["score"] = 0.5
                elif sentiment["label"] == "negative":
                    sentiment["score"] = 0.0
            return sentiments
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
            return []
    return []
# Streamlit UI
st.title("📈 Stock Market Analyzer")
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):").upper()
date = st.date_input("Select Date for Analysis:", datetime.today())

stock_data = None
stock_info = None
news = []

if stock_ticker:
    start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = date.strftime('%Y-%m-%d')
    stock_data = fetch_stock_data(stock_ticker, start_date, end_date)
    stock_info = fetch_current_stock_info(stock_ticker)
    news = fetch_news(stock_ticker)

col1, col2 = st.columns(2)

with col1:
    if stock_data is not None and not stock_data.empty:
        st.subheader("📈 Stock Price Trend")
        fig, ax = plt.subplots()
        ax.plot(stock_data['Date'], stock_data['Close'], marker='o', linestyle='-')
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Closing Price (USD)")
        plt.title(f"{stock_ticker} Closing Prices")
        st.pyplot(fig)
    else:
        st.error("❌ No stock data available! Please check the ticker symbol.")

    if stock_info:
        st.subheader("📊 Current Stock Information")
        st.write(f"**Current Price:** ${stock_info['current_price']:.2f}")
        if stock_info['previous_close']:
            st.write(f"**Previous Close:** ${stock_info['previous_close']:.2f}")
        st.write(f"**Open:** ${stock_info['open']:.2f}")
        st.write(f"**Day High:** ${stock_info['day_high']:.2f}")
        st.write(f"**Day Low:** ${stock_info['day_low']:.2f}")
        st.write(f"**Volume:** {int(stock_info['volume']):,}")
    else:
        st.error("❌ No market data found for this stock.")

with col2:
    st.subheader("📰 Latest News & Sentiment")
    if news:
        sentiments = analyze_sentiment(news)
        for i, article in enumerate(news):
            st.write(f"**News {i+1}:** {article}")
            if sentiments and i < len(sentiments):
                label = sentiments[i]['label'].capitalize()
                score = sentiments[i]['score']
                st.write(f"**Sentiment:** {label} (Score: {score:.1f})")
            else:
                st.write("Sentiment: Not Available")
            st.write("---")
    else:
        st.write("No news available.")

# Recommendation Section
st.subheader("🚀 Investment Recommendation")
if prediction_model and scaler and stock_data is not None and len(stock_data) > 0 and stock_info:  # Check if model, scaler, and data are loaded
    try:
        # 1. Feature Engineering
        stock_data = engineer_features(stock_data.copy()) # Create a copy to avoid modifying original

        # 2. Ensure Required Features (after engineering)
        stock_data = ensure_features(stock_data, REQUIRED_FEATURES)

        # 3. Prepare Input Data for Prediction (using the LAST 30 days, or less if not available)
        input_data = stock_data[REQUIRED_FEATURES].tail(min(30, len(stock_data)))  # Use .tail() and handle cases with less than 30 days
        st.write("Input Data Shape:", input_data.shape) # Debug
        st.write("Input Data:", input_data) # Debug

        # 4. Scaling
        input_data_scaled = scaler.transform(input_data)

        # 5. Prediction
        prediction = prediction_model.predict(input_data_scaled)[0]

        # ... (Recommendation display remains the same)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug Info:") # Provide more debug info
        st.write(f"Model Loaded: {prediction_model is not None}")
        st.write(f"Scaler Loaded: {scaler is not None}")
        st.write(f"Stock Data Length: {len(stock_data) if stock_data is not None else 0}")
        if stock_data is not None:
            st.write(f"Stock Data Columns: {stock_data.columns}")  # Check available columns
else:
    st.warning("⚠️ Not enough data to generate recommendations")
    st.write("Debug Info:")
    st.write(f"Model Loaded: {prediction_model is not None}")
    st.write(f"Scaler Loaded: {scaler is not None}")
    st.write(f"Stock Data Length: {len(stock_data) if stock_data is not None else 0}")
