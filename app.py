import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from datetime import datetime, timedelta
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import ta  # Technical Analysis library

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# -----------------------------
# Load Sentiment Analysis Model (FinBERT)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="yiyanghkust/finbert-tone", top_k=None)
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

sentiment_pipeline = load_sentiment_model()

# -----------------------------
# Load Random Forest Model and Scaler
# -----------------------------
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_prediction_model_and_scaler(model_path, scaler_path):
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        else:
            st.error("Model or Scaler file not found.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

prediction_model, scaler = load_prediction_model_and_scaler(MODEL_PATH, SCALER_PATH)

# -----------------------------
# Define Required Features for Prediction
# -----------------------------
PREDICTION_FEATURES = ["Close", "High", "Low", "Open", "Volume", "Daily_Return"]

# -----------------------------
# Feature Engineering Function
# -----------------------------
def engineer_features(df):
    try:
        df = df.copy()
        df["Daily_Return"] = df["Close"].pct_change()
        df["Moving_Avg"] = df["Close"].rolling(window=10).mean()
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["EMA"] = ta.trend.ema_indicator(df["Close"], window=12)
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        return df

# -----------------------------
# Fetch Stock Data from Yahoo Finance
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(stock_ticker, start_date, end_date):
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")
        if stock_data.empty:
            return None
        stock_data = stock_data.reset_index()
        stock_data["Date"] = stock_data["Date"].astype(str)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# -----------------------------
# Fetch News Data via Google News RSS Feed
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_news(stock_ticker):
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# -----------------------------
# Perform Sentiment Analysis on News Articles
# -----------------------------
def analyze_sentiment(news_articles):
    if news_articles and sentiment_pipeline:
        try:
            sentiments = sentiment_pipeline(news_articles)
            scores = []
            for sentiment in sentiments:
                label = sentiment[0]["label"].lower()
                if label == "positive":
                    score = 1.0
                elif label == "neutral":
                    score = 0.5
                else:
                    score = 0.0
                scores.append(score)
            return np.mean(scores) if scores else 0.5
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
            return 0.5
    return 0.5

# -----------------------------
# Generate Buy/Hold/Sell Recommendation
# -----------------------------
def get_recommendation(avg_sentiment, predicted_price, current_price):
    price_change = ((predicted_price - current_price) / current_price) * 100
    buy_prob = min(max((avg_sentiment * 0.7 + max(price_change, 0) * 0.3) * 100, 0), 100)
    sell_prob = min(max(((1 - avg_sentiment) * 0.7 + max(-price_change, 0) * 0.3) * 100, 0), 100)
    hold_prob = 100 - buy_prob - sell_prob
    return buy_prob, hold_prob, sell_prob

# -----------------------------
# Streamlit UI Layout
# -----------------------------
st.title("📈 Stock Market Analyzer")

stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):").upper()
date = st.date_input("Select Date for Analysis:", datetime.today())

if stock_ticker:
    st.subheader("📈 Stock Price Trend")
    stock_data = fetch_stock_data(stock_ticker, (date - timedelta(days=730)).strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))
    if stock_data is not None and not stock_data.empty:
        stock_data = engineer_features(stock_data)
        chart_data = stock_data.iloc[-30:] if len(stock_data) >= 30 else stock_data
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(chart_data['Date'], chart_data['Close'], marker='o', linestyle='-')
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Closing Price (USD)")
        plt.title(f"{stock_ticker} Closing Prices")
        st.pyplot(fig)
    else:
        st.error("No stock data available! Check the ticker symbol.")

    st.subheader("📰 Latest News Headlines")
    news_articles = fetch_news(stock_ticker)
    for i, article in enumerate(news_articles):
        st.write(f"**News {i+1}:** {article}")
