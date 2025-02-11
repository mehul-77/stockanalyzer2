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

st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Load Sentiment Analysis Model
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

# Load Prediction Model
MODEL_PATH = "random_forest_model.pkl"

@st.cache_resource(show_spinner=False)
def load_prediction_model(model_path):
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("✅ Prediction model loaded successfully!")
            return model
        else:
            st.warning(f"⚠️ Prediction model file not found at: {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading prediction model: {e}")
        return None

prediction_model = load_prediction_model(MODEL_PATH)

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

# Perform Sentiment Analysis
def analyze_sentiment(news_articles):
    if news_articles:
        try:
            return sentiment_pipeline(news_articles)
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
            return []
    return []

# Stock Recommendation Function
def get_recommendation(sentiments, prediction, current_price):
    buy_score = 0
    sell_score = 0
    
    if prediction > current_price:
        buy_score += 50
    else:
        sell_score += 50
    
    for sentiment in sentiments:
        if sentiment['label'] == 'positive':
            buy_score += 25
        elif sentiment['label'] == 'negative':
            sell_score += 25
    
    total_score = buy_score + sell_score
    buy_percentage = (buy_score / total_score) * 100 if total_score > 0 else 50
    sell_percentage = 100 - buy_percentage
    
    return buy_percentage, sell_percentage

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

    with col2:
        st.subheader("📰 Latest News & Sentiment")
        sentiments = analyze_sentiment(news)
        for i, article in enumerate(news):
            st.write(f"**News {i+1}:** {article}")
            st.write(f"Sentiment: {sentiments[i]['label']} (Score: {sentiments[i]['score']:.2f})")
            st.write("---")

    if prediction_model and stock_data is not None and len(stock_data) >= 30:
        try:
            last_30_days_data = stock_data['Close'].values[-30:]
            input_data = last_30_days_data.reshape(1, -1)
            prediction = prediction_model.predict(input_data)[0]
            st.subheader("🔮 Stock Price Prediction")
            st.write(f"Predicted Closing Price: ${prediction:.2f}")
            
            buy_percentage, sell_percentage = get_recommendation(sentiments, prediction, stock_info['current_price'])
            st.subheader("📊 Recommendation")
            st.write(f"**Buy Probability:** {buy_percentage:.2f}%")
            st.write(f"**Sell Probability:** {sell_percentage:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
