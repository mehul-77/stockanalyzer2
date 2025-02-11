import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from transformers import pipeline
from datetime import datetime, timedelta

# Load FinBERT model for sentiment analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_model()

st.title("📊 Stock Market News & Sentiment Analyzer")

# User input
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
date = st.date_input("Select Date for Analysis:", datetime.today())

@st.cache_data
def fetch_stock_data(stock_ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")
        stock_data = stock_data.reset_index()
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_current_stock_info(stock_ticker):
    """Fetch current stock price and other relevant data."""
    try:
        stock_info = yf.Ticker(stock_ticker).info
        return {
            "current_price": stock_info.get("regularMarketPrice"),
            "previous_close": stock_info.get("regularMarketPreviousClose"),
            "open": stock_info.get("regularMarketOpen"),
            "day_high": stock_info.get("regularMarketDayHigh"),
            "day_low": stock_info.get("regularMarketDayLow"),
            "volume": stock_info.get("regularMarketVolume"),
            "market_cap": stock_info.get("marketCap")
        }
    except Exception as e:
        st.error(f"Error fetching current stock info: {e}")
        return {}

@st.cache_data
def fetch_news(stock_ticker):
    """Fetch news headlines from Google News RSS."""
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(news_articles):
    """Perform sentiment analysis on news articles."""
    try:
        if news_articles:
            sentiments = sentiment_pipeline(news_articles)
            return sentiments
        return []
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return []

# Fetch stock data
start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = date.strftime('%Y-%m-%d')
stock_data = fetch_stock_data(stock_ticker, start_date, end_date)

# Fetch current stock info
current_stock_info = fetch_current_stock_info(stock_ticker)

# Display current stock info
if current_stock_info:
    st.subheader("📈 Current Stock Information")
    st.write(f"**Current Price:** ${current_stock_info['current_price']}")
    st.write(f"**Previous Close:** ${current_stock_info['previous_close']}")
    st.write(f"**Open:** ${current_stock_info['open']}")
    st.write(f"**Day High:** ${current_stock_info['day_high']}")
    st.write(f"**Day Low:** ${current_stock_info['day_low']}")
    st.write(f"**Volume:** {current_stock_info['volume']}")
    st.write(f"**Market Cap:** ${current_stock_info['market_cap']:,}")

# Display stock data
if not stock_data.empty:
    st.subheader("📈 Stock Price Trend")
    fig, ax = plt.subplots()
    ax.plot(stock_data['Date'], stock_data['Close'], marker='o', linestyle='-')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.title(f"{stock_ticker} Closing Prices")
    st.pyplot(fig)
else:
    st.error("No stock data available!")

# Fetch and display news
news = fetch_news(stock_ticker)
st.subheader("📰 Latest News")
for article in news:
    st.write(f"- {article}")

# Perform sentiment analysis
sentiments = analyze_sentiment(news)
if sentiments:
    st.subheader("📊 Sentiment Analysis")
    for i, sentiment in enumerate(sentiments):
        st.write(f"**News {i+1}:** {news[i]}")
        st.write(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
else:
    st.warning("No sentiment data available.")

st.success("✅ Analysis Completed! Refresh for updated insights.")
