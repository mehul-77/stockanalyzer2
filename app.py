import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from transformers import pipeline
from datetime import datetime, timedelta

# Ensure page configuration is set first
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Load FinBERT model efficiently
@st.cache_resource
def load_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_model()

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(stock_ticker, start_date, end_date):
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")
        if stock_data.empty:
            return None  # Return None if stock is invalid
        stock_data = stock_data.reset_index()
        stock_data['Date'] = stock_data['Date'].astype(str)
        return stock_data
    except Exception as e:
        return None

# Function to fetch current stock info
@st.cache_data
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
        return None

# Function to fetch news
@st.cache_data
def fetch_news(stock_ticker):
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        return []

# Perform sentiment analysis
def analyze_sentiment(news_articles):
    if news_articles:
        return sentiment_pipeline(news_articles)
    return []

# Streamlit UI with Multiple Pages
st.sidebar.title("📊 Stock Market Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Stock Analysis", "News & Sentiment", "About"])

if page == "Home":
    st.title("📈 Welcome to Stock Market Analyzer")
    st.write("Analyze stock prices, news, and sentiment in one place!")
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Stock_Market_Board.jpg", caption="Stock Market Overview")

elif page == "Stock Analysis":
    st.title("📊 Stock Analysis")
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):").upper()
    date = st.date_input("Select Date for Analysis:", datetime.today())

    if stock_ticker:
        start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')

        stock_data = fetch_stock_data(stock_ticker, start_date, end_date)
        stock_info = fetch_current_stock_info(stock_ticker)

        if stock_data is not None:
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
            st.subheader("📈 Current Stock Information")
            st.write(f"**Current Price:** ${stock_info['current_price']}")
            if stock_info['previous_close']:
                st.write(f"**Previous Close:** ${stock_info['previous_close']}")
            st.write(f"**Open:** ${stock_info['open']}")
            st.write(f"**Day High:** ${stock_info['day_high']}")
            st.write(f"**Day Low:** ${stock_info['day_low']}")
            st.write(f"**Volume:** {stock_info['volume']}")
        else:
            st.error("❌ No market data found for this stock.")

elif page == "News & Sentiment":
    st.title("📰 Stock Market News & Sentiment")
    stock_ticker = st.text_input("Enter Stock Ticker for News Analysis:").upper()

    if stock_ticker:
        news = fetch_news(stock_ticker)
        st.subheader("📰 Latest News")
        for article in news:
            st.write(f"- {article}")

        sentiments = analyze_sentiment(news)
        if sentiments:
            st.subheader("📊 Sentiment Analysis")
            for i, sentiment in enumerate(sentiments):
                st.write(f"**News {i+1}:** {news[i]}")
                st.write(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
        else:
            st.warning("No sentiment data available.")

elif page == "About":
    st.title("ℹ️ About This App")
    st.write("This app provides real-time stock analysis, price trends, market news, and sentiment analysis using AI-driven FinBERT.")
    st.write("Built with ❤️ using Streamlit, Yahoo Finance, and FinBERT.")
