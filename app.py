# main.py
import sys
import subprocess
import pkg_resources
import streamlit as st

# Dependency configuration
REQUIREMENTS = {
    'torch': '2.0.1',
    'transformers': '4.30.0',# app.py
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from google_news import get_google_news
from textblob import TextBlob
from prediction_RandomForest import (
    ensure_features,
    load_rf_model,
    predict_stock_sentiment_rf,
    REQUIRED_FEATURES
)

# Configure page
st.set_page_config(
    page_title="Live Financial Analytics",
    page_icon="💹",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .news-card {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        background: #ffffff;
    }
    .positive {border-left: 4px solid #4CAF50;}
    .negative {border-left: 4px solid #f44336;}
    .neutral {border-left: 4px solid #FFC107;}
    .sentiment-bar {
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .stock-header {
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def fetch_stock_data(ticker, period='1y'):
    """Fetch live stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        st.error(f"No data found for {ticker}")
        return None
    
    # Calculate technical indicators
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['RSI'] = calculate_rsi(hist['Close'])
    hist['Daily_Return'] = hist['Close'].pct_change()
    hist.dropna(inplace=True)
    return hist

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_news_sentiment(news_items):
    """Perform sentiment analysis on news headlines"""
    sentiments = []
    for item in news_items:
        analysis = TextBlob(item['title'])
        sentiment = (analysis.sentiment.polarity + 1) / 2  # Normalize to 0-1
        sentiments.append({
            'title': item['title'],
            'link': item['link'],
            'date': item['date'],
            'sentiment': sentiment,
            'source': item['source']
        })
    return pd.DataFrame(sentiments)

def display_news(news_df):
    """Display news cards with sentiment visualization"""
    st.subheader("📰 Latest Market News")
    
    for _, row in news_df.iterrows():
        sentiment = row['sentiment']
        sentiment_class = "neutral"
        if sentiment > 0.6:
            sentiment_class = "positive"
        elif sentiment < 0.4:
            sentiment_class = "negative"
        
        st.markdown(
            f"""
            <div class="news-card {sentiment_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <small>{row['date']} • {row['source']}</small>
                    <b>Sentiment: {sentiment:.2f}</b>
                </div>
                <h4>{row['title']}</h4>
                <div class="sentiment-bar" style="width: {abs(sentiment-0.5)*200}%; 
                    background: {'#4CAF50' if sentiment > 0.5 else '#f44336'}; 
                    margin-left: {50 - abs(sentiment-0.5)*100}%">
                </div>
                <a href="{row['link']}" target="_blank" style="color: #1a237e; text-decoration: none;">
                    Read more →</a>
            </div>
            """, 
            unsafe_allow_html=True
        )

def display_stock_header(ticker, hist):
    """Show stock price header with key metrics"""
    current_price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2]
    change = current_price - prev_close
    pct_change = (change / prev_close) * 100
    
    st.markdown(f"""
        <div class="stock-header">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <h1>{ticker}</h1>
                    <h2>${current_price:.2f}</h2>
                </div>
                <div style="text-align: right;">
                    <h3>{'↑' if change >= 0 else '↓'} ${abs(change):.2f}</h3>
                    <h3>{pct_change:+.2f}%</h3>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("📊 Live Financial Analytics Dashboard")
    
    # User input
    with st.form("ticker_input"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
        with col2:
            period = st.selectbox(
                "Analysis Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                index=5
            )
        analyze_clicked = st.form_submit_button("Analyze")
    
    if analyze_clicked:
        try:
            with st.spinner("Fetching live market data and news..."):
                # Fetch live data
                hist = fetch_stock_data(ticker, period)
                news_items = get_google_news(ticker)
                news_df = analyze_news_sentiment(news_items)
                
                # Prepare data for ML model
                ml_data = hist.copy()
                ml_data = ensure_features(ml_data, REQUIRED_FEATURES)
                
                # Display results
                st.success("Analysis complete!")
                
                # Stock price header
                display_stock_header(ticker, hist)
                
                # Main columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Price chart
                    st.subheader("Price Movement")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(hist.index, hist['Close'], label='Closing Price')
                    ax.plot(hist.index, hist['SMA_20'], label='20-day SMA')
                    ax.set_ylabel('Price')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Technical indicators
                    st.subheader("Technical Analysis")
                    col1a, col1b, col1c = st.columns(3)
                    with col1a:
                        st.metric("RSI", f"{hist['RSI'].iloc[-1]:.2f}")
                    with col1b:
                        st.metric("Daily Return", f"{hist['Daily_Return'].iloc[-1]:.2%}")
                    with col1c:
                        st.metric("Volume", f"{hist['Volume'].iloc[-1]:,}")
                
                with col2:
                    # AI Prediction
                    st.subheader("AI Prediction")
                    prediction = predict_stock_sentiment_rf(ml_data.tail(1))
                    prediction_color = "#4CAF50" if "Positive" in prediction else "#f44336" if "Negative" in prediction else "#FFC107"
                    st.markdown(
                        f"<h2 style='color: {prediction_color}; text-align: center;'>{prediction}</h2>",
                        unsafe_allow_html=True
                    )
                    
                    # News Section
                    if not news_df.empty:
                        display_news(news_df.head(5))
                    else:
                        st.warning("No recent news found")
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
    'streamlit': '1.23.1',
    'numpy': '1.24.3',
    'pandas': '2.0.3',
    'yfinance': '0.2.18',
    'matplotlib': '3.7.1'
}

def check_dependencies():
    """Verify all required packages are installed and compatible"""
    missing = []
    outdated = []
    for package, required_version in REQUIREMENTS.items():
        try:
            installed = pkg_resources.get_distribution(package)
            if pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(required_version):
                outdated.append(f"{package} ({installed.version} < {required_version})")
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing or outdated:
        st.error("Dependency issues detected!")
        if missing:
            st.error(f"Missing packages: {', '.join(missing)}")
        if outdated:
            st.error(f"Update required: {', '.join(outdated)}")
        st.error("Run: pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu")
        st.stop()

def safe_import(module_name, package_name=None):
    """Import with clear error handling"""
    try:
        return __import__(module_name, fromlist=[package_name] if package_name else None)
    except ImportError as e:
        st.error(f"Critical import failed: {str(e)}")
        st.stop()

# Initialize core dependencies
check_dependencies()
np = safe_import('numpy')
pd = safe_import('pandas')
plt = safe_import('matplotlib.pyplot')
torch = safe_import('torch')
yf = safe_import('yfinance')
ta = safe_import('ta')
transformers = safe_import('transformers')

# Streamlit app configuration
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

def fetch_stock_data(ticker, period='1y'):
    """Fetch and preprocess stock data"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        st.error(f"No data found for {ticker}")
        return None
    
    # Calculate technical indicators
    hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
    hist['RSI'] = ta.momentum.rsi(hist['Close'])
    hist.dropna(inplace=True)
    return hist

def plot_stock_data(data, ticker):
    """Visualize stock data with technical indicators"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price and SMA
    ax1.plot(data.index, data['Close'], label='Closing Price')
    ax1.plot(data.index, data['SMA_20'], label='20-day SMA')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # RSI
    ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='green')
    ax2.set_ylabel('RSI')
    
    plt.suptitle(f"{ticker} Technical Analysis")
    st.pyplot(fig)

# Main application UI
def main():
    st.title("Stock Technical Analyzer")
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    with col2:
        period = st.selectbox("Analysis Period", ['1mo', '3mo', '6mo', '1y', '5y'])
    
    if st.button("Analyze"):
        with st.spinner("Fetching and analyzing data..."):
            try:
                data = fetch_stock_data(ticker, period)
                if data is not None:
                    st.subheader(f"{ticker} Analysis Results")
                    plot_stock_data(data, ticker)
                    
                    # Display raw data
                    with st.expander("View Raw Data"):
                        st.dataframe(data.tail(10))
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
