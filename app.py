# Force-install PyTorch first
import sys
import subprocess

subprocess.check_call([
    sys.executable, 
    "-m", "pip", "install",
    "torch==2.0.1+cpu",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu"
])

# Now import other dependencies
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feedparser
import joblib
import ta
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import warnings
from pytz import timezone
from pandas.tseries.holiday import USFederalHolidayCalendar

# Configuration
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="Professional Stock Analyzer", layout="wide")

# Constants
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
REQUIRED_FEATURES = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "Daily_Return", "Sentiment_Score", "Headlines_Count",
    "Next_Day_Return", "Moving_Avg", "Rolling_Std_Dev",
    "RSI", "EMA", "ROC", "Sentiment_Numeric"
]

# UI Configuration
st.markdown("""
<style>
    .expert-rating { background: #f8f9fa; padding: 25px; border-radius: 15px; margin: 20px 0; }
    .rating-header { color: #2c3e50; font-size: 24px; margin-bottom: 15px; font-weight: 600; }
    .main-rating { font-size: 42px; font-weight: 700; color: #27ae60; margin-bottom: 20px; }
    .breakdown-item { display: flex; justify-content: space-between; margin: 10px 0; padding: 8px 0; border-bottom: 1px solid #ecf0f1; }
    .disclaimer { font-size: 12px; color: #95a5a6; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def get_current_date():
    """Get current date in UTC timezone"""
    return datetime.now(timezone('UTC')).date()

def is_trading_day(selected_date):
    """Check if a date is a US trading day"""
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2020-01-01', end='2030-12-31').date
    return selected_date.weekday() < 5 and selected_date not in holidays

def validate_ticker(ticker):
    """Basic ticker validation"""
    if len(ticker) < 1 or len(ticker) > 5:
        return False
    return ticker.isalpha()

# Model Loading
@st.cache_resource
def load_models():
    """Load ML models with validation"""
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

@st.cache_resource
def load_sentiment_analyzer():
    """Load financial sentiment analyzer with PyTorch backend"""
    try:
        return pipeline(
            "text-classification",
            model="yiyanghkust/finbert-tone",
            framework="pt",  # Explicitly use PyTorch
            top_k=None
        )
    except Exception as e:
        st.error(f"Sentiment engine error: {str(e)}")
        return None

# Data Processing
def engineer_features(df):
    """Generate technical indicators with validation"""
    try:
        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Calculate features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Moving_Avg'] = df['Close'].rolling(window=10).mean()
        df['Rolling_Std_Dev'] = df['Close'].rolling(window=10).std()
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ROC'] = ta.momentum.roc(df['Close'], timeperiod=10)
        
        # Fill missing values
        df = df.fillna(0)
        
        # Ensure correct data types
        return df.astype(np.float32)
    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return df

@st.cache_data(show_spinner=False)
def fetch_market_data(ticker, start_date, end_date):
    """Fetch stock data with comprehensive error handling"""
    try:
        if not validate_ticker(ticker):
            raise ValueError("Invalid ticker format")
            
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError("No historical data available")
            
        # Ensure required columns exist
        for col in ['Adj Close', 'Close', 'Open', 'High', 'Low', 'Volume']:
            if col not in df.columns:
                df[col] = df['Close']  # Fallback to Close price
                
        return df.reset_index()
    except Exception as e:
        st.error(f"Data acquisition failed: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def fetch_news(ticker):
    """Fetch relevant news headlines with error handling"""
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        st.error(f"News fetch error: {str(e)}")
        return []

# Sentiment Analysis
def analyze_news_sentiment(headlines, analyzer):
    """Process news headlines with fallback"""
    if not headlines or not analyzer:
        return []
    try:
        return [max(result, key=lambda x: x['score']) for result in analyzer(headlines)]
    except Exception as e:
        st.warning("News sentiment analysis temporarily unavailable")
        return []

# Main Application
st.title("Professional Stock Analysis Suite")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("NASDAQ Ticker Symbol", "AAPL").upper().strip()
with col2:
    analysis_date = st.date_input("Analysis Date", get_current_date())

# Date Validation
if analysis_date > get_current_date():
    st.error("Future date selection not permitted")
    st.stop()

if analysis_date < get_current_date() - timedelta(days=5*365):
    st.error("Maximum historical data range is 5 years")
    st.stop()

if not is_trading_day(analysis_date):
    st.warning("Selected date is not a trading day. Results may be limited.")

# Core Processing
model, scaler = load_models()
sentiment_analyzer = load_sentiment_analyzer()

if validate_ticker(ticker):
    start_date = (analysis_date - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = analysis_date.strftime('%Y-%m-%d')
    
    with st.spinner("Analyzing market data..."):
        try:
            market_data = fetch_market_data(ticker, start_date, end_date)
            news_headlines = fetch_news(ticker)
            sentiment_results = analyze_news_sentiment(news_headlines, sentiment_analyzer)
            
            # Feature Engineering
            processed_data = engineer_features(market_data.copy())
            
            # Ensure required features
            for feature in REQUIRED_FEATURES:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0.0
                    
            processed_data = processed_data[REQUIRED_FEATURES]
            
            # Data Validation
            if len(processed_data) < 30:
                st.error("Minimum 30 trading days of data required")
                st.stop()
                
            if processed_data.shape[1] != len(REQUIRED_FEATURES):
                st.error(f"Feature dimension mismatch. Expected {len(REQUIRED_FEATURES)}, got {processed_data.shape[1]}")
                st.stop()

            # Prepare input data
            input_data = processed_data.tail(30).values.reshape(-1, len(REQUIRED_FEATURES))
            scaled_data = scaler.transform(input_data)
            
            # Model Prediction
            prediction = model.predict(scaled_data[-1].reshape(1, -1))[0]
            current_price = market_data['Close'].iloc[-1]
            
            # Expert Rating Display
            st.markdown(f"""
            <div class='expert-rating'>
                <div class='rating-header'>Expert Consensus Rating</div>
                <div class='main-rating'>{(prediction/current_price*100)-100:.1f}%</div>
                <div class='breakdown-item">
                    <span>Model Confidence</span>
                    <span>{np.mean([res['score'] for res in sentiment_results]):.0% if sentiment_results else 'N/A'}</span>
                </div>
                <div class='disclaimer'>
                    Aggregated analysis from market data and news sentiment
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Market Summary
            with st.expander("Detailed Market Analysis"):
                st.subheader("Price Trend Analysis")
                fig, ax = plt.subplots(figsize=(10, 4))
                market_data[-90:].plot(x='Date', y='Close', ax=ax)
                st.pyplot(fig)
                
                if sentiment_results:
                    st.subheader("Recent Market Sentiment")
                    cols = st.columns(3)
                    for idx, sentiment in enumerate(sentiment_results[:3]):
                        cols[idx%3].metric(
                            f"Headline {idx+1}",
                            sentiment['label'],
                            f"{sentiment['score']:.0%}"
                        )

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.write("Please try different parameters or check input values")
else:
    st.warning("Please enter a valid NASDAQ ticker symbol (e.g., AAPL, TSLA)")
