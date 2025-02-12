# app.py
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
st.set_page_config(page_title="Stock Analyzer Pro", layout="wide")

# Constants
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
REQUIRED_FEATURES = [
    "Close", "High", "Low", "Open", "Volume", "Adj Close",
    "Daily_Return", "Moving_Avg", "Rolling_Std_Dev",
    "RSI", "EMA", "ROC", "Sentiment_Score"
]

# UI Configuration
st.markdown("""
<style>
    .expert-rating { 
        background: #f8f9fa; 
        padding: 25px; 
        border-radius: 15px; 
        margin: 20px 0; 
    }
    .rating-header { 
        color: #2c3e50; 
        font-size: 24px; 
        margin-bottom: 15px; 
        font-weight: 600; 
    }
    .main-rating { 
        font-size: 42px; 
        font-weight: 700; 
        color: #27ae60; 
        margin-bottom: 20px; 
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def get_current_date():
    return datetime.now(timezone('UTC')).date()

def validate_ticker(ticker):
    return len(ticker) > 0 and ticker.isalpha()

# Model Loading
@st.cache_resource
def load_models():
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Data Processing
def engineer_features(df):
    try:
        df['Daily_Return'] = df['Close'].pct_change()
        df['Moving_Avg'] = df['Close'].rolling(10).mean()
        df['Rolling_Std_Dev'] = df['Close'].rolling(10).std()
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ROC'] = ta.momentum.roc(df['Close'], window=10)
        return df.fillna(0).astype(np.float32)
    except Exception as e:
        st.error(f"Feature engineering error: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def get_stock_data(ticker, days=730):
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date)
        return df.reset_index()
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def get_news(ticker):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en")
        return [entry.title for entry in feed.entries[:3]]
    except:
        return []

# Sentiment Analysis
@st.cache_resource
def load_sentiment_analyzer():
    try:
        return pipeline("text-classification", model="yiyanghkust/finbert-tone")
    except Exception as e:
        st.error(f"Sentiment analysis setup failed: {str(e)}")
        st.stop()

# Main App
def main():
    st.title("Stock Analysis Pro")
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", "AAPL").upper().strip()
    with col2:
        analysis_date = st.date_input("Analysis Date", date.today())

    if not validate_ticker(ticker):
        st.error("Invalid ticker symbol")
        return

    # Load Models
    model, scaler = load_models()
    sentiment_analyzer = load_sentiment_analyzer()

    # Get Data
    with st.spinner("Analyzing market data..."):
        try:
            # Stock Data
            df = get_stock_data(ticker)
            if len(df) < 30:
                st.error("Insufficient historical data")
                return
                
            # Feature Engineering
            processed_data = engineer_features(df)
            input_data = processed_data[REQUIRED_FEATURES].tail(30).values
            
            # Validate Input Shape
            if input_data.shape != (30, len(REQUIRED_FEATURES)):
                st.error("Data shape mismatch")
                return
                
            # Prediction
            scaled_data = scaler.transform(input_data.reshape(-1, len(REQUIRED_FEATURES)))
            prediction = model.predict(scaled_data[-1].reshape(1, -1))[0]
            current_price = df['Close'].iloc[-1]

            # News Analysis
            news = get_news(ticker)
            sentiments = [sentiment_analyzer(headline)[0] for headline in news] if news else []

            # Display Results
            st.markdown(f"""
            <div class='expert-rating'>
                <div class='rating-header'>Expert Rating</div>
                <div class='main-rating'>{prediction/current_price*100-100:.1f}%</div>
                <div class='metric-box'>
                    Current Price: ${current_price:.2f}<br>
                    Predicted Price: ${prediction:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # News Section
            if news:
                st.subheader("Market Sentiment")
                cols = st.columns(3)
                for idx, (headline, sentiment) in enumerate(zip(news, sentiments)):
                    cols[idx%3].markdown(f"""
                    <div class='metric-box'>
                        📰 {headline}<br>
                        <hr>
                        {sentiment['label']} ({sentiment['score']:.0%})
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
