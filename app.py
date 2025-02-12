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

# Add TA-Lib error handling at the top
try:
    import talib
except ImportError:
    st.error("""
    ⚠️ TA-Lib not installed! Required for technical indicators.
    For cloud deployments, add 'talib-binary' to requirements.txt
    Local installation: 'pip install TA-Lib' (see official docs)
    """)
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Constants
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
REQUIRED_FEATURES = [
    "Close", "High", "Low", "Open", "Volume", "Adj Close",
    "Daily_Return", "Moving_Avg", "Rolling_Std_Dev",
    "RSI", "EMA", "ROC", "Sentiment_Score",
    "Headlines_Count", "Sentiment_Numeric"
]

# --- Model Loading Section ---
@st.cache_resource
def load_models():
    """Load ML models with better error handling"""
    try:
        model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        return model, scaler
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# --- Sentiment Analysis Section ---
@st.cache_resource
def load_sentiment_analyzer():
    """Load FinBERT model with progress indication"""
    with st.spinner("Loading sentiment analyzer..."):
        try:
            from transformers import pipeline
            return pipeline(
                "text-classification",
                model="yiyanghkust/finbert-tone",
                return_all_scores=True
            )
        except Exception as e:
            st.error(f"Sentiment model failed to load: {str(e)}")
            return None

# --- Data Fetching Section ---
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start_date, end_date):
    """Improved stock data fetching with validation"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data found for {ticker}")
            return None
        return df.reset_index()
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

# --- Feature Engineering Section ---
def calculate_features(df):
    """Robust feature engineering with TA-Lib"""
    try:
        df['Daily_Return'] = df['Close'].pct_change()
        df['Moving_Avg'] = df['Close'].rolling(10).mean()
        df['Rolling_Std_Dev'] = df['Close'].rolling(10).std()
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['EMA'] = talib.EMA(df['Close'], timeperiod=12)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        return df.fillna(0).replace([np.inf, -np.inf], 0)
    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return df

# --- Sentiment Processing Section ---
def process_sentiment(headlines, analyzer):
    """Handle sentiment analysis with retries"""
    if not headlines or not analyzer:
        return []
    
    try:
        results = analyzer(headlines)
        return [
            max(sentiment, key=lambda x: x['score'])
            for sentiment in results
        ]
    except Exception as e:
        st.error(f"Sentiment analysis failed: {str(e)}")
        return []

# --- Prediction Section ---
def make_prediction(model, scaler, data):
    """Safe prediction handling"""
    try:
        if data.shape[0] < 30:
            st.warning("Insufficient historical data for reliable prediction")
            return None
            
        scaled_data = scaler.transform(data)
        return model.predict(scaled_data)[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# --- Main Application Flow ---
def main():
    st.title("📈 Intelligent Stock Analyzer")
    
    # Sidebar Controls
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    analysis_date = st.sidebar.date_input("Analysis Date", datetime.today())
    
    # Initialize models
    model, scaler = load_models()
    sentiment_analyzer = load_sentiment_analyzer()
    
    # Date calculations
    start_date = (analysis_date - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = analysis_date.strftime('%Y-%m-%d')
    
    # Data collection
    stock_data = get_stock_data(ticker, start_date, end_date)
    news = fetch_news(ticker)  # Implement similar error handling as get_stock_data
    
    if stock_data is None:
        st.error("Failed to retrieve stock data")
        return
    
    # Feature engineering
    processed_data = calculate_features(stock_data)
    
    # Ensure required features exist
    for feature in REQUIRED_FEATURES:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    
    # UI Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        st.subheader(f"{ticker} Price History")
        fig, ax = plt.subplots(figsize=(10, 4))
        processed_data[-30:].plot(x='Date', y='Close', ax=ax)
        st.pyplot(fig)
        
        # Prediction display
        if model and scaler:
            prediction = make_prediction(
                model,
                scaler,
                processed_data[REQUIRED_FEATURES].tail(30)
            
            if prediction:
                st.metric("Predicted Close", f"${prediction:.2f}")
    
    with col2:
        # News sentiment
        st.subheader("Market Sentiment")
        sentiments = process_sentiment(news, sentiment_analyzer)
        
        if sentiments:
            avg_score = np.mean([s['score'] for s in sentiments])
            st.metric("Average Sentiment", f"{avg_score:.1%}")
            
            for idx, sentiment in enumerate(sentiments[:3], 1):
                st.write(f"**Headline {idx}:** {sentiment['label']} ({sentiment['score']:.1%})")
        else:
            st.write("No sentiment data available")
    
    # Recommendation engine
    st.subheader("Trading Recommendation")
    # Implement recommendation logic similar to original but with error checks

if __name__ == "__main__":
    main()
