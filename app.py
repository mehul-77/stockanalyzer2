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
import ta

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Custom CSS for expert rating display
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
    .breakdown-item {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
        padding: 8px 0;
        border-bottom: 1px solid #ecf0f1;
    }
    .breakdown-label {
        color: #7f8c8d;
        font-size: 16px;
    }
    .breakdown-value {
        color: #2c3e50;
        font-weight: 500;
    }
    .disclaimer {
        font-size: 12px;
        color: #95a5a6;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load Sentiment Analysis Model
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="yiyanghkust/finbert-tone",
            return_all_scores=True
        )
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

sentiment_pipeline = load_sentiment_model()

# Load Prediction Model and Scaler
@st.cache_resource(show_spinner=False)
def load_prediction_model():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_prediction_model()

# Feature Engineering and Data Handling
REQUIRED_FEATURES = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "Daily_Return", "Sentiment_Score", "Headlines_Count",
    "Next_Day_Return", "Moving_Avg", "Rolling_Std_Dev",
    "RSI", "EMA", "ROC", "Sentiment_Numeric"
]

def engineer_features(df):
    try:
        df['Daily_Return'] = df['Close'].pct_change()
        df['Moving_Avg'] = df['Close'].rolling(10).mean()
        df['Rolling_Std_Dev'] = df['Close'].rolling(10).std()
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ROC'] = ta.momentum.roc(df['Close'], window=10)
        return df.fillna(0)
    except Exception as e:
        st.error(f"Feature engineering error: {e}")
        return df

def ensure_features(df):
    for feature in REQUIRED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    return df[REQUIRED_FEATURES]

# Data Fetching Functions
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        return df.reset_index() if not df.empty else None
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_news(ticker):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock")
        return [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        st.error(f"News fetch error: {e}")
        return []

# Sentiment Analysis
def analyze_news(news):
    if not news or not sentiment_pipeline:
        return []
    try:
        results = sentiment_pipeline(news)
        return [max(s, key=lambda x: x['score']) for s in results]
    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
        return []

# Prediction and Recommendation
def generate_recommendation(sentiments, prediction, current_price):
    avg_score = np.mean([s['score'] for s in sentiments]) if sentiments else 0.5
    price_change = ((prediction - current_price) / current_price) * 100
    
    buy = min(max((avg_score * 0.7 + max(price_change, 0) * 0.3) * 100, 0), 100)
    sell = min(max(((1 - avg_score) * 0.7 + max(-price_change, 0) * 0.3) * 100, 0), 100)
    hold = 100 - buy - sell
    
    total = buy + sell + hold
    return (
        buy / total * 100,
        hold / total * 100,
        sell / total * 100
    )

# Main App Interface
st.title("📈 Professional Stock Analyzer")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker Symbol", "AAPL").upper()
with col2:
    analysis_date = st.date_input("Analysis Date", datetime.today())

# Data Collection
start_date = (analysis_date - timedelta(days=730)).strftime('%Y-%m-%d')
end_date = analysis_date.strftime('%Y-%m-%d')
stock_data = get_stock_data(ticker, start_date, end_date)
news = get_news(ticker)
sentiments = analyze_news(news)

# Main Display
if stock_data is not None:
    try:
        # Feature Engineering
        processed_data = engineer_features(stock_data.copy())
        processed_data = ensure_features(processed_data)
        
        # Prediction
        input_data = processed_data[REQUIRED_FEATURES].tail(30)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        current_price = processed_data['Close'].iloc[-1]
        
        # Generate Recommendations
        buy, hold, sell = generate_recommendation(sentiments, prediction, current_price)
        
        # Expert Rating Display
        st.markdown(f"""
        <div class='expert-rating'>
            <div class='rating-header'>Expert Consensus Rating</div>
            <div class='main-rating'>{hold:.0f}%</div>
            <div class='breakdown-item'>
                <span class='breakdown-label'>Hold Recommendation</span>
                <span class='breakdown-value'>{hold:.1f}%</span>
            </div>
            <div class='breakdown-item'>
                <span class='breakdown-label'>Model Confidence</span>
                <span class='breakdown-value'>{prediction/current_price*100:.1f}%</span>
            </div>
            <div class='disclaimer'>
                Aggregated analysis from market data and news sentiment
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation Metrics
        cols = st.columns(3)
        metrics = [
            (f"{buy:.1f}%", "BUY", "#27ae60" if buy > 50 else "#95a5a6"),
            (f"{hold:.1f}%", "HOLD", "#f1c40f" if hold > 50 else "#95a5a6"),
            (f"{sell:.1f}%", "SELL", "#e74c3c" if sell > 50 else "#95a5a6")
        ]
        
        for col, (value, label, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: {color}10; border-radius: 10px;">
                    <div style="font-size: 24px; color: {color}; font-weight: 700;">{value}</div>
                    <div style="font-size: 16px; color: #2c3e50; margin-top: 5px;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # Additional Information
        with st.expander("Detailed Analysis"):
            st.subheader("Price Trend")
            fig, ax = plt.subplots(figsize=(10, 4))
            processed_data[-30:].plot(x='Date', y='Close', ax=ax)
            st.pyplot(fig)
            
            st.subheader("News Sentiment")
            if sentiments:
                for i, sentiment in enumerate(sentiments[:3], 1):
                    st.write(f"**Headline {i}:** {sentiment['label']} ({sentiment['score']:.0%})")
            else:
                st.write("No recent news analysis available")

    except Exception as e:
        st.error("System error during analysis. Please try again later.")
else:
    st.warning("Please enter a valid stock ticker to begin analysis")
