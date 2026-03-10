import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from textblob import TextBlob
from GoogleNews import GoogleNews

# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SentiStock: AI-Powered US Stock Analysing and Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():
    try:
        with open("random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


model, scaler = load_models()

# --------------------------------------------------
# Cached Data Functions (Prevents API spam)
# --------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        hist = yf.download(ticker, period="1y", progress=False)

        if hist.empty or 'Close' not in hist.columns:
            return pd.DataFrame()

        return hist

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    try:
        gn = GoogleNews()
        gn.search(f"{ticker} stock news")

        results = gn.results()[:10]

        sentiments = []
        headlines = []

        for result in results:
            analysis = TextBlob(result['title'])
            sentiments.append(analysis.sentiment.polarity)
            headlines.append(result['title'])

        avg_sentiment = np.mean(sentiments) if sentiments else 0

        return {
            'Sentiment_Score': avg_sentiment,
            'Headlines': headlines,
            'Sentiments': sentiments,
            'Sentiment_Numeric': 1 if avg_sentiment > 0 else -1,
            'Headlines_Count': len(headlines)
        }

    except Exception:
        return {
            'Sentiment_Score': 0,
            'Headlines': [],
            'Sentiments': [],
            'Sentiment_Numeric': 0,
            'Headlines_Count': 0
        }

# --------------------------------------------------
# Technical Indicators
# --------------------------------------------------

def compute_rsi(series, window=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    rs = gain / loss

    return 100 - (100 / (1 + rs))


def calculate_technical_indicators(df):

    df['Daily_Return'] = df['Close'].pct_change()

    df['Moving_Avg'] = df['Close'].rolling(window=14).mean()

    df['Rolling_Std_Dev'] = df['Close'].rolling(window=14).std()

    df['RSI'] = compute_rsi(df['Close'])

    df['EMA'] = df['Close'].ewm(span=14).mean()

    df['ROC'] = df['Close'].pct_change(periods=14)

    return df.dropna()

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def prepare_features(stock_data, news_features):

    features = pd.DataFrame({

        'Adj Close': [stock_data['Close'].iloc[-1]],
        'Close': [stock_data['Close'].iloc[-1]],
        'High': [stock_data['High'].iloc[-1]],
        'Low': [stock_data['Low'].iloc[-1]],
        'Open': [stock_data['Open'].iloc[-1]],
        'Volume': [stock_data['Volume'].iloc[-1]],

        'Daily_Return': [stock_data['Daily_Return'].iloc[-1]],
        'Sentiment_Score': [news_features['Sentiment_Score']],

        'Next_Day_Return': [0],

        'Moving_Avg': [stock_data['Moving_Avg'].iloc[-1]],
        'Rolling_Std_Dev': [stock_data['Rolling_Std_Dev'].iloc[-1]],
        'RSI': [stock_data['RSI'].iloc[-1]],
        'EMA': [stock_data['EMA'].iloc[-1]],
        'ROC': [stock_data['ROC'].iloc[-1]],

        'Sentiment_Numeric': [news_features['Sentiment_Numeric']],
        'Headlines_Count': [news_features['Headlines_Count']]

    })

    return features


# --------------------------------------------------
# Recommendation Logic
# --------------------------------------------------

def get_recommendation(probabilities, classes):

    max_index = np.argmax(probabilities)

    recommendation = classes[max_index]

    confidence = probabilities[max_index]

    probs_dict = dict(zip(classes, probabilities))

    if recommendation == 1:
        recommendation = "Buy"

    elif recommendation == 0:
        recommendation = "Sell"

    return recommendation, confidence, probs_dict


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("SentiStock: AI-Powered US Stock Insights 📊")

st.markdown("---")

col1, col2 = st.columns([1, 3])

# Initialize variables safely
stock_data = None
latest_data = None
news_features = None

with col1:

    ticker = st.text_input("Enter US Stock Ticker", value="AAPL").strip().upper()

    if ticker == "":
        st.warning("Please enter a valid stock ticker.")
        st.stop()

    if model is not None and scaler is not None:

        try:

            stock_data = get_stock_data(ticker)

            if stock_data.empty:
                st.warning("No stock data available.")
                st.stop()

            stock_data = calculate_technical_indicators(stock_data)

            latest_data = stock_data.iloc[-1]

            news_features = get_news_sentiment(ticker)

            features = prepare_features(stock_data, news_features)

            if hasattr(scaler, "feature_names_in_"):
                features = features[scaler.feature_names_in_]

            scaled_data = scaler.transform(features)

            pred_probs = model.predict_proba(scaled_data)[0]

            recommendation, confidence, probs = get_recommendation(
                pred_probs, model.classes_
            )

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.stop()

# --------------------------------------------------
# Charts
# --------------------------------------------------

with col2:

    if stock_data is not None and not stock_data.empty:

        st.subheader(f"{ticker} Technical Analysis")

        st.line_chart(stock_data[['Close', 'Moving_Avg']])

        col2_1, col2_2, col2_3 = st.columns(3)

        with col2_1:
            st.metric("Current Price", f"${latest_data['Close']:.2f}")
            st.metric("RSI", f"{latest_data['RSI']:.2f}")

        with col2_2:
            st.metric("14-Day EMA", f"${latest_data['EMA']:.2f}")
            st.metric("Daily Volume", f"{latest_data['Volume']:,.0f}")

        with col2_3:
            st.metric("News Sentiment", f"{news_features['Sentiment_Score']:.2f}")

        st.markdown("---")

        st.subheader("Recent News Headlines")

        for headline, sentiment in zip(
                news_features['Headlines'], news_features['Sentiments']):

            st.write(f"Headline: {headline}")

            st.write(
                f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}"
            )

            st.write("---")


# --------------------------------------------------
# Recommendation
# --------------------------------------------------

st.markdown("---")

st.subheader("Investment Recommendation")

if 'recommendation' in locals():

    col3_1, col3_2 = st.columns([1, 2])

    with col3_1:

        st.metric("Recommendation", recommendation)

        st.progress(confidence)

        st.caption(f"Confidence Level: {confidence*100:.1f}%")

        st.markdown("**Probabilities:**")

        for label, prob in probs.items():
            st.write(f"**{label}:** {prob*100:.1f}%")

    with col3_2:

        if recommendation == "Buy":
            st.success(
                "**Analysis:** Strong positive indicators detected. Consider adding to your portfolio."
            )

        elif recommendation == "Sell":
            st.error(
                "**Analysis:** Negative trends detected. Consider reducing your position."
            )


st.markdown("---")

st.caption("© 2025 US Stock Analyzer. Educational purposes only.")
