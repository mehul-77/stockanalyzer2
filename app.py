import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from textblob import TextBlob
from GoogleNews import GoogleNews

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="SentiStock: AI-Powered US Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------
# Load Models
# ------------------------------------------------

@st.cache_resource
def load_models():
    try:
        with open("random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None


model, scaler = load_models()

# ------------------------------------------------
# Data Fetching (Cached)
# ------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_data(ticker):

    try:
        hist = yf.download(ticker, period="1y", progress=False)

        # Fix MultiIndex issue
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if hist.empty or "Close" not in hist.columns:
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

        for r in results:
            polarity = TextBlob(r["title"]).sentiment.polarity
            sentiments.append(polarity)
            headlines.append(r["title"])

        avg = np.mean(sentiments) if sentiments else 0

        return {
            "Sentiment_Score": avg,
            "Headlines": headlines,
            "Sentiments": sentiments,
            "Sentiment_Numeric": 1 if avg > 0 else -1,
            "Headlines_Count": len(headlines),
        }

    except Exception:
        return {
            "Sentiment_Score": 0,
            "Headlines": [],
            "Sentiments": [],
            "Sentiment_Numeric": 0,
            "Headlines_Count": 0,
        }


# ------------------------------------------------
# Indicators
# ------------------------------------------------

def compute_rsi(series, window=14):

    delta = series.diff()

    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()

    rs = gain / loss

    return 100 - (100 / (1 + rs))


def calculate_indicators(df):

    df["Daily_Return"] = df["Close"].pct_change()
    df["Moving_Avg"] = df["Close"].rolling(14).mean()
    df["Rolling_Std_Dev"] = df["Close"].rolling(14).std()
    df["RSI"] = compute_rsi(df["Close"])
    df["EMA"] = df["Close"].ewm(span=14).mean()
    df["ROC"] = df["Close"].pct_change(14)

    return df.dropna()


# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

def prepare_features(stock, news):

    return pd.DataFrame({

        "Adj Close":[stock["Close"].iloc[-1]],
        "Close":[stock["Close"].iloc[-1]],
        "High":[stock["High"].iloc[-1]],
        "Low":[stock["Low"].iloc[-1]],
        "Open":[stock["Open"].iloc[-1]],
        "Volume":[stock["Volume"].iloc[-1]],
        "Daily_Return":[stock["Daily_Return"].iloc[-1]],

        "Sentiment_Score":[news["Sentiment_Score"]],
        "Next_Day_Return":[0],

        "Moving_Avg":[stock["Moving_Avg"].iloc[-1]],
        "Rolling_Std_Dev":[stock["Rolling_Std_Dev"].iloc[-1]],
        "RSI":[stock["RSI"].iloc[-1]],
        "EMA":[stock["EMA"].iloc[-1]],
        "ROC":[stock["ROC"].iloc[-1]],

        "Sentiment_Numeric":[news["Sentiment_Numeric"]],
        "Headlines_Count":[news["Headlines_Count"]],
    })


# ------------------------------------------------
# Recommendation
# ------------------------------------------------

def get_recommendation(probs, classes):

    idx = np.argmax(probs)

    rec = classes[idx]
    conf = probs[idx]

    probs_dict = dict(zip(classes, probs))

    if rec == 1:
        rec = "Buy"
    elif rec == 0:
        rec = "Sell"

    return rec, conf, probs_dict


# ------------------------------------------------
# UI
# ------------------------------------------------

st.title("SentiStock AI Stock Analyzer 📊")

st.markdown("---")

ticker = st.text_input("Enter US Stock Ticker", value="AAPL").strip().upper()

if ticker == "":
    st.stop()

stock_data = get_stock_data(ticker)

if stock_data.empty:
    st.warning("No stock data found")
    st.stop()

stock_data = calculate_indicators(stock_data)

latest = stock_data.iloc[-1]

news = get_news_sentiment(ticker)

features = prepare_features(stock_data, news)

if hasattr(scaler, "feature_names_in_"):
    features = features[scaler.feature_names_in_]

scaled = scaler.transform(features)

pred = model.predict_proba(scaled)[0]

recommendation, confidence, probs = get_recommendation(pred, model.classes_)

# ------------------------------------------------
# Charts
# ------------------------------------------------

st.subheader(f"{ticker} Technical Analysis")

if "Close" in stock_data.columns and "Moving_Avg" in stock_data.columns:
    st.line_chart(stock_data[["Close", "Moving_Avg"]])

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Price", f"${latest['Close']:.2f}")
    st.metric("RSI", f"{latest['RSI']:.2f}")

with col2:
    st.metric("EMA", f"${latest['EMA']:.2f}")
    st.metric("Volume", f"{latest['Volume']:,.0f}")

with col3:
    st.metric("Sentiment", f"{news['Sentiment_Score']:.2f}")

# ------------------------------------------------
# Recommendation
# ------------------------------------------------

st.markdown("---")

st.subheader("Investment Recommendation")

st.metric("Recommendation", recommendation)

st.progress(confidence)

st.caption(f"Confidence {confidence*100:.1f}%")

for k,v in probs.items():
    st.write(f"{k}: {v*100:.1f}%")

# ------------------------------------------------
# News
# ------------------------------------------------

st.markdown("---")

st.subheader("Recent News")

for h,s in zip(news["Headlines"], news["Sentiments"]):

    st.write(h)

    st.caption(
        "Positive" if s>0 else "Negative" if s<0 else "Neutral"
    )

st.markdown("---")

st.caption("© 2025 SentiStock AI")
