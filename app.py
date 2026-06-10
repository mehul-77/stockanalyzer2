import streamlit as st
import pandas as pd
from backend.data_pipeline import get_stock_data, get_news_sentiment, calculate_indicators
from backend.model_handler import load_models, prepare_features, get_recommendation

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="SentiStock: AI-Powered US Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# Load machine learning models
model, scaler = load_models()

# ------------------------------------------------
# UI Presentation Layer
# ------------------------------------------------

st.title("SentiStock AI Stock Analyzer 📊")
st.markdown("---")

ticker = st.text_input("Enter US Stock Ticker", value="AAPL").strip().upper()

if ticker == "":
    st.stop()

if model is None or scaler is None:
    st.error("Error loading machine learning models. Please ensure the .pkl files exist.")
    st.stop()

# 1. Fetch Stock Data
with st.spinner(f"Fetching market data for {ticker}..."):
    stock_data, stock_error = get_stock_data(ticker)

if stock_error:
    st.warning(f"Could not fetch stock data: {stock_error}")
    st.info("Yahoo Finance may be rate-limiting the server. Please try again later.")
    st.stop()

if stock_data.empty:
    st.warning("No stock data found for this ticker.")
    st.stop()

# 2. Process Technical Indicators
stock_data = calculate_indicators(stock_data)

if stock_data.empty:
    st.warning("Not enough data to calculate technical indicators.")
    st.stop()

latest = stock_data.iloc[-1]

# 3. Fetch News Sentiment
with st.spinner("Analyzing latest news sentiment..."):
    news, news_error = get_news_sentiment(ticker)

if news_error:
    st.warning(f"Error fetching news sentiment: {news_error}")
    # We don't stop here, the model can still predict with 0 sentiment

# 4. Machine Learning Inference
features = prepare_features(stock_data, news)

if hasattr(scaler, "feature_names_in_"):
    # Ensure correct feature ordering
    features = features[scaler.feature_names_in_]

scaled = scaler.transform(features)
pred = model.predict_proba(scaled)[0]
recommendation, confidence, probs = get_recommendation(pred, model.classes_)

# ------------------------------------------------
# Charts and Metrics
# ------------------------------------------------

st.subheader(f"{ticker} Technical Analysis")

if "Close" in stock_data.columns and "Moving_Avg" in stock_data.columns:
    st.line_chart(stock_data[["Close", "Moving_Avg"]])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Price", f"${latest['Close']:.2f}")
    st.metric("RSI", f"{latest['RSI']:.2f}")

with col2:
    st.metric("EMA", f"${latest['EMA']:.2f}")
    st.metric("Volume", f"{latest['Volume']:,.0f}")

with col3:
    st.metric("Sentiment Score", f"{news['Sentiment_Score']:.2f}")

# ------------------------------------------------
# Recommendation Section
# ------------------------------------------------

st.markdown("---")
st.subheader("Investment Recommendation")

st.metric("AI Recommendation", recommendation)
st.progress(float(confidence))
st.caption(f"Confidence Level: {confidence*100:.1f}%")

for k, v in probs.items():
    st.write(f"Probability of class {k}: {v*100:.1f}%")

# ------------------------------------------------
# News Section
# ------------------------------------------------

st.markdown("---")
st.subheader("Recent News Headlines")

if news["Headlines"]:
    for h, s in zip(news["Headlines"], news["Sentiments"]):
        st.write(f"- {h}")
        st.caption("Positive" if s > 0 else "Negative" if s < 0 else "Neutral")
else:
    st.write("No news headlines found recently.")

st.markdown("---")
st.caption("© 2025 SentiStock AI")
st.caption("Disclaimer: This is subject to risk read all market regulations and rules clearly before investing. No legal binding shall be attached to us.")
