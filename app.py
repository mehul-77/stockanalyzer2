import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from GoogleNews import GoogleNews
from transformers import pipeline

# ------------------------------
# Load Random Forest Model & Scaler
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error("Error loading model or scaler: " + str(e))
        return None, None

model, scaler = load_model_and_scaler()

# ------------------------------
# Load FinBERT Sentiment Pipeline
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    try:
        sentiment_pipe = pipeline(
            "text-classification", 
            model="yiyanghkust/finbert-tone", 
            return_all_scores=True
        )
        return sentiment_pipe
    except Exception as e:
        st.error("Error loading sentiment pipeline: " + str(e))
        return None

sentiment_pipe = load_sentiment_pipeline()

# ------------------------------
# Fetch Real-Time Stock Data using yfinance
# ------------------------------
def fetch_stock_data(symbol, period="6mo"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df

# ------------------------------
# Compute Technical Indicators
# ------------------------------
def compute_technical_indicators(df):
    df["Daily_Return"] = df["Close"].pct_change()
    df["Moving_Avg"] = df["Close"].rolling(window=14).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta).clip(lower=0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.fillna(method="bfill", inplace=True)
    return df

# ------------------------------
# Fetch News Articles using GoogleNews
# ------------------------------
def fetch_news(symbol):
    googlenews = GoogleNews(lang='en', period='1d')
    query = f"{symbol} stock"
    googlenews.search(query)
    results = googlenews.result()
    headlines = [item["title"] for item in results][:5]
    return headlines

# ------------------------------
# Analyze Sentiment for News Headlines using FinBERT
# ------------------------------
def analyze_sentiments(headlines):
    scores = []
    news_details = []
    if sentiment_pipe is None:
        return 0.5, []
    for headline in headlines:
        result = sentiment_pipe(headline)
        # Take the label with the highest score from the first (and only) prediction set
        best = max(result[0], key=lambda x: x["score"])
        label = best["label"].lower()
        if label == "positive":
            score = 1.0
        elif label == "neutral":
            score = 0.5
        else:
            score = 0.0
        scores.append(score)
        news_details.append({"headline": headline, "label": best["label"], "score": score})
    avg_sentiment = np.mean(scores) if scores else 0.5
    return avg_sentiment, news_details

# ------------------------------
# Prepare Feature Vector for Prediction
# ------------------------------
def prepare_feature_vector(df, avg_sentiment, headlines_count):
    # Use the latest available data from stock_df and combine with news features
    latest = df.iloc[-1]
    features = {
        "Adj Close": latest["Close"],  # Assuming no separate Adj Close available
        "Close": latest["Close"],
        "High": latest["High"],
        "Low": latest["Low"],
        "Open": latest["Open"],
        "Volume": latest["Volume"],
        "Daily_Return": latest["Daily_Return"],
        "Sentiment_Score": avg_sentiment,
        "Headlines_Count": headlines_count,
        "Next_Day_Return": 0,      # Placeholder (if unavailable)
        "Moving_Avg": latest["Moving_Avg"],
        "Rolling_Std_Dev": 0,      # Placeholder (if unavailable)\n        \"RSI\": latest[\"RSI\"],\n        \"EMA\": 0,           # Placeholder\n        \"ROC\": 0,           # Placeholder\n        \"Sentiment_Numeric\": avg_sentiment\n    }\n    feature_order = [\n        \"Adj Close\", \"Close\", \"High\", \"Low\", \"Open\", \"Volume\",\n        \"Daily_Return\", \"Sentiment_Score\", \"Headlines_Count\",\n        \"Next_Day_Return\", \"Moving_Avg\", \"Rolling_Std_Dev\",\n        \"RSI\", \"EMA\", \"ROC\", \"Sentiment_Numeric\"\n    ]\n    feature_vector = np.array([features[f] for f in feature_order]).reshape(1, -1)\n    return feature_vector\n\n# ------------------------------\n# Interpret Model Prediction (Assuming 3 classes: Buy, Hold, Sell)\n# ------------------------------\ndef interpret_prediction(probs):\n    buy_prob, hold_prob, sell_prob = probs\n    return buy_prob, hold_prob, sell_prob\n\n# ------------------------------\n# Streamlit UI Layout\n# ------------------------------\nst.title(\"NASDAQ Stock Analyzer and Predictor\")\n\nst.sidebar.header(\"Select NASDAQ Stock\")\nstock_symbol = st.sidebar.text_input(\"Enter NASDAQ Symbol (e.g., AAPL, MSFT, TSLA):\", \"AAPL\").upper()\n\nif st.sidebar.button(\"Analyze\"):\n    if not model or not scaler:\n        st.error(\"Model and scaler are not loaded properly.\")\n    else:\n        st.header(f\"Real-Time Analysis for {stock_symbol}\")\n        # Fetch and process stock data\n        stock_df = fetch_stock_data(stock_symbol, period=\"6mo\")\n        stock_df = compute_technical_indicators(stock_df)\n        \n        # Display interactive candlestick chart using Plotly\n        st.subheader(\"Stock Price Chart\")\n        fig = go.Figure(data=[go.Candlestick(x=stock_df[\"Date\"],\n                                             open=stock_df[\"Open\"],\n                                             high=stock_df[\"High\"],\n                                             low=stock_df[\"Low\"],\n                                             close=stock_df[\"Close\"])])\n        fig.update_layout(xaxis_rangeslider_visible=False)\n        st.plotly_chart(fig, use_container_width=True)\n        \n        # Fetch news and perform sentiment analysis\n        st.subheader(\"Latest News & Sentiment Analysis\")\n        headlines = fetch_news(stock_symbol)\n        if headlines:\n            st.write(\"**News Headlines:**\")\n            for hl in headlines:\n                st.markdown(f\"- {hl}\")\n            avg_sentiment, news_details = analyze_sentiments(headlines)\n            st.write(f\"**Average Sentiment Score:** {avg_sentiment:.2f}\")\n            for detail in news_details:\n                st.write(f\"**{detail['headline']}** - {detail['label']} (Score: {detail['score']})\")\n        else:\n            st.write(\"No news available.\")\n            avg_sentiment = 0.5\n        \n        # Prepare features and predict using the Random Forest model\n        feature_vector = prepare_feature_vector(stock_df, avg_sentiment, len(headlines))\n        scaled_features = scaler.transform(feature_vector)\n        prediction_probs = model.predict_proba(scaled_features)\n        if hasattr(prediction_probs, 'tolist'):\n            prediction_probs = prediction_probs.tolist()[0]\n        else:\n            prediction_probs = prediction_probs[0]\n        buy_prob, hold_prob, sell_prob = interpret_prediction(prediction_probs)\n        \n        # Display Expert Rating (Buy/Hold/Sell probabilities)\n        st.subheader(\"Expert Rating\")\n        col1, col2, col3 = st.columns(3)\n        col1.metric(\"Buy\", f\"{buy_prob * 100:.2f}%\")\n        col2.metric(\"Hold\", f\"{hold_prob * 100:.2f}%\")\n        col3.metric(\"Sell\", f\"{sell_prob * 100:.2f}%\")\n        \n        st.write(\"---\")\n        st.write(\"*Final recommendation based on expert rating probabilities.*\")\n"} 

