import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
import urllib.request
import xml.etree.ElementTree as ET
import streamlit as st

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        # Create a session to try bypassing rate limits
        hist = yf.download(ticker, period="1y", progress=False)

        # Handle the MultiIndex issue with newer versions of yfinance
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if hist.empty or "Close" not in hist.columns:
            return pd.DataFrame(), "Rate limit reached or no data found for this ticker."

        return hist, None
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    try:
        # Using Google News RSS Feed directly to bypass scraping blocks
        url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        
        sentiments = []
        headlines = []
        
        # Parse items from RSS
        for item in root.findall('./channel/item')[:10]:
            title = item.find('title').text
            if title:
                polarity = TextBlob(title).sentiment.polarity
                sentiments.append(polarity)
                headlines.append(title)

        avg = np.mean(sentiments) if sentiments else 0

        return {
            "Sentiment_Score": avg,
            "Headlines": headlines,
            "Sentiments": sentiments,
            "Sentiment_Numeric": 1 if avg > 0 else -1,
            "Headlines_Count": len(headlines),
        }, None

    except Exception as e:
        return {
            "Sentiment_Score": 0,
            "Headlines": [],
            "Sentiments": [],
            "Sentiment_Numeric": 0,
            "Headlines_Count": 0,
        }, str(e)

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    if df.empty:
        return df
    
    df["Daily_Return"] = df["Close"].pct_change()
    df["Moving_Avg"] = df["Close"].rolling(14).mean()
    df["Rolling_Std_Dev"] = df["Close"].rolling(14).std()
    df["RSI"] = compute_rsi(df["Close"])
    df["EMA"] = df["Close"].ewm(span=14).mean()
    df["ROC"] = df["Close"].pct_change(14)

    return df.dropna()
