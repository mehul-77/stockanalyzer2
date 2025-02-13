import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from textblob import TextBlob
from GoogleNews import GoogleNews
from sklearn.preprocessing import StandardScaler

# Configuration
st.set_page_config(
    page_title="NASDAQ Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# Helper functions
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

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

def get_news_sentiment(ticker):
    gn = GoogleNews()
    gn.search(f"{ticker} stock news")
    results = gn.results()[:10]  # Get top 10 news

    sentiments = []
    headlines = []
    for result in results:
        analysis = TextBlob(result['title'])
        sentiments.append(analysis.sentiment.polarity)
        headlines.append(result['title'])

    return {
        'Sentiment_Score': np.mean(sentiments) if sentiments else 0,
        'Headlines': headlines,
        'Sentiments': sentiments,
        'Sentiment_Numeric': 1 if np.mean(sentiments) > 0 else -1
    }

def prepare_features(stock_data, news_features):
    # Build a DataFrame from the latest stock row and the news sentiment data.
    features = pd.DataFrame({
        'Adj Close': [stock_data['Close'].iloc[-1]],
        'Close': [stock_data['Close'].iloc[-1]],
        'High': [stock_data['High'].iloc[-1]],
        'Low': [stock_data['Low'].iloc[-1]],
        'Open': [stock_data['Open'].iloc[-1]],
        'Volume': [stock_data['Volume'].iloc[-1]],
        'Daily_Return': [stock_data['Daily_Return'].iloc[-1]],
        'Sentiment_Score': [news_features['Sentiment_Score']],
        'Next_Day_Return': [0],  # Placeholder
        'Moving_Avg': [stock_data['Moving_Avg'].iloc[-1]],
        'Rolling_Std_Dev': [stock_data['Rolling_Std_Dev'].iloc[-1]],
        'RSI': [stock_data['RSI'].iloc[-1]],
        'EMA': [stock_data['EMA'].iloc[-1]],
        'ROC': [stock_data['ROC'].iloc[-1]],
        'Sentiment_Numeric': [news_features['Sentiment_Numeric']]
    })

    required_features = [
        "Adj Close", "Close", "High", "Low", "Open", "Volume",
        "Daily_Return", "Sentiment_Score", "Next_Day_Return",
        "Moving_Avg", "Rolling_Std_Dev", "RSI", "EMA", "ROC", "Sentiment_Numeric"
    ]
    
    for feature in required_features:
        if feature not in features.columns:
            features[feature] = 0

    return features[required_features]

def get_recommendation(probabilities, classes):
    """
    Determines the recommendation based on the highest probability.
    Returns:
      - recommendation (str): The class label with the highest probability.
      - confidence (float): The highest probability.
      - probs_dict (dict): Dictionary of probabilities for each class.
    """
    max_index = np.argmax(probabilities)
    recommendation = classes[max_index]
    confidence = probabilities[max_index]
    probs_dict = dict(zip(classes, probabilities))
    return recommendation, confidence, probs_dict

# UI Components
st.title("NASDAQ Stock Analysis & Prediction Platform 📊")
st.markdown("---")

# Main content
col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.selectbox(
        "Select NASDAQ Stock",
        ("AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"),
        index=0
    )
    
    try:
        # Retrieve stock data and compute indicators
        stock_data = get_stock_data(ticker)
        news_features = get_news_sentiment(ticker)
        processed_data = calculate_technical_indicators(stock_data)
        latest_data = processed_data.iloc[-1]

        # Prepare features and scale them
        features = prepare_features(processed_data, news_features)
        scaled_data = scaler.transform(features)

        # Get probability distribution from the model
        pred_probs = model.predict_proba(scaled_data)[0]
        recommendation, confidence, probs = get_recommendation(pred_probs, model.classes_)
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

with col2:
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
    for headline, sentiment in zip(news_features['Headlines'], news_features['Sentiments']):
        st.write(f"Headline: {headline}")
        st.write(f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
        st.write("---")

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
            st.success("**Analysis:** Strong positive indicators detected. Consider adding to your portfolio.")
        elif recommendation == "Sell":
            st.error("**Analysis:** Negative trends detected. Consider reducing your position.")
        else:
            st.warning("**Analysis:** Neutral market signals. Maintain your current position.")
        
    st.markdown("---")
    st.subheader("Recent News Analysis")
    
    gn = GoogleNews()
    gn.search(f"{ticker} stock news")
    results = gn.results()[:5]
    
    for news in results:
        with st.expander(news['title']):
            st.caption(news['media'])
            st.write(news['desc'])
            st.caption(news['date'])
else:
    st.warning("Select a stock to see analysis")

st.markdown("---")
st.caption("© 2024 NASDAQ Stock Analyzer. For educational purposes only.")
