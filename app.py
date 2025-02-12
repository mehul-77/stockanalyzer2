import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from newsapi import NewsApiClient

# -----------------------------------------
# Load the trained Random Forest model
# -----------------------------------------
model_path = 'random_forest_model.pkl'
model = joblib.load(model_path)

# -----------------------------------------
# Set up (or load) the TF-IDF Vectorizer
# -----------------------------------------
# NOTE: For production, use the vectorizer that was used during model training.
vectorizer = TfidfVectorizer()  # Replace with your pre-fitted vectorizer if available

# -----------------------------------------
# Initialize NewsAPI Client for real-time news fetching
# -----------------------------------------
# Replace 'YOUR_NEWSAPI_KEY' with your actual NewsAPI key.
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"  # <-- INSERT YOUR API KEY HERE
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# -----------------------------------------
# Function to fetch real-time financial news for a given stock ticker
# -----------------------------------------
def fetch_financial_news(stock):
    # Construct a query to search for news articles related to the stock.
    # You can adjust the query as needed (e.g., adding keywords like 'financial' or 'market').
    query = f"{stock} stock"
    response = newsapi.get_everything(q=query,
                                      language='en',
                                      sort_by='publishedAt',
                                      page_size=5)  # Fetch the top 5 recent articles
    articles = response.get("articles", [])
    headlines = [article["title"] for article in articles if article.get("title")]
    return headlines

# -----------------------------------------
# Configure the Streamlit app
# -----------------------------------------
st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")

# App title and header
st.title("📊 Stock Sentiment Analyzer")
st.markdown("## Analyze real-time financial sentiment on your favorite stocks")

# -----------------------------------------
# User Input Section
# -----------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    stock_name = st.text_input(
        "Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):",
        placeholder="Type stock ticker here..."
    )
with col2:
    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

st.markdown("---")

# -----------------------------------------
# Processing & Display Section
# -----------------------------------------
if analyze_button:
    if stock_name:
        st.info(f"Fetching the latest financial news for **{stock_name}**...")
        news_articles = fetch_financial_news(stock_name)

        if news_articles:
            # Preprocess news articles using the TF-IDF vectorizer.
            # If you have a pre-fitted vectorizer, use vectorizer.transform() instead of fit_transform().
            news_features = vectorizer.fit_transform(news_articles)

            # Predict sentiment scores using the trained Random Forest model
            predictions = model.predict(news_features)
            avg_sentiment = predictions.mean()

            # Determine recommendation based on average sentiment score
            if avg_sentiment >= 0.75:
                rec = "🟢 Strong Buy"
                color = "#7CFC00"  # Light Green
            elif avg_sentiment >= 0.5:
                rec = "🟡 Hold"
                color = "#FFFF99"  # Light Yellow
            else:
                rec = "🔴 Sell"
                color = "#FF6347"  # Tomato Red

            # Display the recommendation in a card-like UI
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color};
                            text-align: center; font-size: 24px; font-weight: bold;">
                    {rec}
                </div>
                """, unsafe_allow_html=True
            )

            # Display the average sentiment score and a progress bar
            st.metric(label="Sentiment Score", value=f"{avg_sentiment:.2f}")
            st.progress(avg_sentiment)

            # -----------------------------------------
            # Display the latest news headlines
            # -----------------------------------------
            st.markdown("### Latest News Headlines")
            for article in news_articles:
                st.write(f"- {article}")

            # -----------------------------------------
            # Display a bar chart for individual sentiment scores
            # -----------------------------------------
            st.markdown("### Detailed Sentiment Scores")
            fig, ax = plt.subplots()
            x = np.arange(len(predictions))
            ax.bar(x, predictions, color='skyblue')
            ax.set_xlabel("News Article Index")
            ax.set_ylabel("Sentiment Score")
            ax.set_title("Sentiment Scores per News Article")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Article {i+1}" for i in x])
            st.pyplot(fig)
        else:
            st.error("No financial news found for this stock.")
    else:
        st.warning("Please enter a stock ticker.")
