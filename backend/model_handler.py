import pickle
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_models():
    try:
        with open("random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        return None, None

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
