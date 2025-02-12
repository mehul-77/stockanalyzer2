# main.py
import sys
import subprocess
import pkg_resources
import streamlit as st

# Dependency configuration
REQUIREMENTS = {
    'torch': '2.0.1',
    'transformers': '4.30.0',
    'streamlit': '1.23.1',
    'numpy': '1.24.3',
    'pandas': '2.0.3',
    'yfinance': '0.2.18',
    'matplotlib': '3.7.1'
}

def check_dependencies():
    """Verify all required packages are installed and compatible"""
    missing = []
    outdated = []
    for package, required_version in REQUIREMENTS.items():
        try:
            installed = pkg_resources.get_distribution(package)
            if pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(required_version):
                outdated.append(f"{package} ({installed.version} < {required_version})")
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing or outdated:
        st.error("Dependency issues detected!")
        if missing:
            st.error(f"Missing packages: {', '.join(missing)}")
        if outdated:
            st.error(f"Update required: {', '.join(outdated)}")
        st.error("Run: pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu")
        st.stop()

def safe_import(module_name, package_name=None):
    """Import with clear error handling"""
    try:
        return __import__(module_name, fromlist=[package_name] if package_name else None)
    except ImportError as e:
        st.error(f"Critical import failed: {str(e)}")
        st.stop()

# Initialize core dependencies
check_dependencies()
np = safe_import('numpy')
pd = safe_import('pandas')
plt = safe_import('matplotlib.pyplot')
torch = safe_import('torch')
yf = safe_import('yfinance')
ta = safe_import('ta')
transformers = safe_import('transformers')

# Streamlit app configuration
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

def fetch_stock_data(ticker, period='1y'):
    """Fetch and preprocess stock data"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        st.error(f"No data found for {ticker}")
        return None
    
    # Calculate technical indicators
    hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
    hist['RSI'] = ta.momentum.rsi(hist['Close'])
    hist.dropna(inplace=True)
    return hist

def plot_stock_data(data, ticker):
    """Visualize stock data with technical indicators"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price and SMA
    ax1.plot(data.index, data['Close'], label='Closing Price')
    ax1.plot(data.index, data['SMA_20'], label='20-day SMA')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # RSI
    ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='green')
    ax2.set_ylabel('RSI')
    
    plt.suptitle(f"{ticker} Technical Analysis")
    st.pyplot(fig)

# Main application UI
def main():
    st.title("Stock Technical Analyzer")
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    with col2:
        period = st.selectbox("Analysis Period", ['1mo', '3mo', '6mo', '1y', '5y'])
    
    if st.button("Analyze"):
        with st.spinner("Fetching and analyzing data..."):
            try:
                data = fetch_stock_data(ticker, period)
                if data is not None:
                    st.subheader(f"{ticker} Analysis Results")
                    plot_stock_data(data, ticker)
                    
                    # Display raw data
                    with st.expander("View Raw Data"):
                        st.dataframe(data.tail(10))
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
