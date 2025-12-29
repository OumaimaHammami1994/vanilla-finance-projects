import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pricing.autocalls.vanilla_autocall import Autocall


@st.cache_data
def get_current_price(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception:
        st.warning('Could not fetch market price; using fallback spot. 100')
        return 100


st.title('Vanilla Autocall Monte Carlo Pricer')

st.header('Autocall Parameters')

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input('Undelying Ticker', value='AAPL')
    st.caption("Enter the stock symbol (e.g., AAPL for Apple Inc.)")

    # Fetch current price
    current_price = get_current_price(ticker)
    if current_price is not None:
        st.write(f"Current price of {ticker}: ${current_price:.2f}")
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    st.caption("The theoretical rate of return of an investment with zero risk.")

    sigma = st.slider('Sigma (Volatility) (%)', 0, 100, 20)
    st.caption("A measure of the stock's price variability.  ")
    n_paths = st.number_input('Monte Carlo paths', value=20000, min_value=100, step=100)
    #show_paths = st.number_input('Show simulated paths (count)', value=50, min_value=0, max_value=500, step=1)

with col2:
    notional = st.number_input('Notional', value=1000.0, step=1.0)
    coupon = st.number_input('Coupon (annual, fraction)', value=0.04, min_value=0.0, max_value=1.0, step=0.01)
    coupon_barrier = st.number_input('Coupon barrier level', value=0.9, min_value=0.0, max_value=10.0, step=0.01)
    autocall_barrier = st.number_input('Autocall barrier level', value=1.0, min_value=0.0, max_value=10.0, step=0.01)
    protection_barrier = st.number_input('Protection barrier level', value=0.7, min_value=0.0, max_value=10.0, step=0.01)
    T = st.number_input('Time to maturity (years)', value=1.0, min_value=0.01, step=0.01)
    obs_count = st.slider('Number of observation dates including maturity', 1, 12, 4)

    



# Build observation times evenly spaced up to T
#obs_times = [(i + 1) / obs_count * T for i in range(obs_count)]
# st.write('Observation times (years):', obs_times)

if st.button('Price Autocall'):
    ac = Autocall(S0=current_price, notional=notional, coupon=coupon, coupon_barrier=coupon_barrier,
                   autocall_barrier=autocall_barrier, protection_barrier=protection_barrier, 
                   n_obs=obs_count, T=T, r=risk_free_rate/100, sigma=sigma/100)
    price, stderr = ac._calculate_autocall_price(n_paths=int(n_paths))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr
    st.success(f'Price: {price:.2f} (StdErr: {stderr:.4f})')
    st.write(f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')


