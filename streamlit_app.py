import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pricing.autocalls.vanilla_autocall import Autocall
from pricing.autocalls.multi_index_autocall import MultiIndexAutocall


@st.cache_data
def get_current_price(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception:
        st.warning(f'Could not fetch market price for {ticker}; using fallback spot 100')
        return 100


@st.cache_data
def get_current_prices(tickers: list):
    prices = []
    corr_matrix = None
    if len(tickers) > 1:
        # Fetch historical data for correlation
        try:
            data = yf.download(tickers, period="1y", interval="1d")['Close']
            # Compute log returns correlation
            log_returns = np.log(data / data.shift(1)).dropna()
            corr_matrix = log_returns.corr().values
            # For constant rho, take average of off-diagonal
            rho = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        except Exception:
            st.warning("Could not fetch historical data for correlation; using rho=0")
            rho = 0.0
    else:
        rho = 0.0
    
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1d")
            prices.append(float(data['Close'].iloc[-1]))
        except Exception:
            st.warning(f'Could not fetch market price for {ticker}; using fallback spot 100')
            prices.append(100)
    
    return prices, rho


st.title('Autocall Monte Carlo Pricer')

st.header('Autocall Parameters')

col1, col2 = st.columns(2)
with col1:
    tickers = st.multiselect('Underlying Tickers', options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], default=['AAPL'])
    st.caption("Select one or more stock symbols")

    # Fetch current prices
    if tickers:
        current_prices, default_rho = get_current_prices(tickers)
        for ticker, price in zip(tickers, current_prices):
            st.write(f"Current price of {ticker}: ${price:.2f}")
    else:
        st.warning("Please select at least one ticker")
        current_prices = []
        default_rho = 0.0

    if len(tickers) > 1:
        basket_type = st.selectbox('Basket Calculation Method', options=['mean', 'min', 'max'], index=0)
        st.caption("How to aggregate the basket for barrier checks: mean (average), min (worst performing), max (best performing)")
        rho = st.slider('Correlation between stocks', -1.0, 1.0, default_rho, 0.01)
        st.caption("Correlation coefficient between stock returns")
    else:
        basket_type = 'mean'  # Not used for single
        rho = 0.0  # Not used

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

if st.button('Price Autocall') and tickers:
    if len(tickers) == 1:
        ac = Autocall(S0=current_prices[0], notional=notional, coupon=coupon, coupon_barrier=coupon_barrier,
                       autocall_barrier=autocall_barrier, protection_barrier=protection_barrier, 
                       n_obs=obs_count, T=T, r=risk_free_rate/100, sigma=sigma/100)
        price, stderr = ac.price_monte_carlo(n_paths=int(n_paths))
    else:
        sigma_list = [sigma/100] * len(tickers)  # Assume same sigma for all
        ac = MultiIndexAutocall(S0_list=current_prices, notional=notional, coupon=coupon, coupon_barrier=coupon_barrier,
                                 autocall_barrier=autocall_barrier, protection_barrier=protection_barrier, 
                                 n_obs=obs_count, T=T, r=risk_free_rate/100, sigma_list=sigma_list, basket_type=basket_type, rho=rho)
        price, stderr = ac.price_monte_carlo(n_paths=int(n_paths))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr
    st.success(f'Price: {price:.2f} (StdErr: {stderr:.4f})')
    st.write(f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')


