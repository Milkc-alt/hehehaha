import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import time
from scipy.ndimage import gaussian_filter1d
from arch import arch_model
import matplotlib.pyplot as plt

st.title("Stock Price Probability")

# User inputs
pds = st.selectbox('Choose number of days for the EMA', ['1mo', '3mo', '6mo', '1y'])
interval = st.selectbox("Interval of period", ['1m', '2m', '5m', '10m', '30m'])
ticker = st.text_input('Stock Symbol (e.g., AAPL)', 'AAPL')

periods = -1 * st.number_input('Number of days (for EMA)', min_value=1, value=20)

if ticker:
    # Calculate EMA and MACD
    SMA_data = yf.download(tickers=ticker, period=pds, interval='1d')
    day_close_data = SMA_data['Close']
    SMA = day_close_data.iloc[periods:-2].mean()
    EMA = day_close_data.iloc[-1]*2/(1+abs(periods))+SMA*(1-(2/(1+abs(periods))))
    MACD_SMA_1 = day_close_data.iloc[-12:-2].mean()
    MACD_SMA_2 = day_close_data.iloc[-26:-2].mean()
    MACD = (day_close_data.iloc[-1]*9/(1+12)+MACD_SMA_1*(1-9/(1+12))) - (day_close_data.iloc[-1]*9/(1+26)+MACD_SMA_2*(1-9/(1+26)))

    # Historical data and log returns
    historical_data = yf.download(tickers=ticker, start='2023-1-1', end='2025-5-11')
    historical_prices = historical_data['Close'].values
    log_returns = np.log(historical_prices[1:] / historical_prices[:-1])

    long_return = yf.download(tickers=ticker, period='1y', interval='1d')
    log_returns_series = np.log(long_return['Close'] / long_return['Close'].shift(1)).dropna().values.flatten()
    log_returns_series = pd.Series(log_returns_series)

    # GARCH model
    try:
        model = arch_model(log_returns_series, vol='EGarch', p=1, q=1, rescale=True)
        GARCH_results = model.fit(disp="off")
        sigma = GARCH_results.conditional_volatility.iloc[-1]
        if np.isnan(sigma) or sigma < 1e-6:
            raise ValueError("Sigma too small, using fallback std dev.")
    except Exception as e:
        st.warning(f"GARCH failed or returned bad sigma: {e}")
        sigma = log_returns_series.std()

    S0 = float(historical_data['Close'].iloc[-1])
    treasury_data = yf.download(tickers='^TNX', interval='1d')
    sp500_data = yf.download(tickers='^GSPC', start='2009-1-1', end='2025-01-01', interval='1mo')
    risk_free_rate = float(treasury_data['Close'].iloc[-1]) / 100
    sp500_close = sp500_data['Close']

    percentage_returns = (sp500_close[1:].values - sp500_close[:-1].values) / sp500_close[:-1].values
    month_return_rate = np.mean(percentage_returns)
    annualized_return_rate = month_return_rate * 12
    market_risk_premium = annualized_return_rate - risk_free_rate
    mu = market_risk_premium * 0.8 + risk_free_rate

    # Monte Carlo simulation
    T = 2
    N = 252 * 2
    M = 100000
    dt = T / N

    sigma *= 0.8
    mu *= 0.8

    price_paths = np.zeros((M, N))
    price_paths[:, 0] = S0

    for i in range(1, N):
        Z = np.random.standard_normal(M)
        J = np.random.poisson(lam=0.02 * dt, size=M)
        jump_size = np.clip(np.random.normal(loc=0, scale=0.01, size=M), -0.01, 0.01)
        jump_multiplier = np.clip(np.exp(jump_size), 0.98, 1.02)
        price_paths[:, i] = np.log(price_paths[:, i - 1]) + ((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        price_paths[:, i] = np.clip(price_paths[:, i], 1e-2, 1e5)
        price_paths[:, i] *= np.where(J > 0, jump_multiplier, 1.0)
        price_paths[:, i] = np.exp(price_paths[:, i])
        price_paths[:, i] = np.maximum(price_paths[:, i], 0.01)

    #price_paths_smoothed = gaussian_filter1d(price_paths, sigma=2, axis=1)

    final_prices = price_paths_smoothed[:, -1]
    prob_up = np.mean(final_prices > S0)
    prob_down = 1 - prob_up

    st.subheader("Simulation Results")
    st.metric("Probability Price Increases", f"{prob_up:.2%}")
    st.metric("Probability Price Decreases", f"{prob_down:.2%}")
    st.write(f"Min simulated price: {final_prices.min():.2f}")
    st.write(f"Max simulated price: {final_prices.max():.2f}")
    st.write(f"Estimated Volatility (sigma): {sigma:.4f}")
    st.write(f"Expected Return (mu): {mu:.4f}")

    # Plot single simulated path
    st.subheader("Sample Simulated Path")
    fig, ax = plt.subplots()
    ax.plot(price_paths[0, :])
    ax.set_title(f"Simulated Price Path for {ticker}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # Live Price Display
    st.subheader("Live Price Feed")
    if st.button("Start Live Feed"):
        placeholder = st.empty()
        for _ in range(10):  # Limit loop for Streamlit UI (not infinite)
            data = yf.download(tickers=ticker, period='1d', interval=interval)
            latest_price = data['Close'].iloc[-1]
            latest_high = data['High'].iloc[-1]
            latest_volume = data['Volume'].iloc[-1]
            with placeholder.container():
                st.write(f"Price: {latest_price:.2f}, High: {latest_high:.2f}, Volume: {latest_volume:.0f}")
                st.write(f"EMA: {EMA:.2f}, MACD: {MACD:.2f}")
            time.sleep(5)
