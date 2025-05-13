import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Simulator", layout="wide")

# ------------------------ Sidebar Inputs ------------------------
st.sidebar.title("Simulation Settings")
ticker = st.sidebar.text_input("Stock Symbol", value="AAPL")
ema_window = st.sidebar.slider("EMA Window (days)", 5, 100, 20)
sim_days = st.sidebar.slider("Simulation Length (days)", 50, 1000, 252)
sim_paths = st.sidebar.slider("Number of Paths", 1000, 100000, 10000, step=1000)
theta = st.sidebar.slider("Mean Reversion Strength (θ)", 0.01, 1.0, 0.1)
jump_freq = st.sidebar.slider("Avg Jumps per Year", 0.0, 10.0, 2.0)
run_sim = st.sidebar.button("Run Simulation")

# ------------------------ Fetch Stock Data ------------------------
@st.cache_data
def get_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d")
    return data

data = get_data(ticker)
close_prices = data['Close'].dropna()

# ------------------------ Calculate EMA & MACD ------------------------
ema = close_prices.ewm(span=ema_window, adjust=False).mean()
ema_12 = close_prices.ewm(span=12, adjust=False).mean()
ema_26 = close_prices.ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26
signal = macd.ewm(span=9, adjust=False).mean()

# ------------------------ Plot Real Stock Data ------------------------
st.header(f"{ticker} Historical Price with EMA")
fig1, ax1 = plt.subplots()
ax1.plot(close_prices, label="Close")
ax1.plot(ema, label=f"{ema_window}-Day EMA", linestyle="--")
ax1.legend()
ax1.grid()
st.pyplot(fig1)

st.header(f"{ticker} MACD")
fig2, ax2 = plt.subplots()
ax2.plot(macd, label="MACD", color='green')
ax2.plot(signal, label="Signal", color='red')
ax2.axhline(0, linestyle='--', color='gray')
ax2.legend()
ax2.grid()
st.pyplot(fig2)

# ------------------------ Monte Carlo Simulation ------------------------
if run_sim:
    st.header("Monte Carlo Simulation")

    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    series = pd.Series(log_returns.values.flatten())

    # EGARCH volatility estimate
    try:
        model = arch_model(series, vol='EGarch', p=1, q=1, rescale=True)
        result = model.fit(disp='off')
        sigma = result.conditional_volatility.iloc[-1]
        if np.isnan(sigma) or sigma < 1e-6:
            raise ValueError
    except:
        sigma = series.std()

    # Risk-free rate from 10Y treasury
    rfr_data = yf.download("^TNX", period="1y", interval="1d")
    rfr = np.mean(rfr_data['Close']) / 100 * 365

    # Market return from S&P 500
    sp500 = yf.download("^GSPC", period="1y", interval="1mo")['Close']
    monthly_returns = sp500.pct_change().dropna()
    expected_return = monthly_returns.mean() * 12
    market_risk_premium = expected_return - rfr
    mu = market_risk_premium * 0.8 + rfr

    # Simulation setup
    S0 = close_prices.iloc[-1]
    T = sim_days / 252
    N = sim_days
    dt = T / N
    long_mean = np.log(S0)

    paths = np.zeros((sim_paths, N))
    paths[:, 0] = S0

    for i in range(1, steps):
        Z = np.random.normal(0, 1, num_simulations)
        J = np.random.binomial(1, jump_prob, num_simulations)
        jump_size = np.random.normal(jump_mean, jump_std, num_simulations)
        jump_mult = np.clip(np.exp(jump_size), 0.95, 1.05)

        log_prev = np.log(paths[:, i - 1])
        log_new = log_prev + theta * (float(long_mean) - log_prev) * dt + sigma * np.sqrt(dt) * Z
        paths[:, i] = np.exp(log_new)
        paths[:, i] *= np.where(J > 0, jump_mult, 1.0)
        paths[:, i] = np.maximum(paths[:, i], 0.01)

    smooth_path = gaussian_filter1d(paths, sigma=2, axis=1)

    # Plot first simulated path
    fig3, ax3 = plt.subplots()
    ax3.plot(smooth_path[0], color='purple')
    ax3.set_title("Simulated Price Path (1 Sample)")
    ax3.grid()
    st.pyplot(fig3)

    final_prices = smooth_path[:, -1]
    prob_up = np.mean(final_prices > S0)
    prob_down = 1 - prob_up

    # Stats Output
    st.subheader("📊 Simulation Probabilities & Stats")
    st.markdown(f"""
    - **Start Price (S₀):** `${S0:.2f}`
    - **Probability Price Increases:** `{prob_up:.2%}`  
    - **Probability Price Decreases:** `{prob_down:.2%}`  
    - **Min Simulated Price:** `${final_prices.min():.2f}`  
    - **Max Simulated Price:** `${final_prices.max():.2f}`  
    - **Mean Final Price:** `${final_prices.mean():.2f}`
    - **Median Final Price:** `${np.median(final_prices):.2f}`
    - **Volatility (σ):** `{sigma:.4f}`  
    - **Expected Return (μ):** `{mu:.4f}`
    """)

    # Histogram
    st.subheader("📈 Distribution of Final Prices")
    fig4, ax4 = plt.subplots()
    ax4.hist(final_prices, bins=100, color='skyblue', edgecolor='black')
    ax4.set_xlabel("Final Price")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Histogram of Final Simulated Prices")
    ax4.grid()
    st.pyplot(fig4)

# ------------------------ Footer ------------------------
st.markdown("---")
st.caption("Built with 💻 using Streamlit | Monte Carlo GBM with Jumps, EGARCH Volatility & Mean Reversion")
