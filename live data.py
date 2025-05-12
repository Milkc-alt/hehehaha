import yfinance
import numpy
import pandas
import streamlit as st
import scipy.ndimage
from scipy.ndimage import gaussian_filter1d
from arch import arch_model
import pandas

pds = st.text_input('choose number of days for the EMA (1mo,3mo,6mo,1y)')
interval = st.text_input("intervals of period (1m,2m,5m,10m,30m)")
ticker = st.text_input('symbol')

periods = -1 * int(input('number of days in days'))

SMA_data = yfinance.download(tickers=ticker,period=pds,interval='1d')
day_close_data = SMA_data['Close']
SMA = day_close_data.iloc[periods:-2].mean()
EMA = day_close_data.iloc[-1]*2/(1+periods)+SMA*(1-(2/(1+periods)))
MACD_SMA_1 = day_close_data.iloc[-12:-2].mean()
MACD_SMA_2 = day_close_data.iloc[-26:-2].mean()
MACD =  (day_close_data.iloc[-1]*9/(1+12)+MACD_SMA_1*(1-9/(1+12))) - (day_close_data.iloc[-1]*9/(1+26)+MACD_SMA_2*(1-9/(1+26)))

historical_data = yfinance.download(tickers=ticker, start='2023-1-1', end='2025-5-11')
historical_prices = historical_data['Close'].values
log_returns = numpy.log(historical_prices[1:]/historical_prices[:-1])

long_return = yfinance.download(tickers=ticker,period='1y',interval='1d')

# Get proper log returns over 1 year
log_returns_series = numpy.log(long_return['Close'] / long_return['Close'].shift(1)).dropna()
log_returns_series = log_returns_series.values.flatten()  # ensure 1D array
log_returns_series = pandas.Series(log_returns_series)    # safely convert to Series

# Fit GARCH model on this series
try:
    model = arch_model(log_returns_series, vol='EGarch', p=1, q=1, rescale=True)
    GARCH_results = model.fit(disp="off")
    sigma = GARCH_results.conditional_volatility.iloc[-1]

    # Protect against absurdly small sigma
    if numpy.isnan(sigma) or sigma < 1e-6:
        raise ValueError("Sigma too small, using fallback std dev.")
except Exception as e:
    print("GARCH failed or returned bad sigma:", e)
    sigma = log_returns_series.std()

S0 = float(historical_data['Close'].iloc[-1])
treasury_data = yfinance.download(tickers='^TNX',interval='1d')
sp500_data = yfinance.download(tickers='^GSPC',start='2009-1-1', end = '2025-01-01', interval='1mo')
risk_free_rate = float(treasury_data['Close'].iloc[-1]) /100
sp500_close = sp500_data['Close']

# Calculate percentage returns
percentage_returns = (sp500_close[1:].values - sp500_close[:-1].values) / sp500_close[:-1].values
month_return_rate = numpy.mean(percentage_returns)
annualized_return_rate = month_return_rate*12
market_risk_premium = annualized_return_rate - risk_free_rate
#sigma = numpy.std(log_returns)*numpy.sqrt(252)*0.9 #annual volatility
mu = market_risk_premium*0.8 + risk_free_rate #expected annual return
T = 2 #time in years
N = 252*2 #time steps
M = 100000
dt = T/N
#sigma = conditional_volatility.iloc[-1]
sigma = float(sigma)
sigma *= 0.8
mu = float(mu)   # ensure scalar
mu *= 0.8

price_paths = numpy.zeros((M,N))
price_paths[:,0] = S0

for i in range(1, N):
    Z = numpy.random.standard_normal(M)  # Brownian motion
    J = numpy.random.poisson(lam=0.05*dt, size=M)  # Poisson jump: 0 or 1 typically
    jump_size = numpy.clip(numpy.random.normal(loc=0, scale=0.01, size=M), -0.01, 0.01)
    jump_multiplier = numpy.clip(numpy.exp(jump_size), 0.98, 1.02)  # Limit jumps to ±10%
    # GBM with jump component
    price_paths[:, i] = numpy.log(price_paths[:,i-1]) + ((mu - 0.5 * sigma ** 2) * dt + sigma * numpy.sqrt(dt) * Z)
    price_paths[:, i] = numpy.clip(price_paths[:, i], 1e-2, 1e5)  # Cap prices to stay within reasonable range
    price_paths[:, i] *= numpy.where(J > 0, jump_multiplier, 1.0)
    price_paths[:, i] = numpy.exp(price_paths[:,i])
    # Apply max loss threshold to prevent extreme low values
    price_paths[:, i] = numpy.maximum(price_paths[:, i], 0.01)  # Prevent price from getting too close to 0

price_paths_smoothed = gaussian_filter1d(price_paths, sigma=2, axis=1)

final_prices = price_paths_smoothed[:,-1]
final_prices = numpy.array(final_prices)
prob_up = numpy.mean(final_prices>S0)
prob_down = 1-prob_up
print(f'Probability Price Increases: {prob_up: .2%}')
print(f'Probability Price decreases: {prob_down: .2%}')
print(f"Min simulated price: {final_prices.min()}")
print(f"Max simulated price: {final_prices.max()}")
print(sigma)
print(mu)
import matplotlib.pyplot as plt

plt.plot(price_paths[0, :])  # Plot a single path to visualize
plt.title(f'Simulated Price Path for {ticker}')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.show()

while True:
   data = yfinance.download(tickers=ticker, period='1d', interval=interval)
   latest_price = data['Close',].iloc[-1]  # Get the latest closing price
   latest_high = data['High'].iloc[-1]
   latest_volume = data['Volume'].iloc[-1]
   print(f'Price: {latest_price},High: {latest_high},Volume: {latest_volume},EMA: {EMA},MACD: {MACD}')
   time.sleep(5)

