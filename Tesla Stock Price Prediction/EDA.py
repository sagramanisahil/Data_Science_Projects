import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("TESLA.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Close price vs Date
plt.figure()
plt.plot(df['Date'], df['Close'])
plt.title("Close Price vs Date")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Volume vs Date
plt.figure()
plt.plot(df['Date'], df['Volume'])
plt.title("Trading Volume vs Date")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

# OHLC Prices Over Time
plt.figure()
plt.plot(df['Date'], df['Open'], label='Open')
plt.plot(df['Date'], df['High'], label='High')
plt.plot(df['Date'], df['Low'], label='Low')
plt.plot(df['Date'], df['Close'], label='Close')
plt.title("OHLC Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Daily Price Volatility
df['Daily_Range'] = df['High'] - df['Low']

plt.figure()
plt.plot(df['Date'], df['Daily_Range'])
plt.title("Daily Price Volatility")
plt.xlabel("Date")
plt.ylabel("High - Low")
plt.show()

# Distribution of Close Price 
plt.figure()
plt.hist(df['Close'], bins=20)
plt.title("Distribution of Close Price")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.show()

# Distribution of Volume
plt.figure()
plt.hist(df['Volume'], bins=20)
plt.title("Distribution of Trading Volume")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.show()

# Correlation Matrix
plt.figure()
plt.imshow(df.corr())
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()