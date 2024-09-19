# Import Libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

def calculate_RSI(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Download Stock Data
stock_symbol = 'SPY'
data = yf.download(stock_symbol, start='2020-01-01', end='2024-08-05')

# Get acutal data
a_data = yf.download(stock_symbol, start='2024-08-06', end='2024-08-17')
a_data.dropna(inplace=True)
a_data = a_data['Close']

# Compute MACD
short_window = 12
long_window = 26
signal_window = 9


data['RSI'] = calculate_RSI(data, 10)
data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

# Select relevant features: Closing price, Volume, MACD
data = data[['Close', 'Volume', 'MACD', 'RSI']]

# Drop any NaN values that might exist
data.dropna(inplace=True)

# Prepare the dataset for multiple days ahead prediction
n_future_days = 5  # Predict the next 5 trading days
n_past_days = 60   # Use past 60 days to predict future

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the dataset with multiple future days
X_train = []
y_train = []
for i in range(n_past_days, len(scaled_data) - n_future_days):
    X_train.append(scaled_data[i - n_past_days:i])
    y_train.append(scaled_data[i:i + n_future_days, 0])  # Predict 'Close' for future days

X_train, y_train = np.array(X_train), np.array(y_train)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(n_future_days))  # Output layer adjusted for future days

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict the next few trading days
predictions = model.predict(X_train[-1].reshape(1, n_past_days, X_train.shape[2]))

# Flatten predictions and reshape to match the dimensions
predictions = predictions.flatten().reshape(-1, 1)

# Create an array of zeros for other features to inverse transform
dummy_array = np.zeros((n_future_days, scaled_data.shape[1] - 1))

# Concatenate and inverse transform
predicted_prices = scaler.inverse_transform(np.hstack((predictions, dummy_array)))[:,0]  # Only take 'Close' values
predicted_prices

# Display predicted prices for the next few trading days
print("Predicted prices for the next {} days:".format(n_future_days))
print(predicted_prices)

# Display actual prices of the trading days
print("Actual prices for the next {} days:".format(n_future_days))
print(a_data)

