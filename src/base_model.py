import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Data Collection
stock_data = yf.download('SPY', start='2010-01-01', end='2023-01-01')
data = stock_data['Close'].values.reshape(-1, 1)

# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Feature Engineering (Sliding Window)
window_size = 60
X, y = [], []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 4: Define the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Step 5: Training the Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=25, batch_size=32)

# Step 6: Evaluation
test_data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')['Close'].values.reshape(-1, 1)
scaled_test_data = scaler.transform(test_data)
X_test, y_test = [], []
for i in range(window_size, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-window_size:i, 0])
    y_test.append(scaled_test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(test_data[window_size:], color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
