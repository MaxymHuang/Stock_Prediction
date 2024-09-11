# Import Libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import logging

def calculate_RSI(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Set logging level for warnings
tf.get_logger().setLevel(logging.ERROR)

# Check if TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Set mixed precision policy for faster training on supported GPUs
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Download Stock Data
stock_symbol = 'SPY'
data = yf.download(stock_symbol, start='2020-01-01', end='2024-08-05')

# Compute MACD
short_window = 12
long_window = 26
signal_window = 9

data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

# Select relevant features: Closing price, Volume, MACD
data = data[['Close', 'Volume', 'MACD']]

# Drop any NaN values that may result from MACD calculation
data = data.dropna()

# Normalize the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the Data into Training and Testing Sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create Datasets for LSTM Model
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 0])  # Target is the closing price
    return np.array(X), np.array(Y)

time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1, dtype='float32'))  # Ensure output is float32

model.compile(optimizer='adam', loss='mean_squared_error')

# Define a learning rate schedule function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1).numpy())

# Create a LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(scheduler)

# Train the Model with Learning Rate Scheduler
model.fit(X_train, y_train, batch_size=1, epochs=25, callbacks=[lr_scheduler])

# Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform Back to Original Scale (Only the 'Close' price needs to be transformed back)
train_predict = scaler.inverse_transform(np.hstack((train_predict, np.zeros((train_predict.shape[0], 2)))))
y_train = scaler.inverse_transform(np.hstack((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 2)))))
test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], 2)))))
y_test = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2)))))

# Plot Predictions
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, 0] = train_predict[:, 0]

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, 0] = test_predict[:, 0]

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(scaler.inverse_transform(scaled_data)[:, 0], label='Actual Price')
plt.plot(range(time_step, len(train_predict) + time_step), train_predict[:, 0], label='Train Prediction')
plt.plot(range(len(train_predict) + (time_step * 2) + 1, len(scaled_data) - 1), test_predict[:, 0], label='Test Prediction')
plt.legend()
plt.show()

