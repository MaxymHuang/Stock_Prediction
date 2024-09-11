import yfinance as yf
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Fetch stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Step 2: Create target variable (price movement in next 2 days)
stock_data['Price_Diff'] = stock_data['Close'].shift(-2) - stock_data['Close']
stock_data['Target'] = (stock_data['Price_Diff'] > 0).astype(int)
stock_data.dropna(inplace=True)

# Step 3: Feature creation
stock_data['Open-Close'] = stock_data['Open'] - stock_data['Close']
stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
stock_data.dropna(inplace=True)

# Step 4: Split the data
features = ['Open-Close', 'High-Low', 'SMA_5', 'SMA_10', 'Volume_Change']
X = stock_data[features]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize KNN model with k=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# Step 6: Train the KNN model
knn_model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model accuracy: {accuracy * 100:.2f}%")
