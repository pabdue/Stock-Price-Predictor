# --------------------------------------------------------------------------------
# Name: Pablo Duenas
# Date: May 2023
# Description: This code uses historical stock price data to create a predictive 
# model based on LSTM neural networks, which aims to forecast future stock prices 
# for a predefined list of stocks. 
# --------------------------------------------------------------------------------
import yfinance as yf
import csv
from datetime import date
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to update stock data in a CSV file
def updateCSV(stock):
    """
    Downloads historical stock data for the given stock symbol and saves it to a CSV file.
    
    Parameters:
    stock (str): Stock symbol (e.g., 'GOOGL').

    Returns:
    None
    """
    # Get today's date
    today = date.today().strftime("%Y-%m-%d")

    # Set the start date as January 1, 2018
    start_date = '2018-01-01'

    # Retrieve and save the stock data as CSV files
    data = yf.download(stock, start=start_date, end=today)
    filepath = f"Stock-Price-Predictor/Datasets/{stock}_data.csv"
    data.to_csv(filepath)
    print(f"Data for {stock} saved to {filepath}")

# Function to display prediction results
def displayResults(predictions):
    """
    Display the prediction results for each stock.

    Parameters:
    predictions (list): List of prediction results for each stock.

    Returns:
    None
    """
    for info in predictions:
        print('-------------------------')
        print('Stock:',info[0])
        print('Current Price:', info[1])
        print(info[2],'\n', info[3])
        print('-------------------------')


def currentStockPrice(CSV_file):
    with open(CSV_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        last_row = data[-1]
        adjClose = last_row[4]
    return adjClose

# Function to create input sequences and labels
def create_sequences(data, sequence_length):
    """
    Create input sequences and corresponding labels from the given data.

    Parameters:
    data (numpy.ndarray): The input data.
    sequence_length (int): Length of input sequences.

    Returns:
    numpy.ndarray: Input sequences.
    numpy.ndarray: Corresponding labels.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Function to perform stock price prediction
def predict(stock):
    """
    Predict the future stock price for the given stock symbol using an LSTM model.

    Parameters:
    stock (str): Stock symbol (e.g., 'GOOGL').

    Returns:
    list: Prediction results for the stock.
    """
    filePath = f"Stock-Price-Predictor/Datasets/{stock}_data.csv"
    results = []
    results.append(stock)
    results.append(currentStockPrice(filePath))
    
    # Load the stock data from the CSV file
    data = pd.read_csv(filePath)

    # Extract the 'Close' prices as the target variable
    target = data['Close'].values

    # Scale the target variable to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = scaler.fit_transform(target.reshape(-1, 1))

    # Split the data into training and testing sets
    train_size = int(len(target_scaled) * 0.8)
    train_data = target_scaled[:train_size]
    test_data = target_scaled[train_size:]

    # Define the sequence length for input data
    sequence_length = 10

    # Create input sequences and labels
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Scale back the predictions to the original range
    predictions = scaler.inverse_transform(predictions)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test))**2))
    results.append(f'\033[31mRoot mean squared error (RMSE): \033[33m{rmse}\033[0m')

    # Make a prediction for stock price
    last_sequence = np.array([test_data[-sequence_length:]])
    prediction = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    results.append(f"\033[32mPredicted price for {stock}: \033[33m{predicted_price}\033[0m")
    return results

# Main execution
stocks = ['GOOGL', 'MSFT', 'AAPL', 'INTC', 'ADBE']
predictions = []

for stock in stocks:
    updateCSV(stock)
    stockInfo = predict(stock)
    predictions.append(stockInfo)

# Display prediction results
for info in predictions:
    print('-------------------------')
    print('Stock:',info[0])
    print('Current Price:', info[1])
    print(info[2],'\n', info[3])
    print('-------------------------')
