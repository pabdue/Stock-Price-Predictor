# Stock Price Predictor

This Python script predicts future stock prices for a set of predefined stocks using historical price data and a Long Short-Term Memory (LSTM) neural network model.

## Features

- Updates historical stock data for a list of specified stocks
- Trains an LSTM model to predict future stock prices
- Displays prediction results including root mean squared error (RMSE) and predicted price

## Getting Started

1. Install the required dependencies:
   - yfinance
   - numpy
   - pandas
   - scikit-learn
   - keras

2. Modify the list of stocks (`stocks`) in the script to include the stocks you want to predict.

3. Run: `stock_price_predictor.py`

4. The script will update the historical stock data, train the model, and display the predictions for each stock.

## Usage

- The `updateCSV(stock)` function fetches and updates historical stock data for the given stock symbol.
- The `predict(stock)` function trains an LSTM model on the historical data and predicts future stock prices.
- The `displayResults(predictions)` function displays the prediction results.

## Disclaimer

- The predictions are based on historical data and should be used for informational purposes only.

## License

This project is licensed under the [MIT License](LICENSE).
