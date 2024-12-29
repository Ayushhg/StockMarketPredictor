# Stock Market Predictor

## Overview
The **Stock Market Predictor** is a machine learning-based application designed to forecast stock prices and trends using historical data. The project uses a GRU-LSTM hybrid model to predict stock performance over different time periods (week, month, quarter). It integrates sentiment analysis, financial ratios, and real-time data retrieval to provide comprehensive stock performance insights.

The app allows users to input stock symbols, visualize stock price trends, and access key financial metrics to assess the profitability and performance of stocks.

## Key Features
- **GRU-LSTM Hybrid Model**: A combination of Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) networks for stock price prediction.
- **Sentiment Analysis**: Analyzes market sentiment to enhance prediction accuracy.
- **Financial Ratios**: Includes profitability ratios, liquidity ratios, solvency ratios, EPS, P/E, P/B, Dividend Yield, SMA, EMA, RSI, and MACD.
- **Visualization**: Displays stock price trends and financial ratios in interactive charts.
- **Real-Time Data**: Fetches up-to-date stock data using the Yfinance API.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-market-predictor.git
2. Navigate to the project directory:
   ```bash
   cd stock-market-predictor
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

Run the Streamlit app:
   ```bash
   streamlit run app.py
```

Open the app in your browser by navigating to http://localhost:8501.

Input a stock symbol (e.g., "AAPL" for Apple) to retrieve the stock data.

For Indian Stocks, add .NS to the Symbol for Example TATAMOTORS.NS

View the stock price prediction and financial analysis, including charts and metrics like:

Profitability Ratios
Liquidity Ratios
Solvency Ratios
EPS, P/E, P/B, Dividend Yield
Moving Averages (SMA, EMA)
RSI, MACD
Model Explanation
GRU-LSTM Hybrid Model: This model leverages the strengths of both GRU and LSTM layers to capture the temporal dependencies in stock price movements. It provides accurate predictions for future stock prices.

Sentiment Analysis: Analyzes the sentiment of news articles and social media content to predict stock movement trends.

Financial Ratios: The app calculates key ratios and metrics to evaluate the financial health of a company and predict its stock performance.

Technologies Used
Python
TensorFlow/Keras (for model training)
Streamlit (for web app)
Yfinance API (for stock data retrieval)
Pandas, Numpy, Matplotlib (for data manipulation and visualization)
Contribution
Feel free to fork the project and submit pull requests. Contributions are welcome!
