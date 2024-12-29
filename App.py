import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import sys
import codecs

#sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

FINAL_SCORE = 0

stock_name = st.text_input("Enter Stock Symbol (e.g., AAPL):")

if stock_name:
    try:
        stock_data = yf.download(stock_name, start=datetime.now() - timedelta(days=1000), end=datetime.now())
        stock_data["symbol"] = stock_name
        if not stock_data.empty:
            filename = f"{stock_name}_data.csv"
            stock_data.to_csv(filename)
            st.success(f"Data saved to {filename}")
        else:
            st.error("No data found for the given symbol.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


st.title("Stock Price Predictor")

# selected_stock = st.selectbox("Select a stock", stock_symbols)

df =pd.read_csv(f"{stock_name}_data.csv", header= 0)

# Rename the first column to "Date"
df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# Drop the first row
df = df.iloc[1:].reset_index(drop=True)

df.symbol.value_counts()
df.symbol.unique()
df.symbol.unique()[0:20]
st.write(df.info())
st.write(df.describe())
df.isnull().sum()
df.Date.unique()
pd.DataFrame(df.Date.unique()).to_excel('dates.xlsx')
df.duplicated().sum()
comp_plot = ['TCS.NS', 'RELIANCE.NS', 'TATAMOTORS.NS', 'MRF.NS', 'HDFCBANK.NS',
       'ADANIPOWER.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS',
       'LICI.NS', 'INFY.NS', 'ITC.NS', 'HCLTECH.NS', 'ADANIENT.NS',
       'AXISBANK.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS',
       'MARUTI.NS', 'COALINDIA.NS']


def plotter(code):
    global closing_stock ,opening_stock
    f, axs = plt.subplots(2,2,figsize=(15,8))
    plt.subplot(212)
    company = df[df['symbol']==code]
    company = company.Open.values.astype('float32')
    company = company.reshape(-1, 1)
    opening_stock = company
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel(code + " open stock prices")
    plt.title('prices Vs Time')
    plt.plot(company , 'g')
    plt.subplot(211)
    company_close = df[df['symbol']==code]
    company_close = company_close.Close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time')
    plt.ylabel(code + " close stock prices")
    plt.title('prices Vs Time')
    plt.grid(True)
    plt.plot(company_close , 'b')
    st.pyplot(plt)

plotter(stock_name)
plt.clf()
stocks= np.array (df[df.symbol.isin ([stock_name])].Close)

stocks = stocks.reshape(len(stocks) , 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
stocks = scaler.fit_transform(stocks)

train = int(len(stocks) * 0.80)

test = len(stocks) - train

train = stocks[0:train]
train.shape

test = stocks[len(train) : ]

def process_data(data , n_features):
    dataX, dataY = [], []
    for i in range(len(data)-n_features):
        a = data[i:(i+n_features), 0]
        dataX.append(a)
        dataY.append(data[i + n_features, 0])
    return np.array(dataX), np.array(dataY)

n_features = 2

trainX, trainY = process_data(train, n_features)

pd.DataFrame(trainX).head(10)

pd.DataFrame(trainY)

testX, testY = process_data(test, n_features)

stocksX, stocksY = process_data(stocks, n_features)

trainX = trainX.reshape(trainX.shape[0] , 1 ,trainX.shape[1])

testX = testX.reshape(testX.shape[0] , 1 ,testX.shape[1])

stocksX= stocksX.reshape(stocksX.shape[0] , 1 ,stocksX.shape[1])

import math
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam , SGD , RMSprop

filepath="stock_weights1.keras"
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

model = Sequential()

model.add(GRU(256 , input_shape = (1 , n_features) , return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(64 ,  activation = 'relu'))


model.add(Dense(1))

st.write(model.summary())

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.0005) , metrics = ['mean_squared_error'])

history = model.fit(trainX, trainY, epochs=10 , batch_size = 128 ,
          callbacks = [checkpoint , lr_reduce] , validation_data = (testX,testY))

test_pred = model.predict(testX)

test_pred = scaler.inverse_transform(test_pred)

testY = testY.reshape(testY.shape[0] , 1)

testY = scaler.inverse_transform(testY)

from sklearn.metrics import r2_score
x = r2_score(testY,test_pred)
st.write("R2 Score: ",x)

st.write("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(testY , 'b')
plt.plot(test_pred , 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
st.pyplot(plt)
plt.clf()



train_pred = model.predict(trainX)
train_pred = scaler.inverse_transform(train_pred)
trainY = trainY.reshape(trainY.shape[0] , 1)
trainY = scaler.inverse_transform(trainY)
st.write('Display Accuracy Training Data')
st.write(r2_score(trainY,train_pred))

st.write("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(trainY  , 'b')
plt.plot(train_pred, 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
st.pyplot(plt)
plt.clf()
stocks_pred = model.predict(stocksX)
stocks_pred = scaler.inverse_transform(stocks_pred)
stocksY = stocksY.reshape(stocksY.shape[0] , 1)
stocksY = scaler.inverse_transform(stocksY)
print ('Display Accuracy Training Data')
st.write(r2_score(stocksY,stocks_pred))


plt.rcParams["figure.figsize"] = (15,7)
plt.plot(stocksY  , 'b')
plt.plot(stocks_pred, 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
st.pyplot(plt)

results= df[df.symbol.isin ([stock_name])]
results= results [2:]
results = results.reset_index(drop=True)
df_stocks_pred= pd.DataFrame(stocks_pred, columns = ['Close_Prediction'])
results= pd.concat([results,df_stocks_pred],axis =1)
results.to_excel('results.xlsx')
st.dataframe(results)

if results['Close_Prediction'].iloc[-1] < results['Close_Prediction'].iloc[-2]:
    st.write("Stock Prediction For Tomorrow: Lower")
    ML_result = 'lower'
elif results['Close_Prediction'].iloc[-1] > results['Close_Prediction'].iloc[-2]:
    st.write("Stock Prediction For Tomorrow: Higher")
    ML_result = 'Higher'
else:
    st.write("Stock Prediction For Tomorrow: Same")
    ML_result = 'Same'


#SentimentAnalysis
from newsapi import NewsApiClient
from textblob import TextBlob

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='012f78db2d1642ad83735f2a30537bb0')

# # Fetch stock data from yfinance
# def fetch_stock_data(symbol):
#     end_date = datetime.date.today()
#     start_date = end_date - datetime.timedelta(days=90)
#     stock = yf.download(symbol, start=start_date, end=end_date)
#     return stock

# Fetch news articles and perform sentiment analysis
def fetch_news_and_analyze(full_name):
    articles = newsapi.get_everything(q=full_name, language='en', sort_by='relevancy', page_size=10)
    sentiment_scores = []
    titles = []

    for article in articles['articles']:
        title = article['title']
        description = article['description']
        content = title + " " + (description if description else "")

        # Sentiment analysis using TextBlob
        blob = TextBlob(content)
        sentiment = blob.sentiment.polarity
        sentiment_scores.append(sentiment)
        titles.append(title)
    
    return titles, sentiment_scores

def get_stock_name(symbol):
    stock = yf.Ticker(symbol)
    return stock.info.get('longName', symbol)  # Fallback to symbol if longName is not available



if stock_name:
    # Fetch the full name of the stock
    full_name = get_stock_name(stock_name)

    # Display full name of stock
    st.write(f"Full Name of Stock: {full_name}")

    # Fetch stock data
    # stock_data = fetch_stock_data(stock_name)
    
    # Display stock data
    st.subheader(f"Stock Data for {stock_name}")
    st.line_chart(stock_data['Close'])

    # Fetch news and analyze sentiment
    if full_name:
        titles, sentiment_scores = fetch_news_and_analyze(full_name)
        
        # Display sentiment analysis
        st.subheader(f"Sentiment Analysis for {full_name}")
        sentiment_df = pd.DataFrame({
            'Article Title': titles,
            'Sentiment Score': sentiment_scores
        })
        st.write(sentiment_df)
        sentiment_result = "Negative"
        # Display overall sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        if(avg_sentiment>0):
          FINAL_SCORE+=1
          sentiment_result = "Positive"
        st.write(f"Overall Sentiment: {'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'}")
        

### Mathematical Analysis

# Streamlit App Title

# Input for Stock Symbol
symbol = stock_name

if symbol:
    # Fetch stock data using yfinance
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period="1y")

    # Display stock information and history
    st.subheader(f"{symbol} Stock Data")
    st.write(stock_data.tail())
    
    # Line chart for stock price
    st.line_chart(stock_data['Close'], width=0, height=300, use_container_width=True)

    # Fetch detailed financial data
    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    dividends = stock.dividends

    # Calculate Profitability Ratios
    net_income = income_stmt.get("Net Income", [None])[0]
    revenue = income_stmt.get("Total Revenue", [None])[0]
    gross_profit = income_stmt.get("Gross Profit", [None])[0]
    profitability_ratios = {
        "Net Profit Margin": (net_income / revenue) if revenue else "Data not available",
        "Gross Margin": (gross_profit / revenue) if revenue and gross_profit else "Data not available",
    }

    # Calculate Liquidity Ratios
    current_assets = balance_sheet.get("Total Current Assets", [None])[0]
    current_liabilities = balance_sheet.get("Total Current Liabilities", [None])[0]
    inventory = balance_sheet.get("Inventory", [None])[0]
    liquidity_ratios = {
        "Current Ratio": (current_assets / current_liabilities) if current_assets and current_liabilities else "Data not available",
        "Quick Ratio": ((current_assets - inventory) / current_liabilities) if current_assets and current_liabilities and inventory else "Data not available",
    }

    # Calculate Solvency Ratios
    total_debt = balance_sheet.get("Total Debt", [None])[0]
    shareholder_equity = balance_sheet.get("Shareholder Equity", [None])[0]
    solvency_ratios = {
        "Debt to Equity": (total_debt / shareholder_equity) if total_debt and shareholder_equity else "Data not available",
    }

    # Dividend Yield
    dividend_yield = (dividends.sum() / stock_data["Close"][-1]) if not dividends.empty else "Data not available"
    
    #last 10 days data to compare SMA and EMAs
    stock_data_for_sma = yf.download(symbol, period="10d", interval="1d")
    closing_prices_for_sma = stock_data['Close'].to_numpy()

    # Calculate Technical Indicators
    sma_50 = stock_data['Close'].rolling(window=50).mean().iloc[-1]
    temp_ans=0
    for i in closing_prices_for_sma:
      if sma_50>=i:
        temp_ans+=1
    if temp_ans>=9:
      FINAL_SCORE+=1
      SMA_result = 'Good'
    else:
      SMA_result = 'Bad'

    ema_50 = stock_data['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    for i in closing_prices_for_sma:
      if ema_50>=i:
        temp_ans+=1
    if temp_ans>=9:
      FINAL_SCORE+=1
      EMA_result = 'Good'
    else:
      EMA_result = 'Bad'
    
    # RSI Calculation
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    RSI_result = 'Bad'
    if rsi>40:
      if rsi<60:
        FINAL_SCORE+=1
        RSI_result = 'Good'
    
    # MACD Calculation
    exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    MACD_result = 'Bad'
    if (macd>0).all():
      FINAL_SCORE+=1
      MACD_result = 'Good'

    st.subheader("Technical Indicators")
    st.write(f"50-Day SMA: {sma_50}")
    st.write(f"50-Day EMA: {ema_50}")
    st.write(f"RSI (14): {rsi}")
    st.write(f"MACD (12, 26): {macd.iloc[-1]}")
    st.write(f"Dividend Yield: {dividend_yield}")

    # Explanation for Ratios and Indicators
    st.subheader("Indicator Explanations")
    st.write("""
    **Net Profit Margin**: Indicates profitability. Higher values suggest good cost management and profit generation.

    **Gross Margin**: Reflects the efficiency of production. Higher values generally indicate healthier profit potential.

    **Current Ratio**: Measures liquidity. A value above 1 suggests that the company can cover its short-term liabilities.

    **Quick Ratio**: Similar to Current Ratio but excludes inventory. Indicates immediate liquidity.

    **Debt to Equity**: Measures leverage. Lower values indicate a conservative use of debt.

    **Dividend Yield**: Higher yield often attracts income-focused investors.

    **50-Day SMA**: A simple average of stock prices over 50 days. Helps identify price trends.

    **50-Day EMA**: Like SMA, but more sensitive to recent prices. Useful for spotting reversals.

    **RSI (14)**: Indicates overbought (>70) or oversold (<30) conditions.
    
    **MACD**: Shows trend direction and momentum. Positive MACD suggests an upward trend.
    """)

    st.write("Final Score = ",FINAL_SCORE,"Out of 5")

    stock_extra_info = yf.Ticker(symbol)
    dict_info = stock_extra_info.info

    # Remove 'companyOfficers' from the dictionary
    if 'companyOfficers' in dict_info:
        del dict_info['companyOfficers']

    # Print the remaining information
    st.write(dict_info)

    dividend_result = 'Bad'
    trailingPE_result = 'Bad'
    Beta_result = 'Bad'
    returnOnEquity_result = 'Bad'
    debtToEquity_result = 'Bad'
    profitMargin_result = 'Bad'
    revenueGrowth_result = 'Bad'
    earningGrowth_result = 'Bad'
    operatingMargins_result = 'Bad'
    current_ratio_result = 'Bad'
    
    overall_score = 0
    # 1. Check Dividend Rate
    
    if 'dividendRate' in dict_info and dict_info['dividendRate'] > 50:  # Example threshold for good dividend rate
        overall_score += 1
        dividend_result = 'Good'
    
    # 2. Check P/E Ratio
    if 'trailingPE' in dict_info and 15 <= dict_info['trailingPE'] <= 25:
        overall_score += 1
        trailingPE_result = 'Good'
    
    # 3. Check Beta
    if 'beta' in dict_info and dict_info['beta'] < 1:
        overall_score += 1
        Beta_result = 'Good'
    
    # 4. Check Return on Equity (ROE)
    if 'returnOnEquity' in dict_info and dict_info['returnOnEquity'] > 0.15:
        overall_score += 1
        returnOnEquity_result = 'Good'
    
    # 5. Check Debt-to-Equity Ratio
    if 'debtToEquity' in dict_info and dict_info['debtToEquity'] < 1:
        overall_score += 1
        debtToEquity_result = 'Good'
    
    # 6. Check Profit Margins
    if 'profitMargins' in dict_info and dict_info['profitMargins'] > 0.10:
        overall_score += 1
        profitMargin_result = 'Good'
    
    # 7. Check Revenue Growth
    if 'revenueGrowth' in dict_info and dict_info['revenueGrowth'] > 0.05:
        overall_score += 1
        revenueGrowth_result = 'Good'
    
    # 8. Check Earnings Growth
    if 'earningsGrowth' in dict_info and dict_info['earningsGrowth'] > 0.05:
        overall_score += 1
        earningGrowth_result = 'Good'
    
    # 9. Check Operating Margins
    if 'operatingMargins' in dict_info and dict_info['operatingMargins'] > 0.15:
        overall_score += 1
        operatingMargins_result = 'Good'
    
    # 10. Check Current Ratio
    if 'currentRatio' in dict_info and dict_info['currentRatio'] > 1:
        overall_score += 1
        current_ratio_result = 'Good'
    
    # Calculate overall score as the ratio of good traits to total traits
    final_score = overall_score

    score = final_score
    st.write(f"Overall Score for {symbol}: {score:.2f}")


from fpdf import FPDF

# Content to save
content = f"""
Machine Learning Algorithms's Recommendation: {ML_result}
Simple Moving Average Result: {SMA_result}
Sentiment Analysis Result: {sentiment_result}
Exponential Moving Average Result: {EMA_result}
RSI Result: {RSI_result}
MACD Result: {MACD_result}
Dividend Result: {dividend_result}
Dividend Yield Result: {dividend_result}    
Trailing PE Result: {trailingPE_result}
BETA Result: {Beta_result}
Return On Equity Result: {returnOnEquity_result}
Debt To Equity Result: {debtToEquity_result}
Profit Margin Result: {profitMargin_result}
Revenue Growth Result: {revenueGrowth_result}
Earning Growth Result: {earningGrowth_result}
Operating Margin Result: {operatingMargins_result}
Current Ratio Result: {current_ratio_result}
"""

# Save to a text file
with open("stock_prediction.txt", "w") as text_file:
    text_file.write(content)

st.write("Files created successfully!")
