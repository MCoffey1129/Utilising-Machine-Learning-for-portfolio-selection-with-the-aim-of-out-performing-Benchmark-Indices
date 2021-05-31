
# Section 1 - Import packages required for the assignment

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests
from bs4 import BeautifulSoup as bs
from alpha_vantage.fundamentaldata import FundamentalData




# Output all columns
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)

# Section 2.1 - Get the tickers for all stocks that trade on the NYSE

# The list of stocks traded on the NYSE was sourced from the NASDAQ website on the 31/05/2021

stocks = pd.read_csv(r'Files\NYSE_stocks_2020_05_31.csv')
stocks.head() # see what the first 5 stocks in the list look like
stocks.shape  # 3,128 stocks with 11 different features
stocks.describe() # Min IPO year is 1986 which is correct and no cases have an IPO post 2021
stocks.dtypes  # No issues with any of the data types
stocks.isnull().sum()
# There is a large number of Nulls for Market cap (472), Country (589),
# IPO year (1520), sector (1184) and industry (1184).
# Given that these unpopulated fields are populated in yahoo finance I will only use the ticker value going forward


# Removing unwanted columns
stock_symbol = stocks['Symbol']
stock_symbol.head()
stock_symbol.shape  # 3,128 stocks with only the stock symbol column


# Section 1.2 - Get required info from Yahoo finance on the stocks

# Convert the pandas dataframe into a list which we then pass through yahoo finance
stock_list = list(stock_symbol)
print(type(stock_list))  # List
print(len(stock_list))  # Length is unchanged, all 3,128 stocks are in the list



ticker_info = []
for i in stock_list:
    yf_info = yf.Ticker(i)
    ticker_info.append(yf_info)

print(ticker_info)
print(len(ticker_info))

SP_500 = yf.Ticker('DHI')
SP_500 = SP_500.get_info()
print(SP_500)

Amazon = yf.Ticker('AMZN')
Amzn_info_df = Amazon.get_info()
print(Amzn_info_df)
print(Amazon.earnings)

Tesla = yf.Ticker('TSLA')
Tesla_info_df = Tesla.get_info()
print(Tesla_info_df)

Ptf = yf.Ticker(('TSLA','AMZN'))


amzn_price = pd.DataFrame(Amazon.history(period="max"))
print(amzn_price)
amzn_price.describe()

close = amzn_price['Close']
print(close)

plt.plot(close)
plt.plot(amzn_price['Open'])
plt.legend(["Close", "Open"])
plt.show()



API_key  = "N12W0SC4D3H7IMJ1"

base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'EARNINGS',
         'symbol': 'IBM',
         'apikey': API_key}

response = requests.get(base_url, params=params)

print(response.json())
test1 = pd.DataFrame(response.json())
print(test1)