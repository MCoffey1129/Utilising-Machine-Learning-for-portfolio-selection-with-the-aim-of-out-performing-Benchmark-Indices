
# Section 1 - Import packages required for the assignment

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests
from bs4 import BeautifulSoup as bs


# Output all columns
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)

# Section 2.1 - Get the tickers for all stocks that trade on the NYSE

# The list of stocks traded on the NYSE was sourced from the NASDAQ website on the 31/05/2021

stocks = pd.read_csv(r'Files\NYSE_stocks_2020_05_31.csv')
stocks.head() # see what the first 5 stocks in the list look like
stocks.shape  # 3,127 stocks across 11 columns





# Ticker stuff.............

# Section 2.1 - Import data from Yahoo finance
Stock_list = ['AMZN', 'TSLA']
ticker_info = []
for i in Stock_list:
    a = yf.Ticker(i)
    ticker_info.append(a)

print(ticker_info)

SP_500 = yf.Ticker('^GSPC')
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



# Section 2.2 - Import CSV required for the Project