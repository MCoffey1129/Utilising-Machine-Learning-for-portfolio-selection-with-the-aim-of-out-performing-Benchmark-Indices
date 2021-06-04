
# Section 1 - Import packages required for the assignment

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests
from bs4 import BeautifulSoup as bs
import re




# Output all columns
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)

# Section 2.1 - Get the tickers for all stocks that trade on the NYSE & the NASDAQ and import them into Pycharm

# The list of stocks traded on either the NYSE or the NASDAQ was sourced from the NASDAQ website
# (https://www.nasdaq.com/market-activity/stocks/screener) on the 05/06/2021

stocks = pd.read_csv(r'Files\NYSE_NASDAQ_stocks_20210604.csv')

# Section 2.2 - Exploratory data analysis

stocks.head()  # print out the first 5 rows
stocks.shape  # 7,377 stocks with 11 different features
stocks.describe()  # No issues with any of the data, the min IPO year is 1972 and the max IPO year is 2021
stocks.dtypes  # IPO year should be an integer
stocks.isnull().sum()
# There is a large number of Nulls for % Change (3), Market cap (476), Country (597),
# IPO year (3100), sector (1910) and industry (1911).
# Given that these unpopulated fields are populated in yahoo finance I will only use the ticker value going forward

# Stock name
stock_df = stocks[['Symbol','Name']]
stock_name_list = np.strip(list(stock_df.to_records(index=False)))
print(stock_name_list)


# Check to see how many of the 7,377 stocks are either notes or warrants
# We use regex to print out a list of tuples containing first the list position of where the Note or list
# is contained and the second is the whether the word is notes or warrants in that position
# There are 557 entries which are either Notes or Warrants in our lists which need to be removed
# (422 warrants and 135 Notes)

regex1 = r'[Ww]arrant|[Nn]otes|[Dd]ebenture'
regex2 = r'\W'
removal_list1 = []
removal_list2 = []
upd_stock_list = []

for i in range(len(stock_name_list)):
    reg1 = re.findall(regex1, stock_name_list[i][1].strip())
    reg2 = re.findall(regex2, stock_name_list[i][0].strip())
    if reg1 != []:
        removal_list1.append((i, reg1))
    elif reg2 != [] and reg2 != ['/']:
        removal_list2.append((i, reg2))
    else:
        upd_stock_list.append(stock_name_list[i][0].strip())

print(removal_list1)
print(removal_list2)

print(len(stock_name_list)) # 7377
print(len(removal_list1)) # 581 (422 warrants, 135 notes and 24 debentures)
print(len(removal_list2)) # 432 (preferred stock)
print(len(upd_stock_list)) # 6364 (7377 - 581 - 432)

######################################################################################################################
            # Section 2.3 - Get required info from Alpha Vantage
######################################################################################################################

# We want to try and find the stock information on the 6,364 stocks.
# We need to ensure that we do not bring in any forward looking metrics including market cap and the number of full
# time employees

API_key  = "OSPJN1YHMULW3OEO"

# stk_list = ['TSLA']
stk_list = ['TSLA','AMZN']

for stock in stk_list:
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'OVERVIEW',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    Temp_data = pd.DataFrame(output['Name'])

print(output['Name'])


AssetType
Name
Exchange
Currency
Country
Sector
Industry







EPS_data = pd.DataFrame()

for stock in stk_list:
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'Earnings',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    Temp_data = pd.DataFrame(output['quarterlyEarnings'])
    Temp_data['ticker'] = output['symbol']
    EPS_data = EPS_data.append(Temp_data, ignore_index=True)

print(EPS_data)

print(len(av_stock_list))
List1 = list(av_stock_list[0:10])
print(List1)

base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'Earnings',
          'symbol': 'AAIC',
          'apikey': API_key}

response = requests.get(base_url, params=params)

output = response.json()
Temp_data = pd.DataFrame(output['annualEarnings'])
Temp_data['ticker'] = output['symbol']
print(Temp_data)

output = response.json()
print(output)


base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'LISTING_STATUS',
          'date': '2019-12-31',
          'state': 'active',
          'apikey': API_key}

response = requests.get(base_url, params=params)

output = response.json()
print(output)