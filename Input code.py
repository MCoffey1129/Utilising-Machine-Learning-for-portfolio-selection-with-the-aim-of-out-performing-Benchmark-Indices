###############################################################################################################
# Section 1 - Data Import
# Import the stock information used in the project
#        1.0 - Import packages required for the data import
#        1.1 - Import the Stock Symbols for each of the stocks which will be used in the project
#        1.2 - Complete some initial exploratory analysis on this data
#        1.3 - Structure the data in order to import the required Stock information from AlphaVantage
#        1.4 - Import the data from Alpha Vantage (Please note I did not create a function to import these
#              tables as there were small differences in the import process)
###############################################################################################################

###############################################################################################################
# Section 1.0 - Import packages required for the data import
###############################################################################################################

import pandas as pd
import requests
import re
import time

#
###############################################################################################################
# Section 1.1 - Import the Stock Symbols for each of the stocks which will be used in the project
#
#             - Get the stock symbols for all stocks that trade on the NYSE & the NASDAQ and import
#               them into Pycharm
###############################################################################################################

# The list of stocks traded on either the NYSE or the NASDAQ was sourced from the NASDAQ website
# (https://www.nasdaq.com/market-activity/stocks/screener) on the 05/06/2021.

# Import the CSV into Python
# Please update the below links to point to the location where you have saved these files
stocks = pd.read_csv(r'Files\NYSE_NASDAQ_stocks_20210604.csv')

###############################################################################################################
# Section 1.2 - Complete some initial exploratory analysis on this data
###############################################################################################################

stocks.head()  # print out the first 5 rows
stocks.shape  # 7,377 stocks with 11 different features
stocks.describe()  # No issues with the numeric data, the min IPO year is 1972 and the max IPO year is 2021

stocks.dtypes  # IPO year should be an integer, last sale and % change are objects, this would be an issue if we
# were planning on using these fields

stocks.isnull().sum()
# There is a large number of Nulls in the data, for % Change (3), Market cap (476), Country (597),
# IPO year (3100), sector (1910) and industry (1911).
# We will look to drop all fields except Symbol and Name and look to source fields such as Country, Sector, Industry and
# Market Cap from Alpha Vantage

###############################################################################################################
# Section 1.3 - Structure the data in order to import the required Stock information from AlphaVantage
###############################################################################################################

# Create a list of tuples containing stock symbols and stock names which will be used to pull information
# from Alpha Vantage

stock_df = stocks[['Symbol', 'Name']]
stock_name_list = (list(stock_df.to_records(index=False)))
print(stock_name_list)

# Of the 7,377  stocks which are reported on the NYSE and NASDAQ a number of these are either Warrants, Notes or
# Debentures which will have a different valuation basis than common stock and ordinary shares and may skew our
# model results so we will remove these from our data (these cases will either be referred to as Warrants, Notes or
# Debentures in the stock name)
# To avoid doubling up on certain stocks we will remove Preferred shares from our dataset (Preferred Shares have a
# Stock symbol which contains a '^').
# Please note we will still have preferred stocks in our dataset which we will remove later before modelling.


# We pass two regex expressions through the for loop, the first expression searches for the words Warrant, Notes or
# Debentures so as to remove these stocks from the updated list (please note it searches for the capitalised and
# non-capitalised versions of these words).
# The second regex expression searches the Stock Symbol for non-word characters, please note the stock symbol column
# contains only non-word characters of "^" and "/" and we do not want to remove stock symbols which contain "/"

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

print(len(stock_name_list))  # 7377 stocks - original list
print(len(removal_list1))  # 581 entries (422 warrants, 135 notes and 24 debentures)
print(len(removal_list2))  # 432 entries (preferred stock)
print(len(upd_stock_list))  # 6364 (7377 - 581 - 432), this is our updated list removing Warrants, Notes etc.

######################################################################################################################
# Section 1.4 - Import the data from Alpha Vantage

# When importing data from Alpha Vantage we need to ensure that we do not bring in any forward
# looking metrics i.e. the number of full time employees in 2021

# We will import six different tables from Alpha Vantage:
#       1.4.1 : Company Overview
#       1.4.2 : EPS (earnings per share data)
#       1.4.3 : Income Statement data
#       1.4.4 : Balance sheet information
#       1.4.5 : Cash flow statements
#       1.4.6 : Monthly adjusted stock prices

######################################################################################################################

# Section 1.4.1 -  Company Overview

# API key received from Alpha Vantage
# API key can be downloaded from the Alpha Vantage website
API_key = "XXXXXXXXXXXXX"

# List the columns we would like to keep from the Company overview import ("columns" list)
# Please note there is a sleep function in the for loop used to import the data to ensure we do not go above the
# max API call restriction

# The imported data is in a json file format. We convert the json output into a list and in-turn into a dataframe
# and append it on to the overview dataframe.
# If there is no information for a particular stock nothing gets appended to the overview dataframe

Overview_df = pd.DataFrame()
columns = ['Symbol', 'AssetType', 'Name', 'Exchange', 'Currency', 'Country', 'Sector', 'Industry']

for stock in upd_stock_list:
    time.sleep(0.75)
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'OVERVIEW',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    if output == {} or list(output.keys())[0] == 'Error Message':
        Temp_data_df = pd.DataFrame()
    else:
        Temp_data_df = pd.DataFrame(list(output.values())).transpose()
        Temp_data_df.columns = [list(output.keys())]
        Temp_data_df = Temp_data_df[columns]

    Overview_df = Overview_df.append(Temp_data_df, ignore_index=True)

print(Overview_df)  # Company Overview information available for 5,919 of the 6,364 stocks (93%)

# Write out the table to a CSV to a location on the C drive.
# Overview_df.to_csv(r'Files\Overview_df.csv', index=False, header=True)

# Section 1.4.2 - EPS

# The imported data is in a json file format. We convert the json output into a list and in-turn into a dataframe
# and append it on to the eps dataframe.
# If there is no information for a particular stock nothing gets appended to the eps dataframe
# We will only take in information from Dec '14 onwards as we only have reliable Income Statement information
# from Jun '16 onward (see 'AAPL')

Temp_data = pd.DataFrame()
eps_data = pd.DataFrame()

for stock in upd_stock_list:
    time.sleep(0.75)
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'EARNINGS',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    if output == {} or list(output.keys())[0] == 'Error Message':
        Temp_data = pd.DataFrame()
    else:
        Temp_data = pd.DataFrame(output['quarterlyEarnings'])
        Temp_data['Symbol'] = output['symbol']

    eps_data = eps_data.append(Temp_data, ignore_index=True)
    eps_data = eps_data.loc[eps_data['fiscalDateEnding'] > '2014-11-30']

print(eps_data)

# Write out the CSV
# eps_data.to_csv(r'Files\eps_data.csv', index=False, header=True)

# Section 1.4.3 - Income Statement

# The imported data is in a json file format. We convert the json output into a list and in-turn into a dataframe
# and append it on to the Income Statement dataframe.
# If there is no information for a particular stock nothing gets appended to the Income Statement dataframe


Temp_data = pd.DataFrame()
inc_st_data = pd.DataFrame()

for stock in upd_stock_list:
    time.sleep(0.75)
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'INCOME_STATEMENT',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    if output == {} or list(output.keys())[0] == 'Error Message':
        Temp_data = pd.DataFrame()
    else:
        Temp_data = pd.DataFrame(output['quarterlyReports'])
        Temp_data['Symbol'] = output['symbol']

    inc_st_data = inc_st_data.append(Temp_data, ignore_index=True)

print(inc_st_data)

# Write out the CSV
# inc_st_data.to_csv(r'Files\inc_st_data.csv', index=False, header=True)

# Section 1.4.4 - Balance Sheet

# The imported data is in a json file format. We convert the json output into a list and in-turn into a dataframe
# and append it on to the Balance Sheet dataframe.
# If there is no information for a particular stock nothing gets appended to the Balance Sheet dataframe

Temp_data = pd.DataFrame()
BS_data = pd.DataFrame()

for stock in upd_stock_list:
    time.sleep(0.75)
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'BALANCE_SHEET',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    if output == {} or list(output.keys())[0] == 'Error Message':
        Temp_data = pd.DataFrame()
    else:
        Temp_data = pd.DataFrame(output['quarterlyReports'])
        Temp_data['Symbol'] = output['symbol']

    BS_data = BS_data.append(Temp_data, ignore_index=True)

print(BS_data)

# Write out the CSV
# BS_data.to_csv(r'Files\BS_data.csv', index=False, header=True)

# Section 1.4.5 - Cash_flow

# The imported data is in a json file format. We convert the json output into a list and in-turn into a dataframe
# and append it on to the Cash Flow statement dataframe.
# If there is no information for a particular stock nothing gets appended to the Cash Flow statement dataframe

Temp_data = pd.DataFrame()
CF_data = pd.DataFrame()

for stock in upd_stock_list:
    time.sleep(0.75)
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'CASH_FLOW',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    if output == {} or list(output.keys())[0] == 'Error Message':
        Temp_data = pd.DataFrame()
    else:
        Temp_data = pd.DataFrame(output['quarterlyReports'])
        Temp_data['Symbol'] = output['symbol']

    CF_data = CF_data.append(Temp_data, ignore_index=True)

print(CF_data)

# Write out the CSV
# CF_data.to_csv(r'Files\CF_data.csv', index=False, header=True)

# Section 1.4.6 - Monthly stock prices

# The imported data is in a json file format. We convert the json output into a list and in-turn into a dataframe
# and append it on to the Monthly prices dataframe.


upd_stock_list1 = upd_stock_list[5780:]
print(upd_stock_list1)

Temp_data = pd.DataFrame()
monthly_prices = pd.DataFrame()

for stock in upd_stock_list1:
    time.sleep(0.75)
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
              'outputsize': 'full',
              'symbol': stock,
              'apikey': API_key}

    response = requests.get(base_url, params=params)

    output = response.json()
    if output == {} or list(output.keys())[0] == 'Error Message':
        Temp_data = pd.DataFrame()
    else:
        Temp_data = pd.DataFrame(output['Monthly Adjusted Time Series']).transpose().rename_axis('dt').reset_index()
        Temp_data['Symbol'] = stock

    monthly_prices = monthly_prices.append(Temp_data, ignore_index=True)
    monthly_prices = monthly_prices.loc[monthly_prices['dt'] > '2014-11-30']

print(monthly_prices)

# Write out the CSV
# monthly_prices.to_csv(r'Files\monthly_prices.csv', index=False, header=True)
