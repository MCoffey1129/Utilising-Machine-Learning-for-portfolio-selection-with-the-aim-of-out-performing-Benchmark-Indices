###############################################################################################################
# Section 2 - Exploratory data analysis
###############################################################################################################


###############################################################################################################
# Section 2.0 - Import the required packages and functions
###############################################################################################################

# Packages
import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import copy
import calendar


# Functions

# timer decorator - to check the length of time functions have taken to run

def timer(func):
    """A decorator that prints how long a function took to run."""

    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result.
        result = func(*args, **kwargs)
        # Get the total time it took to run, and print it.
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result

    return wrapper


# Function for returning a pandas dataframe containing unique symbols for the input table
@timer
def unique_column(input_table, column):
    """A function for returning a pandas dataframe containing unique column for the input table"""
    output = pd.DataFrame(input_table[column].unique(), columns=[column])
    return output


# Create a
@timer
def pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320):
    """A function used to display the requested number of rows and columns when printing the data"""

    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)

    pd.set_option('display.width', display_width)


@timer
def seaborn_lm_plt(input_table, close_val, future_value):
    seaborn_tab = input_table.loc[
        (input_table['close_price'] < close_val) & (input_table['future_price'] < future_value)]
    sns.lmplot(data=seaborn_tab, x='close_price', y='future_price', hue='month', palette='deep', legend=False,
               scatter_kws={"s": 10, "alpha": 0.2})
    plt.xlabel('Close Price', size=12)
    plt.ylabel('Future Price', size=12)
    plt.legend(loc='upper left')
    plt.title("Close Price v Future Price", fontdict={'size': 16})
    plt.tight_layout()
    plt.show()


# Handling null values
def null_value_pc(table):
    missing_tbl = pd.DataFrame(table.isnull().sum(), columns = ['num missing'])
    missing_tbl['missing_pc'] = missing_tbl['num missing'] / mdl_data.shape[0]
    print(missing_tbl)


#    plt.title("Close Price (< $" + str(close_val) + ")" + " v Future Price (< $" + str(future_value) + ")"
#        , fontdict={'size': 16})

# Function which converts a string value of "none" to missing
# @timer
# def string_conv(input_table):
#     """Function which converts a string value of "none" to missing"""
#     return input_table = inpu

###############################################################################################################
# Section 2.1 - Exploratory data analysis
###############################################################################################################


# There are a further 445 stocks for which AlphaVantage does not have Company information on (please note
# 390 of these cases are ETF (exchange traded funds which we want to remove from our model)

# Original list of stocks in NYSE or NASDAQ                                                 =  7,377
# Minus Warrants, Notes, Debentures and Preferred Shares                                     - 1,013
# Minus ETFs which do not contain company info and should not be included in our model       -   390
# Minus 55 stocks for which most (e.g. WSO/B and HVT/A are preferred stocks)                 -    55
# Equals the total number of stocks in our Company overview file (overview_df)              =  5,919

# Only use stocks for which we have information on earnings, cashflow etc. (see below)       -   837
# Final number of stocks used in our model                                                  =  5,082

pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320)

# Import the CSVs into Python

company_overview = pd.read_csv(r'Files\Overview_df.csv')
eps_data = pd.read_csv(r'Files\eps_data.csv')
inc_st_data = pd.read_csv(r'Files\inc_st_data.csv')
bs_data = pd.read_csv(r'Files\BS_data.csv')
cf_data = pd.read_csv(r'Files\CF_data.csv')
monthly_stock_prices = pd.read_csv(r'Files\monthly_prices.csv')

# Pull in the unique stock symbols for each table (time taken for each call of the function <0.0s)
co_symbol_unique = unique_column(company_overview, 'Symbol')
eps_symbol_unique = unique_column(eps_data, 'Symbol')
inc_st_symbol_unique = unique_column(inc_st_data, 'Symbol')
bs_symbol_unique = unique_column(bs_data, 'Symbol')
cf_symbol_unique = unique_column(cf_data, 'Symbol')
sp_symbol_unique = unique_column(monthly_stock_prices, 'Symbol')

# We only want to keep stocks which are contained in each file, in order to achieve this we complete run an
# inner join on each of the datasets containing the unique stock symbols
# There are 5,082 stocks which are in every file, this is what we will use going forward
symbols_in_all_files = \
    pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(co_symbol_unique, eps_symbol_unique, how='inner', on='Symbol')
                                        , inc_st_symbol_unique, how='inner', on='Symbol')
                               , bs_symbol_unique, how='inner', on='Symbol')
                      , cf_symbol_unique, how='inner', on='Symbol')
             , sp_symbol_unique, how='inner', on='Symbol')

symbols_in_all_files.shape  # Check complete that there are 5,082 stocks in this table

# Update the company overview such that it will contain only the stocks which are contained in each file

company_overview_upd = pd.merge(company_overview, symbols_in_all_files, how='inner', on='Symbol')
company_overview_upd.shape  # updated file contains 5,082 stocks as expected

# Update our initial dataframe such that it is in the correct form required for modelling.
# As the approach is a 6 month hold and sell strategy we want to get the stock information off quarter
# so that we do not have any issues aro....


dates = ['2021-01', '2020-07', '2020-01', '2019-07', '2019-01', '2018-07', '2018-01', '2017-07']
dates_df = pd.DataFrame(dates, columns=['dt'])
dates_df.index = pd.to_datetime(dates_df['dt']).dt.to_period('M')
dates_df = dates_df.drop(columns=dates_df.columns[0])  # drop the second dt field
print(dates_df)
dates_df.columns

company_overview_dt = pd.DataFrame()

for i in dates:
    company_overview_upd['dt'] = i
    company_overview_dt = company_overview_dt.append(company_overview_upd, ignore_index=True)

company_overview_dt.head()
company_overview_dt.tail()
company_overview_dt.shape  # 40,656 rows (5,082 * 8 timeframes), the for loop was run correctly
company_overview_dt.info()  # dt is an object, we want this to be a datetime and we want to set it as our index.

# Change the dt field to a datetime object and set it as the index
company_overview_dt.index = pd.to_datetime(company_overview_dt['dt']).dt.to_period('M')
company_overview_dt = company_overview_dt.drop(columns=company_overview_dt.columns[8])  # drop the second dt field
company_overview_dt.head()
company_overview_dt.tail()

##################################################################################################################
# Section 2.2 - Financial Results data including earnings, income statement, balance sheet and cash flow statement
##################################################################################################################


financial_results = \
    pd.merge(pd.merge(pd.merge(eps_data, inc_st_data, how='inner', on=['fiscalDateEnding', 'Symbol'])
                      , bs_data.drop(labels=['reportedCurrency'], axis=1), how='inner',
                      on=['fiscalDateEnding', 'Symbol'])
             , cf_data.drop(labels=['netIncome', 'reportedCurrency'], axis=1), how='inner',
             on=['fiscalDateEnding', 'Symbol'])

financial_results.head(40)

# Initially checked the data for duplicate columns (these are flagged as _x and _y after which I re-run the
# join removing these columns to avoid the duplication.

eps_data.shape  # 7 columns
inc_st_data.shape  # 27 columns
bs_data.shape  # 39 columns
cf_data.shape  # 30 columns
financial_results.shape  # 94 columns ( 7 + 25 (27-2 columns joining on) + 36 (39-2-1(rep currency)) + 26 (30-2-2) )
# 78,126 rows

columns_df = pd.DataFrame(financial_results.columns, columns=['Columns'])
columns_df = columns_df.sort_values(by='Columns')
print(columns_df)  # There are no longer any duplicate columns

# Check
financial_results.info()  # Every field is saved as a character field
financial_results.describe()  # No numeric fields so the output is not useful
financial_results.isnull().sum()  # There are a large number of nulls

# Replace 'None' with NaN in order to convert the character fields to Numeric,
# we will have to take another look at reportedCurrency later but for now we will convert it
# to numeric
financial_results = financial_results.replace('None', np.nan)
symb_curr_df = pd.DataFrame(financial_results[['Symbol', 'reportedCurrency']])

financial_results_reorder = pd.concat(
    [symb_curr_df, financial_results.drop(labels=['Symbol', 'reportedCurrency'], axis=1)]
    , axis=1)

financial_results_reorder.info()  # reordering looks to be correct
financial_results_reorder.shape  # No change in the number of rows (78,126) and columns 94
# Convert the numeric fields to floats

financial_results_reorder.iloc[:, 4:] = \
    financial_results_reorder.iloc[:, 4:].astype(float)

financial_results_reorder.info()  # Each of the numeric fields have been converted to floats
financial_results_reorder.shape  # No change in the number of rows (78,126) and columns 94

# Convert the Date fields to dates
financial_results_reorder['fiscalDateEnding'] = pd.to_datetime(financial_results_reorder['fiscalDateEnding']) \
    .dt.to_period('M')

financial_results_reorder['reportedDate'] = pd.to_datetime(financial_results_reorder['reportedDate']).dt.to_period('M')

# We need to mould the data we will be looking to take in the most recent Financial information for July and Jan
# each year. The most recent data at Jan will be the quarterly results published in Jan of the current year
# or Dec or Nov of the previous year. Below we update the report date to accomplish the above.

# If the report date is in Dec or Nov we update the reported date to be Jan of the following year

financial_results_reorder.loc[((financial_results_reorder['reportedDate'].dt.month == 12)
                              | (financial_results_reorder['reportedDate'].dt.month == 11)
                              | (financial_results_reorder['reportedDate'].dt.month == 10))
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 1)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 12)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 11)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 10), 'dt_yr'] \
    = financial_results_reorder['reportedDate'].dt.year + 1

financial_results_reorder.loc[((financial_results_reorder['reportedDate'].dt.month == 12)
                              | (financial_results_reorder['reportedDate'].dt.month == 11)
                              | (financial_results_reorder['reportedDate'].dt.month == 10))
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 1)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 12)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 11)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 10),
                              'dt_month'] = 1

# If the report date is in June or May we update the reported date to be July

financial_results_reorder.loc[((financial_results_reorder['reportedDate'].dt.month == 6)
                              | (financial_results_reorder['reportedDate'].dt.month == 5)
                              | (financial_results_reorder['reportedDate'].dt.month == 4))
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 7)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 6)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 5)
                              & (financial_results_reorder['reportedDate'].shift(1).dt.month != 4),
                              'dt_month'] = 7

financial_results_reorder['dt_yr'].fillna(financial_results_reorder['reportedDate'].dt.year, inplace=True)
financial_results_reorder['dt_month'].fillna(financial_results_reorder['reportedDate'].dt.month, inplace=True)


# Combine the year and month column which will be converted to an updated report date field
financial_results_reorder['dt_str'] = financial_results_reorder['dt_yr'].astype(int).map(str) + "-" + \
                                      financial_results_reorder['dt_month'].astype(int).map(str)

financial_results_reorder.head(40)


# Create a date field called 'dt' and assign it as the index
financial_results_reorder['dt'] = pd.to_datetime(financial_results_reorder['dt_str']).dt.to_period('M')
financial_results_reorder.index = financial_results_reorder['dt']

financial_results_reorder.head(40)

financial_results_reorder.shape  # No change in the number of rows (78,126) but there are 4 extra columns (98)
financial_results_reorder.columns  # 4 columns which are not required, these are the last 4 columns

financial_results_reorder = financial_results_reorder.drop(columns=financial_results_reorder.columns[[-1, -2, -3, -4]])
financial_results_reorder.head(5)
financial_results_reorder.shape  # the 4 columns have been dropped (94) and no change to the number of rows (78,126)
financial_results_reorder.columns

# Get the lagged data for each of the numeric columns
# When looking at the 6 month forecasted growth/decline of a share price we do not want to look at just the
# revenue in the current quarter but rather the revenue growth across the year for that share

# columns are the list of all numeric fields we want to get the 4 lagged values for

columns = list(financial_results_reorder.iloc[:, 4:].columns)

for i in columns:
    for j in range(1, 5):
        # Get the 4 quarter lags on to the same row i.e. the column totalRevenue_2Q_lag is the totalRevenue
        # from 6 months previous
        financial_results_reorder[i + '_' + str(j) + 'Q_lag'] = financial_results_reorder[i].shift(-j)
        # The below code ensures that we are not taking in financial data from an incorrect symbol
        financial_results_reorder.loc[financial_results_reorder['Symbol'].shift(-j) !=
                                      financial_results_reorder['Symbol'], i + '_' + str(j) + 'Q_lag'] = np.nan

financial_results_reorder.head(30)
print(financial_results_reorder.loc[financial_results_reorder['Symbol'] == 'ZNTL'])  # Calculation looks correct
financial_results_reorder.shape  # Number of rows are unchanged at 78,126 but we now have 454 columns
# The number of columns equals 90 (the number of numeric columns) * 5 (4 lagged periods) + 4 (character fields) = 454


# We only have 8 timeframes which we are modelling on so we can delete all other time points
# These timepooints are saved in the dates_df
financial_results_reorder = pd.merge(financial_results_reorder, dates_df, left_index=True, right_index=True)
financial_results_reorder.head(20)
financial_results_reorder.tail(20)

financial_results_reorder.index.unique()  # 8 unique timepoints as expected
financial_results_reorder.shape  # the number of columns are unchanged at 454 as expected
# the number of rows have decreased to 31,399


##################################################################################################################
# Section 2.3 - Bring in the monthly stock prices
##################################################################################################################

monthly_stock_prices.head(20)

monthly_stock_prices['dt_m'] = pd.to_datetime(monthly_stock_prices['dt'], format="%d/%m/%Y").dt.to_period('M')

monthly_stock_prices['close_price'] = monthly_stock_prices["5. adjusted close"].astype(float)
stock_prices = monthly_stock_prices[['dt', 'dt_m', 'Symbol', 'close_price']]
stk_prices = copy.deepcopy(stock_prices)

print(stk_prices)

# Create a month column as we will need to decipher whether investing for 6 months in Jan or at July is more
# profitable
stk_prices['month'] = stk_prices['dt_m'].dt.month
stk_prices['month'] = stk_prices['month'].apply(lambda x: calendar.month_abbr[x])
print(stk_prices)

# columns = ['close_price']

for j in [1, 3, 6, 12, -5, -6]:
    # Get the historic and future stock prices
    stk_prices['close_price' + '_' + str(j) + 'M_lag'] = stk_prices['close_price'].shift(-j)
    # The below code ensures that we are not taking in stk_prices from an incorrect symbol
    stk_prices.loc[stk_prices['Symbol'].shift(-j) !=
                   stk_prices['Symbol'], 'close_price' + '_' + str(j) + 'M_lag'] = np.nan

print(stk_prices)
stk_prices.info()

# Checks
stk_prices[['dt', 'dt_m', 'Symbol', 'close_price', 'close_price_-5M_lag', 'close_price_-6M_lag']].tail(100)
stk_prices.columns
stk_prices.describe()  # There are no negative stock prices but the max stock price looks very high

chk_tbl = stk_prices.sort_values(by=['close_price'], ascending=False)
chk_tbl.head(10)  # I will take the 10 largest values and check the values on yahoo finance
# The values look correct as per Yahoo finance


# Create the forecasted stock price - please note we have only 5 months of forecasted stock information
# for trades made in Jan '21
stk_prices.loc[stk_prices['dt_m'] == '2021-01', 'future_price'] = stk_prices['close_price_-5M_lag']
stk_prices.loc[stk_prices['dt_m'] != '2021-01', 'future_price'] = stk_prices['close_price_-6M_lag']
stk_prices.tail(100)

# Make dt_m the index
stk_prices.drop("dt", inplace=True, axis=1)
stk_prices.index = stk_prices['dt_m'].rename("dt")

# Drop unneeded columns
stk_prices = stk_prices.drop(['dt_m', 'close_price_-5M_lag', 'close_price_-6M_lag'], axis=1)
stk_prices.head(50)
stk_prices.columns

# Keep only the dates which are required
stk_prices = pd.merge(stk_prices, dates_df, left_index=True, right_index=True)
print(stk_prices)
stk_prices.index.unique()  # 8 unique timepoints as expected
stk_prices.head(20)

# Plot the close price against the forecasted price (6 months later)
# Looking at the plots possibly investing in stocks at July for a 6 month period is more
# profitable than investing in Jan. Obviously we will have to fit all of the data to see if the date of investing
# is statistically significant

# function(table,close_price to be lower than, future price to be lower than)
seaborn_lm_plt(stk_prices, 10, 50)  # cases which have a price of less than $10 and a future price less than $50
seaborn_lm_plt(stk_prices, 100, 500)
seaborn_lm_plt(stk_prices, 5, 20)
seaborn_lm_plt(stk_prices, 5, 1000000)  # Outlier is Gamestop share increase from July '20 to Jan '21

# Code for checking the stocks with the largest 6 month gains who had a share price of less than 5 euro
# GameStop's (GME) share price increased from €4.01 in July 2020 to €320.99 in Jan '21
# This is an outlier as the share increase was not a result of the fundamentals of the company.
a = stk_prices.loc[stk_prices['close_price'] < 5]
b = copy.deepcopy(a)
b['diff'] = b['future_price'] - b['close_price']
print(b.sort_values(by=['diff'], ascending=False))

##################################################################################################################
# Section 3.1 - Feature....
##################################################################################################################

# Join the three tables together by index
#      - company_overview_dt (table containing the name, country, Sector and Industry of the companies)
#      - financial_results_reorder (table containing the financial results (income statement, bs etc.)
#      - stk_prices (table containing the monthly stock prices of each company)


mdl_data = \
    pd.merge(pd.merge(company_overview_dt, financial_results_reorder, how='left', on=['dt', 'Symbol'])
             , stk_prices, how='left', on=['dt', 'Symbol'])

company_overview_dt.shape  # 40,656 rows and 8 columns
financial_results_reorder.shape  # 31,399 rows and 454 columns
stk_prices.shape  # 38,078 rows and 8 columns
mdl_data.shape  # 40,656 rows and 468 columns (454 + 8 + 8 - 2 (Symbol which we are joining on))


# Handling null values
null_value_pc(mdl_data) # there are a large number of Null values to deal with in all but 6 columns

# Sector
mdl_data['Sector'].unique()  # nan and 'None' in the column
mdl_data['Sector'].replace(to_replace = [np.nan,'None'], value = ['Unknown','Unknown'] , inplace=True)
mdl_data['Sector'].unique()  # no nan or 'None' values in the column
mdl_data['Sector'].isnull().sum() # 0 value returned

# Industry
mdl_data['Industry'].isnull().sum() # 120 missing values
mdl_data['Industry'].unique()  # 'None' values in the column
mdl_data['Industry'].replace(to_replace = [np.nan,'None'], value = ['Unknown','Unknown'] , inplace=True)
mdl_data['Industry'].unique()  # no 'None' values in the column
mdl_data['Industry'].isnull().sum() # 0 value returned


null_value_pc(mdl_data) # Sector and industry no longer have missing values


# The most important null field which needs to be investigated is the target variable which is the 'future_price' field
# After investigating a number of the fields which are missing I have concluded that the stock prices wer not there
# for that timepoint and so the rows should be deleted.
mdl_data.loc[mdl_data['future_price'].isnull()]
mdl_data.loc[mdl_data['Symbol'] == 'AAC', ['Symbol', 'close_price', 'future_price']]

mdl_data['future_price'].isnull().sum()  #  We are looking to drop 6,266 rows
mdl_data.shape  # currently 40,656 rows and 468 columns
mdl_data.dropna(how='all', subset=['future_price'], inplace=True)
mdl_data.shape # updated dataset has 34,390 rows (40,656 - 6,266) and 468 columns


null_value_pc(mdl_data)
# There are 3,823 cases that have a missing 'fiscalDateEnding' and 'reportedDate.' These rows correspond to rows with
# missing EPS and revenue information and so should be dropped from the model

mdl_data.dropna(how='all', subset=['fiscalDateEnding'], inplace=True)
mdl_data.shape # updated dataset has 30,567 rows (34,390 - 3,823) and 468 columns

null_value_pc(mdl_data)