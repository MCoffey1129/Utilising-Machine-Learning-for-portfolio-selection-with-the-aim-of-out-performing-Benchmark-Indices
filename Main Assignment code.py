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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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
@timer
def null_value_pc(table):
    missing_tbl = pd.DataFrame(table.isnull().sum(), columns=['num missing'])
    missing_tbl['missing_pc'] = missing_tbl['num missing'] / mdl_data.shape[0]
    print(missing_tbl)


# Function used for margin calculations
@timer
def margin_calcs(input_num, input_den, output_col):
    for j in [0, 1, 2, 4]:
        if j == 0:
            mdl_input_data[output_col] = mdl_input_data[input_num] / mdl_input_data[input_den]

        else:
            mdl_input_data[output_col + '_' + str(j) + 'Q_lag'] = mdl_input_data[input_num + '_' + str(j) + 'Q_lag'] \
                                                                  / mdl_input_data[input_den + '_' + str(j) + 'Q_lag']

            mdl_input_data[output_col + '_' + str(j) + 'Q_gth'] = (mdl_input_data[output_col]
                                                                       - mdl_input_data[
                                                                           output_col + '_' + str(j) + 'Q_lag'])


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

# Create a new column called Book_value which is equal to Assets less liabilities
financial_results_reorder['book_value'] = financial_results_reorder['totalAssets'] \
                                          - financial_results_reorder['totalLiabilities']

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
    for j in [1, 2, 4]:
        # Get the 4 quarter lags on to the same row i.e. the column totalRevenue_2Q_lag is the totalRevenue
        # from 6 months previous
        financial_results_reorder[i + '_' + str(j) + 'Q_lag'] = financial_results_reorder[i].shift(-j)
        financial_results_reorder[i + '_' + str(j) + 'Q_gth'] = (financial_results_reorder[i] -
                                                                     financial_results_reorder[i].shift(-j)) / \
                                                                abs(financial_results_reorder[i].shift(-j))
        # The below code ensures that we are not taking in financial data from an incorrect symbol
        financial_results_reorder.loc[financial_results_reorder['Symbol'].shift(-j) !=
                                      financial_results_reorder['Symbol'], i + '_' + str(j) + 'Q_lag'] = np.nan
        financial_results_reorder.loc[financial_results_reorder['Symbol'].shift(-j) !=
                                      financial_results_reorder['Symbol'], i + '_' + str(j) + 'Q_gth'] = np.nan


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
    # Get the historic and future stock prices growths
    if j >= 0:
        stk_prices['close_price' + '_' + str(j) + 'M_lag'] = stk_prices['close_price'].shift(-j)
        stk_prices['close_price' + '_' + str(j) + 'M_gth'] = \
            (stk_prices['close_price'] - stk_prices['close_price'].shift(-j)) / stk_prices['close_price'].shift(-j)
    else:
        stk_prices['close_price' + '_' + str(j) + 'M_lag'] = stk_prices['close_price'].shift(-j)
        stk_prices['close_price' + '_' + str(j) + 'M_gth'] = \
            (stk_prices['close_price'].shift(-j) - stk_prices['close_price']) / stk_prices['close_price']
    # The below code ensures that we are not taking in stk_prices from an incorrect symbol
    stk_prices.loc[stk_prices['Symbol'].shift(-j) !=
                   stk_prices['Symbol'], 'close_price' + '_' + str(j) + 'M_gth'] = np.nan
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
stk_prices.loc[stk_prices['dt_m'] == '2021-01', 'future_price_gth'] = stk_prices['close_price_-5M_gth']
stk_prices.loc[stk_prices['dt_m'] != '2021-01', 'future_price_gth'] = stk_prices['close_price_-6M_gth']
stk_prices.tail(100)

# Make dt_m the index
stk_prices.drop("dt", inplace=True, axis=1)
stk_prices.index = stk_prices['dt_m'].rename("dt")
stk_prices.head(25)
stk_prices.columns

# Drop unneeded columns
stk_prices = stk_prices.drop(['dt_m', 'close_price_-5M_gth', 'close_price_-6M_gth',
                              'close_price_-5M_lag', 'close_price_-6M_lag'], axis=1)
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

# Code for checking the stocks with the largest 6 month gains on companies who had a share price of less than 5 euro
# GameStop's (GME) share price increased from €4.01 in July 2020 to €320.99 in Jan '21
# This is an outlier as the share increase was not a result of the fundamentals of the company.
a = stk_prices.loc[stk_prices['close_price'] < 5]
b = copy.deepcopy(a)
b['diff'] = b['future_price'] - b['close_price']
print(b.sort_values(by=['diff'], ascending=False))

##################################################################################################################
# Section 3.1 - Replacing nulls
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
stk_prices.shape  # 38,078 rows and 13 columns
mdl_data.shape  # 40,656 rows and 473 columns (454 + 8 + 13 - 2 (Symbol which we are joining on))

# Handling null values
null_value_pc(mdl_data)  # there are a large number of Null values to deal with in all but 6 columns

# Sector
mdl_data['Sector'].unique()  # nan and 'None' in the column
mdl_data['Sector'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data['Sector'].unique()  # no nan or 'None' values in the column
mdl_data['Sector'].isnull().sum()  # 0 value returned

# Industry
mdl_data['Industry'].isnull().sum()  # 120 missing values
mdl_data['Industry'].unique()  # 'None' values in the column
mdl_data['Industry'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data['Industry'].unique()  # no 'None' values in the column
mdl_data['Industry'].isnull().sum()  # 0 value returned

null_value_pc(mdl_data)  # Sector and industry no longer have missing values

# We need to drop the 'future_price' from our model to prevent data leakage
mdl_data.drop('future_price', axis=1, inplace=True)
mdl_data.shape  # 40,656 rows and 468 columns (469 - 1)

# The most important null field which needs to be investigated is the target variable which is the 'future_price_gth'
# field
# After investigating a number of the fields which are missing I have concluded that the stock prices were not there
# for that timepoint and so the rows should be deleted.
mdl_data.loc[mdl_data['future_price_gth'].isnull()]
mdl_data.loc[mdl_data['Symbol'] == 'AAC', ['Symbol', 'close_price', 'future_price_gth']]

mdl_data['future_price_gth'].isnull().sum()  # We are looking to drop 6,266 rows
mdl_data.shape  # currently 40,656 rows and 468 columns
mdl_data.dropna(how='all', subset=['future_price_gth'], inplace=True)
mdl_data.shape  # updated dataset has 34,390 rows (40,656 - 6,266) and 468 columns

null_value_pc(mdl_data)
# There are 3,823 cases that have a missing 'fiscalDateEnding' and 'reportedDate.' These rows correspond to rows with
# missing EPS and revenue information and so should be dropped from the model

mdl_data.dropna(how='all', subset=['fiscalDateEnding'], inplace=True)
mdl_data.shape  # updated dataset has 30,567 rows (34,390 - 3,823) and 468 columns

null_value_pc(mdl_data)  # drop the reportedCurrency column as we already have a Currency column
mdl_data.drop('reportedCurrency', axis=1, inplace=True)
mdl_data.shape  # updated dataset has 30,567 rows  and 467 columns (468 -1)

null_value_pc(mdl_data)
# replace the estimated EPS with the reported EPS when missing, surprise and surprise Percentage should then be zero

mdl_data['estimatedEPS'].fillna(mdl_data['reportedEPS'], inplace=True)
mdl_data['estimatedEPS'].isnull().sum()  # no missing values

# Replace all other missing values with 0
# Revenue, gross profit and other values which were null had very few nulls......
mdl_data.fillna(0, inplace=True)
mdl_data.isnull().sum()  # no missing values

mdl_data.dtypes
mdl_data.drop(['fiscalDateEnding', 'reportedDate'], axis=1, inplace=True)
mdl_data.shape  # updated dataset has 30,567 rows  and 465 columns (467 -2)

##################################################################################################################
# Section 3.2 - Feature selection
##################################################################################################################

mdl_input_data = mdl_data

# Create a market cap column
mdl_input_data['market_cap'] = mdl_input_data['commonStockSharesOutstanding'] * mdl_input_data['close_price']
mdl_input_data['market_cap_1Q_lag'] = mdl_input_data['commonStockSharesOutstanding_1Q_lag'] \
                                      * mdl_input_data['close_price_3M_lag']
mdl_input_data['market_cap_2Q_lag'] = mdl_input_data['commonStockSharesOutstanding_2Q_lag'] \
                                      * mdl_input_data['close_price_6M_lag']
mdl_input_data['market_cap_4Q_lag'] = mdl_input_data['commonStockSharesOutstanding_4Q_lag'] \
                                      * mdl_input_data['close_price_12M_lag']

# Create new features required for modelling i.e. P/E, gross margin, net margin etc.

# Profitability metrics
margin_calcs('grossProfit', 'totalRevenue', 'gross_margin')
margin_calcs('researchAndDevelopment', 'totalRevenue', 'r&d_margin')
margin_calcs('ebitda', 'totalRevenue', 'ebitda_margin')
margin_calcs('netIncome', 'totalRevenue', 'net_margin')
margin_calcs('netIncome', 'totalAssets', 'ret_on_assets')
margin_calcs('netIncome', 'totalShareholderEquity', 'ret_on_equity')

# Liquidity metrics
margin_calcs('operatingCashflow', 'totalCurrentLiabilities', 'op_cf')
margin_calcs('totalCurrentAssets', 'totalCurrentLiabilities', 'current_ratio')

# Metrics for checking the debt level of a company (Solvency ratios)
margin_calcs('totalLiabilities', 'totalAssets', 'debt_to_assets')
margin_calcs('totalLiabilities', 'totalShareholderEquity', 'debt_to_equity')
margin_calcs('ebitda', 'interestAndDebtExpense', 'int_cov_ratio')

# Valuation ratios
margin_calcs('market_cap', 'netIncome', 'p_to_e')
margin_calcs('market_cap', 'book_value', 'p_to_b')
margin_calcs('market_cap', 'totalRevenue', 'p_to_r')
margin_calcs('market_cap', 'operatingCashflow', 'p_to_op_cf')
margin_calcs('market_cap', 'cashflowFromInvestment', 'p_to_inv_cf')
margin_calcs('market_cap', 'cashflowFromFinancing', 'p_to_fin_cf')

# Dividends per share
margin_calcs('dividendPayoutCommonStock', 'commonStockSharesOutstanding', 'div_yield')

# Inventory issues
margin_calcs('inventory', 'costofGoodsAndServicesSold', 'inv_ratio')


mdl_input_data.loc[mdl_input_data['Symbol'] == 'AAIC', ['totalRevenue', 'totalRevenue_1Q_lag', 'totalRevenue_2Q_lag'
    ,'totalRevenue_4Q_lag','market_cap', 'market_cap_1Q_lag', 'market_cap_2Q_lag'
    ,'market_cap_4Q_lag', 'p_to_r', 'p_to_r_1Q_lag', 'p_to_r_2Q_lag', 'p_to_r_4Q_lag'
    ,'totalRevenue_1Q_gth', 'totalRevenue_2Q_gth', 'totalRevenue_4Q_gth'
    ,'p_to_r_1Q_gth', 'p_to_r_2Q_gth', 'p_to_r_4Q_gth']]


# Assess the character variables
print(pd.DataFrame(mdl_input_data.dtypes, columns=['datatype']).sort_values('datatype'))
mdl_input_data.info()
# useful for putting all of the character fields at the bottom of the print.

# There are 9 character fields before we get dummy values for these fields we need to look into them:
#    - Symbol has 4,566 unique values and Name has 4,397 unique values, we will drop these features as otherwise we will
#      be modelling at too low a level
#    - Asset type, currency and country all only have one unique value and as such are redundant to the model
#    - exchange(2) and month (2) will be included in the model
#    - We will investigate if Industry (148) should be included or if Sector (13) gives us enough information
#    - about the company

char_columns = ['Symbol', 'AssetType', 'Name', 'month', 'Exchange', 'Currency', 'Country', 'Sector', 'Industry']
unique_vals = []
for entry in char_columns:
    cnt_entry_i = unique_column(mdl_input_data, entry).shape[0]
    unique_vals.append([entry, cnt_entry_i])

print(unique_vals)

# Without doing any statistical tests we can see that there is clearly large differences between different industries
# We will drop sector from our model and keep Industry
mdl_input_data[['Sector', 'Industry', 'future_price_gth']].groupby(by=['Sector', 'Industry']).mean()

# Drop the required columns in a new dataframe called "model_input_data"
mdl_input_data = mdl_input_data.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector'], axis=1)

print(pd.DataFrame(mdl_input_data.dtypes, columns=['datatype']).sort_values('datatype'))  # 3 character fields remaining

##################################################################################################################
# Section 4 - Modelling
##################################################################################################################

# The features which we would expect to be important in predicting share price
# are P/E growth, net margin growth, P/B growth etc. so we would expect that the important features which
# will come out of a first "very rough" run of the model will be revenue, net profit, market cap etc.

# Please note I first tried a simpler version of the model containing only balance sheet and income statment
# information and the R squared was only 1.5%, test set had a negative R squared
# The original random forest model had an R squared of 1.3% where most of the test scores were actually negative
#

X = mdl_input_data.iloc[:, :-1]
X = pd.get_dummies(data=X, drop_first=True)
y = mdl_input_data.iloc[:, -1:]

X_train = X[X.index < '2019-07']
y_train = y[y.index < '2019-07']
X_test = X[X.index == '2019-07']
y_test = y[y.index == '2019-07']

# Feature Scaling
# See "1.Data" what we are doing below is (x - mu) / sd
# We have scaled both the dependent and independent variables (xi any yi)
sc_X_train = StandardScaler()
sc_X_test = StandardScaler()
sc_y_train = StandardScaler()
sc_y_test = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)
X_test = sc_X_test.fit_transform(X_test)
y_train = sc_y_train.fit_transform(y_train)
y_test = sc_y_test.fit_transform(y_test)

# Linear regression model

# Build model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.score(X_train, y_train)  # 2.77%
lin_reg.score(X_test, y_test)  # << 0%

lin_reg.coef_[0]
print(X)

# Display two vectors, the y predicted v y train
y_train_pred = pd.DataFrame(lin_reg.predict(X_train), columns=['y_train_pred'])
y_train_df = pd.DataFrame(y_train, columns=['y_train'])
lin_reg_pred = pd.concat([y_train_df, y_train_pred.set_index(y_train_df.index)], axis=1)
print(lin_reg_pred)

# Visually comparing the predicted values for profit versus actual
sns.scatterplot(data=lin_reg_pred, x='y_train_pred', y='y_train', palette='deep', legend=False)
plt.xlabel('Predicted Stock Price', size=12)
plt.ylabel('Actual Stock Price', size=12)
plt.title("Predicted v Actual Stock Price", fontdict={'size': 16})
plt.tight_layout()
plt.show()

lin_reg_pred.sort_values(by=['y_train_pred'])

# Run a random forest to check what are the most important features in predicting future stock prices
X_train_rf = X_train
y_train_rf = y_train.ravel()
X_test_rf = X_test
y_test_rf = y_test.ravel()

np.shape(X_train_rf)

# Grid Search

rfr = RandomForestRegressor(criterion='mse')
param_grid = [{'n_estimators': [50, 100, 200], 'max_depth': [2, 4, 8], 'max_features': ['auto', 'sqrt']
                  , 'random_state': [21]}]

# Create a GridSearchCV object
grid_rf_reg = GridSearchCV(
    estimator=rfr,
    param_grid=param_grid,
    scoring='r2',
    n_jobs=-1,
    cv=3)

print(grid_rf_reg)

grid_rf_reg.fit(X_train_rf, y_train_rf)  # Fitting 3 folds

best_rsqr = grid_rf_reg.best_score_
best_parameters = grid_rf_reg.best_params_
print("Best R squared: : {:.2f} %".format(best_rsqr * 100))
print("Best Parameters:", best_parameters)

# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_reg.cv_results_)
print(cv_results_df)

# Get feature importances from our random forest model
importances = grid_rf_reg.best_estimator_.feature_importances_
imp_df = pd.DataFrame(list(importances), columns=['Feature Importance'])
print(imp_df)

sorted_index = np.argsort(importances)[::-1]
print(sorted_index)

# Get the index of importances from greatest importance to least
index_df = pd.DataFrame(list(np.argsort(importances[::-1])), columns=['column_index'])
print(index_df)

rf_imp_df = pd.concat([index_df, imp_df, pd.DataFrame(X.columns[sorted_index], columns=['Feature Name'])], axis=1)
rf_imp_df.sort_values(by=['Feature Importance'], inplace=True)
print(rf_imp_df)
