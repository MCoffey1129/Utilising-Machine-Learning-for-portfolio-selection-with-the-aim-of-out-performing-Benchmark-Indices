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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from numpy import inf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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
    missing_tbl['missing_pc'] = missing_tbl['num missing'] / table.shape[0]
    return missing_tbl


# Function used for margin calculations
@timer
def margin_calcs(input_num, input_den, output_col):
    for j in [0, 1, 2, 4]:
        if j == 0:
            mdl_input_data[output_col] = mdl_input_data[input_num] / mdl_input_data[input_den]
            mdl_input_data[output_col].replace([np.inf, -np.inf], 0, inplace=True)

        else:
            mdl_input_data[output_col + '_' + str(j) + 'Q_lag'] = mdl_input_data[input_num + '_' + str(j) + 'Q_lag'] \
                                                                  / mdl_input_data[input_den + '_' + str(j) + 'Q_lag']

            mdl_input_data[output_col + '_' + str(j) + 'Q_gth'] = (mdl_input_data[output_col]
                                                                   - mdl_input_data[
                                                                       output_col + '_' + str(j) + 'Q_lag'])

            mdl_input_data[output_col + '_' + str(j) + 'Q_lag'].replace([np.inf, -np.inf], 0, inplace=True)

            mdl_input_data[output_col + '_' + str(j) + 'Q_gth'].replace([np.inf, -np.inf], 0, inplace=True)


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
    pd.merge(pd.merge(pd.merge(eps_data, inc_st_data, how='left', on=['fiscalDateEnding', 'Symbol'])
                      , bs_data.drop(labels=['reportedCurrency'], axis=1), how='inner',
                      on=['fiscalDateEnding', 'Symbol'])
             , cf_data.drop(labels=['netIncome', 'reportedCurrency'], axis=1), how='left',
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
financial_results_reorder.head()
for i in columns:
    for j in [1, 2, 4]:
        # Get the 4 quarter lags on to the same row i.e. the column totalRevenue_2Q_lag is the totalRevenue
        # from 6 months previous
        financial_results_reorder[i + '_' + str(j) + 'Q_lag'] = financial_results_reorder[i].shift(-j)
        financial_results_reorder[i + '_' + str(j) + 'Q_gth'] = (financial_results_reorder[i] -
                                                                 financial_results_reorder[i].shift(-j)) / \
                                                                abs(financial_results_reorder[i].shift(-j))

        financial_results_reorder[i + '_' + str(j) + 'Q_gth'].replace([np.inf, -np.inf], 0, inplace=True)

        # The below code ensures that we are not taking in financial data from an incorrect symbol
        financial_results_reorder.loc[financial_results_reorder['Symbol'].shift(-j) !=
                                      financial_results_reorder['Symbol'], i + '_' + str(j) + 'Q_lag'] = np.nan
        financial_results_reorder.loc[financial_results_reorder['Symbol'].shift(-j) !=
                                      financial_results_reorder['Symbol'], i + '_' + str(j) + 'Q_gth'] = np.nan

print(financial_results_reorder.loc[(financial_results_reorder['Symbol'] == 'AAIC') |
                                    (financial_results_reorder['Symbol'] == 'A'), [
                                        'Symbol', 'surprise', 'surprise_1Q_lag', 'surprise_2Q_lag', 'surprise_4Q_lag'
                                        , 'surprise_1Q_gth', 'surprise_2Q_gth', 'surprise_4Q_gth']])

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
stk_prices.loc[stk_prices['future_price_gth'] >= 0.1, 'gt_10pc_gth'] = 1
stk_prices.loc[stk_prices['future_price_gth'] < 0.1, 'gt_10pc_gth'] = 0
stk_prices.tail(100)

# Make dt_m the index
stk_prices.drop("dt", inplace=True, axis=1)
stk_prices.index = stk_prices['dt_m'].rename("dt")
stk_prices.head(25)
stk_prices.columns
stk_prices['gt_10pc_gth'].value_counts()

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
# seaborn_lm_plt(stk_prices, 10, 50)  # cases which have a price of less than $10 and a future price less than $50
# seaborn_lm_plt(stk_prices, 100, 500)
# seaborn_lm_plt(stk_prices, 5, 20)
# seaborn_lm_plt(stk_prices, 5, 1000000)  # Outlier is Gamestop share increase from July '20 to Jan '21


# Code for checking the stocks with the largest 6 month gains on companies who had a share price of less than 5 euro
# GameStop's (GME) share price increased from €4.01 in July 2020 to €320.99 in Jan '21
# This is an outlier as the share increase was not a result of the fundamentals of the company.
a = stk_prices.loc[stk_prices['close_price'] < 5]
b = copy.deepcopy(a)
b['diff'] = b['future_price'] - b['close_price']
print(b.sort_values(by=['diff'], ascending=False))

##################################################################################################################
# Section 3.1 - Finalise the data required
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

mdl_input_data = mdl_data

# Create a market cap column
mdl_input_data['market_cap'] = mdl_input_data['commonStockSharesOutstanding'] * mdl_input_data['close_price']
mdl_input_data['market_cap_1Q_lag'] = mdl_input_data['commonStockSharesOutstanding_1Q_lag'] \
                                      * mdl_input_data['close_price_3M_lag']
mdl_input_data['market_cap_2Q_lag'] = mdl_input_data['commonStockSharesOutstanding_2Q_lag'] \
                                      * mdl_input_data['close_price_6M_lag']
mdl_input_data['market_cap_4Q_lag'] = mdl_input_data['commonStockSharesOutstanding_4Q_lag'] \
                                      * mdl_input_data['close_price_12M_lag']

# Create a simplified EV column (free cashflow equals op cf - capex)
mdl_input_data['EV_simple'] = mdl_input_data['market_cap'].fillna(0) - mdl_input_data['currentDebt'].fillna(0) \
                              + mdl_input_data['operatingCashflow'].fillna(0) \
                              - mdl_input_data['capitalExpenditures'].fillna(0)
mdl_input_data['EV_simple_1Q_lag'] = mdl_input_data['market_cap_1Q_lag'].fillna(0) \
                                     - mdl_input_data['currentDebt_1Q_lag'].fillna(0) \
                                     + mdl_input_data['operatingCashflow_1Q_lag'].fillna(0) \
                                     - mdl_input_data['capitalExpenditures_1Q_lag'].fillna(0)
mdl_input_data['EV_simple_2Q_lag'] = mdl_input_data['market_cap_2Q_lag'].fillna(0) \
                                     - mdl_input_data['currentDebt_2Q_lag'].fillna(0) \
                                     + mdl_input_data['operatingCashflow_2Q_lag'].fillna(0) \
                                     - mdl_input_data['capitalExpenditures_2Q_lag'].fillna(0)
mdl_input_data['EV_simple_4Q_lag'] = mdl_input_data['market_cap_4Q_lag'].fillna(0) \
                                     - mdl_input_data['currentDebt_4Q_lag'].fillna(0) \
                                     + mdl_input_data['operatingCashflow_4Q_lag'].fillna(0) \
                                     - mdl_input_data['capitalExpenditures_4Q_lag'].fillna(0)

mdl_input_data.loc[mdl_input_data['Symbol'] == 'APLT', 'EV_simple']
# Create new features required for modelling i.e. P/E, gross margin, net margin etc.

# Profitability metrics
margin_calcs('grossProfit', 'totalRevenue', 'gross_margin')
margin_calcs('researchAndDevelopment', 'totalRevenue', 'r&d_margin')
margin_calcs('ebitda', 'totalRevenue', 'ebitda_margin')
margin_calcs('ebit', 'totalRevenue', 'ebit_margin')
margin_calcs('netIncome', 'totalRevenue', 'net_margin')
margin_calcs('netIncome', 'totalAssets', 'ret_on_assets')
margin_calcs('netIncome', 'totalShareholderEquity', 'ret_on_equity')
margin_calcs('operatingCashflow', 'totalRevenue', 'cf_to_rev')

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
margin_calcs('market_cap', 'ebitda', 'p_to_ebitda')
margin_calcs('EV_simple', 'netIncome', 'ev_to_e')
margin_calcs('EV_simple', 'book_value', 'ev_to_b')
margin_calcs('EV_simple', 'totalRevenue', 'ev_to_r')
margin_calcs('EV_simple', 'ebitda', 'ev_to_ebitda')
margin_calcs('market_cap', 'operatingCashflow', 'p_to_op_cf')
margin_calcs('market_cap', 'cashflowFromInvestment', 'p_to_inv_cf')
margin_calcs('market_cap', 'cashflowFromFinancing', 'p_to_fin_cf')
margin_calcs('EV_simple', 'operatingCashflow', 'ev_to_op_cf')

# Dividends per share
margin_calcs('dividendPayoutCommonStock', 'commonStockSharesOutstanding', 'div_yield')

# Inventory issues
margin_calcs('inventory', 'costofGoodsAndServicesSold', 'inv_ratio')

industry_avg = mdl_input_data[['Industry', 'p_to_e', 'p_to_b', 'p_to_r', 'ev_to_e', 'ev_to_b', 'ev_to_r',
                               'div_yield', 'debt_to_equity', 'p_to_op_cf', 'ev_to_op_cf', 'p_to_ebitda',
                               'ev_to_ebitda',
                               'int_cov_ratio', 'gross_margin', 'ebitda_margin', 'ebit_margin', 'op_cf', 'market_cap',
                               'totalRevenue_2Q_gth', 'netIncome_2Q_gth', 'reportedEPS', 'surprisePercentage']] \
    .groupby(by=['Industry', 'dt']).mean()

industry_avg_rename = industry_avg.rename(columns=lambda s: s + '_ind_avg')
print(industry_avg_rename)

mdl_input_data_upd = pd.merge(mdl_input_data, industry_avg_rename, how='left', on=['dt', 'Industry'])

for col1 in list(industry_avg):
    mdl_input_data_upd[col1 + '_v_ind_avg'] = mdl_input_data_upd[col1] - mdl_input_data_upd[col1 + '_ind_avg']
    mdl_input_data_upd.drop([col1 + '_ind_avg'], axis=1, inplace=True)

'surprisePercentage_v_ind_avg'

# sns.boxplot(data=mdl_input_data_upd, x='gt_10pc_gth', y='surprisePercentage_v_ind_avg', whis=10)
# plt.yscale('log')
# plt.show()

mdl_input_data_upd[['p_to_r_v_ind_avg', 'gt_10pc_gth']].groupby(by=['gt_10pc_gth']).mean()

# Checks
print(mdl_input_data_upd.loc[
          mdl_input_data['Symbol'] == 'AAIC', ['totalRevenue', 'totalRevenue_1Q_lag', 'totalRevenue_2Q_lag'
              , 'totalRevenue_4Q_lag', 'market_cap', 'market_cap_1Q_lag', 'market_cap_2Q_lag'
              , 'market_cap_4Q_lag', 'p_to_r', 'p_to_r_1Q_lag', 'p_to_r_2Q_lag', 'p_to_r_4Q_lag'
              , 'totalRevenue_1Q_gth', 'totalRevenue_2Q_gth', 'totalRevenue_4Q_gth'
              , 'p_to_r_1Q_gth', 'p_to_r_2Q_gth', 'p_to_r_4Q_gth'
              , 'netIncome', 'netIncome_1Q_lag', 'netIncome_2Q_lag', 'netIncome_4Q_lag'
              , 'netIncome_1Q_gth', 'netIncome_2Q_gth', 'netIncome_4Q_gth'
              , 'p_to_e', 'p_to_e_1Q_lag', 'p_to_e_2Q_lag', 'p_to_e_4Q_lag'
              , 'p_to_e_1Q_gth', 'p_to_e_2Q_gth', 'p_to_e_4Q_gth'
              , 'surprise', 'surprise_1Q_lag', 'surprise_2Q_lag', 'surprise_4Q_lag'
              , 'surprise_1Q_gth', 'surprise_2Q_gth', 'surprise_4Q_gth']])

ds = mdl_input_data[mdl_input_data_upd.isin([np.inf, -np.inf])].sum()
print(ds)

##################################################################################################################
# Section 3.2 - Replacing nulls and Feature Selection
##################################################################################################################


# Split the data into train and test in order to prevent data leakage. Any decision made around dropping columns
# or replacing nulls needs to be completed assuming we have no information on the test set
mdl_data_test = mdl_input_data_upd[mdl_input_data_upd.index == '2020-07']
mdl_data_train = mdl_input_data_upd[mdl_input_data_upd.index < '2020-07']
mdl_deploy = mdl_input_data_upd[mdl_input_data_upd.index == '2021-01']

# Drop any rows where fiscalDateEnding is null (we will have no revenue or profit information for these rows)
null_value_pc(mdl_data_train)
mdl_data_train = mdl_data_train.dropna(how='all', subset=['fiscalDateEnding'])
mdl_data_test = mdl_data_test.dropna(how='all', subset=['fiscalDateEnding'])

# Drop any rows where totalRevenue is Null
null_value_pc(mdl_data_train)
mdl_data_train = mdl_data_train.dropna(how='all', subset=['totalRevenue'])
mdl_data_test = mdl_data_test.dropna(how='all', subset=['totalRevenue'])

# Drop any rows where grossProfit is Null
null_value_pc(mdl_data_train)
mdl_data_train = mdl_data_train.dropna(how='all', subset=['grossProfit'])
mdl_data_test = mdl_data_test.dropna(how='all', subset=['grossProfit'])

# Drop any rows which do not contain the target variable. Drop these cases from both test and Live
mdl_data_train = mdl_data_train.dropna(how='all', subset=['gt_10pc_gth'])
mdl_data_test = mdl_data_test.dropna(how='all', subset=['gt_10pc_gth'])

mdl_data_train = mdl_data_train.drop('reportedCurrency', axis=1)
mdl_data_test = mdl_data_test.drop('reportedCurrency', axis=1)

# Remaining null values
null_df = null_value_pc(mdl_data_train)  # there are a large number of Null values to deal with in all but 6 columns
null_df = null_df.sort_values(by='missing_pc')
print(null_df)

# Numeric fields
mdl_train_numeric = mdl_data_train.iloc[:, 10:]
mdl_train_numeric.drop(columns='month', inplace=True)
mdl_train_numeric.head()

mdl_train_numeric = mdl_train_numeric.fillna(0)

null_value_pc(mdl_train_numeric)  # there are non non-zero values

# Get the correlation between each of the fields and the target variable
col_list = list(mdl_train_numeric.columns)
col_list.remove('future_price_gth')
print(col_list)
corr_list = []

for col in col_list:
    corr = pd.DataFrame(mdl_train_numeric[[col, 'future_price_gth']].corr())
    corr = corr.iloc[0, 1]
    corr_list.append(corr)

print(len(col_list))
print(corr_list)
print(len(corr_list))

corr_df = pd.concat([pd.DataFrame(col_list, columns=['columns']), pd.DataFrame(corr_list
                                                                               , columns=['Correlation'])], axis=1)

corr_df_nulls = pd.merge(null_df, corr_df, left_index=True, right_on='columns')
print(corr_df_nulls.sort_values(by='Correlation'))

# Dataframe containing each of the correlations
all_corrs = mdl_train_numeric.corr()

# Sector
mdl_data_train['Sector'].unique()  # nan and 'None' in the column
mdl_data_train['Sector'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_train['Sector'].unique()  # no nan or 'None' values in the column
mdl_data_train['Sector'].isnull().sum()  # 0 value returned

mdl_data_test['Sector'].unique()  # nan and 'None' in the column
mdl_data_test['Sector'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_test['Sector'].unique()  # no nan or 'None' values in the column
mdl_data_test['Sector'].isnull().sum()  # 0 value returned

# Industry
mdl_data_train['Industry'].isnull().sum()  # 7 missing values
mdl_data_train['Industry'].unique()  # 'None' values in the column
mdl_data_train['Industry'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_train['Industry'].unique()  # no 'None' values in the column
mdl_data_train['Industry'].isnull().sum()  # 0 value returned

mdl_data_test['Industry'].isnull().sum()  # 4 missing values
mdl_data_test['Industry'].unique()  # 'None' values in the column
mdl_data_test['Industry'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_test['Industry'].unique()  # no 'None' values in the column
mdl_data_test['Industry'].isnull().sum()  # 0 value returned

null_value_pc(mdl_data_train)  # Sector and industry no longer have missing values

# We need to drop the 'future_price' and 'future_price_gth' from our model
future_test_pc_df = mdl_data_test[['future_price_gth']]
mdl_data_train.drop(['future_price', 'future_price_gth', 'fiscalDateEnding', 'reportedDate'], axis=1, inplace=True)
mdl_data_test.drop(['future_price', 'future_price_gth', 'fiscalDateEnding', 'reportedDate'], axis=1, inplace=True)

mdl_data_train.shape
future_test_pc_df.head()

##


# Assess the character variables
print(pd.DataFrame(mdl_data_train.dtypes, columns=['datatype']).sort_values('datatype'))
mdl_data_train.info()
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
    cnt_entry_i = unique_column(mdl_data_train, entry).shape[0]
    unique_vals.append([entry, cnt_entry_i])

print(unique_vals)

# Without doing any statistical tests we can see that there is clearly large differences between different industries
# We will drop sector from our model and keep Industry
mdl_data_train[['Sector', 'Industry', 'gt_10pc_gth']].groupby(by=['Sector', 'Industry']).mean()

# Drop the required columns in a new dataframe called "model_input_data"
symbol_test_df = mdl_data_test[['Symbol']]
mdl_data_train = mdl_data_train.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector'], axis=1)
mdl_data_test = mdl_data_test.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector'], axis=1)

print(pd.DataFrame(mdl_data_train.dtypes, columns=['datatype']).sort_values('datatype'))  # 3 character fields remaining

# Replace all other missing values with 0
# Revenue, gross profit and other values which were null had very few nulls......
null_value_pc(mdl_data_train)

# mdl_data_train.fillna(0, inplace=True)
# mdl_data_test.fillna(0, inplace=True)
# mdl_data_train.isnull().sum()  # no missing values
# mdl_data_test.isnull().sum()  # no missing values
# mdl_data_train.describe()

# mdl_data_train['gt_10pc_gth'] = mdl_data_train['gt_10pc_gth'].astype(int)
# mdl_data_test['gt_10pc_gth'] = mdl_data_test['gt_10pc_gth'].astype(int)

mdl_data_test.info()

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

X_train_df = mdl_data_train.drop(['gt_10pc_gth', 'month'], axis=1)
X_train_df = pd.get_dummies(data=X_train_df, drop_first=True)
y_train_df = mdl_data_train['gt_10pc_gth']
X_test_df = mdl_data_test.drop(['gt_10pc_gth'], axis=1)
X_test_df = pd.get_dummies(data=X_test_df, drop_first=True)
X_test_df = X_test_df.drop(['Industry_Oil & Gas Pipelines'], axis=1)
y_test_df = mdl_data_test['gt_10pc_gth']

X_train_df.shape
X_test_df.shape
y_train_df.shape
y_test_df.shape

##
# y_train_df.to_csv(r'Files\y_train_df.csv', index=True, header=True)

X_train = X_train_df.values
y_train = y_train_df.values
X_test = X_test_df.values
y_test = y_test_df.values

y_train = y_train.reshape(len(y_train),
                          1)  # For feature scaling you need a 2D array as this is what the StandardScaler expects
y_test = y_test.reshape(len(y_test), 1)
print(y_train)

# Feature Scaling
# Scale the values such that each are in the range [0,1]
# Scaling is necessary for feature selection and modelling
sc_X_train = MinMaxScaler()
sc_X_test = MinMaxScaler()
X_train = sc_X_train.fit_transform(X_train)
X_test = sc_X_test.fit_transform(X_test)

# Impute missing values
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5, weights = "uniform")
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.fit_transform(X_test)

# Univariate Feature selection


# select_feature = SelectKBest(chi2, k=1000).fit(X_train, y_train)
# select_features_df = pd.DataFrame({'Feature': list(X_train_df.columns),
#                                    'Scores' : select_feature.scores_})
# a = select_features_df.sort_values(by='Scores', ascending=False)
# a.head(50)
# a.loc[a['Feature']=='Industry_Food Distribution']
#
# X_train = select_feature.transform(X_train)
# X_test = select_feature.transform(X_test)
##


X_train_rv = X_train
y_train_rv = y_train.ravel()
X_test_rv = X_test
y_test_rv = y_test.ravel()
np.shape(X_train_rv)
np.shape(X_test_rv)
np.shape(y_train_rv)
np.shape(y_test_rv)

# X_train_rv_df = pd.DataFrame(X_train_rv)
# y_train_rv_df = pd.DataFrame(y_train_rv)
# X_test_rv_df = pd.DataFrame(X_test_rv)
# y_test_rv_df = pd.DataFrame(y_test_rv)
#
# X_train_rv_df.to_csv(r'Files\X_train_rv_df.csv', index=False, header=True)
# y_train_rv_df.to_csv(r'Files\y_train_rv_df.csv', index=False, header=True)
# X_test_rv_df.to_csv(r'Files\X_test_rv_df.csv', index=False, header=True)
# y_test_rv_df.to_csv(r'Files\y_test_rv_df.csv', index=False, header=True)

X_train_rv_df = pd.read_csv(r'Files\X_train_rv_df.csv')
y_train_rv_df = pd.read_csv(r'Files\y_train_rv_df.csv')
X_test_rv_df = pd.read_csv(r'Files\X_test_rv_df.csv')
y_test_rv_df = pd.read_csv(r'Files\y_test_rv_df.csv')

X_train_rv = X_train_rv_df.values
y_train_rv = y_train_rv_df.values.ravel()
X_test_rv = X_test_rv_df.values
y_test_rv = y_test_rv_df.values.ravel()

y_test_rv.shape

# Recursive feature elimination
# from sklearn.feature_selection import RFECV
# rfecv= RFECV(estimator=RandomForestClassifier(), step=100, cv=5, scoring='precision')
# rfecv= rfecv.fit(X_train_rv, y_train_rv)
# print('Optimal number of features : ' , rfecv.n_features_)
# print('Best features :' , X_train_df.columns[rfecv.support_])
#
#
# plt.figure()
# plt.xlabel("No of features selected")
# plt.ylabel("Cross Validation score")
# plt.plot(range(1,len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

level0.append(('knn', KNeighborsClassifier()))
level0.append(('r_forest', RandomForestClassifier()))
level0.append(('log_reg', LogisticRegression(max_iter=1000)))


# classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#classifier = RandomForestClassifier(n_estimators = 10)
classifier = LogisticRegression(C=5,max_iter=1000)
# classifier = LogisticRegression(C=1, max_iter=1000, random_state=1)

classifier.fit(X_train_rv, y_train_rv)
#
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#
y_pred_train = classifier.predict(X_train_rv)
cm = confusion_matrix(y_train_rv, y_pred_train)
print(cm)

y_pred = classifier.predict(X_test_rv)
cm = confusion_matrix(y_test_rv, y_pred)
print(cm)  # only 1 incorrect prediction
print(classification_report(y_test_rv, y_pred))

# Random Forest
# Run a random forest to check what are the most important features in predicting future stock prices


#################################################################################################################
# Section 0 - Grid Search to find the best hyperparameters to use
#################################################################################################################
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

model_params = {
    'XGB': {'model': XGBClassifier(),
                            'params': {'learning_rate': [0.1,0.3,0.5], 'max_depth': [3,6,9],
                                       'gamma' : [0,1,5]}},

    'CatBoost': {'model': CatBoostClassifier(),
                 'params': {'learning_rate': [0.3, 0.1, 0.01], 'max_depth': [3, 4, 5],
                            'n_estimators': [100, 200, 300]}}

    'random_forest': {'model': RandomForestClassifier(criterion='entropy', random_state=1),
                      'params': {'n_estimators': [200, 500, 1000], 'max_features': ['sqrt', 'log2'],
                                 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}},

    'knn': {'model': KNeighborsClassifier(algorithm='kd_tree'),
            'params': {'n_neighbors': [5, 10, 15, 25, 50, 100]}
            }
}

#
print(model_params)

scores = []
all_scores = []

# Random Forest CV mean precision of 57%, logistic regression of 52% and KNN of 48%
# With KNN I did not have enough memory to run further tests. Check a value of 0.1 for logistic regression

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], n_jobs=-1, scoring='f1', cv=10,
                       return_train_score=True, verbose=2)
    clf.fit(X_train_rv, y_train_rv)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    all_scores.append({
        'model': model_name,
        'avg_score': clf.cv_results_['mean_test_score'],
        'std_test_score': clf.cv_results_['std_test_score'],
        'params': clf.cv_results_['params']
    })

print(scores)

print(all_scores)

scores_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(scores_df)


#################################################################################################################
# Section 4.1 - Stacking the models with the best hyperparameters
#################################################################################################################

# Define the base models
# Please note the Random Forest
from sklearn.ensemble import StackingClassifier


level0 = list()

level0.append(('knn', KNeighborsClassifier()))
level0.append(('r_forest', RandomForestClassifier()))
level0.append(('XGB', XGBClassifier(n_jobs=-1)))
level0.append(('CB', CatBoostClassifier()))

# define meta learner model
level1 = XGBClassifier()

# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1, verbose=2)

# fit the model on all available data
model.fit(X_train_rv, y_train_rv)

# predict the test data
y_pred = model.predict(X_test_rv)

cm = confusion_matrix(y_test_rv, y_pred)
print(cm)
print(classification_report(y_test_rv, y_pred))

# Probabilities
stkd_probs = model.predict_proba(X_test_rv)
stkd_probs_df = pd.DataFrame(stkd_probs[:,1], columns = ['Stacked_mdl_prob'])

stkd_mdl_results = pd.concat([symbol_test_df,
                                mdl_data_test[['Industry','gt_10pc_gth']].set_index(symbol_test_df.index),
                                future_test_pc_df.set_index(symbol_test_df.index),
                                stkd_probs_df.set_index(symbol_test_df.index)], axis=1)

stkd_mdl_results.sort_values(by=['Stacked_mdl_prob'], inplace=True, ignore_index=True, ascending=False)
stkd_mdl_results.to_csv(r'Files\stkd_mdl_results.csv', index=False, header=True)

stkd_mdl_results['avg_ret_of_portfolio'] = stkd_mdl_results['future_price_gth'].cumsum() / ( stkd_mdl_results.index + 1)
stkd_mdl_results['avg_ret_of_market'] = stkd_mdl_results['future_price_gth'].mean()
print(stkd_mdl_results)

sns.set()
sns.lineplot(data=stkd_mdl_results.reset_index(),x='index', y='avg_ret_of_portfolio')
sns.lineplot(data=stkd_mdl_results.reset_index(),x='index', y='avg_ret_of_market', ls='--')
plt.show()


#################################################################################################################
# Section 4.2 - Genetic Algorithm
#################################################################################################################

# Genetic Algorithms
# Assign the values outlined to the inputs
from tpot import TPOTClassifier

# number_generations = 3  # nothing to do with the dataset
# population_size = 4  # Start with 4 algorithms
# offspring_size = 3
# scoring_function = 'precision'

# Create the tpot classifier
# tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
#                           offspring_size=offspring_size, scoring=scoring_function,
#                           verbosity=2, random_state=2, cv=5)
#
# # Fit the classifier to the training data
# tpot_clf.fit(X_train_rv, y_train_rv)
#
# # Score on the test set
# tpot_clf.score(X_test_rv, y_test_rv)


#################################################################################################################
# Section 4.3 - Boosted models
#################################################################################################################

# XGBoost



xgbc = XGBClassifier()
cbc = CatBoostClassifier()


model_params_XGB = {
    'XGB': {'model': XGBClassifier(),
                            'params': {'learning_rate': [0.3,0.1,0.01], 'max_depth': [3,5,7],
                                       'gamma' : [0,1,5]}}}

model_params_CB = {
    'CatBoost': {'model': CatBoostClassifier(),
                            'params': {'learning_rate': [0.3,0.1,0.01], 'max_depth': [3,5,7],
                                       'n_estimators' : [100,200,500]}}}


gscb = grid_search(model_params_CB,
            X_train_rv,
            y=None,
            cv=5,
            scoring='precision',
            verbose=True
            )


gsxgb =



# Fit the model
gscb.fit(X_train_rv, y_train_rv)

#returns the estimator with the best performance
print(gscb.best_estimator_)

#returns the best score
print(gscb.best_score_)

#returns the best parameters
print(gscb.best_params_)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test_rv)
cm = confusion_matrix(y_test_rv, y_pred)
print(cm)
accuracy_score(y_test_rv, y_pred)  #

# Applying k-Fold Cross Validation
# The average accuracy across the K fold cross validations is 96.53% making it the best performing algorithm
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train_rv, y=y_train_rv, cv=3)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Precision: {:.2f} %").format(precision.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

# Predict with a model
gbt_preds = clf_gbt.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))




#################################################################################################################
# Section 4.4 - Neural Networks
#################################################################################################################

# Artificial Neural Network
import tensorflow as tf
from keras.layers import Dropout
from keras.layers import LeakyReLU

tf.random.set_seed(2)

# Initializing the ANN
# Using a sequential neutral network
ann_1 = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# The first hidden layer contains 6 nodes from the Dense class and the activation function is the rectifier function
# Activation = max(x,0) (which is 0 for negative values and then increase linearly until 1)
ann_1.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Adding the second hidden layer
# The second hidden layer contains 6 nodes and the activation function is the rectifier function
# Activation = max(x,0) (which is 0 for negative values and then increase linearly until 1)
ann_1.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann_1.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Adding the output layer
# The output layer contains 1 node and the activation function is the sigmoid function
# Activation = 1 / (1 + e(-x)) (which is very useful when predicting probabilities)
# If we wanted three output variables the output layer would require 3 nodes/neurons
ann_1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
# Optimizer used is Adaptive Moment Estimation (Adam), the training cost in the case of Adam uses stochastic gradient
# descent.
# For binary outcomes we always have to use binary_crossentropy and for non-binary is categorical_crossentropy
# For metrics we can put in 'mse', 'mae', 'mape', 'cosine' (for numeric output node you would also
# change loss to 'mse'
ann_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision()])

# Training the ANN on the Training set
# We are doing batch learning, a good rule of thumb is to use 32
# epochs is the number of times we run over the data, in our case we run over the data 100 times
history_1 = ann_1.fit(X_train_rv, y_train_rv, batch_size=32, epochs=100)

# Fit the second Neural Network
ann_2 = tf.keras.models.Sequential()
ann_2.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann_2.add(Dropout(0.2))
ann_2.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann_2.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann_2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision()])
history_2 = ann_2.fit(X_train_rv, y_train_rv, batch_size=32, epochs=100)

# Fit the third Neural Network
ann_3 = tf.keras.models.Sequential()
ann_3.add(tf.keras.layers.Dense(units=50, activation=tf.keras.activations.tanh))
ann_2.add(Dropout(0.1))
ann_3.add(tf.keras.layers.Dense(units=25, activation=tf.keras.activations.tanh))
ann_3.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.tanh))
ann_3.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision()])
history_3 = ann_3.fit(X_train_rv, y_train_rv, batch_size=32, epochs=100)

# Plot losses
# Once we've fit a model, we usually check the training loss curve to make sure it's flattened out.
# The history returned from model.fit() is a dictionary that has an entry, 'loss', which is the
# training loss. We want to ensure this has more or less flattened out at the end of our training.


# Plot the losses from the fit
plt.plot(history_1.history['precision'], label='NN_1')
plt.plot(history_2.history['precision'], label='NN_2')
plt.plot(history_3.history['precision'], label='NN_3')
plt.title('Precision v Loss')
plt.legend()
plt.show()

y_pred = ann.predict(X_test_rv)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test_rv.reshape(len(y_test_rv), 1)), 1))

# Making the Confusion Matrix
# The model has an accuracy score of 86.5%
# But the recall is only 53% for customer's leaving (this is to be expected). See the classification report.
cm = confusion_matrix(y_test_rv, y_pred)
print(cm)
cr = classification_report(y_test_rv, y_pred)
print(cr)
accuracy_score(y_test_rv, y_pred)

y_pred.shape

# Write out the CSV
y_pred_df = pd.DataFrame(y_pred, columns=['prob'])
y_pred.to_csv(r'Files\y_pred.csv', index=True, header=True)
print(y_pred_df)

chk = pd.concat([mdl_data_test, y_pred_df.set_index(mdl_data_test.index),
                 future_test_pc_df.set_index(mdl_data_test.index),
                 symbol_test_df.set_index(mdl_data_test.index)], axis=1)

# chk.to_csv(r'Files\chk.csv', index=True, header=True)

# print(chk)


# Make predictions from the 3 neural net models
train_pred1 = model_1.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2 = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3 = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds = np.mean(np.hstack((test_pred1, test_pred2, test_pred3)), axis=1)
print(test_preds[-5:])

#################################################################################################################
# Section 4.4 - Comparing the models
#################################################################################################################


#################################################################################################################
# Section 5 - Top 30 cases from each model versus using Max Drawdown versus Max Sharpe Ratio
#################################################################################################################

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

# Calculate expected returns mu from the set of stock prices dataset
mu = expected_returns.mean_historical_return(stock_prices)

# Calculate the covariance matrix S
Sigma = risk_models.sample_cov(stock_prices)

# Obtain the efficient frontier
ef = EfficientFrontier(mu, Sigma)
print(mu, Sigma)

# Calculate weights for the maximum Sharpe ratio portfolio
raw_weights_maxsharpe = ef.max_sharpe()
cleaned_weights_maxsharpe = ef.clean_weights()
print(raw_weights_maxsharpe, cleaned_weights_maxsharpe)

# Historical drawdown
# Calculate the running maximum
running_max = np.maximum.accumulate(cum_rets)

# Ensure the value never drops below 1
running_max[running_max < 1] = 1

# Calculate the percentage drawdown
drawdown = (cum_rets) / running_max - 1

# Plot the results
drawdown.plot()
plt.show()

#################################################################################################################
# Section 6 - Implementation of the model at Jan '21
#################################################################################################################
