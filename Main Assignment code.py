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
            mdl_input_data[output_col].replace([np.inf,-np.inf],0, inplace=True)

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
    'Symbol' , 'surprise', 'surprise_1Q_lag', 'surprise_2Q_lag', 'surprise_4Q_lag'
    ,'surprise_1Q_gth', 'surprise_2Q_gth', 'surprise_4Q_gth']])

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

# Create new features required for modelling i.e. P/E, gross margin, net margin etc.

# Profitability metrics
margin_calcs('grossProfit', 'totalRevenue', 'gross_margin')
margin_calcs('researchAndDevelopment', 'totalRevenue', 'r&d_margin')
margin_calcs('ebitda', 'totalRevenue', 'ebitda_margin')
margin_calcs('ebit', 'totalRevenue', 'ebit_margin')
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

mdl_input_data[['Industry', 'p_to_e', 'p_to_b', 'p_to_r', 'div_yield', 'debt_to_equity'
                'int_cov_ratio', 'gross_margin', 'ebitda_margin ]].groupby(by=['Industry','dt']).mean()

# Checks
print(mdl_input_data.loc[mdl_input_data['Symbol'] == 'AAIC', ['totalRevenue', 'totalRevenue_1Q_lag', 'totalRevenue_2Q_lag'
    ,'totalRevenue_4Q_lag','market_cap', 'market_cap_1Q_lag', 'market_cap_2Q_lag'
    ,'market_cap_4Q_lag', 'p_to_r', 'p_to_r_1Q_lag', 'p_to_r_2Q_lag', 'p_to_r_4Q_lag'
    ,'totalRevenue_1Q_gth', 'totalRevenue_2Q_gth', 'totalRevenue_4Q_gth'
    ,'p_to_r_1Q_gth', 'p_to_r_2Q_gth', 'p_to_r_4Q_gth'
    ,'netIncome', 'netIncome_1Q_lag', 'netIncome_2Q_lag', 'netIncome_4Q_lag'
    ,'netIncome_1Q_gth', 'netIncome_2Q_gth', 'netIncome_4Q_gth'
    , 'p_to_e', 'p_to_e_1Q_lag' , 'p_to_e_2Q_lag' , 'p_to_e_4Q_lag'
    ,'p_to_e_1Q_gth', 'p_to_e_2Q_gth', 'p_to_e_4Q_gth'
    ,'surprise', 'surprise_1Q_lag', 'surprise_2Q_lag', 'surprise_4Q_lag'
    ,'surprise_1Q_gth', 'surprise_2Q_gth', 'surprise_4Q_gth']])


ds = mdl_input_data[mdl_input_data.isin([np.inf, -np.inf])].sum()
print(ds)


##################################################################################################################
# Section 3.2 - Replacing nulls and Feature Selection
##################################################################################################################


# Split the data into train and test in order to prevent data leakage. Any decision made around dropping columns
# or replacing nulls needs to be completed assuming we have no information on the test set
mdl_data_train = mdl_input_data[mdl_input_data.index < '2019-07']
mdl_data_test = mdl_input_data[mdl_input_data.index == '2019-07']


# Drop any rows where fiscalDateEnding is null (we will have no revenue or profit information for these rows)
null_value_pc(mdl_data_train)
mdl_data_train = mdl_data_train.dropna(how='all', subset=['fiscalDateEnding'])
mdl_data_test = mdl_data_test.dropna(how='all', subset=['fiscalDateEnding'])

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
mdl_train_numeric = mdl_data_train.iloc[:,10:]
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
    corr = pd.DataFrame(mdl_train_numeric[[col,'future_price_gth']].corr())
    corr = corr.iloc[0,1]
    corr_list.append(corr)

print(len(col_list))
print(corr_list)
print(len(corr_list))

corr_df = pd.concat([pd.DataFrame(col_list,columns=['columns']), pd.DataFrame(corr_list
                                                                              ,columns=['Correlation'])], axis=1)

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
mdl_data_train.drop(['future_price','future_price_gth', 'fiscalDateEnding', 'reportedDate'], axis=1, inplace=True)
mdl_data_test.drop(['future_price','future_price_gth', 'fiscalDateEnding', 'reportedDate'], axis=1, inplace=True)

mdl_data_train.shape
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
mdl_data_train = mdl_data_train.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector'], axis=1)

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

# ax = plt.gca()
# sns.scatterplot(data = mdl_data_train, x= 'paymentsForRepurchaseOfPreferredStock', y='future_price_gth')
# ax.set_yscale('log')
# ax.set_xscale('log')

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

X_train_df = mdl_data_train.drop(['gt_10pc_gth'], axis=1)
X_train_df = pd.get_dummies(data=X_train_df, drop_first=True)
y_train_df = mdl_data_train['gt_10pc_gth']
X_test_df = mdl_data_test.drop(['gt_10pc_gth'], axis=1)
X_test_df = pd.get_dummies(data=X_test_df, drop_first=True)
y_test_df = mdl_data_test['gt_10pc_gth']

##
# y_train_df.to_csv(r'Files\y_train_df.csv', index=True, header=True)

X_train = X_train_df.values
y_train = y_train_df.values
X_test = X_test_df.values
y_test = y_test_df.values

y_train = y_train.reshape(len(y_train),1)  # For feature scaling you need a 2D array as this is what the StandardScaler expects
y_test = y_test.reshape(len(y_test),1)
print(y_train)



# Feature Scaling
# Scale the values such that each are in the range [0,1]
# Scaling is necessary for feature selection and modelling
sc_X_train = MinMaxScaler()
sc_X_test = MinMaxScaler()
X_train = sc_X_train.fit_transform(X_train)
X_test = sc_X_test.fit_transform(X_test)


# Impute missing values
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights = "uniform")
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

# Univariate Feature selection
select_feature = SelectKBest(chi2, k=100).fit(X_train, y_train)
select_features_df = pd.DataFrame({'Feature': list(X_train_df.columns),
                                   'Scores' : select_feature.scores_})
select_features_df.sort_values(by='Scores', ascending=False)

X_train_chi = select_feature.transform(X_train)
X_test_chi = select_feature.transform(X_test)

X_train_rv = X_train
y_train_rv = y_train.ravel()
X_test_rv = X_test
y_test_rv = y_test.ravel()
np.shape(X_train_rv)


# Recursive feature elimination
from sklearn.feature_selection import RFECV
rfecv= RFECV(estimator=RandomForestClassifier(), step=100, cv=5, scoring='precision')
rfecv= rfecv.fit(X_train_rv, y_train_rv)
print('Optimal number of features : ' , rfecv.n_features_)
print('Best features :' , X_train_df.columns[rfecv.support_])


plt.figure()
plt.xlabel("No of features selected")
plt.ylabel("Cross Validation score")
plt.plot(range(1,len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train_rv, y_train_rv)


from sklearn.metrics import confusion_matrix, accuracy_score , classification_report

y_pred = classifier.predict(X_test_rv)
cm = confusion_matrix(y_test_rv, y_pred)
print(cm)  # only 1 incorrect prediction
print(classification_report(y_test, y_pred))

print(y_test_rv)

# Random Forest
# Run a random forest to check what are the most important features in predicting future stock prices


# Grid Search

rf_class = RandomForestClassifier(criterion='entropy')
param_grid = {'n_estimators' : [200], 'max_features': ['auto', 'sqrt','log2']}

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='precision',
    n_jobs=-1,
    cv=5,
    refit=True, return_train_score=True)

print(grid_rf_class)

grid_rf_class.fit(X_train_rv, y_train_rv)


# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ["params"]]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df["rank_test_score"] == 1]
print(best_row)

# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Fit the best model
clf= RandomForestClassifier(criterion='entropy', max_features ='log2', n_estimators=200)
clf.fit(X_train_rv, y_train_rv)

y_pred = clf.predict(X_test_rv)
print(classification_report(y_test_rv, y_pred))




# Genetic Algorithms
# Assign the values outlined to the inputs
from tpot import TPOTClassifier

number_generations = 3  # nothing to do with the dataset
population_size = 4  # Start with 4 algorithms
offspring_size = 3
scoring_function = 'precision'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train_rv, y_train_rv)

# Score on the test set - 33.6%
tpot_clf.score(X_test_rv, y_test_rv)