###############################################################################################################
# Section 1 - Data Import -ctd
#
#         1.5 - Company Overview table
#         1.6 - Financial Results table
#         1.7 - Monthly Stock Prices
###############################################################################################################


###############################################################################################################
# Import the required packages and functions
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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
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
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tpot import TPOTClassifier
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout


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
    """A function for returning a pandas dataframe containing unique column for the input table supplied"""
    output = pd.DataFrame(input_table[column].unique(), columns=[column])
    return output


# Function for plotting the closing stock price versus the future stock price (stock price 6 months later)
# Given the large outliers (i.e. accounts with very large stock gains we have inserted a line of code to remove these
# cases from the plot)
@timer
def seaborn_lm_plt(input_table, close_val, future_value):
    """A function for plotting the closing stock price versus the future stock price"""

    # Seaborn table used in the plot removes any large movements which make the plot difficult to interpret
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


# Function used to create a table containing the number of missing values in each column of a table
@timer
def null_value_pc(table):
    """Function used to assess the number (and %) of missing values in each column of a specific table"""

    missing_tbl = pd.DataFrame(table.isnull().sum(), columns=['num missing'])
    missing_tbl['missing_pc'] = missing_tbl['num missing'] / table.shape[0]
    return missing_tbl


# Function used for margin calculations
@timer
def margin_calcs(input_num, input_den, output_col):
    """Function used to calculate different ratios (i.e. Price to Earnings) which are used in the modelling process"""

    # The ratios are calculated at the date, 1 quarter prior (3 months prior), 2 quarters prior 6 months prior
    # and 4 quarters prior (12 months prior).
    for j in [0, 1, 2, 4]:
        if j == 0:
            mdl_input_data[output_col] = mdl_input_data[input_num] / mdl_input_data[input_den]
            # Replace inf values with zero
            mdl_input_data[output_col].replace([np.inf, -np.inf], 0, inplace=True)

        else:
            mdl_input_data[output_col + '_' + str(j) + 'Q_lag'] = mdl_input_data[input_num + '_' + str(j) + 'Q_lag'] \
                                                                  / mdl_input_data[input_den + '_' + str(j) + 'Q_lag']

            # We are interested in the growth rate of these ratios
            mdl_input_data[output_col + '_' + str(j) + 'Q_gth'] = (mdl_input_data[output_col]
                                                                   - mdl_input_data[
                                                                       output_col + '_' + str(j) + 'Q_lag'])
            # Replace inf values with zero
            mdl_input_data[output_col + '_' + str(j) + 'Q_lag'].replace([np.inf, -np.inf], 0, inplace=True)
            # Replace inf values with zero
            mdl_input_data[output_col + '_' + str(j) + 'Q_gth'].replace([np.inf, -np.inf], 0, inplace=True)


# Function for dropping rows
@timer
def drop_row(tbl, lst):
    """Function used to delete rows in a table (tbl) when a column within the list are NULL"""
    output = tbl.dropna(subset=lst, how='any')
    return output


# Function for dropping columns
@timer
def drop_column(tbl, lst):
    """Function used to drop columns in a table """

    output = tbl.drop(lst, axis=1)
    return output


# Function which calculates the precision value used in our ANN


def precision_m(y_true, y_pred):
    """Manual calculation of precision which is required as our ANN metric"""

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


###############################################################################################################
# Section 1.5 - Company Overview
#
#              - We transform our company overview dataframe into a date dependent table containing each of
#                the dates required for our model
###############################################################################################################


# There are a further 445 stocks for which AlphaVantage does not have Company information on (please note
# 390 of these cases are ETF (exchange traded funds which we want to remove from our model)

# Original list of stocks in NYSE or NASDAQ                                                 =  7,377
# Minus Warrants, Notes, Debentures and Preferred Shares                                     - 1,013
# Minus ETFs which do not contain company info and should not be included in our model       -   390
# Minus 55 stocks for which most (e.g. WSO/B and HVT/A are preferred stocks)                 -    55
# Equals the total number of stocks in our Company overview file (overview_df)              =  5,919

# Only use stocks for which we have information on earnings, balance sheet info etc.         -   837
# Final number of stocks used in our model                                                  =  5,082

# # Display 1000 columns as a default
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)

# Import the stock data CSVs into Python
# Please update the below links to point to the location where you have saved these files
# Please note the BS_data file has been split into two on GITHUB due to the file size
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

# We only want to keep stocks which are contained in each file, in order to achieve this we  run an
# inner join on each of the datasets containing the unique stock symbols
# There are 5,082 stocks which are in every file, this is the list of stocks we will use going forward
symbols_in_all_files = \
    pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(co_symbol_unique, eps_symbol_unique, how='inner', on='Symbol')
                                        , inc_st_symbol_unique, how='inner', on='Symbol')
                               , bs_symbol_unique, how='inner', on='Symbol')
                      , cf_symbol_unique, how='inner', on='Symbol')
             , sp_symbol_unique, how='inner', on='Symbol')

symbols_in_all_files.shape  # Check complete - there are 5,082 stocks in this table

# Update the company overview such that it will contain only the stocks for which we have earnings, income and
# balance sheet information on

company_overview_upd = pd.merge(company_overview, symbols_in_all_files, how='inner', on='Symbol')
company_overview_upd.shape  # Check - updated file contains 5,082 stocks as expected

# The modelling approach we are using in this assignment is a 6 month hold and sell strategy in which we will purchase
# the stocks at Jan and July and sell the stocks 6 months later (i.e. if we purchase the stocks at July '18 we
# will sell these stocks at Jan '19)


# We will create a dataframe which contains just an index of dates (which are spaced by 6 months) which start at July
# '17 and end at Jan '20. This represents the shape required to create our model

dates = ['2021-01', '2020-07', '2020-01', '2019-07', '2019-01', '2018-07', '2018-01', '2017-07']
dates_df = pd.DataFrame(dates, columns=['dt'])
dates_df.index = pd.to_datetime(dates_df['dt']).dt.to_period('M')
dates_df = dates_df.drop(columns=dates_df.columns[0])  # drop the second dt field
print(dates_df)


# We transform our company overview dataframe into a date dependent table containing each of the dates required
# for our model (essentially multiplying the number of rows in the table by 8)

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
# Section 1.6 - Financial Results data including earnings, income statement, balance sheet and cash flow statement
##################################################################################################################

# Create a financial results dataset which is a merge of earnings, income statement, balance sheet and
# Cash flow statement
financial_results = \
    pd.merge(pd.merge(pd.merge(eps_data, inc_st_data, how='left', on=['fiscalDateEnding', 'Symbol'])
                      , bs_data.drop(labels=['reportedCurrency'], axis=1), how='left',
                      on=['fiscalDateEnding', 'Symbol'])
             , cf_data.drop(labels=['netIncome', 'reportedCurrency'], axis=1), how='left',
             on=['fiscalDateEnding', 'Symbol'])

financial_results.head(40)
print(financial_results)

# Initially checked the data for duplicate columns (these are flagged as _x and _y after which I re-run the
# join removing these columns to avoid the duplication.

eps_data.shape  # 7 columns (109,888 rows)
inc_st_data.shape  # 27 columns
bs_data.shape  # 39 columns
cf_data.shape  # 30 columns
financial_results.shape  # 94 columns ( 7 + 25 (27-2 columns joining on) + 36 (39-2-1(rep currency)) + 26 (30-2-2) )
# 109,888 rows (equal to the number of rows in the eps_data table


columns_df = pd.DataFrame(financial_results.columns, columns=['Columns'])
columns_df = columns_df.sort_values(by='Columns')
print(columns_df)  # There are no longer any duplicate columns

# Check
financial_results.info()  # Every field is saved as a character field (numeric and date fields will have to be updated)
financial_results.describe()  # No numeric fields so the output is not useful
financial_results.isnull().sum()  # There are a large number of nulls in the dataset

# Replace 'None' with NaN. We will reorder the dataframe to keep the character and date fields to the left and the
# numeric fields to the right, this will make data manipulation a lot easier
financial_results = financial_results.replace('None', np.nan)
symb_curr_df = pd.DataFrame(financial_results[['Symbol', 'reportedCurrency']])

financial_results_reorder = pd.concat(
    [symb_curr_df, financial_results.drop(labels=['Symbol', 'reportedCurrency'], axis=1)]
    , axis=1)

financial_results_reorder.info()  # reordering looks to be correct
financial_results_reorder.shape  # No change in the number of rows (109,888) and columns (94)

# Convert the numeric fields which are currently stored as character fields to floats
financial_results_reorder.iloc[:, 4:] = \
    financial_results_reorder.iloc[:, 4:].astype(float)

financial_results_reorder.info()  # Each of the numeric fields have been converted to floats
financial_results_reorder.shape  # No change in the number of rows (109,888) and columns (94)

# Create a new column called Book_value which is equal to Assets less liabilities
financial_results_reorder['book_value'] = financial_results_reorder['totalAssets'] \
                                          - financial_results_reorder['totalLiabilities']

financial_results_reorder.shape  # No extra rows added (109,888) and one extra column (95)

# Convert the Date fields to dates
financial_results_reorder['fiscalDateEnding'] = pd.to_datetime(financial_results_reorder['fiscalDateEnding']) \
    .dt.to_period('M')

financial_results_reorder['reportedDate'] = pd.to_datetime(financial_results_reorder['reportedDate']).dt.to_period('M')

# Similar to the transformation performed on the Company overview tables above
# we will update the financial results table to ensure it is in the correct format
# for our model

# For the Jan row of our data we will require the most recent Financial information for Jan (the same applies for July)
# The most recent information at Jan each year will be the quarterly results pre Jan (i.e. the most recent reported
# EPS for symbol A at Jan '21 is the reported EPS for A on the 23rd of November 2020).
# Below we update the report date to ensure that we bring in the most up-to-date financial information on each stock
# ensuring that the stock information is known at that date.

# If the report date is in Dec or Nov and the subsequent report date is not Jan we update the reported date
# to be Jan of the following year

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

# If the report date is in Apr or May and the subsequent report date is not July we update the reported date
# to be July for that case

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

# Create a date field called 'dt' and assign it as the index - this date field contains up-to-date financial
# information on each stock at Jan and Dec each year
financial_results_reorder['dt'] = pd.to_datetime(financial_results_reorder['dt_str']).dt.to_period('M')
financial_results_reorder.index = financial_results_reorder['dt']

financial_results_reorder.head(40)

financial_results_reorder.shape  # No change in the number of rows (109,888) but there are 4 extra columns (99)
financial_results_reorder.columns  # 4 columns which are not required, these are the last 4 columns

financial_results_reorder = financial_results_reorder.drop(columns=financial_results_reorder.columns[[-1, -2, -3, -4]])
financial_results_reorder.head(5)
financial_results_reorder.shape  # the 4 columns have been dropped (95) and no change to the number of rows (109,888)
financial_results_reorder.columns

# The purpose of the below code is to get the prior month, three month and twelve month financial information
# on each stock.
# For modelling purposes this will allow us to assess the growth rate impact of these financial fields on future stock
# price movement (i.e. knowing the most recent revenue figure for Apple at July '19 might not tell us a lot about what
# the stock price will be at Jan '20 but knowing the six month and twelve month revenue growth may give us a better
# indication)


# Create a list called "columns" containing the column names of all the numeric fields in the
# financial_results_reorder dataset
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

# Check - ensure that the lag and growth calculation are correct
print(financial_results_reorder.loc[(financial_results_reorder['Symbol'] == 'AG') |
                                    (financial_results_reorder['Symbol'] == 'STC'), [
                                        'Symbol', 'totalRevenue', 'totalRevenue_1Q_lag', 'totalRevenue_2Q_lag',
                                        'totalRevenue_4Q_lag', 'totalRevenue_1Q_gth', 'totalRevenue_2Q_gth',
                                        'totalRevenue_4Q_gth']])

financial_results_reorder.head(30)
financial_results_reorder.shape  # Number of rows are unchanged at 109,888 but we now have 641 columns
# The number of columns equals 91 (the number of numeric columns) * 7 (3 lagged fields and 3 growth fields)
#  + 4 (character fields) = 641

# We only have 8 timeframes which we are modelling on so we can delete all other time points
# These timepoints are saved in the dates_df
financial_results_reorder = pd.merge(financial_results_reorder, dates_df, left_index=True, right_index=True)
financial_results_reorder.head(20)
financial_results_reorder.tail(20)

financial_results_reorder.index.unique()  # 8 unique timepoints as expected
financial_results_reorder.shape  # the number of columns are unchanged at 641 as expected
# the number of rows has decreased to 35,564


##################################################################################################################
# Section 1.7 - Bring in the monthly stock prices
##################################################################################################################

# Check the top 20 entries in the table
# The only two columns we are interested in is "dt" and "5. adjusted close"
monthly_stock_prices.head(20)
monthly_stock_prices.info()

# Convert the "dt" field into a date type (similar to the other two tables)
monthly_stock_prices['dt_m'] = pd.to_datetime(monthly_stock_prices['dt'], format="%d/%m/%Y").dt.to_period('M')

# Create a new column called "close_price" and leave it equal to "5. adjusted close".
monthly_stock_prices['close_price'] = monthly_stock_prices["5. adjusted close"].astype(float)
stock_prices = monthly_stock_prices[['dt', 'dt_m', 'Symbol', 'close_price']]
stk_prices = copy.deepcopy(stock_prices)

print(stk_prices)

# Create a month column which represents the month we invest in our portfolio
stk_prices['month'] = stk_prices['dt_m'].dt.month
stk_prices['month'] = stk_prices['month'].apply(lambda x: calendar.month_abbr[x])
print(stk_prices)

# for loop which generates the one month three month, six month and twelve month lag share price
# as well as the five and six month future share price
# The future share price will be our modelled future price and the lagged share prices will be used to
# asses the relationship between historic share price growths and future share price movement
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
stk_prices[['dt', 'dt_m', 'Symbol', 'close_price', 'close_price_1M_lag', 'close_price_3M_lag', 'close_price_12M_lag'
    , 'close_price_-5M_lag', 'close_price_-6M_lag']].tail(25)  # Calculation is correct
stk_prices.columns
stk_prices.describe()  # There are no negative stock prices but the max stock price looks very high

chk_tbl = stk_prices.sort_values(by=['close_price'], ascending=False)
chk_tbl.head(10)  # The values look correct when checked against Yahoo finance for the timepoint

# Create the forecasted stock price - please note we have only 5 months of forecasted stock information
# for trades made in Jan '21 so the future stock price at that date will be 5 months in the future
# We create the target variable for our model, "gt_10pc_gth" which is 1 if the stock price increased by 10%
# in the 6 months and 0 if the stock price did not.
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
stk_prices['gt_10pc_gth'].value_counts()  # 38% of cases increased by 10% in the 6 months (a small class imbalance)

# Drop unneeded columns
stk_prices = stk_prices.drop(['dt_m', 'close_price_-5M_gth', 'close_price_-6M_gth',
                              'close_price_-5M_lag', 'close_price_-6M_lag'], axis=1)
stk_prices.head(50)
stk_prices.columns

# Keep a copy of the monthly stock price info
stk_prices_all_months = copy.deepcopy(stk_prices)

# Keep only the dates which are required
stk_prices = pd.merge(stk_prices, dates_df, left_index=True, right_index=True)
print(stk_prices)
stk_prices.index.unique()  # 8 unique timepoints as expected
stk_prices.head(20)

# Plot the close price against the forecasted price (6 months later)
# Looking at the plots possibly investing in stocks at July for a 6 month period is more
# profitable than investing in Jan.

# function(table,close_price to be lower than, future price to be lower than)
seaborn_lm_plt(stk_prices, 10, 50)  # cases which have a price of less than $10 and a future price less than $50
seaborn_lm_plt(stk_prices, 100, 500)
seaborn_lm_plt(stk_prices, 5, 20)
seaborn_lm_plt(stk_prices, 5, 1000000)  # Some large stock price increases make it difficult to see the overall impact


# Code for checking the stocks with the largest 6 month gains on companies who had a share price of less than 5 euro
# GameStop's (GME) share price increased from €4.01 in July 2020 to €320.99 in Jan '21
# This is an outlier as the share increase was not a result of the fundamentals of the company.
a = stk_prices.loc[stk_prices['close_price'] < 5]
b = copy.deepcopy(a)
b['diff'] = b['future_price'] - b['close_price']
print(b.sort_values(by=['diff'], ascending=False))

##################################################################################################################
# Section 2 - Creation of new modelled features
##################################################################################################################

# Join the three tables together by index
#      - company_overview_dt (table containing the name, country, Sector and Industry of the companies)
#      - financial_results_reorder (table containing the financial results (income statement, bs etc.)
#      - stk_prices (table containing the monthly stock prices of each company)


mdl_data = \
    pd.merge(pd.merge(company_overview_dt, financial_results_reorder, how='left', on=['dt', 'Symbol'])
             , stk_prices, how='left', on=['dt', 'Symbol'])

company_overview_dt.shape  # 40,656 rows and 8 columns
financial_results_reorder.shape  # 35,564 rows and 641 columns
stk_prices.shape  # 38,078 rows and 14 columns
mdl_data.shape  # 40,656 rows and 661 columns (641 + 8 + 14 - 2 (Symbol which we are joining on))

# Create a new table which is a copy of mdl_data
mdl_input_data = mdl_data

# Create a market cap column (including lagged values)
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

# Compare the key ratios against the industry average for that time point

# Create the industry average ratios
industry_avg = mdl_input_data[['Industry', 'p_to_e', 'p_to_b', 'p_to_r', 'ev_to_e', 'ev_to_b', 'ev_to_r',
                               'div_yield', 'debt_to_equity', 'p_to_op_cf', 'ev_to_op_cf', 'p_to_ebitda',
                               'ev_to_ebitda',
                               'int_cov_ratio', 'gross_margin', 'ebitda_margin', 'ebit_margin', 'op_cf', 'market_cap',
                               'totalRevenue_2Q_gth', 'netIncome_2Q_gth', 'reportedEPS', 'surprisePercentage']] \
    .groupby(by=['Industry', 'dt']).mean()

industry_avg_rename = industry_avg.rename(columns=lambda s: s + '_ind_avg')
print(industry_avg_rename)

# Join the industry averages on to the overall dataset
mdl_input_data_upd = pd.merge(mdl_input_data, industry_avg_rename, how='left', on=['dt', 'Industry'])

# Create a field which compares the stock's ratios against the industry average
# Drop the industry average field.
for col1 in list(industry_avg):
    mdl_input_data_upd[col1 + '_v_ind_avg'] = mdl_input_data_upd[col1] - mdl_input_data_upd[col1 + '_ind_avg']
    mdl_input_data_upd.drop([col1 + '_ind_avg'], axis=1, inplace=True)

# Boxplot which can be used to view the population of a variable which had a share price growth of greater than
# 10% versus the population that did not
sns.boxplot(data=mdl_input_data_upd, x='gt_10pc_gth', y='ebitda_margin_v_ind_avg', whis=10)
plt.yscale('log')
plt.show()


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
# Section 3 - Data Preparation
##################################################################################################################

# Split the data into train, test and deploy datasets in order to prevent data leakage.
# Any decision made around dropping columns or replacing nulls needs to be completed assuming we have
# no information on the test or deploy dataset
mdl_data_train = mdl_input_data_upd[mdl_input_data_upd.index < '2020-07']
mdl_data_test = mdl_input_data_upd[mdl_input_data_upd.index == '2020-07']
mdl_deploy = mdl_input_data_upd[mdl_input_data_upd.index == '2021-01']

# Drop any rows where fiscalDateEnding, Net Income, revenue or where the target value is NULL, drop these rows in
# each of the datasets
drop_list = ['fiscalDateEnding', 'totalRevenue', 'netIncome', 'gt_10pc_gth']
mdl_data_train = drop_row(mdl_data_train, drop_list)
mdl_data_test = drop_row(mdl_data_test, drop_list)
mdl_deploy = drop_row(mdl_deploy, drop_list)

# Check the Null values in the train dataset
null_value_pc(mdl_data_train)

# Drop the reportedCurrency, fiscalDateEnding and reportedDate columns
drop_col = ['reportedCurrency', 'fiscalDateEnding', 'reportedDate']
mdl_data_train = drop_column(mdl_data_train, drop_col)
mdl_data_test = drop_column(mdl_data_test, drop_col)
mdl_deploy = drop_column(mdl_deploy, drop_col)

mdl_data_train.shape

# Remaining null values
null_df = null_value_pc(mdl_data_train)  # there are a large number of Null values to deal with in all but 6 columns
null_df = null_df.sort_values(by='missing_pc')
print(null_df)

# The below section assess the correlation between the numeric feature columns and the target columns in the training
# data only

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

# Pairplots give a visual representation of the correlation between the feature variables
# sns.pairplot(mdl_data_train.iloc[:, 10:14].fillna(0))


# After reviewing the above there is a high degree of correlation within a number of our feature variables, there is
# also a number of feature variables which have a very low correlation with the target variable. The columns which
# have a high correlation with the target variable are features we engineered.
# We will assess what columns we should drop from the dataset later in the code

# Sector
mdl_data_train['Sector'].unique()  # nan and 'None' in the column
mdl_data_train['Sector'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_train['Sector'].unique()  # no nan or 'None' values in the column
mdl_data_train['Sector'].isnull().sum()  # 0 value returned

mdl_data_test['Sector'].unique()  # nan and 'None' in the column
mdl_data_test['Sector'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_test['Sector'].unique()  # no nan or 'None' values in the column
mdl_data_test['Sector'].isnull().sum()  # 0 value returned

mdl_deploy['Sector'].unique()  # nan and 'None' in the column
mdl_deploy['Sector'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_deploy['Sector'].unique()  # no nan or 'None' values in the column
mdl_deploy['Sector'].isnull().sum()  # 0 value returned

# Industry
mdl_data_train['Industry'].isnull().sum()  # 14 missing values
mdl_data_train['Industry'].unique()  # 'None' values in the column
mdl_data_train['Industry'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_train['Industry'].unique()  # no 'None' values in the column
mdl_data_train['Industry'].isnull().sum()  # 0 value returned

mdl_data_test['Industry'].isnull().sum()  # 6 missing values
mdl_data_test['Industry'].unique()  # 'None' values in the column
mdl_data_test['Industry'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_data_test['Industry'].unique()  # no 'None' values in the column
mdl_data_test['Industry'].isnull().sum()  # 0 value returned

mdl_deploy['Industry'].isnull().sum()  # 7 missing values
mdl_deploy['Industry'].unique()  # 'None' values in the column
mdl_deploy['Industry'].replace(to_replace=[np.nan, 'None'], value=['Unknown', 'Unknown'], inplace=True)
mdl_deploy['Industry'].unique()  # no 'None' values in the column
mdl_deploy['Industry'].isnull().sum()  # 0 value returned

null_value_pc(mdl_data_train)  # Sector and industry no longer have missing values

# We need to drop the 'future_price' and 'future_price_gth' from our model
future_test_pc_df = mdl_data_test[['future_price_gth']]
future_deploy_pc_df = mdl_deploy[['future_price_gth']]

drop_col = ['future_price_gth', 'future_price']
mdl_data_train = drop_column(mdl_data_train, drop_col)
mdl_data_test = drop_column(mdl_data_test, drop_col)
mdl_deploy = drop_column(mdl_deploy, drop_col)

mdl_data_train.shape
future_test_pc_df.head()

##

# Assess the character variables
print(pd.DataFrame(mdl_data_train.dtypes, columns=['datatype']).sort_values('datatype'))
mdl_data_train.info()
# useful for putting all of the character fields at the bottom of the print.

# There are 9 character fields before we get dummy values for these fields we need to look into them:
#    - Symbol has 4,205 unique values and Name has 4,075 unique values, we will drop these features as otherwise we will
#      be modelling at too low of a level
#    - Asset type, currency and country all only have one unique value and as such are redundant to the model
#    - Exchange(2) will be included in the model
#    - We will investigate if Industry (148) should be included or if Sector (13) gives us enough information
#    - about the stock

char_columns = ['Symbol', 'AssetType', 'Name', 'month', 'Exchange', 'Currency', 'Country', 'Sector', 'Industry']
unique_vals = []
for entry in char_columns:
    cnt_entry_i = unique_column(mdl_data_train, entry).shape[0]
    unique_vals.append([entry, cnt_entry_i])

print(unique_vals)

# The variability in the average return within a Sector is high.
# We will drop sector from our model and keep Industry
mdl_data_train[['Sector', 'Industry', 'gt_10pc_gth']].groupby(by=['Sector', 'Industry']).mean()

# Drop the required columns in a new dataframe called "model_input_data"
symbol_test_df = mdl_data_test[['Symbol']]
symbol_deploy_df = mdl_deploy[['Symbol']]

mdl_data_train = mdl_data_train.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector', 'month'], axis=1)
mdl_data_test = mdl_data_test.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector', 'month'], axis=1)
mdl_deploy = mdl_deploy.drop(['Symbol', 'AssetType', 'Name', 'Currency', 'Country', 'Sector', 'month'], axis=1)

print(pd.DataFrame(mdl_data_train.dtypes, columns=['datatype']).sort_values('datatype'))  # 3 character fields remaining

##################################################################################################################
# Section 3 Data prep ctd - Removing Nulls, feature scaling, dimensionality reduction
##################################################################################################################


# Create the feature variable dataframes X and the target y

X_train_df = mdl_data_train.drop(['gt_10pc_gth'], axis=1)
X_train_df = pd.get_dummies(data=X_train_df, drop_first=True)
y_train_df = mdl_data_train['gt_10pc_gth']

X_test_df = mdl_data_test.drop(['gt_10pc_gth'], axis=1)
X_test_df = pd.get_dummies(data=X_test_df, drop_first=True)
X_test_df = X_test_df.drop(['Industry_Oil & Gas Pipelines'], axis=1)
y_test_df = mdl_data_test['gt_10pc_gth']

X_deploy_df = mdl_deploy.drop(['gt_10pc_gth'], axis=1)
X_deploy_df = pd.get_dummies(data=X_deploy_df, drop_first=True)
y_deploy_df = mdl_deploy['gt_10pc_gth']

X_train_df.shape
X_test_df.shape
X_deploy_df.shape
y_train_df.shape
y_test_df.shape
y_deploy_df.shape

# Convert the dataframes to a numpy array

X_train = X_train_df.values
y_train = y_train_df.values
X_test = X_test_df.values
y_test = y_test_df.values
X_deploy = X_deploy_df.values
y_deploy = y_deploy_df.values

# Reshape the target variables
y_train = y_train.reshape(len(y_train),
                          1)  # For feature scaling you need a 2D array as this is what the StandardScaler expects
y_test = y_test.reshape(len(y_test), 1)
y_deploy = y_deploy.reshape(len(y_deploy), 1)
print(y_train)

# Feature Scaling
# Scale the values such that each are in the range [0,1]
# Scaling is necessary for feature selection and modelling
sc = MinMaxScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_deploy = sc.transform(X_deploy)

##################################################################################################################
 # Please note the KNNImputer is very slow
##################################################################################################################

# Impute missing values - given time constraints took the default value of 5
# It would be worth investigating the optimal value for KNNImputer.
# Please note the below code takes a long time to run, the results have been written out and saved on GITHUB as
# a result so you do not have to run this
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights = "uniform")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_deploy = imputer.transform(X_deploy)



X_train_rv = X_train
y_train_rv = y_train.ravel()
X_test_rv = X_test
y_test_rv = y_test.ravel()
X_deploy_rv = X_deploy
y_deploy_rv = y_deploy.ravel()
np.shape(X_train_rv)
np.shape(X_test_rv)
np.shape(X_deploy_rv)
np.shape(y_train_rv)
np.shape(y_test_rv)
np.shape(y_deploy_rv)

# X_train_rv_df = pd.DataFrame(X_train_rv)
# y_train_rv_df = pd.DataFrame(y_train_rv)
# X_test_rv_df = pd.DataFrame(X_test_rv)
# y_test_rv_df = pd.DataFrame(y_test_rv)
# X_deploy_rv_df = pd.DataFrame(X_deploy_rv)
# y_deploy_rv_df = pd.DataFrame(y_deploy_rv)

# X_train_rv_df.to_csv(r'Files\X_train_rv_df.csv', index=False, header=True)
# y_train_rv_df.to_csv(r'Files\y_train_rv_df.csv', index=False, header=True)
# X_test_rv_df.to_csv(r'Files\X_test_rv_df.csv', index=False, header=True)
# y_test_rv_df.to_csv(r'Files\y_test_rv_df.csv', index=False, header=True)
# X_deploy_rv_df.to_csv(r'Files\X_deploy_rv_df.csv', index=False, header=True)
# y_deploy_rv_df.to_csv(r'Files\y_deploy_rv_df.csv', index=False, header=True)

##################################################################################################################

# Import the results of KNN Imputer

# Please note these files have been provided
# X_train_rv_df = pd.read_csv(r'Files\X_train_rv_df.csv')
# y_train_rv_df = pd.read_csv(r'Files\y_train_rv_df.csv')
# X_test_rv_df = pd.read_csv(r'Files\X_test_rv_df.csv')
# y_test_rv_df = pd.read_csv(r'Files\y_test_rv_df.csv')
# X_deploy_rv_df = pd.read_csv(r'Files\X_deploy_rv_df.csv')
# y_deploy_rv_df = pd.read_csv(r'Files\y_deploy_rv_df.csv')

# Convert to numpy arrays
# X_train_rv = X_train_rv_df.values
# y_train_rv = y_train_rv_df.values.ravel()
# X_test_rv = X_test_rv_df.values
# y_test_rv = y_test_rv_df.values.ravel()
# X_deploy_rv = X_deploy_rv_df.values
# y_deploy_rv = y_deploy_rv_df.values.ravel()

# Recursive feature elimination
# Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
# step is the number of features removed at each iteration - this is purely for information purposes
# rfecv = RFECV(estimator=RandomForestClassifier(), step=100, cv=5, scoring='f1')
# rfecv = rfecv.fit(X_train_rv, y_train_rv)
# print('Optimal number of features : ', rfecv.n_features_)
# print('Best features :', X_train_df.columns[rfecv.support_])

# Plot the change in f1 score as a result of removing features
##
# plt.figure()
# plt.xlabel("No of model features removed (in 00's)")
# plt.ylabel("Cross Validation score")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.title("Recursive Feature Elimination")
# plt.show()

# Univariate Feature selection
# Another approach which was not used in this model is to reduce the number of features by only  selecting the best
# K number of features (determined based on the features chi-squared)
# select_feature = SelectKBest(chi2, k=1000).fit(X_train, y_train)
# select_features_df = pd.DataFrame({'Feature': list(X_train_df.columns),
#                                    'Scores' : select_feature.scores_})
#
# X_train = select_feature.transform(X_train)
# X_test = select_feature.transform(X_test)
##

# Adopt PCA to reduce the number of columns to 100
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_rv)
X_test_pca = pca.transform(X_test_rv)
X_deploy_pca = pca.transform(X_deploy_rv)
pca.explained_variance_ratio_.cumsum()  # 88% of the variance is retained

y_train_pca = y_train_rv
y_test_pca = y_test_rv
y_deploy_pca = y_deploy_rv




#################################################################################################################
# Section 4 - Modelling
#         4.1 - Grid Search to find the best hyperparameters to use for RF, KNN, XGBoost and CatBoost
#         4.2 - Random Forest
#         4.3 - Stacked Model
#         4.4 - Genetic Algorithm
#         4.5 - Neural Networks
#         4.6 - Min Drawdown and diversification
#################################################################################################################


#################################################################################################################
#  Section 4.1 - GridSearchCV
#################################################################################################################

# Model Parameters run through the GridSearch CV
# Please note with more processing power and less of a time constraint I would be looking to run more parameters
# through the Grid
model_params = {

    'random_forest': {'model': RandomForestClassifier(random_state=1),
                      'params': {'criterion': ['gini', 'entropy'],
                                 'n_estimators': [5, 100, 200, 500], 'max_features': ['auto', 'log2'],
                                 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}},

    'svm' : {'model': SVC(random_state=1),
             'params' : {'C': [0.25, 0.5, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.3, 0.5, 0.9]}},

    'knn': {'model': KNeighborsClassifier(algorithm='kd_tree'),
            'params': {'n_neighbors': [5, 10, 15, 25, 50, 100]}}
}

print(model_params)

scores = []
all_scores = []

# Fit the model_params to the GridSearch below.

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], n_jobs=-1, scoring='precision', cv=5,
                       return_train_score=True, verbose=2)
    clf.fit(X_train_pca, y_train_pca)
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
# random_forest    0.378879  {'criterion': 'gini', 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 10,
#                             'n_estimators': 500}
# svm              0.535546  {'C': 0.25, 'gamma': 0.3, 'kernel': 'rbf'}
# knn              0.464635  {'n_neighbors': 100}


# # Grid searchCV for XGBoost
# Please note for using XGBoost you should use the DMatrix, for this project I have not to keep it a little bit easier
# to follow

xgb_params = {'learning_rate': [0.05, 0.1, 0.3, 0.5], 'max_depth': [3, 6, 9, 15],
              'gamma': [0, 1, 5], 'min_child_weight': [1, 3, 5]}

classifier = XGBClassifier()
grid_search = GridSearchCV(classifier, param_grid=xgb_params, cv=5, n_jobs=-1, verbose=3)
grid_search.fit(X_train_pca, y_train_pca)

print(grid_search.best_params_)
# {'gamma': 5, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1}

#################################################################################################################
# Section 4.2 - Random forest model
#################################################################################################################

# Random Forest Classifier  - using the optimal hyperparameters post using GridSearchCV

rf_cf = RandomForestClassifier(criterion= 'gini', max_features='log2', min_samples_leaf=4, \
                                min_samples_split=10, n_estimators=500, random_state=1)


# Fit the model
rf_cf.fit(X_train_pca, y_train_pca)

# Assess the performance of the model on the test data
y_rf_pred = rf_cf.predict(X_test_pca)
cm_rf = confusion_matrix(y_test_pca, y_rf_pred)
print(cm_rf)
print(classification_report(y_test_pca, y_rf_pred))

# Assess the performance of the model on the deployment data
y_rf_dep_pred = rf_cf.predict(X_deploy_pca)
cm_rf = confusion_matrix(y_deploy_pca, y_rf_dep_pred)
print(cm_rf)
print(classification_report(y_deploy_pca, y_rf_dep_pred))

# Random Forest model - Test dataset returns
rf_test_probs = rf_cf.predict_proba(X_test_pca)
rf_test_probs_df = pd.DataFrame(rf_test_probs[:, 1], columns=['rf_mdl_prob'])

rf_test_mdl_results = pd.concat([symbol_test_df,
                                 mdl_data_test[['Industry', 'gt_10pc_gth']].set_index(symbol_test_df.index),
                                 future_test_pc_df.set_index(symbol_test_df.index),
                                 rf_test_probs_df.set_index(symbol_test_df.index)], axis=1)

rf_test_mdl_results.sort_values(by=['rf_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

rf_test_mdl_results['Top30_ret'] = rf_test_mdl_results.iloc[:30]['future_price_gth'].mean()
rf_test_mdl_results['Top100_ret'] = rf_test_mdl_results.iloc[:100]['future_price_gth'].mean()
print(rf_test_mdl_results)

# Random Forest model - deploy dataset returns
rf_deploy_probs = rf_cf.predict_proba(X_deploy_pca)
rf_deploy_probs_df = pd.DataFrame(rf_deploy_probs[:, 1], columns=['rf_mdl_prob'])

rf_deploy_mdl_results = pd.concat([symbol_deploy_df,
                                   mdl_deploy[['Industry', 'gt_10pc_gth']].set_index(symbol_deploy_df.index),
                                   future_deploy_pc_df.set_index(symbol_deploy_df.index),
                                   rf_deploy_probs_df.set_index(symbol_deploy_df.index)], axis=1)

rf_deploy_mdl_results.sort_values(by=['rf_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

rf_deploy_mdl_results['Top30_ret'] = rf_deploy_mdl_results.iloc[:30]['future_price_gth'].mean()
rf_deploy_mdl_results['Top100_ret'] = rf_deploy_mdl_results.iloc[:100]['future_price_gth'].mean()
print(rf_deploy_mdl_results)

#################################################################################################################
# Section 4.3 - Stacking the models
#################################################################################################################

# Define the base models
level0 = list()

level0.append(('knn', KNeighborsClassifier(algorithm='kd_tree', n_neighbors=100)))
level0.append(('r_forest', RandomForestClassifier(criterion= 'gini', max_features='log2', min_samples_leaf=4, \
                                                  min_samples_split=10, n_estimators=500, random_state=1)))
level0.append(('nb', GaussianNB()))
level0.append(('svm', SVC(C=0.25, gamma=0.3, kernel='rbf', random_state=1)))
level0.append(('XGB', XGBClassifier(gamma=5, learning_rate=0.05, max_depth=3, min_child_weight=1 , n_jobs=-1
                                    , random_state=1)))


# define meta learner model
level1 = XGBClassifier(random_state=1)

# define the stacking ensemble
stk_mdl = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1, verbose=2)

# fit the model on all available data
stk_mdl.fit(X_train_pca, y_train_pca)

# Check the model on the test set
y_stk_pred = stk_mdl.predict(X_test_pca)
cm = confusion_matrix(y_test_pca, y_stk_pred)
print(cm)
print(classification_report(y_test_pca, y_stk_pred))

# Check the model on the deploy set
y_stk_dep_pred = stk_mdl.predict(X_deploy_pca)
cm = confusion_matrix(y_deploy_pca, y_stk_dep_pred)
print(cm)
print(classification_report(y_deploy_pca, y_stk_dep_pred))

# Stacked model - Test dataset returns

stkd_test_probs = stk_mdl.predict_proba(X_test_pca)
stkd_test_probs_df = pd.DataFrame(stkd_test_probs[:, 1], columns=['Stacked_mdl_prob'])

stkd_test_mdl_results = pd.concat([symbol_test_df,
                                   mdl_data_test[['Industry', 'gt_10pc_gth']].set_index(symbol_test_df.index),
                                   future_test_pc_df.set_index(symbol_test_df.index),
                                   stkd_test_probs_df.set_index(symbol_test_df.index)], axis=1)

stkd_test_mdl_results.sort_values(by=['Stacked_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

stkd_test_mdl_results['Top30_ret'] = stkd_test_mdl_results.iloc[:30]['future_price_gth'].mean()
stkd_test_mdl_results['Top100_ret'] = stkd_test_mdl_results.iloc[:100]['future_price_gth'].mean()
print(stkd_test_mdl_results)

# Stacked model - deploy dataset returns
stkd_deploy_probs = stk_mdl.predict_proba(X_deploy_pca)
stkd_deploy_probs_df = pd.DataFrame(stkd_deploy_probs[:, 1], columns=['Stacked_mdl_prob'])

stkd_deploy_mdl_results = pd.concat([symbol_deploy_df,
                                     mdl_deploy[['Industry', 'gt_10pc_gth']].set_index(symbol_deploy_df.index),
                                     future_deploy_pc_df.set_index(symbol_deploy_df.index),
                                     stkd_deploy_probs_df.set_index(symbol_deploy_df.index)], axis=1)

stkd_deploy_mdl_results.sort_values(by=['Stacked_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

stkd_deploy_mdl_results['Top30_ret'] = stkd_deploy_mdl_results.iloc[:30]['future_price_gth'].mean()
stkd_deploy_mdl_results['Top100_ret'] = stkd_deploy_mdl_results.iloc[:100]['future_price_gth'].mean()
print(stkd_deploy_mdl_results)

#################################################################################################################
# Section 4.4 - Genetic Algorithm
#################################################################################################################

# Genetic Algorithms
# Assign the values outlined to the inputs

number_generations = 5  # 5 generations of the algorithm
population_size = 20  # Start with 20 algorithms
offspring_size = 3
scoring_function = 'precision'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=1, cv=5)

# # Fit the classifier to the training data
tpot_clf.fit(X_train_pca, y_train_pca)

# # Score on the test set
tpot_clf.score(X_test_pca, y_test_pca)
# Best pipeline: MLPClassifier(input_matrix, alpha=0.1, learning_rate_init=0.001)


# Assess the performance of the model on the test dataset
y_ga_pred = tpot_clf.predict(X_test_pca)
cm = confusion_matrix(y_test_pca, y_ga_pred)
print(cm)
print(classification_report(y_test_pca, y_ga_pred))

# Assess the performance of the model on the deployment dataset
y_ga_dep_pred = tpot_clf.predict(X_deploy_pca)
cm = confusion_matrix(y_deploy_pca, y_ga_dep_pred)
print(cm)
print(classification_report(y_deploy_pca, y_ga_dep_pred))

# Genetic Algorithm model - Test dataset returns

ga_test_probs = tpot_clf.predict_proba(X_test_pca)
ga_test_probs_df = pd.DataFrame(ga_test_probs[:, 1], columns=['ga_mdl_prob'])

ga_test_mdl_results = pd.concat([symbol_test_df,
                                 mdl_data_test[['Industry', 'gt_10pc_gth']].set_index(symbol_test_df.index),
                                 future_test_pc_df.set_index(symbol_test_df.index),
                                 ga_test_probs_df.set_index(symbol_test_df.index)], axis=1)

ga_test_mdl_results.sort_values(by=['ga_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

ga_test_mdl_results['Top30_ret'] = ga_test_mdl_results.iloc[:30]['future_price_gth'].mean()
ga_test_mdl_results['Top100_ret'] = ga_test_mdl_results.iloc[:100]['future_price_gth'].mean()
print(ga_test_mdl_results)

# Genetic Algorithm model  - deploy dataset returns
ga_deploy_probs = tpot_clf.predict_proba(X_deploy_pca)
ga_deploy_probs_df = pd.DataFrame(ga_deploy_probs[:, 1], columns=['ga_mdl_prob'])

ga_deploy_mdl_results = pd.concat([symbol_deploy_df,
                                   mdl_deploy[['Industry', 'gt_10pc_gth']].set_index(symbol_deploy_df.index),
                                   future_deploy_pc_df.set_index(symbol_deploy_df.index),
                                   ga_deploy_probs_df.set_index(symbol_deploy_df.index)], axis=1)

ga_deploy_mdl_results.sort_values(by=['ga_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

ga_deploy_mdl_results['Top30_ret'] = ga_deploy_mdl_results.iloc[:30]['future_price_gth'].mean()
ga_deploy_mdl_results['Top100_ret'] = ga_deploy_mdl_results.iloc[:100]['future_price_gth'].mean()
print(ga_deploy_mdl_results)

#################################################################################################################
# Section 4.5 - Neural Networks
#################################################################################################################

# Artificial Neural Network
# Seed the network
tf.random.set_seed(1)

# Initializing the ANN
# Using a sequential neutral network
ann_1 = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# The first hidden layer contains 30 nodes from the Dense class and the activation function is the rectifier function
# Activation = max(x,0) (which is 0 for negative values and then increases linearly until 1)
ann_1.add(tf.keras.layers.Dense(units=30, activation='relu'))

# Adding the second hidden layer
# The second and third hidden layer contains 30 nodes and the activation function is the rectifier function
ann_1.add(tf.keras.layers.Dense(units=30, activation='relu'))
ann_1.add(tf.keras.layers.Dense(units=30, activation='relu'))

# Adding the output layer
# The output layer contains 1 node and the activation function is the sigmoid function
# Activation = 1 / (1 + e(-x)) (which is very useful when predicting probabilities)
# If we wanted three output variables the output layer would require 3 nodes/neurons
ann_1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the ANN

# Compiling the ANN
# Optimizer used is Adaptive Moment Estimation (Adam), the training cost in the case of Adam uses stochastic gradient
# descent.
# For binary outcomes we will use binary_crossentropy as the loss
ann_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_m])

# Training the ANN on the Training set
# We are doing batch learning, with a batch_size of 20
# epochs is the number of times we run over the data, in our case we run over the data 200 times
history_1 = ann_1.fit(X_train_pca, y_train_pca, batch_size=20, validation_split=0.2, epochs=200)


# Plot the precision by epoch
# plt.clf()
plt.plot(history_1.history['precision_m'], label='Actual')
plt.plot(history_1.history['val_precision_m'], label='Validation')
plt.title('NN1 Precision by Epoch')
plt.xlabel('# of epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()


# Fit the second Neural Network - has  2 hidden layers containing 200 and 100 nodes respectively
# which the second layer has a 50% dropout.
ann_2 = tf.keras.models.Sequential()
ann_2.add(tf.keras.layers.Dense(units=200, activation='relu'))
ann_2.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann_2.add(Dropout(0.5))
ann_2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_m])
history_2 = ann_2.fit(X_train_pca, y_train_pca, batch_size=50, validation_split=0.2, epochs=100)

# Plot the precision by epoch
# plt.clf()
plt.plot(history_2.history['precision_m'], label='Actual')
plt.plot(history_2.history['val_precision_m'], label='Validation')
plt.title('NN2 Precision by Epoch')
plt.xlabel('# of epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()


# Fit the third Neural Network - which has a 10% dropout on each layer and a different activation function
ann_3 = tf.keras.models.Sequential()
ann_3.add(tf.keras.layers.Dense(units=100, activation=tf.keras.activations.tanh))
ann_3.add(tf.keras.layers.Dense(units=30, activation=tf.keras.activations.tanh))
ann_3.add(Dropout(0.1))
ann_3.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.tanh))
ann_3.add(Dropout(0.1))
ann_3.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.1, nesterov=True)
ann_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_m])
history_3 = ann_3.fit(X_train_pca, y_train_pca, batch_size=100, validation_split=0.2, epochs=200)

# Plot the precision from the fitted models
# plt.clf()
plt.plot(history_3.history['precision_m'], label='Actual')
plt.plot(history_3.history['val_precision_m'], label='Validation')
plt.title('NN3 Precision by Epoch')
plt.xlabel('# of epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Assess the model on the test data
nn_test_prob = np.mean(np.hstack((ann_1.predict(X_test_pca), ann_2.predict(X_test_pca), ann_3.predict(X_test_pca))),
                       axis=1)
nn_test_prob_tf = (nn_test_prob > 0.5)

cm = confusion_matrix(y_test_pca, nn_test_prob_tf)
print(cm)
cr = classification_report(y_test_pca, nn_test_prob_tf)
print(cr)

# Assess the model on the deploy data
nn_deploy_prob = np.mean(np.hstack((ann_1.predict(X_deploy_pca), ann_2.predict(X_deploy_pca),
                                    ann_3.predict(X_deploy_pca))), axis=1)
nn_deploy_prob_tf = (nn_deploy_prob > 0.5)
cm = confusion_matrix(y_deploy_pca, nn_deploy_prob_tf)
print(cm)
cr = classification_report(y_deploy_pca, nn_deploy_prob_tf)
print(cr)

# Artificial Neural Networks model - Test dataset returns

nn_test_probs_df = pd.DataFrame(nn_test_prob, columns=['nn_mdl_prob'])

nn_test_mdl_results = pd.concat([symbol_test_df,
                                 mdl_data_test[['Industry', 'gt_10pc_gth']].set_index(symbol_test_df.index),
                                 future_test_pc_df.set_index(symbol_test_df.index),
                                 nn_test_probs_df.set_index(symbol_test_df.index)], axis=1)

nn_test_mdl_results.sort_values(by=['nn_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

nn_test_mdl_results['Top30_ret'] = nn_test_mdl_results.iloc[:30]['future_price_gth'].mean()
nn_test_mdl_results['Top100_ret'] = nn_test_mdl_results.iloc[:100]['future_price_gth'].mean()
print(nn_test_mdl_results)

# Artificial Neural Network model  - deploy dataset returns

nn_deploy_probs_df = pd.DataFrame(nn_deploy_prob, columns=['nn_mdl_prob'])

nn_deploy_mdl_results = pd.concat([symbol_deploy_df,
                                   mdl_deploy[['Industry', 'gt_10pc_gth']].set_index(symbol_deploy_df.index),
                                   future_deploy_pc_df.set_index(symbol_deploy_df.index),
                                   nn_deploy_probs_df.set_index(symbol_deploy_df.index)], axis=1)

nn_deploy_mdl_results.sort_values(by=['nn_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

nn_deploy_mdl_results['Top30_ret'] = nn_deploy_mdl_results.iloc[:30]['future_price_gth'].mean()
nn_deploy_mdl_results['Top100_ret'] = nn_deploy_mdl_results.iloc[:100]['future_price_gth'].mean()
print(nn_deploy_mdl_results)

#################################################################################################################
# Section 4.6 - Top 30 cases from each model versus using min Drawdown versus a fully diversified model
#################################################################################################################

# Join the Test and deploy data probabilities for each model
combined_mdl_test = \
        pd.merge(stkd_test_mdl_results[['Symbol', 'Industry', 'gt_10pc_gth', 'future_price_gth', 'Stacked_mdl_prob']]
        , nn_test_mdl_results[['Symbol', 'nn_mdl_prob']], how='left', on=['Symbol'])


combined_mdl_test['avg_mdl_prob'] = combined_mdl_test.iloc[:, 4:6].mean(axis=1)

combined_mdl_test.sort_values(by=['avg_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

combined_mdl_test['Top30_ret'] = combined_mdl_test.iloc[:30]['future_price_gth'].mean()
combined_mdl_test['Top100_ret'] = combined_mdl_test.iloc[:100]['future_price_gth'].mean()
combined_mdl_test = combined_mdl_test.iloc[:201]
print(combined_mdl_test)

# Deploy dataset
combined_mdl_deploy = \
        pd.merge(stkd_deploy_mdl_results[['Symbol', 'Industry', 'gt_10pc_gth', 'future_price_gth', 'Stacked_mdl_prob']]
        , nn_deploy_mdl_results[['Symbol', 'nn_mdl_prob']], how='left', on=['Symbol'])

combined_mdl_deploy['avg_mdl_prob'] = combined_mdl_deploy.iloc[:, 4:6].mean(axis=1)

combined_mdl_deploy.sort_values(by=['avg_mdl_prob'], inplace=True, ignore_index=True, ascending=False)

combined_mdl_deploy['Top30_ret'] = combined_mdl_deploy.iloc[:30]['future_price_gth'].mean()
combined_mdl_deploy['Top100_ret'] = combined_mdl_deploy.iloc[:100]['future_price_gth'].mean()
combined_mdl_deploy = combined_mdl_deploy.iloc[:200]  # Only look at the top 200 as an option
print(combined_mdl_deploy)
#######


# Stock prices for every month
monthly_stock_prices_all = stk_prices_all_months[['Symbol', 'close_price']].sort_values(by=['Symbol', 'dt'])

# Historical drawdown
# Calculate the running maximum - first pivot the data
monthly_sp_all = monthly_stock_prices_all.pivot_table(index=['dt'], columns='Symbol', values='close_price')

mnthly_retrns = monthly_sp_all.pct_change()
cum_mnthly_returns = (1 + mnthly_retrns).cumprod()

# Calculate the running maximum
running_max = np.fmax.accumulate(cum_mnthly_returns)

# Ensure the value never drops below 1
running_max[running_max < 1] = 1

# Calculate the percentage drawdown
drawdown = (cum_mnthly_returns) / running_max - 1

# Plot the results - as you can see the drawdown on 'AAME' is a lot larger than 'AAPL'
drawdown[['AAME', 'AAPL']].plot()

largest_drawdown = pd.DataFrame(drawdown.loc[drawdown.index < '2020-07'].min(), columns=['largest_drawdown'])
largest_drawdown.tail()

# Test Drawdown portfolio

test_drawdown_ptf = pd.merge(combined_mdl_test, largest_drawdown, how='left', on=['Symbol'])
test_drawdown_ptf.head(10)

test_drawdown_ptf.sort_values(by=['largest_drawdown'], inplace=True, ignore_index=True, ascending=False)
test_drawdown_ptf['Top30_ret_ddown'] = test_drawdown_ptf.iloc[:30]['future_price_gth'].mean()
test_drawdown_ptf.head()

# Deploy Drawdown portfolio

deploy_drawdown_ptf = pd.merge(combined_mdl_deploy, largest_drawdown, how='left', on=['Symbol'])
deploy_drawdown_ptf.head(10)

deploy_drawdown_ptf.sort_values(by=['largest_drawdown'], inplace=True, ignore_index=True, ascending=False)
deploy_drawdown_ptf['Top30_ret_ddown'] = deploy_drawdown_ptf.iloc[:30]['future_price_gth'].mean()
deploy_drawdown_ptf.head()

# Results csv
# test_drawdown_ptf.to_csv(r'Files\results_test.csv', index=False, header=True)
# deploy_drawdown_ptf.to_csv(r'Files\results_deploy.csv', index=False, header=True)


# Plot the close price against the future price for the test and deploy set
# Tables used to populate the last two pages of the assignment


# Test
stk_prices_test = stk_prices.loc[stk_prices.index == '2020-07']
stk_prices_test = stk_prices_test[['Symbol', 'close_price', 'future_price']]
stk_prices_test.isnull().sum()

stk_prices_test = drop_row(stk_prices_test, ['future_price'])

stk_prices_test_upd = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(stk_prices_test,
                                                                   company_overview[['Symbol', 'Sector']], how='left',
                                                                   on=['Symbol']),
                                                          rf_test_mdl_results[['Symbol', 'rf_mdl_prob']].iloc[:30],
                                                          how='left', on=['Symbol']),
                                                 stkd_test_mdl_results[['Symbol', 'Stacked_mdl_prob']].iloc[:30],
                                                 how='left', on=['Symbol']),
                                        ga_test_mdl_results[['Symbol', 'ga_mdl_prob']].iloc[:30], how='left',
                                        on=['Symbol']),
                               nn_test_mdl_results[['Symbol', 'nn_mdl_prob']].iloc[:30], how='left', on=['Symbol'])

stk_prices_test_upd.iloc[:, 4:] = stk_prices_test_upd.iloc[:, 4:].fillna(0)
stk_prices_test_upd.isnull().sum()

stk_prices_test_upd.iloc[:, 4:] = stk_prices_test_upd.iloc[:, 4:].apply(lambda x: [y if y <= 0 else 1 for y in x])


# Scatter Plot
# plt.clf()
sns.scatterplot(data=stk_prices_test_upd.loc[stk_prices_test_upd['rf_mdl_prob'] == 0],
                x='close_price', y='future_price', palette='deep', alpha=0.3, zorder=1)
sns.scatterplot(data=stk_prices_test_upd.loc[stk_prices_test_upd['rf_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='RF')
sns.scatterplot(data=stk_prices_test_upd.loc[stk_prices_test_upd['Stacked_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='Stacked')
sns.scatterplot(data=stk_prices_test_upd.loc[stk_prices_test_upd['ga_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='Genetic Algo')
sns.scatterplot(data=stk_prices_test_upd.loc[stk_prices_test_upd['nn_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='Neural Net')
plt.plot([-1, 10000], [-1, 10000], linewidth=2, color='r', linestyle='--')
plt.xlabel('Close Price', size=12)
plt.xscale('log')
plt.ylabel('Future Price', size=12)
plt.yscale('log')
# plt.legend(loc='upper left')
plt.title("Close Price v Future Price (log scale)", fontdict={'size': 16})
plt.tight_layout()
plt.show()

# Create a stock price difference column
stk_prices_test_upd['stk_price_gth'] = (stk_prices_test_upd['future_price'] - stk_prices_test_upd['close_price']) \
                                       / stk_prices_test_upd['close_price']

stk_prices_test_upd = stk_prices_test_upd.loc[stk_prices_test_upd['stk_price_gth'] < 10]

# Boxplot and swarmplot to display the model returns by Sector
sns.boxplot(x="Sector", y="stk_price_gth", data=stk_prices_test_upd, zorder=10)
sns.swarmplot(x=stk_prices_test_upd["Sector"], y="stk_price_gth",
              data=stk_prices_test_upd.loc[stk_prices_test_upd['rf_mdl_prob'] == 1],
              label='RF', color='orange', zorder=100)
sns.swarmplot(x=stk_prices_test_upd["Sector"], y="stk_price_gth",
              data=stk_prices_test_upd.loc[stk_prices_test_upd['Stacked_mdl_prob'] == 1],
              label='Stacked', color='green', zorder=100)
sns.swarmplot(x=stk_prices_test_upd["Sector"], y="stk_price_gth",
              data=stk_prices_test_upd.loc[stk_prices_test_upd['ga_mdl_prob'] == 1],
              label='Genetic Algo', color='red', zorder=100)
sns.swarmplot(x=stk_prices_test_upd["Sector"], y="stk_price_gth",
              data=stk_prices_test_upd.loc[stk_prices_test_upd['nn_mdl_prob'] == 1],
              label='Neural Net', color='purple', zorder=100)
plt.xticks(rotation=45)
plt.xlabel('Sector', size=12)
plt.ylabel('Stock Price Growth', size=12)
# plt.legend(loc='upper right')
plt.title("Model Returns by Sector", fontdict={'size': 16})
plt.tight_layout()
plt.show()

# Deploy
stk_prices_deploy = stk_prices.loc[stk_prices.index == '2021-01']
stk_prices_deploy = stk_prices_deploy[['Symbol', 'close_price', 'future_price']]
stk_prices_deploy.isnull().sum()

stk_prices_deploy = drop_row(stk_prices_deploy, ['future_price'])

stk_prices_deploy_upd = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(stk_prices_deploy,
                                                                     company_overview[['Symbol', 'Sector']], how='left',
                                                                     on=['Symbol']),
                                                            rf_deploy_mdl_results[['Symbol', 'rf_mdl_prob']].iloc[:30],
                                                            how='left', on=['Symbol']),
                                                   stkd_deploy_mdl_results[['Symbol', 'Stacked_mdl_prob']].iloc[:30],
                                                   how='left', on=['Symbol']),
                                          ga_deploy_mdl_results[['Symbol', 'ga_mdl_prob']].iloc[:30], how='left',
                                          on=['Symbol']),
                                 nn_deploy_mdl_results[['Symbol', 'nn_mdl_prob']].iloc[:30], how='left', on=['Symbol'])

stk_prices_deploy_upd.iloc[:, 4:] = stk_prices_deploy_upd.iloc[:, 4:].fillna(0)
stk_prices_deploy_upd.isnull().sum()

stk_prices_deploy_upd.iloc[:, 4:] = stk_prices_deploy_upd.iloc[:, 4:].apply(lambda x: [y if y <= 0 else 1 for y in x])

# Scatter Plot
# plt.clf()
sns.scatterplot(data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['rf_mdl_prob'] == 0],
                x='close_price', y='future_price', palette='deep', alpha=0.3, zorder=1)
sns.scatterplot(data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['rf_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='RF')
sns.scatterplot(data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['Stacked_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='Stacked')
sns.scatterplot(data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['ga_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='Genetic Algo')
sns.scatterplot(data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['nn_mdl_prob'] == 1],
                x='close_price', y='future_price', palette='deep', alpha=1, zorder=100, label='Neural Net')
plt.plot([-1, 10000], [-1, 10000], linewidth=2, color='r', linestyle='--')
plt.xlabel('Close Price', size=12)
plt.xscale('log')
plt.ylabel('Future Price', size=12)
plt.yscale('log')
# plt.legend(loc='upper left')
plt.title("Close Price v Future Price (log scale)", fontdict={'size': 16})
plt.tight_layout()
plt.show()

# Create a stock price difference column
stk_prices_deploy_upd['stk_price_gth'] = (stk_prices_deploy_upd['future_price'] - stk_prices_deploy_upd['close_price']) \
                                         / stk_prices_deploy_upd['close_price']

stk_prices_deploy_upd = stk_prices_deploy_upd.loc[stk_prices_deploy_upd['stk_price_gth'] < 10]

# Boxplot and swarmplot to display the model returns by Sector
sns.boxplot(x="Sector", y="stk_price_gth", data=stk_prices_deploy_upd, zorder=10)
sns.swarmplot(x=stk_prices_deploy_upd["Sector"], y="stk_price_gth",
              data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['rf_mdl_prob'] == 1],
              label='RF', color='orange', zorder=100)
sns.swarmplot(x=stk_prices_deploy_upd["Sector"], y="stk_price_gth",
              data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['Stacked_mdl_prob'] == 1],
              label='Stacked', color='green', zorder=100)
sns.swarmplot(x=stk_prices_deploy_upd["Sector"], y="stk_price_gth",
              data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['ga_mdl_prob'] == 1],
              label='Genetic Algo', color='red', zorder=100)
sns.swarmplot(x=stk_prices_deploy_upd["Sector"], y="stk_price_gth",
              data=stk_prices_deploy_upd.loc[stk_prices_deploy_upd['nn_mdl_prob'] == 1],
              label='Neural Net', color='purple', zorder=100)
plt.xticks(rotation=45)
plt.xlabel('Sector', size=12)
plt.ylabel('Stock Price Growth', size=12)
# plt.legend(loc='upper right')
plt.title("Model Returns by Sector", fontdict={'size': 16})
plt.tight_layout()
plt.show()

stk_prices_deploy_upd.groupby(by='Sector').sum()
