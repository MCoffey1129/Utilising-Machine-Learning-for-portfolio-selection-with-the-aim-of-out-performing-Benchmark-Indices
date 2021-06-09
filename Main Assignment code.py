###############################################################################################################
# Section 2 - Exploratory data analysis
###############################################################################################################


###############################################################################################################
# Section 2.0 - Import the required packages and functions
###############################################################################################################

# Packages
import pandas as pd
import time as time
from datetime import timedelta



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
def unique_symbol(input_table):
    """A function for returning a pandas dataframe containing unique symbols for the input table"""
    output = pd.DataFrame(input_table['Symbol'].unique(), columns=['Symbol'])
    return output



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



# Import the CSVs into Python

company_overview = pd.read_csv(r'Files\Overview_df.csv')
eps_data = pd.read_csv(r'Files\eps_data.csv')
inc_st_data = pd.read_csv(r'Files\inc_st_data.csv')
bs_data = pd.read_csv(r'Files\BS_data.csv')
cf_data = pd.read_csv(r'Files\CF_data.csv')
monthly_stock_prices = pd.read_csv(r'Files\monthly_prices.csv')

# Pull in the unique stock symbols for each table (time taken for each call of the function <0.0s)
co_symbol_unique = unique_symbol(company_overview)
eps_symbol_unique = unique_symbol(eps_data)
inc_st_symbol_unique = unique_symbol(inc_st_data)
bs_symbol_unique = unique_symbol(bs_data)
cf_symbol_unique = unique_symbol(cf_data)
sp_symbol_unique = unique_symbol(monthly_stock_prices)

# We only want to keep stocks which are contained in each file, in order to achieve this we complete run an
# inner join on each of the datasets containing the unique stock symbols
# There are 5,082 stocks which are in every file, this is what we will use going forward
symbols_in_all_files = \
    pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(co_symbol_unique, eps_symbol_unique, how='inner', on='Symbol')
                                        , inc_st_symbol_unique, how='inner', on='Symbol')
                               , bs_symbol_unique, how='inner', on='Symbol')
                      , cf_symbol_unique, how='inner', on='Symbol')
             , sp_symbol_unique, how='inner', on='Symbol')

symbols_in_all_files.shape # Check complete that there are 5,082 stocks in this table


# Update the company overview such that it will contain only the stocks which are contained in each file

company_overview_upd = pd.merge(company_overview, symbols_in_all_files, how='inner', on='Symbol')
company_overview_upd.shape # updated file contains 5,082 stocks as expected


# Update our initial dataframe such that it is in the correct form required for modelling.
# As the approach is a 6 month hold and sell strategy we want to get the stock information off quarter
# so that we do not have any issues aro....


dates = ['2021-01', '2020-07', '2020-01', '2019-07', '2019-01', '2018-07', '2018-01', '2017-07']
company_overview_dt = pd.DataFrame()

for i in dates:
    company_overview_upd['dt'] = i
    company_overview_dt = company_overview_dt.append(company_overview_upd, ignore_index=True)

company_overview_dt.head()
company_overview_dt.tail()
company_overview_dt.shape # 40,656 rows (5,082 * 8 timeframes), the for loop was run correctly
company_overview_dt.info()  # dt is an object, we want this to be a datetime and we want to set it as our index.


# Change the dt field to a datetime object and set it as the index
company_overview_dt.index = pd.to_datetime(company_overview_dt['dt']).dt.to_period('M')
company_overview_dt = company_overview_dt.drop(columns=company_overview_dt.columns[8])  # drop the second dt field
company_overview_dt.head()
company_overview_dt.tail()


#
eps_data.head(40)
eps_data.info()
eps_data['fiscalDateEnding'] = pd.to_datetime(eps_data['fiscalDateEnding']).dt.to_period('M')
eps_data['reportedDate'] = pd.to_datetime(eps_data['reportedDate']).dt.to_period('M')

delta = timedelta(years=1)

eps_data.loc[(eps_data['reportedDate'].dt.month == 12) | (eps_data['reportedDate'].dt.month == 11) |
             (eps_data['reportedDate'].dt.month == 10), 'dt'] = eps_data['reportedDate'].dt.month + delta

print(eps_data['reportedDate'].dt.month)

print(eps_data['reportedDate'].month)