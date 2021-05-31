
# Section 1 - Import packages required for the assignment

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

# Section 2

# Section 2.1 - Import data from Yahoo finance
Stock_list = ['AMZN', 'TSLA']
ticker_info = []
for i in Stock_list:
    a = yf.Ticker(i)
    ticker_info.append(a)

print(ticker_info)

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