******************************************************************************************************************************************
There are two python scripts in this project:

1. Input_code: Code to generate the data required for the model the output of which are the csvs saved on GITHUB e.g. eps_data.csv etc.

2. Main assignment code - imports the csvs generated from the Input_code script and creates our models   

There is one small change required in the code, the return on the "minimum drawdown" portfolio should be 13.2% not 9.2%. The code should be updated as below:

deploy_drawdown_ptf['Top30_ret_ddown'] = test_drawdown_ptf.iloc[:31]['future_price_gth'].mean()
should read
deploy_drawdown_ptf['Top30_ret_ddown'] = deploy_drawdown_ptf.iloc[:31]['future_price_gth'].mean()
******************************************************************************************************************************************

The Project Explained:

The aim of this project is to assess if a 6 month buy and hold trading strategy based on Machine Learning portfolio selection can out-perform the S&P 500 for the same period (the purchase and sale is assumed to occur on the last trading day of January and June each year). The list of stocks used in the project are sourced from the NASDAQ website.


Classification models rather than regression models are used to generate stock portfolios. Each of the models will be trained to predict if a stock will increase by 10% or more 6 months from that date. The probabilities generated from the models will be used to generate the stock portfolios i.e. the higher the probability the model thinks a stock price will increase by 10% or more 6 months from that date the more likely the stock will be included in the portfolio.



The following models will be used in this project:

•	Random Forest model, 

•	Stacked model – RF, KNN, XGBoost and CatBoost as Base models and XGBoost as the meta model.

•	Genetic algorithm

•	Average of 3 diverse Neural Networks





Project process:

1.	Data Import
The data used in the model was imported from the Alpha Vantage website via an API key and includes:

•	Company_overview: 

Name, Sector and Industry are used while the other elements are deleted as they are forward looking which would positively skew the model

•	Earnings, Income_Statement, Balance Sheet and Cash_Flow_statement:

The quarterly reports published by each company is  imported into Python

•	Time_Series_Monthly_Adjusted:

The monthly adjusted stock price for each stock is imported into Python.
The target for our classification model is “gt_10pc_gth” which is 1 if the account’s stock price increased by 10% after 6 months and 0 otherwise




2.	Feature Engineering

Large institutional investors tend to use a combination of favourable valuation, profitability, solvency and liquidity ratios combined with forecasted discounted cash-flow models to assess if a stock is undervalued. 

Feature engineering was applied to the data to obtain these ratios.




3.	Data Preparation

3.1 Data was split into the training set (data from July '17 to Jan '19, test dataset (July '20) and the deployment dataset (Jan '21)


•	The model is trained on the training data and is evaluated on both the Test and Deploy dataset.

o	The training data consists of 6 trading windows from Jul ’17 to Jan ’19 (c. 75% of the data)

o	The test data for which the training model is tested on consists of one trading window at Jul ’20 (c. 12.5% of the data)

o	The training model is subsequently tested on the deploy dataset  to ensure that the models perform well across multiple time points (c. 12.5% of the data). Please note this trading window is only 4 months as we do not have data as of the last trading day of July. 

3.2 Missing data & scaling features


Missing data:

•	Any characters not required for modelling were dropped.

•	A cross validation test on the training data proved that using KNNImputer (n_neighbors=5) to impute missing numeric fields yielded a higher cross validation score over setting the remaining missing numeric fields to 0. Going forward different values for “n_neighbors” should be tested.


Scaling data:

•	Scale the values such that each are in the range [0,1]
Scaling was necessary for feature selection and modelling (completed using MinMaxScaler()) 


Feature reduction:

•	The training dataset contained 1,012 features, if the model performs well on the training set there is a greater risk of overfitting)

•	SelectKbest was considered but it is difficult to know at what chi squared level features should be dropped

•	Recursive feature elimination performs Feature ranking with recursive feature elimination and cross-validated selection of the best number of features. The f1 score drops from 27.43% to 13.65% after removing 100 of the features. Given this decrease and the high degree of correlation between features reducing the feature dimensionality via PCA was used. The number of features was reduced to 100 which led to a minor c.10% decrease in variance. The cross validated training set precision increased as a result of applying PCA and there was reduced overfitting in the Artificial Neural Networks





4.	Modelling

4.1 Hyperparamter tuning:

•	The hyperparameters were tuned via GridSearchCV but were not used in the modelling process, instead we first assess a simplified version of each of the models in order to calculate a generalisable “Combined” model 


4.2	Random Forest

•	A simple random forest model using 5 decision trees is applied (n_estimators = 5)

•	This model was evaluated on its own and as a base estimator of a “Stacked” model.



4.3	K Nearest Neighbours

•	A simplified version of the KNN model was used in this project setting n-neighbours = 5 i.e. the model estimates how likely a data point is to be a member of one group or the other depending on what group the data points the 5 nearest to it are in.

4.4	XGBoost and CatBoost

•	XGBoost and CatBoost are implementations of gradient boosted decision trees .

•	The default settings for both Catboost and XGBoost were used in this model.

•	Both are used as Base estimators for the “Stacked” model while XGBoost is also used as a the meta model


4.5	Genetic Algorithm:

•	A simple algorithm was used  containing a population size of 5 an offspring size of 3 and 3 generations of the algorithm. The best pipeline was DecisionTreeClassifier(input_matrix, criterion=gini, max_depth=3, min_samples_leaf=10, min_samples_split=9)



4.6	Neural Networks

•	Three diverse neural networks were run on the data and the average probabilities from the 3 models were 

•	The first neural network has 3 hidden layers with 30 node in each, the activation function is relu, there is no dropout and it is run across 500 epochs with a batch size of 20.

•	The second neural network has 2 hidden layers with 50 node in each, the activation function is relu, there is a 10% dropout on the second hidden layer and it is run across 500 epoch with a batch size of 100.

•	The third neural network has 3 hidden layers with 30 node in each, the activation function is tanh, there is no dropout and it is run across 500 epoch with a batch size of 200.


•	Although the third model is not as accurate for predictions as the first two averaging in the probabilities with the first two Neural Networks makes the model more generalizable

•	This multiple neural network (average of the 3 models) is the model that we will test the trading strategy on.





5.	Results

The S&P 500 return from Jul ’20 to Jan ’21 was 13.5% and the return from Jan ’21 to May ’21 was 13.2%. 

Definitions in the code (please note the portfolio size of 30 and 100 is arbitrary but the general consensus is that close to 30 stocks in a portfolio is ideal, snip from Investing.com - "While there is no consensus answer, there is a reasonable range for the ideal number of stocks to hold in a portfolio: for investors in the United States, the number is about 20 to 30 stocks"):

•	“Precision” refers to the precision of each of the models on the test and deployment data.

•	“Top 30 return” refers to the 6 month stock price return of the portfolio generated by taking the 30 stocks with the highest modelled probability of having a 6 month stock price growth of greater than 10%

•	“Top 100 return,” is the same as “Top 30 return” only it includes 100  stocks 

•	The “Combined model” takes an average of the probabilities of the 4 models (MNN, GA, Stacked Model and Random Forest)

•	The “Min drawdown” portfolio takes the 200 stocks with the highest modelled probability of having a share price greater than 10% and takes only the top 30 stocks with the smallest drawdown over the last 4 years.

•	The “Diversified” portfolio simply takes the stock with the highest probability from each industry and combines each of these into an overall well diversified portfolio (this was completed using the Combined model).  




5.1	Findings:

•	The “Combined” “Top 30” return model which is an average of all the models generates 25.9% return in the Test data and 29.8% return on the Deployment data (assuming equal weights in the portfolio). An investment of €10k at Jul ’20 would have returned €16.34k at the end of May ’21 (63.4% return) versus 28.5% return from the S&P 500 for the same period. 

•	The model performed well on the Deployment data for certain Sectors, for Banks-Diversified (90% of cases predicted to increase by more than 10% were correctly predicted), REIT-Industrial (91% correctly predicted) and machinery (80% correctly predicted) but did very poorly for electrical equipment (40%), internet content (31%) and software applications (14%) 

•	The simple random forest model performs well in both the test and deployment dataset. 

•	The Stacked model does very well in the “Test” dataset but not as well in the “Deployment” dataset. 

•	The Multiple Neural Networks results are quite like the stacked model.





Other interesting findings (test dataset):

•	There are two stocks in the Genetic Algorithm portfolio which out-performed in the period, Riot Blockchain Inc which had a stock price return of 677% in the period and Bit Digital which had a return of 528% in the period. However, the Generic Algorithm stock portfolio only contains Technology stocks. This makes the portfolio more risky as there is little diversification in the portfolio. The model will have to be firstly checked for why it has such a high weighting toward Technology stocks.

•	There are four Sectors in which none of the models had any stocks “Consumer Defensive,” “Utilities,” “None” and “Other.” The lack of selection will need to be reviewed but it is most likely a result of poor performance of these sectors over the last number of years.

•	Random Forest and Neural Networks have a good spread of stocks across Sectors. The Neural Network was the only model to choose stocks in the Basic Material Sector for which none of the returns were above the median return of that Sector (this will have to be reviewed).



Other interesting findings (test dataset):

•	The genetic algorithm which only had Technology shares in its Test portfolio now have an equal amount of Technology and Healthcare stocks in its 
“Top 30” portfolio (11 of each). It only has one stock in the Real Estate sector but that was the highest performing Real Estate stock for the period “Nam Tai Property Inc.” which has seen a 365% share price increase YTD. 

•	The Random Forest model has moved from not having any Communication Services stocks in its Top 30 to having 23 stocks in the Deploy dataset. This includes the best performing communication stock in the period “Moxian Inc” which had a 653% return in the 6-month period. The lack of diversification is an issue especially when it had stocks across numerous industries in Jul ’20.

•	The sectors “Other,” “None” and “Utilities,” have again not been chosen by any model. There are two utility Sector stocks which the Combined model predicted would have a greater than 10% growth in the period, these cases had a combined average growth rate in the period of 16% (the model is correctly predicting Utility stocks which will increase in value) but they do not have a high enough probability to make it into the Top30 stock portfolio of any model.

•	The Neural Network model has a high number of underperforming consumer cyclical stocks, we will have to review why this is the case.



