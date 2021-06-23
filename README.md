# UCDPA_MartinCoffey


Name: Martin Coffey

Project Title:
Utilising Machine Learning for portfolio selection with the aim of out-performing benchmark indices



Introduction


The aim of this project is to assess if a 6 month buy and hold trading strategy based on Machine Learning portfolio selection can out-perform the 
S&P 500 for the same period (the purchase and sale is assumed to occur on the last trading day of January and June each year). The list 
of stocks used in the project are sourced from the NASDAQ website.

Classification models rather than regression models will be used to generate stock portfolios. Each of the models will be trained to predict
if a stock will increase by 10% or more 6 months from that date. The probabilities generated from the models will be used to generate the stock
portfolios i.e. the higher the probability the model thinks a stock price will increase by 10% or more 6 months from that date the more likely the
stock will be included in the portfolio.



The following models will be used in this project:
•	Random Forest model, 
•	Stacked model – RF, KNN, XGBoost and CatBoost as Base models and XGBoost as the meta model.
•	Genetic algorithm
•	Average of 3 diverse Neural Networks





Modelling approach

 
				Figure 1: Modelling approach


1.	Data Import
The data used in the model was imported from the Alpha Vantage website via an API key and includes:
•	Company_overview: 
Name, Sector and Industry are used while the other elements are deleted as they are forward looking which would positively skew the model

•	Earnings, Income_Statement, Balance Sheet and Cash_Flow_statement:
The quarterly reports published by each company is  imported into Python

•	Time_Series_Monthly_Adjusted:
The monthly adjusted stock price for each stock is imported into Python.
The target for our classification model is “gt_10pc_gth” which is 1 if the account’s stock price increased by 10% after 6 months and 0 otherwise.

T



2.	Feature Engineering

Large institutional investors tend to use a combination of favourable valuation, profitability, solvency and liquidity ratios combined with forecasted discounted cash-flow models to assess if a stock is undervalued. 

Feature engineering was applied to the data to obtain these ratios.

 

3.	Data Preparation

3.1 Train/Test/Deploy split:

 

Figure 2: Train, test and deploy datasets

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
•	The hyperparameters were tuned via GridSearchCV but were not used in the modelling process, instead we first assess a simplified version of each of the models in order to calculate a powerful “Combined” model 


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

Model	    No of hidden layers	    No of nodes in each layer	        Activation function	    Dropout	                                  Batc size	No of epochs
NN1	              3	                  30                                    	relu	          No dropout	                              20	    500
NN2              	2	                  50	                                    relu	    10% on the second hidden Layer	               100	    500
NN3             	3                 	30                                    	tanh	           No dropout	                              200	    500



 
Table 4: ANN table 

o	 Although the third model is not as accurate for predictions as the first two averaging in the probabilities with the first two Neural Networks makes the model more generalizable

o	This multiple neural network (average of the 3 models) is the model that we will test the trading strategy on



5.	Results
The S&P 500 return from Jul ’20 to Jan ’21 was 13.5% and the return from Jan ’21 to May ’21 was 13.2%. 

Definitions in the below table:
•	“Precision” refers to the test and deployment model precision.
•	“Top 30 return” refers to the 6 month stock price return of the portfolio generated by taking the 30 stocks with the highest modelled probability of having a 6 month stock price growth of greater than 10%
•	“Top 100 return,” is the same as “Top 30 return” only it includes 100  stocks 
•	The “Combined model” takes an average of the probabilities of the 4 models (MNN, GA, Stacked Model and Random Forest)
•	The “Min drawdown” portfolio takes the 200 stocks with the highest modelled probability of having a share price greater than 10% and takes only the top 30 stocks with the smallest drawdown over the last 4 years.
•	The “Diversified” portfolio simply takes the stock with the highest probability from each industry and combines each of these into an overall well diversified portfolio (this was completed using the Combined model).  

                                 Test dataset (Jul '20)					             Deployment dataset (Jan '21)			
                         Precision	  Top 30 return	Top 100 return			 Precision	    Top 30 return	    Top 100 return	
Random Forest	            68.0%	         18.4%	      29.0%			          62.0%	            28.6%            32.1%	
Stacked Model	            68.0%	         41.3%	      51.1%			          57.0%	            15.1%	           17.5%	
Genetic Algorithm   	    72.0%	         72.3%	      52.9%			          43.0%           	13.8%	           13.2%	
Multiple Neural Networks	69.0%	          51.6%     	64.6%			          57.0%	            18.8%	           17.5%	
									
									
	Top 30 return	Top 100 return	Min drawdown	Diversified		Top 30 return	Top 100 return	Min drawdown	Diversified
Combined model	25.9%	41.1%	9.2%	33.0%		29.8%	19.5%	9.2%	11.2%


5.1	Findings:
•	The “Combined” “Top 30” return model which is an average of all the models generates 25.9% return in the Test data and 29.8% return on the Deployment data (assuming equal weights in the portfolio). An investment of €10k at Jul ’20 would have returned €16.34k at the end of May ’21 (63.4% return) versus 28.5% return from the S&P 500 for the same period. 
•	The model performed well on the Deployment data for certain Sectors, for Banks-Diversified (90% of cases predicted to increase by more than 10% were correctly predicted), REIT-Industrial (91% correctly predicted) and machinery (80% correctly predicted) but did very poorly for electrical equipment (40%), internet content (31%) and software applications (14%) 

•	The simple random forest model performs well in both the test and deployment dataset. 

•	The Stacked model does very well in the “Test” dataset but not as well in the “Deployment” dataset. 
•	The Multiple Neural Networks results are quite like the stacked model.




