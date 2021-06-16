
# Linear regression model

# Build model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.score(X_train, y_train)  # 19.44%
lin_reg.score(X_test, y_test)  # << 0% - we have overfit the model

lin_reg.coef_[0]
print(X)

# Display two vectors, the y predicted v y train
y_pred_df = pd.DataFrame(lin_reg.predict(X_test), columns=['y_pred'])
y_test_df = pd.DataFrame(y_test, columns=['y_test'])
lin_reg_pred = pd.concat([y_test_df, y_pred_df.set_index(y_test_df.index)], axis=1)
print(lin_reg_pred)

# Visually comparing the predicted values for profit versus actual
sns.scatterplot(data=lin_reg_pred, x='y_pred', y='y_test', palette='deep', legend=False)
plt.xlabel('Predicted Stock Price', size=12)
plt.ylabel('Actual Stock Price', size=12)
plt.title("Predicted v Actual Stock Price", fontdict={'size': 16})
plt.tight_layout()
plt.show()
#
# Run a random forest to check what are the most important features in predicting future stock prices
X_train_rf = X_train
y_train_rf = y_train.ravel()
X_test_rf = X_test
y_test_rf = y_test.ravel()

np.shape(X_train_rf)

# Grid Search

rfr = RandomForestRegressor(criterion='mse')
param_grid = [{'n_estimators': [20, 50, 100, 200], 'max_depth': [2, 4, 8], 'max_features': ['auto', 'sqrt']
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

#
# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))
print(sorted_index)
X.columns[733]


#
labels = np.array(X.columns)[sorted_index]
print(labels)

print(importances[sorted_index])

#


feature_imp_df = pd.concat([pd.DataFrame(importances[sorted_index], columns=['Importance']),
                 pd.DataFrame(labels, columns=['Feature'])], axis=1)

#
print(feature_imp_df)
feature_imp_df[feature_imp_df['Feature'] == 'p_to_e_4Q_gth']





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


model_params = {
    'svm' : {'model' : SVC(kernel = 'rbf', random_state = 0),
             'params' : {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
                {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}},

    'random_forest' : { 'model': RandomForestClassifier(criterion='entropy', random_state=0),
                        'params': {'n_estimators' : [50,100,200,500], 'max_features': ['auto', 'sqrt','log2']
                                   'class_weight' : [{0:0.3, 1:0.7},{0:0.2, 1:0.8},{0:0.1, 1:0.9}, {0:0.05, 1:0.95}}},

    'knn' : { 'model' : KNeighborsClassifier(),
              'params' : {'n_neighbours':[2,3,5,9,15,25], 'p': [1,2], leaf_size : [1,2,12,25,100,200]}
             }
}

