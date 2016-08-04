from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
# import seaborn as sns

test_categorical = pd.read_csv('test_categorical.csv', dtype={'parcel_parcel_number': object})
control_categorical = pd.read_csv('control_categorical.csv', dtype={'parcel_parcel_number': object})
control_regression = pd.read_csv('control_regression.csv', dtype={'parcel_parcel_number': object})

price = control_regression['sale_sale_price']
control_regression.drop('parcel_parcel_number', axis=1, inplace=True)
control_regression.drop('sale_sale_price', axis=1, inplace=True)
control_regression.drop('Unnamed: 0', axis=1, inplace=True)
control_categorical.drop('Unnamed: 0', axis=1, inplace=True)
control_categorical.drop('parcel_parcel_number', axis=1, inplace=True)
test_categorical.info()



def error(model, x, y):
    error = 0
    prediction = model.predict(x)
    error = (np.abs(prediction - y)/np.minimum(prediction, y))
    error = np.sum(error)/len(y)
    return error

random_forest = RandomForestRegressor(max_features='log2', max_depth=3, n_estimators=1000,
                                      n_jobs=-1)

x_train, x_test, y_train, y_test = train_test_split(test_categorical, price,
                                                    test_size=0.33,
                                                    random_state=1)

cx_train, cx_test, cy_train, cy_test = train_test_split(control_categorical,
                                                        price,
                                                        test_size=0.33,
                                                        random_state=1)

crx_train, crx_test, cry_train, cry_test = train_test_split(control_regression,
                                                            price,
                                                            test_size=0.33,
                                                            random_state=1)

crx_train.info()


random_forest.fit(cx_train, cy_train)
error(random_forest, cx_test, cy_test)
importance = random_forest.feature_importances_
importance = importance/np.max(importance)
# std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
idx = np.argsort(importance)[::-1]
features_to_keep = idx[importance > np.mean(importance)]
features_to_keep.shape
cx_train = cx_train[idx[features_to_keep]]
cx_test = cx_test[idx[features_to_keep]]
random_forest.fit(cx_train, cy_train)
error(random_forest, cx_test, cy_test)

importance[features_to_keep].shape
print('Feature ranking:')
for f in xrange(cx_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, cx_train.columns[f], importance[idx[f]]))

plt.figure(figsize=(10, 12))
plt.title('Feature importances')
plt.xticks(features_to_keep)
plt.bar(xrange(cx_train.shape[1]), importance[features_to_keep], align='center', color='b')
plt.show()

random_forest.fit(x_train, y_train)
error(random_forest, x_test, y_test)
importance = random_forest.feature_importances_
importance = importance/np.max(importance)
# std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
idx = np.argsort(importance)[::-1]
features_to_keep = idx[importance > np.mean(importance)]
features_to_keep.shape
x_train = x_train[idx[features_to_keep]]
x_test = x_test[idx[features_to_keep]]
random_forest.fit(x_train, y_train)
error(random_forest, x_test, y_test)

importance[features_to_keep].shape
print('Feature ranking:')
for f in xrange(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, x_train.columns[f], importance[idx[f]]))

plt.figure(figsize=(10, 12))
plt.title('Feature importances')
plt.xticks(features_to_keep)
plt.bar(xrange(x_train.shape[1]), importance[features_to_keep], align='center', color='b')
plt.show()





ols = sm.OLS(y_train, x_train)
ols_fit = ols.fit()
ols_fit.summary2()
error(ols, cry_test, crx_test)
