from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split  # , KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('test.csv')
control_regression = pd.read_csv('control_regression.csv')
control = pd.read_csv('control.csv')

price = control_regression['sale_sale_price']
df.drop('sale_sale_price', axis='columns', inplace=True)
df.drop('parcel_parcel_number', axis='columns', inplace=True)

ctrl_price = control_regression['sale_sale_price']
control_regression.drop('sale_sale_price', axis='columns', inplace=True)
control_regression.drop('parcel_parcel_number', axis='columns', inplace=True)


x_train, x_test, y_train, y_test = train_test_split(df, price,
                                                    test_size=0.33,
                                                    random_state=1)

crx_train, crx_test, cry_train, cry_test = train_test_split(control_regression,
                                                            ctrl_price,
                                                            test_size=0.33,
                                                            random_state=1)

cx_train, cx_test, cy_train, cy_test = train_test_split(control,
                                                        ctrl_price,
                                                        test_size=0.33,
                                                        random_state=1)


def error(model, x, y):
    error = 0
    prediction = model.predict(x)
    error = (np.abs(prediction - y)/np.minimum(prediction, y))
    error = np.sum(error)/len(y)
    return error

random_forest = RandomForestRegressor(max_features='sqrt', n_estimators=1000,
                                      oob_score=True, max_depth=6, n_jobs=-1)
ols = LinearRegression(n_jobs=-1)
boost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)
sm_ols = sm.OLS(y_train, x_train)
ctrl_sm_ols = sm.OLS(cy_train, cx_train)


# This takes > 12 hours to run with random forest, so... take that with a grain of salt

random_forest.fit(x_train, y_train)
importance = random_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
idx = np.argsort(importance)[::-1]
features_to_keep = idx[importance > np.mean(importance)]
features_to_keep.shape
features = x_train[idx[features_to_keep]]
reduced_x_test = x_test[idx[features_to_keep]]

random_forest.fit(features, y_train)
random_forest.score(reduced_x_test, y_test)
error(random_forest, reduced_x_test, y_test)

print('Feature ranking:')
for f in xrange(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, x_train.columns[f], importance[idx[f]]))

plt.figure(figsize=(12, 12))
plt.title('Feature importances')
plt.bar(xrange(x_train.shape[1]), importance[idx], align='center', color='b', yerr=std[idx])
plt.show()


error(random_forest, x_test, y_test)

random_forest.fit(cx_train, cy_train)
error(random_forest, cx_test, cy_test)

random_forest.fit(crx_train, cry_train)
error(random_forest, crx_test, cry_test)

# SKlearn OLS
ols.fit(x_train, y_train)
error(ols, x_test, y_test)
ols.fit(crx_train, cry_train)
error(ols, crx_test, cry_test)
ols.fit(cx_train, cy_train)
error(ols, cx_test, cy_test)
#
# boost.fit(crx_train, cry_train)
# boost.score(crx_test, cry_test)
#
boost.fit(features, y_train)
boost.score(reduced_x_test, y_test)
error(boost, reduced_x_test, y_test)

boost.fit(cx_train, cy_train)

error(boost, cx_test, cy_test)


# SM OLS
sm_fit = sm_ols.fit()
print(sm_fit.summary2())

ctrl_sm_fit = ctrl_sm_ols.fit()
print(ctrl_sm_fit.summary2())

ctrl_cat_sm_fit = ctrl_cat_sm_ols.fit()
print (ctrl_cat_sm_fit.summary2())
