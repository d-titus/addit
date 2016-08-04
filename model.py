from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split  # , KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('test.csv')
control = pd.read_csv('control.csv')

price = control['sale_sale_price']
df.drop('sale_sale_price', axis='columns', inplace=True)
df.drop('parcel_parcel_number', axis='columns', inplace=True)

ctrl_price = control['sale_sale_price']
control.drop('sale_sale_price', axis='columns', inplace=True)
control.drop('parcel_parcel_number', axis='columns', inplace=True)


x_train, x_test, y_train, y_test = train_test_split(df, price,
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

random_forest = RandomForestRegressor(max_features='log2', n_estimators=1000,
                                      n_jobs=-1)
ols = LinearRegression(n_jobs=-1)
boost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)
sm_ols = sm.OLS(y_train, x_train)
ctrl_sm_ols = sm.OLS(cy_train, cx_train)

random_forest.fit(x_train, y_train)
error(random_forest, x_test, y_test)


# SKlearn OLS
ols.fit(x_train, y_train)
error(ols, x_test, y_test)
ols.fit(cx_train, cy_train)
error(ols, cx_test, cy_test)
#
# boost.fit(crx_train, cry_train)
# boost.score(crx_test, cry_test)
#
boost.fit(x_train, y_train)
boost.score(x_test, y_test)
error(boost, x_test, y_test)
boost.fit(cx_train, cy_train)
error(boost, cx_test, cy_test)


# SM OLS
sm_fit = sm_ols.fit()
print(sm_fit.summary2())

ctrl_sm_fit = ctrl_sm_ols.fit()
print(ctrl_sm_fit.summary2())
