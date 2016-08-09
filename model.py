from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score  # , KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import ttest_ind


df = pd.read_csv('test.csv')
control = pd.read_csv('control.csv')

df.drop('LUDescription_nan', axis=1, level=None, inplace=True, errors='raise')
price = df['sale_sale_price']
df.drop('sale_sale_price', axis='columns', inplace=True)
df.drop('parcel_parcel_number', axis='columns', inplace=True)

ctrl_price = control['sale_sale_price']
control.drop('sale_sale_price', axis='columns', inplace=True)
control.drop('parcel_parcel_number', axis='columns', inplace=True)
control.drop('Unnamed: 0', axis='columns', inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df, price,
                                                    test_size=0.33,
                                                    random_state=0)

cx_train, cx_test, cy_train, cy_test = train_test_split(control,
                                                        ctrl_price,
                                                        test_size=0.33,
                                                        random_state=0)


def error(model, x, y):
    error = 0
    prediction = model.predict(x)
    error = (np.abs(prediction - y)/np.minimum(prediction, y))
    error = np.sum(error)/len(y)
    return error

random_forest = RandomForestRegressor(max_features='log2', n_estimators=1000,
                                      n_jobs=-1)
ols = LinearRegression(n_jobs=-1)
boost = GradientBoostingRegressor(loss='huber', n_estimators=1000, learning_rate=0.05)
sm_ols = sm.OLS(y_train, x_train)
ctrl_sm_ols = sm.OLS(cy_train, cx_train)

# SKlearn OLS
ols.fit(x_train, y_train)
ols_score = cross_val_score(ols, x_test, y_test, scoring=None, cv=10, n_jobs=1,
                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
print 'OLS R2 test: {}'.format(ols_score.mean())
# print 'OLS test Reletive Difference: {}'.format(error(ols, x_test, y_test))
ols.fit(cx_train, cy_train)
ols_ctrl = cross_val_score(ols, cx_test, cy_test, scoring=None, cv=10, n_jobs=1,
                           verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
print 'OLS R2 ctrl: {}'.format(ols_ctrl.mean())

print ttest_ind(ols_score, ols_ctrl, axis=0, equal_var=True, nan_policy='propagate')

# print 'OLS control Reletive Difference: {}'.format(error(ols, cx_test, cy_test))
#
# boost.fit(crx_train, cry_train)
# boost.score(crx_test, cry_test)
#
boost.fit(x_train, y_train)
boost_score = cross_val_score(boost, x_test, y_test, scoring=None, cv=10,
                              n_jobs=1, verbose=0, fit_params=None,
                              pre_dispatch='2*n_jobs')
print 'Boosting R2 test: {}'.format(boost_score.mean())

# print 'Gradient Boosting test R2: {}'.format(boost.score(x_test, y_test))
# print 'Gradient Boosting test Reletive Difference: {}'.format(error(boost, x_test, y_test))
boost.fit(cx_train, cy_train)
boost_ctrl = cross_val_score(boost, cx_test, y=cy_test, scoring=None, cv=10,
                             n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
print 'Boosting R2 ctrl: {}'.format(boost_ctrl.mean())
# print 'Gradient Boosting control R2: {}'.format(boost.score(cx_test, cy_test))
# print 'Gradient Boosting control Reletive Difference: {}'.format(error(boost, cx_test, cy_test))
print ttest_ind(boost_score, boost_ctrl, axis=0, equal_var=True, nan_policy='propagate')



# (Random Forest)
random_forest.fit(x_train, y_train)
random_forest.score(x_test, y_test)
test_score = cross_val_score(random_forest, x_test, y_test, scoring=None,
                             cv=10, n_jobs=1, verbose=0, fit_params=None,
                             pre_dispatch='2*n_jobs')
print 'Random Forest R2 test: {}'.format(test_score.mean())


print error(random_forest, x_test, y_test)

random_forest.fit(cx_train, cy_train)
control_score = cross_val_score(random_forest, cx_test, cy_test, scoring=None,
                                cv=10, n_jobs=1, verbose=0, fit_params=None,
                                pre_dispatch='2*n_jobs')

print 'Random Forest R2 test: {}'.format(control_score.mean())

print ttest_ind(test_score, control_score, axis=0, equal_var=True, nan_policy='propagate')


print error(random_forest, cx_test, cy_test)

# SM OLS
sm_fit = sm_ols.fit()
print(sm_fit.summary2())
print error(sm_fit, x_test, y_test)

ctrl_sm_fit = ctrl_sm_ols.fit()
print(ctrl_sm_fit.summary2())
print error(ctrl_sm_fit, cx_test, cy_test)
