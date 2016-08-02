from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

df = pd.read_csv('test_categorical.csv')
control_regression = pd.read_csv('control_regression.csv')
control_categorical = pd.read_csv('categorical_control.csv')

price = control_regression['sale_sale_price']
# df.drop('sale_sale_price', axis='columns', inplace=True)
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

ccx_train, ccx_test, ccy_train, ccy_test = train_test_split(control_categorical,
                                                            ctrl_price,
                                                            test_size=0.33,
                                                            random_state=1)


def error(model, x, y):
    # Mean
    error = 0
    prediction = model.predict(x)
    for i in prediction:
        if prediction[i] > y[i]:
            error = np.abs(prediction[i] - y[i])/prediction[i]
        else:
            error = np.abs(prediction[i] - y[i])/y[i]
    return error

random_forest = RandomForestRegressor(max_features='sqrt', n_estimators=1000,
                                      oob_score=True, max_depth=6, n_jobs=-1)
ols = LinearRegression(n_jobs=-1)
boost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)
sm_x_train = sm.add_constant(x_train)
sm_ols = sm.OLS(y_train, x_train)
ctrl_sm_ols = sm.OLS(cry_train, crx_train)

random_forest.fit(x_train, y_train)
random_forest.score(x_test, y_test)
random_forest.oob_score_

random_forest.fit(ccx_train, ccy_train)
random_forest.score(ccx_test, ccy_test)
random_forest.oob_score_

random_forest.fit(crx_train, cry_train)
random_forest.score(crx_test, cry_test)
random_forest.oob_score_

# SKlearn OLS
ols.fit(x_train, y_train)
ols.score(x_test, y_test)
ols.fit(crx_train, cry_train)
ols.score(crx_test, cry_test)
ols.fit(ccx_train, ccy_train)
ols.score(ccx_test, ccy_test)

boost.fit(crx_train, cry_train)
boost.score(crx_test, cry_test)

boost.fit(x_train, y_train)
boost.score(x_test, y_test)

# SM OLS
sm_fit = sm_ols.fit()
print(sm_fit.summary2())

ctrl_sm_fit = ctrl_sm_ols.fit()
print(ctrl_sm_fit.summary2())


np_control = np.array(control)

for i in xrange(len(control.columns)):
    print 'Column: {}, VIF: {}'.format(control.columns[i], VIF(np_control, i))

control.drop('structure_year_built', axis='columns', inplace=True)

control.columns

for i in xrange(len(control.columns)):
    print 'Column: {}, VIF: {}'.format(control.columns[i], VIF(np_control, i))

control.drop('structure_quality', axis='columns', inplace=True)

control.columns

for i in xrange(len(control.columns)):
    print 'Column: {}, VIF: {}'.format(control.columns[i], VIF(np_control, i))

control.drop('structure_condition', axis=1, inplace=True)


ctrl_sm_ols = sm.OLS(cy_train, cx_train)
ctrl_sm_fit = ctrl_sm_old.fit()
print(ctrl_sm_fit.summary2())
