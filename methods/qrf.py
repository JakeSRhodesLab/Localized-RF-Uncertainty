# For quantile regression forest


def quantile_regression_forest(X, y, X_test, y_test = None, random_state = None, **kwargs):

    qrf = RandomForestQuantileRegressor(random_state = random_state, **kwargs)
    qrf.fit(X, y)

    y_pred = qrf.predict(X, quantiles=[0.025, 0.5, 0.975])



"""For each of the regression methods, we need to return calculate the 
prediction intervals for a variety of quantiles (0.8, 0.8, 0.95, 0.975, 0.99).
For each of these quantiles, we need to produce the following:
* 5-number summery
* width
* coverage

Do the same for training (OOB?) and test sets"""