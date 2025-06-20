# File for all regression interval methods using the RFGAP class.
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from rfgap import RFGAP
from rfgap.helpers import is_in_interval, get_coverage, get_width_stats


def extract_width_stats(width_stats):
	return {
		'width_mean': width_stats[0],
		'width_std': width_stats[1],
		'width_min': width_stats[2],
		'width_q1': width_stats[3],
		'width_median': width_stats[4],
		'width_q3': width_stats[5],
		'width_max': width_stats[6],
	}


def get_qrf_results(X, y, X_test=None, y_test=None,
					prox_method=None, level=0.95,
					random_state=None, k=None, **kwargs):
	# Quantile Regression Forest
	qrf = RandomForestQuantileRegressor(random_state=random_state, oob_score=True)
	qrf.fit(X, y)

	quantiles = [(1 - level) / 2, 0.5, 1 - (1 - level) / 2]
	qrf_prediction_test = qrf.predict(X=X_test, quantiles=quantiles)
	qrf_lwr_test, qrf_pred_test, qrf_upr_test = qrf_prediction_test.T

	qrf_prediction = qrf.predict(X, quantiles=quantiles, oob_score=True)
	qrf_lwr, qrf_pred, qrf_upr = qrf_prediction.T

	# Width and coverage
	width_test = extract_width_stats(get_width_stats(qrf_lwr_test, qrf_upr_test))
	width = extract_width_stats(get_width_stats(qrf_lwr, qrf_upr))
	coverage_test = get_coverage(y_test, qrf_lwr_test, qrf_upr_test)
	coverage = get_coverage(y, qrf_lwr, qrf_upr)

	base_results = {
		'method': 'qrf',
		'prox_method': np.nan,
		'level': level,
		'random_state': random_state,
		'oob_score_': qrf.oob_score_,
		'k': np.nan,
		'coverage_test': coverage_test,
		'coverage': coverage,
		**{f'{k}_test': v for k, v in width_test.items()},
		**width,
	}

	plot_results = {
		**base_results,
		'lwr_test': qrf_lwr_test, 'pred_test': qrf_pred_test, 'upr_test': qrf_upr_test,
		'lwr': qrf_lwr, 'pred': qrf_pred, 'upr': qrf_upr,
	}

	return qrf, plot_results, base_results


def get_zrf_results(X, y, X_test=None, y_test=None,
					prox_method=None, level=0.95,
					random_state=None, k=None, **kwargs):
	# Zhang RF Intervals
	zrf, zrf_lwr_test, zrf_pred_test, zrf_upr_test = rfintervals(
		X, y, X_test=X_test, y_test=y_test, level=level, random_state=random_state
	)
	zrf_lwr, zrf_pred, zrf_upr = zrf.oob_pred_lwr_, zrf.oob_prediction_, zrf.oob_pred_upr_

	width_test = extract_width_stats(get_width_stats(zrf_lwr_test, zrf_upr_test))
	width = extract_width_stats(get_width_stats(zrf_lwr, zrf_upr))
	coverage_test = get_coverage(y_test, zrf_lwr_test, zrf_upr_test)
	coverage = get_coverage(y, zrf_lwr, zrf_upr)

	base_results = {
		'method': 'zrf',
		'prox_method': np.nan,
		'level': level,
		'random_state': random_state,
		'oob_score_': zrf.oob_score_,
		'k': np.nan,
		'coverage_test': coverage_test,
		'coverage': coverage,
		**{f'{k}_test': v for k, v in width_test.items()},
		**width,
	}

	plot_results = {
		**base_results,
		'lwr_test': zrf_lwr_test, 'pred_test': zrf_pred_test, 'upr_test': zrf_upr_test,
		'lwr': zrf_lwr, 'pred': zrf_pred, 'upr': zrf_upr,
	}

	return zrf, plot_results, base_results


def get_rf_results(X, y, X_test=None, y_test=None,
				   prox_method='rfgap', level=0.95,
				   random_state=None, k='auto', **kwargs):
	# Proximity-Based (RFGAP)
	rf = RFGAP(y=y, random_state=random_state, oob_score=True, **kwargs)
	rf.fit(X, y)

	rf_lwr_test, rf_pred_test, rf_upr_test = rf.predict_with_intervals(X_test=X_test, n_neighbors=k, level=level)
	rf_lwr, rf_pred, rf_upr = rf.oob_pred_lwr_, rf.oob_prediction_, rf.oob_pred_upr_

	width_test = extract_width_stats(get_width_stats(rf_lwr_test, rf_upr_test))
	width = extract_width_stats(get_width_stats(rf_lwr, rf_upr))
	coverage_test = get_coverage(y_test, rf_lwr_test, rf_upr_test)
	coverage = get_coverage(y, rf_lwr, rf_upr)

	base_results = {
		'method': 'proximity_based',
		'prox_method': prox_method,
		'level': level,
		'random_state': random_state,
		'oob_score_': rf.oob_score_,
		'k': k,
		'coverage_test': coverage_test,
		'coverage': coverage,
		**{f'{k}_test': v for k, v in width_test.items()},
		**width,
	}

	plot_results = {
		**base_results,
		'lwr_test': rf_lwr_test, 'pred_test': rf_pred_test, 'upr_test': rf_upr_test,
		'lwr': rf_lwr, 'pred': rf_pred, 'upr': rf_upr,
	}

	return rf, plot_results, base_results
