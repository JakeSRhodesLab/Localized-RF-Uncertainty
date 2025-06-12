import numpy as np

def is_in_interval(y, lwr, upr):
    """Check if y is within [lwr, upr] interval (inclusive)."""
    return np.logical_and(y >= lwr, y <= upr)

def get_coverage(y, y_lwr, y_upr):
    """Return the proportion of y values within [y_lwr, y_upr] intervals."""
    return np.mean(is_in_interval(y, y_lwr, y_upr))

def get_width_stats(y_lwr, y_upr):
    """Return summary statistics (mean, std, min, Q1, median, Q3, max) of interval widths."""
    widths = y_upr - y_lwr
    return (
        np.mean(widths),
        np.std(widths),
        np.min(widths),
        np.quantile(widths, 0.25),
        np.median(widths),
        np.quantile(widths, 0.75),
        np.max(widths),
    )
