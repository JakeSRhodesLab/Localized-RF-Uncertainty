# Import all methods, broken down by classification and regression

from .get_regression_results import get_qrf_results, get_zrf_results, get_rf_results
from .get_classification_results import get_classification_results


classification_methods = {
    'rf': get_classification_results,
}

regression_methods = {
    'qrf': get_qrf_results,
    'zrf': get_zrf_results,
    'rf': get_rf_results
}


