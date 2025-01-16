"""
Diagnostics for estimated gwXGBoost models
"""
import numpy as np


def calculate_cv_value(gwXGBoost):
    """
    Calculate the cross-validation (CV) value.
    The formula is: CV = sum((y_obj - y_pred)^2) / n
    :param gwXGBoost: The already fitted Geographically Weighted XGBoost model
    :return: the calculated CV value
    """
    # Check if gwXGBoost has necessary attributes
    if not hasattr(gwXGBoost, 'n_samples') or not hasattr(gwXGBoost, 'residuals'):
        raise AttributeError("gwXGBoost must have 'n_samples' and 'residuals' attributes.")
    n_samples = gwXGBoost.n_samples
    # Convert residuals to numpy array and perform vectorized square operation
    residuals = np.array(gwXGBoost.residuals)
    cv_value = np.sum(residuals ** 2)
    return cv_value / n_samples
