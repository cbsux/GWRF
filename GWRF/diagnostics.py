"""
Diagnostics for estimated gwrf models
"""
import numpy as np


def calculate_cv_value(gwrf):
    """
    Calculate the cross-validation (CV) value.
    The formula is: CV = sum((y_obj - y_pred)^2) / n
    :param gwrf: The already fitted Geographically Weighted Random Forest model
    :return: the calculated CV value
    """
    # Check if gwrf has necessary attributes
    if not hasattr(gwrf, 'n_samples') or not hasattr(gwrf, 'residuals'):
        raise AttributeError("gwrf must have 'n_samples' and 'residuals' attributes.")
    n_samples = gwrf.n_samples
    # Convert residuals to numpy array and perform vectorized square operation
    residuals = np.array(gwrf.residuals)
    cv_value = np.sum(residuals ** 2)
    return cv_value / n_samples
