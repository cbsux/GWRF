from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import xgboost as xgb


# Define the spherical distance calculation function
def haversine_metric(x, y):
    """
    Convert latitude and longitude to radians and calculate the spherical distance.
    :param x: Coordinates array.
    :param y: Coordinates array.
    :return: Spherical distance array.
    """
    x_rad = np.radians(x).reshape(-1, 2)
    y_rad = np.radians(y).reshape(-1, 2)
    return haversine_distances(x_rad, y_rad) * 6371.0  # Earth's radius is approximately 6371 kilometers


# Create a NearestNeighbors instance using existing dat
# Create a NearestNeighbors instance using existing data
def create_kdtree(coords, algorithm='kd_tree', n_neighbors=15, leaf_size=40, spherical=False):
    """
    Create a NearestNeighbors instance using existing data.
    :param coords: Coordinates data.
    :param algorithm: Algorithm for NearestNeighbors, default is 'kd_tree'.
    :param n_neighbors: Number of nearest neighbors, default is 15.
    :param leaf_size: Leaf size, default is 40.
    :param spherical: Whether to use spherical metric, default is False.
    :return: NearestNeighbors instance.
    """
    try:
        if spherical:
            kdtree = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,
                                      leaf_size=leaf_size, metric=haversine_metric)
        else:
            kdtree = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,
                                      leaf_size=leaf_size, metric='euclidean')
        kdtree.fit(coords)
        return kdtree
    except Exception as e:
        print(f"Error occurred during kdtree creation: {e}")


# Custom weighted objective function
def weighted_mse_objective2(preds: np.ndarray, dtrain: xgb.DMatrix):
    """
    Custom weighted objective function for XGBoost.
    :param preds: Predicted values.
    :param dtrain: XGBoost DMatrix object.
    :return: Gradient and Hessian arrays.
    """
    y = dtrain.get_label()
    weights = dtrain.get_weight()
    grad = 2 * weights * (preds - y)
    hess = 2 * weights
    return grad, hess


def weighted_mse_objective1(y_true, y_pred, sample_weight=None):
    """
    Another custom weighted objective function.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param sample_weight: Sample weights, default is None.
    :return: Gradient and Hessian arrays.
    """
    # Ensure sample_weight is not None, if it is None, use all weights as 1
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    residual = y_pred - y_true
    grad = 2 * residual * sample_weight
    hess = 2 * sample_weight
    return grad, hess


# Custom weighted mean squared error evaluation metric
def weighted_mse_metric(preds: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    weights = dtrain.get_weight()
    weighted_mse = np.mean(weights * (preds - y) ** 2)
    return 'weighted_mse', weighted_mse
