import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.inspection import partial_dependence
from joblib import Parallel, delayed
from .kernels import Kernel
from .utils import create_kdtree, weighted_mse_objective1
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class GWXGBoost(BaseEstimator, RegressorMixin):

    def __init__(self, coords, feature, target, n_estimators=10, max_depth=3, bandwidth=10.0, kernel='bisquare',
                 criterion='mse', fixed=False, spherical=False, n_jobs=-1, random_state=None, feature_names=None):
        """
        Initialize the GWXGBoost model.
        :param coords: Coordinate data.
        :param feature: Feature data.
        :param target: Target data.
        :param n_estimators: Number of estimators, default is 10.
        :param max_depth: Maximum depth, default is 3.
        :param bandwidth: Bandwidth, default is 10.0.
        :param kernel: Kernel type, default is 'bisquare'.
        :param criterion: Criterion, default is'mse'.
        :param fixed: Whether the bandwidth is fixed, default is False.
        :param spherical: Whether to use spherical metric, default is False.
        :param n_jobs: Number of jobs, default is -1.
        :param random_state: Random state, default is None.
        :param feature_names: Feature names, default is None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.coords = coords
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.fixed = fixed
        self.spherical = spherical
        self.X_, self.Y = check_X_y(feature, target)
        self.n_samples = self.X_.shape[0]
        self.kdtree = None
        self.models = []
        self.residuals = []
        self.feature_names = feature_names

    def _get_kdtree(self):
        """
        Get the kd-tree. If not initialized, create it.
        """
        if self.kdtree is None:
            self.kdtree = create_kdtree(self.coords, n_neighbors=self.n_samples, spherical=self.spherical)

    def _compute_weights(self, i, bandwidth, coords_i=None):
        """
        Compute weights for the samples.
        :param i: Index of the sample.
        :param bandwidth: Bandwidth value.
        :param coords_i: Optional coordinates, default is None.
        :return: Weight array.
        """
        if bandwidth == np.inf:
            wi = np.ones(self.n_samples)
            return wi
        try:
            if self.kdtree is None:
                self._get_kdtree()
            wi = Kernel(i, self.coords, self.kdtree, bw=bandwidth, fixed=self.fixed,
                        function=self.kernel, coords_i=coords_i).kernel
        except BaseException:
            raise
        return wi

    def _local_fit(self, i):
        """
        Fit a local model for a specific sample.
        :param i: Index of the sample.
        :return: Trained XGBoost model and residual.
        """
        weights = self._compute_weights(i, self.bandwidth)

        # Train a weighted model
        # ① Use the sample_weight parameter in XGBRegressor for weighted training
        # Convert numpy array to DataFrame, preserving feature names
        X_train = pd.DataFrame(np.delete(self.X_, i, axis=0), columns=self.feature_names)
        Y_train = np.delete(self.Y, i, axis=0)
        weights_train = np.delete(weights, i, axis=0)

        xgbRegressor = xgb.XGBRegressor(n_estimators=self.n_estimators,
                                        max_depth=self.max_depth,
                                        objective=weighted_mse_objective1,
                                        gamma=0,
                                        min_child_weight=1,
                                        colsample_bytree=1,
                                        reg_alpha=0,
                                        reg_lambda=1,
                                        random_state=self.random_state
                                        )
        xgbRegressor.fit(X_train, Y_train, sample_weight=weights_train)
        y_pred = xgbRegressor.predict(self.X_[i].reshape(1, -1))
        residual = self.Y[i] - y_pred
        return xgbRegressor, residual

    def fit(self):
        """
        Fit the GWXGBoost model.
        :return: The fitted GWXGBoost model.
        """
        local_result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._local_fit)(i) for i in range(self.n_samples)
        )
        local_result_list = list(zip(*local_result))
        self.models = np.array(local_result_list[0]).reshape(-1)
        self.residuals = np.array(local_result_list[1]).reshape(-1)
        return self

    def predict(self, pred_coords, pred_x):
        """
        Predict using the GWXGBoost model.
        :param pred_coords: Coordinates for prediction.
        :param pred_x: Feature data for prediction.
        :return: Predicted values.
        """
        pred_x = check_array(pred_x)
        pred_coords = check_array(pred_coords)
        pred_nums = pred_x.shape[0]
        y_pred = np.zeros(pred_nums)

        for i in range(pred_nums):
            weights = self._compute_weights(i=i, bandwidth=self.bandwidth, coords_i=pred_coords[i])
            weighted_sum = 0
            weight_sum = 0
            for j in range(self.n_samples):
                weighted_sum += weights[j] * self.models[j].predict(pred_x[i].reshape(1, -1))
                weight_sum += weights[j]
            y_pred[i] = weighted_sum / weight_sum

        return y_pred

    def get_local_feature_importance(self, model_index, importance_type='weight'):
        """
        Get the local feature importance of a specific model.
        :param model_index: Index of the model.
        :param importance_type: Importance type, default is 'weight'.
        :return: Local feature importance.
        """
        model = self.models[model_index]
        local_importance = model.get_booster().get_score(importance_type=importance_type)
        return local_importance

    def plot_local_feature_importance(self, model_index, importance_type='weight'):
        """
        Plot the local feature importance of a specific model.
        :param model_index: Index of the model.
        :param importance_type: Importance type, default is 'weight'.
        """
        local_importance = self.get_local_feature_importance(model_index, importance_type=importance_type)
        print(local_importance)
        xgb.plot_importance(local_importance, importance_type=importance_type)
        plt.show()

    def get_global_feature_importance(self, importance_type='weight'):
        """
        Get the global feature importance.
        :param importance_type: Importance type, default is 'weight'.
        :return: Global feature importance.
        """
        feature_importance = np.array(
            [model.get_booster().get_score(importance_type=importance_type) for model in self.models])
        total_importance = {}
        for importance in feature_importance:
            for feature, value in importance.items():
                if feature in total_importance:
                    total_importance[feature] += value
                else:
                    total_importance[feature] = value
        average_importance = {feature: round(value / len(feature_importance), 2) for feature, value in total_importance.items()}
        return average_importance

    def plot_global_feature_importance(self, importance_type='weight'):
        """
        Plot the global feature importance.
        :param importance_type: Importance type, default is 'weight'.
        """
        feature_importance = self.get_global_feature_importance(importance_type=importance_type)
        xgb.plot_importance(feature_importance, importance_type=importance_type)
        plt.show()

    def get_local_partial_dependence(self, model_index, feature_index, grid_size=100):
        """
        Get the local partial dependence of a specific model.
        :param model_index: Index of the model.
        :param feature_index: Index of the feature.
        :param grid_size: Grid size, default is 100.
        :return: Local partial dependence result.
        """
        check_is_fitted(self.models[model_index])
        model = self.models[model_index]
        X = self.X_
        result = partial_dependence(model, features=feature_index, X=X, grid_resolution=grid_size)
        return result

    def plot_local_partial_dependence(self, model_index, feature_index, grid_size=100):
        """
        Plot the local partial dependence of a specific model.
        :param model_index: Index of the model.
        :param feature_index: Index of the feature.
        :param grid_size: Grid size, default is 100.
        """
        result = self.get_local_partial_dependence(model_index, feature_index, grid_size)
        grid_values_list = result['grid_values']
        if len(feature_index) == 1:
            average_list = result['average']
        else:
            average_list = result['average'][0]

        if len(feature_index) == 1:
            plt.plot(grid_values_list[0], average_list[0])
            plt.xlabel(f'{self.feature_names[feature_index[0]]}')
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence for {self.feature_names[feature_index[0]]}')
            plt.show()
        else:
            X1, X2 = grid_values_list[0], grid_values_list[1]
            X1, X2 = np.meshgrid(X1, X2)
            plt.contourf(X1, X2, average_list, cmap='viridis')
            plt.xlabel(f'{self.feature_names[feature_index[0]]}')
            plt.ylabel(f'{self.feature_names[feature_index[1]]}')
            plt.title(
                f'Partial Dependence for {self.feature_names[feature_index[0]]} '
                f'and {self.feature_names[feature_index[1]]}')
            plt.colorbar()
            plt.show()

    def get_global_partial_dependence(self, feature_index, grid_size=100):
        """
        Get the global partial dependence.
        :param feature_index: Index of the feature.
        :param grid_size: Grid size, default is 100.
        :return: Averaged global partial dependence result.
        """
        all_results = []
        for model in self.models:
            check_is_fitted(model)
            result = partial_dependence(model, features=feature_index, X=self.X_, grid_resolution=grid_size)
            all_results.append(result)
        # 计算平均值
        averaged_result = {}
        for key in all_results[0].keys():
            averaged_result[key] = np.mean([result[key] for result in all_results], axis=0)
        return averaged_result

    def plot_global_partial_dependence(self, feature_index, grid_size=100):
        """
        Plot the global partial dependence.
        :param feature_index: Index of the feature.
        :param grid_size: Grid size, default is 100.
        """
        result = self.get_global_partial_dependence(feature_index, grid_size)
        grid_values_list = result['grid_values']

        if len(feature_index) == 1:
            average_list = result['average']
        else:
            average_list = result['average'][0]

        if len(feature_index) == 1:
            plt.plot(grid_values_list[0], average_list[0])
            plt.xlabel(f'{self.feature_names[feature_index[0]]}')
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence for {self.feature_names[feature_index[0]]}')
            plt.show()
        else:
            X1, X2 = grid_values_list[0], grid_values_list[1]
            X1, X2 = np.meshgrid(X1, X2)
            plt.contourf(X1, X2, average_list, cmap='viridis')
            plt.xlabel(f'{self.feature_names[feature_index[0]]}')
            plt.ylabel(f'{self.feature_names[feature_index[1]]}')
            plt.title(
                f'Partial Dependence for {self.feature_names[feature_index[0]]} '
                f'and {self.feature_names[feature_index[1]]}')
            plt.colorbar()
            plt.show()

    def save(self, model_path):
        """
        Save the model to a file.
        :param model_path: Path to save the model.
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, model_path):
        """
        Load the GWRF model from a file.
        :param model_path: Path to the saved model file.
        :return: The loaded GWRF model.
        """
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        if not isinstance(loaded_model, GWXGBoost):
            raise TypeError("The loaded object is not an instance of GWRF.")
        return loaded_model
