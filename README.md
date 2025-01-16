# GWXGBoost
## I. Overview
GWXGBoost (Geographically Weighted XGBoost) is a Python library designed to incorporate the concept of 
geographical weighting into the XGBoost model, enabling regression analysis of data with spatial 
correlation. It integrates powerful tools such as `numpy`, `xgboost`, and `sklearn`, providing users with a 
range of functionalities from model training and prediction to feature importance assessment and partial 
dependence analysis, while also supporting model saving and loading.

## II. Dependency Libraries
① If you want to directly use the Python library related to GWXGBoost for regression, you can install it directly using the following command:
```bash
pip install GWXGBoost
```

② If you want to modify, innovate and learn using relevant code, you can clone the code to your local machine:
```bash
 git clone https://github.com/cbsux/GWXGBoost.git
```
Then install it in the relevant directory:
```bash
pip install -r requirements.txt
```

## III. Usage 
The GWXGBoost library contains the following several modules:
 - `model`: Training, prediction, feature importance and local dependency analysis of the GWXGBoost model.
 - `select_bandwidth`: Defines the function of optimal bandwidth selection, which is selected by obtaining the optimal cross-validation value.
 - `kernels`: Provides kernel function calculations for geographically weighted models, supports multiple kernel function types, and calculates weight values based on bandwidth and distance, etc.
 - `search`: Provides two bandwidth search methods, the golden section method and the equal-spacing search method, for selecting the optimal bandwidth value.
 - `utils`: Provides some utility functions, such as the calculation of spherical distance, the instantiation of KDTree, the customization of weighted objective functions, etc.<br>
GWXGBoost provides a series of functions for spatial data regression analysis. The following part will introduce how to use the GWXGBoost library for model training, prediction, feature importance and local dependency analysis, etc.
### i. Model Initialization
```python
from GWXGBoost.model import GWXGBoost
gwModel = GWXGBoost(coords, feature, target, n_estimators=10, max_depth=3, bandwidth=10.0, kernel='bisquare',
                  criterion='mse', fixed=False, spherical=False, n_jobs=-1, random_state=None, feature_names=None)
```
- `coords`: A two-dimensional array representing the longitude and latitude coordinates of the samples.
- `feature`: A two-dimensional array representing the feature data of the samples.
- `target`: A one-dimensional array representing the target data of the samples.
- `n_estimators`: An integer representing the number of trees in XGBoost, defaulting to 10.
- `max_depth`: An integer representing the maximum depth of each tree, defaulting to 3.
- `bandwidth`: A floating-point number representing the bandwidth in geographical weighting, defaulting to 10.0.
- `kernel`: A string representing the kernel function in geographical weighting, with optional values of gaussian, exponential, bisquare, and tricube, defaulting to bisquare.
- `criterion`: A string representing the evaluation criterion in the random forest, with optional values of mse and mae, defaulting to mse.
- `fixed`: A boolean value indicating whether to use a fixed bandwidth, defaulting to False.
- `spherical`: A boolean value indicating whether to use spherical distance, defaulting to False.
- `n_jobs`: An integer representing the number of parallel computations, defaulting to -1.
- `random_state`: An integer representing the random seed, defaulting to None.
- `feature_names`: A list representing the names of the features, defaulting to None.
- `return`: The GWXGBoost model.<br>
Initialize a GWXGBoost model using the GWXGBoost class. The input coords are the longitude and latitude coordinates of the samples, feature is the feature data of the samples, and target is the target data of the samples. n_estimators, max_depth, bandwidth, kernel, criterion, fixed, spherical, n_jobs, and random_state represent the parameters of the XGBoost model, respectively. feature_names are the names of the features.

### ii. Model Training
```python
gwModel.fit()
```
Calling the `fit()` method will train the GWXGBoost model. This method uses Parallel for parallel computation. For each sample, it calculates weights using spatial information (longitude and latitude), and then fits a local XGBoost model. These local models are trained with weighted data, taking into account the spatial proximity of samples.
### iii. Model Prediction
```python
pred = gwModel.predict(pred_coords, pred_x)
```
- `pred_coords`: A two-dimensional array representing the longitude and latitude coordinates of the samples to be predicted.
- `pred_x`: A two-dimensional array representing the feature data of the samples to be predicted.
- `return`: A one-dimensional array representing the prediction results.<br>
Use the `predict()` method to make predictions on new data. The input `pred_coords` are the coordinates of the prediction data, and `pred_x` is the feature data of the prediction. For each prediction sample, a weighted average prediction is made based on its weights (calculated using spatial information) and the trained local models.
### iv. Feature Importance
```python
# Get and plot local feature importance
local_importance = gwModel.get_local_feature_importance(model_index, importance_type='weight')
gwModel.plot_local_feature_importance(model_index, importance_type='weight')

# Get and plot global feature importance
global_importance = gwModel.get_global_feature_importance(importance_type='weight')
gwModel.plot_global_feature_importance(importance_type='weight')
```
- `model_index`: An integer representing the index of the local model.
- `importance_type`: A string representing the type of feature importance, with optional values of weight, gain, and cover, defaulting to weight.
- `return`: Feature importance.<br>
 the get_local_feature_importance and plot_local_feature_importance methods to get and plot local feature importance, and use the get_global_feature_importance and plot_global_feature_importance methods to get and plot global feature importance.

### v. Partial Dependence Analysis
```python
# Get and plot local partial dependence
local_partial = gwModel.get_local_partial_dependence(model_index, feature_index)
gwModel.plot_local_partial_dependence(model_index, feature_index)

# Get and plot global partial dependence
global_partial = gwModel.get_global_partial_dependence(feature_index)
gwModel.plot_global_partial_dependence(feature_index)
```
- `model_index`: An integer representing the index of the local model.
- `feature_index`: An integer representing the index of the feature.
- `return`: Partial dependence.<br>
Use the get_local_partial_dependence and plot_local_partial_dependence methods to get and plot local partial dependence, and use the get_global_partial_dependence and plot_global_partial_dependence methods to get and plot global partial dependence.

### vi. Bandwidth Selection
```python
from GWXGBoost.select_bandwidth import SelectBandwidth
sele = SelectBandwidth(coords=coords, feature=X, target=y, n_estimators=10, max_depth=4, kernel='gaussian',
                      criterion='mse', fixed=False, spherical=False, n_jobs=4, random_state=1234).search(verbose=True)
print(bw)
```
- `coords`: A two-dimensional array representing the longitude and latitude coordinates of the samples.
- `feature`: A two-dimensional array representing the feature data of the samples.
- `target`: A one-dimensional array representing the target data of the samples.
- `n_estimators`: An integer representing the number of trees in XGBoost, defaulting to 10.
- `max_depth`: An integer representing the maximum depth of each tree, defaulting to 4.
- `kernel`: A string representing the kernel function in geographical weighting, with optional values of gaussian, exponential, bisquare, and tricube, defaulting to gaussian.
- `criterion`: A string representing the evaluation criterion in the random forest, with optional values of mse and mae, defaulting to mse.
- `fixed`: A boolean value indicating whether to use a fixed bandwidth, defaulting to False.
- `spherical`: A boolean value indicating whether to use spherical distance, defaulting to False.
- `n_jobs`: An integer representing the number of parallel computations, defaulting to 4.
- `random_state`: An integer representing the random seed, defaulting to 1234.
- `return`: A floating-point number representing the optimal bandwidth.<br>
Initialize a bandwidth selector using the `SelectBandwidth` class. The input `coords` are the longitude and latitude coordinates of the samples, `feature` is the feature data of the samples, and `target` is the target data of the samples. `n_estimators`, `max_depth`, `kernel`, `criterion`, `fixed`, `spherical`, `n_jobs`, and `random_state` represent the parameters of the XGBoost model, respectively. Call the `search` method for bandwidth selection, which returns the optimal bandwidth through cross-validation.
## IV. Example of Code Usage
Here is an example of using the GWXGBoost library:

```python

import libpysal as ps
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from GWXGBoost.select_bandwidth import SelectBandwidth
from GWXGBoost.model import GWXGBoost

data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
coords = np.array(list(zip(data.by_col('X'), data.by_col('Y'))))
y = np.array(data.by_col('PctBach')).reshape(-1)
rural = np.array(data.by_col('PctRural')).reshape((-1, 1))
pov = np.array(data.by_col('PctPov')).reshape((-1, 1))
african_amer = np.array(data.by_col('PctBlack')).reshape((-1, 1))
feature_names = ['PctRural', 'PctPov', 'PctBlack']
X = np.hstack([rural, pov, african_amer])

print("######################################\n"
      "############Select_Bandwidth##########\n"
      "######################################")
bw = SelectBandwidth(coords=coords, feature=X, target=y, n_estimators=10, max_depth=3,
                     kernel='gaussian', criterion='mse', fixed=False, spherical=False, n_jobs=4,
                     random_state=1234).search(verbose=True)
print(bw)  # 71.0

print("######################################\n"
      "################GWXGBoost##################\n"
      "######################################")
gwModel = GWXGBoost(coords=coords, feature=X, target=y, n_estimators=10,
                  max_depth=3, bandwidth=71.0, kernel='bisquare', fixed=False, spherical=False,
                  n_jobs=4, random_state=1234,
                  feature_names=feature_names)
print("training")
gwModel.fit()
print("predicting")
y_pred = gwModel.predict(coords, X)

print("R2: ", r2_score(y_pred, y))  # 0.832256420036751
print("EV: ", explained_variance_score(y_pred, y))  # 0.8417855829906713
print("MSE: ", mean_squared_error(y_pred, y))  # 3.378118906427371

print("######################################\n"
      "################可视化##################\n"
      "######################################")

gwModel.plot_local_feature_importance(model_index=2)
gwModel.plot_global_feature_importance(importance_type='gain')
gwModel.plot_local_partial_dependence(model_index=0, feature_index=[0, 1])
gwModel.plot_global_partial_dependence(feature_index=[1])
```


#### lobal_feature_importance
![Image text](https://raw.githubusercontent.com/cbsux/GWXGBoost/master/doc/images/local_feature_importance.png)
This image shows the local feature importance, indicating the importance of features for a specific local model. It helps identify which features are more influential in different geographic regions.
#### global_feature_importance
![Image text](https://raw.githubusercontent.com/cbsux/GWXGBoost/master/doc/images/global_feature_importance.png)
This image shows the global feature importance, giving an overall view of feature importance across all models, helping to understand which features are most important in the entire dataset.
#### local_partial_dependence
![Image text](https://raw.githubusercontent.com/cbsux/GWXGBoost/master/doc/images/local_partial_dependence.png)
This image shows the local partial dependence, indicating how the target variable changes with a particular feature in a local model, providing insights into local relationships.
#### global_partial_dependence
![Image text](https://raw.githubusercontent.com/cbsux/GWXGBoost/master/doc/images/global_partial_dependence.png)
This image shows the global partial dependence, showing the relationship between a feature and the target variable across the entire dataset, helping to understand the overall relationship.


