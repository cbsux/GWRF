# GWXGBoost
## I. Overview
GWXGBoost (Geographically Weighted XGBoost) is a Python library designed to incorporate the concept of 
geographical weighting into the XGBoost model, enabling regression analysis of data with spatial 
correlation. It integrates powerful tools such as `numpy`, `xgboost`, and `sklearn`, providing users with a 
range of functionalities from model training and prediction to feature importance assessment and partial 
dependence analysis, while also supporting model saving and loading.

## II. Dependency Libraries
Before using the GWXGBoost library, ensure that the following dependency libraries are installed:
```bash
pip install -r requirement.txt
```

## III. Usage of the Code
### i. Training the Model

```python
from GWXGBoost import model

model = GWXGBoost(coords, feature, target, n_estimators=10, max_depth=3, bandwidth=10.0, kernel='bisquare',
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
model.fit()
```
Calling the fit method will train the GWXGBoost model. This method uses Parallel for parallel computation, fitting a local XGBoost model for each sample, calculating weights using spatial information, and performing weighted training.

### iii. Model Prediction
```python
pred = model.predict(pred_coords, pred_x)
```
- `pred_coords`: A two-dimensional array representing the longitude and latitude coordinates of the samples to be predicted.
- `pred_x`: A two-dimensional array representing the feature data of the samples to be predicted.
- `return`: A one-dimensional array representing the prediction results.<br>
Use the predict method to make predictions on new data. The input pred_coords are the coordinates of the prediction data, and pred_x is the feature data of the prediction. For each prediction sample, a weighted average prediction is made based on its weights and the trained local models.

### iv. Feature Importance
```python
# Get and plot local feature importance
local_importance = model.get_local_feature_importance(model_index, importance_type='weight')
model.plot_local_feature_importance(model_index, importance_type='weight')

# Get and plot global feature importance
global_importance = model.get_global_feature_importance(importance_type='weight')
model.plot_global_feature_importance(importance_type='weight')
```
- `model_index`: An integer representing the index of the local model.
- `importance_type`: A string representing the type of feature importance, with optional values of weight, gain, and cover, defaulting to weight.
- `return`: Feature importance.<br>
Use the get_local_feature_importance and plot_local_feature_importance methods to get and plot local feature importance, and use the get_global_feature_importance and plot_global_feature_importance methods to get and plot global feature importance.

### v. Partial Dependence Analysis
```python
# Get and plot local partial dependence
local_partial = model.get_local_partial_dependence(model_index, feature_index)
model.plot_local_partial_dependence(model_index, feature_index)

# Get and plot global partial dependence
global_partial = model.get_global_partial_dependence(feature_index)
model.plot_global_partial_dependence(feature_index)
```
- `model_index`: An integer representing the index of the local model.
- `feature_index`: An integer representing the index of the feature.
- `return`: Partial dependence.<br>
Use the get_local_partial_dependence and plot_local_partial_dependence methods to get and plot local partial dependence, and use the get_global_partial_dependence and plot_global_partial_dependence methods to get and plot global partial dependence.

### vi. Bandwidth Selection
```python
from select_bandwidth import SelectBandwidth
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
- `Initialize` a bandwidth selector using the SelectBandwidth class. The input coords are the longitude and latitude coordinates of the samples, feature is the feature data of the samples, and target is the target data of the samples. n_estimators, max_depth, kernel, criterion, fixed, spherical, n_jobs, and random_state represent the parameters of the XGBoost model , respectively. Call the search method for bandwidth selection, returning the optimal bandwidth.

## IV. Example of Code Usage
Here is an example of using the GWXGBoost library:

```python

import libpysal as ps
import numpy as np
from GWXGBoost import model
from select_bandwidth import SelectBandwidth
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

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
model = GWXGBoost(coords=coords, feature=X, target=y, n_estimators=10,
                  max_depth=3, bandwidth=71.0, kernel='bisquare', fixed=False, spherical=False,
                  n_jobs=4, random_state=1234,
                  feature_names=feature_names)
print("training")
model.fit()
print("predicting")
y_pred = model.predict(coords, X)

print("R2: ", r2_score(y_pred, y))  # 0.832256420036751
print("EV: ", explained_variance_score(y_pred, y))  # 0.8417855829906713
print("MSE: ", mean_squared_error(y_pred, y))  # 3.378118906427371

print("######################################\n"
      "################可视化##################\n"
      "######################################")

model.plot_local_feature_importance(model_index=2)
model.plot_global_feature_importance(importance_type='gain')
model.plot_local_partial_dependence(model_index=0, feature_index=[0, 1])
model.plot_global_partial_dependence(feature_index=[1])
```


#### lobal_feature_importance
![Image text](https://github.com/cbsux/GWXGBoost/blob/master/doc/images/local_feature_importance.png)
#### global_feature_importance
![Image text](https://github.com/cbsux/GWXGBoost/blob/master/doc/images/global_feature_importance.png)
#### local_partial_dependence
![Image text](https://github.com/cbsux/GWXGBoost/blob/master/doc/images/local_partial_dependence.png)
#### global_partial_dependence
![Image text](https://github.com/cbsux/GWXGBoost/blob/master/doc/images/global_partial_dependence.png)


