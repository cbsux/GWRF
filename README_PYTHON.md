# GWRF 库

## 一、概述
GWRF（Geographically Weighted Random Forest）是一个 Python 库，旨在将地理加权概念融入随机森林模型，
实现对具有空间相关性数据的回归分析。它集成了 `numpy`、`xgboost` 和 `sklearn` 等强大工具，为用户提供了从模型训练、预测到特征重要性评估和部分依赖分析等一系列功能，同时支持模型的保存和加载。

## 二、依赖库
在使用 GWRF 库之前，请确保已安装以下依赖库：
```bash
pip install -r requirement.txt
```
安装相关依赖库后，可以直接安装 GWRF 库：
```bash
pip install gwrf
```

## 三、代码的使用方法
### 1. 训练模型
```python
from gwrf import GWRF
model = GWRF(coords, feature, target, n_estimators=10, max_depth=3, bandwidth=10.0, kernel='bisquare',
           criterion='mse', fixed=False, spherical=False, n_jobs=-1, random_state=None, feature_names=None)
```
- `coords`：二维数组，表示样本的经纬度坐标。
- `feature`：二维数组，表示样本的特征数据。
- `target`：一维数组，表示样本的目标数据。
- `n_estimators`：整数，表示XGBoost中树的数量，默认为 10。
- `max_depth`：整数，表示每棵树的最大深度，默认为 3。
- `bandwidth`：浮点数，表示地理加权中的带宽，默认为 10.0。
- `kernel`：字符串，表示地理加权中的核函数，可选值为 `gaussian`、`exponential`、`bisquare` 和 `tricube`，默认为 `bisquare`。
- `criterion`：字符串，表示随机森林中的评估标准，可选值为 `mse` 和 `mae`，默认为 `mse`。
- `fixed`：布尔值，表示是否使用固定带宽，默认为 `False`。
- `spherical`：布尔值，表示是否使用球面距离，默认为 `False`。
- `n_jobs`：整数，表示并行计算的数量，默认为 `-1`。
- `random_state`：整数，表示随机种子，默认为 `None`。
- `feature_names`：列表，表示特征的名称，默认为 `None`。
- `return`：GWRF 模型。<br>
使用`GWRF`类初始化一个 GWRF 模型。输入`coords`是样本的经纬度坐标，`feature`是样本的特征数据，`target`是样本的目标数据。`n_estimators`、`max_depth`、`bandwidth`、`kernel`、`criterion`、`fixed`、`spherical`、`n_jobs` 和 `random_state` 分别表示 XGBoost 模型和 GWRF 模型的参数。`feature_names` 是特征的名称。
### 2. 模型训练
```python
model.fit()
```
调用`fit`方法将对`GWRF`模型进行训练。该方法使用`Parallel`并行计算，为每个样本拟合一个局部`XGBoost`模型，利用空间信息计算权重并进行加权训练。

### 3. 模型预测
```python
pred = model.predict(pred_coords, pred_x)
```
- `pred_coords`：二维数组，表示待预测样本的经纬度坐标。
- `pred_x`：二维数组，表示待预测样本的特征数据。
- `return`：一维数组，表示预测结果。<br>
使用`predict`方法对新数据进行预测。输入`pred_coords`是预测数据的坐标，`pred_x` 是预测数据的特征。对于每个预测样本，根据其权重和已训练的局部模型进行加权平均预测。

### 4. 特征重要性
```python
# 获取和绘制局部特征重要性
local_importance = model.get_local_feature_importance(model_index, importance_type='weight')
model.plot_local_feature_importance(model_index, importance_type='weight')

# 获取和绘制全局特征重要性
global_importance = model.get_global_feature_importance(importance_type='weight')
model.plot_global_feature_importance(importance_type='weight')
```
- `model_index`：整数，表示局部模型的索引。
- `importance_type`：字符串，表示特征重要性的类型，可选值为 `weight`、`gain` 和 `cover`，默认为 `weight`。
- `return`：特征重要性。<br>
使用`get_local_feature_importance`和`plot_local_feature_importance`方法获取和绘制局部特征重要性，使用`get_global_feature_importance`和`plot_global_feature_importance`方法获取和绘制全局特征重要性。

### 5. 部分依赖分析
```python
# 获取和绘制局部部分依赖
local_partial = model.get_local_partial_dependence(model_index, feature_index)
model.plot_local_partial_dependence(model_index, feature_index)

# 获取和绘制全局部分依赖
global_partial = model.get_global_partial_dependence(feature_index)
model.plot_global_partial_dependence(feature_index)
```
- `model_index`：整数，表示局部模型的索引。
- `feature_index`：整数，表示特征的索引。
- `return`：偏依赖性。<br>
使用`get_local_partial_dependence`和`plot_local_partial_dependence`方法获取和绘制局部部分依赖，使用`get_global_partial_dependence`和`plot_global_partial_dependence`方法获取和绘制全局部分依赖。

### 6. 带宽选择
```python
from select_bandwidth import SelectBandwidth
sele = SelectBandwidth(coords=coords, feature=X, target=y, n_estimators=10, max_depth=4, kernel='gaussian',
                      criterion='mse', fixed=False, spherical=False, n_jobs=4, random_state=1234).search(verbose=True)
print(bw)
```
- `coords`：二维数组，表示样本的经纬度坐标。
- `feature`：二维数组，表示样本的特征数据。
- `target`：一维数组，表示样本的目标数据。
- `n_estimators`：整数，表示XGBoost中树的数量，默认为 10。
- `max_depth`：整数，表示每棵树的最大深度，默认为 4。
- `kernel`：字符串，表示地理加权中的核函数，可选值为 `gaussian`、`exponential`、`bisquare` 和 `tricube`，默认为 `gaussian`。
- `criterion`：字符串，表示随机森林中的评估标准，可选值为 `mse` 和 `mae`，默认为 `mse`。
- `fixed`：布尔值，表示是否使用固定带宽，默认为 `False`。
- `spherical`：布尔值，表示是否使用球面距离，默认为 `False`。
- `n_jobs`：整数，表示并行计算的数量，默认为 `4`。
- `random_state`：整数，表示随机种子，默认为 `1234`。
- `return`：浮点数，表示最优带宽。<br>
使用`SelectBandwidth`类初始化一个带宽选择器。输入`coords`是样本的经纬度坐标，`feature`是样本的特征数据，`target`是样本的目标数据。`n_estimators`、`max_depth`、`kernel`、`criterion`、`fixed`、`spherical`、`n_jobs` 和 `random_state` 分别表示 XGBoost 模型和 GWRF 模型的参数。调用`search`方法进行带宽选择，返回最优带宽。


## 四、代码使用示例
下面是一个使用 GWRF 库的示例：
```python
import libpysal as ps
import numpy as np
from gwrf import GWRF
from select_bandwidth import SelectBandwidth
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

# 从libpysal加载相关数据，进行预处理
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
print(bw) # 71.0


print("######################################\n"
      "################GWRF##################\n"
      "######################################")
model = GWRF(coords=coords, feature=X, target=y, n_estimators=10,
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
model.plot_local_partial_dependence(model_index=0, feature_index=[0,1])
model.plot_global_partial_dependence(feature_index=[1])
```