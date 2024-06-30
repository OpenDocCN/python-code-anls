# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_outlier_detection_bench.py`

```
"""
==========================================
Evaluation of outlier detection estimators
==========================================

This example compares two outlier detection algorithms, namely
:ref:`local_outlier_factor` (LOF) and :ref:`isolation_forest` (IForest), on
real-world datasets available in :class:`sklearn.datasets`. The goal is to show
that different algorithms perform well on different datasets and contrast their
training speed and sensitivity to hyperparameters.

The algorithms are trained (without labels) on the whole dataset assumed to
contain outliers.

1. The ROC curves are computed using knowledge of the ground-truth labels
and displayed using :class:`~sklearn.metrics.RocCurveDisplay`.

2. The performance is assessed in terms of the ROC-AUC.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Dataset preprocessing and model training
# ========================================
#
# Different outlier detection models require different preprocessing. In the
# presence of categorical variables,
# :class:`~sklearn.preprocessing.OrdinalEncoder` is often a good strategy for
# tree-based models such as :class:`~sklearn.ensemble.IsolationForest`, whereas
# neighbors-based models such as :class:`~sklearn.neighbors.LocalOutlierFactor`
# would be impacted by the ordering induced by ordinal encoding. To avoid
# inducing an ordering, one should rather use
# :class:`~sklearn.preprocessing.OneHotEncoder`.
#
# Neighbors-based models may also require scaling of the numerical features (see
# for instance :ref:`neighbors_scaling`). In the presence of outliers, a good
# option is to use a :class:`~sklearn.preprocessing.RobustScaler`.

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)


def make_estimator(name, categorical_columns=None, iforest_kw=None, lof_kw=None):
    """Create an outlier detection estimator based on its name."""
    # 根据传入的算法名称选择相应的异常检测器
    if name == "LOF":
        # 如果选择的算法是 LOF，则使用 LocalOutlierFactor，并根据参数配置创建实例
        outlier_detector = LocalOutlierFactor(**(lof_kw or {}))
        # 如果没有指定分类变量列，则使用 RobustScaler 进行预处理
        if categorical_columns is None:
            preprocessor = RobustScaler()
        else:
            # 如果有分类变量列，使用 ColumnTransformer 对分类变量进行独热编码处理，
            # 其余的数值特征使用 RobustScaler 进行预处理
            preprocessor = ColumnTransformer(
                transformers=[("categorical", OneHotEncoder(), categorical_columns)],
                remainder=RobustScaler(),
            )
    else:  # 如果name == "IForest"，执行以下代码块
        # 使用传入的iforest_kw参数（如果没有则为空字典），创建IsolationForest异常检测器对象
        outlier_detector = IsolationForest(**(iforest_kw or {}))
        
        # 如果categorical_columns为None，则预处理器为None
        if categorical_columns is None:
            preprocessor = None
        else:
            # 使用OrdinalEncoder对分类列进行编码，处理未知值为指定值-1
            ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            # 创建ColumnTransformer对象，用于对数据进行列变换
            preprocessor = ColumnTransformer(
                transformers=[
                    # 对分类列使用ordinal_encoder进行变换
                    ("categorical", ordinal_encoder, categorical_columns),
                ],
                # 对剩余的列不做任何变换，直接传递
                remainder="passthrough",
            )

    # 返回一个Pipeline对象，包含预处理器和异常检测器
    return make_pipeline(preprocessor, outlier_detector)
# %%
# 下面的 `fit_predict` 函数返回 X 的平均异常值分数。

from time import perf_counter


def fit_predict(estimator, X):
    tic = perf_counter()
    # 如果 estimator 的最后一个模型是 LocalOutlierFactor，则进行拟合并获取异常值分数
    if estimator[-1].__class__.__name__ == "LocalOutlierFactor":
        estimator.fit(X)
        y_pred = estimator[-1].negative_outlier_factor_
    else:  # 如果是 IsolationForest，则直接获取决策函数的输出作为异常值分数
        y_pred = estimator.fit(X).decision_function(X)
    toc = perf_counter()
    # 打印执行时间
    print(f"Duration for {model_name}: {toc - tic:.2f} s")
    return y_pred


# %%
# 在本例中，我们每个部分处理一个数据集。在加载数据后，目标被修改为包含两类：
# 0 代表正常值，1 代表异常值。由于 scikit-learn 文档的计算限制，某些数据集的样本
# 大小通过分层的 :class:`~sklearn.model_selection.train_test_split` 进行了减少。
#
# 此外，我们将 `n_neighbors` 设置为预期异常值数量的估计值
# `expected_n_anomalies = n_samples * expected_anomaly_fraction`。
# 这是一个很好的启发式方法，只要异常值的比例不是非常低，原因是 `n_neighbors`
# 应该至少大于较少人口的簇中的样本数（参见
# :ref:`sphx_glr_auto_examples_neighbors_plot_lof_outlier_detection.py`）。

# KDDCup99 - SA 数据集
# ---------------------
#
# :ref:`kddcup99_dataset` 是使用封闭网络和手动注入攻击生成的。SA 数据集是它的一个子集，
# 通过简单选择所有正常数据和约 3% 的异常比例获得。

# %%
import numpy as np

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split

X, y = fetch_kddcup99(
    subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True
)
y = (y != b"normal.").astype(np.int32)
X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

n_samples, anomaly_frac = X.shape[0], y.mean()
print(f"{n_samples} datapoints with {y.sum()} anomalies ({anomaly_frac:.02%})")

# %%
# SA 数据集包含 41 个特征，其中 3 个是分类特征:
# "protocol_type", "service" 和 "flag".

# %%
y_true = {}
y_pred = {"LOF": {}, "IForest": {}}
model_names = ["LOF", "IForest"]
cat_columns = ["protocol_type", "service", "flag"]

y_true["KDDCup99 - SA"] = y
for model_name in model_names:
    model = make_estimator(
        name=model_name,
        categorical_columns=cat_columns,
        lof_kw={"n_neighbors": int(n_samples * anomaly_frac)},
        iforest_kw={"random_state": 42},
    )
    y_pred[model_name]["KDDCup99 - SA"] = fit_predict(model, X)

# %%
# Forest covertypes 数据集
# -------------------------
#
# :ref:`covtype_dataset` 是一个多类数据集，其中目标是给定森林区域中主要树种。它包含
# 54 个特征，其中一些特征 ("Wilderness_Area" 和 "Soil_Type") 已经进行了二进制编码。
# Though originally meant as a classification task, one can regard inliers as
# samples encoded with label 2 and outliers as those with label 4.

# %%
# 导入必要的库和模块
from sklearn.datasets import fetch_covtype

# 从covtype数据集中获取特征X和标签y，并将数据框作为返回值
X, y = fetch_covtype(return_X_y=True, as_frame=True)

# 筛选出标签为2和4的样本索引，并将对应的X和y值保存
s = (y == 2) + (y == 4)
X = X.loc[s]
y = y.loc[s]

# 将标签y转换为0和1，其中0表示inliers，1表示outliers
y = (y != 2).astype(np.int32)

# 将数据集X和y分割为训练集和测试集，保留5%的训练集数据，并使用stratify方式分层抽样
X, _, y, _ = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)

# 保存X用于后续使用
X_forestcover = X

# 计算样本数量和异常数据比例，并打印相关信息
n_samples, anomaly_frac = X.shape[0], y.mean()
print(f"{n_samples} datapoints with {y.sum()} anomalies ({anomaly_frac:.02%})")

# %%
# 将forestcover数据集的真实标签y存储到y_true字典中，并对每个模型进行预测
y_true["forestcover"] = y
for model_name in model_names:
    # 根据模型名称创建估计器，设置LOF参数和iForest参数
    model = make_estimator(
        name=model_name,
        lof_kw={"n_neighbors": int(n_samples * anomaly_frac)},
        iforest_kw={"random_state": 42},
    )
    # 使用估计器拟合数据并预测标签，存储到y_pred字典中
    y_pred[model_name]["forestcover"] = fit_predict(model, X)

# %%
# Ames Housing dataset
# --------------------
#
# The `Ames housing dataset <http://www.openml.org/d/43926>`_ is originally a
# regression dataset where the target are sales prices of houses in Ames, Iowa.
# Here we convert it into an outlier detection problem by regarding houses with
# price over 70 USD/sqft. To make the problem easier, we drop intermediate
# prices between 40 and 70 USD/sqft.

# %%
# 导入matplotlib.pyplot库
import matplotlib.pyplot as plt

# 从openml获取Ames housing数据集的特征X和目标y，并将数据框作为返回值
from sklearn.datasets import fetch_openml
X, y = fetch_openml(name="ames_housing", version=1, return_X_y=True, as_frame=True)

# 计算每个房屋的价格每平方英尺，并更新目标y
y = y.div(X["Lot_Area"])

# 处理pandas版本变化后的空值映射
X["Misc_Feature"] = X["Misc_Feature"].cat.add_categories("NoInfo").fillna("NoInfo")
X["Mas_Vnr_Type"] = X["Mas_Vnr_Type"].cat.add_categories("NoInfo").fillna("NoInfo")

# 删除Lot_Area列，并根据房价选择40到70之外的数据
X.drop(columns="Lot_Area", inplace=True)
mask = (y < 40) | (y > 70)
X = X.loc[mask]
y = y.loc[mask]

# 绘制房价分布直方图
y.hist(bins=20, edgecolor="black")
plt.xlabel("House price in USD/sqft")
_ = plt.title("Distribution of house prices in Ames")

# %%
# 将y转换为二进制标签，大于70USD/sqft的房屋标记为1，其他为0
y = (y > 70).astype(np.int32)

# 计算样本数量和异常数据比例，并打印相关信息
n_samples, anomaly_frac = X.shape[0], y.mean()
print(f"{n_samples} datapoints with {y.sum()} anomalies ({anomaly_frac:.02%})")

# %%
# The dataset contains 46 categorical features. In this case it is easier use a
# :class:`~sklearn.compose.make_column_selector` to find them instead of passing
# a list made by hand.

# %%
# 导入make_column_selector函数
from sklearn.compose import make_column_selector as selector

# 创建选择器以查找数据集中的分类列
categorical_columns_selector = selector(dtype_include="category")
cat_columns = categorical_columns_selector(X)

# 将ames_housing数据集的真实标签y存储到y_true字典中，并对每个模型进行预测
y_true["ames_housing"] = y
for model_name in model_names:
    # 根据模型名称创建估计器，设置分类列、LOF参数和iForest参数
    model = make_estimator(
        name=model_name,
        categorical_columns=cat_columns,
        lof_kw={"n_neighbors": int(n_samples * anomaly_frac)},
        iforest_kw={"random_state": 42},
    )
    # 使用估计器拟合数据并预测标签，存储到y_pred字典中
    y_pred[model_name]["ames_housing"] = fit_predict(model, X)

# %%
# Cardiotocography dataset
# ------------------------
#
# The `Cardiotocography dataset <http://www.openml.org/d/1466>`_ is a multiclass
# 导入必要的库和模块
import math
from sklearn.metrics import RocCurveDisplay

# 定义绘图参数
cols = 2
pos_label = 0  # 正类标签为0
datasets_names = y_true.keys()
rows = math.ceil(len(datasets_names) / cols)

# 创建绘图窗口
fig, axs = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(10, rows * 4))

# 遍历数据集名称和模型名称，生成ROC曲线图
for ax, dataset_name in zip(axs.ravel(), datasets_names):
    for model_idx, model_name in enumerate(model_names):
        # 使用真实标签和预测标签绘制ROC曲线
        display = RocCurveDisplay.from_predictions(
            y_true[dataset_name],
            y_pred[model_name][dataset_name],
            pos_label=pos_label,
            name=model_name,
            ax=ax,
            # 如果是最后一个模型，则绘制随机分类的机会水平
            plot_chance_level=(model_idx == len(model_names) - 1),
            chance_level_kw={"linestyle": ":"},
        )
    ax.set_title(dataset_name)

# 调整子图布局，设置间距
_ = plt.tight_layout(pad=2.0)
# %%
# 从 forestcover 数据集中获取特征 X 和目标变量 y_true["forestcover"]
X = X_forestcover
y = y_true["forestcover"]

# 获取样本数量
n_samples = X.shape[0]

# 根据样本数量计算 n_neighbors_list，即 LOF 算法中的邻居数列表
n_neighbors_list = (n_samples * np.array([0.2, 0.02, 0.01, 0.001])).astype(np.int32)

# 创建包含 RobustScaler 和 LocalOutlierFactor 的管道模型
model = make_pipeline(RobustScaler(), LocalOutlierFactor())

# 定义不同线条样式的列表
linestyles = ["solid", "dashed", "dashdot", ":", (5, (10, 3))]

# 创建图表和坐标轴
fig, ax = plt.subplots()

# 遍历 linestyles 和 n_neighbors_list，进行模型拟合和可视化
for model_idx, (linestyle, n_neighbors) in enumerate(zip(linestyles, n_neighbors_list)):
    # 设置 LocalOutlierFactor 的 n_neighbors 参数
    model.set_params(localoutlierfactor__n_neighbors=n_neighbors)
    
    # 拟合模型
    model.fit(X)
    
    # 获取 LOF 算法的负异常因子作为预测分数
    y_pred = model[-1].negative_outlier_factor_
    
    # 绘制 ROC 曲线并显示
    display = RocCurveDisplay.from_predictions(
        y,
        y_pred,
        pos_label=pos_label,
        name=f"n_neighbors = {n_neighbors}",
        ax=ax,
        plot_chance_level=(model_idx == len(n_neighbors_list) - 1),
        chance_level_kw={"linestyle": (0, (1, 10))},
        linestyle=linestyle,
        linewidth=2,
    )

# 设置图表标题
_ = ax.set_title("RobustScaler with varying n_neighbors\non forestcover dataset")

# %%
# 我们观察到邻居数对模型性能有重要影响。如果可以访问（至少部分）地面真实标签，
# 那么根据需要调整 `n_neighbors` 是非常重要的。一种方便的方法是探索 `n_neighbors`
# 值的数量级与预期污染程度的数量级相近。

# %%
# 导入预处理器和 LOF 算法
from sklearn.preprocessing import MinMaxScaler, SplineTransformer, StandardScaler

# 定义预处理器列表
preprocessor_list = [
    None,
    RobustScaler(),
    StandardScaler(),
    MinMaxScaler(),
    SplineTransformer(),
]

# 定义预期异常分数
expected_anomaly_fraction = 0.02

# 设置 LOF 算法的 n_neighbors 参数
lof = LocalOutlierFactor(n_neighbors=int(n_samples * expected_anomaly_fraction))

# 创建图表和坐标轴
fig, ax = plt.subplots()

# 遍历 preprocessor_list 和 linestyles，进行模型拟合和可视化
for model_idx, (linestyle, preprocessor) in enumerate(zip(linestyles, preprocessor_list)):
    # 创建包含预处理器和 LOF 的管道模型
    model = make_pipeline(preprocessor, lof)
    
    # 拟合模型
    model.fit(X)
    
    # 获取 LOF 算法的负异常因子作为预测分数
    y_pred = model[-1].negative_outlier_factor_
    
    # 绘制 ROC 曲线并显示
    display = RocCurveDisplay.from_predictions(
        y,
        y_pred,
        pos_label=pos_label,
        name=str(preprocessor).split("(")[0],
        ax=ax,
        plot_chance_level=(model_idx == len(preprocessor_list) - 1),
        chance_level_kw={"linestyle": (0, (1, 10))},
        linestyle=linestyle,
        linewidth=2,
    )

# 设置图表标题
_ = ax.set_title("Fixed n_neighbors with varying preprocessing\non forestcover dataset")

# %%
# 一方面，:class:`~sklearn.preprocessing.RobustScaler` 默认使用 IQR（四分位距）
# 独立地缩放每个特征。它通过减去中位数并通过 IQR 进行除法来对数据进行中心化和缩放。
# The
# IQR is robust to outliers: the median and interquartile range are less
# affected by extreme values than the range, the mean and the standard
# deviation. Furthermore, :class:`~sklearn.preprocessing.RobustScaler` does not
# squash marginal outlier values, contrary to
# :class:`~sklearn.preprocessing.StandardScaler`.
#
# On the other hand, :class:`~sklearn.preprocessing.MinMaxScaler` scales each
# feature individually such that its range maps into the range between zero and
# one. If there are outliers in the data, they can skew it towards either the
# minimum or maximum values, leading to a completely different distribution of
# data with large marginal outliers: all non-outlier values can be collapsed
# almost together as a result.
#
# We also evaluated no preprocessing at all (by passing `None` to the pipeline),
# :class:`~sklearn.preprocessing.StandardScaler` and
# :class:`~sklearn.preprocessing.SplineTransformer`. Please refer to their
# respective documentation for more details.
#
# Note that the optimal preprocessing depends on the dataset, as shown below:

# %% Start of a new cell in a Jupyter notebook

# Assigning X as the cardiotocography dataset and y as its corresponding true labels
X = X_cardiotocography
y = y_true["cardiotocography"]

# Setting the number of samples and expected anomaly fraction based on X's shape
n_samples, expected_anomaly_fraction = X.shape[0], 0.025

# Initializing the Local Outlier Factor (LOF) detector with a number of neighbors
# proportional to the expected anomaly fraction
lof = LocalOutlierFactor(n_neighbors=int(n_samples * expected_anomaly_fraction))

# Creating a new figure and axis for plotting
fig, ax = plt.subplots()

# Iterating over models specified by linestyles and preprocessor_list
for model_idx, (linestyle, preprocessor) in enumerate(
    zip(linestyles, preprocessor_list)
):
    # Creating a pipeline with the current preprocessor and LOF model
    model = make_pipeline(preprocessor, lof)
    
    # Fitting the pipeline to the data X
    model.fit(X)
    
    # Extracting negative outlier factor predictions from the fitted model
    y_pred = model[-1].negative_outlier_factor_
    
    # Displaying ROC curve with predictions y_pred against true labels y
    display = RocCurveDisplay.from_predictions(
        y,
        y_pred,
        pos_label=pos_label,
        name=str(preprocessor).split("(")[0],
        ax=ax,
        # Setting chance level for plot aesthetics
        plot_chance_level=(model_idx == len(preprocessor_list) - 1),
        chance_level_kw={"linestyle": (0, (1, 10))},
        linestyle=linestyle,
        linewidth=2,
    )

# Setting the title of the plot
ax.set_title(
    "Fixed n_neighbors with varying preprocessing\non cardiotocography dataset"
)

# Displaying the plot
plt.show()
```