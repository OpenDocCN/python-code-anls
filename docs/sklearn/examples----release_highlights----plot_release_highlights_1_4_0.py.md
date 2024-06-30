# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_1_4_0.py`

```
# ruff: noqa
"""
=======================================
Release Highlights for scikit-learn 1.4
=======================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 1.4! Many bug fixes
and improvements were added, as well as some new key features. We detail
below a few of the major features of this release. **For an exhaustive list of
all the changes**, please refer to the :ref:`release notes <release_notes_1_4>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

# %%
# HistGradientBoosting Natively Supports Categorical DTypes in DataFrames
# -----------------------------------------------------------------------
# :class:`ensemble.HistGradientBoostingClassifier` and
# :class:`ensemble.HistGradientBoostingRegressor` now directly supports dataframes with
# categorical features.  Here we have a dataset with a mixture of
# categorical and numerical features:
from sklearn.datasets import fetch_openml

# Fetch the 'adult' dataset from openml, version 2, and return X (features) and y (target)
X_adult, y_adult = fetch_openml("adult", version=2, return_X_y=True)

# Remove columns "education-num" and "fnlwgt" from the features
X_adult = X_adult.drop(["education-num", "fnlwgt"], axis="columns")

# Print the data types of the remaining columns in X_adult
X_adult.dtypes

# %%
# By setting `categorical_features="from_dtype"`, the gradient boosting classifier
# treats the columns with categorical dtypes as categorical features in the
# algorithm:
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_adult, y_adult, random_state=0)

# Create a HistGradientBoostingClassifier instance, setting categorical features from dtype
hist = HistGradientBoostingClassifier(categorical_features="from_dtype")

# Fit the classifier on the training data
hist.fit(X_train, y_train)

# Obtain decision function scores on the test data
y_decision = hist.decision_function(X_test)

# Print ROC AUC score
print(f"ROC AUC score is {roc_auc_score(y_test, y_decision)}")

# %%
# Polars output in `set_output`
# -----------------------------
# scikit-learn's transformers now support polars output with the `set_output` API.
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create a Polars DataFrame with height and pet columns
df = pl.DataFrame({"height": [120, 140, 150, 110, 100], "pet": ["dog", "cat", "dog", "cat", "cat"]})

# Define a ColumnTransformer with StandardScaler for numerical column and OneHotEncoder for categorical column
preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), ["height"]),
        ("categorical", OneHotEncoder(sparse_output=False), ["pet"]),
    ],
    verbose_feature_names_out=False,
)

# Set the output to transform as 'polars'
preprocessor.set_output(transform="polars")

# Transform the DataFrame df using the preprocessor
df_out = preprocessor.fit_transform(df)
df_out

# %%
# Print the type of the output DataFrame df_out
print(f"Output type: {type(df_out)}")

# %%
# Missing value support for Random Forest
# ---------------------------------------
# The classes :class:`ensemble.RandomForestClassifier` and
# :class:`ensemble.RandomForestRegressor` now support missing values. When training
# every individual tree, the splitter evaluates each potential threshold with the
# 导入 NumPy 库，用于处理数据
import numpy as np
# 导入随机森林分类器模型
from sklearn.ensemble import RandomForestClassifier

# 创建包含缺失值的数组 X，reshape(-1, 1) 将其转换为列向量
X = np.array([0, 1, 6, np.nan]).reshape(-1, 1)
# 目标标签 y
y = [0, 0, 1, 1]

# 使用随机森林分类器拟合数据
forest = RandomForestClassifier(random_state=0).fit(X, y)
# 对 X 进行预测
forest.predict(X)

# %%
# 添加树模型中单调约束的支持
# ----------------------------------------------------------
# 自 scikit-learn 0.23 起，我们增加了直方图梯度提升树中的单调约束支持，
# 现在我们在所有其他树模型中也支持此特性，包括树、随机森林、极端随机树和精确梯度提升。
# 在这里，我们展示了在回归问题上，随机森林模型如何支持此特性。
import matplotlib.pyplot as plt
# 导入偏依赖展示工具
from sklearn.inspection import PartialDependenceDisplay
# 导入随机森林回归器模型
from sklearn.ensemble import RandomForestRegressor

# 创建样本数量为 500 的随机数据集
n_samples = 500
rng = np.random.RandomState(0)
X = rng.randn(n_samples, 2)
noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
y = 5 * X[:, 0] + np.sin(10 * np.pi * X[:, 0]) - noise

# 使用随机森林回归器拟合数据，分别创建有约束和无约束的模型
rf_no_cst = RandomForestRegressor().fit(X, y)
rf_cst = RandomForestRegressor(monotonic_cst=[1, 0]).fit(X, y)

# 创建偏依赖展示对象，显示有约束和无约束模型在特征 0 上的偏依赖图
disp = PartialDependenceDisplay.from_estimator(
    rf_no_cst,
    X,
    features=[0],
    feature_names=["feature 0"],
    line_kw={"linewidth": 4, "label": "unconstrained", "color": "tab:blue"},
)
PartialDependenceDisplay.from_estimator(
    rf_cst,
    X,
    features=[0],
    line_kw={"linewidth": 4, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
disp.axes_[0, 0].plot(
    X[:, 0], y, "o", alpha=0.5, zorder=-1, label="samples", color="tab:green"
)
disp.axes_[0, 0].set_ylim(-3, 3)
disp.axes_[0, 0].set_xlim(-1, 1)
disp.axes_[0, 0].legend()
plt.show()

# %%
# 丰富的估计器显示
# ---------------------------
# 估计器显示已经丰富化：例如，如果我们查看上面定义的 `forest`：
forest

# %%
# 点击图表右上角的 "?" 图标可以访问估计器的文档。
#
# 此外，当估计器被拟合时，显示会从橙色变为蓝色。当你悬停在 "i" 图标上时，
# 也可以获取这些信息。
from sklearn.base import clone

clone(forest)  # 克隆的对象尚未拟合

# %%
# 元数据路由支持
# ------------------------
# 许多元估计器和交叉验证程序现在支持元数据路由，详细列在 :ref:`用户指南 <metadata_routing_models>`。
# 例如，这是如何使用样本权重和 :class:`~model_selection.GroupKFold` 进行嵌套交叉验证的示例：
import sklearn
# 导入获取评分器函数
from sklearn.metrics import get_scorer
# 导入生成回归数据集的函数
from sklearn.datasets import make_regression
# 导入 Lasso 回归模型
from sklearn.linear_model import Lasso
# 导入网格搜索交叉验证模型
from sklearn.model_selection import GridSearchCV, cross_validate, GroupKFold

# 目前默认情况下元数据路由是禁用的，需要显式启用。
# 设置 `enable_metadata_routing` 标志以启用元数据路由，用于某些 sklearn 模型的元数据路由支持
sklearn.set_config(enable_metadata_routing=True)

# 设置样本数量
n_samples = 100
# 使用 `make_regression` 生成具有 5 个特征的回归数据集 `X` 和目标值 `y`，噪声为 0.5
X, y = make_regression(n_samples=n_samples, n_features=5, noise=0.5)
# 创建随机数生成器对象 `rng`，并生成包含 `n_samples` 个元素的随机分组数据 `groups`
rng = np.random.RandomState(7)
groups = rng.randint(0, 10, size=n_samples)
# 生成 `n_samples` 个样本权重 `sample_weights`
sample_weights = rng.rand(n_samples)
# 创建 `Lasso` 回归估计器对象，并设置使用样本权重
estimator = Lasso().set_fit_request(sample_weight=True)
# 设置超参数网格 `hyperparameter_grid`，包含不同的 `alpha` 值
hyperparameter_grid = {"alpha": [0.1, 0.5, 1.0, 2.0]}
# 设置用于内部交叉验证的评分器 `scoring_inner_cv`，使用样本权重
scoring_inner_cv = get_scorer("neg_mean_squared_error").set_score_request(
    sample_weight=True
)
# 创建 `GroupKFold` 对象 `inner_cv`，用于内部交叉验证的分组交叉验证
inner_cv = GroupKFold(n_splits=5)

# 创建 `GridSearchCV` 对象 `grid_search`，用于在超参数网格上执行估计器的交叉验证搜索
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=hyperparameter_grid,
    cv=inner_cv,
    scoring=scoring_inner_cv,
)

# 创建 `GroupKFold` 对象 `outer_cv`，用于外部交叉验证的分组交叉验证
outer_cv = GroupKFold(n_splits=5)
# 创建评分器字典 `scorers`，包含名称为 "mse" 的评分器，使用样本权重
scorers = {
    "mse": get_scorer("neg_mean_squared_error").set_score_request(sample_weight=True)
}
# 执行外部交叉验证 `cross_validate`，返回交叉验证结果和估计器对象
results = cross_validate(
    grid_search,
    X,
    y,
    cv=outer_cv,
    scoring=scorers,
    return_estimator=True,
    params={"sample_weight": sample_weights, "groups": groups},
)
# 输出测试集上的交叉验证误差
print("cv error on test sets:", results["test_mse"])

# 设置 `enable_metadata_routing` 标志为 `False`，以避免影响其他脚本
# 设置为默认的 `False`
sklearn.set_config(enable_metadata_routing=False)

# %%
# 改进稀疏数据上 PCA 的内存和运行时效率
# -------------------------------------------------------------
# 现在 PCA 能够原生地处理稀疏矩阵，使用 `arpack` 求解器，
# 通过 `scipy.sparse.linalg.LinearOperator` 来避免在执行数据集协方差矩阵的特征值分解时
# 实例化大型稀疏矩阵。
#
from sklearn.decomposition import PCA
import scipy.sparse as sp
from time import time

# 创建稀疏矩阵 `X_sparse`，形状为 (1000, 1000)，随机初始化
X_sparse = sp.random(m=1000, n=1000, random_state=0)
# 将稀疏矩阵转换为密集矩阵 `X_dense`
X_dense = X_sparse.toarray()

# 计时开始
t0 = time()
# 在稀疏数据 `X_sparse` 上使用 `arpack` 求解器执行 PCA，选择前 10 个主成分
PCA(n_components=10, svd_solver="arpack").fit(X_sparse)
# 计算稀疏数据运行时间
time_sparse = time() - t0

# 计时重新开始
t0 = time()
# 在密集数据 `X_dense` 上使用 `arpack` 求解器执行 PCA，选择前 10 个主成分
PCA(n_components=10, svd_solver="arpack").fit(X_dense)
# 计算密集数据运行时间
time_dense = time() - t0

# 输出稀疏数据和密集数据运行时间比例
print(f"Speedup: {time_dense / time_sparse:.1f}x")
```