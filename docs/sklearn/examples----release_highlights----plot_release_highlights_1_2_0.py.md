# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_1_2_0.py`

```
# ruff: noqa
"""
=======================================
Release Highlights for scikit-learn 1.2
=======================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 1.2! Many bug fixes
and improvements were added, as well as some new key features. We detail
below a few of the major features of this release. **For an exhaustive list of
all the changes**, please refer to the :ref:`release notes <release_notes_1_2>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

# %%
# Pandas output with `set_output` API
# -----------------------------------
# scikit-learn's transformers now support pandas output with the `set_output` API.
# To learn more about the `set_output` API see the example:
# :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py` and
# # this `video, pandas DataFrame output for scikit-learn transformers
# (some examples) <https://youtu.be/5bCg8VfX2x8>`__.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer

# Load the iris dataset into a pandas DataFrame
X, y = load_iris(as_frame=True, return_X_y=True)

# Define column names for sepal and petal features
sepal_cols = ["sepal length (cm)", "sepal width (cm)"]
petal_cols = ["petal length (cm)", "petal width (cm)"]

# Create a ColumnTransformer with two transformers:
# - StandardScaler for sepal columns
# - KBinsDiscretizer for petal columns
preprocessor = ColumnTransformer(
    [
        ("scaler", StandardScaler(), sepal_cols),
        ("kbin", KBinsDiscretizer(encode="ordinal"), petal_cols),
    ],
    verbose_feature_names_out=False,
).set_output(transform="pandas")  # Set output to pandas DataFrame

# Fit and transform the input data X using the preprocessor
X_out = preprocessor.fit_transform(X)
X_out.sample(n=5, random_state=0)  # Sample 5 rows from the transformed data

# %%
# Interaction constraints in Histogram-based Gradient Boosting Trees
# ------------------------------------------------------------------
# :class:`~ensemble.HistGradientBoostingRegressor` and
# :class:`~ensemble.HistGradientBoostingClassifier` now supports interaction constraints
# with the `interaction_cst` parameter. For details, see the
# :ref:`User Guide <interaction_cst_hgbt>`. In the following example, features are not
# allowed to interact.
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor

# Load the diabetes dataset into a pandas DataFrame
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Initialize a HistGradientBoostingRegressor with interaction constraints
hist_no_interact = HistGradientBoostingRegressor(
    interaction_cst=[[i] for i in range(X.shape[1])], random_state=0
)

# Fit the regressor to the data
hist_no_interact.fit(X, y)

# %%
# New and enhanced displays
# -------------------------
# :class:`~metrics.PredictionErrorDisplay` provides a way to analyze regression
# models in a qualitative manner.
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

# Create a figure with two subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Display the actual vs predicted plot using PredictionErrorDisplay
_ = PredictionErrorDisplay.from_estimator(
    hist_no_interact, X, y, kind="actual_vs_predicted", ax=axs[0]
)

# Display the residual vs predicted plot using PredictionErrorDisplay
_ = PredictionErrorDisplay.from_estimator(
    hist_no_interact, X, y, kind="residual_vs_predicted", ax=axs[1]
)

# %%
# :class:`~model_selection.LearningCurveDisplay` is now available to plot
# results from :func:`~model_selection.learning_curve`.
from sklearn.model_selection import LearningCurveDisplay

# 使用 `LearningCurveDisplay` 类来绘制学习曲线，显示模型的性能变化情况
_ = LearningCurveDisplay.from_estimator(
    hist_no_interact, X, y, cv=5, n_jobs=2, train_sizes=np.linspace(0.1, 1, 5)
)

# %%
# :class:`~inspection.PartialDependenceDisplay` exposes a new parameter
# `categorical_features` to display partial dependence for categorical features
# using bar plots and heatmaps.
from sklearn.datasets import fetch_openml

# 从 OpenML 数据库中获取 Titanic 数据集的特征和目标变量，并将其作为 pandas DataFrame 返回
X, y = fetch_openml(
    "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
)
# 从数据集中选择数值和分类类型的特征，并且移除 'body' 列
X = X.select_dtypes(["number", "category"]).drop(columns=["body"])

# %%
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline

# 指定需要进行序数编码的分类特征列表
categorical_features = ["pclass", "sex", "embarked"]
# 创建数据预处理和模型管道，其中对分类特征进行序数编码，其余特征保持不变，使用梯度提升回归模型
model = make_pipeline(
    ColumnTransformer(
        transformers=[("cat", OrdinalEncoder(), categorical_features)],
        remainder="passthrough",
    ),
    HistGradientBoostingRegressor(random_state=0),
).fit(X, y)

# %%
from sklearn.inspection import PartialDependenceDisplay

# 创建绘图画布和坐标系，用于显示模型的部分依赖图
fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
_ = PartialDependenceDisplay.from_estimator(
    model,
    X,
    features=["age", "sex", ("pclass", "sex")],
    categorical_features=categorical_features,
    ax=ax,
)

# %%
# Faster parser in :func:`~datasets.fetch_openml`
# -----------------------------------------------
# :func:`~datasets.fetch_openml` now supports a new `"pandas"` parser that is
# more memory and CPU efficient. In v1.4, the default will change to
# `parser="auto"` which will automatically use the `"pandas"` parser for dense
# data and `"liac-arff"` for sparse data.
X, y = fetch_openml(
    "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
)
# 显示数据集的前几行，以验证数据加载和解析的正确性
X.head()

# %%
# Experimental Array API support in :class:`~discriminant_analysis.LinearDiscriminantAnalysis`
# --------------------------------------------------------------------------------------------
# Experimental support for the `Array API <https://data-apis.org/array-api/latest/>`_
# specification was added to :class:`~discriminant_analysis.LinearDiscriminantAnalysis`.
# The estimator can now run on any Array API compliant libraries such as
# `CuPy <https://docs.cupy.dev/en/stable/overview.html>`__, a GPU-accelerated array
# library. For details, see the :ref:`User Guide <array_api>`.
# 添加对 Array API 的实验性支持，允许 :class:`~discriminant_analysis.LinearDiscriminantAnalysis`
# 在符合 Array API 的库上运行，如 GPU 加速的 `CuPy` 库。

# %%
# Improved efficiency of many estimators
# --------------------------------------
# In version 1.1 the efficiency of many estimators relying on the computation of
# pairwise distances (essentially estimators related to clustering, manifold
# learning and neighbors search algorithms) was greatly improved for float64
# dense input. Efficiency improvement especially were a reduced memory footprint
# and a much better scalability on multi-core machines.
# In version 1.2, the efficiency of these estimators was further improved for all
# 1.1 版本中，对于依赖于计算成对距离的众多估算器（尤其是与聚类、流形学习和邻居搜索算法相关的估算器），
# 在 float64 密集输入上的效率得到了显著提高。特别是内存占用减少，多核机器的可扩展性大大提高。
# 在 1.2 版本中，这些估算器的效率进一步得到了提高。
# 在 float32 和 float64 数据集上测试稠密和稀疏输入的组合，但不包括稀疏-稠密和稠密-稀疏的组合，这些组合针对欧氏距离和平方欧氏距离度量方式不适用。
# 受影响的估算器的详细列表可以在版本变更日志（changelog）的 :ref:`changelog <release_notes_1_2>` 中找到。
```