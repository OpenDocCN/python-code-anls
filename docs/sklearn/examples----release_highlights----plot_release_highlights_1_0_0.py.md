# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_1_0_0.py`

```
# ruff: noqa
"""
=======================================
Release Highlights for scikit-learn 1.0
=======================================

.. currentmodule:: sklearn

We are very pleased to announce the release of scikit-learn 1.0! The library
has been stable for quite some time, releasing version 1.0 is recognizing that
and signalling it to our users. This release does not include any breaking
changes apart from the usual two-release deprecation cycle. For the future, we
do our best to keep this pattern.

This release includes some new key features as well as many improvements and
bug fixes. We detail below a few of the major features of this release. **For
an exhaustive list of all the changes**, please refer to the :ref:`release
notes <release_notes_1_0>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

##############################################################################
# Keyword and positional arguments
# ---------------------------------------------------------
# The scikit-learn API exposes many functions and methods which have many input
# parameters. For example, before this release, one could instantiate a
# :class:`~ensemble.HistGradientBoostingRegressor` as::
#
#         HistGradientBoostingRegressor("squared_error", 0.1, 100, 31, None,
#             20, 0.0, 255, None, None, False, "auto", "loss", 0.1, 10, 1e-7,
#             0, None)
#
# Understanding the above code requires the reader to go to the API
# documentation and to check each and every parameter for its position and
# its meaning. To improve the readability of code written based on scikit-learn,
# now users have to provide most parameters with their names, as keyword
# arguments, instead of positional arguments. For example, the above code would
# be::
#
#     HistGradientBoostingRegressor(
#         loss="squared_error",
#         learning_rate=0.1,
#         max_iter=100,
#         max_leaf_nodes=31,
#         max_depth=None,
#         min_samples_leaf=20,
#         l2_regularization=0.0,
#         max_bins=255,
#         categorical_features=None,
#         monotonic_cst=None,
#         warm_start=False,
#         early_stopping="auto",
#         scoring="loss",
#         validation_fraction=0.1,
#         n_iter_no_change=10,
#         tol=1e-7,
#         verbose=0,
#         random_state=None,
#     )
#
# which is much more readable. Positional arguments have been deprecated since
# version 0.23 and will now raise a ``TypeError``. A limited number of
# positional arguments are still allowed in some cases, for example in
# :class:`~decomposition.PCA`, where ``PCA(10)`` is still allowed, but ``PCA(10,
# False)`` is not allowed.

##############################################################################
# Spline Transformers
# ---------------------------------------------------------
# No additional code or comments are present beyond this section.
# 导入必要的库
import numpy as np
from sklearn.preprocessing import SplineTransformer

# 创建一个简单的一维数组作为输入特征
X = np.arange(5).reshape(5, 1)

# 创建 SplineTransformer 对象，设置 B-样条的阶数为2，结点数为3
spline = SplineTransformer(degree=2, n_knots=3)

# 对输入数据进行拟合和转换
spline.fit_transform(X)


##############################################################################
# Quantile Regressor
# --------------------------------------------------------------------------
# 分位数回归估计因变量 y 在给定自变量 X 条件下的中位数或其他分位数，而普通最小二乘法
# 则估计条件均值。
#
# 作为一个线性模型，新的 QuantileRegressor 类提供了对第 q 个分位数的线性预测，
# 其中权重或系数 w 通过以下最小化问题求解：
#
# .. math::
#     \min_{w} {\frac{1}{n_{\text{samples}}}
#     \sum_i PB_q(y_i - X_i w) + \alpha ||w||_1}.
#
# 这里 PB_q(t) 表示分位数损失（也称为线性损失），详见 sklearn.metrics.mean_pinball_loss，
#
# .. math::
#     PB_q(t) = q \max(t, 0) + (1 - q) \max(-t, 0) =
#     \begin{cases}
#         q t, & t > 0, \\
#         0,    & t = 0, \\
#         (1-q) t, & t < 0
#     \end{cases}
#
# 同时还有由参数 alpha 控制的 L1 惩罚，类似于 linear_model.Lasso。
#
# 请查看以下示例以了解其工作原理，并参阅 User Guide 中的 quantile_regression 以获取更多细节。

##############################################################################
# Feature Names Support
# --------------------------------------------------------------------------
# 当在 fit 过程中传递一个 pandas dataframe 给一个 estimator 时，
# estimator 将设置一个 feature_names_in_ 属性，其中包含特征名称。
# 注意，只有当 dataframe 的列名全为字符串时，特征名称支持才会启用。
# feature_names_in_ 用于检查传递给非 fit 过程（如 predict）的 dataframe 的列名
# 是否与特征匹配。
# 导入 StandardScaler 类，用于数据标准化
from sklearn.preprocessing import StandardScaler
# 导入 pandas 库，用于数据处理和分析
import pandas as pd

# 创建一个包含两行三列数据的 DataFrame
X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
# 使用 StandardScaler 对象拟合数据 X，计算均值和标准差
scalar = StandardScaler().fit(X)
# 获取标准化后的特征名称
scalar.feature_names_in_

# %%
# 支持 :term:`get_feature_names_out` 的转换器，例如已有 `get_feature_names`
# 方法和输入输出一对一对应的转换器，例如 :class:`~preprocessing.StandardScaler`。
# :term:`get_feature_names_out` 的支持将在未来的版本中添加到所有其他转换器中。
# 此外，可以使用 :meth:`compose.ColumnTransformer.get_feature_names_out`
# 方法来组合转换器的特征名称：
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 创建一个包含宠物类型和年龄的 DataFrame
X = pd.DataFrame({"pet": ["dog", "cat", "fish"], "age": [3, 7, 1]})
# 创建一个 ColumnTransformer 对象，包含数值型和分类型特征的转换
preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), ["age"]),
        ("categorical", OneHotEncoder(), ["pet"]),
    ],
    verbose_feature_names_out=False,
).fit(X)

# 获取转换后的特征名称
preprocessor.get_feature_names_out()

# %%
# 当此 `preprocessor` 与 pipeline 结合使用时，分类器使用的特征名称通过切片和调用
# :term:`get_feature_names_out` 方法获得：
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 创建目标变量 y
y = [1, 0, 1]
# 创建一个 pipeline 包含 preprocessor 和 LogisticRegression 分类器
pipe = make_pipeline(preprocessor, LogisticRegression())
# 对 pipeline 进行拟合
pipe.fit(X, y)
# 获取 pipeline 中除最后一步外的所有步骤的特征名称
pipe[:-1].get_feature_names_out()
# complexity of a kernelized One-Class SVM is at best quadratic in the number
# of samples. :class:`~linear_model.SGDOneClassSVM` is thus well suited for
# datasets with a large number of training samples (> 10,000) for which the SGD
# variant can be several orders of magnitude faster. Please check this
# :ref:`example
# <sphx_glr_auto_examples_miscellaneous_plot_anomaly_comparison.py>` to see how
# it's used, and the :ref:`User Guide <sgd_online_one_class_svm>` for more
# details.
#
# .. figure:: ../miscellaneous/images/sphx_glr_plot_anomaly_comparison_001.png
#    :target: ../miscellaneous/plot_anomaly_comparison.html
#    :align: center

##############################################################################
# Histogram-based Gradient Boosting Models are now stable
# --------------------------------------------------------------------------
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` and
# :class:`~ensemble.HistGradientBoostingClassifier` are no longer experimental
# and can simply be imported and used as::
#
#     from sklearn.ensemble import HistGradientBoostingClassifier

##############################################################################
# New documentation improvements
# ------------------------------
# This release includes many documentation improvements. Out of over 2100
# merged pull requests, about 800 of them are improvements to our
# documentation.
```