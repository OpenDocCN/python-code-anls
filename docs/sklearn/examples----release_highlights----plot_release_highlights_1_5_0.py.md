# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_1_5_0.py`

```
# ruff: noqa
"""
=======================================
Release Highlights for scikit-learn 1.5
=======================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 1.5! Many bug fixes
and improvements were added, as well as some key new features. Below we
detail the highlights of this release. **For an exhaustive list of
all the changes**, please refer to the :ref:`release notes <release_notes_1_5>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

# %%
# FixedThresholdClassifier: Setting the decision threshold of a binary classifier
# -------------------------------------------------------------------------------
# All binary classifiers of scikit-learn use a fixed decision threshold of 0.5
# to convert probability estimates (i.e. output of `predict_proba`) into class
# predictions. However, 0.5 is almost never the desired threshold for a given
# problem. :class:`~model_selection.FixedThresholdClassifier` allows wrapping any
# binary classifier and setting a custom decision threshold.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

# Generate synthetic data for classification
X, y = make_classification(n_samples=10_000, weights=[0.9, 0.1], random_state=0)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train a logistic regression classifier with default threshold of 0.5
classifier_05 = LogisticRegression(C=1e6, random_state=0).fit(X_train, y_train)
# Display confusion matrix for classifier_05 on the test data
_ = ConfusionMatrixDisplay.from_estimator(classifier_05, X_test, y_test)

# %%
# Lowering the threshold, i.e. allowing more samples to be classified as the positive
# class, increases the number of true positives at the cost of more false positives
# (as is well known from the concavity of the ROC curve).
from sklearn.model_selection import FixedThresholdClassifier

# Create a FixedThresholdClassifier instance with a threshold of 0.1
classifier_01 = FixedThresholdClassifier(classifier_05, threshold=0.1)
# Fit the classifier with the new threshold on the training data
classifier_01.fit(X_train, y_train)
# Display confusion matrix for classifier_01 on the test data
_ = ConfusionMatrixDisplay.from_estimator(classifier_01, X_test, y_test)

# %%
# TunedThresholdClassifierCV: Tuning the decision threshold of a binary classifier
# --------------------------------------------------------------------------------
# The decision threshold of a binary classifier can be tuned to optimize a
# given metric, using :class:`~model_selection.TunedThresholdClassifierCV`.
#
# It is particularly useful to find the best decision threshold when the model
# is meant to be deployed in a specific application context where we can assign
# different gains or costs for true positives, true negatives, false positives,
# and false negatives.
#
# Let's illustrate this by considering an arbitrary case where:
#
# - each true positive gains 1 unit of profit, e.g. euro, year of life in good
#   health, etc.;
# - true negatives gain or cost nothing;
# - each false negative costs 2;
# - each false positive costs 0.1.
#
# Our metric quantifies the average profit per sample, which is defined by the
# following Python function:
from sklearn.metrics import confusion_matrix

# 定义一个自定义评分函数，计算每个样本的平均利润
def custom_score(y_observed, y_pred):
    # 计算混淆矩阵并展开为四个元素，返回正常化后的值
    tn, fp, fn, tp = confusion_matrix(y_observed, y_pred, normalize="all").ravel()
    # 返回根据公式计算的自定义评分
    return tp - 2 * fn - 0.1 * fp

print("Untuned decision threshold: 0.5")
# 打印未调整的决策阈值
print(f"Custom score: {custom_score(y_test, classifier_05.predict(X_test)):.2f}")

# %%
# It is interesting to observe that the average gain per prediction is negative
# which means that this decision system is making a loss on average.
#
# Tuning the threshold to optimize this custom metric gives a smaller threshold
# that allows more samples to be classified as the positive class. As a result,
# the average gain per prediction improves.
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer

# 创建一个自定义的评分器，用于优化决策阈值
custom_scorer = make_scorer(
    custom_score, response_method="predict", greater_is_better=True
)
# 使用交叉验证选择最佳决策阈值的分类器
tuned_classifier = TunedThresholdClassifierCV(
    classifier_05, cv=5, scoring=custom_scorer
).fit(X, y)

# 打印调整后的最佳决策阈值
print(f"Tuned decision threshold: {tuned_classifier.best_threshold_:.3f}")
# 打印使用最佳决策阈值时的自定义评分
print(f"Custom score: {custom_score(y_test, tuned_classifier.predict(X_test)):.2f}")

# %%
# We observe that tuning the decision threshold can turn a machine
# learning-based system that makes a loss on average into a beneficial one.
#
# In practice, defining a meaningful application-specific metric might involve
# making those costs for bad predictions and gains for good predictions depend on
# auxiliary metadata specific to each individual data point such as the amount
# of a transaction in a fraud detection system.
#
# To achieve this, :class:`~model_selection.TunedThresholdClassifierCV`
# leverages metadata routing support (:ref:`Metadata Routing User
# Guide<metadata_routing>`) allowing to optimize complex business metrics as
# detailed in :ref:`Post-tuning the decision threshold for cost-sensitive
# learning
# <sphx_glr_auto_examples_model_selection_plot_cost_sensitive_learning.py>`.

# %%
# Performance improvements in PCA
# -------------------------------
# :class:`~decomposition.PCA` has a new solver, `"covariance_eigh"`, which is
# up to an order of magnitude faster and more memory efficient than the other
# solvers for datasets with many data points and few features.
from sklearn.datasets import make_low_rank_matrix
from sklearn.decomposition import PCA

# 生成一个低秩矩阵作为输入数据
X = make_low_rank_matrix(
    n_samples=10_000, n_features=100, tail_strength=0.1, random_state=0
)

# 使用新的求解器 "covariance_eigh" 进行PCA分析
pca = PCA(n_components=10, svd_solver="covariance_eigh").fit(X)
# 打印解释方差比例的总和
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2f}")


# %%
# The new solver also accepts sparse input data:
from scipy.sparse import random

# 生成稀疏的随机输入数据
X = random(10_000, 100, format="csr", random_state=0)

# 使用新的求解器 "covariance_eigh" 进行PCA分析
pca = PCA(n_components=10, svd_solver="covariance_eigh").fit(X)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2f}")
# 打印主成分分析（PCA）解释的方差总和，保留两位小数

# %%
# `"full"` solver 已经得到改进，使用更少的内存并允许更快的转换。默认的 `svd_solver="auto"`
# 选项利用了新的求解器，能够为稀疏数据集选择合适的求解器。
#
# 类似于大多数其他 PCA 求解器，新的 `"covariance_eigh"` 求解器在输入数据作为 PyTorch
# 或 CuPy 数组传入时，可以利用 GPU 进行计算，启用了 :ref:`Array API <array_api>` 的实验支持。

# %%
# ColumnTransformer 可以通过下标访问
# -----------------------------------
# :class:`~compose.ColumnTransformer` 的转换器现在可以直接通过名称进行访问。
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X = np.array([[0, 1, 2], [3, 4, 5]])
column_transformer = ColumnTransformer(
    [("std_scaler", StandardScaler(), [0]), ("one_hot", OneHotEncoder(), [1, 2])]
)

column_transformer.fit(X)

print(column_transformer["std_scaler"])
print(column_transformer["one_hot"])

# %%
# SimpleImputer 支持自定义的填补策略
# -----------------------------------
# :class:`~impute.SimpleImputer` 现在支持使用一个计算列向量非缺失值的标量值的可调用函数进行填补策略。

from sklearn.impute import SimpleImputer

X = np.array(
    [
        [-1.1, 1.1, 1.1],
        [3.9, -1.2, np.nan],
        [np.nan, 1.3, np.nan],
        [-0.1, -1.4, -1.4],
        [-4.9, 1.5, -1.5],
        [np.nan, 1.6, 1.6],
    ]
)


def smallest_abs(arr):
    """返回一维数组中的最小绝对值。"""
    return np.min(np.abs(arr))


imputer = SimpleImputer(strategy=smallest_abs)

imputer.fit_transform(X)

# %%
# pairwise_distances 支持非数值数组的配对距离计算
# ----------------------------------------------
# :func:`~metrics.pairwise_distances` 现在可以使用可调用的度量来计算非数值数组之间的距离。

from sklearn.metrics import pairwise_distances

X = ["cat", "dog"]
Y = ["cat", "fox"]


def levenshtein_distance(x, y):
    """返回两个字符串之间的Levenshtein距离。"""
    if x == "" or y == "":
        return max(len(x), len(y))
    if x[0] == y[0]:
        return levenshtein_distance(x[1:], y[1:])
    return 1 + min(
        levenshtein_distance(x[1:], y),
        levenshtein_distance(x, y[1:]),
        levenshtein_distance(x[1:], y[1:]),
    )


pairwise_distances(X, Y, metric=levenshtein_distance)
```