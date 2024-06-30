# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_isolation_forest.py`

```
"""
=======================
IsolationForest example
=======================

An example using :class:`~sklearn.ensemble.IsolationForest` for anomaly
detection.

The :ref:`isolation_forest` is an ensemble of "Isolation Trees" that "isolate"
observations by recursive random partitioning, which can be represented by a
tree structure. The number of splittings required to isolate a sample is lower
for outliers and higher for inliers.

In the present example we demo two ways to visualize the decision boundary of an
Isolation Forest trained on a toy dataset.

"""

# %%
# Data generation
# ---------------
#
# We generate two clusters (each one containing `n_samples`) by randomly
# sampling the standard normal distribution as returned by
# :func:`numpy.random.randn`. One of them is spherical and the other one is
# slightly deformed.
#
# For consistency with the :class:`~sklearn.ensemble.IsolationForest` notation,
# the inliers (i.e. the gaussian clusters) are assigned a ground truth label `1`
# whereas the outliers (created with :func:`numpy.random.uniform`) are assigned
# the label `-1`.

import numpy as np

from sklearn.model_selection import train_test_split

# Define number of samples and outliers
n_samples, n_outliers = 120, 40
# Initialize random number generator
rng = np.random.RandomState(0)
# Define covariance matrix for cluster 1
covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
# Generate cluster 1 data points
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
# Generate cluster 2 data points
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
# Generate outlier data points
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

# Concatenate all data points into X
X = np.concatenate([cluster_1, cluster_2, outliers])
# Create labels: inliers labeled as 1, outliers labeled as -1
y = np.concatenate(
    [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
# We can visualize the resulting clusters:

import matplotlib.pyplot as plt

# Scatter plot to visualize clusters
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
# Create legend elements for scatter plot
handles, labels = scatter.legend_elements()
# Set plot to square shape
plt.axis("square")
# Add legend with labels for inliers and outliers
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
# Set plot title
plt.title("Gaussian inliers with \nuniformly distributed outliers")
plt.show()

# %%
# Training of the model
# ---------------------

from sklearn.ensemble import IsolationForest

# Initialize Isolation Forest model with parameters
clf = IsolationForest(max_samples=100, random_state=0)
# Fit Isolation Forest model to training data
clf.fit(X_train)

# %%
# Plot discrete decision boundary
# -------------------------------
#
# Use :class:`~sklearn.inspection.DecisionBoundaryDisplay` to visualize a discrete
# decision boundary. The background color represents whether a sample in that
# given area is predicted to be an outlier or not. The scatter plot displays
# the true labels.

from sklearn.inspection import DecisionBoundaryDisplay

# Create decision boundary display object
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    alpha=0.5,
)
# Scatter plot of data points with true labels
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
# Set title for the plot
disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
# Set plot to square shape
plt.axis("square")
# 显示图例，包括处理过的句柄和标签
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
# 展示绘制的图形
plt.show()

# %%
# 绘制路径长度决策边界
# --------------------
#
# 通过设置 `response_method="decision_function"`，背景的
# :class:`~sklearn.inspection.DecisionBoundaryDisplay` 表示观测的正常性度量。
# 这种分数由随机森林中平均路径长度表示，路径长度本身由隔离给定样本所需的叶子深度
# （或等效地，分割次数）给出。
#
# 当随机森林集体为隔离某些特定样本产生较短的路径长度时，它们极有可能是异常值，
# 正常性度量接近于 `0`。类似地，较长的路径对应的值接近 `1`，更可能是内点。
#

# 从估算器创建路径长度决策边界显示对象
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    alpha=0.5,
)
# 在显示对象的坐标轴上绘制散点图，用分类标签 `y` 进行着色
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
# 设置坐标轴的标题
disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
# 设置坐标轴的刻度相等
plt.axis("square")
# 显示图例，包括处理过的句柄和标签
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
# 在坐标轴上添加颜色条
plt.colorbar(disp.ax_.collections[1])
# 展示绘制的图形
plt.show()
```