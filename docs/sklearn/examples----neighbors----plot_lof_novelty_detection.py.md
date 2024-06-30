# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_lof_novelty_detection.py`

```
"""
=================================================
Novelty detection with Local Outlier Factor (LOF)
=================================================

The Local Outlier Factor (LOF) algorithm is an unsupervised anomaly detection
method which computes the local density deviation of a given data point with
respect to its neighbors. It considers as outliers the samples that have a
substantially lower density than their neighbors. This example shows how to
use LOF for novelty detection. Note that when LOF is used for novelty
detection you MUST not use predict, decision_function and score_samples on the
training set as this would lead to wrong results. You must only use these
methods on new unseen data (which are not in the training set). See
:ref:`User Guide <outlier_detection>`: for details on the difference between
outlier detection and novelty detection and how to use LOF for outlier
detection.

The number of neighbors considered, (parameter n_neighbors) is typically
set 1) greater than the minimum number of samples a cluster has to contain,
so that other samples can be local outliers relative to this cluster, and 2)
smaller than the maximum number of close by samples that can potentially be
local outliers.
In practice, such information is generally not available, and taking
n_neighbors=20 appears to work well in general.

"""

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate normal (not abnormal) training observations
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate new normal (not abnormal) observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model for novelty detection (novelty=True)
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
# DO NOT use predict, decision_function and score_samples on X_train as this
# would give wrong results but only on new unseen data (not used in X_train),
# e.g. X_test, X_outliers or the meshgrid
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the learned frontier, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Title of the plot
plt.title("Novelty Detection with LOF")
# Fill contours of the decision function with specified levels and colormap
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# Draw contours of the decision function at level 0
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
# Fill contours of the decision function at levels 0 and maximum value with colors
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

# Size of points in scatter plot
s = 40
# Scatter plot of normal training observations
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
# 绘制散点图，显示测试集中的数据点，使用蓝紫色标记，设置点的大小为 s，边界颜色为黑色
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")

# 绘制散点图，显示异常数据集中的数据点，使用金色标记，设置点的大小为 s，边界颜色为黑色
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")

# 设置图的坐标轴范围紧贴数据范围
plt.axis("tight")

# 设置 x 轴的显示范围为 (-5, 5)
plt.xlim((-5, 5))

# 设置 y 轴的显示范围为 (-5, 5)
plt.ylim((-5, 5))

# 添加图例，分别标注学习到的分类边界、训练集数据点、新的正常数据点和新的异常数据点
plt.legend(
    [mlines.Line2D([], [], color="darkred"), b1, b2, c],
    [
        "learned frontier",  # 学习到的分类边界
        "training observations",  # 训练集数据点
        "new regular observations",  # 新的正常数据点
        "new abnormal observations",  # 新的异常数据点
    ],
    loc="upper left",  # 图例位置为左上角
    prop=matplotlib.font_manager.FontProperties(size=11),  # 图例文字大小设置为 11
)

# 设置 x 轴标签，显示格式为 "errors novel regular: %d/40 ; errors novel abnormal: %d/40"，其中填入具体的错误计数
plt.xlabel(
    "errors novel regular: %d/40 ; errors novel abnormal: %d/40"
    % (n_error_test, n_error_outliers)
)

# 显示图形
plt.show()
```