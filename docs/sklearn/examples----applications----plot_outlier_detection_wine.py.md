# `D:\src\scipysrc\scikit-learn\examples\applications\plot_outlier_detection_wine.py`

```
"""
====================================
Outlier detection on a real data set
====================================

This example illustrates the need for robust covariance estimation
on a real data set. It is useful both for outlier detection and for
a better understanding of the data structure.

We selected two sets of two variables from the Wine data set
as an illustration of what kind of analysis can be done with several
outlier detection tools. For the purpose of visualization, we are working
with two-dimensional examples, but one should be aware that things are
not so trivial in high-dimension, as it will be pointed out.

In both examples below, the main result is that the empirical covariance
estimate, as a non-robust one, is highly influenced by the heterogeneous
structure of the observations. Although the robust covariance estimate is
able to focus on the main mode of the data distribution, it sticks to the
assumption that the data should be Gaussian distributed, yielding some biased
estimation of the data structure, but yet accurate to some extent.
The One-Class SVM does not assume any parametric form of the data distribution
and can therefore model the complex shape of the data much better.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# First example
# -------------
#
# The first example illustrates how the Minimum Covariance Determinant
# robust estimator can help concentrate on a relevant cluster when outlying
# points exist. Here the empirical covariance estimation is skewed by points
# outside of the main cluster. Of course, some screening tools would have pointed
# out the presence of two clusters (Support Vector Machines, Gaussian Mixture
# Models, univariate outlier detection, ...). But had it been a high-dimensional
# example, none of these could be applied that easily.
from sklearn.covariance import EllipticEnvelope
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import OneClassSVM

# Define different outlier detection estimators
estimators = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=0.25),
    "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
        contamination=0.25
    ),
    "OCSVM": OneClassSVM(nu=0.25, gamma=0.35),
}

# %%
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

# Load Wine dataset and select two variables for illustration
X = load_wine()["data"][:, [1, 2]]  # two clusters

# Create a plot for visualization
fig, ax = plt.subplots()
colors = ["tab:blue", "tab:orange", "tab:red"]

# Learn a frontier for outlier detection with several classifiers
legend_lines = []
for color, (name, estimator) in zip(colors, estimators.items()):
    # Fit the outlier detection estimator to the data
    estimator.fit(X)
    
    # Plot the decision boundary of the estimator
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="decision_function",
        plot_method="contour",
        levels=[0],
        colors=color,
        ax=ax,
    )
    
    # Add legend lines for each estimator
    legend_lines.append(mlines.Line2D([], [], color=color, label=name))
ax.scatter(X[:, 0], X[:, 1], color="black")
# 在坐标轴上绘制散点图，X[:, 0] 表示数据集 X 的第一列，X[:, 1] 表示第二列，使用黑色表示点的颜色

bbox_args = dict(boxstyle="round", fc="0.8")
# 定义一个文本框的样式参数，圆角矩形，填充颜色为浅灰色

arrow_args = dict(arrowstyle="->")
# 定义箭头样式参数，箭头朝向右侧

ax.annotate(
    "outlying points",
    xy=(4, 2),
    xycoords="data",
    textcoords="data",
    xytext=(3, 1.25),
    bbox=bbox_args,
    arrowprops=arrow_args,
)
# 在图中添加注释文本 "outlying points"，位置在坐标 (4, 2)，注释和箭头的坐标都是基于数据坐标系
# 注释文本的起始位置在坐标 (3, 1.25)，使用之前定义的文本框和箭头样式参数进行装饰

ax.legend(handles=legend_lines, loc="upper center")
# 在图中添加图例，图例的句柄为 legend_lines，位置在图的上中部

_ = ax.set(
    xlabel="ash",
    ylabel="malic_acid",
    title="Outlier detection on a real data set (wine recognition)",
)
# 设置坐标轴的标签和标题，横坐标标签为 "ash"，纵坐标标签为 "malic_acid"
# 图的标题为 "Outlier detection on a real data set (wine recognition)"

# %%
# Second example
# --------------
#
# The second example shows the ability of the Minimum Covariance Determinant
# robust estimator of covariance to concentrate on the main mode of the data
# distribution: the location seems to be well estimated, although the
# covariance is hard to estimate due to the banana-shaped distribution. Anyway,
# we can get rid of some outlying observations. The One-Class SVM is able to
# capture the real data structure, but the difficulty is to adjust its kernel
# bandwidth parameter so as to obtain a good compromise between the shape of
# the data scatter matrix and the risk of over-fitting the data.
X = load_wine()["data"][:, [6, 9]]  # "banana"-shaped
# 从 wine 数据集中加载数据，选择第 7 列和第 10 列作为数据集 X，这些数据集呈现出香蕉形状分布

fig, ax = plt.subplots()
# 创建一个新的图形和一个包含单个轴的子图

colors = ["tab:blue", "tab:orange", "tab:red"]
# 定义颜色数组，分别为蓝色、橙色和红色

# Learn a frontier for outlier detection with several classifiers
legend_lines = []
for color, (name, estimator) in zip(colors, estimators.items()):
    estimator.fit(X)
    # 使用数据集 X 对每个分类器进行拟合

    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="decision_function",
        plot_method="contour",
        levels=[0],
        colors=color,
        ax=ax,
    )
    # 使用拟合的分类器生成决策边界展示，方法为轮廓图，颜色为当前循环的 color

    legend_lines.append(mlines.Line2D([], [], color=color, label=name))
    # 将当前分类器的图例句柄添加到 legend_lines 数组中

ax.scatter(X[:, 0], X[:, 1], color="black")
# 在轴上绘制散点图，X[:, 0] 表示数据集 X 的第一列，X[:, 1] 表示第二列，使用黑色表示点的颜色

ax.legend(handles=legend_lines, loc="upper center")
# 在轴上添加图例，图例的句柄为 legend_lines，位置在轴的上中部

ax.set(
    xlabel="flavanoids",
    ylabel="color_intensity",
    title="Outlier detection on a real data set (wine recognition)",
)
# 设置轴的标签和标题，横坐标标签为 "flavanoids"，纵坐标标签为 "color_intensity"
# 轴的标题为 "Outlier detection on a real data set (wine recognition)"

plt.show()
# 显示绘制的图形
```