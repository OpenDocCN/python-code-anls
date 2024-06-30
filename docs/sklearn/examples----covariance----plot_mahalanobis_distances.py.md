# `D:\src\scipysrc\scikit-learn\examples\covariance\plot_mahalanobis_distances.py`

```
# %%
# Generate data
# --------------
#
# 首先，生成一个包含125个样本和2个特征的数据集。
# 这两个特征都服从均值为0的高斯分布，但是特征1具有标准差
# deviation equal to 2 and feature 2 has a standard deviation equal to 1. Next,
# 25 samples are replaced with Gaussian outlier samples where feature 1 has
# a standard deviation equal to 1 and feature 2 has a standard deviation equal
# to 7.

import numpy as np

# for consistent results
np.random.seed(7)

n_samples = 125
n_outliers = 25
n_features = 2

# generate Gaussian data of shape (125, 2)
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2.0
# Generate data points with specified covariance for inliers
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
# add some outliers
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.0
# Replace the last n_outliers samples with outlier data points
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

# %%
# Comparison of results
# ---------------------
#
# Below, we fit MCD and MLE based covariance estimators to our data and print
# the estimated covariance matrices. Note that the estimated variance of
# feature 2 is much higher with the MLE based estimator (7.5) than
# that of the MCD robust estimator (1.2). This shows that the MCD based
# robust estimator is much more resistant to the outlier samples, which were
# designed to have a much larger variance in feature 2.

import matplotlib.pyplot as plt

from sklearn.covariance import EmpiricalCovariance, MinCovDet

# fit a MCD robust estimator to data
robust_cov = MinCovDet().fit(X)
# fit a MLE estimator to data
emp_cov = EmpiricalCovariance().fit(X)
print(
    "Estimated covariance matrix:\nMCD (Robust):\n{}\nMLE:\n{}".format(
        robust_cov.covariance_, emp_cov.covariance_
    )
)

# %%
# To better visualize the difference, we plot contours of the
# Mahalanobis distances calculated by both methods. Notice that the robust
# MCD based Mahalanobis distances fit the inlier black points much better,
# whereas the MLE based distances are more influenced by the outlier
# red points.
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(10, 5))
# Plot data set
inlier_plot = ax.scatter(X[:, 0], X[:, 1], color="black", label="inliers")
outlier_plot = ax.scatter(
    X[:, 0][-n_outliers:], X[:, 1][-n_outliers:], color="red", label="outliers"
)
ax.set_xlim(ax.get_xlim()[0], 10.0)
ax.set_title("Mahalanobis distances of a contaminated data set")

# Create meshgrid of feature 1 and feature 2 values
xx, yy = np.meshgrid(
    np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
    np.linspace(plt.ylim()[0], plt.ylim()[1], 100),
)
zz = np.c_[xx.ravel(), yy.ravel()]
# Calculate the MLE based Mahalanobis distances of the meshgrid
mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = plt.contour(
    xx, yy, np.sqrt(mahal_emp_cov), cmap=plt.cm.PuBu_r, linestyles="dashed"
)
# Calculate the MCD based Mahalanobis distances
mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = ax.contour(
    xx, yy, np.sqrt(mahal_robust_cov), cmap=plt.cm.YlOrBr_r, linestyles="dotted"
)

# Add legend
inlier_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='inliers')
outlier_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                               markersize=10, label='outliers')
ax.legend(handles=[inlier_legend, outlier_legend])
# 创建图例，包含四个元素：蓝色虚线，橙色点线，内点图(inlier_plot)，外点图(outlier_plot)
ax.legend(
    [
        mlines.Line2D([], [], color="tab:blue", linestyle="dashed"),
        mlines.Line2D([], [], color="tab:orange", linestyle="dotted"),
        inlier_plot,
        outlier_plot,
    ],
    ["MLE dist", "MCD dist", "inliers", "outliers"],  # 设置图例标签
    loc="upper right",  # 设置图例位置为右上角
    borderaxespad=0,  # 边界填充设置为0
)

plt.show()  # 显示图形

# %%
# 最后，突显基于MCD的马氏距离识别异常值的能力。我们对马氏距离取立方根，
# 得到近似正态分布（正如Wilson和Hilferty [2]_建议的那样），然后用箱线图绘制
# 内点和异常点的值。对于基于MCD的马氏距离，异常样本的分布与内点样本的分布更为分离。
fig, (ax1, ax2) = plt.subplots(1, 2)  # 创建包含两个子图的图形
plt.subplots_adjust(wspace=0.6)  # 调整子图之间的间距

# 计算样本的MLE马氏距离的立方根
emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
# 绘制箱线图
ax1.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=0.25)
# 绘制个别样本
ax1.plot(
    np.full(n_samples - n_outliers, 1.26),
    emp_mahal[:-n_outliers],
    "+k",
    markeredgewidth=1,
)
ax1.plot(np.full(n_outliers, 2.26), emp_mahal[-n_outliers:], "+k", markeredgewidth=1)
ax1.axes.set_xticklabels(("inliers", "outliers"), size=15)  # 设置X轴标签
ax1.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)  # 设置Y轴标签
ax1.set_title("Using non-robust estimates\n(Maximum Likelihood)")  # 设置子图标题

# 计算样本的MCD马氏距离的立方根
robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
# 绘制箱线图
ax2.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]], widths=0.25)
# 绘制个别样本
ax2.plot(
    np.full(n_samples - n_outliers, 1.26),
    robust_mahal[:-n_outliers],
    "+k",
    markeredgewidth=1,
)
ax2.plot(np.full(n_outliers, 2.26), robust_mahal[-n_outliers:], "+k", markeredgewidth=1)
ax2.axes.set_xticklabels(("inliers", "outliers"), size=15)  # 设置X轴标签
ax2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)  # 设置Y轴标签
ax2.set_title("Using robust estimates\n(Minimum Covariance Determinant)")  # 设置子图标题

plt.show()  # 显示图形
```