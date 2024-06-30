# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_lof_outlier_detection.py`

```
# %%
# Generate data with outliers
# ---------------------------

# %%
import numpy as np

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成正常数据点
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# 生成异常数据点
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 合并正常数据点和异常数据点
X = np.r_[X_inliers, X_outliers]

# 计算异常数据点的数量
n_outliers = len(X_outliers)

# 创建一个标签数组，1 表示正常数据点，-1 表示异常数据点
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# %%
# Fit the model for outlier detection (default)
# ---------------------------------------------

# 使用 `fit_predict` 计算训练样本的预测标签
# （当 LOF 用于异常检测时，估计器没有 `predict`、`decision_function` 和 `score_samples` 方法）

from sklearn.neighbors import LocalOutlierFactor

# 创建 LOF 模型，设置参数 `n_neighbors=20` 和 `contamination=0.1`
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# 拟合模型并预测样本标签
y_pred = clf.fit_predict(X)

# 计算预测标签与真实标签不同的数量
n_errors = (y_pred != ground_truth).sum()

# 获取异常值因子（negative_outlier_factor_）
X_scores = clf.negative_outlier_factor_

# %%
# Plot results
# ------------

# %%
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

# 自定义函数：更新图例标记的大小
def update_legend_marker_size(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([20])

# 绘制数据点的散点图
plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")

# 根据异常值因子绘制圆圈，圆圈的半径与异常值因子成反比
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))

# 设置图例并调用自定义的图例处理程序
plt.legend(
    handler_map={plt.scatter: HandlerPathCollection(update_func=update_legend_marker_size)},
)
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}


# 创建一个字典 handler_map，其中包含一个键为 scatter 的项和对应的值为 HandlerPathCollection 对象
# HandlerPathCollection 对象使用 update_legend_marker_size 函数作为更新函数
)
# 关闭 Matplotlib 的图形绘制
plt.title("Local Outlier Factor (LOF)")
# 显示绘制的图形
plt.show()
```