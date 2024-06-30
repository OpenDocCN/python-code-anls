# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_regression.py`

```
"""
============================
Nearest Neighbors regression
============================

Demonstrate the resolution of a regression problem
using a k-Nearest Neighbor and the interpolation of the
target using both barycenter and constant weights.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


# %%
# Generate sample data
# --------------------
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from sklearn import neighbors  # 导入 sklearn 中的 neighbors 模块，包含 k-近邻算法


np.random.seed(0)  # 设置随机种子以确保结果可重现
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # 在 [0, 5] 范围内生成均匀分布的 40 个数据点，并按列排序
T = np.linspace(0, 5, 500)[:, np.newaxis]  # 生成 500 个等间距的点，并将其转换为列向量
y = np.sin(X).ravel()  # 对 X 中每个元素求正弦值，并展平成一维数组作为目标值 y

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))  # 对部分 y 值添加随机噪声

# %%
# Fit regression model
# --------------------
n_neighbors = 5  # 设置近邻数为 5

for i, weights in enumerate(["uniform", "distance"]):  # 遍历权重选项列表
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)  # 创建 K 近邻回归模型对象
    y_ = knn.fit(X, y).predict(T)  # 使用模型拟合数据 X, y，并预测 T 的结果 y_

    plt.subplot(2, 1, i + 1)  # 创建子图，2 行 1 列，当前第 i+1 个子图
    plt.scatter(X, y, color="darkorange", label="data")  # 绘制原始数据散点图
    plt.plot(T, y_, color="navy", label="prediction")  # 绘制预测结果曲线
    plt.axis("tight")  # 设置坐标轴适应数据范围
    plt.legend()  # 显示图例
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))  # 设置子图标题

plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()  # 显示图形
```