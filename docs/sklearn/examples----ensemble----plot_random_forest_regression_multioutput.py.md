# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_random_forest_regression_multioutput.py`

```
"""
============================================================
Comparing random forests and the multi-output meta estimator
============================================================

An example to compare multi-output regression with random forest and
the :ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator.

This example illustrates the use of the
:ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator
to perform multi-output regression. A random forest regressor is used,
which supports multi-output regression natively, so the results can be
compared.

The random forest regressor will only ever predict values within the
range of observations or closer to zero for each of the targets. As a
result the predictions are biased towards the centre of the circle.

Using a single underlying feature the model learns both the
x and y coordinate as output.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算

from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.multioutput import MultiOutputRegressor  # 导入多输出回归器

# Create a random dataset
rng = np.random.RandomState(1)  # 创建随机数生成器 rng，并设置种子为 1
X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)  # 生成排序的随机数据 X
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T  # 生成 y 数据，分别是 sin 和 cos 函数的结果
y += 0.5 - rng.rand(*y.shape)  # 对 y 数据添加噪声

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=400, test_size=200, random_state=4
)  # 使用 train_test_split 函数划分数据集为训练集和测试集，设置随机种子为 4

max_depth = 30  # 定义随机森林最大深度为 30
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=0)
)  # 创建多输出回归器，使用随机森林回归器作为基础模型
regr_multirf.fit(X_train, y_train)  # 在训练集上拟合多输出回归器

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)  # 创建普通随机森林回归器
regr_rf.fit(X_train, y_train)  # 在训练集上拟合普通随机森林回归器

# Predict on new data
y_multirf = regr_multirf.predict(X_test)  # 在测试集上进行多输出回归器的预测
y_rf = regr_rf.predict(X_test)  # 在测试集上进行普通随机森林回归器的预测

# Plot the results
plt.figure()  # 创建图形
s = 50  # 点的大小
a = 0.4  # 透明度
plt.scatter(
    y_test[:, 0],
    y_test[:, 1],
    edgecolor="k",
    c="navy",
    s=s,
    marker="s",
    alpha=a,
    label="Data",
)  # 绘制测试集真实数据散点图
plt.scatter(
    y_multirf[:, 0],
    y_multirf[:, 1],
    edgecolor="k",
    c="cornflowerblue",
    s=s,
    alpha=a,
    label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test),
)  # 绘制多输出回归器的预测结果散点图
plt.scatter(
    y_rf[:, 0],
    y_rf[:, 1],
    edgecolor="k",
    c="c",
    s=s,
    marker="^",
    alpha=a,
    label="RF score=%.2f" % regr_rf.score(X_test, y_test),
)  # 绘制普通随机森林回归器的预测结果散点图
plt.xlim([-6, 6])  # 设置 x 轴范围
plt.ylim([-6, 6])  # 设置 y 轴范围
plt.xlabel("target 1")  # 设置 x 轴标签
plt.ylabel("target 2")  # 设置 y 轴标签
plt.title("Comparing random forests and the multi-output meta estimator")  # 设置图标题
plt.legend()  # 显示图例
plt.show()  # 展示图形
```