# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_huber_vs_ridge.py`

```
"""
=======================================================
HuberRegressor vs Ridge on dataset with strong outliers
=======================================================

Fit Ridge and HuberRegressor on a dataset with outliers.

The example shows that the predictions in ridge are strongly influenced
by the outliers present in the dataset. The Huber regressor is less
influenced by the outliers since the model uses the linear loss for these.
As the parameter epsilon is increased for the Huber regressor, the decision
function approaches that of the ridge.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入 matplotlib.pyplot 作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并重命名为 np
import numpy as np

# 从 sklearn.datasets 导入 make_regression 生成模拟回归数据的工具函数
from sklearn.datasets import make_regression
# 从 sklearn.linear_model 导入 HuberRegressor 和 Ridge 回归模型
from sklearn.linear_model import HuberRegressor, Ridge

# 使用随机种子生成一个确定性的随机数生成器
rng = np.random.RandomState(0)
# 使用 make_regression 生成一个包含异常值的模拟数据集
X, y = make_regression(
    n_samples=20, n_features=1, random_state=0, noise=4.0, bias=100.0
)

# 添加四个强烈的异常值到数据集中
X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.0
X_outliers[2:, :] += X.min() - X.mean() / 4.0
y_outliers[:2] += y.min() - y.mean() / 4.0
y_outliers[2:] += y.max() + y.mean() / 4.0
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))

# 绘制数据点，蓝色圆点表示数据点
plt.plot(X, y, "b.")

# 使用一系列的 epsilon 值拟合 HuberRegressor 回归器
colors = ["r-", "b-", "y-", "m-"]
# 在数据范围内生成均匀分布的点作为 x 值
x = np.linspace(X.min(), X.max(), 7)
epsilon_values = [1, 1.5, 1.75, 1.9]
for k, epsilon in enumerate(epsilon_values):
    # 创建 HuberRegressor 对象，设置 alpha 为 0.0，epsilon 为当前值
    huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
    # 使用 HuberRegressor 对象拟合数据集 X, y
    huber.fit(X, y)
    # 计算拟合直线的系数
    coef_ = huber.coef_ * x + huber.intercept_
    # 绘制拟合直线，使用不同颜色的线条代表不同 epsilon 值的 Huber loss
    plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

# 拟合一个 Ridge 回归器，用于与 Huber 回归器比较
ridge = Ridge(alpha=0.0, random_state=0)
ridge.fit(X, y)
# 获取 Ridge 回归器的系数
coef_ridge = ridge.coef_
# 计算 Ridge 回归器拟合直线的系数
coef_ = ridge.coef_ * x + ridge.intercept_
# 绘制 Ridge 回归器的拟合直线，使用绿色线条
plt.plot(x, coef_, "g-", label="ridge regression")

# 设置图表标题和坐标轴标签
plt.title("Comparison of HuberRegressor vs Ridge")
plt.xlabel("X")
plt.ylabel("y")
# 添加图例，显示线条含义
plt.legend(loc=0)
# 显示图形
plt.show()
```