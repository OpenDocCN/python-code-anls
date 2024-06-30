# `D:\src\scipysrc\scikit-learn\examples\tree\plot_tree_regression.py`

```
"""
===================================================================
Decision Tree Regression
===================================================================

A 1D regression with decision tree.

The :ref:`decision trees <tree>` is
used to fit a sine curve with addition noisy observation. As a result, it
learns local linear regressions approximating the sine curve.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""

# 导入必要的模块和库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

from sklearn.tree import DecisionTreeRegressor  # 从 sklearn 库中导入决策树回归器

# 创建随机数据集
rng = np.random.RandomState(1)  # 创建随机数生成器对象，种子为 1
X = np.sort(5 * rng.rand(80, 1), axis=0)  # 生成排序后的随机 X 数据，形状为 (80, 1)
y = np.sin(X).ravel()  # 计算对应的正弦值并展平为一维数组
y[::5] += 3 * (0.5 - rng.rand(16))  # 向每第五个元素加入噪声

# 拟合回归模型
regr_1 = DecisionTreeRegressor(max_depth=2)  # 创建最大深度为 2 的决策树回归模型
regr_2 = DecisionTreeRegressor(max_depth=5)  # 创建最大深度为 5 的决策树回归模型
regr_1.fit(X, y)  # 在数据集上拟合深度为 2 的决策树回归模型
regr_2.fit(X, y)  # 在数据集上拟合深度为 5 的决策树回归模型

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  # 生成测试数据，形状为 (500, 1)
y_1 = regr_1.predict(X_test)  # 使用深度为 2 的模型进行预测
y_2 = regr_2.predict(X_test)  # 使用深度为 5 的模型进行预测

# 绘制结果
plt.figure()  # 创建一个新的图形窗口
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")  # 绘制散点图，显示数据点
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)  # 绘制预测结果曲线，深度为 2
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)  # 绘制预测结果曲线，深度为 5
plt.xlabel("data")  # 设置 x 轴标签
plt.ylabel("target")  # 设置 y 轴标签
plt.title("Decision Tree Regression")  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图形
```