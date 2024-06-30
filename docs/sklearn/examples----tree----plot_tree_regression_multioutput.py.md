# `D:\src\scipysrc\scikit-learn\examples\tree\plot_tree_regression_multioutput.py`

```
"""
===================================================================
Multi-output Decision Tree Regression
===================================================================

An example to illustrate multi-output regression with decision tree.

The :ref:`decision trees <tree>`
is used to predict simultaneously the noisy x and y observations of a circle
given a single underlying feature. As a result, it learns local linear
regressions approximating the circle.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归模型

# 创建随机数据集
rng = np.random.RandomState(1)  # 创建随机数生成器
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)  # 生成排序后的随机数据
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T  # 根据X生成带有正弦和余弦噪声的y数据
y[::5, :] += 0.5 - rng.rand(20, 2)  # 对每隔五个样本的y数据加上随机噪声

# 拟合回归模型
regr_1 = DecisionTreeRegressor(max_depth=2)  # 创建最大深度为2的决策树回归模型
regr_2 = DecisionTreeRegressor(max_depth=5)  # 创建最大深度为5的决策树回归模型
regr_3 = DecisionTreeRegressor(max_depth=8)  # 创建最大深度为8的决策树回归模型
regr_1.fit(X, y)  # 使用数据拟合第一个模型
regr_2.fit(X, y)  # 使用数据拟合第二个模型
regr_3.fit(X, y)  # 使用数据拟合第三个模型

# 预测
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]  # 生成测试数据
y_1 = regr_1.predict(X_test)  # 对第一个模型进行预测
y_2 = regr_2.predict(X_test)  # 对第二个模型进行预测
y_3 = regr_3.predict(X_test)  # 对第三个模型进行预测

# 绘图
plt.figure()  # 创建新的图形
s = 25  # 散点大小
plt.scatter(y[:, 0], y[:, 1], c="navy", s=s, edgecolor="black", label="data")  # 绘制原始数据散点图
plt.scatter(
    y_1[:, 0],
    y_1[:, 1],
    c="cornflowerblue",
    s=s,
    edgecolor="black",
    label="max_depth=2",
)  # 绘制最大深度为2的预测结果散点图
plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s, edgecolor="black", label="max_depth=5")  # 绘制最大深度为5的预测结果散点图
plt.scatter(
    y_3[:, 0], y_3[:, 1], c="orange", s=s, edgecolor="black", label="max_depth=8"
)  # 绘制最大深度为8的预测结果散点图
plt.xlim([-6, 6])  # 设置x轴范围
plt.ylim([-6, 6])  # 设置y轴范围
plt.xlabel("target 1")  # 设置x轴标签
plt.ylabel("target 2")  # 设置y轴标签
plt.title("Multi-output Decision Tree Regression")  # 设置图标题
plt.legend(loc="best")  # 显示图例，位置自动选择最佳位置
plt.show()  # 显示图形
```