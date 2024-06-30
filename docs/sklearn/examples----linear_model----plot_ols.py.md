# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ols.py`

```
"""
=========================================================
Linear Regression Example
=========================================================
The example below uses only the first feature of the `diabetes` dataset,
in order to illustrate the data points within the two-dimensional plot.
The straight line can be seen in the plot, showing how linear regression
attempts to draw a straight line that will best minimize the
residual sum of squares between the observed responses in the dataset,
and the responses predicted by the linear approximation.

The coefficients, residual sum of squares and the coefficient of
determination are also calculated.

"""

# Code source: Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# 导入 matplotlib 库用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于科学计算
import numpy as np

# 导入 sklearn 中的数据集和线性回归模型
from sklearn import datasets, linear_model
# 导入评估指标：均方误差和决定系数
from sklearn.metrics import mean_squared_error, r2_score

# 加载糖尿病数据集，返回特征 X 和目标 y
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 仅使用第三个特征（索引为2）作为输入特征
diabetes_X = diabetes_X[:, np.newaxis, 2]

# 将数据集分割为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标分割为训练集和测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 创建线性回归对象
regr = linear_model.LinearRegression()

# 使用训练集训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用测试集进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)

# 打印模型的系数
print("Coefficients: \n", regr.coef_)
# 打印均方误差
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 打印决定系数：1 表示完美预测
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# 绘制输出结果
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

# 展示绘制的图形
plt.show()
```