# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_underfitting_overfitting.py`

```
"""
============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.model_selection import cross_val_score  # 导入交叉验证函数
from sklearn.pipeline import Pipeline  # 导入管道工具，用于组合多个处理步骤
from sklearn.preprocessing import PolynomialFeatures  # 导入多项式特征生成器


def true_fun(X):
    return np.cos(1.5 * np.pi * X)  # 定义真实函数


np.random.seed(0)

n_samples = 30  # 样本数量
degrees = [1, 4, 15]  # 不同的多项式阶数

X = np.sort(np.random.rand(n_samples))  # 生成随机样本并排序
y = true_fun(X) + np.random.randn(n_samples) * 0.1  # 根据真实函数添加随机噪声

plt.figure(figsize=(14, 5))  # 创建图像窗口

for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)  # 在图中创建子图
    plt.setp(ax, xticks=(), yticks=())  # 设置子图的坐标轴刻度为空

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)  # 创建指定阶数的多项式特征生成器
    linear_regression = LinearRegression()  # 创建线性回归模型
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),  # 将多项式特征生成器添加到管道中
            ("linear_regression", linear_regression),  # 将线性回归模型添加到管道中
        ]
    )
    pipeline.fit(X[:, np.newaxis], y)  # 在训练数据上拟合管道模型

    # Evaluate the models using crossvalidation
    scores = cross_val_score(
        pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
    )  # 使用交叉验证计算模型的均方误差（MSE）

    X_test = np.linspace(0, 1, 100)  # 生成用于测试的数据点
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")  # 绘制模型的预测曲线
    plt.plot(X_test, true_fun(X_test), label="True function")  # 绘制真实函数曲线
    plt.scatter(X, y, edgecolor="b", s=20, label="Samples")  # 绘制样本数据点
    plt.xlabel("x")  # 设置 x 轴标签
    plt.ylabel("y")  # 设置 y 轴标签
    plt.xlim((0, 1))  # 设置 x 轴范围
    plt.ylim((-2, 2))  # 设置 y 轴范围
    plt.legend(loc="best")  # 设置图例位置
    plt.title(
        "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()
        )
    )  # 设置子图标题，显示阶数和均方误差（MSE）

plt.show()  # 显示图形
```