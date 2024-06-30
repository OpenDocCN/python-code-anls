# `D:\src\scipysrc\scikit-learn\benchmarks\bench_glm.py`

```
"""
A comparison of different methods in GLM

Data comes from a random square matrix.

"""

# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 sklearn 库中的 linear_model 模块
from sklearn import linear_model

# 如果直接运行当前脚本，则导入 matplotlib.pyplot 库，并使用别名 plt
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 迭代次数
    n_iter = 40

    # 分别创建存储 Ridge、OLS 和 LassoLars 方法运行时间的空数组
    time_ridge = np.empty(n_iter)
    time_ols = np.empty(n_iter)
    time_lasso = np.empty(n_iter)

    # 创建维度数组，从 500 到 20000，步长为 500
    dimensions = 500 * np.arange(1, n_iter + 1)

    # 开始迭代
    for i in range(n_iter):
        # 打印当前迭代次数
        print("Iteration %s of %s" % (i, n_iter))

        # 根据当前迭代次数计算样本数和特征数
        n_samples, n_features = 10 * i + 3, 10 * i + 3

        # 生成随机的样本矩阵 X 和目标值 Y
        X = np.random.randn(n_samples, n_features)
        Y = np.random.randn(n_samples)

        # 开始计时
        start = datetime.now()
        # 创建 Ridge 回归模型对象
        ridge = linear_model.Ridge(alpha=1.0)
        # 使用 X, Y 进行 Ridge 回归模型拟合
        ridge.fit(X, Y)
        # 计算 Ridge 方法的运行时间并存储
        time_ridge[i] = (datetime.now() - start).total_seconds()

        # 继续计时
        start = datetime.now()
        # 创建 OLS（普通最小二乘法）回归模型对象
        ols = linear_model.LinearRegression()
        # 使用 X, Y 进行 OLS 回归模型拟合
        ols.fit(X, Y)
        # 计算 OLS 方法的运行时间并存储
        time_ols[i] = (datetime.now() - start).total_seconds()

        # 继续计时
        start = datetime.now()
        # 创建 LassoLars 回归模型对象
        lasso = linear_model.LassoLars()
        # 使用 X, Y 进行 LassoLars 回归模型拟合
        lasso.fit(X, Y)
        # 计算 LassoLars 方法的运行时间并存储
        time_lasso[i] = (datetime.now() - start).total_seconds()

    # 绘制图像
    plt.figure("scikit-learn GLM benchmark results")
    plt.xlabel("Dimensions")
    plt.ylabel("Time (s)")
    plt.plot(dimensions, time_ridge, color="r")
    plt.plot(dimensions, time_ols, color="g")
    plt.plot(dimensions, time_lasso, color="b")

    # 添加图例和轴标签
    plt.legend(["Ridge", "OLS", "LassoLars"], loc="upper left")
    plt.axis("tight")
    plt.show()
```