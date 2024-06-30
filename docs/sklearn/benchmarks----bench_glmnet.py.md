# `D:\src\scipysrc\scikit-learn\benchmarks\bench_glmnet.py`

```
"""
To run this, you'll need to have installed.

  * glmnet-python
  * scikit-learn (of course)

Does two benchmarks

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

In both cases, only 10% of the features are informative.
"""

# 导入所需的库
import gc  # 引入垃圾回收模块，用于释放内存
from time import time  # 导入时间模块中的时间函数

import numpy as np  # 导入numpy库，用于科学计算

from sklearn.datasets import make_regression  # 导入sklearn中的make_regression函数，用于生成回归数据

alpha = 0.1
# alpha = 0.01


def rmse(a, b):
    # 计算均方根误差（Root Mean Squared Error，RMSE）
    return np.sqrt(np.mean((a - b) ** 2))


def bench(factory, X, Y, X_test, Y_test, ref_coef):
    gc.collect()  # 执行垃圾回收，释放内存

    # 记录开始时间
    tstart = time()
    # 使用给定的工厂函数创建分类器，并进行拟合
    clf = factory(alpha=alpha).fit(X, Y)
    # 计算经过的时间
    delta = time() - tstart
    # 记录结束时间

    # 打印执行时间
    print("duration: %0.3fs" % delta)
    # 打印预测值和实际值之间的均方根误差
    print("rmse: %f" % rmse(Y_test, clf.predict(X_test)))
    # 打印系数的平均绝对差异
    print("mean coef abs diff: %f" % abs(ref_coef - clf.coef_.ravel()).mean())
    return delta


if __name__ == "__main__":
    # 延迟导入matplotlib.pyplot
    import matplotlib.pyplot as plt
    from glmnet.elastic_net import Lasso as GlmnetLasso  # 从glmnet中导入Lasso回归模型

    from sklearn.linear_model import Lasso as ScikitLasso  # 从sklearn中导入Lasso回归模型

    scikit_results = []  # 存储sklearn结果的列表
    glmnet_results = []  # 存储glmnet结果的列表
    n = 20  # 迭代次数
    step = 500  # 增加的样本数的步长
    n_features = 1000  # 特征数
    n_informative = n_features / 10  # 信息特征数为总特征数的十分之一
    n_test_samples = 1000  # 测试样本数

    for i in range(1, n + 1):
        print("==================")
        print("Iteration %s of %s" % (i, n))
        print("==================")

        # 生成回归数据集
        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples,
            n_features=n_features,
            noise=0.1,
            n_informative=n_informative,
            coef=True,
        )

        # 划分训练集和测试集
        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[: (i * step)]
        Y = Y[: (i * step)]

        print("benchmarking scikit-learn: ")
        # 进行benchmark，记录结果到scikit_results
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: ")
        # 进行benchmark，记录结果到glmnet_results
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    # 绘制图表
    plt.clf()
    xx = range(0, n * step, step)
    plt.title("Lasso regression on sample dataset (%d features)" % n_features)
    plt.plot(xx, scikit_results, "b-", label="scikit-learn")
    plt.plot(xx, glmnet_results, "r-", label="glmnet")
    plt.legend()
    plt.xlabel("number of samples to classify")
    plt.ylabel("Time (s)")
    plt.show()

    # 现在进行一个benchmark，其中点的数量是固定的，变量是特征的数量

    scikit_results = []  # 重置存储sklearn结果的列表
    glmnet_results = []  # 重置存储glmnet结果的列表
    n = 20  # 迭代次数
    step = 100  # 增加的特征数的步长
    n_samples = 500
    # 对于给定的范围内的每一个整数 i，执行以下操作
    for i in range(1, n + 1):
        # 打印分隔线，标明当前迭代的编号和总迭代次数
        print("==================")
        print("Iteration %02d of %02d" % (i, n))
        print("==================")
        # 计算当前迭代中的特征数量
        n_features = i * step
        # 根据特征数量计算信息特征的数量
        n_informative = n_features / 10

        # 使用 make_regression 生成具有指定特性的回归数据集
        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples,  # 样本数量
            n_features=n_features,  # 特征数量
            noise=0.1,  # 噪声水平
            n_informative=n_informative,  # 信息特征数量
            coef=True,  # 返回真实系数
        )

        # 从生成的数据中分离出测试集
        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[:n_samples]  # 取前 n_samples 个样本作为训练集
        Y = Y[:n_samples]  # 对应的训练集标签

        # 打印信息，开始 benchmarking scikit-learn 的执行时间
        print("benchmarking scikit-learn: ")
        # 使用 bench 函数评估 ScikitLasso 模型的性能，并将结果添加到 scikit_results 列表中
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        # 打印信息，开始 benchmarking glmnet 的执行时间
        print("benchmarking glmnet: ")
        # 使用 bench 函数评估 GlmnetLasso 模型的性能，并将结果添加到 glmnet_results 列表中
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    # 生成一个包含从 100 到 100 + n * step，步长为 step 的数组 xx
    xx = np.arange(100, 100 + n * step, step)
    # 创建一个新的图形窗口，用于展示 scikit-learn 和 glmnet 的性能对比结果
    plt.figure("scikit-learn vs. glmnet benchmark results")
    # 设置图形的标题，指明所展示的是在高维空间中的回归，以及样本数量
    plt.title("Regression in high dimensional spaces (%d samples)" % n_samples)
    # 绘制 scikit-learn 和 glmnet 的性能对比曲线
    plt.plot(xx, scikit_results, "b-", label="scikit-learn")
    plt.plot(xx, glmnet_results, "r-", label="glmnet")
    # 添加图例
    plt.legend()
    # 设置 x 轴和 y 轴的标签
    plt.xlabel("number of features")
    plt.ylabel("Time (s)")
    # 自动调整坐标轴范围
    plt.axis("tight")
    # 展示图形
    plt.show()
```