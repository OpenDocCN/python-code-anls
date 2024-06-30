# `D:\src\scipysrc\scikit-learn\benchmarks\bench_tree.py`

```
"""
To run this, you'll need to have installed.

  * scikit-learn

Does two benchmarks

First, we fix a training set, increase the number of
samples to classify and plot number of classified samples as a
function of time.

In the second benchmark, we increase the number of dimensions of the
training set, classify a sample and plot the time taken as a function
of the number of dimensions.
"""

import gc  # 引入垃圾回收模块
from datetime import datetime  # 引入日期时间模块

import matplotlib.pyplot as plt  # 引入绘图模块
import numpy as np  # 引入数值计算模块

# to store the results
scikit_classifier_results = []  # 存储分类器的结果
scikit_regressor_results = []  # 存储回归器的结果

mu_second = 0.0 + 10**6  # 定义微秒数，1秒 = 10^6 微秒


def bench_scikit_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""
    from sklearn.tree import DecisionTreeClassifier  # 从scikit-learn中导入决策树分类器

    gc.collect()  # 手动触发垃圾回收

    # start time
    tstart = datetime.now()  # 获取当前时间
    clf = DecisionTreeClassifier()  # 创建决策树分类器对象
    clf.fit(X, Y).predict(X)  # 拟合数据并进行预测
    delta = datetime.now() - tstart  # 计算时间差
    # stop time

    scikit_classifier_results.append(delta.seconds + delta.microseconds / mu_second)  # 将耗时结果加入列表


def bench_scikit_tree_regressor(X, Y):
    """Benchmark with scikit-learn decision tree regressor"""
    from sklearn.tree import DecisionTreeRegressor  # 从scikit-learn中导入决策树回归器

    gc.collect()  # 手动触发垃圾回收

    # start time
    tstart = datetime.now()  # 获取当前时间
    clf = DecisionTreeRegressor()  # 创建决策树回归器对象
    clf.fit(X, Y).predict(X)  # 拟合数据并进行预测
    delta = datetime.now() - tstart  # 计算时间差
    # stop time

    scikit_regressor_results.append(delta.seconds + delta.microseconds / mu_second)  # 将耗时结果加入列表


if __name__ == "__main__":
    print("============================================")
    print("Warning: this is going to take a looong time")
    print("============================================")

    n = 10  # 迭代次数
    step = 10000  # 每次迭代增加的样本数
    n_samples = 10000  # 初始样本数
    dim = 10  # 初始维度
    n_classes = 10  # 类别数
    for i in range(n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))  # 打印当前迭代次数信息
        print("============================================")
        n_samples += step  # 增加样本数
        X = np.random.randn(n_samples, dim)  # 生成服从标准正态分布的随机样本数据
        Y = np.random.randint(0, n_classes, (n_samples,))  # 生成随机整数类标签
        bench_scikit_tree_classifier(X, Y)  # 执行分类器性能测试
        Y = np.random.randn(n_samples)  # 生成服从标准正态分布的随机目标数据
        bench_scikit_tree_regressor(X, Y)  # 执行回归器性能测试

    xx = range(0, n * step, step)  # x轴数据范围
    plt.figure("scikit-learn tree benchmark results")  # 创建绘图窗口
    plt.subplot(211)  # 创建子图
    plt.title("Learning with varying number of samples")  # 设置子图标题
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")  # 绘制分类器结果曲线
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")  # 绘制回归器结果曲线
    plt.legend(loc="upper left")  # 设置图例位置
    plt.xlabel("number of samples")  # 设置x轴标签
    plt.ylabel("Time (s)")  # 设置y轴标签

    scikit_classifier_results = []  # 清空分类器结果列表
    scikit_regressor_results = []  # 清空回归器结果列表
    n = 10  # 迭代次数
    step = 500  # 每次迭代增加的维度数
    start_dim = 500  # 初始维度
    n_classes = 10  # 类别数

    dim = start_dim  # 设置当前维度为初始维度
    # 循环从 0 到 n-1，执行以下操作
    for i in range(0, n):
        # 打印分隔线，标识进入第 i 次迭代
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        # 增加维度参数 dim
        dim += step
        # 生成一个 100x(dim) 的随机数组 X
        X = np.random.randn(100, dim)
        # 生成一个长度为 100 的随机整数数组 Y，取值范围为 [0, n_classes)
        Y = np.random.randint(0, n_classes, (100,))
        # 调用 bench_scikit_tree_classifier 函数，评估分类器性能
        bench_scikit_tree_classifier(X, Y)
        # 生成一个长度为 100 的随机浮点数数组 Y
        Y = np.random.randn(100)
        # 调用 bench_scikit_tree_regressor 函数，评估回归器性能
        bench_scikit_tree_regressor(X, Y)

    # 生成一个从 start_dim 开始，步长为 step，共 n 个元素的数组 xx
    xx = np.arange(start_dim, start_dim + n * step, step)
    # 绘制子图，标题为 "Learning in high dimensional spaces"
    plt.subplot(212)
    # 设置子图标题
    plt.title("Learning in high dimensional spaces")
    # 绘制分类器结果曲线，颜色为绿色
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    # 绘制回归器结果曲线，颜色为红色
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    # 显示图例，位置在左上角
    plt.legend(loc="upper left")
    # 设置 x 轴标签为 "number of dimensions"
    plt.xlabel("number of dimensions")
    # 设置 y 轴标签为 "Time (s)"
    plt.ylabel("Time (s)")
    # 自动调整坐标轴范围
    plt.axis("tight")
    # 显示图形
    plt.show()
```