# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_parallel_pairwise.py`

```
`
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import time

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels
from sklearn.utils import check_random_state


# 定义一个装饰器函数，用于绘制函数运行时间的对比图
def plot(func):
    # 使用随机数生成器，种子为0
    random_state = check_random_state(0)
    # 初始化列表，用于存储单核和多核的运行时间
    one_core = []
    multi_core = []
    # 定义样本数量的范围，从1000到6000，步长为1000
    sample_sizes = range(1000, 6000, 1000)

    # 遍历样本数量
    for n_samples in sample_sizes:
        # 生成随机数据，形状为(n_samples, 300)
        X = random_state.rand(n_samples, 300)

        # 记录单核计算的开始时间
        start = time.time()
        # 调用func函数，使用1个核进行计算
        func(X, n_jobs=1)
        # 计算单核计算耗时，添加到one_core列表
        one_core.append(time.time() - start)

        # 记录多核计算的开始时间
        start = time.time()
        # 调用func函数，使用所有可用核进行计算
        func(X, n_jobs=-1)
        # 计算多核计算耗时，添加到multi_core列表
        multi_core.append(time.time() - start)

    # 创建一个新的图形窗口，标题为“scikit-learn parallel func函数名 benchmark results”
    plt.figure("scikit-learn parallel %s benchmark results" % func.__name__)
    # 绘制样本数量与单核计算时间的折线图
    plt.plot(sample_sizes, one_core, label="one core")
    # 绘制样本数量与多核计算时间的折线图
    plt.plot(sample_sizes, multi_core, label="multi core")
    # 设置x轴标签为“n_samples”
    plt.xlabel("n_samples")
    # 设置y轴标签为“Time (s)”
    plt.ylabel("Time (s)")
    # 设置图形标题为“Parallel func函数名”
    plt.title("Parallel %s" % func.__name__)
    # 显示图例
    plt.legend()


# 定义计算欧氏距离的函数，调用pairwise_distances，设置距离度量为“euclidean”
def euclidean_distances(X, n_jobs):
    return pairwise_distances(X, metric="euclidean", n_jobs=n_jobs)


# 定义计算RBF核函数的函数，调用pairwise_kernels，设置核函数为“rbf”，gamma参数为0.1
def rbf_kernels(X, n_jobs):
    return pairwise_kernels(X, metric="rbf", n_jobs=n_jobs, gamma=0.1)


# 调用plot函数，传入欧氏距离计算函数
plot(euclidean_distances)
# 调用plot函数，传入RBF核计算函数
plot(rbf_kernels)
# 显示所有绘制的图形
plt.show()
```