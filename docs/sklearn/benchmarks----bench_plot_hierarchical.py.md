# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_hierarchical.py`

```
# 导入所需的库和模块
from collections import defaultdict  # 导入默认字典模块
from time import time  # 导入时间模块

import numpy as np  # 导入NumPy库
from numpy import random as nr  # 导入NumPy的随机模块

from sklearn.cluster import AgglomerativeClustering  # 导入层次聚类模块


def compute_bench(samples_range, features_range):
    it = 0  # 初始化迭代计数器
    results = defaultdict(lambda: [])  # 创建默认字典用于存储结果，默认值为空列表

    max_it = len(samples_range) * len(features_range)  # 计算总迭代次数
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1  # 每次迭代计数器加一
            print("==============================")
            print("Iteration %03d of %03d" % (it, max_it))  # 打印迭代信息
            print("n_samples %05d; n_features %02d" % (n_samples, n_features))
            print("==============================")
            print()
            data = nr.randint(-50, 51, (n_samples, n_features))  # 生成随机数据

            for linkage in ("single", "average", "complete", "ward"):
                print(linkage.capitalize())  # 打印链接方法名称
                tstart = time()  # 记录开始时间
                AgglomerativeClustering(linkage=linkage, n_clusters=10).fit(data)  # 执行层次聚类

                delta = time() - tstart  # 计算执行时间
                print("Speed: %0.3fs" % delta)  # 打印执行时间
                print()

                results[linkage].append(delta)  # 将执行时间记录到结果字典中

    return results  # 返回执行时间结果字典


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图

    samples_range = np.linspace(1000, 15000, 8).astype(int)  # 设置样本数量范围
    features_range = np.array([2, 10, 20, 50])  # 设置特征数量范围

    results = compute_bench(samples_range, features_range)  # 执行性能测试并获取结果

    max_time = max([max(i) for i in [t for (label, t) in results.items()]])  # 获取最长的执行时间

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:4]  # 获取颜色映射
    lines = {linkage: None for linkage in results.keys()}  # 创建存储线条对象的字典
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)  # 创建子图和图形对象
    fig.suptitle("Scikit-learn agglomerative clustering benchmark results", fontsize=16)  # 设置总标题
    for c, (label, timings) in zip(colors, sorted(results.items())):
        timing_by_samples = np.asarray(timings).reshape(
            samples_range.shape[0], features_range.shape[0]
        )  # 将执行时间转换成数组形式

        for n in range(timing_by_samples.shape[1]):
            ax = axs.flatten()[n]  # 获取当前子图对象
            (lines[label],) = ax.plot(
                samples_range, timing_by_samples[:, n], color=c, label=label
            )  # 绘制执行时间曲线
            ax.set_title("n_features = %d" % features_range[n])  # 设置子图标题
            if n >= 2:
                ax.set_xlabel("n_samples")  # 设置x轴标签
            if n % 2 == 0:
                ax.set_ylabel("time (s)")  # 设置y轴标签

    fig.subplots_adjust(right=0.8)  # 调整子图布局
    fig.legend(
        [lines[link] for link in sorted(results.keys())],  # 设置图例
        sorted(results.keys()),
        loc="center right",
        fontsize=8,
    )

    plt.show()  # 显示绘制的图形
```