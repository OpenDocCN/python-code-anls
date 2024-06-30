# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_fastkmeans.py`

```
# 导入必要的模块和函数
from collections import defaultdict  # 导入 defaultdict 类实现默认值字典
from time import time  # 导入 time 函数用于计时

import numpy as np  # 导入 NumPy 库并重命名为 np
from numpy import random as nr  # 从 NumPy 中导入 random 模块并重命名为 nr

from sklearn.cluster import KMeans, MiniBatchKMeans  # 从 sklearn.cluster 导入 KMeans 和 MiniBatchKMeans 类


def compute_bench(samples_range, features_range):
    # 初始化迭代计数器和结果字典
    it = 0
    results = defaultdict(lambda: [])
    chunk = 100  # 设置 MiniBatchKMeans 的批处理大小

    max_it = len(samples_range) * len(features_range)  # 计算总迭代次数
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1
            print("==============================")
            print("Iteration %03d of %03d" % (it, max_it))
            print("==============================")
            print()

            # 生成指定范围内随机整数的二维数组
            data = nr.randint(-50, 51, (n_samples, n_features))

            print("K-Means")
            tstart = time()  # 记录开始时间
            kmeans = KMeans(init="k-means++", n_clusters=10).fit(data)  # 使用 K-Means 算法拟合数据集

            delta = time() - tstart  # 计算执行时间
            print("Speed: %0.3fs" % delta)  # 打印运行时间
            print("Inertia: %0.5f" % kmeans.inertia_)  # 打印聚类的惯性
            print()

            results["kmeans_speed"].append(delta)  # 将 K-Means 运行时间添加到结果字典中
            results["kmeans_quality"].append(kmeans.inertia_)  # 将 K-Means 的惯性值添加到结果字典中

            print("Fast K-Means")
            # 使用小批量 K-Means 算法拟合数据集
            mbkmeans = MiniBatchKMeans(
                init="k-means++", n_clusters=10, batch_size=chunk
            )
            tstart = time()  # 记录开始时间
            mbkmeans.fit(data)  # 拟合数据集
            delta = time() - tstart  # 计算执行时间
            print("Speed: %0.3fs" % delta)  # 打印运行时间
            print("Inertia: %f" % mbkmeans.inertia_)  # 打印聚类的惯性
            print()
            print()

            results["MiniBatchKMeans Speed"].append(delta)  # 将小批量 K-Means 运行时间添加到结果字典中
            results["MiniBatchKMeans Quality"].append(mbkmeans.inertia_)  # 将小批量 K-Means 的惯性值添加到结果字典中

    return results  # 返回结果字典


def compute_bench_2(chunks):
    results = defaultdict(lambda: [])  # 初始化结果字典
    n_features = 50000  # 设置数据集特征数
    means = np.array(  # 定义聚类中心均值数组
        [
            [1, 1],
            [-1, -1],
            [1, -1],
            [-1, 1],
            [0.5, 0.5],
            [0.75, -0.5],
            [-1, 0.75],
            [1, 0],
        ]
    )
    X = np.empty((0, 2))  # 初始化空的数据集 X
    for i in range(8):
        X = np.r_[X, means[i] + 0.8 * np.random.randn(n_features, 2)]  # 生成具有偏差的聚类数据

    max_it = len(chunks)  # 计算迭代次数
    it = 0  # 初始化迭代计数器
    for chunk in chunks:
        it += 1
        print("==============================")
        print("Iteration %03d of %03d" % (it, max_it))
        print("==============================")
        print()

        print("Fast K-Means")
        tstart = time()  # 记录开始时间
        mbkmeans = MiniBatchKMeans(init="k-means++", n_clusters=8, batch_size=chunk)  # 使用小批量 K-Means 算法拟合数据集

        mbkmeans.fit(X)  # 拟合数据集
        delta = time() - tstart  # 计算执行时间
        print("Speed: %0.3fs" % delta)  # 打印运行时间
        print("Inertia: %0.3fs" % mbkmeans.inertia_)  # 打印聚类的惯性
        print()

        results["MiniBatchKMeans Speed"].append(delta)  # 将小批量 K-Means 运行时间添加到结果字典中
        results["MiniBatchKMeans Quality"].append(mbkmeans.inertia_)  # 将小批量 K-Means 的惯性值添加到结果字典中

    return results  # 返回结果字典


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import axes3d  # 导入 matplotlib 的 3D 图形工具包
    import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块用于绘图

    samples_range = np.linspace(50, 150, 5).astype(int)  # 生成样本范围数组
    # 定义一组特征范围，从150到50000之间生成5个整数作为特征数量的列表
    features_range = np.linspace(150, 50000, 5).astype(int)
    
    # 定义一组分块大小范围，从500到10000之间生成15个整数作为分块大小的列表
    chunks = np.linspace(500, 10000, 15).astype(int)
    
    # 调用 compute_bench 函数计算基于给定样本范围和特征范围的性能结果
    results = compute_bench(samples_range, features_range)
    
    # 调用 compute_bench_2 函数计算基于给定分块大小范围的性能结果
    results_2 = compute_bench_2(chunks)
    
    # 找到结果中标签包含 "speed" 的最大时间值
    max_time = max(
        [max(i) for i in [t for (label, t) in results.items() if "speed" in label]]
    )
    
    # 找到结果中标签不包含 "speed" 的最大惯性值
    max_inertia = max(
        [max(i) for i in [t for (label, t) in results.items() if "speed" not in label]]
    )
    
    # 创建一个名为 "scikit-learn K-Means benchmark results" 的图形窗口
    fig = plt.figure("scikit-learn K-Means benchmark results")
    
    # 遍历已排序的结果字典，根据标签选择合适的子图和设置Z轴的范围
    for c, (label, timings) in zip("brcy", sorted(results.items())):
        if "speed" in label:
            ax = fig.add_subplot(2, 2, 1, projection="3d")
            ax.set_zlim3d(0.0, max_time * 1.1)
        else:
            ax = fig.add_subplot(2, 2, 2, projection="3d")
            ax.set_zlim3d(0.0, max_inertia * 1.1)
    
        # 创建一个二维网格，用样本范围和特征范围作为X和Y轴，将时间数据转换为Z轴数据并绘制曲面
        X, Y = np.meshgrid(samples_range, features_range)
        Z = np.asarray(timings).reshape(samples_range.shape[0], features_range.shape[0])
        ax.plot_surface(X, Y, Z.T, cstride=1, rstride=1, color=c, alpha=0.5)
        ax.set_xlabel("n_samples")
        ax.set_ylabel("n_features")
    
    i = 0
    # 遍历已排序的第二组结果字典，绘制分块大小与时间之间的关系
    for c, (label, timings) in zip("br", sorted(results_2.items())):
        i += 1
        ax = fig.add_subplot(2, 2, i + 2)
        y = np.asarray(timings)
        ax.plot(chunks, y, color=c, alpha=0.8)
        ax.set_xlabel("Chunks")
        ax.set_ylabel(label)
    
    # 显示图形窗口
    plt.show()
```