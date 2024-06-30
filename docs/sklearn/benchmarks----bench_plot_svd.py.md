# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_svd.py`

```
"""
Singular Value Decomposition（SVD）的性能评测（精确和近似）

数据大部分为低秩，但有一个“胖”无限尾巴。
"""

# 导入必要的库
import gc  # 垃圾回收模块，用于清理内存
from collections import defaultdict  # 默认字典，用于存储结果
from time import time  # 计时功能

import numpy as np  # 数组操作库
from scipy.linalg import svd  # SciPy的SVD函数

from sklearn.datasets import make_low_rank_matrix  # 生成低秩矩阵的函数
from sklearn.utils.extmath import randomized_svd  # scikit-learn的随机SVD函数

# 计算性能的函数，接受不同的样本和特征范围
def compute_bench(samples_range, features_range, n_iter=3, rank=50):
    it = 0

    results = defaultdict(lambda: [])  # 使用默认字典来存储结果，每个键对应一个空列表

    max_it = len(samples_range) * len(features_range)  # 最大迭代次数
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1
            print("====================")
            print("Iteration %03d of %03d" % (it, max_it))
            print("====================")
            
            # 生成低秩矩阵
            X = make_low_rank_matrix(
                n_samples, n_features, effective_rank=rank, tail_strength=0.2
            )

            gc.collect()  # 手动进行垃圾回收，清理内存
            print("benchmarking scipy svd: ")
            tstart = time()  # 记录开始时间
            svd(X, full_matrices=False)  # 使用SciPy进行SVD分解
            results["scipy svd"].append(time() - tstart)  # 记录耗时

            gc.collect()  # 再次清理内存
            print("benchmarking scikit-learn randomized_svd: n_iter=0")
            tstart = time()  # 记录开始时间
            randomized_svd(X, rank, n_iter=0)  # 使用scikit-learn的随机SVD，迭代次数为0
            results["scikit-learn randomized_svd (n_iter=0)"].append(time() - tstart)  # 记录耗时

            gc.collect()  # 再次清理内存
            print("benchmarking scikit-learn randomized_svd: n_iter=%d " % n_iter)
            tstart = time()  # 记录开始时间
            randomized_svd(X, rank, n_iter=n_iter)  # 使用scikit-learn的随机SVD，指定迭代次数
            results["scikit-learn randomized_svd (n_iter=%d)" % n_iter].append(
                time() - tstart  # 记录耗时
            )

    return results  # 返回性能结果的字典


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import axes3d  # 导入3D绘图工具
    import matplotlib.pyplot as plt  # 导入绘图库

    samples_range = np.linspace(2, 1000, 4).astype(int)  # 生成样本数量范围
    features_range = np.linspace(2, 1000, 4).astype(int)  # 生成特征数量范围
    results = compute_bench(samples_range, features_range)  # 计算性能结果

    label = "scikit-learn singular value decomposition benchmark results"
    fig = plt.figure(label)  # 创建图形对象
    ax = fig.gca(projection="3d")  # 获取3D坐标轴

    # 对每个结果进行排序，并用不同颜色绘制
    for c, (label, timings) in zip("rbg", sorted(results.items())):
        X, Y = np.meshgrid(samples_range, features_range)  # 创建网格数据
        Z = np.asarray(timings).reshape(samples_range.shape[0], features_range.shape[0])  # 转换耗时数据为数组
        # 绘制表面图
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, color=c)
        # 添加标签
        ax.plot([1], [1], [1], color=c, label=label)

    # 设置坐标轴标签和图例
    ax.set_xlabel("n_samples")
    ax.set_ylabel("n_features")
    ax.set_zlabel("Time (s)")
    ax.legend()  # 显示图例
    plt.show()  # 展示图形
```