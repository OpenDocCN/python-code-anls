# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_incremental_pca.py`

```
"""
========================
IncrementalPCA benchmark
========================

Benchmarks for IncrementalPCA

"""

# 导入必要的库
import gc  # 垃圾回收模块，用于清理内存
from collections import defaultdict  # 默认字典，用于存储计数或者默认值
from time import time  # 时间模块，用于计算程序执行时间

import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数组操作库

from sklearn.datasets import fetch_lfw_people  # 导入 LFW 数据集
from sklearn.decomposition import PCA, IncrementalPCA  # PCA 和 IncrementalPCA 方法


def plot_results(X, y, label):
    """绘制结果曲线"""
    plt.plot(X, y, label=label, marker="o")


def benchmark(estimator, data):
    """评估方法性能"""
    gc.collect()  # 执行垃圾回收，清理内存
    print("Benching %s" % estimator)  # 打印评估的方法名称
    t0 = time()  # 记录开始时间
    estimator.fit(data)  # 使用给定数据拟合估计器
    training_time = time() - t0  # 计算训练时间
    data_t = estimator.transform(data)  # 对数据进行变换
    data_r = estimator.inverse_transform(data_t)  # 对变换后的数据进行逆变换
    reconstruction_error = np.mean(np.abs(data - data_r))  # 计算重构误差
    return {"time": training_time, "error": reconstruction_error}  # 返回时间和误差的字典


def plot_feature_times(all_times, batch_size, all_components, data):
    """绘制特征提取时间图"""
    plt.figure()
    plot_results(all_components, all_times["pca"], label="PCA")  # 绘制 PCA 曲线
    plot_results(
        all_components, all_times["ipca"], label="IncrementalPCA, bsize=%i" % batch_size
    )  # 绘制 IncrementalPCA 曲线
    plt.legend(loc="upper left")  # 设置图例位置
    plt.suptitle(
        "Algorithm runtime vs. n_components\n                  LFW, size %i x %i"
        % data.shape
    )  # 设置总标题
    plt.xlabel("Number of components (out of max %i)" % data.shape[1])  # 设置 x 轴标签
    plt.ylabel("Time (seconds)")  # 设置 y 轴标签


def plot_feature_errors(all_errors, batch_size, all_components, data):
    """绘制特征提取误差图"""
    plt.figure()
    plot_results(all_components, all_errors["pca"], label="PCA")  # 绘制 PCA 曲线
    plot_results(
        all_components,
        all_errors["ipca"],
        label="IncrementalPCA, bsize=%i" % batch_size,
    )  # 绘制 IncrementalPCA 曲线
    plt.legend(loc="lower left")  # 设置图例位置
    plt.suptitle("Algorithm error vs. n_components\nLFW, size %i x %i" % data.shape)  # 设置总标题
    plt.xlabel("Number of components (out of max %i)" % data.shape[1])  # 设置 x 轴标签
    plt.ylabel("Mean absolute error")  # 设置 y 轴标签


def plot_batch_times(all_times, n_features, all_batch_sizes, data):
    """绘制批处理时间图"""
    plt.figure()
    plot_results(all_batch_sizes, all_times["pca"], label="PCA")  # 绘制 PCA 曲线
    plot_results(all_batch_sizes, all_times["ipca"], label="IncrementalPCA")  # 绘制 IncrementalPCA 曲线
    plt.legend(loc="lower left")  # 设置图例位置
    plt.suptitle(
        "Algorithm runtime vs. batch_size for n_components %i\n                  LFW,"
        " size %i x %i" % (n_features, data.shape[0], data.shape[1])
    )  # 设置总标题
    plt.xlabel("Batch size")  # 设置 x 轴标签
    plt.ylabel("Time (seconds)")  # 设置 y 轴标签


def plot_batch_errors(all_errors, n_features, all_batch_sizes, data):
    """绘制批处理误差图"""
    plt.figure()
    plot_results(all_batch_sizes, all_errors["pca"], label="PCA")  # 绘制 PCA 曲线
    plot_results(all_batch_sizes, all_errors["ipca"], label="IncrementalPCA")  # 绘制 IncrementalPCA 曲线
    plt.legend(loc="lower left")  # 设置图例位置
    plt.suptitle(
        "Algorithm error vs. batch_size for n_components %i\n                  LFW,"
        " size %i x %i" % (n_features, data.shape[0], data.shape[1])
    )  # 设置总标题
    plt.xlabel("Batch size")  # 设置 x 轴标签
    plt.ylabel("Mean absolute error")  # 设置 y 轴标签


def fixed_batch_size_comparison(data):
    """固定批处理大小的比较"""
    all_features = [
        i.astype(int) for i in np.linspace(data.shape[1] // 10, data.shape[1], num=5)
    ]  # 生成特征数组
    # 定义批处理大小
    batch_size = 1000
    # 用于存储所有不同组件数量下的运行时间
    all_times = defaultdict(list)
    # 用于存储所有不同组件数量下的误差率
    all_errors = defaultdict(list)
    # 遍历所有特征数
    for n_components in all_features:
        # 创建 PCA 对象，设定主成分数为当前 n_components
        pca = PCA(n_components=n_components)
        # 创建增量 PCA 对象，设定主成分数和批处理大小
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        # 执行基准测试，并将结果存入字典 results_dict 中
        results_dict = {
            k: benchmark(est, data) for k, est in [("pca", pca), ("ipca", ipca)]
        }

        # 将结果按键名排序后，将时间和误差率添加到 all_times 和 all_errors 中
        for k in sorted(results_dict.keys()):
            all_times[k].append(results_dict[k]["time"])
            all_errors[k].append(results_dict[k]["error"])

    # 绘制特征处理时间图表
    plot_feature_times(all_times, batch_size, all_features, data)
    # 绘制特征处理误差图表
    plot_feature_errors(all_errors, batch_size, all_features, data)
def variable_batch_size_comparison(data):
    # 生成批次大小列表，每个大小是一个整数，范围从数据集行数的十分之一到整个数据集行数，共10个批次大小
    batch_sizes = [
        i.astype(int) for i in np.linspace(data.shape[0] // 10, data.shape[0], num=10)
    ]

    # 遍历每个主成分数量的值，这些值是从数据集列数的十分之一到整个数据集列数，共4个主成分数量
    for n_components in [
        i.astype(int) for i in np.linspace(data.shape[1] // 10, data.shape[1], num=4)
    ]:
        # 用于存储所有算法执行时间的默认字典
        all_times = defaultdict(list)
        # 用于存储所有算法误差的默认字典
        all_errors = defaultdict(list)
        
        # 创建 PCA 对象，并设置主成分数量为当前循环的 n_components
        pca = PCA(n_components=n_components)
        # 创建随机化 PCA 对象，设置主成分数量为当前循环的 n_components，并启用随机化方法
        rpca = PCA(
            n_components=n_components, svd_solver="randomized", random_state=1999
        )
        
        # 用来存储每个算法的评估结果的字典，包括 "pca" 和 "rpca"
        results_dict = {
            k: benchmark(est, data) for k, est in [("pca", pca), ("rpca", rpca)]
        }

        # 将 PCA 和 Randomized PCA 的时间和误差数据扁平化，并添加到 all_times 和 all_errors 中
        all_times["pca"].extend([results_dict["pca"]["time"]] * len(batch_sizes))
        all_errors["pca"].extend([results_dict["pca"]["error"]] * len(batch_sizes))
        all_times["rpca"].extend([results_dict["rpca"]["time"]] * len(batch_sizes))
        all_errors["rpca"].extend([results_dict["rpca"]["error"]] * len(batch_sizes))
        
        # 遍历批次大小列表，为每个批次大小创建 Incremental PCA 对象，并评估其性能
        for batch_size in batch_sizes:
            ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            results_dict = {k: benchmark(est, data) for k, est in [("ipca", ipca)]}
            all_times["ipca"].append(results_dict["ipca"]["time"])
            all_errors["ipca"].append(results_dict["ipca"]["error"])

        # 绘制批次时间变化图表，显示每种算法的执行时间随批次大小的变化
        plot_batch_times(all_times, n_components, batch_sizes, data)
        # 绘制批次误差变化图表，显示每种算法的误差随批次大小的变化
        plot_batch_errors(all_errors, n_components, batch_sizes, data)
```