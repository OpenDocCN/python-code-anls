# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_neighbors.py`

```
"""
Plot the scaling of the nearest neighbors algorithms with k, D, and N
"""

from time import time  # 导入时间计算函数

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库
from matplotlib import ticker  # 导入绘图辅助工具

from sklearn import datasets, neighbors  # 导入数据集和最近邻模块


def get_data(N, D, dataset="dense"):
    """
    根据参数生成指定类型的数据集

    Parameters:
    -----------
    N : int
        数据集样本数目
    D : int
        数据集维度
    dataset : str, optional
        数据集类型，可以是'dense'或'digits'

    Returns:
    --------
    X : numpy.ndarray
        生成的数据集
    """
    if dataset == "dense":
        np.random.seed(0)
        return np.random.random((N, D))  # 返回随机生成的密集型数据集
    elif dataset == "digits":
        X, _ = datasets.load_digits(return_X_y=True)
        i = np.argsort(X[0])[::-1]
        X = X[:, i]
        return X[:N, :D]  # 返回加载的手写数字数据集的前N个样本和D维特征
    else:
        raise ValueError("invalid dataset: %s" % dataset)


def barplot_neighbors(
    Nrange=2 ** np.arange(1, 11),
    Drange=2 ** np.arange(7),
    krange=2 ** np.arange(10),
    N=1000,
    D=64,
    k=5,
    leaf_size=30,
    dataset="digits",
):
    """
    绘制最近邻算法在不同参数下的性能比较图

    Parameters:
    -----------
    Nrange : numpy.ndarray, optional
        样本数范围
    Drange : numpy.ndarray, optional
        维度范围
    krange : numpy.ndarray, optional
        k近邻数范围
    N : int, optional
        默认样本数
    D : int, optional
        默认维度数
    k : int, optional
        默认k近邻数
    leaf_size : int, optional
        叶子大小参数
    dataset : str, optional
        数据集类型，默认为手写数字数据集
    """
    algorithms = ("kd_tree", "brute", "ball_tree")  # 定义最近邻算法

    fiducial_values = {"N": N, "D": D, "k": k}

    # ------------------------------------------------------------
    # varying N
    N_results_build = {alg: np.zeros(len(Nrange)) for alg in algorithms}  # 初始化存储建模时间的字典
    N_results_query = {alg: np.zeros(len(Nrange)) for alg in algorithms}  # 初始化存储查询时间的字典

    for i, NN in enumerate(Nrange):
        print("N = %i (%i out of %i)" % (NN, i + 1, len(Nrange)))
        X = get_data(NN, D, dataset)
        for algorithm in algorithms:
            nbrs = neighbors.NearestNeighbors(
                n_neighbors=min(NN, k), algorithm=algorithm, leaf_size=leaf_size
            )  # 初始化最近邻对象
            t0 = time()  # 记录开始时间
            nbrs.fit(X)  # 拟合数据
            t1 = time()  # 记录建模完成时间
            nbrs.kneighbors(X)  # 查询k近邻
            t2 = time()  # 记录查询完成时间

            N_results_build[algorithm][i] = t1 - t0  # 存储建模时间
            N_results_query[algorithm][i] = t2 - t1  # 存储查询时间

    # ------------------------------------------------------------
    # varying D
    D_results_build = {alg: np.zeros(len(Drange)) for alg in algorithms}  # 初始化存储建模时间的字典
    D_results_query = {alg: np.zeros(len(Drange)) for alg in algorithms}  # 初始化存储查询时间的字典

    for i, DD in enumerate(Drange):
        print("D = %i (%i out of %i)" % (DD, i + 1, len(Drange)))
        X = get_data(N, DD, dataset)
        for algorithm in algorithms:
            nbrs = neighbors.NearestNeighbors(
                n_neighbors=k, algorithm=algorithm, leaf_size=leaf_size
            )  # 初始化最近邻对象
            t0 = time()  # 记录开始时间
            nbrs.fit(X)  # 拟合数据
            t1 = time()  # 记录建模完成时间
            nbrs.kneighbors(X)  # 查询k近邻
            t2 = time()  # 记录查询完成时间

            D_results_build[algorithm][i] = t1 - t0  # 存储建模时间
            D_results_query[algorithm][i] = t2 - t1  # 存储查询时间

    # ------------------------------------------------------------
    # varying k
    k_results_build = {alg: np.zeros(len(krange)) for alg in algorithms}  # 初始化存储建模时间的字典
    k_results_query = {alg: np.zeros(len(krange)) for alg in algorithms}  # 初始化存储查询时间的字典

    X = get_data(N, D, dataset)  # 获取数据集
    for i, kk in enumerate(krange):
        # 打印当前 k 的值和进度信息
        print("k = %i (%i out of %i)" % (kk, i + 1, len(krange)))
        for algorithm in algorithms:
            # 使用给定参数创建最近邻对象
            nbrs = neighbors.NearestNeighbors(
                n_neighbors=kk, algorithm=algorithm, leaf_size=leaf_size
            )
            # 记录开始拟合模型的时间
            t0 = time()
            # 使用数据 X 拟合最近邻模型
            nbrs.fit(X)
            # 记录拟合模型完成的时间
            t1 = time()
            # 查询最近邻
            nbrs.kneighbors(X)
            # 记录查询的时间
            t2 = time()

            # 记录构建模型时间到结果字典
            k_results_build[algorithm][i] = t1 - t0
            # 记录查询时间到结果字典
            k_results_query[algorithm][i] = t2 - t1

    plt.figure(figsize=(8, 11))

    for sbplt, vals, quantity, build_time, query_time in [
        (311, Nrange, "N", N_results_build, N_results_query),
        (312, Drange, "D", D_results_build, D_results_query),
        (313, krange, "k", k_results_build, k_results_query),
    ]:
        # 在指定的子图位置创建子图，并使用对数坐标轴
        ax = plt.subplot(sbplt, yscale="log")
        # 显示网格
        plt.grid(True)

        tick_vals = []
        tick_labels = []

        # 计算底部起点
        bottom = 10 ** np.min(
            [min(np.floor(np.log10(build_time[alg]))) for alg in algorithms]
        )

        for i, alg in enumerate(algorithms):
            # 确定条形图的位置和宽度
            xvals = 0.1 + i * (1 + len(vals)) + np.arange(len(vals))
            width = 0.8

            # 绘制构建时间的条形图，并设置颜色
            c_bar = plt.bar(xvals, build_time[alg] - bottom, width, bottom, color="r")
            # 绘制查询时间的条形图，并设置颜色
            q_bar = plt.bar(xvals, query_time[alg], width, build_time[alg], color="b")

            # 设置 x 轴刻度值和标签
            tick_vals += list(xvals + 0.5 * width)
            tick_labels += ["%i" % val for val in vals]

            # 在子图上方添加算法名称文本框
            plt.text(
                (i + 0.02) / len(algorithms),
                0.98,
                alg,
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(facecolor="w", edgecolor="w", alpha=0.5),
            )

            # 设置 y 轴标签
            plt.ylabel("Time (s)")

        # 设置 x 轴主要刻度定位器和格式化器
        ax.xaxis.set_major_locator(ticker.FixedLocator(tick_vals))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))

        # 调整 x 轴标签的旋转角度和字体大小
        for label in ax.get_xticklabels():
            label.set_rotation(-90)
            label.set_fontsize(10)

        # 设置子图标题
        title_string = "Varying %s" % quantity

        # 设置描述信息字符串
        descr_string = ""

        for s in "NDk":
            if s == quantity:
                pass
            else:
                descr_string += "%s = %i, " % (s, fiducial_values[s])

        descr_string = descr_string[:-2]

        # 在子图上方左侧和右侧添加标题和描述信息
        plt.text(
            1.01,
            0.5,
            title_string,
            transform=ax.transAxes,
            rotation=-90,
            ha="left",
            va="center",
            fontsize=20,
        )

        plt.text(
            0.99,
            0.5,
            descr_string,
            transform=ax.transAxes,
            rotation=-90,
            ha="right",
            va="center",
        )

        # 设置整个图的标题
        plt.gcf().suptitle("%s data set" % dataset.capitalize(), fontsize=16)

    # 添加图例并显示
    plt.figlegend((c_bar, q_bar), ("construction", "N-point query"), "upper right")
# 如果当前脚本作为主程序执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 调用 barplot_neighbors 函数，绘制关于 digits 数据集的条形图
    barplot_neighbors(dataset="digits")
    # 调用 barplot_neighbors 函数，绘制关于 dense 数据集的条形图
    barplot_neighbors(dataset="dense")
    # 显示所有已经绘制的图形
    plt.show()
```