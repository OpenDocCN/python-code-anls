# `D:\src\scipysrc\scikit-learn\examples\applications\plot_prediction_latency.py`

```
"""
==================
Prediction Latency
==================

This is an example showing the prediction latency of various scikit-learn
estimators.

The goal is to measure the latency one can expect when doing predictions
either in bulk or atomic (i.e. one by one) mode.

The plots represent the distribution of the prediction latency as a boxplot.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import gc                # 导入垃圾回收模块，用于内存管理
import time              # 导入时间模块，用于测量程序运行时间
from collections import defaultdict   # 导入defaultdict类，用于创建默认字典

import matplotlib.pyplot as plt   # 导入matplotlib库的pyplot模块，用于绘图
import numpy as np        # 导入NumPy库，用于数值计算

from sklearn.datasets import make_regression   # 导入make_regression函数，用于生成回归数据集
from sklearn.ensemble import RandomForestRegressor   # 导入随机森林回归器
from sklearn.linear_model import Ridge, SGDRegressor   # 导入岭回归和随机梯度下降回归器
from sklearn.model_selection import train_test_split   # 导入train_test_split函数，用于数据集划分
from sklearn.preprocessing import StandardScaler   # 导入StandardScaler类，用于特征标准化
from sklearn.svm import SVR   # 导入支持向量回归器
from sklearn.utils import shuffle   # 导入shuffle函数，用于数据随机化


def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return "__file__" in globals()


# %%
# Benchmark and plot helper functions
# -----------------------------------


def atomic_benchmark_estimator(estimator, X_test, verbose=False):
    """Measure runtime prediction of each instance."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    for i in range(n_instances):
        instance = X_test[[i], :]
        start = time.time()
        estimator.predict(instance)
        runtimes[i] = time.time() - start
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose):
    """Measure runtime prediction of the whole input."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_bulk_repeats, dtype=float)
    for i in range(n_bulk_repeats):
        start = time.time()
        estimator.predict(X_test)
        runtimes[i] = time.time() - start
    runtimes = np.array(list(map(lambda x: x / float(n_instances), runtimes)))
    if verbose:
        print(
            "bulk_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def benchmark_estimator(estimator, X_test, n_bulk_repeats=30, verbose=False):
    """
    Measure runtimes of prediction in both atomic and bulk mode.

    Parameters
    ----------
    estimator : already trained estimator supporting `predict()`
    X_test : test input
    n_bulk_repeats : how many times to repeat when evaluating bulk mode

    Returns
    -------
    atomic_runtimes, bulk_runtimes : a pair of `np.array` which contain the
    runtimes in seconds.

    """
    atomic_runtimes = atomic_benchmark_estimator(estimator, X_test, verbose)
    bulk_runtimes = bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose)
    return atomic_runtimes, bulk_runtimes
# 这个函数用于生成具有指定参数的回归数据集
def generate_dataset(n_train, n_test, n_features, noise=0.1, verbose=False):
    """Generate a regression dataset with the given parameters."""
    # 如果 verbose 参数为 True，则打印生成数据集的信息
    if verbose:
        print("generating dataset...")

    # 使用 make_regression 生成回归数据集，包括特征矩阵 X、目标值 y 和真实系数 coef
    X, y, coef = make_regression(
        n_samples=n_train + n_test, n_features=n_features, noise=noise, coef=True
    )

    # 设定一个随机种子，用于分割训练集和测试集的随机状态
    random_seed = 13
    # 将数据集 X 和 y 按照指定比例分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=random_seed
    )
    # 对训练集的特征矩阵 X 和目标值 y 进行随机打乱
    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)

    # 对特征矩阵 X 进行标准化处理
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    # 对目标值 y 进行标准化处理
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]
    y_test = y_scaler.transform(y_test[:, None])[:, 0]

    # 手动进行垃圾回收，释放内存
    gc.collect()
    # 如果 verbose 参数为 True，则打印数据集生成完成的信息
    if verbose:
        print("ok")
    # 返回处理后的训练集和测试集数据
    return X_train, y_train, X_test, y_test


def boxplot_runtimes(runtimes, pred_type, configuration):
    """
    Plot a new `Figure` with boxplots of prediction runtimes.

    Parameters
    ----------
    runtimes : list of `np.array` of latencies in micro-seconds
    cls_names : list of estimator class names that generated the runtimes
    pred_type : 'bulk' or 'atomic'

    """

    # 创建一个新的图形 Figure，设置尺寸为 10x6
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 使用 plt.boxplot 绘制预测运行时间的箱线图
    bp = plt.boxplot(
        runtimes,
    )

    # 生成每个估计器配置信息的字符串列表，用于作为 x 轴刻度标签
    cls_infos = [
        "%s\n(%d %s)"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
        )
        for estimator_conf in configuration["estimators"]
    ]
    # 设置 x 轴刻度标签为生成的估计器配置信息字符串列表
    plt.setp(ax1, xticklabels=cls_infos)
    # 设置箱线图的盒子颜色为黑色
    plt.setp(bp["boxes"], color="black")
    # 设置箱线图的须线颜色为黑色
    plt.setp(bp["whiskers"], color="black")
    # 设置异常值的颜色为红色，标记为 "+"
    plt.setp(bp["fliers"], color="red", marker="+")

    # 设置 y 轴网格线为灰色，透明度为 0.5
    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    # 将坐标轴设为图形的底层
    ax1.set_axisbelow(True)
    # 设置图形标题，显示预测类型（bulk 或 atomic）和特征数目
    ax1.set_title(
        "Prediction Time per Instance - %s, %d feats."
        % (pred_type.capitalize(), configuration["n_features"])
    )
    # 设置 y 轴标签为 "Prediction Time (us)"
    ax1.set_ylabel("Prediction Time (us)")

    # 显示图形
    plt.show()


def benchmark(configuration):
    """Run the whole benchmark."""
    # 使用 generate_dataset 生成指定配置的训练集和测试集
    X_train, y_train, X_test, y_test = generate_dataset(
        configuration["n_train"], configuration["n_test"], configuration["n_features"]
    )

    # 初始化一个空字典用于存储统计数据
    stats = {}
    # 遍历每个估计器配置，进行性能评估
    for estimator_conf in configuration["estimators"]:
        # 打印当前正在评估的估计器实例的信息
        print("Benchmarking", estimator_conf["instance"])
        # 使用当前估计器实例对训练集进行拟合
        estimator_conf["instance"].fit(X_train, y_train)
        # 手动进行垃圾回收，释放内存
        gc.collect()
        # 获取当前估计器实例的预测时间信息
        a, b = benchmark_estimator(estimator_conf["instance"], X_test)
        # 将结果存储在 stats 字典中，键为估计器名称，值包含 "atomic" 和 "bulk" 两种类型的预测时间
        stats[estimator_conf["name"]] = {"atomic": a, "bulk": b}

    # 获取所有估计器的名称列表
    cls_names = [
        estimator_conf["name"] for estimator_conf in configuration["estimators"]
    ]
    # 计算并转换成微秒的原子预测时间列表
    runtimes = [1e6 * stats[clf_name]["atomic"] for clf_name in cls_names]
    # 绘制原子预测时间的箱线图
    boxplot_runtimes(runtimes, "atomic", configuration)
    # 根据给定的分类器名称列表 cls_names，使用 stats 字典中每个分类器名对应的 "bulk" 字段计算运行时间（以微秒为单位）
    runtimes = [1e6 * stats[clf_name]["bulk"] for clf_name in cls_names]
    
    # 调用 boxplot_runtimes 函数，将运行时间数据（runtimes）、标签字符串和配置字典中的测试数量配置为参数传递
    boxplot_runtimes(runtimes, "bulk (%d)" % configuration["n_test"], configuration)
# 评估特征数量对预测时间影响的函数
def n_feature_influence(estimators, n_train, n_test, n_features, percentile):
    """
    Estimate influence of the number of features on prediction time.

    Parameters
    ----------
    estimators : dict of (name (str), estimator) to benchmark
        需要进行基准测试的估算器字典，键为名称，值为估算器对象
    n_train : nber of training instances (int)
        训练实例的数量
    n_test : nber of testing instances (int)
        测试实例的数量
    n_features : list of feature-space dimensionality to test (int)
        需要测试的特征空间维度列表
    percentile : percentile at which to measure the speed (int [0-100])
        用于测量速度的百分位数

    Returns:
    --------
    percentiles : dict(estimator_name,
                       dict(n_features, percentile_perf_in_us))
        百分位数字典，包含每个估算器名称对应的字典，该字典包含特征数量到百分位性能（微秒）的映射

    """
    percentiles = defaultdict(defaultdict)
    for n in n_features:
        print("benchmarking with %d features" % n)
        X_train, y_train, X_test, y_test = generate_dataset(n_train, n_test, n)
        for cls_name, estimator in estimators.items():
            estimator.fit(X_train, y_train)
            gc.collect()  # 执行垃圾收集
            runtimes = bulk_benchmark_estimator(estimator, X_test, 30, False)
            percentiles[cls_name][n] = 1e6 * np.percentile(runtimes, percentile)  # 计算百分位性能并转换为微秒
    return percentiles


# 绘制特征数量对预测时间影响的图表
def plot_n_features_influence(percentiles, percentile):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["r", "g", "b"]
    for i, cls_name in enumerate(percentiles.keys()):
        x = np.array(sorted(percentiles[cls_name].keys()))
        y = np.array([percentiles[cls_name][n] for n in x])
        plt.plot(
            x,
            y,
            color=colors[i],
        )
    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title("Evolution of Prediction Time with #Features")
    ax1.set_xlabel("#Features")
    ax1.set_ylabel("Prediction Time at %d%%-ile (us)" % percentile)
    plt.show()


# 基准测试不同估算器的吞吐量
def benchmark_throughputs(configuration, duration_secs=0.1):
    """benchmark throughput for different estimators."""
    X_train, y_train, X_test, y_test = generate_dataset(
        configuration["n_train"], configuration["n_test"], configuration["n_features"]
    )
    throughputs = dict()
    for estimator_config in configuration["estimators"]:
        estimator_config["instance"].fit(X_train, y_train)
        start_time = time.time()
        n_predictions = 0
        while (time.time() - start_time) < duration_secs:
            estimator_config["instance"].predict(X_test[[0]])
            n_predictions += 1
        throughputs[estimator_config["name"]] = n_predictions / duration_secs
    return throughputs


# 绘制各估算器的吞吐量基准测试结果图表
def plot_benchmark_throughput(throughputs, configuration):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["r", "g", "b"]
    cls_infos = [
        "%s\n(%d %s)"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
        )
        for estimator_conf in configuration["estimators"]
    ]
    # 从配置中获取每个评估器的吞吐量值列表
    cls_values = [
        throughputs[estimator_conf["name"]]
        for estimator_conf in configuration["estimators"]
    ]
    # 绘制柱状图，显示各评估器的吞吐量
    plt.bar(range(len(throughputs)), cls_values, width=0.5, color=colors)
    # 设置 x 轴刻度位置为等距分布，使每个柱状图对应一个评估器
    ax.set_xticks(np.linspace(0.25, len(throughputs) - 0.75, len(throughputs)))
    # 设置 x 轴刻度标签为评估器的信息，字体大小为 10
    ax.set_xticklabels(cls_infos, fontsize=10)
    # 计算 y 轴最大值，用于设置图表的纵坐标范围
    ymax = max(cls_values) * 1.2
    # 设置 y 轴范围从 0 到 ymax
    ax.set_ylim((0, ymax))
    # 设置 y 轴标签为 "Throughput (predictions/sec)"
    ax.set_ylabel("Throughput (predictions/sec)")
    # 设置图表标题，显示评估器吞吐量和特征数
    ax.set_title(
        "Prediction Throughput for different estimators (%d features)"
        % configuration["n_features"]
    )
    # 显示整个图表
    plt.show()
# %%
# 对各种回归器进行批量/原子预测速度的基准测试
# -------------------------------------------------------------
# 定义配置字典，包括训练样本数、测试样本数、特征数及回归器列表
configuration = {
    "n_train": int(1e3),
    "n_test": int(1e2),
    "n_features": int(1e2),
    "estimators": [
        {
            "name": "Linear Model",
            "instance": SGDRegressor(
                penalty="elasticnet", alpha=0.01, l1_ratio=0.25, tol=1e-4
            ),
            "complexity_label": "non-zero coefficients",
            "complexity_computer": lambda clf: np.count_nonzero(clf.coef_),
        },
        {
            "name": "RandomForest",
            "instance": RandomForestRegressor(),
            "complexity_label": "estimators",
            "complexity_computer": lambda clf: clf.n_estimators,
        },
        {
            "name": "SVR",
            "instance": SVR(kernel="rbf"),
            "complexity_label": "support vectors",
            "complexity_computer": lambda clf: len(clf.support_vectors_),
        },
    ],
}
# 调用基准测试函数，传入配置字典
benchmark(configuration)

# %%
# 基准测试特征数对预测速度的影响
# --------------------------------------------------
# 设定特征影响的百分位数
percentile = 90
# 计算不同特征数的影响百分位数
percentiles = n_feature_influence(
    {"ridge": Ridge()},
    configuration["n_train"],
    configuration["n_test"],
    [100, 250, 500],
    percentile,
)
# 绘制特征数影响图表
plot_n_features_influence(percentiles, percentile)

# %%
# 基准测试吞吐量
# --------------------
# 计算配置字典中各回归器的吞吐量
throughputs = benchmark_throughputs(configuration)
# 绘制吞吐量基准测试图表
plot_benchmark_throughput(throughputs, configuration)
```