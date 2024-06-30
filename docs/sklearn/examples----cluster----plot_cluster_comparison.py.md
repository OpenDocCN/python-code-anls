# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_cluster_comparison.py`

```
"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.

"""

import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500  # 设置每个数据集的样本数量为500
seed = 30  # 设置随机数种子为30，确保结果的可重复性

# 生成包含噪声圆的数据集
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
# 生成包含噪声月亮形状的数据集
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
# 生成高斯分布的数据集
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
# 生成无结构数据集
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
# 对数据进行线性变换
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
# 生成具有不同方差的高斯分布数据集
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,  # 高斯混合数据集的参数：阻尼系数
            "preference": -220,  # 高斯混合数据集的参数：首选项
            "n_clusters": 2,  # 高斯混合数据集的参数：聚类数量
            "min_samples": 7,  # 高斯混合数据集的参数：最小样本数
            "xi": 0.1,  # 高斯混合数据集的参数：XI
        },
    ),
    (
        varied,
        {
            "eps": 0.18,  # 变化方差数据集的参数：EPS
            "n_neighbors": 2,  # 变化方差数据集的参数：近邻数
            "min_samples": 7,  # 变化方差数据集的参数：最小样本数
            "xi": 0.01,  # 变化方差数据集的参数：XI
            "min_cluster_size": 0.2,  # 变化方差数据集的参数：最小聚类大小
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,  # 各向异性数据集的参数：EPS
            "n_neighbors": 2,  # 各向异性数据集的参数：近邻数
            "min_samples": 7,  # 各向异性数据集的参数：最小样本数
            "xi": 0.1,  # 各向异性数据集的参数：XI
            "min_cluster_size": 0.2,  # 各向异性数据集的参数：最小聚类大小
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),  # Blob数据集的参数：最小样本数、XI、最小聚类大小
    (no_structure, {}),  # 无结构数据集，无额外参数
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # 创建 MeanShift 聚类对象，使用指定的带宽和二进制种子初始化
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # 创建 MiniBatchKMeans 聚类对象，使用指定的簇数和随机种子初始化
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
    )
    # 创建 AgglomerativeClustering 聚类对象，使用指定的簇数、连接方式和连接矩阵初始化
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    # 创建 SpectralClustering 聚类对象，使用指定的簇数、特征解算器和相似度初始化
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=params["random_state"],
    )
    # 创建 DBSCAN 聚类对象，使用指定的距离阈值初始化
    dbscan = cluster.DBSCAN(eps=params["eps"])
    # 创建 HDBSCAN 聚类对象，使用指定的最小样本数、最小簇大小和是否允许单簇初始化
    hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
    )
    # 创建 OPTICS 聚类对象，使用指定的最小样本数、xi 和最小簇大小初始化
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    # 创建 AffinityPropagation 聚类对象，使用指定的阻尼、首选项和随机种子初始化
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"],
        preference=params["preference"],
        random_state=params["random_state"],
    )
    # 创建 AgglomerativeClustering 聚类对象，使用指定的连接方式、距离度量和簇数初始化
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        metric="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    # 创建 Birch 聚类对象，使用指定的簇数初始化
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    # 创建 GaussianMixture 聚类对象，使用指定的簇数、协方差类型和随机种子初始化
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"],
        covariance_type="full",
        random_state=params["random_state"],
    )

    clustering_algorithms = (
        ("MiniBatch\nKMeans", two_means),
        ("Affinity\nPropagation", affinity_propagation),
        ("MeanShift", ms),
        ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("Agglomerative\nClustering", average_linkage),
        ("DBSCAN", dbscan),
        ("HDBSCAN", hdbscan),
        ("OPTICS", optics),
        ("BIRCH", birch),
        ("Gaussian\nMixture", gmm),
    )
    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # 捕获与 kneighbors_graph 相关的警告
        with warnings.catch_warnings():
            # 忽略特定的 UserWarning，避免因连接矩阵的连接组件数目大于1而提前终止树的构建
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            # 忽略图不完全连接的警告，这可能导致谱嵌入效果不如预期
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            # 使用算法对象对数据 X 进行拟合
            algorithm.fit(X)

        t1 = time.time()
        # 如果算法对象具有 labels_ 属性，则使用它作为聚类结果
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            # 否则，使用 predict 方法得到聚类结果
            y_pred = algorithm.predict(X)

        # 在绘图中创建子图，放置于给定数据集和聚类算法的位置
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            # 在第一行的每个子图标题上显示聚类算法的名称
            plt.title(name, size=18)

        # 生成聚类结果的颜色列表，根据 y_pred 的不同值分配不同颜色
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # 将黑色添加到颜色列表中，用于标记异常值（如果存在）
        colors = np.append(colors, ["#000000"])
        # 绘制散点图，根据聚类结果着色
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # 设置坐标轴的范围
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        # 清除 x 和 y 轴的刻度
        plt.xticks(())
        plt.yticks(())
        # 在子图的右下角显示运行时间
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        # 更新子图编号
        plot_num += 1
# 显示 matplotlib 中当前的图形
plt.show()
```