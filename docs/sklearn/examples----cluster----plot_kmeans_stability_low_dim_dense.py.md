# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_kmeans_stability_low_dim_dense.py`

```
"""
============================================================
Empirical evaluation of the impact of k-means initialization
============================================================

Evaluate the ability of k-means initializations strategies to make
the algorithm convergence robust, as measured by the relative standard
deviation of the inertia of the clustering (i.e. the sum of squared
distances to the nearest cluster center).

The first plot shows the best inertia reached for each combination
of the model (``KMeans`` or ``MiniBatchKMeans``), and the init method
(``init="random"`` or ``init="k-means++"``) for increasing values of the
``n_init`` parameter that controls the number of initializations.

The second plot demonstrates one single run of the ``MiniBatchKMeans``
estimator using a ``init="random"`` and ``n_init=1``. This run leads to
a bad convergence (local optimum), with estimated centers stuck
between ground truth clusters.

The dataset used for evaluation is a 2D grid of isotropic Gaussian
clusters widely spaced.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import check_random_state, shuffle

# Initialize a random state for reproducibility
random_state = np.random.RandomState(0)

# Number of runs (with randomly generated dataset) for each strategy to estimate standard deviation
n_runs = 5

# Array specifying the range of n_init values to evaluate for initialization robustness
n_init_range = np.array([1, 5, 10, 15, 20])

# Parameters for generating datasets
n_samples_per_center = 100
grid_size = 3
scale = 0.1
n_clusters = grid_size ** 2


def make_data(random_state, n_samples_per_center, grid_size, scale):
    """
    Generate isotropic Gaussian clusters in a 2D grid.
    
    Parameters:
    - random_state: RandomState instance or None, for reproducibility
    - n_samples_per_center: Number of samples per cluster center
    - grid_size: Size of the grid (grid_size x grid_size clusters)
    - scale: Standard deviation of the Gaussian noise added to clusters
    
    Returns:
    - X: Generated samples
    - y: Labels indicating cluster membership
    """
    random_state = check_random_state(random_state)
    centers = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    noise = random_state.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1])
    )

    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center for i in range(n_clusters_true)])
    return shuffle(X, y, random_state=random_state)


# Part 1: Quantitative evaluation of various init methods

plt.figure()
plots = []
legends = []

# List of cases to evaluate, each specifying a clustering model, init method, and plotting format
cases = [
    (KMeans, "k-means++", {}, "^-"),
    (KMeans, "random", {}, "o-"),
    (MiniBatchKMeans, "k-means++", {"max_no_improvement": 3}, "x-"),
    (MiniBatchKMeans, "random", {"max_no_improvement": 3, "init_size": 500}, "d-"),
]

for factory, init, params, format in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(n_init_range), n_runs))
    # 循环执行多次聚类运行，每次运行使用不同的 run_id
    for run_id in range(n_runs):
        # 使用 make_data 函数生成数据集 X 和对应的标签 y
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        
        # 对于 n_init_range 中的每个元素 n_init，依次执行以下操作
        for i, n_init in enumerate(n_init_range):
            # 使用 factory 函数创建 KMeans 聚类器 km 对象
            km = factory(
                n_clusters=n_clusters,
                init=init,
                random_state=run_id,
                n_init=n_init,
                **params,
            ).fit(X)
            
            # 将每次聚类的 inertia（总距离的平方和）记录在 inertia 数组中的第 i 行、第 run_id 列
            inertia[i, run_id] = km.inertia_
    
    # 使用 matplotlib 绘制误差条图，显示每个 n_init 对应的平均 inertia 和标准差
    p = plt.errorbar(
        n_init_range, inertia.mean(axis=1), inertia.std(axis=1), fmt=format
    )
    
    # 将误差条图对象的第一个元素添加到 plots 列表中
    plots.append(p[0])
    
    # 将当前聚类工厂函数 factory 的名称和初始值 init 组合成字符串，添加到 legends 列表中
    legends.append("%s with %s init" % (factory.__name__, init))
# 设置 x 轴标签为 "n_init"
plt.xlabel("n_init")
# 设置 y 轴标签为 "inertia"
plt.ylabel("inertia")
# 添加图例，使用之前定义的 plots 和 legends 变量
plt.legend(plots, legends)
# 设置图表标题，展示 k-means 不同初始化方式在多次运行中的平均惯性
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)

# Part 2: Qualitative visual inspection of the convergence

# 生成用于可视化的数据集 X 和对应的标签 y
X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
# 使用 MiniBatchKMeans 进行聚类，指定参数包括簇数、初始化方式、初始化次数和随机种子
km = MiniBatchKMeans(
    n_clusters=n_clusters, init="random", n_init=1, random_state=random_state
).fit(X)

# 创建新的图表
plt.figure()
# 遍历每个簇
for k in range(n_clusters):
    # 获取属于当前簇的数据点索引
    my_members = km.labels_ == k
    # 根据簇的索引 k，从颜色映射 cm.nipy_spectral 中获取颜色
    color = cm.nipy_spectral(float(k) / n_clusters, 1)
    # 绘制属于当前簇的数据点
    plt.plot(X[my_members, 0], X[my_members, 1], ".", c=color)
    # 获取当前簇的簇中心点
    cluster_center = km.cluster_centers_[k]
    # 绘制簇中心点
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=color,
        markeredgecolor="k",
        markersize=6,
    )
    # 设置子图标题，展示单次随机初始化下 MiniBatchKMeans 的聚类结果
    plt.title(
        "Example cluster allocation with a single random init\nwith MiniBatchKMeans"
    )

# 展示图表
plt.show()
```