# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_hdbscan.py`

```
# -*- coding: utf-8 -*-
"""
====================================
Demo of HDBSCAN clustering algorithm
====================================
.. currentmodule:: sklearn

In this demo we will take a look at :class:`cluster.HDBSCAN` from the
perspective of generalizing the :class:`cluster.DBSCAN` algorithm.
We'll compare both algorithms on specific datasets. Finally we'll evaluate
HDBSCAN's sensitivity to certain hyperparameters.

We first define a couple utility functions for convenience.
"""
# %%
# 导入必要的库
import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy

from sklearn.cluster import DBSCAN, HDBSCAN  # 从sklearn中导入DBSCAN和HDBSCAN聚类算法
from sklearn.datasets import make_blobs  # 从sklearn.datasets中导入make_blobs函数，用于生成聚类数据


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    """
    绘制数据点的散点图，用不同颜色和大小表示不同的聚类结果。

    Parameters:
    -----------
    X : ndarray
        输入数据集，每行是一个样本，每列是一个特征。
    labels : ndarray
        每个样本的聚类标签。
    probabilities : ndarray, optional
        每个样本属于其所在聚类的概率。
    parameters : dict, optional
        其他需要在图标题中显示的参数信息。
    ground_truth : bool, optional
        是否使用真实的聚类标签。
    ax : matplotlib.axes.Axes, optional
        绘图所使用的轴对象。

    Returns:
    --------
    None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))  # 创建一个新的绘图窗口

    labels = labels if labels is not None else np.ones(X.shape[0])  # 如果没有提供标签，默认所有点为一个类
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])  # 如果没有提供概率，默认所有点概率为1

    # 移除黑色，用于表示噪声点
    unique_labels = set(labels)  # 获取唯一的聚类标签
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]  # 生成一组颜色，用于表示不同的聚类标签

    # 根据点属于其聚类的概率确定点的大小
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}

    # 遍历每个聚类标签及其对应的颜色
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 用黑色表示噪声点
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]  # 获取属于当前聚类的样本索引
        for ci in class_index:
            # 绘制每个样本点
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",  # 噪声点用"x"，其他点用"o"
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],  # 设置点的大小
            )

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 计算聚类的数量
    preamble = "True" if ground_truth else "Estimated"  # 如果使用真实标签则显示"True"，否则显示"Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"  # 设置图标题，显示聚类数量信息
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())  # 将参数字典格式化为字符串
        title += f" | {parameters_str}"  # 在标题中添加参数信息
    ax.set_title(title)  # 设置图的标题
    plt.tight_layout()  # 调整图的布局，使其更加紧凑


# %%
# 生成示例数据
# --------------------
# HDBSCAN相对于DBSCAN的一个显著优势是其开箱即用的稳健性。
# 特别是在数据的异质混合物上表现显著。
# 类似于DBSCAN，它可以模拟任意形状和分布，但不像DBSCAN那样需要指定一个
# 需要任意和敏感的`eps`超参数。
#
# 例如，下面我们从三个双向和各向同性高斯分布的混合中生成一个数据集。
centers = [[1, 1], [-1, -1], [1.5, -1.5]]  # 设置高斯分布的中心点坐标
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=[0.4, 0.1, 0.75], random_state=0
)  # 使用make_blobs生成具有不同标准差的高斯混合数据集
plot(X, labels=labels_true, ground_truth=True)  # 调用plot函数绘制数据集的真实聚类情况

# %%
# 尺度不变性
# -----------------
# 值得记住的是，虽然DBSCAN为`eps`提供了一个默认值
# parameter, it hardly has a proper default value and must be tuned for the
# specific dataset at use.
#
# As a simple demonstration, consider the clustering for a `eps` value tuned
# for one dataset, and clustering obtained with the same value but applied to
# rescaled versions of the dataset.
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
# 使用 DBSCAN 算法进行聚类，设置 eps 参数为 0.3
dbs = DBSCAN(eps=0.3)
# 对于三种不同的数据尺度进行循环处理
for idx, scale in enumerate([1, 0.5, 3]):
    # 对数据进行缩放后，应用 DBSCAN 算法进行聚类
    dbs.fit(X * scale)
    # 绘制聚类结果的图像，同时传入参数 scale 和固定的 eps=0.3
    plot(X * scale, dbs.labels_, parameters={"scale": scale, "eps": 0.3}, ax=axes[idx])

# %%
# Indeed, in order to maintain the same results we would have to scale `eps` by
# the same factor.
fig, axis = plt.subplots(1, 1, figsize=(12, 5))
# 通过设置较大的 eps 值（0.9），对数据进行缩放处理（乘以3倍）后应用 DBSCAN 算法进行聚类
dbs = DBSCAN(eps=0.9).fit(3 * X)
# 绘制聚类结果的图像，同时传入参数 scale=3 和固定的 eps=0.9
plot(3 * X, dbs.labels_, parameters={"scale": 3, "eps": 0.9}, ax=axis)
# %%
# While standardizing data (e.g. using
# :class:`sklearn.preprocessing.StandardScaler`) helps mitigate this problem,
# great care must be taken to select the appropriate value for `eps`.
#
# HDBSCAN is much more robust in this sense: HDBSCAN can be seen as
# clustering over all possible values of `eps` and extracting the best
# clusters from all possible clusters (see :ref:`User Guide <HDBSCAN>`).
# One immediate advantage is that HDBSCAN is scale-invariant.
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
# 使用 HDBSCAN 算法进行聚类
hdb = HDBSCAN()
for idx, scale in enumerate([1, 0.5, 3]):
    # 对数据进行缩放后，应用 HDBSCAN 算法进行聚类
    hdb.fit(X * scale)
    # 绘制聚类结果的图像，同时传入参数 scale 和不同的其他参数
    plot(
        X * scale,
        hdb.labels_,
        hdb.probabilities_,
        ax=axes[idx],
        parameters={"scale": scale},
    )
# %%
# Multi-Scale Clustering
# ----------------------
# HDBSCAN is much more than scale invariant though -- it is capable of
# multi-scale clustering, which accounts for clusters with varying density.
# Traditional DBSCAN assumes that any potential clusters are homogeneous in
# density. HDBSCAN is free from such constraints. To demonstrate this we
# consider the following dataset
centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=[0.2, 0.35, 1.35, 1.35], random_state=0
)
plot(X, labels=labels_true, ground_truth=True)

# %%
# This dataset is more difficult for DBSCAN due to the varying densities and
# spatial separation:
#
# - If `eps` is too large then we risk falsely clustering the two dense
#   clusters as one since their mutual reachability will extend
#   clusters.
# - If `eps` is too small, then we risk fragmenting the sparser clusters
#   into many false clusters.
#
# Not to mention this requires manually tuning choices of `eps` until we
# find a tradeoff that we are comfortable with.
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
params = {"eps": 0.7}
# 使用给定的 eps 值对数据进行聚类，绘制聚类结果的图像
dbs = DBSCAN(**params).fit(X)
plot(X, dbs.labels_, parameters=params, ax=axes[0])
params = {"eps": 0.3}
# 使用不同的 eps 值对数据进行聚类，绘制聚类结果的图像
dbs = DBSCAN(**params).fit(X)
plot(X, dbs.labels_, parameters=params, ax=axes[1])

# %%
# To properly cluster the two dense clusters, we would need a smaller value of
# 使用 HDBSCAN 算法拟合数据集 X，不指定任何参数
hdb = HDBSCAN().fit(X)
# 绘制数据集 X 的散点图，标记每个点的聚类标签和聚类概率
plot(X, hdb.labels_, hdb.probabilities_)

# %%
# HDBSCAN 能够自适应数据集的多尺度结构，无需用户调整参数。
# 虽然任何足够复杂的数据集都可能需要调整，但本例表明，HDBSCAN 能够提供质量更好的聚类结果，
# 而这些结果通过 DBSCAN 是无法获得的。

# %%
# 超参数的鲁棒性
# -------------------------
# 最终，在任何实际应用中调整参数将是一个重要步骤，因此让我们来看看 HDBSCAN 的一些最重要的超参数。
# 虽然 HDBSCAN 摆脱了 DBSCAN 的 `eps` 参数，但它仍然具有一些超参数，如 `min_cluster_size` 和 `min_samples`，
# 这些超参数调整了其关于密度的结果。我们将看到，由于这些参数的清晰含义，HDBSCAN 对各种实际示例相对鲁棒。

PARAM = ({"min_cluster_size": 5}, {"min_cluster_size": 3}, {"min_cluster_size": 25})
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for i, param in enumerate(PARAM):
    # 使用不同的参数对数据集 X 运行 HDBSCAN 算法
    hdb = HDBSCAN(**param).fit(X)
    labels = hdb.labels_

    # 绘制数据集 X 的散点图，标记每个点的聚类标签和聚类概率，并显示当前使用的参数
    plot(X, labels, hdb.probabilities_, param, ax=axes[i])

# %%
# `min_samples`
# ^^^^^^^^^^^^^
# `min_samples` 是一个点周围邻域内的样本数，包括该点本身，才能被认为是核心点。
# `min_samples` 默认为 `min_cluster_size`。
# 与 `min_cluster_size` 类似，较大的 `min_samples` 值可以增加模型对噪声的鲁棒性，但可能会忽略或丢弃
# 可能有效但较小的聚类。
# 在找到适当的 `min_cluster_size` 值后，最好调整 `min_samples` 值。

PARAM = (
    {"min_cluster_size": 20, "min_samples": 5},
    {"min_cluster_size": 20, "min_samples": 3},
    {"min_cluster_size": 20, "min_samples": 25},
)
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for i, param in enumerate(PARAM):
    # 使用不同的参数对数据集 X 运行 HDBSCAN 算法
    hdb = HDBSCAN(**param).fit(X)
    labels = hdb.labels_
    plot(X, labels, hdb.probabilities_, param, ax=axes[i])



    # 在给定的坐标轴上绘制图形
    plot(
        X,              # 输入数据集 X
        labels,         # 数据点的标签
        hdb.probabilities_,  # HDBSCAN 算法计算的每个点属于聚类的概率
        param,          # 绘图参数（可能是颜色、形状等）
        ax=axes[i]      # 使用指定的子图 axes[i] 进行绘制
    )
# %%
# `dbscan_clustering`
# ^^^^^^^^^^^^^^^^^^^
# 在 `fit` 过程中，`HDBSCAN` 构建了一个单链接树，该树编码了所有点在所有
# :class:`~cluster.DBSCAN` 的 `eps` 参数值下的聚类情况。
# 因此，我们可以高效地绘制和评估这些聚类，而无需完全重新计算诸如核心距离、
# 互可达性和最小生成树等中间值。我们只需指定要进行聚类的 `cut_distance`
# （相当于 `eps`）即可。

PARAM = (
    {"cut_distance": 0.1},
    {"cut_distance": 0.5},
    {"cut_distance": 1.0},
)
# 创建一个 HDBSCAN 的实例
hdb = HDBSCAN()
# 使用数据 X 来拟合模型
hdb.fit(X)
# 创建一个包含 len(PARAM) 行和 1 列的图形布局
fig, axes = plt.subplots(len(PARAM), 1, figsize=(10, 12))
# 对 PARAM 中的每个参数进行迭代
for i, param in enumerate(PARAM):
    # 使用指定的参数调用 dbscan_clustering 方法来获取标签
    labels = hdb.dbscan_clustering(**param)

    # 在当前轴上绘制数据 X 的散点图，使用 labels 进行着色，显示 hdb 的概率值，使用 param 进行参数设置
    plot(X, labels, hdb.probabilities_, param, ax=axes[i])
```