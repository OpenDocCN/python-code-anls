# `D:\src\scipysrc\scikit-learn\examples\neighbors\approximate_nearest_neighbors.py`

```
"""
=====================================
Approximate nearest neighbors in TSNE
=====================================

This example presents how to chain KNeighborsTransformer and TSNE in a pipeline.
It also shows how to wrap the packages `nmslib` and `pynndescent` to replace
KNeighborsTransformer and perform approximate nearest neighbors. These packages
can be installed with `pip install nmslib pynndescent`.

Note: In KNeighborsTransformer we use the definition which includes each
training point as its own neighbor in the count of `n_neighbors`, and for
compatibility reasons, one extra neighbor is computed when `mode == 'distance'`.
Please note that we do the same in the proposed `nmslib` wrapper.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# First we try to import the packages and warn the user in case they are
# missing.
import sys

try:
    import nmslib
except ImportError:
    # 提示用户缺少必需的 'nmslib' 包
    print("The package 'nmslib' is required to run this example.")
    sys.exit()

try:
    from pynndescent import PyNNDescentTransformer
except ImportError:
    # 提示用户缺少必需的 'pynndescent' 包
    print("The package 'pynndescent' is required to run this example.")
    sys.exit()

# %%
# We define a wrapper class for implementing the scikit-learn API to the
# `nmslib`, as well as a loading function.
import joblib
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric="euclidean", method="sw-graph", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        # 记录训练样本数量
        self.n_samples_fit_ = X.shape[0]

        # 根据选择的距离度量初始化 nmslib 的空间
        space = {
            "euclidean": "l2",
            "cosine": "cosinesimil",
            "l1": "l1",
            "l2": "l2",
        }[self.metric]

        # 初始化 nmslib 对象并添加数据点批次
        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(X.copy())
        self.nmslib_.createIndex()
        return self
    # 定义一个方法 `transform`，用于将输入数据集 X 进行转换
    def transform(self, X):
        # 获取输入数据集的样本数量
        n_samples_transform = X.shape[0]

        # 由于每个样本被视为自身的邻居，因此需要计算额外的一个邻居
        n_neighbors = self.n_neighbors + 1

        # 如果设定的并行工作线程数 `n_jobs` 小于 0
        if self.n_jobs < 0:
            # 与 joblib 中对于负值 `n_jobs` 的处理相同：
            # 特别是，`n_jobs == -1` 表示 "尽可能多的线程与 CPU 核心数匹配"。
            num_threads = joblib.cpu_count() + self.n_jobs + 1
        else:
            # 否则，使用设定的并行工作线程数 `n_jobs`
            num_threads = self.n_jobs

        # 使用 nmslib 中的 knnQueryBatch 方法计算 X 的 k 近邻查询结果
        results = self.nmslib_.knnQueryBatch(
            X.copy(), k=n_neighbors, num_threads=num_threads
        )

        # 将查询结果分解为索引和距离两个部分
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        # 创建稀疏矩阵的指针数组 `indptr`
        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)

        # 使用 CSR 格式创建稀疏矩阵 `kneighbors_graph`
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        # 返回最终的 k 近邻图矩阵 `kneighbors_graph`
        return kneighbors_graph
def load_mnist(n_samples):
    """Load MNIST, shuffle the data, and return only n_samples."""
    # 使用 fetch_openml 函数加载 MNIST 数据集，将 as_frame 参数设为 False
    mnist = fetch_openml("mnist_784", as_frame=False)
    # 对数据和标签进行洗牌，使用 random_state=2 进行随机种子控制
    X, y = shuffle(mnist.data, mnist.target, random_state=2)
    # 返回前 n_samples 个样本的特征数据 X，并将像素值缩放到 [0, 1] 区间
    return X[:n_samples] / 255, y[:n_samples]

# %%
# 对不同的近邻转换器进行性能基准测试（Benchmark）
import time

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline

datasets = [
    ("MNIST_10000", load_mnist(n_samples=10_000)),
    ("MNIST_20000", load_mnist(n_samples=20_000)),
]

n_iter = 500
perplexity = 30
metric = "euclidean"
# TSNE 需要一定数量的近邻数，这取决于 perplexity 参数。
# 我们加一是因为每个样本本身也作为自己的一个近邻。
n_neighbors = int(3.0 * perplexity + 1) + 1

tsne_params = dict(
    init="random",  # 对于稀疏矩阵，不支持 pca 初始化
    perplexity=perplexity,
    method="barnes_hut",
    random_state=42,
    n_iter=n_iter,
    learning_rate="auto",
)

transformers = [
    (
        "KNeighborsTransformer",
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance", metric=metric),
    ),
    (
        "NMSlibTransformer",
        NMSlibTransformer(n_neighbors=n_neighbors, metric=metric),
    ),
    (
        "PyNNDescentTransformer",
        PyNNDescentTransformer(
            n_neighbors=n_neighbors, metric=metric, parallel_batch_queries=True
        ),
    ),
]

for dataset_name, (X, y) in datasets:
    msg = f"Benchmarking on {dataset_name}:"
    print(f"\n{msg}\n" + str("-" * len(msg)))

    for transformer_name, transformer in transformers:
        longest = np.max([len(name) for name, model in transformers])
        start = time.time()
        transformer.fit(X)
        fit_duration = time.time() - start
        print(f"{transformer_name:<{longest}} {fit_duration:.3f} sec (fit)")
        start = time.time()
        Xt = transformer.transform(X)
        transform_duration = time.time() - start
        print(f"{transformer_name:<{longest}} {transform_duration:.3f} sec (transform)")
        if transformer_name == "PyNNDescentTransformer":
            start = time.time()
            Xt = transformer.transform(X)
            transform_duration = time.time() - start
            print(
                f"{transformer_name:<{longest}} {transform_duration:.3f} sec"
                " (transform)"
            )

# %%
# 示例输出：
#
#     Benchmarking on MNIST_10000:
#     ----------------------------
#     KNeighborsTransformer  0.007 sec (fit)
#     KNeighborsTransformer  1.139 sec (transform)
#     NMSlibTransformer      0.208 sec (fit)
#     NMSlibTransformer      0.315 sec (transform)
#     PyNNDescentTransformer 4.823 sec (fit)
#     PyNNDescentTransformer 4.884 sec (transform)
#     PyNNDescentTransformer 0.744 sec (transform)
#
#     Benchmarking on MNIST_20000:
#     ----------------------------
#     KNeighborsTransformer  0.011 sec (fit)
# %%
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
from matplotlib.ticker import NullFormatter  # 导入 NullFormatter 用于设置坐标轴格式

# 定义不同的转换器列表，每个元素是一个元组，包含转换器名称和对应的转换器对象
transformers = [
    ("TSNE with internal NearestNeighbors", TSNE(metric=metric, **tsne_params)),  # 使用内置的最近邻算法进行 t-SNE 转换
    (
        "TSNE with KNeighborsTransformer",  # 使用 KNeighborsTransformer 进行 t-SNE 转换
        make_pipeline(
            KNeighborsTransformer(
                n_neighbors=n_neighbors, mode="distance", metric=metric
            ),  # 创建 KNeighborsTransformer 对象，配置参数为邻居数量、模式和距离度量
            TSNE(metric="precomputed", **tsne_params),  # 使用预计算的距离度量进行 t-SNE 转换
        ),
    ),
    (
        "TSNE with NMSlibTransformer",  # 使用 NMSlibTransformer 进行 t-SNE 转换
        make_pipeline(
            NMSlibTransformer(n_neighbors=n_neighbors, metric=metric),  # 创建 NMSlibTransformer 对象，配置参数为邻居数量和距离度量
            TSNE(metric="precomputed", **tsne_params),  # 使用预计算的距离度量进行 t-SNE 转换
        ),
    ),
]

# 初始化绘图
nrows = len(datasets)  # 图表行数为数据集数量
ncols = np.sum([1 for name, model in transformers if "TSNE" in name])  # 图表列数为包含 TSNE 的转换器数量
fig, axes = plt.subplots(  # 创建子图，指定行数、列数、布局大小
    nrows=nrows, ncols=ncols, squeeze=False, figsize=(5 * ncols, 4 * nrows)
)
axes = axes.ravel()  # 将二维子图数组展平成一维数组
i_ax = 0  # 初始化子图索引

# 遍历数据集
for dataset_name, (X, y) in datasets:
    msg = f"Benchmarking on {dataset_name}:"  # 打印当前数据集的提示信息
    print(f"\n{msg}\n" + str("-" * len(msg)))  # 打印带有分隔线的提示信息

    # 遍历转换器列表
    for transformer_name, transformer in transformers:
        longest = np.max([len(name) for name, model in transformers])  # 计算转换器名称的最大长度
        start = time.time()  # 记录开始时间
        Xt = transformer.fit_transform(X)  # 对数据集 X 进行拟合和转换
        transform_duration = time.time() - start  # 计算拟合转换耗时
        print(
            f"{transformer_name:<{longest}} {transform_duration:.3f} sec"
            " (fit_transform)"  # 打印转换器名称和拟合转换耗时信息
        )

        # 绘制 t-SNE 嵌入图像，颜色映射使用 viridis
        axes[i_ax].set_title(transformer_name + "\non " + dataset_name)  # 设置子图标题
        axes[i_ax].scatter(
            Xt[:, 0],
            Xt[:, 1],
            c=y.astype(np.int32),
            alpha=0.2,
            cmap=plt.cm.viridis,
        )  # 绘制散点图
        axes[i_ax].xaxis.set_major_formatter(NullFormatter())  # 设置 x 轴主要刻度格式为空
        axes[i_ax].yaxis.set_major_formatter(NullFormatter())  # 设置 y 轴主要刻度格式为空
        axes[i_ax].axis("tight")  # 调整子图边界
        i_ax += 1  # 更新子图索引

fig.tight_layout()  # 调整布局使子图不重叠
plt.show()  # 显示图像

# %%
# 示例输出::
#
#     Benchmarking on MNIST_10000:
#     ----------------------------
#     TSNE with internal NearestNeighbors 24.828 sec (fit_transform)
# TSNE with KNeighborsTransformer     20.111 sec (fit_transform)
# TSNE with NMSlibTransformer         21.757 sec (fit_transform)
#
# Benchmarking on MNIST_20000:
# ----------------------------
# TSNE with internal NearestNeighbors 51.955 sec (fit_transform)
# TSNE with KNeighborsTransformer     50.994 sec (fit_transform)
# TSNE with NMSlibTransformer         43.536 sec (fit_transform)
#
# 我们可以观察到，默认的 :class:`~sklearn.manifold.TSNE` 估计器使用其内部的
# :class:`~sklearn.neighbors.NearestNeighbors` 实现，在性能上大致等同于使用
# :class:`~sklearn.manifold.TSNE` 和 :class:`~sklearn.neighbors.KNeighborsTransformer`
# 组合的管道。这是因为这两个管道在内部都依赖于执行精确邻居搜索的
# :class:`~sklearn.neighbors.NearestNeighbors` 实现。近似搜索 `NMSlibTransformer`
# 在较小的数据集上已经比精确搜索稍快，但预计在样本数量较大的数据集上，这种速度
# 差异会更显著。
#
# 但需要注意，并非所有近似搜索方法都能保证提高默认的精确搜索方法的速度：事实上，
# 自 scikit-learn 1.1 以来，精确搜索实现已经显著改进。此外，暴力精确搜索方法在
# `fit` 时不需要构建索引。因此，在 `transform` 阶段通过近似搜索获得整体性能提升，
# 需要近似搜索在 `fit` 时建立索引所花费的额外时间小于 `transform` 阶段的性能收益。
#
# 最后，TSNE 算法本身也具有较高的计算强度，与最近邻搜索步骤的速度提升无关。因此，
# 将最近邻搜索步骤加速5倍，并不意味着整体管道速度也会提升5倍。
```