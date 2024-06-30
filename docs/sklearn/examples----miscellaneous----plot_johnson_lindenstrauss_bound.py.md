# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_johnson_lindenstrauss_bound.py`

```
# 导入必要的库和模块
import sys
from time import time

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from sklearn.datasets import fetch_20newsgroups_vectorized, load_digits  # 导入 sklearn 的数据集加载模块
from sklearn.metrics.pairwise import euclidean_distances  # 导入 sklearn 中的欧氏距离计算模块
from sklearn.random_projection import (  # 导入 sklearn 中的随机投影模块及其函数
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

# %%
# 理论上的边界
# ==================
# 随机投影引入的失真由以下事实来断言：`p` 定义了一个 eps-embedding，
# 在这个过程中控制了成对距离的失真。

# 定义并引用 Johnson-Lindenstrauss 引理，详细说明了随机投影的作用
# 参考链接：https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

# 第一个图表展示了随着样本数 `n_samples` 的增加，
# 为保证 `eps`-embedding 所需的最小维数 `n_components` 随着对数增长。

# 可接受失真的范围
eps_range = np.linspace(0.1, 0.99, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

# 样本数的范围（观察值）用于嵌入
n_samples_range = np.logspace(1, 9, 9)

plt.figure()
for eps, color in zip(eps_range, colors):
    # 计算保证 eps-embedding 所需的最小维数 `n_components`
    min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
    plt.loglog(n_samples_range, min_n_components, color=color)

plt.legend([f"eps = {eps:0.1f}" for eps in eps_range], loc="lower right")
plt.xlabel("观察数用于 eps-embedding")
plt.ylabel("最小维数")
plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")
plt.show()


# %%
# 第二个图表展示了增加可接受失真 `eps` 可以显著减少给定样本数 `n_samples`
# 的最小维数 `n_components`。

# 可接受失真的范围
eps_range = np.linspace(0.01, 0.99, 100)

# 样本数的范围（观察值）用于嵌入
n_samples_range = np.logspace(2, 6, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

plt.figure()
for n_samples, color in zip(n_samples_range, colors):
    # 计算 Johnson-Lindenstrauss 维数下限，基于给定的样本数量和ε范围
    min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
    # 使用半对数坐标绘制图形，横坐标为ε范围，纵坐标为对应的最小维数下限，指定颜色
    plt.semilogy(eps_range, min_n_components, color=color)
plt.legend([f"n_samples = {n}" for n in n_samples_range], loc="upper right")
# 创建图例，显示不同样本数量的标签，位于图像的右上角

plt.xlabel("Distortion eps")
# 设置 x 轴标签为 "Distortion eps"

plt.ylabel("Minimum number of dimensions")
# 设置 y 轴标签为 "Minimum number of dimensions"

plt.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")
# 设置图像标题为 "Johnson-Lindenstrauss bounds:\nn_components vs eps"

plt.show()
# 显示绘制的图像

# %%
# Empirical validation
# ====================
#
# We validate the above bounds on the 20 newsgroups text document
# (TF-IDF word frequencies) dataset or on the digits dataset:
#
# - for the 20 newsgroups dataset some 300 documents with 100k
#   features in total are projected using a sparse random matrix to smaller
#   euclidean spaces with various values for the target number of dimensions
#   ``n_components``.
#
# - for the digits dataset, some 8x8 gray level pixels data for 300
#   handwritten digits pictures are randomly projected to spaces for various
#   larger number of dimensions ``n_components``.
#
# The default dataset is the 20 newsgroups dataset. To run the example on the
# digits dataset, pass the ``--use-digits-dataset`` command line argument to
# this script.

if "--use-digits-dataset" in sys.argv:
    data = load_digits().data[:300]
else:
    data = fetch_20newsgroups_vectorized().data[:300]

# %%
# For each value of ``n_components``, we plot:
#
# - 2D distribution of sample pairs with pairwise distances in original
#   and projected spaces as x- and y-axis respectively.
#
# - 1D histogram of the ratio of those distances (projected / original).

n_samples, n_features = data.shape
print(
    f"Embedding {n_samples} samples with dim {n_features} using various "
    "random projections"
)

n_components_range = np.array([300, 1_000, 10_000])
dists = euclidean_distances(data, squared=True).ravel()

# select only non-identical samples pairs
nonzero = dists != 0
dists = dists[nonzero]

for n_components in n_components_range:
    t0 = time()
    rp = SparseRandomProjection(n_components=n_components)
    projected_data = rp.fit_transform(data)
    print(
        f"Projected {n_samples} samples from {n_features} to {n_components} in "
        f"{time() - t0:0.3f}s"
    )
    if hasattr(rp, "components_"):
        n_bytes = rp.components_.data.nbytes
        n_bytes += rp.components_.indices.nbytes
        print(f"Random matrix with size: {n_bytes / 1e6:0.3f} MB")

    projected_dists = euclidean_distances(projected_data, squared=True).ravel()[nonzero]

    plt.figure()
    min_dist = min(projected_dists.min(), dists.min())
    max_dist = max(projected_dists.max(), dists.max())
    plt.hexbin(
        dists,
        projected_dists,
        gridsize=100,
        cmap=plt.cm.PuBu,
        extent=[min_dist, max_dist, min_dist, max_dist],
    )
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for n_components=%d" % n_components)
    cb = plt.colorbar()
    cb.set_label("Sample pairs counts")

    rates = projected_dists / dists
    # 打印平均距离比率和标准差
    print(f"Mean distances rate: {np.mean(rates):.2f} ({np.std(rates):.2f})")
    
    # 创建一个新的图形窗口
    plt.figure()
    
    # 绘制距离比率的直方图，设置50个柱子，范围在0.0到2.0之间，边缘颜色为黑色，进行归一化
    plt.hist(rates, bins=50, range=(0.0, 2.0), edgecolor="k", density=True)
    
    # 设置x轴标签为“Squared distances rate: projected / original”
    plt.xlabel("Squared distances rate: projected / original")
    
    # 设置y轴标签为“Distribution of samples pairs”
    plt.ylabel("Distribution of samples pairs")
    
    # 设置标题，使用给定的n_components作为参数
    plt.title("Histogram of pairwise distance rates for n_components=%d" % n_components)
    
    # TODO: 计算eps的期望值，并将其作为垂直线/区域添加到前述的图中
    # 作为下一步工作，计算eps的期望值，并将其作为垂直线或区域添加到前面的图中
# 展示当前的图形窗口，显示所有已创建的图形
plt.show()

# %%
# 我们可以看到，对于低值的 ``n_components``，分布是宽泛的，
# 有许多扭曲的对和一个偏斜的分布（由于距离始终为正数，左侧有零比率的硬限制），
# 而对于较大的 `n_components`，扭曲被控制住了，
# 而且随机投影能很好地保留距离。
#
# 备注
# =======
#
# 根据约翰逊-林登斯特劳斯引理（JL引理），投影 300 个样本如果要保持不太大的扭曲，
# 将至少需要几千个维度，这与原始数据集的特征数量无关。
#
# 因此，在仅有 64 个输入空间特征的数字数据集上使用随机投影是没有意义的：
# 这种情况下无法实现降维。
#
# 而在二十个新闻组数据集上，可以将维度从 56,436 降低到 10,000，
# 同时合理地保持成对距离。
```