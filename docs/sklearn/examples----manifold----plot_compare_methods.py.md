# `D:\src\scipysrc\scikit-learn\examples\manifold\plot_compare_methods.py`

```
"""
=========================================
Comparison of Manifold Learning methods
=========================================

An illustration of dimensionality reduction on the S-curve dataset
with various manifold learning methods.

For a discussion and comparison of these algorithms, see the
:ref:`manifold module page <manifold>`

For a similar example, where the methods are applied to a
sphere dataset, see :ref:`sphx_glr_auto_examples_manifold_plot_manifold_sphere.py`

Note that the purpose of the MDS is to find a low-dimensional
representation of the data (here 2D) in which the distances respect well
the distances in the original high-dimensional space, unlike other
manifold-learning algorithms, it does not seeks an isotropic
representation of the data in the low-dimensional space.

"""

# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

# %%
# Dataset preparation
# -------------------
#
# We start by generating the S-curve dataset.

import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker

from sklearn import datasets, manifold

n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

# %%
# Let's look at the original data. Also define some helping
# functions, which we will use further on.


def plot_3d(points, points_color, title):
    """
    Plot the 3D scatter plot of points colored by points_color.

    Parameters:
    - points: array-like, shape (n_samples, 3)
        The 3D points to plot.
    - points_color: array-like, shape (n_samples,)
        Color values corresponding to each point.
    - title: str
        Title for the plot.
    """
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    """
    Plot the 2D scatter plot of points colored by points_color.

    Parameters:
    - points: array-like, shape (n_samples, 2)
        The 2D points to plot.
    - points_color: array-like, shape (n_samples,)
        Color values corresponding to each point.
    - title: str
        Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    """
    Add a 2D scatter plot to the given axes.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes to plot on.
    - points: array-like, shape (n_samples, 2)
        The 2D points to plot.
    - points_color: array-like, shape (n_samples,)
        Color values corresponding to each point.
    - title: str, optional
        Title for the plot.
    """
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


plot_3d(S_points, S_color, "Original S-curve samples")

# %%
# Define algorithms for the manifold learning
# -------------------------------------------
#
# Manifold learning is an approach to non-linear dimensionality reduction.
# Algorithms for this task are based on the idea that the dimensionality of
# many data sets is only artificially high.
#
# Read more in the :ref:`User Guide <manifold>`.
n_neighbors = 12  # 用于恢复局部线性结构的邻居数量
n_components = 2  # 流形的坐标数目

# %%
# 局部线性嵌入
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 局部线性嵌入（Locally Linear Embedding, LLE）可以看作一系列局部主成分分析，
# 这些分析在全局范围内进行比较，以找到最佳的非线性嵌入。
# 在用户指南中可以阅读更多信息 :ref:`User Guide <locally_linear_embedding>`.

params = {
    "n_neighbors": n_neighbors,  # 邻居数量
    "n_components": n_components,  # 嵌入的坐标数
    "eigen_solver": "auto",  # 特征值求解器
    "random_state": 0,  # 随机数种子
}

# 创建标准局部线性嵌入对象
lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
# 对数据进行标准局部线性嵌入变换
S_standard = lle_standard.fit_transform(S_points)

# 创建局部切线空间对齐的局部线性嵌入对象
lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
# 对数据进行局部切线空间对齐的局部线性嵌入变换
S_ltsa = lle_ltsa.fit_transform(S_points)

# 创建Hessian特征映射的局部线性嵌入对象
lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
# 对数据进行Hessian特征映射的局部线性嵌入变换
S_hessian = lle_hessian.fit_transform(S_points)

# 创建修改的局部线性嵌入对象
lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
# 对数据进行修改的局部线性嵌入变换
S_mod = lle_mod.fit_transform(S_points)

# %%
fig, axs = plt.subplots(
    nrows=2, ncols=2, figsize=(7, 7), facecolor="white", constrained_layout=True
)
fig.suptitle("局部线性嵌入", size=16)

lle_methods = [
    ("标准局部线性嵌入", S_standard),
    ("局部切线空间对齐", S_ltsa),
    ("Hessian特征映射", S_hessian),
    ("修改的局部线性嵌入", S_mod),
]
for ax, method in zip(axs.flat, lle_methods):
    name, points = method
    add_2d_scatter(ax, points, S_color, name)

plt.show()

# %%
# Isomap嵌入
# ^^^^^^^^^^^^^^^^
#
# 通过等距映射进行非线性降维。
# Isomap 寻找一个低维嵌入，同时保持所有点之间的测地距离。
# 在用户指南中可以阅读更多信息 :ref:`User Guide <isomap>`.

isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
# 对数据进行Isomap嵌入变换
S_isomap = isomap.fit_transform(S_points)

plot_2d(S_isomap, S_color, "Isomap嵌入")

# %%
# 多维尺度分析（MDS）
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# 多维尺度分析寻求数据的低维表示，其中距离很好地保持了原始高维空间中的距离。
# 在用户指南中可以阅读更多信息 :ref:`User Guide <multidimensional_scaling>`.

md_scaling = manifold.MDS(
    n_components=n_components,
    max_iter=50,
    n_init=4,
    random_state=0,
    normalized_stress=False,
)
# 对数据进行多维尺度分析变换
S_scaling = md_scaling.fit_transform(S_points)

plot_2d(S_scaling, S_color, "多维尺度分析")

# %%
# 用于非线性降维的谱嵌入
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 此实现使用拉普拉斯特征映射，通过图拉普拉斯的谱分解找到数据的低维表示。
# 在用户指南中可以阅读更多信息 :ref:`User Guide <spectral_embedding>`.

spectral = manifold.SpectralEmbedding(
    n_components=n_components, n_neighbors=n_neighbors, random_state=42
)
# 使用谱嵌入方法将数据集 S_points 进行降维处理，并返回处理后的结果
S_spectral = spectral.fit_transform(S_points)

# 调用自定义函数 plot_2d 绘制二维图像，展示谱嵌入的结果
plot_2d(S_spectral, S_color, "Spectral Embedding")

# %%
# t-SNE (T-distributed Stochastic Neighbor Embedding)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# t-SNE 将数据点之间的相似性转换为联合概率，并尝试最小化低维嵌入和高维数据之间的
# Kullback-Leibler 散度。t-SNE 的成本函数不是凸的，即不同的初始化可能导致不同的结果。
# 更多详细信息请参阅用户指南中的 :ref:`User Guide <t_sne>`。

# 创建 t-SNE 对象，并设置参数：降维后的维度数、困惑度、初始化方式、最大迭代次数和随机种子
t_sne = manifold.TSNE(
    n_components=n_components,
    perplexity=30,
    init="random",
    max_iter=250,
    random_state=0,
)

# 使用 t-SNE 对象对数据集 S_points 进行降维处理
S_t_sne = t_sne.fit_transform(S_points)

# 调用自定义函数 plot_2d 绘制二维图像，展示 t-SNE 的降维结果
plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")

# %%
```