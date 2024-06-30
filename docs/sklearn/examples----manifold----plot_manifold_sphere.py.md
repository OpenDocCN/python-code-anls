# `D:\src\scipysrc\scikit-learn\examples\manifold\plot_manifold_sphere.py`

```
"""
=============================================
Manifold Learning methods on a severed sphere
=============================================

An application of the different :ref:`manifold` techniques
on a spherical data-set. Here one can see the use of
dimensionality reduction in order to gain some intuition
regarding the manifold learning methods. Regarding the dataset,
the poles are cut from the sphere, as well as a thin slice down its
side. This enables the manifold learning techniques to
'spread it open' whilst projecting it onto two dimensions.

For a similar example, where the methods are applied to the
S-curve dataset, see :ref:`sphx_glr_auto_examples_manifold_plot_compare_methods.py`

Note that the purpose of the :ref:`MDS <multidimensional_scaling>` is
to find a low-dimensional representation of the data (here 2D) in
which the distances respect well the distances in the original
high-dimensional space, unlike other manifold-learning algorithms,
it does not seeks an isotropic representation of the data in
the low-dimensional space. Here the manifold problem matches fairly
that of representing a flat map of the Earth, as with
`map projection <https://en.wikipedia.org/wiki/Map_projection>`_

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time

import matplotlib.pyplot as plt

# Unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from sklearn.utils import check_random_state

# Variables for manifold learning.
n_neighbors = 10  # 邻居数量设为10
n_samples = 1000  # 样本数设为1000

# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * np.pi

# Sever the poles from the sphere.
indices = (t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8)))
colors = p[indices]
x, y, z = (
    np.sin(t[indices]) * np.cos(p[indices]),  # x坐标
    np.sin(t[indices]) * np.sin(p[indices]),  # y坐标
    np.cos(t[indices]),  # z坐标
)

# Plot our dataset.
fig = plt.figure(figsize=(15, 8))  # 创建图形对象
plt.suptitle(
    "Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14  # 设置总标题
)

ax = fig.add_subplot(251, projection="3d")  # 在第1个子图中创建3D坐标轴
ax.scatter(x, y, z, c=p[indices], cmap=plt.cm.rainbow)  # 绘制散点图
ax.view_init(40, -10)  # 设置视角

sphere_data = np.array([x, y, z]).T  # 转置数组，使其符合sklearn的输入要求

# Perform Locally Linear Embedding Manifold learning
methods = ["standard", "ltsa", "hessian", "modified"]
labels = ["LLE", "LTSA", "Hessian LLE", "Modified LLE"]

for i, method in enumerate(methods):
    t0 = time()  # 记录开始时间
    trans_data = (
        manifold.LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=2, method=method, random_state=42
        )
        .fit_transform(sphere_data)  # 执行LLE算法降维
        .T  # 转置数据以匹配matplotlib的要求
    )
    t1 = time()  # 记录结束时间
    print("%s: %.2g sec" % (methods[i], t1 - t0))  # 输出算法名称和执行时间

    ax = fig.add_subplot(252 + i)  # 在第2列开始的位置创建子图
    # 绘制散点图，以 trans_data 的第一列作为 x 轴数据，第二列作为 y 轴数据，使用 colors 数组指定每个点的颜色，颜色映射为彩虹色
    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    
    # 设置图表标题，包含标签和执行时间的格式化字符串
    plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    
    # 设置 x 轴主刻度格式为空，即不显示主刻度标签
    ax.xaxis.set_major_formatter(NullFormatter())
    
    # 设置 y 轴主刻度格式为空，即不显示主刻度标签
    ax.yaxis.set_major_formatter(NullFormatter())
    
    # 自动调整坐标轴范围，使所有数据点都能显示在图中
    plt.axis("tight")
# Perform Isomap Manifold learning.
t0 = time()  # 记录开始时间
trans_data = (
    manifold.Isomap(n_neighbors=n_neighbors, n_components=2)  # 创建 Isomap 对象，设置参数
    .fit_transform(sphere_data)  # 对数据进行 Isomap 变换
    .T  # 转置结果
)
t1 = time()  # 记录结束时间
print("%s: %.2g sec" % ("ISO", t1 - t0))  # 打印运行时间

ax = fig.add_subplot(257)  # 在图形中添加子图
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)  # 绘制散点图
plt.title("%s (%.2g sec)" % ("Isomap", t1 - t0))  # 设置子图标题
ax.xaxis.set_major_formatter(NullFormatter())  # 设置 X 轴主刻度格式为空
ax.yaxis.set_major_formatter(NullFormatter())  # 设置 Y 轴主刻度格式为空
plt.axis("tight")  # 调整坐标轴范围

# Perform Multi-dimensional scaling.
t0 = time()  # 记录开始时间
mds = manifold.MDS(2, max_iter=100, n_init=1, random_state=42)  # 创建 MDS 对象，设置参数
trans_data = mds.fit_transform(sphere_data).T  # 对数据进行 MDS 变换并转置结果
t1 = time()  # 记录结束时间
print("MDS: %.2g sec" % (t1 - t0))  # 打印运行时间

ax = fig.add_subplot(258)  # 在图形中添加子图
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)  # 绘制散点图
plt.title("MDS (%.2g sec)" % (t1 - t0))  # 设置子图标题
ax.xaxis.set_major_formatter(NullFormatter())  # 设置 X 轴主刻度格式为空
ax.yaxis.set_major_formatter(NullFormatter())  # 设置 Y 轴主刻度格式为空
plt.axis("tight")  # 调整坐标轴范围

# Perform Spectral Embedding.
t0 = time()  # 记录开始时间
se = manifold.SpectralEmbedding(
    n_components=2, n_neighbors=n_neighbors, random_state=42
)  # 创建 Spectral Embedding 对象，设置参数
trans_data = se.fit_transform(sphere_data).T  # 对数据进行 Spectral Embedding 变换并转置结果
t1 = time()  # 记录结束时间
print("Spectral Embedding: %.2g sec" % (t1 - t0))  # 打印运行时间

ax = fig.add_subplot(259)  # 在图形中添加子图
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)  # 绘制散点图
plt.title("Spectral Embedding (%.2g sec)" % (t1 - t0))  # 设置子图标题
ax.xaxis.set_major_formatter(NullFormatter())  # 设置 X 轴主刻度格式为空
ax.yaxis.set_major_formatter(NullFormatter())  # 设置 Y 轴主刻度格式为空
plt.axis("tight")  # 调整坐标轴范围

# Perform t-distributed stochastic neighbor embedding.
t0 = time()  # 记录开始时间
tsne = manifold.TSNE(n_components=2, random_state=0)  # 创建 t-SNE 对象，设置参数
trans_data = tsne.fit_transform(sphere_data).T  # 对数据进行 t-SNE 变换并转置结果
t1 = time()  # 记录结束时间
print("t-SNE: %.2g sec" % (t1 - t0))  # 打印运行时间

ax = fig.add_subplot(2, 5, 10)  # 在图形中添加子图
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)  # 绘制散点图
plt.title("t-SNE (%.2g sec)" % (t1 - t0))  # 设置子图标题
ax.xaxis.set_major_formatter(NullFormatter())  # 设置 X 轴主刻度格式为空
ax.yaxis.set_major_formatter(NullFormatter())  # 设置 Y 轴主刻度格式为空
plt.axis("tight")  # 调整坐标轴范围

plt.show()  # 显示所有子图
```