# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_segmentation_toy.py`

```
"""
===========================================
Spectral clustering for image segmentation
===========================================

In this example, an image with connected circles is generated and
spectral clustering is used to separate the circles.

In these settings, the :ref:`spectral_clustering` approach solves the problem
know as 'normalized graph cuts': the image is seen as a graph of
connected voxels, and the spectral clustering algorithm amounts to
choosing graph cuts defining regions while minimizing the ratio of the
gradient along the cut, and the volume of the region.

As the algorithm tries to balance the volume (ie balance the region
sizes), if we take circles with different sizes, the segmentation fails.

In addition, as there is no useful information in the intensity of the image,
or its gradient, we choose to perform the spectral clustering on a graph
that is only weakly informed by the gradient. This is close to performing
a Voronoi partition of the graph.

In addition, we use the mask of the objects to restrict the graph to the
outline of the objects. In this example, we are interested in
separating the objects one from the other, and not from the background.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate the data
# -----------------
import numpy as np

l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

# Define circles based on given centers and radii
circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1**2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2**2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3**2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4**2

# %%
# Plotting four circles
# ---------------------
img = circle1 + circle2 + circle3 + circle4

# We use a mask that limits to the foreground: the problem that we are
# interested in here is not separating the objects from the background,
# but separating them one from the other.
mask = img.astype(bool)

# Convert the image to float and add random noise
img = img.astype(float)
img += 1 + 0.2 * np.random.randn(*img.shape)

# %%
# Convert the image into a graph with the value of the gradient on the
# edges.
from sklearn.feature_extraction import image

graph = image.img_to_graph(img, mask=mask)

# %%
# Take a decreasing function of the gradient resulting in a segmentation
# that is close to a Voronoi partition
graph.data = np.exp(-graph.data / graph.data.std())

# %%
# Perform spectral clustering using the arpack solver since amg is
# numerically unstable on this example. Then plot the results.
import matplotlib.pyplot as plt

from sklearn.cluster import spectral_clustering

labels = spectral_clustering(graph, n_clusters=4, eigen_solver="arpack")
label_im = np.full(mask.shape, -1.0)
label_im[mask] = labels

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].matshow(img)
axs[1].matshow(label_im)

plt.show()
# %%
# 绘制两个圆圈
# --------------------
# 这里我们重复上面的过程，但只考虑我们生成的前两个圆圈。
# 注意，在这种情况下，由于区域大小更容易平衡，所以圆圈之间的分隔更清晰。

# 将两个圆圈的图像叠加在一起
img = circle1 + circle2
# 创建一个布尔掩码，将图像转换为布尔类型（True/False）
mask = img.astype(bool)
# 将图像转换为浮点类型
img = img.astype(float)

# 在图像的每个像素上加上一些随机噪声
img += 1 + 0.2 * np.random.randn(*img.shape)

# 根据图像和掩码创建图形的表示，掩码指定了哪些像素是有效的
graph = image.img_to_graph(img, mask=mask)
# 对图形的数据进行指数转换，用于图形分析
graph.data = np.exp(-graph.data / graph.data.std())

# 使用谱聚类算法对图形进行聚类，分成两类
labels = spectral_clustering(graph, n_clusters=2, eigen_solver="arpack")
# 创建一个与掩码形状相同的标签图像，初始化为-1.0
label_im = np.full(mask.shape, -1.0)
# 将聚类标签放入标签图像中对应的有效区域
label_im[mask] = labels

# 创建包含两个子图的图像窗口
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# 在第一个子图中显示原始图像
axs[0].matshow(img)
# 在第二个子图中显示聚类标签的图像
axs[1].matshow(label_im)

# 显示图像窗口
plt.show()
```