# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_coin_segmentation.py`

```
"""
================================================
Segmenting the picture of greek coins in regions
================================================

This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogeneous regions.

This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.

There are three options to assign labels:

* 'kmeans' spectral clustering clusters samples in the embedding space
  using a kmeans algorithm
* 'discrete' iteratively searches for the closest partition
  space to the embedding space of spectral clustering.
* 'cluster_qr' assigns labels using the QR factorization with pivoting
  that directly determines the partition in the embedding space.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time  # 导入时间模块，用于计时

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算
from scipy.ndimage import gaussian_filter  # 导入 scipy.ndimage 中的高斯滤波函数
from skimage.data import coins  # 从 skimage.data 中导入 coins 数据集
from skimage.transform import rescale  # 导入 skimage.transform 中的 rescale 函数

from sklearn.cluster import spectral_clustering  # 导入 sklearn.cluster 中的 spectral_clustering 函数
from sklearn.feature_extraction import image  # 导入 sklearn.feature_extraction 中的 image 模块

# load the coins as a numpy array
orig_coins = coins()  # 载入 coins 数据集，存储为 numpy 数组 orig_coins

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)  # 对原始图像进行高斯平滑处理
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect", anti_aliasing=False)  # 将图像按照0.2的比例重新缩放

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(rescaled_coins)  # 将图像转换为图形结构，边的权重基于梯度值

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 10  # 设置 beta 参数，用于计算边的权重
eps = 1e-6  # 设置一个极小值，防止除零错误
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps  # 应用指数函数计算新的边权重

# The number of segmented regions to display needs to be chosen manually.
# The current version of 'spectral_clustering' does not support determining
# the number of good quality clusters automatically.
n_regions = 26  # 设置要显示的分割区域的数量

# %%
# Compute and visualize the resulting regions

# Computing a few extra eigenvectors may speed up the eigen_solver.
# The spectral clustering quality may also benefit from requesting
# extra regions for segmentation.
n_regions_plus = 3  # 设置要计算的额外特征向量的数量

# Apply spectral clustering using the default eigen_solver='arpack'.
# Any implemented solver can be used: eigen_solver='arpack', 'lobpcg', or 'amg'.
# Choosing eigen_solver='amg' requires an extra package called 'pyamg'.
# The quality of segmentation and the speed of calculations is mostly determined
# by the choice of the solver and the value of the tolerance 'eigen_tol'.
# TODO: varying eigen_tol seems to have no effect for 'lobpcg' and 'amg' #21243.
for assign_labels in ("kmeans", "discretize", "cluster_qr"):
    t0 = time.time()  # 记录开始时间
    # 使用谱聚类算法对图形进行聚类分析，生成每个节点的标签
    labels = spectral_clustering(
        graph,
        n_clusters=(n_regions + n_regions_plus),  # 设定聚类的数量，这里是区域数加上附加区域数的总和
        eigen_tol=1e-7,  # 特征值分解的容差值
        assign_labels=assign_labels,  # 分配标签的策略，由外部传入
        random_state=42,  # 设置随机数种子，保证结果的可重复性
    )
    
    # 计算运行时间，并将标签重新调整为与原始图像相同的形状
    t1 = time.time()
    labels = labels.reshape(rescaled_coins.shape)
    
    # 创建一个新的图像窗口，并显示灰度化的原始图像
    plt.figure(figsize=(5, 5))
    plt.imshow(rescaled_coins, cmap=plt.cm.gray)
    
    # 隐藏图像的 x 和 y 轴刻度
    plt.xticks(())
    plt.yticks(())
    
    # 构造图像标题，显示谱聚类的策略及运行时间
    title = "Spectral clustering: %s, %.2fs" % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
    
    # 根据每个区域的标签，绘制轮廓线，并使用不同颜色区分各个区域
    for l in range(n_regions):
        colors = [plt.cm.nipy_spectral((l + 4) / float(n_regions + 4))]
        plt.contour(labels == l, colors=colors)
        # 若要查看每个单独区域的轮廓线，可以取消下面这行代码的注释：plt.pause(0.5)
# 显示 matplotlib 的绘图窗口
plt.show()

# TODO: 在 #21194 合并并且 #21243 修复后，检查哪个特征值求解器是最好的，
# 并在这个示例中明确设置 eigen_solver='arpack'、'lobpcg' 或 'amg'，
# 并设置 eigen_tol。
```