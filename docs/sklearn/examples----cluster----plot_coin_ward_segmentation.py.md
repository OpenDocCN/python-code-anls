# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_coin_ward_segmentation.py`

```
"""
======================================================================
A demo of structured Ward hierarchical clustering on an image of coins
======================================================================

Compute the segmentation of a 2D image with Ward hierarchical
clustering. The clustering is spatially constrained in order
for each segmented region to be in one piece.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate data
# -------------

# 导入包含硬币图像数据的skimage库
from skimage.data import coins

# 从skimage库中获取原始硬币图像数据
orig_coins = coins()

# %%
# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.

# 导入numpy库
import numpy as np
# 导入scipy库中的高斯滤波函数
from scipy.ndimage import gaussian_filter
# 导入skimage库中的图像缩放函数
from skimage.transform import rescale

# 对原始硬币图像应用高斯滤波，sigma=2
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
# 将经过滤波处理后的图像缩小到原始图像的20%，使用反射模式，关闭反锯齿
rescaled_coins = rescale(
    smoothened_coins,
    0.2,
    mode="reflect",
    anti_aliasing=False,
)

# 将二维数组转换为一维数组，以便后续聚类分析
X = np.reshape(rescaled_coins, (-1, 1))

# %%
# Define structure of the data
# ----------------------------
#
# Pixels are connected to their neighbors.

# 导入sklearn库中图像特征提取模块中的网格到图形函数
from sklearn.feature_extraction.image import grid_to_graph

# 使用rescaled_coins的形状作为参数，生成像素的邻接图
connectivity = grid_to_graph(*rescaled_coins.shape)

# %%
# Compute clustering
# ------------------

# 导入时间库
import time as time

# 导入sklearn库中的凝聚聚类模块
from sklearn.cluster import AgglomerativeClustering

# 输出信息：计算结构化的层次聚类...
print("Compute structured hierarchical clustering...")
# 记录开始时间
st = time.time()
# 指定聚类数目为27，使用ward链接方式和之前生成的像素邻接图进行凝聚聚类
n_clusters = 27  # number of regions
ward = AgglomerativeClustering(
    n_clusters=n_clusters, linkage="ward", connectivity=connectivity
)
# 对数据X进行聚类分析
ward.fit(X)
# 将聚类标签重新整形为与rescaled_coins相同的形状
label = np.reshape(ward.labels_, rescaled_coins.shape)
# 输出聚类分析所用时间
print(f"Elapsed time: {time.time() - st:.3f}s")
# 输出像素总数
print(f"Number of pixels: {label.size}")
# 输出聚类数目
print(f"Number of clusters: {np.unique(label).size}")

# %%
# Plot the results on an image
# ----------------------------
#
# Agglomerative clustering is able to segment each coin however, we have had to
# use a ``n_cluster`` larger than the number of coins because the segmentation
# is finding a large in the background.

# 导入matplotlib库中的绘图模块
import matplotlib.pyplot as plt

# 创建一个5x5英寸大小的图像
plt.figure(figsize=(5, 5))
# 将rescaled_coins以灰度图像方式显示
plt.imshow(rescaled_coins, cmap=plt.cm.gray)
# 对每个聚类标签画出轮廓，使用nipy_spectral颜色映射
for l in range(n_clusters):
    plt.contour(
        label == l,
        colors=[
            plt.cm.nipy_spectral(l / float(n_clusters)),
        ],
    )
# 关闭坐标轴显示
plt.axis("off")
# 显示图像
plt.show()
```