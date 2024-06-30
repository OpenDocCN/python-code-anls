# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_digits_agglomeration.py`

```
"""
=========================================================
Feature agglomeration
=========================================================

These images show how similar features are merged together using
feature agglomeration.

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库
import matplotlib.pyplot as plt  # 导入用于绘图的matplotlib库
import numpy as np  # 导入numpy库，用于数值计算

from sklearn import cluster, datasets  # 导入sklearn中的cluster和datasets模块
from sklearn.feature_extraction.image import grid_to_graph  # 导入图像特征提取模块中的grid_to_graph函数

# 载入手写数字数据集
digits = datasets.load_digits()
images = digits.images  # 获取手写数字的图像数据
X = np.reshape(images, (len(images), -1))  # 将图像数据转换为二维数组形式
connectivity = grid_to_graph(*images[0].shape)  # 构建图像的连接性结构

# 使用特征聚合进行聚类，将特征聚合为32个簇
agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
agglo.fit(X)  # 对数据进行特征聚合
X_reduced = agglo.transform(X)  # 转换原始数据到降维后的数据空间
X_restored = agglo.inverse_transform(X_reduced)  # 将降维后的数据恢复到原始数据空间
images_restored = np.reshape(X_restored, images.shape)  # 将恢复后的数据重新转换为图像形式

# 绘制图像
plt.figure(1, figsize=(4, 3.5))
plt.clf()
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91)

# 绘制原始数据图像和聚合后的数据图像
for i in range(4):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
    if i == 1:
        plt.title("Original data")
    plt.subplot(3, 4, 4 + i + 1)
    plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16, interpolation="nearest")
    if i == 1:
        plt.title("Agglomerated data")
    plt.xticks(())
    plt.yticks(())

# 绘制标签图像
plt.subplot(3, 4, 10)
plt.imshow(
    np.reshape(agglo.labels_, images[0].shape),
    interpolation="nearest",
    cmap=plt.cm.nipy_spectral,
)
plt.xticks(())
plt.yticks(())
plt.title("Labels")
plt.show()
```