# `D:\src\scipysrc\scikit-learn\examples\bicluster\plot_spectral_coclustering.py`

```
"""
==============================================
A demo of the Spectral Co-Clustering algorithm
==============================================

This example demonstrates how to generate a dataset and bicluster it
using the Spectral Co-Clustering algorithm.

The dataset is generated using the ``make_biclusters`` function, which
creates a matrix of small values and implants bicluster with large
values. The rows and columns are then shuffled and passed to the
Spectral Co-Clustering algorithm. Rearranging the shuffled matrix to
make biclusters contiguous shows how accurately the algorithm found
the biclusters.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score

# 使用 make_biclusters 函数生成一个具有大小值矩阵和大值双聚类的数据集
data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=5, shuffle=False, random_state=0
)

# 绘制原始数据集的热图
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# 将行和列随机排列以生成打乱的数据集
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

# 绘制打乱后的数据集的热图
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

# 创建 SpectralCoclustering 模型对象并拟合数据
model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)

# 计算聚类结果的一致性得分
score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))

# 打印一致性得分
print("consensus score: {:.3f}".format(score))

# 根据模型的行和列标签重新排列数据，以显示双聚类
fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

# 绘制重新排列后的数据集的热图，以显示双聚类
plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

# 显示所有绘图
plt.show()
```