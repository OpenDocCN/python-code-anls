# `D:\src\scipysrc\scikit-learn\examples\manifold\plot_mds.py`

```
"""
=========================
Multi-dimensional scaling
=========================

An illustration of the metric and non-metric MDS on generated noisy data.

The reconstructed points using the metric MDS and non metric MDS are slightly
shifted to avoid overlapping.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需库
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

# 定义一个极小值
EPSILON = np.finfo(np.float32).eps

# 设定样本数和随机种子
n_samples = 20
seed = np.random.RandomState(seed=3)

# 生成随机的二维样本数据
X_true = seed.randint(0, 20, 2 * n_samples).astype(float)
X_true = X_true.reshape((n_samples, 2))

# 将数据中心化（均值归零）
X_true -= X_true.mean()

# 计算样本点之间的欧氏距离作为相似度
similarities = euclidean_distances(X_true)

# 向相似度矩阵添加噪声
noise = np.random.rand(n_samples, n_samples)
noise = noise + noise.T
noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
similarities += noise

# 创建度量多维缩放（MDS）对象，使用欧氏距离作为不相似度度量
mds = manifold.MDS(
    n_components=2,
    max_iter=3000,
    eps=1e-9,
    random_state=seed,
    dissimilarity="precomputed",
    n_jobs=1,
)
# 对相似度矩阵进行MDS降维，获取嵌入空间位置
pos = mds.fit(similarities).embedding_

# 创建非度量多维缩放（NMDS）对象，不使用欧氏距离作为度量
nmds = manifold.MDS(
    n_components=2,
    metric=False,
    max_iter=3000,
    eps=1e-12,
    dissimilarity="precomputed",
    random_state=seed,
    n_jobs=1,
    n_init=1,
)
# 对相似度矩阵进行NMDS降维，使用前面得到的MDS位置作为初始值，获取嵌入空间位置
npos = nmds.fit_transform(similarities, init=pos)

# 将数据重新缩放以与真实数据一致
pos *= np.sqrt((X_true**2).sum()) / np.sqrt((pos**2).sum())
npos *= np.sqrt((X_true**2).sum()) / np.sqrt((npos**2).sum())

# 使用PCA对原始数据和降维后的数据进行旋转
clf = PCA(n_components=2)
X_true = clf.fit_transform(X_true)
pos = clf.fit_transform(pos)
npos = clf.fit_transform(npos)

# 创建图形窗口和坐标轴
fig = plt.figure(1)
ax = plt.axes([0.0, 0.0, 1.0, 1.0])

# 绘制散点图展示原始数据、MDS结果和NMDS结果
s = 100
plt.scatter(X_true[:, 0], X_true[:, 1], color="navy", s=s, lw=0, label="True Position")
plt.scatter(pos[:, 0], pos[:, 1], color="turquoise", s=s, lw=0, label="MDS")
plt.scatter(npos[:, 0], npos[:, 1], color="darkorange", s=s, lw=0, label="NMDS")
plt.legend(scatterpoints=1, loc="best", shadow=False)

# 将相似度矩阵进行归一化处理，计算边的权重
similarities = similarities.max() / (similarities + EPSILON) * 100
np.fill_diagonal(similarities, 0)

# 绘制边（线段）连接相似的样本点
start_idx, end_idx = np.where(pos)
segments = [
    [X_true[i, :], X_true[j, :]] for i in range(len(pos)) for j in range(len(pos))
]
values = np.abs(similarities)
lc = LineCollection(
    segments, zorder=0, cmap=plt.cm.Blues, norm=plt.Normalize(0, values.max())
)
lc.set_array(similarities.flatten())
lc.set_linewidths(np.full(len(segments), 0.5))
ax.add_collection(lc)

# 显示图形
plt.show()
```