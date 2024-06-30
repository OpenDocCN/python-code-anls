# `D:\src\scipysrc\scikit-learn\examples\bicluster\plot_spectral_biclustering.py`

```
"""
=============================================
A demo of the Spectral Biclustering algorithm
=============================================

This example demonstrates how to generate a checkerboard dataset and bicluster
it using the :class:`~sklearn.cluster.SpectralBiclustering` algorithm. The
spectral biclustering algorithm is specifically designed to cluster data by
simultaneously considering both the rows (samples) and columns (features) of a
matrix. It aims to identify patterns not only between samples but also within
subsets of samples, allowing for the detection of localized structure within the
data. This makes spectral biclustering particularly well-suited for datasets
where the order or arrangement of features is fixed, such as in images, time
series, or genomes.

The data is generated, then shuffled and passed to the spectral biclustering
algorithm. The rows and columns of the shuffled matrix are then rearranged to
plot the biclusters found.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate sample data
# --------------------
# We generate the sample data using the
# :func:`~sklearn.datasets.make_checkerboard` function. Each pixel within
# `shape=(300, 300)` represents with it's color a value from a uniform
# distribution. The noise is added from a normal distribution, where the value
# chosen for `noise` is the standard deviation.
#
# As you can see, the data is distributed over 12 cluster cells and is
# relatively well distinguishable.
from matplotlib import pyplot as plt

from sklearn.datasets import make_checkerboard

# Define the number of clusters and generate checkerboard data
n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=42
)

# Display the original dataset as a matrix plot
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")
_ = plt.show()

# %%
# We shuffle the data and the goal is to reconstruct it afterwards using
# :class:`~sklearn.cluster.SpectralBiclustering`.
import numpy as np

# Creating lists of shuffled row and column indices
rng = np.random.RandomState(0)
row_idx_shuffled = rng.permutation(data.shape[0])
col_idx_shuffled = rng.permutation(data.shape[1])

# %%
# We redefine the shuffled data and plot it. We observe that we lost the
# structure of the original data matrix.
data = data[row_idx_shuffled][:, col_idx_shuffled]

# Display the shuffled dataset as a matrix plot
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")
_ = plt.show()

# %%
# Fitting `SpectralBiclustering`
# ------------------------------
# We fit the model and compare the obtained clusters with the ground truth. Note
# that when creating the model we specify the same number of clusters that we
# used to create the dataset (`n_clusters = (4, 3)`), which will contribute to
# obtaining a good result.
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

# Initialize the SpectralBiclustering model with specified parameters
model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)

# Fit the model to the shuffled data
model.fit(data)
# 计算两组双聚类的相似性得分
score = consensus_score(
    model.biclusters_, (rows[:, row_idx_shuffled], columns[:, col_idx_shuffled])
)
# 打印双聚类的一致性得分
print(f"consensus score: {score:.1f}")

# %%
# 得分在0到1之间，1表示完美匹配。它展示了双聚类的质量。

# %%
# 绘制结果
# ----------------
# 现在，根据:class:`~sklearn.cluster.SpectralBiclustering`模型分配的行和列标签
# 按升序重新排列数据，然后再次绘图。`row_labels_`范围从0到3，而`column_labels_`
# 范围从0到2，表示每行4个聚类和每列3个聚类。

# 首先重新排列行，然后重新排列列。
reordered_rows = data[np.argsort(model.row_labels_)]
reordered_data = reordered_rows[:, np.argsort(model.column_labels_)]

plt.matshow(reordered_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")
_ = plt.show()

# %%
# 最后一步，我们希望展示模型分配的行和列标签之间的关系。因此，我们使用
# :func:`numpy.outer`创建一个网格，它接受排序后的`row_labels_`和`column_labels_`，
# 并为每个标签添加1，以确保标签从1开始而不是从0开始，以便更好地可视化。
plt.matshow(
    np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
    cmap=plt.cm.Blues,
)
plt.title("Checkerboard structure of rearranged data")
plt.show()

# %%
# 行和列标签向量的外积显示了棋盘结构的表示，其中不同的行和列标签组合由不同的蓝色阴影表示。
```