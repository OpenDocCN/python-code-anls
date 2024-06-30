# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_mean_shift.py`

```
# 引入 NumPy 库，用于数值计算
import numpy as np

# 从 sklearn 库中引入 MeanShift 聚类算法和带宽估计函数
from sklearn.cluster import MeanShift, estimate_bandwidth
# 从 sklearn 库中引入 make_blobs 函数，用于生成样本数据
from sklearn.datasets import make_blobs

# %%
# 生成样本数据
# --------------------
# 定义三个中心点的坐标作为聚类的中心
centers = [[1, 1], [-1, -1], [1, -1]]
# 生成包含 10000 个样本的数据集，以 centers 为中心，标准差为 0.6
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# %%
# 使用 MeanShift 进行聚类计算
# ---------------------------------

# 通过指定 quantile 和 n_samples 参数，自动计算带宽 bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# 创建 MeanShift 聚类对象 ms，并指定带宽参数和使用二进制种子点初始化
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# 对数据集 X 进行聚类
ms.fit(X)
# 获取每个样本点的聚类标签
labels = ms.labels_
# 获取聚类中心点的坐标
cluster_centers = ms.cluster_centers_

# 获取唯一的聚类标签值
labels_unique = np.unique(labels)
# 获取聚类簇的数量
n_clusters_ = len(labels_unique)

# 打印估计的聚类数量
print("number of estimated clusters : %d" % n_clusters_)

# %%
# 绘制聚类结果
# -----------
import matplotlib.pyplot as plt

# 创建图像窗口
plt.figure(1)
# 清空当前图像
plt.clf()

# 定义颜色和标记形状
colors = ["#dede00", "#377eb8", "#f781bf"]
markers = ["x", "o", "^"]

# 遍历每个聚类簇
for k, col in zip(range(n_clusters_), colors):
    # 获取属于当前聚类簇 k 的样本点
    my_members = labels == k
    # 获取当前聚类簇的中心点坐标
    cluster_center = cluster_centers[k]
    # 绘制属于当前聚类簇的样本点
    plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
    # 绘制当前聚类簇的中心点
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        markers[k],
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )

# 设置图像标题，显示估计的聚类数量
plt.title("Estimated number of clusters: %d" % n_clusters_)
# 显示图像
plt.show()
```