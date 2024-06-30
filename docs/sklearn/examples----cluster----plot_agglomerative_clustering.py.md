# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_agglomerative_clustering.py`

```
# 聚合聚类（凝聚聚类）的例子，包括有结构和无结构的情况

# 这个例子展示了在数据中引入连接图以捕获局部结构的效果。连接图简单地是数据中的20个最近邻的图。

# 引入连接图有两个优点。首先，使用稀疏连接矩阵进行聚类通常更快。

# 其次，当使用连接矩阵时，单链接、平均链接和完全链接通常会不稳定，并倾向于创建一些非常快速增长的群集。
# 平均链接和完全链接通过在合并时考虑两个群集之间的所有距离来抵抗这种渗流行为（而单链接通过只考虑群集之间的最短距离来夸张这种行为）。
# 连接图打破了平均链接和完全链接的这种机制，使它们更像更脆弱的单链接。这种效应在非常稀疏的图中更为显著
# （尝试减少 kneighbors_graph 中的邻居数量）并且在完全链接中尤为明显。特别是，在图中有非常少量的邻居时，
# 强加的几何形状接近于单链接，已知它具有这种渗流不稳定性。

# 作者：scikit-learn 开发者
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import time  # 导入时间模块用于性能计算

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 用于绘图
import numpy as np  # 导入 numpy 用于数值计算

# 导入聚类算法和邻居图生成工具
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# 生成示例数据
n_samples = 1500  # 数据样本数量
np.random.seed(0)  # 设置随机种子以保证可重复性
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)  # x 坐标
y = t * np.sin(t)  # y 坐标

X = np.concatenate((x, y))  # 将 x 和 y 连接成一个特征向量
X += 0.7 * np.random.randn(2, n_samples)  # 加入一些随机噪声
X = X.T  # 转置以得到样本数量 x 特征数量的矩阵

# 创建捕捉局部连接性的图。较大数量的邻居将提供更均匀的群集，但计算成本更高。
# 非常大数量的邻居将导致更均匀分布的群集大小，但可能不会强加数据的局部流形结构。
knn_graph = kneighbors_graph(X, 30, include_self=False)

# 对于两种连接性选项（有结构和无结构）
for connectivity in (None, knn_graph):
    # 对不同的聚类数量(n_clusters)进行迭代，分别为30和3
    for n_clusters in (30, 3):
        # 创建一个大小为(10, 4)的新图像窗口
        plt.figure(figsize=(10, 4))
        # 对四种不同的连接方法(linkage)进行迭代，分别为"average", "complete", "ward", "single"
        for index, linkage in enumerate(("average", "complete", "ward", "single")):
            # 在1行4列的布局中，创建第 index+1 个子图
            plt.subplot(1, 4, index + 1)
            # 创建聚合聚类模型，指定连接方式(linkage)、连接性(connectivity)，以及聚类数量(n_clusters)
            model = AgglomerativeClustering(
                linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
            )
            # 记录开始拟合模型的时间
            t0 = time.time()
            # 对数据 X 进行拟合
            model.fit(X)
            # 计算拟合模型所花费的时间
            elapsed_time = time.time() - t0
            # 绘制散点图，根据模型的标签(model.labels_)对数据 X 进行着色，使用颜色映射 plt.cm.nipy_spectral
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
            # 设置子图的标题，显示连接方式(linkage)和所花费的时间
            plt.title(
                "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
                fontdict=dict(verticalalignment="top"),
            )
            # 设置坐标轴比例相等
            plt.axis("equal")
            # 关闭坐标轴显示
            plt.axis("off")

            # 调整子图之间的间距和位置，使得整体布局更合理
            plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
            # 设置总标题，显示聚类数量(n_clusters)和连接性(connectivity)是否为非空
            plt.suptitle(
                "n_cluster=%i, connectivity=%r"
                % (n_clusters, connectivity is not None),
                size=17,
            )
# 显示当前所有已创建的图形
plt.show()
```