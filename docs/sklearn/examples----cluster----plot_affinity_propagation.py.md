# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_affinity_propagation.py`

```
# 引入必要的库：numpy 用于数值计算，sklearn 中的相关模块用于聚类算法和评估指标，matplotlib 用于绘图
import numpy as np

from sklearn import metrics  # 导入评估指标模块
from sklearn.cluster import AffinityPropagation  # 导入 Affinity Propagation 聚类算法
from sklearn.datasets import make_blobs  # 导入 make_blobs 用于生成聚类数据

# %%
# 生成样本数据
# --------------------
centers = [[1, 1], [-1, -1], [1, -1]]
# 生成具有三个中心的样本数据，总共300个样本，每个簇的标准差为0.5，随机数种子为0
X, labels_true = make_blobs(
    n_samples=300, centers=centers, cluster_std=0.5, random_state=0
)

# %%
# 计算 Affinity Propagation 聚类
# ----------------------------
# 创建 Affinity Propagation 聚类对象，设置首选项为-50，随机数种子为0，并在数据 X 上拟合
af = AffinityPropagation(preference=-50, random_state=0).fit(X)
# 获取聚类中心的索引
cluster_centers_indices = af.cluster_centers_indices_
# 获取每个样本的聚类标签
labels = af.labels_

# 计算聚类的数量
n_clusters_ = len(cluster_centers_indices)

# 打印聚类数量的估计值
print("Estimated number of clusters: %d" % n_clusters_)
# 打印同质性指标
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# 打印完整性指标
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# 打印V度量
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# 打印调整兰德指数
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# 打印调整的互信息
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
# 打印轮廓系数
print(
    "Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(X, labels, metric="sqeuclidean")
)

# %%
# 绘制结果
# -----------
import matplotlib.pyplot as plt

# 关闭所有已有的图形窗口
plt.close("all")
# 创建新的图形窗口
plt.figure(1)
# 清除当前图形窗口中的内容

# 设置颜色循环器，使用 plt.cm.viridis 生成包含四种颜色的颜色映射
colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))

# 遍历每个聚类，绘制对应的散点图和聚类中心
for k, col in zip(range(n_clusters_), colors):
    # 获取当前聚类的成员索引
    class_members = labels == k
    # 获取当前聚类的中心点坐标
    cluster_center = X[cluster_centers_indices[k]]
    # 绘制当前聚类的成员点
    plt.scatter(
        X[class_members, 0], X[class_members, 1], color=col["color"], marker="."
    )
    # 绘制当前聚类的中心点
    plt.scatter(
        cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
    )
    # 绘制每个成员点与中心点之间的连线
    for x in X[class_members]:
        plt.plot(
            [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
        )

# 设置图形的标题
plt.title("Estimated number of clusters: %d" % n_clusters_)
# 展示图形
plt.show()
```