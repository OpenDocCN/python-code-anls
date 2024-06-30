# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_linkage_comparison.py`

```
# %%
# 导入必要的库和模块
import time  # 导入时间模块，用于计时
import warnings  # 导入警告模块，处理警告信息
from itertools import cycle, islice  # 导入迭代工具，用于循环迭代

import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from sklearn import cluster, datasets  # 导入sklearn库的cluster和datasets模块，用于聚类和数据集操作
from sklearn.preprocessing import StandardScaler  # 导入数据预处理模块，用于数据标准化

# %%
# 生成数据集。我们选择足够大以查看算法的可扩展性，但不要太大以避免运行时间过长

n_samples = 1500  # 数据集样本数

# 生成不同形状和噪声水平的数据集
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=170
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=170)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=170)
rng = np.random.RandomState(170)
no_structure = rng.rand(n_samples, 2), None

# 生成异性分布的数据集
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# 生成具有不同方差的数据集
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170
)

# %%
# 运行聚类算法并绘图

# 设置绘图参数
plt.figure(figsize=(9 * 1.3 + 2, 14.5))  # 设置图像大小
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)  # 调整子图之间的间距

plot_num = 1  # 初始化子图编号

default_base = {"n_neighbors": 10, "n_clusters": 3}  # 默认参数

datasets = [
    (noisy_circles, {"n_clusters": 2}),  # 噪声圆环数据集
    (noisy_moons, {"n_clusters": 2}),    # 噪声月牙数据集
    (varied, {"n_neighbors": 2}),        # 具有变化方差的数据集
    (aniso, {"n_neighbors": 2}),         # 异性分布数据集
    (blobs, {}),                         # 簇状数据集
    (no_structure, {}),                  # 无结构数据集
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # 使用数据集特定值更新参数
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset  # 提取数据集特征和标签

    # 标准化数据集以便更容易选择参数
    X = StandardScaler().fit_transform(X)

    # ============
    # 创建聚类对象
    # ============
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward"
    )  # Ward层次聚类
    complete = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="complete"
    )  # 完全连接层次聚类
    # 创建平均连接聚类对象，使用给定的聚类数目和链接方式
    average = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="average"
    )
    # 创建单链接聚类对象，使用给定的聚类数目和链接方式
    single = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="single"
    )

    # 定义多个聚类算法和对应的名称
    clustering_algorithms = (
        ("Single Linkage", single),
        ("Average Linkage", average),
        ("Complete Linkage", complete),  # complete 变量未在代码中定义
        ("Ward Linkage", ward),  # ward 变量未在代码中定义
    )

    # 遍历每个聚类算法及其名称
    for name, algorithm in clustering_algorithms:
        t0 = time.time()  # 记录开始时间

        # 捕获与 kneighbors_graph 相关的警告信息
        with warnings.catch_warnings():
            # 过滤特定类型的警告信息以忽略
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            # 使用算法对数据进行拟合
            algorithm.fit(X)

        t1 = time.time()  # 记录结束时间
        # 如果算法对象有 labels_ 属性，则使用其标签作为预测结果
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            # 否则调用 predict 方法获取预测结果
            y_pred = algorithm.predict(X)

        # 在图中创建子图区域用于绘制数据点及其分类结果
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)  # 在第一个数据集的第一列添加标题

        # 为每个类别分配不同颜色，并绘制数据点
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # 设置坐标轴范围和标签
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        # 在右下角添加执行时间信息
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1  # 更新子图编号
# 显示当前的 Matplotlib 图形
plt.show()
```