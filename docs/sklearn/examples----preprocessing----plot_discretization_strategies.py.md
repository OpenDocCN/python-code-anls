# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_discretization_strategies.py`

```
    # 引入matplotlib.pyplot库，用于绘图
    import matplotlib.pyplot as plt
    # 引入numpy库，用于科学计算
    import numpy as np

    # 从sklearn.datasets中引入make_blobs函数，用于生成聚类数据
    from sklearn.datasets import make_blobs
    # 从sklearn.preprocessing中引入KBinsDiscretizer类，用于特征分箱处理
    from sklearn.preprocessing import KBinsDiscretizer

    # 定义分箱的策略列表
    strategies = ["uniform", "quantile", "kmeans"]

    # 设置样本数量
    n_samples = 200
    # 设置聚类中心点
    centers_0 = np.array([[0, 0], [0, 5], [2, 4], [8, 8]])
    centers_1 = np.array([[0, 0], [3, 1]])

    # 构造数据集
    random_state = 42
    X_list = [
        np.random.RandomState(random_state).uniform(-3, 3, size=(n_samples, 2)),
        make_blobs(
            n_samples=[
                n_samples // 10,
                n_samples * 4 // 10,
                n_samples // 10,
                n_samples * 4 // 10,
            ],
            cluster_std=0.5,
            centers=centers_0,
            random_state=random_state,
        )[0],
        make_blobs(
            n_samples=[n_samples // 5, n_samples * 4 // 5],
            cluster_std=0.5,
            centers=centers_1,
            random_state=random_state,
        )[0],
    ]

    # 创建一个图像对象，设置图像大小
    figure = plt.figure(figsize=(14, 9))
    i = 1
    # 遍历数据集列表
    for ds_cnt, X in enumerate(X_list):
        # 创建子图，行数为数据集列表长度，列数为策略列表长度加1
        ax = plt.subplot(len(X_list), len(strategies) + 1, i)
        # 绘制散点图
        ax.scatter(X[:, 0], X[:, 1], edgecolors="k")
        # 如果是第一个数据集，设置子图标题
        if ds_cnt == 0:
            ax.set_title("Input data", size=14)

        # 构造网格点
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # 设置子图的x轴和y轴范围
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # 设置子图的x轴和y轴刻度为空
        ax.set_xticks(())
        ax.set_yticks(())

        # 子图序号加1
        i += 1
        # 使用KBinsDiscretizer对数据集进行转换处理
        # transform the dataset with KBinsDiscretizer
    # 对每种指定的分箱策略循环处理
    for strategy in strategies:
        # 使用指定的分箱策略创建 KBinsDiscretizer 对象
        enc = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=strategy)
        # 根据输入数据 X 训练分箱器
        enc.fit(X)
        # 对网格数据进行分箱编码转换
        grid_encoded = enc.transform(grid)

        # 在子图中创建一个新的 Axes 对象
        ax = plt.subplot(len(X_list), len(strategies) + 1, i)

        # 绘制水平条纹
        horizontal = grid_encoded[:, 0].reshape(xx.shape)
        ax.contourf(xx, yy, horizontal, alpha=0.5)
        # 绘制垂直条纹
        vertical = grid_encoded[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, vertical, alpha=0.5)

        # 绘制原始数据点的散点图
        ax.scatter(X[:, 0], X[:, 1], edgecolors="k")
        # 设置 x 轴和 y 轴的范围
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # 设置 x 轴和 y 轴的刻度为空，即不显示刻度
        ax.set_xticks(())
        ax.set_yticks(())
        # 如果是第一个数据集，则设置子图标题为当前策略的名称
        if ds_cnt == 0:
            ax.set_title("strategy='%s'" % (strategy,), size=14)

        # 更新子图索引
        i += 1
# 调整图形布局以确保子图之间的间距合适
plt.tight_layout()
# 显示当前所有已创建的图形
plt.show()
```