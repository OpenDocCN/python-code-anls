# `numpy-ml\numpy_ml\plots\gmm_plots.py`

```
# 禁用 flake8 检查
# 导入 numpy 库，并重命名为 np
import numpy as np
# 从 sklearn.datasets.samples_generator 模块中导入 make_blobs 函数
from sklearn.datasets.samples_generator import make_blobs
# 从 scipy.stats 模块中导入 multivariate_normal 函数
from scipy.stats import multivariate_normal
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 seaborn 库，并重命名为 sns
import seaborn as sns

# 设置 seaborn 库的样式为白色
sns.set_style("white")
# 设置 seaborn 库的上下文为 paper，字体比例为 1
sns.set_context("paper", font_scale=1)

# 从 numpy_ml.gmm 模块中导入 GMM 类
from numpy_ml.gmm import GMM
# 从 matplotlib.colors 模块中导入 ListedColormap 类

# 定义函数 plot_countour，用于绘制等高线图
def plot_countour(X, x, y, z, ax, xlim, ylim):
    # 定义函数 fixed_aspect_ratio，用于设置 matplotlib 图的固定纵横比
    def fixed_aspect_ratio(ratio, ax):
        """
        Set a fixed aspect ratio on matplotlib plots
        regardless of axis units
        """
        # 获取当前图的 x 和 y 轴范围
        xvals, yvals = ax.get_xlim(), ax.get_ylim()

        # 计算 x 和 y 轴范围
        xrange = xvals[1] - xvals[0]
        yrange = yvals[1] - yvals[0]
        # 设置图的纵横比
        ax.set_aspect(ratio * (xrange / yrange), adjustable="box")

    # 在随机分布的数据点上绘制等高线
    ax.contour(x, y, z, 6, linewidths=0.5, colors="k")

    # 设置 x 和 y 轴的范围
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # 调用函数设置固定纵横比
    fixed_aspect_ratio(1, ax)
    return ax

# 定义函数 plot_clusters，用于绘制聚类结果
def plot_clusters(model, X, ax):
    # 获取聚类数目
    C = model.C

    # 计算 x 和 y 轴的范围
    xmin = min(X[:, 0]) - 0.1 * (max(X[:, 0]) - min(X[:, 0]))
    xmax = max(X[:, 0]) + 0.1 * (max(X[:, 0]) - min(X[:, 0]))
    ymin = min(X[:, 1]) - 0.1 * (max(X[:, 1]) - min(X[:, 1]))
    ymax = max(X[:, 1]) + 0.1 * (max(X[:, 1]) - min(X[:, 1]))

    # 遍历每个聚类
    for c in range(C):
        # 创建多元正态分布对象
        rv = multivariate_normal(model.mu[c], model.sigma[c], allow_singular=True)

        # 在 x 和 y 轴上生成均匀间隔的点
        x = np.linspace(xmin, xmax, 500)
        y = np.linspace(ymin, ymax, 500)

        # 生成网格点
        X1, Y1 = np.meshgrid(x, y)
        xy = np.column_stack([X1.flat, Y1.flat])

        # 计算网格点处的密度值
        Z = rv.pdf(xy).reshape(X1.shape)
        # 调用函数绘制等高线图
        ax = plot_countour(X, X1, Y1, Z, ax=ax, xlim=(xmin, xmax), ylim=(ymin, ymax))
        # 在图上标记聚类中心
        ax.plot(model.mu[c, 0], model.mu[c, 1], "ro")

    # 绘制数据点
    cm = ListedColormap(sns.color_palette().as_hex())
    # 获取每个数据点的聚类标签
    labels = model.Q.argmax(1)
    # 将标签列表转换为集合，去除重复的标签
    uniq = set(labels)
    # 遍历每个唯一的标签
    for i in uniq:
        # 根据标签值筛选出对应的数据点，并在散点图上绘制
        ax.scatter(X[labels == i, 0], X[labels == i, 1], c=cm.colors[i - 1], s=30)
    # 返回绘制好的散点图对象
    return ax
# 定义一个绘图函数
def plot():
    # 创建一个包含 4x4 子图的图形对象
    fig, axes = plt.subplots(4, 4)
    # 设置图形大小为 10x10 英寸
    fig.set_size_inches(10, 10)
    # 遍历所有子图
    for i, ax in enumerate(axes.flatten()):
        # 设置样本数量为 150，特征数量为 2，随机生成类别数量
        n_ex = 150
        n_in = 2
        n_classes = np.random.randint(2, 4)
        # 生成聚类数据
        X, y = make_blobs(
            n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=i
        )
        # 数据中心化
        X -= X.mean(axis=0)

        # 进行 10 次运行，选择最佳拟合
        best_elbo = -np.inf
        for k in range(10):
            # 创建 GMM 模型对象
            _G = GMM(C=n_classes, seed=k * 3)
            # 拟合数据
            ret = _G.fit(X, max_iter=100, verbose=False)
            # 如果拟合失败，则重新拟合
            while ret != 0:
                print("Components collapsed; Refitting")
                ret = _G.fit(X, max_iter=100, verbose=False)

            # 选择最佳 ELBO 值的模型
            if _G.best_elbo > best_elbo:
                best_elbo = _G.best_elbo
                G = _G

        # 绘制聚类结果
        ax = plot_clusters(G, X, ax)
        # 设置 x 轴刻度标签为空
        ax.xaxis.set_ticklabels([])
        # 设置 y 轴刻度标签为空
        ax.yaxis.set_ticklabels([])
        # 设置子图标题，显示类别数量和最终 ELBO 值
        ax.set_title("# Classes: {}; Final VLB: {:.2f}".format(n_classes, G.best_elbo))

    # 调整子图布局
    plt.tight_layout()
    # 保存图形为图片文件
    plt.savefig("img/plot.png", dpi=300)
    # 关闭所有图形对象
    plt.close("all")
```