# `D:\src\scipysrc\scikit-learn\examples\manifold\plot_t_sne_perplexity.py`

```
"""
=============================================================================
t-SNE: The effect of various perplexity values on the shape
=============================================================================

An illustration of t-SNE on the two concentric circles and the S-curve
datasets for different perplexity values.

We observe a tendency towards clearer shapes as the perplexity value increases.

The size, the distance and the shape of clusters may vary upon initialization,
perplexity values and does not always convey a meaning.

As shown below, t-SNE for higher perplexities finds meaningful topology of
two concentric circles, however the size and the distance of the circles varies
slightly from the original. Contrary to the two circles dataset, the shapes
visually diverge from S-curve topology on the S-curve dataset even for
larger perplexity values.

For further details, "How to Use t-SNE Effectively"
https://distill.pub/2016/misread-tsne/ provides a good discussion of the
effects of various parameters, as well as interactive plots to explore
those effects.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time  # 导入时间函数，用于计算程序运行时间

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
from matplotlib.ticker import NullFormatter  # 导入 NullFormatter，用于坐标轴格式设置

from sklearn import datasets, manifold  # 导入 sklearn 库中的 datasets 和 manifold 模块

n_samples = 150  # 设定数据样本数量
n_components = 2  # 设定降维后的维度数为 2
(fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))  # 创建一个 3x5 的图像子图布局，figsize 设置图像大小
perplexities = [5, 30, 50, 100]  # 不同 perplexity 值的列表

X, y = datasets.make_circles(  # 创建一个环形数据集 X 和对应的标签 y
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=0
)

red = y == 0  # 筛选标签为 0 的数据点
green = y == 1  # 筛选标签为 1 的数据点

ax = subplots[0][0]  # 获取第一个子图
ax.scatter(X[red, 0], X[red, 1], c="r")  # 绘制红色类别的散点图
ax.scatter(X[green, 0], X[green, 1], c="g")  # 绘制绿色类别的散点图
ax.xaxis.set_major_formatter(NullFormatter())  # 设置 x 轴主刻度格式为空
ax.yaxis.set_major_formatter(NullFormatter())  # 设置 y 轴主刻度格式为空
plt.axis("tight")  # 调整坐标轴范围

for i, perplexity in enumerate(perplexities):  # 遍历不同 perplexity 值
    ax = subplots[0][i + 1]  # 获取当前 perplexity 对应的子图

    t0 = time()  # 记录当前时间
    tsne = manifold.TSNE(  # 创建 t-SNE 模型
        n_components=n_components,
        init="random",
        random_state=0,
        perplexity=perplexity,
        max_iter=300,
    )
    Y = tsne.fit_transform(X)  # 对数据进行降维
    t1 = time()  # 记录降维后的时间
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))  # 打印 perplexity 和运行时间
    ax.set_title("Perplexity=%d" % perplexity)  # 设置子图标题
    ax.scatter(Y[red, 0], Y[red, 1], c="r")  # 绘制降维后红色类别的散点图
    ax.scatter(Y[green, 0], Y[green, 1], c="g")  # 绘制降维后绿色类别的散点图
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置 x 轴主刻度格式为空
    ax.yaxis.set_major_formatter(NullFormatter())  # 设置 y 轴主刻度格式为空
    ax.axis("tight")  # 调整坐标轴范围

# Another example using s-curve
X, color = datasets.make_s_curve(n_samples, random_state=0)  # 创建 S-curve 数据集

ax = subplots[1][0]  # 获取 S-curve 数据集的第一个子图
ax.scatter(X[:, 0], X[:, 2], c=color)  # 绘制 S-curve 数据集的散点图
ax.xaxis.set_major_formatter(NullFormatter())  # 设置 x 轴主刻度格式为空
ax.yaxis.set_major_formatter(NullFormatter())  # 设置 y 轴主刻度格式为空

for i, perplexity in enumerate(perplexities):  # 遍历不同 perplexity 值
    ax = subplots[1][i + 1]  # 获取当前 perplexity 对应的子图

    t0 = time()  # 记录当前时间
    tsne = manifold.TSNE(  # 创建 t-SNE 模型
        n_components=n_components,
        init="random",
        random_state=0,
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=300,
    )
    Y = tsne.fit_transform(X)  # 对数据进行降维
    # 记录当前时间
    t1 = time()
    # 打印包含 perplexity 的 S-curve 运行时间
    print("S-curve, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

    # 设置图表标题，显示 perplexity 值
    ax.set_title("Perplexity=%d" % perplexity)
    # 绘制散点图，Y[:, 0] 和 Y[:, 1] 分别是数据的两个特征维度，使用 color 进行着色
    ax.scatter(Y[:, 0], Y[:, 1], c=color)
    # 设置 x 轴的主要刻度格式为空
    ax.xaxis.set_major_formatter(NullFormatter())
    # 设置 y 轴的主要刻度格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    # 根据数据范围自动调整坐标轴
    ax.axis("tight")
# 创建一个在二维空间中均匀分布的网格，用于可视化
x = np.linspace(0, 1, int(np.sqrt(n_samples)))  # 在0到1之间生成平均间隔的数列，数目为n_samples的平方根
xx, yy = np.meshgrid(x, x)  # 创建网格坐标矩阵xx和yy
X = np.hstack(
    [
        xx.ravel().reshape(-1, 1),  # 将xx展平并重新形状为列向量，与下一行组合成X矩阵的第一列
        yy.ravel().reshape(-1, 1),  # 将yy展平并重新形状为列向量，与上一行组合成X矩阵的第二列
    ]
)
color = xx.ravel()  # 将xx展平作为颜色值
ax = subplots[2][0]  # 获取绘图对象的子图中的第三行第一列的图表

# 在子图中绘制散点图，并设置x轴和y轴的主要刻度格式为空
ax.scatter(X[:, 0], X[:, 1], c=color)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

# 遍历不同的perplexity值，进行t-SNE降维并可视化
for i, perplexity in enumerate(perplexities):
    ax = subplots[2][i + 1]  # 获取绘图对象的子图中的第三行第i+1列的图表

    t0 = time()  # 记录开始时间
    tsne = manifold.TSNE(
        n_components=n_components,  # 输出的降维空间维度数
        init="random",  # 初始化的方法为随机
        random_state=0,  # 随机数生成器的种子，保证可重复性
        perplexity=perplexity,  # t-SNE的perplexity参数，影响降维结果的分布紧密程度
        max_iter=400,  # 最大迭代次数
    )
    Y = tsne.fit_transform(X)  # 对数据X进行t-SNE降维转换
    t1 = time()  # 记录结束时间
    print("uniform grid, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))  # 打印降维所用时间

    ax.set_title("Perplexity=%d" % perplexity)  # 设置子图标题，显示当前perplexity值
    ax.scatter(Y[:, 0], Y[:, 1], c=color)  # 在子图中绘制t-SNE降维后的散点图，并使用相同的颜色值
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置x轴主要刻度格式为空
    ax.yaxis.set_major_formatter(NullFormatter())  # 设置y轴主要刻度格式为空
    ax.axis("tight")  # 调整子图坐标轴范围，使所有数据点可见

plt.show()  # 显示所有绘制的子图
```