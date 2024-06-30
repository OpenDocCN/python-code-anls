# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_kde_1d.py`

```
# 导入必要的库和模块：matplotlib.pyplot用于绘图，numpy用于数值计算，scipy.stats.norm用于正态分布，sklearn.neighbors.KernelDensity用于核密度估计
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# ----------------------------------------------------------------------
# 绘制直方图到核密度估计的过渡过程

# 设置随机种子
np.random.seed(1)
# 样本数量
N = 20
# 生成混合分布的样本数据
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]
# 在指定范围内生成用于绘图的数据点
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
# 设定直方图的分箱边界
bins = np.linspace(-5, 10, 10)

# 创建包含四个子图的图像框架
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# 调整子图之间的垂直和水平间距
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# 直方图 1
ax[0, 0].hist(X[:, 0], bins=bins, fc="#AAAAFF", density=True)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# 直方图 2
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc="#AAAAFF", density=True)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# Tophat 核密度估计
kde = KernelDensity(kernel="tophat", bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF")
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian 核密度估计
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF")
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

# 在每个子图中绘制样本数据点
for axi in ax.ravel():
    axi.plot(X[:, 0], np.full(X.shape[0], -0.01), "+k")
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

# 在左侧两个子图中添加标签
for axi in ax[:, 0]:
    # 设置 Y 轴的标签为 "Normalized Density"
    axi.set_ylabel("Normalized Density")
# ----------------------------------------------------------------------
# 遍历第一行中的每个子图，并设置它们的 x 轴标签为 "x"
for axi in ax[1, :]:
    axi.set_xlabel("x")

# ----------------------------------------------------------------------
# 绘制所有可用的核密度估计图
X_plot = np.linspace(-6, 6, 1000)[:, None]  # 在指定范围内生成用于绘图的数据点
X_src = np.zeros((1, 1))  # 创建一个包含单个零元素的数组作为源数据点

# 创建一个2x3的子图布局，并调整其间距和边距
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)

# 自定义格式化函数，用于 x 轴刻度标签的显示
def format_func(x, loc):
    if x == 0:
        return "0"
    elif x == 1:
        return "h"
    elif x == -1:
        return "-h"
    else:
        return "%ih" % x

# 遍历不同的核函数类型，并在子图中填充对应的核密度估计图
for i, kernel in enumerate(
    ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
):
    axi = ax.ravel()[i]  # 获取当前子图对象
    # 计算当前核函数对应的核密度估计值，并绘制在当前子图中
    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
    axi.fill(X_plot[:, 0], np.exp(log_dens), "-k", fc="#AAAAFF")  # 填充估计值的曲线
    axi.text(-2.6, 0.95, kernel)  # 在指定位置添加核函数名称的文本标签

    # 设置 x 轴主要刻度的格式化方式和间隔
    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())  # 隐藏 y 轴刻度

    axi.set_ylim(0, 1.05)  # 设置 y 轴显示范围
    axi.set_xlim(-2.9, 2.9)  # 设置 x 轴显示范围

ax[0, 1].set_title("Available Kernels")  # 设置特定子图的标题为 "Available Kernels"

# ----------------------------------------------------------------------
# 绘制一维密度示例
N = 100  # 数据点数量
np.random.seed(1)
# 生成混合分布的样本数据 X
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]  # 生成用于绘图的数据点

# 计算真实密度函数的值作为比较基准
true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(5, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()  # 创建新的图形和子图对象
# 填充真实密度函数的曲线
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ["navy", "cornflowerblue", "darkorange"]
kernels = ["gaussian", "tophat", "epanechnikov"]
lw = 2

# 针对不同的核函数类型，绘制对应的核密度估计曲线
for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color=color,
        lw=lw,
        linestyle="-",
        label="kernel = '{0}'".format(kernel),
    )

ax.text(6, 0.38, "N={0} points".format(N))  # 在指定位置添加包含数据点数量的文本标签

ax.legend(loc="upper left")  # 在左上角添加图例
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")  # 绘制数据点的位置

ax.set_xlim(-4, 9)  # 设置 x 轴显示范围
ax.set_ylim(-0.02, 0.4)  # 设置 y 轴显示范围
plt.show()  # 显示绘制的图形
```