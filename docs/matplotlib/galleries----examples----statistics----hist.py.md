# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\hist.py`

```
"""
==========
Histograms
==========

How to plot histograms with Matplotlib.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from matplotlib import colors  # 从matplotlib中导入colors模块
from matplotlib.ticker import PercentFormatter  # 从matplotlib.ticker中导入PercentFormatter类，用于格式化坐标轴显示百分比

# 创建一个具有固定种子的随机数生成器，以保证可重现性
rng = np.random.default_rng(19680801)

# %%
# 生成数据并绘制简单直方图
# -----------------------------------------
#
# 要生成一个1D直方图，我们只需要一个数字向量。对于2D直方图，我们需要第二个向量。
# 我们将在下面生成这两个向量，并显示每个向量的直方图。

N_points = 100000  # 数据点数目
n_bins = 20  # 直方图柱子的数目

# 生成两个正态分布
dist1 = rng.standard_normal(N_points)
dist2 = 0.4 * rng.standard_normal(N_points) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# 可以使用 *bins* 关键字参数设置柱子的数目。
axs[0].hist(dist1, bins=n_bins)  # 绘制dist1的直方图
axs[1].hist(dist2, bins=n_bins)  # 绘制dist2的直方图

plt.show()


# %%
# 更新直方图颜色
# -------------------------
#
# 直方图方法返回（除其他外）一个 ``patches`` 对象。这使我们可以访问绘制对象的属性。
# 使用这个对象，我们可以编辑直方图来符合我们的喜好。让我们根据每个柱子的y值来改变柱子的颜色。

fig, axs = plt.subplots(1, 2, tight_layout=True)

# N 是每个柱子的计数，bins 是每个柱子的下限
N, bins, patches = axs[0].hist(dist1, bins=n_bins)

# 我们将按高度进行颜色编码，但您可以使用任何标量
fracs = N / N.max()

# 我们需要将数据归一化到 0..1 以适应整个颜色图的范围
norm = colors.Normalize(fracs.min(), fracs.max())

# 现在，我们循环遍历我们的对象，并相应地设置每个对象的颜色
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# 我们也可以通过总计数归一化我们的输入
axs[1].hist(dist1, bins=n_bins, density=True)

# 现在我们格式化y轴以显示百分比
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))


# %%
# 绘制2D直方图
# -------------------
#
# 要绘制2D直方图，我们只需要两个相同长度的向量，分别对应直方图的每个轴。

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(dist1, dist2)


# %%
# 自定义你的直方图
# --------------------------
#
# 自定义2D直方图类似于1D情况，您可以控制视觉组件，如柱子大小或颜色归一化。

fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True,
                        tight_layout=True)

# 我们可以增加每个轴上的柱子数目
axs[0].hist2d(dist1, dist2, bins=40)

# 同时定义颜色的归一化
axs[1].hist2d(dist1, dist2, bins=40, norm=colors.LogNorm())

# 我们还可以为每个轴定义自定义的柱子数目
axs[2].hist2d(dist1, dist2, bins=(80, 10), norm=colors.LogNorm())

# %%
#
# .. admonition:: References
#    
#    This section provides references to functions, methods, classes, and modules used in this example:
#    
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`: Used for creating histograms in Matplotlib.
#    - `matplotlib.pyplot.hist2d`: Used for creating 2D histograms in Matplotlib.
#    - `matplotlib.ticker.PercentFormatter`: Used for formatting tick labels as percentages in Matplotlib.
```