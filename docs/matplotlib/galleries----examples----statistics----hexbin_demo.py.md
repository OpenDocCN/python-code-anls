# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\hexbin_demo.py`

```
"""
=====================
Hexagonal binned plot
=====================

`~.Axes.hexbin` is a 2D histogram plot, in which the bins are hexagons and
the color represents the number of data points within each bin.
"""

# 导入matplotlib.pyplot库并命名为plt
import matplotlib.pyplot as plt
# 导入numpy库并命名为np
import numpy as np

# 设定随机种子以保证结果可复现性
np.random.seed(19680801)

# 设定数据点数量
n = 100_000
# 生成服从标准正态分布的随机数数组x
x = np.random.standard_normal(n)
# 生成服从特定线性关系的随机数数组y
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
# 确定x和y的范围
xlim = x.min(), x.max()
ylim = y.min(), y.max()

# 创建包含两个子图的图像窗口
fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

# 在第一个子图上绘制hexbin图，设置网格大小和颜色映射
hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
# 设定第一个子图的x和y轴限制
ax0.set(xlim=xlim, ylim=ylim)
# 设置第一个子图的标题
ax0.set_title("Hexagon binning")
# 添加第一个子图的colorbar，并设定标签为'counts'
cb = fig.colorbar(hb, ax=ax0, label='counts')

# 在第二个子图上绘制hexbin图，设置网格大小为50，颜色映射为'inferno'，并设定bins为'log'
hb = ax1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
# 设定第二个子图的x和y轴限制
ax1.set(xlim=xlim, ylim=ylim)
# 设置第二个子图的标题
ax1.set_title("With a log color scale")
# 添加第二个子图的colorbar，并设定标签为'counts'
cb = fig.colorbar(hb, ax=ax1, label='counts')

# 展示图像
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hexbin` / `matplotlib.pyplot.hexbin`
```