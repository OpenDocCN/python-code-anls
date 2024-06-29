# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\histogram_bihistogram.py`

```
"""
===========
Bihistogram
===========

How to plot a bihistogram with Matplotlib.
"""

# 导入 matplotlib 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np

# Create a random number generator with a fixed seed for reproducibility
# 使用固定种子创建随机数生成器 rng，确保结果可重现性
rng = np.random.default_rng(19680801)

# %%
# Generate data and plot a bihistogram
# ------------------------------------
#
# To generate a bihistogram we need two datasets (each being a vector of numbers).
# We will plot both histograms using plt.hist() and set the weights of the second
# one to be negative. We'll generate data below and plot the bihistogram.

# 设置生成数据点的数量
N_points = 10_000

# Generate two normal distributions
# 生成两个正态分布的数据集
dataset1 = np.random.normal(0, 1, size=N_points)
dataset2 = np.random.normal(1, 2, size=N_points)

# Use a constant bin width to make the two histograms easier to compare visually
# 使用固定的箱宽度来更容易地比较两个直方图的视觉效果
bin_width = 0.25
bins = np.arange(np.min([dataset1, dataset2]),
                    np.max([dataset1, dataset2]) + bin_width, bin_width)

fig, ax = plt.subplots()

# Plot the first histogram
# 绘制第一个直方图，使用 dataset1 的数据和指定的箱子边界
ax.hist(dataset1, bins=bins, label="Dataset 1")

# Plot the second histogram
# (notice the negative weights, which flip the histogram upside down)
# 绘制第二个直方图，使用 dataset2 的数据，权重设为负数以翻转直方图的方向
ax.hist(dataset2, weights=-np.ones_like(dataset2), bins=bins, label="Dataset 2")
# 添加一条水平线在 y=0 处，以突出两个直方图的分界线
ax.axhline(0, color="k")
# 添加图例
ax.legend()

# 显示图形
plt.show()
```