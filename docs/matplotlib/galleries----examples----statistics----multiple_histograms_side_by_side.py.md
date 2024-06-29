# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\multiple_histograms_side_by_side.py`

```
"""
==========================================
Producing multiple histograms side by side
==========================================

This example plots horizontal histograms of different samples along
a categorical x-axis. Additionally, the histograms are plotted to
be symmetrical about their x-position, thus making them very similar
to violin plots.

To make this highly specialized plot, we can't use the standard ``hist``
method. Instead, we use ``barh`` to draw the horizontal bars directly. The
vertical positions and lengths of the bars are computed via the
``np.histogram`` function. The histograms for all the samples are
computed using the same range (min and max values) and number of bins,
so that the bins for each sample are in the same vertical positions.

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块用于绘图
import numpy as np  # 导入numpy模块用于数值计算

np.random.seed(19680801)  # 设置随机种子以确保结果可复现
number_of_bins = 20  # 设置直方图的条数

# An example of three data sets to compare
number_of_data_points = 387  # 每个数据集的数据点数目
labels = ["A", "B", "C"]  # 数据集的标签
data_sets = [np.random.normal(0, 1, number_of_data_points),
             np.random.normal(6, 1, number_of_data_points),
             np.random.normal(-3, 1, number_of_data_points)]  # 三个数据集的示例数据

# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))  # 所有数据集的最小和最大值范围
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets
]  # 使用相同的范围和条数计算每个数据集的直方图数据

binned_maximums = np.max(binned_data_sets, axis=1)  # 计算每个数据集直方图的最大高度
x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))  # 计算每个直方图的水平位置

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)  # 计算所有直方图共享的bin边界
heights = np.diff(bin_edges)  # 计算bin的高度
centers = bin_edges[:-1] + heights / 2  # 计算每个bin的中心位置

# Cycle through and plot each histogram
fig, ax = plt.subplots()  # 创建图形和子图对象
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc - 0.5 * binned_data  # 计算每个直方图的左侧位置
    ax.barh(centers, binned_data, height=heights, left=lefts)  # 绘制水平直方图

ax.set_xticks(x_locations, labels)  # 设置x轴刻度和标签

ax.set_ylabel("Data values")  # 设置y轴标签
ax.set_xlabel("Data sets")  # 设置x轴标签

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.barh` / `matplotlib.pyplot.barh`
```