# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\boxplot_vs_violin.py`

```py
"""
===================================
Box plot vs. violin plot comparison
===================================

Note that although violin plots are closely related to Tukey's (1977)
box plots, they add useful information such as the distribution of the
sample data (density trace).

By default, box plots show data points outside 1.5 * the inter-quartile
range as outliers above or below the whiskers whereas violin plots show
the whole range of the data.

A good general reference on boxplots and their history can be found
here: http://vita.had.co.nz/papers/boxplots.pdf

Violin plots require matplotlib >= 1.4.

For more information on violin plots, the scikit-learn docs have a great
section: https://scikit-learn.org/stable/modules/density.html
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))  # 创建一个包含两个子图的画布

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设定随机数种子，保证随机生成的数据可复现性

# generate some random test data
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]  # 生成四组不同标准差的正态分布随机数据

# plot violin plot
axs[0].violinplot(all_data,  # 绘制小提琴图
                  showmeans=False,  # 不显示均值
                  showmedians=True)  # 显示中位数
axs[0].set_title('Violin plot')  # 设置子图标题为 "Violin plot"

# plot box plot
axs[1].boxplot(all_data)  # 绘制箱线图
axs[1].set_title('Box plot')  # 设置子图标题为 "Box plot"

# adding horizontal grid lines
for ax in axs:  # 循环处理每个子图
    ax.yaxis.grid(True)  # 显示 y 轴的网格线
    ax.set_xticks([y + 1 for y in range(len(all_data))],  # 设置 x 轴刻度位置
                  labels=['x1', 'x2', 'x3', 'x4'])  # 设置 x 轴刻度标签
    ax.set_xlabel('Four separate samples')  # 设置 x 轴标签
    ax.set_ylabel('Observed values')  # 设置 y 轴标签

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.boxplot` / `matplotlib.pyplot.boxplot`
#    - `matplotlib.axes.Axes.violinplot` / `matplotlib.pyplot.violinplot`
```