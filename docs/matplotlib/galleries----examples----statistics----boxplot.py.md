# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\boxplot.py`

```
"""
=================================
Artist customization in box plots
=================================

This example demonstrates how to use the various keyword arguments to fully
customize box plots. The first figure demonstrates how to remove and add
individual components (note that the mean is the only value not shown by
default). The second figure demonstrates how the styles of the artists can be
customized. It also demonstrates how to set the limit of the whiskers to
specific percentiles (lower right Axes)

A good general reference on boxplots and their history can be found here:
https://vita.had.co.nz/papers/boxplots.pdf

"""

import matplotlib.pyplot as plt
import numpy as np

# fake data
np.random.seed(19680801)  # 设置随机种子以确保可复现性
data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)  # 生成37x4的对数正态分布的假数据
labels = list('ABCD')  # 设置标签为['A', 'B', 'C', 'D']
fs = 10  # 设置字体大小为10

# %%
# Demonstrate how to toggle the display of different elements:

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)  # 创建2行3列的子图布局，共享y轴
axs[0, 0].boxplot(data, labels=labels)  # 在第一个子图中绘制默认设置的箱线图
axs[0, 0].set_title('Default', fontsize=fs)  # 设置子图标题为'Default'，字体大小为fs

axs[0, 1].boxplot(data, labels=labels, showmeans=True)  # 在第二个子图中显示均值
axs[0, 1].set_title('showmeans=True', fontsize=fs)  # 设置子图标题为'showmeans=True'，字体大小为fs

axs[0, 2].boxplot(data, labels=labels, showmeans=True, meanline=True)  # 在第三个子图中显示均值和均值线
axs[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)  # 设置子图标题为'showmeans=True,\nmeanline=True'，字体大小为fs

axs[1, 0].boxplot(data, labels=labels, showbox=False, showcaps=False)  # 在第四个子图中不显示箱体和边缘线
tufte_title = 'Tufte Style \n(showbox=False,\nshowcaps=False)'  # 设置子图标题为'Tufte Style \n(showbox=False,\nshowcaps=False)'
axs[1, 0].set_title(tufte_title, fontsize=fs)  # 设置子图标题为上述字符串，字体大小为fs

axs[1, 1].boxplot(data, labels=labels, notch=True, bootstrap=10000)  # 在第五个子图中显示缺口和置信区间
axs[1, 1].set_title('notch=True,\nbootstrap=10000', fontsize=fs)  # 设置子图标题为'notch=True,\nbootstrap=10000'，字体大小为fs

axs[1, 2].boxplot(data, labels=labels, showfliers=False)  # 在第六个子图中不显示异常值
axs[1, 2].set_title('showfliers=False', fontsize=fs)  # 设置子图标题为'showfliers=False'，字体大小为fs

for ax in axs.flat:
    ax.set_yscale('log')  # 将每个子图的y轴设置为对数尺度
    ax.set_yticklabels([])  # 不显示y轴刻度标签

fig.subplots_adjust(hspace=0.4)  # 调整子图布局的垂直间距
plt.show()


# %%
# Demonstrate how to customize the display different elements:

boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')  # 设置箱线图的属性：虚线样式、宽度、颜色
flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                  markeredgecolor='none')  # 设置异常值的属性：圆形标记、填充颜色、大小、边缘颜色
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')  # 设置中位数线的属性：点划线样式、宽度、颜色
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')  # 设置均值点的属性：菱形标记、边缘颜色、填充颜色
meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')  # 设置均值线的属性：虚线样式、宽度、颜色

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)  # 创建2行3列的子图布局，共享y轴
axs[0, 0].boxplot(data, boxprops=boxprops)  # 在第一个子图中使用自定义的箱线图属性
axs[0, 0].set_title('Custom boxprops', fontsize=fs)  # 设置子图标题为'Custom boxprops'，字体大小为fs

axs[0, 1].boxplot(data, flierprops=flierprops, medianprops=medianprops)  # 在第二个子图中使用自定义的异常值和中位数线属性
axs[0, 1].set_title('Custom medianprops\nand flierprops', fontsize=fs)  # 设置子图标题为'Custom medianprops\nand flierprops'，字体大小为fs

axs[0, 2].boxplot(data, whis=(0, 100))  # 在第三个子图中设置箱线图的whisker为(0, 100)
axs[0, 2].set_title('whis=(0, 100)', fontsize=fs)  # 设置子图标题为'whis=(0, 100)'，字体大小为fs

axs[1, 0].boxplot(data, meanprops=meanpointprops, meanline=False,
                  showmeans=True)  # 在第四个子图中自定义均值点和不显示均值线但显示均值
axs[1, 0].set_title('Custom mean\nas point', fontsize=fs)  # 设置子图标题为'Custom mean\nas point'，字体大小为fs

plt.show()  # 显示所有子图
# 在第二行的第二个子图上绘制箱线图，使用指定的均值线属性和显示均值线
axs[1, 1].boxplot(data, meanprops=meanlineprops, meanline=True, showmeans=True)
# 设置第二行第二列子图的标题，使用自定义的均值线作为直线，并指定字体大小
axs[1, 1].set_title('Custom mean\nas line', fontsize=fs)

# 在第二行的第三个子图上绘制箱线图，指定箱线图的上下分位数百分比为15%和85%
axs[1, 2].boxplot(data, whis=[15, 85])
# 设置第二行第三列子图的标题，说明使用了特定的百分位数范围，并指定字体大小
axs[1, 2].set_title('whis=[15, 85]\n#percentiles', fontsize=fs)

# 遍历所有子图，将y轴设置为对数刻度，并隐藏y轴刻度标签
for ax in axs.flat:
    ax.set_yscale('log')
    ax.set_yticklabels([])

# 设置整个图形的标题
fig.suptitle("I never said they'd be pretty")
# 调整子图之间的垂直间距
fig.subplots_adjust(hspace=0.4)
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.boxplot` / `matplotlib.pyplot.boxplot`
```