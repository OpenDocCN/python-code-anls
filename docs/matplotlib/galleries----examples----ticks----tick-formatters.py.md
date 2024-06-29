# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\tick-formatters.py`

```
"""
===============
Tick formatters
===============

Tick formatters define how the numeric value associated with a tick on an axis
is formatted as a string.

This example illustrates the usage and effect of the most common formatters.

The tick format is configured via the function `~.Axis.set_major_formatter`
or `~.Axis.set_minor_formatter`. It accepts:

- a format string, which implicitly creates a `.StrMethodFormatter`.
- a function,  implicitly creates a `.FuncFormatter`.
- an instance of a `.Formatter` subclass. The most common are

  - `.NullFormatter`: No labels on the ticks.
  - `.StrMethodFormatter`: Use string `str.format` method.
  - `.FormatStrFormatter`: Use %-style formatting.
  - `.FuncFormatter`: Define labels through a function.
  - `.FixedFormatter`: Set the label strings explicitly.
  - `.ScalarFormatter`: Default formatter for scalars: auto-pick the format string.
  - `.PercentFormatter`: Format labels as a percentage.

  See :ref:`formatters` for a complete list.

"""

import matplotlib.pyplot as plt

from matplotlib import ticker


def setup(ax, title):
    """Set up common parameters for the Axes in the example."""
    # only show the bottom spine
    ax.yaxis.set_major_locator(ticker.NullLocator())  # 设置y轴主要定位器为空定位器，即无主刻度
    ax.spines[['left', 'right', 'top']].set_visible(False)  # 隐藏左、右、上脊柱线

    # define tick positions
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))  # 设置x轴主刻度定位器为1.00的倍数定位器
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))  # 设置x轴次刻度定位器为0.25的倍数定位器

    ax.xaxis.set_ticks_position('bottom')  # 设置x轴刻度位置为底部
    ax.tick_params(which='major', width=1.00, length=5)  # 设置主刻度参数，线宽1.00，长度5
    ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)  # 设置次刻度参数，线宽0.75，长度2.5，标签大小10
    ax.set_xlim(0, 5)  # 设置x轴显示范围
    ax.set_ylim(0, 1)  # 设置y轴显示范围
    ax.text(0.0, 0.2, title, transform=ax.transAxes,  # 在Axes内部的相对位置(0.0, 0.2)处添加文本标题
            fontsize=14, fontname='Monospace', color='tab:blue')


fig = plt.figure(figsize=(8, 8), layout='constrained')  # 创建一个8x8大小的约束布局的图形对象
fig0, fig1, fig2 = fig.subfigures(3, height_ratios=[1.5, 1.5, 7.5])  # 将图形分割为3个子图，设置高度比例

fig0.suptitle('String Formatting', fontsize=16, x=0, ha='left')  # 设置fig0的总标题为'String Formatting'，左对齐
ax0 = fig0.subplots()  # 创建fig0的子图对象ax0

setup(ax0, title="'{x} km'")  # 调用setup函数设置ax0的参数和标题
ax0.xaxis.set_major_formatter('{x} km')  # 设置ax0的x轴主刻度格式化器为'{x} km'


fig1.suptitle('Function Formatting', fontsize=16, x=0, ha='left')  # 设置fig1的总标题为'Function Formatting'，左对齐
ax1 = fig1.subplots()  # 创建fig1的子图对象ax1

setup(ax1, title="def(x, pos): return str(x-5)")  # 调用setup函数设置ax1的参数和标题
ax1.xaxis.set_major_formatter(lambda x, pos: str(x-5))  # 设置ax1的x轴主刻度格式化器为lambda函数


fig2.suptitle('Formatter Object Formatting', fontsize=16, x=0, ha='left')  # 设置fig2的总标题为'Formatter Object Formatting'，左对齐
axs2 = fig2.subplots(7, 1)  # 创建7行1列的子图对象数组axs2

setup(axs2[0], title="NullFormatter()")  # 调用setup函数设置axs2[0]的参数和标题
axs2[0].xaxis.set_major_formatter(ticker.NullFormatter())  # 设置axs2[0]的x轴主刻度格式化器为NullFormatter()

setup(axs2[1], title="StrMethodFormatter('{x:.3f}')")  # 调用setup函数设置axs2[1]的参数和标题
axs2[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))  # 设置axs2[1]的x轴主刻度格式化器为StrMethodFormatter

setup(axs2[2], title="FormatStrFormatter('#%d')")  # 调用setup函数设置axs2[2]的参数和标题
axs2[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("#%d"))  # 设置axs2[2]的x轴主刻度格式化器为FormatStrFormatter
# 创建一个包含固定位置的标签的列表
positions = [0, 1, 2, 3, 4, 5]
# 创建对应位置的标签文本列表
labels = ['A', 'B', 'C', 'D', 'E', 'F']
# 设置第5个子图(axs2[4])的 x 轴主要定位器为固定定位器，使用预定义的位置列表
axs2[4].xaxis.set_major_locator(ticker.FixedLocator(positions))
# 设置第5个子图(axs2[4])的 x 轴主要格式化器为固定格式化器，使用预定义的标签列表
axs2[4].xaxis.set_major_formatter(ticker.FixedFormatter(labels))

# 对第6个子图(axs2[5])进行设置，设置标题为 "ScalarFormatter()"
setup(axs2[5], title="ScalarFormatter()")
# 设置第6个子图(axs2[5])的 x 轴主要格式化器为标量格式化器，使用数学文本
axs2[5].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# 对第7个子图(axs2[6])进行设置，设置标题为 "PercentFormatter(xmax=5)"
setup(axs2[6], title="PercentFormatter(xmax=5)")
# 设置第7个子图(axs2[6])的 x 轴主要格式化器为百分比格式化器，最大值为 5
axs2[6].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))

# 显示绘图
plt.show()
```