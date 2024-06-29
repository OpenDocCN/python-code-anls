# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\demo_tight_layout.py`

```
"""
===============================
Resizing Axes with tight layout
===============================

`~.Figure.tight_layout` attempts to resize subplots in a figure so that there
are no overlaps between Axes objects and labels on the Axes.

See :ref:`tight_layout_guide` for more details and
:ref:`constrainedlayout_guide` for an alternative.

"""

# 导入必要的库和模块
import itertools  # 导入 itertools 模块，用于循环迭代
import warnings   # 导入 warnings 模块，用于警告处理

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图

# 创建一个循环生成不同字体大小的迭代器
fontsizes = itertools.cycle([8, 16, 24, 32])


def example_plot(ax):
    """
    在给定的坐标轴上绘制示例图形，并设置不同的字体大小。

    Parameters:
    ax : matplotlib.axes.Axes
        要绘制图形的坐标轴对象
    """
    ax.plot([1, 2])  # 在坐标轴上绘制简单的折线图
    ax.set_xlabel('x-label', fontsize=next(fontsizes))  # 设置 x 轴标签和字体大小
    ax.set_ylabel('y-label', fontsize=next(fontsizes))  # 设置 y 轴标签和字体大小
    ax.set_title('Title', fontsize=next(fontsizes))     # 设置图表标题和字体大小


# %%

fig, ax = plt.subplots()  # 创建一个包含单个坐标轴的图形对象
example_plot(ax)          # 在第一个坐标轴上绘制示例图形
fig.tight_layout()        # 调整子图的布局，以避免重叠

# %%

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)  # 创建一个2x2的子图布局
example_plot(ax1)  # 在第一个子图上绘制示例图形
example_plot(ax2)  # 在第二个子图上绘制示例图形
example_plot(ax3)  # 在第三个子图上绘制示例图形
example_plot(ax4)  # 在第四个子图上绘制示例图形
fig.tight_layout()  # 调整子图的布局，以避免重叠

# %%

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)  # 创建一个包含两个垂直排列的子图的图形对象
example_plot(ax1)  # 在第一个子图上绘制示例图形
example_plot(ax2)  # 在第二个子图上绘制示例图形
fig.tight_layout()  # 调整子图的布局，以避免重叠

# %%

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)  # 创建一个包含两个水平排列的子图的图形对象
example_plot(ax1)  # 在第一个子图上绘制示例图形
example_plot(ax2)  # 在第二个子图上绘制示例图形
fig.tight_layout()  # 调整子图的布局，以避免重叠

# %%

fig, axs = plt.subplots(nrows=3, ncols=3)  # 创建一个3x3的子图布局
for ax in axs.flat:
    example_plot(ax)  # 在每个子图上绘制示例图形
fig.tight_layout()    # 调整子图的布局，以避免重叠

# %%

plt.figure()  # 创建一个新的图形对象
ax1 = plt.subplot(221)  # 创建一个2x2网格中的第一个子图
ax2 = plt.subplot(223)  # 创建一个2x2网格中的第三个子图
ax3 = plt.subplot(122)  # 创建一个1x2网格中的第二个子图
example_plot(ax1)  # 在第一个子图上绘制示例图形
example_plot(ax2)  # 在第二个子图上绘制示例图形
example_plot(ax3)  # 在第三个子图上绘制示例图形
plt.tight_layout()  # 调整子图的布局，以避免重叠

# %%

plt.figure()  # 创建一个新的图形对象
ax1 = plt.subplot2grid((3, 3), (0, 0))  # 在3x3网格的(0,0)位置创建子图
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)  # 在3x3网格的(0,1)位置创建跨两列的子图
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)  # 在3x3网格的(1,0)位置创建跨两列和两行的子图
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)  # 在3x3网格的(1,2)位置创建跨两行的子图
example_plot(ax1)  # 在第一个子图上绘制示例图形
example_plot(ax2)  # 在第二个子图上绘制示例图形
example_plot(ax3)  # 在第三个子图上绘制示例图形
example_plot(ax4)  # 在第四个子图上绘制示例图形
plt.tight_layout()  # 调整子图的布局，以避免重叠

# %%

fig = plt.figure()  # 创建一个新的图形对象

gs1 = fig.add_gridspec(3, 1)  # 添加一个3行1列的网格布局对象
ax1 = fig.add_subplot(gs1[0])  # 在第一个网格位置创建子图
ax2 = fig.add_subplot(gs1[1])  # 在第二个网格位置创建子图
ax3 = fig.add_subplot(gs1[2])  # 在第三个网格位置创建子图
example_plot(ax1)  # 在第一个子图上绘制示例图形
example_plot(ax2)  # 在第二个子图上绘制示例图形
example_plot(ax3)  # 在第三个子图上绘制示例图形
gs1.tight_layout(fig, rect=[None, None, 0.45, None])  # 调整第一个网格布局的子图布局，指定区域

gs2 = fig.add_gridspec(2, 1)  # 添加一个2行1列的网格布局对象
ax4 = fig.add_subplot(gs2[0])  # 在第一个网格位置创建子图
ax5 = fig.add_subplot(gs2[1])  # 在第二个网格位置创建子图
example_plot(ax4)  # 在第一个子图上绘制示例图形
example_plot(ax5)  # 在第二个子图上绘制示例图形
with warnings.catch_warnings():
    # 忽略关于子图布局无法处理的警告
    warnings.simplefilter("ignore", UserWarning)
    gs2.tight_layout(fig, rect=[0.45, None, None, None])  # 调整第二个网格布局的子图布局，指定区域

# 匹配两个网格布局的顶部和底部
top = min(gs1.top, gs2.top)
bottom = max(gs1.bottom, gs2.bottom)

gs1.update(top=top, bottom=bottom)  # 更新第一个网格布局的顶部和底部位置
gs2.update(top=top, bottom=bottom)  # 更新第二个网格布局的顶部和底部位置

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.tight_layout` /
#      `matplotlib.pyplot.tight_layout`
#    - `matplotlib.figure.Figure.add_gridspec`
#    - `matplotlib.figure.Figure.add_subplot`
#    - `matplotlib.pyplot.subplot2grid`
```