# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\legend_demo.py`

```
"""
===========
Legend Demo
===========

There are many ways to create and customize legends in Matplotlib. Below
we'll show a few examples for how to do so.

First we'll show off how to make a legend for specific lines.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D

t1 = np.arange(0.0, 2.0, 0.1)
t2 = np.arange(0.0, 2.0, 0.01)

fig, ax = plt.subplots()

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1, = ax.plot(t2, np.exp(-t2))
l2, l3 = ax.plot(t2, np.sin(2 * np.pi * t2), '--o', t1, np.log(1 + t1), '.')
l4, = ax.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 's-.')

ax.legend((l2, l4), ('oscillatory', 'damped'), loc='upper right', shadow=True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
ax.set_title('Damped oscillation')
plt.show()


# %%
# Next we'll demonstrate plotting more complex labels.

x = np.linspace(0, 1)

fig, (ax0, ax1) = plt.subplots(2, 1)

# Plot the lines y=x**n for n=1..4.
for n in range(1, 5):
    ax0.plot(x, x**n, label=f"{n=}")
leg = ax0.legend(loc="upper left", bbox_to_anchor=[0, 1],
                 ncols=2, shadow=True, title="Legend", fancybox=True)
leg.get_title().set_color("red")

# Demonstrate some more complex labels.
ax1.plot(x, x**2, label="multi\nline")
half_pi = np.linspace(0, np.pi / 2)
ax1.plot(np.sin(half_pi), np.cos(half_pi), label=r"$\frac{1}{2}\pi$")
ax1.plot(x, 2**(x**2), label="$2^{x^2}$")
ax1.legend(shadow=True, fancybox=True)

plt.show()


# %%
# Here we attach legends to more complex plots.

fig, axs = plt.subplots(3, 1, layout="constrained")
top_ax, middle_ax, bottom_ax = axs

top_ax.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4, label="Bar 1",
           align="center")
top_ax.bar([0.5, 1.5, 2.5], [0.3, 0.2, 0.2], color="red", width=0.4,
           label="Bar 2", align="center")
top_ax.legend()

middle_ax.errorbar([0, 1, 2], [2, 3, 1], xerr=0.4, fmt="s", label="test 1")
middle_ax.errorbar([0, 1, 2], [3, 2, 4], yerr=0.3, fmt="o", label="test 2")
middle_ax.errorbar([0, 1, 2], [1, 1, 3], xerr=0.4, yerr=0.3, fmt="^",
                   label="test 3")
middle_ax.legend()

bottom_ax.stem([0.3, 1.5, 2.7], [1, 3.6, 2.7], label="stem test")
bottom_ax.legend()

plt.show()

# %%
# Now we'll showcase legend entries with more than one legend key.

fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')

# First plot: two legend keys for a single entry
p1 = ax1.scatter([1], [5], c='r', marker='s', s=100)
p2 = ax1.scatter([3], [2], c='b', marker='o', s=100)
# `plot` returns a list, but we want the handle - thus the comma on the left
p3, = ax1.plot([1, 5], [4, 4], 'm-d')

# Assign two of the handles to the same legend entry by putting them in a tuple
ax1.legend([(p1, p2)], ['scatter 1', 'scatter 2'], loc='upper right', shadow=True)
# 创建一个带有自定义处理程序映射的图例，该映射指定了如何处理多个图例条目，包括元组 (p1, p3)。
l = ax1.legend([(p1, p3), p2], ['two keys', 'one key'], scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

# 第二个图：在同一坐标系上绘制两个堆叠的条形图，并调整图例条目之间的间距。
x_left = [1, 2, 3]
y_pos = [1, 3, 2]
y_neg = [2, 1, 4]

# 绘制负值条形图，并设定特定标签和风格
rneg = ax2.bar(x_left, y_neg, width=0.5, color='w', hatch='///', label='-1')
# 绘制正值条形图，并设定特定标签和风格
rpos = ax2.bar(x_left, y_pos, width=0.5, color='k', label='+1')

# 使用特定的 `HandlerTuple` 处理程序映射来单独处理每个图例条目
l = ax2.legend([(rpos, rneg), (rneg, rpos)], ['pad!=0', 'pad=0'],
               handler_map={(rpos, rneg): HandlerTuple(ndivide=None),
                            (rneg, rpos): HandlerTuple(ndivide=None, pad=0.)})
plt.show()

# %%
# 最后，也可以编写自定义类来定义如何样式化图例。

class HandlerDashedLines(HandlerLineCollection):
    """
    自定义的 LineCollection 实例处理程序。
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 确定有多少条线
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # 将用于线条的垂直空间分为相等的部分
        ydata = np.full_like(xdata, height / (numlines + 1))
        # 对于每条线，创建位于正确位置的线条，并设置虚线样式
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # 设置颜色、虚线样式和线条宽度与 LineCollection 中的线条相同
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines

# 在新的图中绘制数据
x = np.linspace(0, 5, 100)

fig, ax = plt.subplots()
# 获取默认的颜色循环和线条样式
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]
styles = ['solid', 'dashed', 'dashed', 'dashed', 'solid']
# 为每个数据点绘制线条，使用特定的颜色和样式
for i, color, style in zip(range(5), colors, styles):
    # 在坐标系 ax 上绘制正弦波曲线，曲线的纵坐标根据变量 i 调整偏移量
    ax.plot(x, np.sin(x) - .1 * i, c=color, ls=style)
# 创建代理艺术家
# 创建一个包含一个线条的列表，线条的具体坐标不重要
line = [[(0, 0)]]
# 设置代理艺术家
# 使用指定的线条样式和颜色创建线条集合对象
lc = mcol.LineCollection(5 * line, linestyles=styles, colors=colors)
# 创建图例
# 在图中添加一个图例，显示为多条线，使用 HandlerDashedLines 处理程序来处理线条集合对象 lc
ax.legend([lc], ['multi-line'], handler_map={type(lc): HandlerDashedLines()},
          handlelength=2.5, handleheight=3)
# 显示图形
plt.show()
```