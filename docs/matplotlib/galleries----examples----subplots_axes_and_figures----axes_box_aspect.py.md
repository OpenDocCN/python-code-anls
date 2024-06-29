# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axes_box_aspect.py`

```
"""
===============
Axes box aspect
===============

This demo shows how to set the aspect of an Axes box directly via
`~.Axes.set_box_aspect`. The box aspect is the ratio between Axes height
and Axes width in physical units, independent of the data limits.
This is useful to e.g. produce a square plot, independent of the data it
contains, or to have a usual plot with the same axes dimensions next to
an image plot with fixed (data-)aspect.

The following lists a few use cases for `~.Axes.set_box_aspect`.
"""

# %%
# A square Axes, independent of data
# ----------------------------------
#
# Produce a square Axes, no matter what the data limits are.

import matplotlib.pyplot as plt
import numpy as np

fig1, ax = plt.subplots()

ax.set_xlim(300, 400)  # 设置 x 轴数据范围
ax.set_box_aspect(1)   # 设置 Axes 的盒子宽高比为 1，使得绘制的图形为正方形

plt.show()

# %%
# Shared square Axes
# ------------------
#
# Produce shared subplots that are squared in size.
#
fig2, (ax, ax2) = plt.subplots(ncols=2, sharey=True)

ax.plot([1, 5], [0, 10])    # 在第一个子图中绘制一条线
ax2.plot([100, 500], [10, 15])  # 在第二个子图中绘制另一条线

ax.set_box_aspect(1)    # 设置第一个子图的盒子宽高比为 1
ax2.set_box_aspect(1)   # 设置第二个子图的盒子宽高比为 1

plt.show()

# %%
# Square twin Axes
# ----------------
#
# Produce a square Axes, with a twin Axes. The twinned Axes takes over the
# box aspect of the parent.
#

fig3, ax = plt.subplots()

ax2 = ax.twinx()

ax.plot([0, 10])    # 在第一个子图中绘制一条线
ax2.plot([12, 10])  # 在第二个子图中绘制另一条线

ax.set_box_aspect(1)   # 设置第一个子图的盒子宽高比为 1

plt.show()


# %%
# Normal plot next to image
# -------------------------
#
# When creating an image plot with fixed data aspect and the default
# ``adjustable="box"`` next to a normal plot, the Axes would be unequal in
# height. `~.Axes.set_box_aspect` provides an easy solution to that by allowing
# to have the normal plot's Axes use the images dimensions as box aspect.
#
# This example also shows that *constrained layout* interplays nicely with
# a fixed box aspect.

fig4, (ax, ax2) = plt.subplots(ncols=2, layout="constrained")

np.random.seed(19680801)  # Fixing random state for reproducibility
im = np.random.rand(16, 27)
ax.imshow(im)

ax2.plot([23, 45])
ax2.set_box_aspect(im.shape[0]/im.shape[1])  # 设置第二个子图的盒子宽高比为图像的高宽比

plt.show()

# %%
# Square joint/marginal plot
# --------------------------
#
# It may be desirable to show marginal distributions next to a plot of joint
# data. The following creates a square plot with the box aspect of the
# marginal Axes being equal to the width- and height-ratios of the gridspec.
# This ensures that all Axes align perfectly, independent on the size of the
# figure.

fig5, axs = plt.subplots(2, 2, sharex="col", sharey="row",
                         gridspec_kw=dict(height_ratios=[1, 3],
                                          width_ratios=[3, 1]))
axs[0, 1].set_visible(False)
axs[0, 0].set_box_aspect(1/3)   # 设置第一个子图的盒子宽高比为 1/3
axs[1, 0].set_box_aspect(1)     # 设置第三个子图的盒子宽高比为 1
axs[1, 1].set_box_aspect(3/1)   # 设置第四个子图的盒子宽高比为 3/1

np.random.seed(19680801)  # Fixing random state for reproducibility
x, y = np.random.randn(2, 400) * [[.5], [180]]
axs[1, 0].scatter(x, y)
axs[0, 0].hist(x)
axs[1, 1].hist(y, orientation="horizontal")

plt.show()

# %%
# Set data aspect with box aspect
# 创建一个新的图形和一个 Axes 对象
fig6, ax = plt.subplots()

# 在 Axes 上添加一个圆形补丁，圆心为 (5, 3)，半径为 1
ax.add_patch(plt.Circle((5, 3), 1))

# 设置数据方面的比例为 "equal"，使得内容保持等比例，即圆形仍然是圆形
ax.set_aspect("equal", adjustable="datalim")

# 设置盒子的长宽比为 0.5，使得盒子的宽度是其高度的一半
ax.set_box_aspect(0.5)

# 自动调整坐标轴范围以适应所有内容
ax.autoscale()

# 显示图形
plt.show()

# %%
# 多个子图的盒子长宽比
# ----------------------------
#
# 可以在初始化时将盒子长宽比传递给 Axes。以下示例创建了一个 2 行 3 列的子图网格，
# 所有 Axes 都是正方形的。

fig7, axs = plt.subplots(2, 3, subplot_kw=dict(box_aspect=1),
                         sharex=True, sharey=True, layout="constrained")

# 在每个子图上绘制不同的散点图
for i, ax in enumerate(axs.flat):
    ax.scatter(i % 3, -((i // 3) - 0.5)*200, c=[plt.cm.hsv(i / 6)], s=300)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    本示例展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.axes.Axes.set_box_aspect`
```