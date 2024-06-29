# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\align_ylabels.py`

```
"""
==============
Align y-labels
==============

Two methods are shown here, one using a short call to `.Figure.align_ylabels`
and the second a manual way to align the labels.

.. redirect-from:: /gallery/pyplots/align_ylabels
"""
# 导入 matplotlib 的 pyplot 模块，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并简称为 np
import numpy as np

# 定义一个创建图表的函数，参数为子图对象 axs
def make_plot(axs):
    # 定义一个字典，用于设置文本框样式
    box = dict(facecolor='yellow', pad=5, alpha=0.2)

    # 设定随机种子以保证可复现性
    np.random.seed(19680801)
    
    # 获取第一个子图对象 ax1
    ax1 = axs[0, 0]
    # 绘制随机数据折线图
    ax1.plot(2000*np.random.rand(10))
    # 设置子图标题
    ax1.set_title('ylabels not aligned')
    # 设置 y 轴标签及其样式
    ax1.set_ylabel('misaligned 1', bbox=box)
    # 设定 y 轴刻度范围
    ax1.set_ylim(0, 2000)

    # 获取第三个子图对象 ax3
    ax3 = axs[1, 0]
    # 设置 y 轴标签及其样式
    ax3.set_ylabel('misaligned 2', bbox=box)
    # 绘制随机数据折线图
    ax3.plot(np.random.rand(10))

    # 获取第二个子图对象 ax2
    ax2 = axs[0, 1]
    # 设置子图标题
    ax2.set_title('ylabels aligned')
    # 绘制随机数据折线图
    ax2.plot(2000*np.random.rand(10))
    # 设置 y 轴标签及其样式
    ax2.set_ylabel('aligned 1', bbox=box)
    # 设定 y 轴刻度范围
    ax2.set_ylim(0, 2000)

    # 获取第四个子图对象 ax4
    ax4 = axs[1, 1]
    # 绘制随机数据折线图
    ax4.plot(np.random.rand(10))
    # 设置 y 轴标签及其样式
    ax4.set_ylabel('aligned 2', bbox=box)

# 创建一个 2x2 的子图布局
fig, axs = plt.subplots(2, 2)
# 调整子图的布局参数
fig.subplots_adjust(left=0.2, wspace=0.6)
# 调用 make_plot 函数来绘制图表
make_plot(axs)

# 调用 `align_ylabels` 方法，使得第二列子图的 y 轴标签对齐
fig.align_ylabels(axs[:, 1])
# 显示图表
plt.show()

# %%
#
# .. seealso::
#     `.Figure.align_ylabels` and `.Figure.align_labels` for a direct method
#     of doing the same thing.
#     Also :doc:`/gallery/subplots_axes_and_figures/align_labels_demo`
#
#
# Or we can manually align the axis labels between subplots manually using the
# `~.Axis.set_label_coords` method of the y-axis object.  Note this requires
# we know a good offset value which is hardcoded.

# 创建一个新的 2x2 的子图布局
fig, axs = plt.subplots(2, 2)
# 调整子图的布局参数
fig.subplots_adjust(left=0.2, wspace=0.6)

# 调用 make_plot 函数来绘制图表
make_plot(axs)

# 设置手动对齐 y 轴标签的偏移值
labelx = -0.3  # axes coords

# 遍历每一行的第二列子图对象，手动设置 y 轴标签的位置坐标
for j in range(2):
    axs[j, 1].yaxis.set_label_coords(labelx, 0.5)

# 显示图表
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.align_ylabels`
#    - `matplotlib.axis.Axis.set_label_coords`
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_ylabel`
#    - `matplotlib.axes.Axes.set_ylim`
```