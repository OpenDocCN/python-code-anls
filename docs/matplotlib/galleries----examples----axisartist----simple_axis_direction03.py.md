# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\simple_axis_direction03.py`

```py
"""
==========================================
Simple axis tick label and tick directions
==========================================

First subplot moves the tick labels to inside the spines.
Second subplot moves the ticks to inside the spines.
These effects can be obtained for a standard Axes by `~.Axes.tick_params`.
"""

# 导入 matplotlib 库，并导入 mpl_toolkits.axisartist 模块
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

# 定义一个函数 setup_axes，用于设置带有特定坐标轴属性的子图
def setup_axes(fig, pos):
    # 在指定位置添加一个带有 axisartist.Axes 类的子图
    ax = fig.add_subplot(pos, axes_class=axisartist.Axes)
    # 设置 y 轴和 x 轴的刻度位置
    ax.set_yticks([0.2, 0.8])
    ax.set_xticks([0.2, 0.8])
    return ax

# 创建一个新的图形对象，设置大小为 5x2 英寸，并调整子图之间的空白和底部的位置
fig = plt.figure(figsize=(5, 2))
fig.subplots_adjust(wspace=0.4, bottom=0.3)

# 在第一个子图位置创建一个带有特定坐标轴属性的子图 ax1
ax1 = setup_axes(fig, 121)
ax1.set_xlabel("ax1 X-label")  # 设置 ax1 的 x 轴标签
ax1.set_ylabel("ax1 Y-label")  # 设置 ax1 的 y 轴标签

# 将 ax1 的刻度标签方向反转，使其位于坐标轴内侧
ax1.axis[:].invert_ticklabel_direction()

# 在第二个子图位置创建一个带有特定坐标轴属性的子图 ax2
ax2 = setup_axes(fig, 122)
ax2.set_xlabel("ax2 X-label")  # 设置 ax2 的 x 轴标签
ax2.set_ylabel("ax2 Y-label")  # 设置 ax2 的 y 轴标签

# 将 ax2 的主刻度线设置为不向外延伸，使其位于坐标轴内侧
ax2.axis[:].major_ticks.set_tick_out(False)

# 显示绘图结果
plt.show()
```