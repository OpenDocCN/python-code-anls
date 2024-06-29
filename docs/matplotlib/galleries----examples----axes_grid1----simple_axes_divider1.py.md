# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\simple_axes_divider1.py`

```py
"""
=====================
Simple Axes Divider 1
=====================

See also :ref:`axes_grid`.
"""

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt

# 从mpl_toolkits.axes_grid1导入Divider和Size类
from mpl_toolkits.axes_grid1 import Divider, Size


def label_axes(ax, text):
    """Place a label at the center of an Axes, and remove the axis ticks."""
    # 在Axes的中心放置文本标签，同时移除轴刻度
    ax.text(.5, .5, text, transform=ax.transAxes,
            horizontalalignment="center", verticalalignment="center")
    # 设置轴参数，移除所有刻度和标签
    ax.tick_params(bottom=False, labelbottom=False,
                   left=False, labelleft=False)


# %%
# 固定Axes尺寸和固定内边距

# 创建一个尺寸为6x6英寸的图形对象
fig = plt.figure(figsize=(6, 6))
fig.suptitle("Fixed axes sizes, fixed paddings")

# 设置水平和垂直方向的尺寸列表（单位为英寸）
horiz = [Size.Fixed(1.), Size.Fixed(.5), Size.Fixed(1.5), Size.Fixed(.5)]
vert = [Size.Fixed(1.5), Size.Fixed(.5), Size.Fixed(1.)]

# 定义主要绘图区域的矩形范围
rect = (0.1, 0.1, 0.8, 0.8)
# 使用指定的horiz和vert尺寸列表创建一个网格划分对象
div = Divider(fig, rect, horiz, vert, aspect=False)

# 添加第一个Axes，使用div对象的新定位器
ax1 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0))
label_axes(ax1, "nx=0, ny=0")
# 添加第二个Axes，使用div对象的新定位器
ax2 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=2))
label_axes(ax2, "nx=0, ny=2")
# 添加第三个Axes，使用div对象的新定位器
ax3 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=2))
label_axes(ax3, "nx=2, ny=2")
# 添加第四个Axes，使用div对象的新定位器
ax4 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, nx1=4, ny=0))
label_axes(ax4, "nx=2, nx1=4, ny=0")

# %%
# Axes尺寸随图形大小缩放；固定内边距

# 创建另一个尺寸为6x6英寸的图形对象
fig = plt.figure(figsize=(6, 6))
fig.suptitle("Scalable axes sizes, fixed paddings")

# 设置不同的水平和垂直方向的尺寸列表（部分尺寸以图形大小比例缩放）
horiz = [Size.Scaled(1.5), Size.Fixed(.5), Size.Scaled(1.), Size.Scaled(.5)]
vert = [Size.Scaled(1.), Size.Fixed(.5), Size.Scaled(1.5)]

# 定义主要绘图区域的矩形范围
rect = (0.1, 0.1, 0.8, 0.8)
# 使用指定的horiz和vert尺寸列表创建一个网格划分对象
div = Divider(fig, rect, horiz, vert, aspect=False)

# 添加第一个Axes，使用div对象的新定位器
ax1 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0))
label_axes(ax1, "nx=0, ny=0")
# 添加第二个Axes，使用div对象的新定位器
ax2 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=2))
label_axes(ax2, "nx=0, ny=2")
# 添加第三个Axes，使用div对象的新定位器
ax3 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=2))
label_axes(ax3, "nx=2, ny=2")
# 添加第四个Axes，使用div对象的新定位器
ax4 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, nx1=4, ny=0))
label_axes(ax4, "nx=2, nx1=4, ny=0")

# 展示图形
plt.show()
```