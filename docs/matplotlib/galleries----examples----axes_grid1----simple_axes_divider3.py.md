# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\simple_axes_divider3.py`

```py
"""
=====================
Simple axes divider 3
=====================

See also :ref:`axes_grid`.
"""

# 导入 matplotlib 的 pyplot 模块，并简称为 plt
import matplotlib.pyplot as plt

# 从 mpl_toolkits.axes_grid1 中导入 Divider 类和 axes_size 模块中的 Size 类
from mpl_toolkits.axes_grid1 import Divider
import mpl_toolkits.axes_grid1.axes_size as Size

# 创建一个大小为 (5.5, 4) 的新图形对象
fig = plt.figure(figsize=(5.5, 4))

# 指定四个子图的位置和标签，存放在 ax 列表中
rect = (0.1, 0.1, 0.8, 0.8)
ax = [fig.add_axes(rect, label="%d" % i) for i in range(4)]

# 水平方向上的布局设置，包括左边第一个子图、固定间隔 0.5、右边第二个子图
horiz = [Size.AxesX(ax[0]), Size.Fixed(.5), Size.AxesX(ax[1])]

# 垂直方向上的布局设置，包括上面第一个子图、固定间隔 0.5、下面第二个子图
vert = [Size.AxesY(ax[0]), Size.Fixed(.5), Size.AxesY(ax[2])]

# 根据水平和垂直布局创建一个 Divider 对象，将整个图形分割为网格
divider = Divider(fig, rect, horiz, vert, aspect=False)

# 分别设置四个子图的位置定位器
ax[0].set_axes_locator(divider.new_locator(nx=0, ny=0))
ax[1].set_axes_locator(divider.new_locator(nx=2, ny=0))
ax[2].set_axes_locator(divider.new_locator(nx=0, ny=2))
ax[3].set_axes_locator(divider.new_locator(nx=2, ny=2))

# 设置第一个子图的 x 轴范围为 0 到 2
ax[0].set_xlim(0, 2)
# 设置第二个子图的 x 轴范围为 0 到 1
ax[1].set_xlim(0, 1)

# 设置第一个子图的 y 轴范围为 0 到 1
ax[0].set_ylim(0, 1)
# 设置第三个子图的 y 轴范围为 0 到 2
ax[2].set_ylim(0, 2)

# 设置 Divider 对象的纵横比为 1
divider.set_aspect(1.)

# 对所有子图禁用标签的显示（无标签）
for ax1 in ax:
    ax1.tick_params(labelbottom=False, labelleft=False)

# 显示图形
plt.show()
```