# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\viewlims.py`

```py
"""
========
Viewlims
========

Creates two identical panels.  Zooming in on the right panel will show
a rectangle in the first panel, denoting the zoomed region.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入所需的库
import functools  # 导入 functools 库用于创建偏函数

import matplotlib.pyplot as plt  # 导入 matplotlib 库的 pyplot 模块，并命名为 plt
import numpy as np  # 导入 numpy 库，并命名为 np

from matplotlib.patches import Rectangle  # 从 matplotlib.patches 模块中导入 Rectangle 类


# A class that will regenerate a fractal set as we zoom in, so that you
# can actually see the increasing detail.  A box in the left panel will show
# the area to which we are zoomed.
class MandelbrotDisplay:
    def __init__(self, h=500, w=500, niter=50, radius=2., power=2):
        self.height = h  # 设置 MandelbrotDisplay 类的高度属性
        self.width = w  # 设置 MandelbrotDisplay 类的宽度属性
        self.niter = niter  # 设置 MandelbrotDisplay 类的迭代次数属性
        self.radius = radius  # 设置 MandelbrotDisplay 类的半径属性
        self.power = power  # 设置 MandelbrotDisplay 类的幂属性

    def compute_image(self, xlim, ylim):
        self.x = np.linspace(*xlim, self.width)  # 生成 x 轴坐标数组
        self.y = np.linspace(*ylim, self.height).reshape(-1, 1)  # 生成 y 轴坐标数组
        c = self.x + 1.0j * self.y  # 生成复数数组 c
        threshold_time = np.zeros((self.height, self.width))  # 创建高度和宽度的零数组
        z = np.zeros(threshold_time.shape, dtype=complex)  # 创建复数数组 z
        mask = np.ones(threshold_time.shape, dtype=bool)  # 创建布尔类型的全为 True 的掩码数组
        for i in range(self.niter):
            z[mask] = z[mask]**self.power + c[mask]  # 更新 z 数组的值
            mask = (np.abs(z) < self.radius)  # 更新掩码数组
            threshold_time += mask  # 更新阈值时间数组
        return threshold_time  # 返回阈值时间数组

    def ax_update(self, ax):
        ax.set_autoscale_on(False)  # 关闭自动缩放
        self.width, self.height = ax.patch.get_window_extent().size.round().astype(int)  # 获取窗口大小
        ax.images[-1].set(data=self.compute_image(ax.get_xlim(), ax.get_ylim()),  # 更新图像数据和范围
                          extent=(*ax.get_xlim(), *ax.get_ylim()))
        ax.figure.canvas.draw_idle()  # 绘制更新


md = MandelbrotDisplay()  # 创建 MandelbrotDisplay 类的实例对象

fig1, (ax_full, ax_zoom) = plt.subplots(1, 2)  # 创建一个包含两个子图的 Figure 对象
ax_zoom.imshow([[0]], origin="lower")  # 在 ax_zoom 子图上显示空白图像
ax_zoom.set_title("Zoom here")  # 设置 ax_zoom 子图的标题

rect = Rectangle(
    [0, 0], 0, 0, facecolor="none", edgecolor="black", linewidth=1.0)  # 创建一个边框矩形对象
ax_full.add_patch(rect)  # 在 ax_full 子图上添加矩形对象


def update_rect(rect, ax):  # 定义一个函数用于更新矩形对象的位置和大小
    xlo, xhi = ax.get_xlim()  # 获取 x 轴的当前限制范围
    ylo, yhi = ax.get_ylim()  # 获取 y 轴的当前限制范围
    rect.set_bounds((xlo, ylo, xhi - xlo, yhi - ylo))  # 设置矩形的位置和大小
    ax.figure.canvas.draw_idle()  # 绘制更新


# Connect for changing the view limits.
ax_zoom.callbacks.connect("xlim_changed", functools.partial(update_rect, rect))  # 连接 x 轴限制更改事件
ax_zoom.callbacks.connect("ylim_changed", functools.partial(update_rect, rect))  # 连接 y 轴限制更改事件

ax_zoom.callbacks.connect("xlim_changed", md.ax_update)  # 连接 x 轴限制更改事件
ax_zoom.callbacks.connect("ylim_changed", md.ax_update)  # 连接 y 轴限制更改事件

# Initialize: trigger image computation by setting view limits; set colormap limits;
# 设置缩放子图的坐标范围
ax_zoom.set(xlim=(-2, .5), ylim=(-1.25, 1.25))
# 获取缩放子图上的第一个图像对象
im = ax_zoom.images[0]
# 调整第一个图像对象的颜色映射范围，使其范围与当前图像数组的最小值和最大值相同
ax_zoom.images[0].set(clim=(im.get_array().min(), im.get_array().max()))
# 在全视图中显示当前缩放子图的图像数据，使用原点在下方
ax_full.imshow(im.get_array(), extent=im.get_extent(), origin="lower")

# 显示所有绘制的图形
plt.show()
```