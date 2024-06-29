# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_axes_divider.py`

```py
"""
============
Axes divider
============

Axes divider to calculate location of Axes and
create a divider for them using existing Axes instances.
"""

import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块

from matplotlib import cbook  # 导入matplotlib的cbook模块


def get_demo_image():
    z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")  # 从样本数据中获取一个15x15的数组
    return z, (-3, 4, -4, 3)  # 返回数组和其坐标范围


def demo_simple_image(ax):
    Z, extent = get_demo_image()  # 获取演示用的图像数据和其坐标范围

    im = ax.imshow(Z, extent=extent)  # 在给定的Axes上绘制图像
    cb = plt.colorbar(im)  # 添加颜色条到图像
    cb.ax.yaxis.set_tick_params(labelright=False)  # 设置颜色条的y轴刻度参数，右侧标签不显示


def demo_locatable_axes_hard(fig):
    from mpl_toolkits.axes_grid1 import Size, SubplotDivider  # 导入子模块Size和SubplotDivider

    divider = SubplotDivider(fig, 2, 2, 2, aspect=True)  # 创建一个SubplotDivider对象，用于划分Axes的位置

    # Axes for image
    ax = fig.add_subplot(axes_locator=divider.new_locator(nx=0, ny=0))  # 添加用于图像的Axes，使用新的定位器设置位置
    # Axes for colorbar
    ax_cb = fig.add_subplot(axes_locator=divider.new_locator(nx=2, ny=0))  # 添加用于颜色条的Axes，使用新的定位器设置位置

    divider.set_horizontal([
        Size.AxesX(ax),  # 主要Axes
        Size.Fixed(0.05),  # 填充，0.05英寸
        Size.Fixed(0.2),  # 颜色条，0.2英寸
    ])
    divider.set_vertical([Size.AxesY(ax)])  # 设置垂直方向，与主要Axes对齐

    Z, extent = get_demo_image()  # 获取演示用的图像数据和其坐标范围

    im = ax.imshow(Z, extent=extent)  # 在Axes上绘制图像
    plt.colorbar(im, cax=ax_cb)  # 添加颜色条到指定的Axes上
    ax_cb.yaxis.set_tick_params(labelright=False)  # 设置颜色条的y轴刻度参数，右侧标签不显示


def demo_locatable_axes_easy(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # 导入make_axes_locatable函数

    divider = make_axes_locatable(ax)  # 使用make_axes_locatable函数创建Axes的分隔器

    ax_cb = divider.append_axes("right", size="5%", pad=0.05)  # 在右侧附加一个尺寸为5%的Axes作为颜色条，设置填充为0.05
    fig = ax.get_figure()  # 获取当前Axes所属的Figure对象
    fig.add_axes(ax_cb)  # 将颜色条的Axes添加到Figure中

    Z, extent = get_demo_image()  # 获取演示用的图像数据和其坐标范围
    im = ax.imshow(Z, extent=extent)  # 在Axes上绘制图像

    plt.colorbar(im, cax=ax_cb)  # 添加颜色条到指定的Axes上
    ax_cb.yaxis.tick_right()  # 设置颜色条的y轴刻度在右侧显示
    ax_cb.yaxis.set_tick_params(labelright=False)  # 设置颜色条的y轴刻度参数，右侧标签不显示


def demo_images_side_by_side(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # 导入make_axes_locatable函数

    divider = make_axes_locatable(ax)  # 使用make_axes_locatable函数创建Axes的分隔器

    Z, extent = get_demo_image()  # 获取演示用的图像数据和其坐标范围
    ax2 = divider.append_axes("right", size="100%", pad=0.05)  # 在右侧附加一个尺寸为100%的Axes作为第二个图像，设置填充为0.05
    fig1 = ax.get_figure()  # 获取当前Axes所属的Figure对象
    fig1.add_axes(ax2)  # 将第二个图像的Axes添加到Figure中

    ax.imshow(Z, extent=extent)  # 在Axes上绘制第一个图像
    ax2.imshow(Z, extent=extent)  # 在第二个Axes上绘制第二个图像
    ax2.yaxis.set_tick_params(labelleft=False)  # 设置第二个图像的y轴刻度参数，左侧标签不显示


def demo():
    fig = plt.figure(figsize=(6, 6))  # 创建一个大小为6x6英寸的Figure对象

    # PLOT 1
    # simple image & colorbar
    ax = fig.add_subplot(2, 2, 1)  # 添加一个2x2网格中的第一个Axes对象
    demo_simple_image(ax)  # 在该Axes上演示简单的图像和颜色条

    # PLOT 2
    # image and colorbar with draw-time positioning -- a hard way
    demo_locatable_axes_hard(fig)  # 使用复杂的方式在Figure上演示图像和颜色条的位置安排

    # PLOT 3
    # image and colorbar with draw-time positioning -- an easy way
    ax = fig.add_subplot(2, 2, 3)  # 添加一个2x2网格中的第三个Axes对象
    demo_locatable_axes_easy(ax)  # 使用简单的方式在该Axes上演示图像和颜色条的位置安排

    # PLOT 4
    # two images side by side with fixed padding.
    ax = fig.add_subplot(2, 2, 4)  # 添加一个2x2网格中的第四个Axes对象
    demo_images_side_by_side(ax)  # 在该Axes上演示两幅图像并排显示

    plt.show()  # 显示所有绘制的图形


demo()  # 调用demo函数，运行示例代码
```