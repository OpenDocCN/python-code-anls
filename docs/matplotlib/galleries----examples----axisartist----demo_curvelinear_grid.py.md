# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_curvelinear_grid.py`

```py
"""
=====================
Curvilinear grid demo
=====================

Custom grid and ticklines.

This example demonstrates how to use
`~.grid_helper_curvelinear.GridHelperCurveLinear` to define custom grids and
ticklines by applying a transformation on the grid.  This can be used, as
shown on the second plot, to create polar projections in a rectangular box.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import Axes, HostAxes, angle_helper
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
    GridHelperCurveLinear


def curvelinear_test1(fig):
    """
    Grid for custom transform.
    """

    # 定义一个简单的变换函数
    def tr(x, y): return x, y - x
    # 定义变换函数的逆操作函数
    def inv_tr(x, y): return x, y + x

    # 创建一个曲线线性网格助手对象，使用定义的变换函数
    grid_helper = GridHelperCurveLinear((tr, inv_tr))

    # 在图形中添加子图 ax1，使用 Axes 类，指定网格助手为之前创建的 grid_helper
    ax1 = fig.add_subplot(1, 2, 1, axes_class=Axes, grid_helper=grid_helper)

    # 在 ax1 上绘制经过变换的数据点
    xx, yy = tr(np.array([3, 6]), np.array([5, 10]))
    ax1.plot(xx, yy)

    # 设置 ax1 的纵横比例为1
    ax1.set_aspect(1)
    # 设置 ax1 的 x 轴和 y 轴范围
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # 添加浮动轴 "t" 和 "t2" 到 ax1
    ax1.axis["t"] = ax1.new_floating_axis(0, 3)
    ax1.axis["t2"] = ax1.new_floating_axis(1, 7)

    # 在 ax1 上绘制网格线
    ax1.grid(True, zorder=0)


def curvelinear_test2(fig):
    """
    Polar projection, but in a rectangular box.
    """

    # 创建一个仿射变换对象，将角度转换为弧度，并与极坐标变换组合
    tr = Affine2D().scale(np.pi/180, 1) + PolarAxes.PolarTransform(
        apply_theta_transforms=False)

    # 创建一个极坐标投影的极限查找器对象
    extreme_finder = angle_helper.ExtremeFinderCycle(
        nx=20, ny=20,  # 每个方向的采样点数
        lon_cycle=360, lat_cycle=None,
        lon_minmax=None, lat_minmax=(0, np.inf),
    )

    # 创建一个用于度分秒坐标的网格定位器
    grid_locator1 = angle_helper.LocatorDMS(12)

    # 创建一个用于度分秒坐标的刻度标签格式化器
    tick_formatter1 = angle_helper.FormatterDMS()

    # 创建一个曲线线性网格助手对象，使用之前定义的变换和参数
    grid_helper = GridHelperCurveLinear(
        tr, extreme_finder=extreme_finder,
        grid_locator1=grid_locator1, tick_formatter1=tick_formatter1)

    # 在图形中添加子图 ax1，使用 HostAxes 类，指定网格助手为之前创建的 grid_helper
    ax1 = fig.add_subplot(
        1, 2, 2, axes_class=HostAxes, grid_helper=grid_helper)

    # 设置右侧和顶部轴的主刻度标签可见
    ax1.axis["right"].major_ticklabels.set_visible(True)
    ax1.axis["top"].major_ticklabels.set_visible(True)
    # 让右侧轴显示第一个坐标（角度）的刻度标签
    ax1.axis["right"].get_helper().nth_coord_ticks = 0
    # 让底部轴显示第二个坐标（半径）的刻度标签
    ax1.axis["bottom"].get_helper().nth_coord_ticks = 1

    # 设置轴的纵横比为1（保持正方形）
    ax1.set_aspect(1)
    # 设置 x 轴的显示范围
    ax1.set_xlim(-5, 12)
    # 设置 y 轴的显示范围
    ax1.set_ylim(-5, 10)

    # 在 ax1 上绘制网格，zorder=0 表示在底层
    ax1.grid(True, zorder=0)

    # 创建一个使用给定 transform 的 parasite Axes（寄生轴）
    ax2 = ax1.get_aux_axes(tr)
    # 注意，ax2.transData == tr + ax1.transData
    # 在 ax2 中绘制的任何内容将匹配 ax1 的刻度和网格
    ax2.plot(np.linspace(0, 30, 51), np.linspace(10, 10, 51), linewidth=2)

    # 在 ax2 中使用 pcolor 绘制颜色图，传入 x、y 轴坐标和数据
    ax2.pcolor(np.linspace(0, 90, 4), np.linspace(0, 10, 4),
               np.arange(9).reshape((3, 3)))
    # 在 ax2 中使用 contour 绘制轮廓线，传入 x、y 轴坐标和数据
    ax2.contour(np.linspace(0, 90, 4), np.linspace(0, 10, 4),
                np.arange(16).reshape((4, 4)), colors="k")
# 如果当前脚本被直接执行（而不是被作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 创建一个新的图形对象，设置其大小为 7x4 英寸
    fig = plt.figure(figsize=(7, 4))

    # 在创建的图形对象上执行 curvelinear_test1 函数，绘制第一个测试图形
    curvelinear_test1(fig)
    
    # 在创建的图形对象上执行 curvelinear_test2 函数，绘制第二个测试图形
    curvelinear_test2(fig)

    # 显示绘制的图形，阻止脚本的执行直到所有图形窗口被关闭
    plt.show()
```