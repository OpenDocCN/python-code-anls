# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_floating_axes.py`

```py
"""
==========================
``floating_axes`` features
==========================

Demonstration of features of the :mod:`.floating_axes` module:

* Using `~.axes.Axes.scatter` and `~.axes.Axes.bar` with changing the shape of
  the plot.
* Using `~.floating_axes.GridHelperCurveLinear` to rotate the plot and set the
  plot boundary.
* Using `~.Figure.add_subplot` to create a subplot using the return value from
  `~.floating_axes.GridHelperCurveLinear`.
* Making a sector plot by adding more features to
  `~.floating_axes.GridHelperCurveLinear`.
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算

from matplotlib.projections import PolarAxes  # 导入PolarAxes模块，用于极坐标系
from matplotlib.transforms import Affine2D  # 导入Affine2D模块，用于仿射变换
import mpl_toolkits.axisartist.angle_helper as angle_helper  # 导入角度助手模块
import mpl_toolkits.axisartist.floating_axes as floating_axes  # 导入floating_axes模块，用于浮动坐标轴
from mpl_toolkits.axisartist.grid_finder import (DictFormatter, FixedLocator,
                                                 MaxNLocator)  # 导入网格定位器相关模块

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设定随机种子，以便结果可复现


def setup_axes1(fig, rect):
    """
    A simple one.
    """
    tr = Affine2D().scale(2, 1).rotate_deg(30)  # 创建仿射变换对象tr，缩放2倍于x轴，1倍于y轴，并旋转30度

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(-0.5, 3.5, 0, 4),  # 使用tr创建GridHelperCurveLinear对象，设置极限值范围
        grid_locator1=MaxNLocator(nbins=4),  # 使用MaxNLocator创建网格定位器1，设定4个刻度
        grid_locator2=MaxNLocator(nbins=4))  # 使用MaxNLocator创建网格定位器2，设定4个刻度

    ax1 = fig.add_subplot(
        rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)  # 在fig中添加子图ax1，使用FloatingAxes类和grid_helper对象
    ax1.grid()  # 在ax1上添加网格线

    aux_ax = ax1.get_aux_axes(tr)  # 获取与tr相关的辅助坐标轴

    return ax1, aux_ax  # 返回主坐标轴ax1和辅助坐标轴aux_ax


def setup_axes2(fig, rect):
    """
    With custom locator and formatter.
    Note that the extreme values are swapped.
    """
    tr = PolarAxes.PolarTransform(apply_theta_transforms=False)  # 创建极坐标变换对象tr，不应用theta变换

    pi = np.pi  # 设定π值
    angle_ticks = [(0, r"$0$"),
                   (.25*pi, r"$\frac{1}{4}\pi$"),
                   (.5*pi, r"$\frac{1}{2}\pi$")]  # 设定角度刻度及其标签
    grid_locator1 = FixedLocator([v for v, s in angle_ticks])  # 使用FixedLocator创建网格定位器1
    tick_formatter1 = DictFormatter(dict(angle_ticks))  # 使用DictFormatter创建刻度标签格式化器1

    grid_locator2 = MaxNLocator(2)  # 使用MaxNLocator创建网格定位器2，设定2个刻度

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(.5*pi, 0, 2, 1),  # 使用tr创建GridHelperCurveLinear对象，设置极限值范围
        grid_locator1=grid_locator1,  # 指定网格定位器1
        grid_locator2=grid_locator2,  # 指定网格定位器2
        tick_formatter1=tick_formatter1,  # 指定刻度标签格式化器1
        tick_formatter2=None)  # 刻度标签格式化器2为空

    ax1 = fig.add_subplot(
        rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)  # 在fig中添加子图ax1，使用FloatingAxes类和grid_helper对象
    ax1.grid()  # 在ax1上添加网格线

    # create a parasite Axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)  # 获取与tr相关的辅助坐标轴

    aux_ax.patch = ax1.patch  # 设置aux_ax的裁剪路径与ax1相同
    ax1.patch.zorder = 0.9  # 但这会导致裁剪路径被绘制两次，可能覆盖其他图形。因此，将zorder降低一点以防止这种情况。

    return ax1, aux_ax  # 返回主坐标轴ax1和辅助坐标轴aux_ax


def setup_axes3(fig, rect):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(-95, 0)  # 创建平移变换对象tr_rotate，将坐标向左平移95个单位
    # 创建一个仿射变换对象，用来进行坐标变换，将角度转换为弧度
    tr_scale = Affine2D().scale(np.pi/180., 1.)

    # 创建一个仿射变换对象 tr，整合了旋转变换 tr_rotate、缩放变换 tr_scale 和极坐标变换
    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform(
        apply_theta_transforms=False)

    # 创建一个角度定位器，以时分秒格式定位
    grid_locator1 = angle_helper.LocatorHMS(4)
    # 创建一个角度格式化器，以时分秒格式显示
    tick_formatter1 = angle_helper.FormatterHMS()

    # 创建一个最大定位器，用于第二个轴
    grid_locator2 = MaxNLocator(3)

    # 指定角度范围（单位为度）
    ra0, ra1 = 8.*15, 14.*15
    # 指定径向范围
    cz0, cz1 = 0, 14000

    # 创建一个曲线线性网格助手对象 grid_helper
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=None)

    # 在图形 fig 中添加一个子图 ax1，使用 FloatingAxes 类型，并应用 grid_helper
    ax1 = fig.add_subplot(
        rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)

    # 调整左侧坐标轴方向为向下
    ax1.axis["left"].set_axis_direction("bottom")
    # 调整右侧坐标轴方向为向上
    ax1.axis["right"].set_axis_direction("top")

    # 隐藏底部坐标轴
    ax1.axis["bottom"].set_visible(False)
    # 调整顶部坐标轴方向为向下
    ax1.axis["top"].set_axis_direction("bottom")
    # 显示顶部坐标轴刻度和标签
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    # 调整顶部坐标轴主刻度标签方向为向上
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    # 调整顶部坐标轴标签方向为向上
    ax1.axis["top"].label.set_axis_direction("top")

    # 设置左侧坐标轴标签文本为 "cz [km$^{-1}$]"
    ax1.axis["left"].label.set_text(r"cz [km$^{-1}$]")
    # 设置顶部坐标轴标签文本为 "$\alpha_{1950}$"
    ax1.axis["top"].label.set_text(r"$\alpha_{1950}$")
    # 绘制网格
    ax1.grid()

    # 创建一个与主坐标轴 ax1 共享数据变换的寄生坐标轴 aux_ax，在 RA 和 cz 方向
    aux_ax = ax1.get_aux_axes(tr)

    # 设置 aux_ax 的 patch 与 ax1 的 patch 相同，用于裁剪路径
    aux_ax.patch = ax1.patch
    # 增加 ax1 的 patch 的绘制顺序，以便正确裁剪
    ax1.patch.zorder = 0.9

    # 返回主坐标轴 ax1 和寄生坐标轴 aux_ax
    return ax1, aux_ax
# %%
# 创建一个新的图形对象，设置尺寸为宽8英寸，高4英寸
fig = plt.figure(figsize=(8, 4))
# 调整子图之间的空白和整体图形的左右边界
fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

# 调用 setup_axes1 函数设置第一个子图 ax1 和辅助轴 aux_ax1
ax1, aux_ax1 = setup_axes1(fig, 131)
# 在 aux_ax1 上绘制柱状图，数据为 [3, 2, 1, 3]，位置为 [0, 1, 2, 3]
aux_ax1.bar([0, 1, 2, 3], [3, 2, 1, 3])

# 调用 setup_axes2 函数设置第二个子图 ax2 和辅助轴 aux_ax2
ax2, aux_ax2 = setup_axes2(fig, 132)
# 生成随机角度数据 theta 和半径数据 radius，并在 aux_ax2 上绘制散点图
theta = np.random.rand(10) * 0.5 * np.pi
radius = np.random.rand(10) + 1.0
aux_ax2.scatter(theta, radius)

# 调用 setup_axes3 函数设置第三个子图 ax3 和辅助轴 aux_ax3
ax3, aux_ax3 = setup_axes3(fig, 133)

# 生成随机角度数据 theta（以度为单位）和半径数据 radius，并在 aux_ax3 上绘制散点图
theta = (8 + np.random.rand(10) * (14 - 8)) * 15.  # 角度单位为度
radius = np.random.rand(10) * 14000.
aux_ax3.scatter(theta, radius)

# 显示图形
plt.show()
```