# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_axis_direction.py`

```
"""
===================
axis_direction demo
===================
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib.projections import PolarAxes  # 导入 PolarAxes 模块，用于极坐标系的定义
from matplotlib.transforms import Affine2D  # 导入 Affine2D 模块，用于仿射变换
import mpl_toolkits.axisartist as axisartist  # 导入 axisartist 模块，用于自定义坐标轴
import mpl_toolkits.axisartist.angle_helper as angle_helper  # 导入 angle_helper 模块，用于角度辅助计算
import mpl_toolkits.axisartist.grid_finder as grid_finder  # 导入 grid_finder 模块，用于网格线的定义
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
    GridHelperCurveLinear  # 导入 GridHelperCurveLinear 模块，用于定义曲线坐标系的网格辅助器


def setup_axes(fig, rect):
    """Polar projection, but in a rectangular box."""
    # 使用 GridHelperCurveLinear 创建曲线线性网格辅助器对象
    grid_helper = GridHelperCurveLinear(
        (
            Affine2D().scale(np.pi/180., 1.) +
            PolarAxes.PolarTransform(apply_theta_transforms=False)
        ),
        extreme_finder=angle_helper.ExtremeFinderCycle(
            20, 20,
            lon_cycle=360, lat_cycle=None,
            lon_minmax=None, lat_minmax=(0, np.inf),
        ),
        grid_locator1=angle_helper.LocatorDMS(12),  # 使用角度格式的定位器
        grid_locator2=grid_finder.MaxNLocator(5),  # 使用最大 N 定位器，设置为 5
        tick_formatter1=angle_helper.FormatterDMS(),  # 使用角度格式的刻度格式化器
    )
    # 将带有自定义网格辅助器的坐标系添加到图形中
    ax = fig.add_subplot(
        rect, axes_class=axisartist.Axes, grid_helper=grid_helper,
        aspect=1, xlim=(-5, 12), ylim=(-5, 10))  # 设置坐标系的纵横比、X 轴和 Y 轴的范围
    ax.axis[:].toggle(ticklabels=False)  # 关闭所有坐标轴的刻度标签
    ax.grid(color=".9")  # 设置网格线的颜色为浅灰色
    return ax


def add_floating_axis1(ax):
    # 在指定的轴上创建新的浮动坐标轴 "lat"
    ax.axis["lat"] = axis = ax.new_floating_axis(0, 30)
    axis.label.set_text(r"$\theta = 30^{\circ}$")  # 设置坐标轴标签文本为角度值
    axis.label.set_visible(True)  # 设置坐标轴标签可见
    return axis


def add_floating_axis2(ax):
    # 在指定的轴上创建新的浮动坐标轴 "lon"
    ax.axis["lon"] = axis = ax.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")  # 设置坐标轴标签文本为半径值
    axis.label.set_visible(True)  # 设置坐标轴标签可见
    return axis


fig = plt.figure(figsize=(8, 4), layout="constrained")  # 创建指定大小的图形对象

# 循环创建不同位置的坐标系，并添加浮动坐标轴 "lat"
for i, d in enumerate(["bottom", "left", "top", "right"]):
    ax = setup_axes(fig, rect=241+i)  # 设置坐标系的位置和尺寸
    axis = add_floating_axis1(ax)  # 添加浮动坐标轴 "lat"
    axis.set_axis_direction(d)  # 设置浮动坐标轴的方向
    ax.set(title=d)  # 设置坐标系标题

# 循环创建不同位置的坐标系，并添加浮动坐标轴 "lon"
for i, d in enumerate(["bottom", "left", "top", "right"]):
    ax = setup_axes(fig, rect=245+i)  # 设置坐标系的位置和尺寸
    axis = add_floating_axis2(ax)  # 添加浮动坐标轴 "lon"
    axis.set_axis_direction(d)  # 设置浮动坐标轴的方向
    ax.set(title=d)  # 设置坐标系标题

plt.show()  # 显示绘制的图形
```