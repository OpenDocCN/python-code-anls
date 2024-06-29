# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\simple_axis_pad.py`

```py
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np
# 从 matplotlib.projections 中导入 PolarAxes 类
from matplotlib.projections import PolarAxes
# 从 matplotlib.transforms 中导入 Affine2D 类
from matplotlib.transforms import Affine2D
# 导入 mpl_toolkits.axisartist 库，并重命名为 axisartist
import mpl_toolkits.axisartist as axisartist
# 从 mpl_toolkits.axisartist.angle_helper 中导入 ExtremeFinderCycle 类
import mpl_toolkits.axisartist.angle_helper as angle_helper
# 从 mpl_toolkits.axisartist.grid_finder 中导入 MaxNLocator 类
import mpl_toolkits.axisartist.grid_finder as grid_finder
# 从 mpl_toolkits.axisartist.grid_helper_curvelinear 中导入 GridHelperCurveLinear 类
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

# 定义函数 setup_axes，用于设置极坐标投影在矩形框内
def setup_axes(fig, rect):
    """Polar projection, but in a rectangular box."""
    
    # 使用 Affine2D 缩放角度，创建极坐标变换，禁止应用 theta 转换
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform(
        apply_theta_transforms=False)

    # 设置角度查找器，週期为 360 度，纬度週期为无穷大
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0, np.inf),
                                                     )

    # 设置角度定位器，以度-分-秒方式显示
    grid_locator1 = angle_helper.LocatorDMS(12)
    # 设置网格定位器，最多显示 5 条线
    grid_locator2 = grid_finder.MaxNLocator(5)

    # 设置角度格式化器，以度-分-秒方式显示
    tick_formatter1 = angle_helper.FormatterDMS()

    # 创建曲线线性网格辅助对象
    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1
                                        )

    # 在图形上添加子图 ax1，使用 axisartist.Axes 类，传入网格辅助对象
    ax1 = fig.add_subplot(
        rect, axes_class=axisartist.Axes, grid_helper=grid_helper)
    # 隐藏所有坐标轴
    ax1.axis[:].set_visible(False)
    # 设置纵横比为 1
    ax1.set_aspect(1.)
    # 设置 x 轴范围
    ax1.set_xlim(-5, 12)
    # 设置 y 轴范围
    ax1.set_ylim(-5, 10)

    # 返回创建的子图对象 ax1
    return ax1


# 定义函数 add_floating_axis1，添加新的浮动坐标轴 lat 到 ax1
def add_floating_axis1(ax1):
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 30)
    axis.label.set_text(r"$\theta = 30^{\circ}$")
    axis.label.set_visible(True)

    # 返回添加的浮动坐标轴对象 axis
    return axis


# 定义函数 add_floating_axis2，添加新的浮动坐标轴 lon 到 ax1
def add_floating_axis2(ax1):
    ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")
    axis.label.set_visible(True)

    # 返回添加的浮动坐标轴对象 axis
    return axis


# 创建图形对象 fig，设置尺寸为 (9, 3)
fig = plt.figure(figsize=(9, 3.))
# 调整子图布局参数
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                    wspace=0.01, hspace=0.01)


# 定义函数 ann，用于在 ax1 上注释文本 d
def ann(ax1, d):
    if plt.rcParams["text.usetex"]:
        d = d.replace("_", r"\_")

    # 在图形上添加注释文本 d，位置为 (0.5, 1)，偏移量为 (5, -5)
    ax1.annotate(d, (0.5, 1), (5, -5),
                 xycoords="axes fraction", textcoords="offset points",
                 va="top", ha="center")


# 在子图 1 上设置极坐标轴
ax1 = setup_axes(fig, rect=141)
# 添加浮动坐标轴 lat 到 ax1
axis = add_floating_axis1(ax1)
# 在 ax1 上注释默认文本
ann(ax1, r"default")

# 在子图 2 上设置极坐标轴
ax1 = setup_axes(fig, rect=142)
# 添加浮动坐标轴 lat 到 ax1
axis = add_floating_axis1(ax1)
# 设置主刻度标签的间距为 10
axis.major_ticklabels.set_pad(10)
# 在 ax1 上注释文本，指示设置了刻度标签的间距
ann(ax1, r"ticklabels.set_pad(10)")

# 在子图 3 上设置极坐标轴
ax1 = setup_axes(fig, rect=143)
# 添加浮动坐标轴 lat 到 ax1
axis = add_floating_axis1(ax1)
# 设置坐标轴标签的间距为 20
axis.label.set_pad(20)
# 在 ax1 上注释文本，指示设置了坐标轴标签的间距
ann(ax1, r"label.set_pad(20)")

# 在子图 4 上设置极坐标轴
ax1 = setup_axes(fig, rect=144)
# 添加浮动坐标轴 lat 到 ax1
axis = add_floating_axis1(ax1)
# 设置主坐标轴的刻度线朝外
axis.major_ticks.set_tick_out(True)

# 在ax1上添加注释，内容是"ticks.set_tick_out(True)"
ann(ax1, "ticks.set_tick_out(True)")

# 显示绘图
plt.show()
```