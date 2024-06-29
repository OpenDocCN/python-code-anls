# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\tests\test_grid_helper_curvelinear.py`

```py
import numpy as np  # 导入 NumPy 库，用于数值计算

import matplotlib.pyplot as plt  # 导入 Matplotlib 库的 pyplot 模块，用于绘图
from matplotlib.path import Path  # 导入 Matplotlib 库的 Path 类，用于处理路径
from matplotlib.projections import PolarAxes  # 导入 Matplotlib 库的 PolarAxes 投影
from matplotlib.ticker import FuncFormatter  # 导入 Matplotlib 库的 FuncFormatter 类，用于格式化刻度
from matplotlib.transforms import Affine2D, Transform  # 导入 Matplotlib 库的 Affine2D 和 Transform 类
from matplotlib.testing.decorators import image_comparison  # 导入 Matplotlib 测试工具的 image_comparison 函数

from mpl_toolkits.axisartist import SubplotHost  # 导入 Matplotlib 的 axisartist 模块中的 SubplotHost 类
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory  # 导入 axes_grid1 的 parasite_axes 模块中的 host_axes_class_factory 函数
from mpl_toolkits.axisartist import angle_helper  # 导入 axisartist 模块的 angle_helper 助手函数
from mpl_toolkits.axisartist.axislines import Axes  # 导入 axisartist 模块的 Axes 类
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
    GridHelperCurveLinear  # 导入 axisartist 模块的 GridHelperCurveLinear 类


@image_comparison(['custom_transform.png'], style='default', tol=0.2)
def test_custom_transform():
    class MyTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            """
            Resolution is the number of steps to interpolate between each input
            line segment to approximate its path in transformed space.
            """
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            x, y = ll.T
            return np.column_stack([x, y - x])

        transform_non_affine = transform

        def transform_path(self, path):
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        transform_path_non_affine = transform_path

        def inverted(self):
            return MyTransformInv(self._resolution)

    class MyTransformInv(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            x, y = ll.T
            return np.column_stack([x, y + x])

        def inverted(self):
            return MyTransform(self._resolution)

    fig = plt.figure()  # 创建一个新的图形对象

    SubplotHost = host_axes_class_factory(Axes)  # 使用 host_axes_class_factory 创建 SubplotHost 类对象

    tr = MyTransform(1)  # 创建 MyTransform 类的实例 tr，分辨率为 1
    grid_helper = GridHelperCurveLinear(tr)  # 使用 tr 创建 GridHelperCurveLinear 实例 grid_helper
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)  # 在图形 fig 上创建一个 SubplotHost 对象 ax1
    fig.add_subplot(ax1)  # 将 ax1 添加到 fig 图形中

    ax2 = ax1.get_aux_axes(tr, viewlim_mode="equal")  # 在 ax1 上获取辅助坐标轴 ax2，使用 tr 作为变换，并设置视图限制模式为 "equal"
    ax2.plot([3, 6], [5.0, 10.])  # 在 ax2 上绘制数据点

    ax1.set_aspect(1.)  # 设置 ax1 的纵横比为 1
    ax1.set_xlim(0, 10)  # 设置 ax1 的 X 轴范围为 0 到 10
    ax1.set_ylim(0, 10)  # 设置 ax1 的 Y 轴范围为 0 到 10

    ax1.grid(True)  # 在 ax1 上显示网格


@image_comparison(['polar_box.png'], style='default', tol=0.04)
def test_polar_box():
    fig = plt.figure(figsize=(5, 5))  # 创建一个大小为 5x5 的图形对象

    # PolarAxes.PolarTransform 需要弧度值，但我们想要的是以度为单位的坐标系
    tr = (Affine2D().scale(np.pi / 180., 1.) +
          PolarAxes.PolarTransform(apply_theta_transforms=False))
    # 创建一个变换 tr，首先将角度转换为弧度，然后应用极坐标变换，不应用 theta 转换

    # 极坐标投影，涉及循环，并且在其坐标中有限制，需要特殊方法来查找极值
    # (在视图内的坐标的最小值和最大值)。
    # 创建一个角度辅助查找器对象，用于帮助定位极值点
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0, np.inf))

    # 创建一个网格辅助对象，使用曲线线性网格帮助器
    grid_helper = GridHelperCurveLinear(
        tr,  # 使用指定的变换对象来构建网格帮助器
        extreme_finder=extreme_finder,  # 使用前面创建的极值查找器对象
        grid_locator1=angle_helper.LocatorDMS(12),  # 使用角度的DMS定位器来定位网格
        tick_formatter1=angle_helper.FormatterDMS(),  # 使用角度的DMS格式化器来格式化刻度标签
        tick_formatter2=FuncFormatter(lambda x, p: "eight" if x == 8 else f"{int(x)}"),  # 自定义刻度标签的格式化器
    )

    # 在图中创建一个子图主机对象，使用上述网格帮助器
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    # 设置右边轴的主刻度标签可见
    ax1.axis["right"].major_ticklabels.set_visible(True)
    # 设置顶部轴的主刻度标签可见
    ax1.axis["top"].major_ticklabels.set_visible(True)

    # 设置右边轴显示第一个坐标（角度）的刻度标签
    ax1.axis["right"].get_helper().nth_coord_ticks = 0
    # 设置底部轴显示第二个坐标（半径）的刻度标签
    ax1.axis["bottom"].get_helper().nth_coord_ticks = 1

    # 将创建的子图主机对象添加到图形中
    fig.add_subplot(ax1)

    # 创建一个新的浮动轴对象“lat”，并将其添加到ax1上
    ax1.axis["lat"] = axis = grid_helper.new_floating_axis(0, 45, axes=ax1)
    axis.label.set_text("Test")  # 设置轴标签文本为"Test"
    axis.label.set_visible(True)  # 设置轴标签可见
    axis.get_helper().set_extremes(2, 12)  # 设置轴辅助对象的极限值范围为2到12

    # 创建一个新的浮动轴对象“lon”，并将其添加到ax1上
    ax1.axis["lon"] = axis = grid_helper.new_floating_axis(1, 6, axes=ax1)
    axis.label.set_text("Test 2")  # 设置轴标签文本为"Test 2"
    axis.get_helper().set_extremes(-180, 90)  # 设置轴辅助对象的极限值范围为-180到90

    # 创建一个依附轴对象ax2，使用给定的变换方式“tr”
    ax2 = ax1.get_aux_axes(tr, viewlim_mode="equal")
    assert ax2.transData == tr + ax1.transData  # 确保ax2的数据变换与ax1相匹配
    # 在ax2中绘制一条曲线，其刻度和网格将与ax1匹配
    ax2.plot(np.linspace(0, 30, 50), np.linspace(10, 10, 50))

    # 设置ax1的纵横比为1.0
    ax1.set_aspect(1.)
    # 设置ax1的X轴范围为-5到12
    ax1.set_xlim(-5, 12)
    # 设置ax1的Y轴范围为-5到10
    ax1.set_ylim(-5, 10)

    # 在ax1上显示网格
    ax1.grid(True)
# Remove tol & kerning_factor when this test image is regenerated.
# 当重新生成此测试图像时，移除 tol 和 kerning_factor 参数

@image_comparison(['axis_direction.png'], style='default', tol=0.13)
# 通过图像比较验证生成的图像与参考图像 'axis_direction.png' 在默认样式下是否一致，容忍度为 0.13

def test_axis_direction():
    # 设置文本的字符间距因子为 6
    plt.rcParams['text.kerning_factor'] = 6

    # 创建一个大小为 5x5 的图形
    fig = plt.figure(figsize=(5, 5))

    # PolarAxes.PolarTransform 使用弧度制，但我们希望使用度数制作我们的坐标系
    tr = (Affine2D().scale(np.pi / 180., 1.) +
          PolarAxes.PolarTransform(apply_theta_transforms=False))
    # 构建坐标转换对象 tr，将角度转换为弧度，并禁用应用 theta 变换

    # 极坐标投影，涉及循环，并且在其坐标中有限制，需要特殊方法来找到视图内的极值点
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0, np.inf),
                                                     )
    # 极坐标的极值点查找器，lon_cycle 表示经度循环周期为 360 度，lat_minmax 设定纬度的最小值和最大值为 0 和正无穷

    grid_locator1 = angle_helper.LocatorDMS(12)
    tick_formatter1 = angle_helper.FormatterDMS()
    # 使用 DMS 格式化的刻度定位器和格式化器

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1)
    # 创建曲线线性网格助手，使用给定的转换 tr，极值点查找器，定位器和格式化器

    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)
    # 创建子图 ax1，并指定使用 grid_helper 作为网格助手

    for axis in ax1.axis.values():
        axis.set_visible(False)
    # 隐藏 ax1 中所有坐标轴

    fig.add_subplot(ax1)
    # 将 ax1 添加到图形中

    # 创建新的浮动坐标轴 lat1，位于水平位置 0 到 130 之间，使用左侧方向
    ax1.axis["lat1"] = axis = grid_helper.new_floating_axis(
        0, 130,
        axes=ax1, axis_direction="left")
    axis.label.set_text("Test")
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(0.001, 10)
    # 设置坐标轴标签文本为 "Test"，并设置可见性为 True，设置极值范围为 0.001 到 10

    # 创建新的浮动坐标轴 lat2，位于水平位置 0 到 50 之间，使用右侧方向
    ax1.axis["lat2"] = axis = grid_helper.new_floating_axis(
        0, 50,
        axes=ax1, axis_direction="right")
    axis.label.set_text("Test")
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(0.001, 10)
    # 设置坐标轴标签文本为 "Test"，并设置可见性为 True，设置极值范围为 0.001 到 10

    # 创建新的浮动坐标轴 lon，位于垂直位置 1 到 10 之间，使用底部方向
    ax1.axis["lon"] = axis = grid_helper.new_floating_axis(
        1, 10,
        axes=ax1, axis_direction="bottom")
    axis.label.set_text("Test 2")
    axis.get_helper().set_extremes(50, 130)
    axis.major_ticklabels.set_axis_direction("top")
    axis.label.set_axis_direction("top")
    # 设置坐标轴标签文本为 "Test 2"，设置极值范围为 50 到 130，设置主要刻度标签方向为顶部，设置标签方向为顶部

    grid_helper.grid_finder.grid_locator1.set_params(nbins=5)
    grid_helper.grid_finder.grid_locator2.set_params(nbins=5)
    # 设置网格查找器的参数，第一个轴的分 bin 数量为 5，第二个轴的分 bin 数量为 5

    ax1.set_aspect(1.)
    # 设置 ax1 的纵横比为 1
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-4, 12)
    # 设置 ax1 的 x 轴和 y 轴的显示范围

    ax1.grid(True)
    # 在 ax1 上显示网格
```