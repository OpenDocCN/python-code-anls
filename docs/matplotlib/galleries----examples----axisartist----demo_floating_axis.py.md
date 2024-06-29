# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_floating_axis.py`

```
"""
==================
floating_axis demo
==================

Axis within rectangular frame.

The following code demonstrates how to put a floating polar curve within a
rectangular box. In order to get a better sense of polar curves, please look at
:doc:`/gallery/axisartist/demo_curvelinear_grid`.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.projections import PolarAxes  # 导入极坐标轴对象
from matplotlib.transforms import Affine2D  # 导入仿射变换对象
from mpl_toolkits.axisartist import GridHelperCurveLinear, HostAxes  # 导入辅助坐标系对象和主坐标系对象
import mpl_toolkits.axisartist.angle_helper as angle_helper  # 导入角度辅助函数


def curvelinear_test2(fig):
    """Polar projection, but in a rectangular box."""
    # see demo_curvelinear_grid.py for details
    # 创建一个仿射变换对象，将角度转换为弧度，但不应用额外的角度变换
    tr = Affine2D().scale(np.pi / 180., 1.) + PolarAxes.PolarTransform(
        apply_theta_transforms=False)

    # 使用角度辅助函数创建一个极值查找器对象
    extreme_finder = angle_helper.ExtremeFinderCycle(20,
                                                     20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0, np.inf),
                                                     )

    # 使用角度辅助函数创建一个坐标轴定位器对象
    grid_locator1 = angle_helper.LocatorDMS(12)

    # 使用角度辅助函数创建一个坐标轴格式化器对象
    tick_formatter1 = angle_helper.FormatterDMS()

    # 创建一个曲线线性网格辅助对象
    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )

    # 在图形中添加一个基于网格辅助对象的主坐标轴对象
    ax1 = fig.add_subplot(axes_class=HostAxes, grid_helper=grid_helper)

    # Now creates floating axis

    # 创建一个浮动坐标轴，其第一个坐标（theta）固定在60度处
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 60)
    axis.label.set_text(r"$\theta = 60^{\circ}$")  # 设置坐标轴标签文本
    axis.label.set_visible(True)  # 设置坐标轴标签可见

    # 创建一个浮动坐标轴，其第二个坐标（r）固定在6处
    ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")  # 设置坐标轴标签文本

    ax1.set_aspect(1.)  # 设置坐标轴纵横比为1（保持正圆形）
    ax1.set_xlim(-5, 12)  # 设置x轴显示范围
    ax1.set_ylim(-5, 10)  # 设置y轴显示范围

    ax1.grid(True)  # 显示网格


fig = plt.figure(figsize=(5, 5))  # 创建一个大小为5x5英寸的图形对象
curvelinear_test2(fig)  # 调用函数生成图形
plt.show()  # 显示图形
```