# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\radar_chart.py`

```
"""
======================================
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider or star chart [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
`matplotlib.axis` to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axis.

.. [1] https://en.wikipedia.org/wiki/Radar_chart
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.patches import Circle, RegularPolygon  # 导入 Circle 和 RegularPolygon 类，用于绘制图形
from matplotlib.path import Path  # 导入 Path 类，用于定义路径
from matplotlib.projections import register_projection  # 导入 register_projection 函数，用于注册投影
from matplotlib.projections.polar import PolarAxes  # 导入 PolarAxes 类，用于极坐标轴的定义
from matplotlib.spines import Spine  # 导入 Spine 类，用于定义坐标轴的脊柱线
from matplotlib.transforms import Affine2D  # 导入 Affine2D 类，用于仿射变换


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)  # 计算均匀分布的角度

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)
    class RadarAxes(PolarAxes):
        # 定义雷达图的自定义坐标系类，继承自极坐标系
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            # 初始化函数，调用父类的初始化方法
            super().__init__(*args, **kwargs)
            # 设置极坐标系的起始角度为北方向
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            # 覆盖父类的填充方法，使得默认情况下线条是封闭的
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            # 覆盖父类的绘图方法，使得默认情况下线条是封闭的
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            # 将线条首尾相接闭合的方法
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            # 如果首尾点不重合，则添加首点使线条闭合
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            # 设置变量标签的方法，根据传入的标签设置极坐标系的刻度标签
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # 生成极坐标系的背景图形
            # Axes patch 必须在 (0.5, 0.5) 处居中，并且半径为 0.5
            # 在坐标轴上
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            # 生成极坐标系的脊柱线
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # 脊柱类型必须是 'left'/'right'/'top'/'bottom'/'circle'
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon 生成半径为 1 的中心在 (0, 0) 的多边形，
                # 我们需要的是半径为 0.5 中心在 (0.5, 0.5) 的多边形，在坐标轴上
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
# 导入必要的绘图库
import matplotlib.pyplot as plt

# 定义雷达图工厂函数，并设置多边形框架
def radar_factory(num_vars, frame='circle'):
    # 这里应该是雷达图工厂的具体实现，但在注释中没有给出详细说明

# 从 example_data 函数获取数据
data = example_data()

# 弹出数据的第一项，即化学物质名称列表
spoke_labels = data.pop(0)

# 创建一个包含4个子图的画布，每个子图使用雷达投影
fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                        subplot_kw=dict(projection='radar'))

# 调整子图之间的间距和顶部/底部的空白区域
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

# 定义用于不同情景数据显示的颜色列表
colors = ['b', 'r', 'g', 'm', 'y']

# 绘制来自示例数据的四种情景的雷达图，每种情景在一个独立的子图中
    # 遍历 axs.flat 中的每个子图 ax，并同时迭代 data 中的元组 (title, case_data)
    for ax, (title, case_data) in zip(axs.flat, data):
        # 设置极坐标图 ax 的径向网格线位置
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        # 设置 ax 的标题，包括字体粗细、大小、位置、水平和垂直对齐方式
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        # 遍历 case_data 中的数据 d 和对应的颜色 colors，绘制极坐标图上的线条和填充区域
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        # 设置极坐标图的变量标签
        ax.set_varlabels(spoke_labels)

    # 在左上角的子图 axs[0, 0] 上添加图例，位置相对于子图的绝对位置
    labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')

    # 在图形中添加主标题，包括水平对齐方式、颜色、字体粗细和大小
    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    # 显示整个图形
    plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.path`: Provides path construction and manipulation functions.
#    - `matplotlib.path.Path`: Represents a geometric path in 2D space.
#    - `matplotlib.spines`: Manages the appearance and behavior of axis spines.
#    - `matplotlib.spines.Spine`: Represents a single spine on a figure.
#    - `matplotlib.projections`: Deals with different types of plot projections.
#    - `matplotlib.projections.polar`: Specific utilities for polar plots.
#    - `matplotlib.projections.polar.PolarAxes`: Represents polar axes for plotting.
#    - `matplotlib.projections.register_projection`: Registers a custom projection type with Matplotlib.
```