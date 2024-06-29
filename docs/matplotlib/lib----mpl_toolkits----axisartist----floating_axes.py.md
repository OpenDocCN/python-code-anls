# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\floating_axes.py`

```py
"""
An experimental support for curvilinear grid.
"""

# TODO :
# see if tick_iterator method can be simplified by reusing the parent method.

import functools

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path

from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory

# 导入自定义的模块
from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple

# 定义浮动坐标轴艺术家助手类
class FloatingAxisArtistHelper(
        grid_helper_curvelinear.FloatingAxisArtistHelper):
    pass

# 定义固定坐标轴艺术家助手类，继承自浮动坐标轴艺术家助手类
class FixedAxisArtistHelper(grid_helper_curvelinear.FloatingAxisArtistHelper):

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        # 调用父类的初始化方法
        lon1, lon2, lat1, lat2 = grid_helper.grid_finder.extreme_finder(*[None] * 5)
        # 根据方向确定坐标轴值和坐标轴方向
        value, nth_coord = _api.check_getitem(
            dict(left=(lon1, 0), right=(lon2, 0), bottom=(lat1, 1), top=(lat2, 1)),
            side=side)
        super().__init__(grid_helper, nth_coord, value, axis_direction=side)
        # 如果未指定 nth_coord_ticks，则默认为 nth_coord
        if nth_coord_ticks is None:
            nth_coord_ticks = nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.value = value
        self.grid_helper = grid_helper
        self._side = side

    def update_lim(self, axes):
        # 更新坐标轴辅助器的限制
        self.grid_helper.update_lim(axes)
        # 更新内部的网格信息
        self._grid_info = self.grid_helper._grid_info
    def get_tick_iterators(self, axes):
        """获取刻度迭代器

        根据指定的坐标轴和网格信息，生成用于绘制刻度的迭代器。

        Parameters:
        axes (matplotlib.axes.Axes): Matplotlib 中的坐标轴对象

        Returns:
        generator: 生成器对象，用于迭代每个刻度的位置、角度和标签
        """
        grid_finder = self.grid_helper.grid_finder

        lat_levs, lat_n, lat_factor = self._grid_info["lat_info"]
        yy0 = lat_levs / lat_factor

        lon_levs, lon_n, lon_factor = self._grid_info["lon_info"]
        xx0 = lon_levs / lon_factor

        extremes = self.grid_helper.grid_finder.extreme_finder(*[None] * 5)
        xmin, xmax = sorted(extremes[:2])
        ymin, ymax = sorted(extremes[2:])

        def trf_xy(x, y):
            """坐标转换函数

            将数据坐标转换为图形坐标。

            Parameters:
            x (numpy.ndarray): X 坐标数组
            y (numpy.ndarray): Y 坐标数组

            Returns:
            numpy.ndarray: 转换后的图形坐标数组
            """
            trf = grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack(np.broadcast_arrays(x, y))).T

        if self.nth_coord == 0:
            mask = (ymin <= yy0) & (yy0 <= ymax)
            (xx1, yy1), (dxx1, dyy1), (dxx2, dyy2) = \
                grid_helper_curvelinear._value_and_jacobian(
                    trf_xy, self.value, yy0[mask], (xmin, xmax), (ymin, ymax))
            labels = self._grid_info["lat_labels"]

        elif self.nth_coord == 1:
            mask = (xmin <= xx0) & (xx0 <= xmax)
            (xx1, yy1), (dxx2, dyy2), (dxx1, dyy1) = \
                grid_helper_curvelinear._value_and_jacobian(
                    trf_xy, xx0[mask], self.value, (xmin, xmax), (ymin, ymax))
            labels = self._grid_info["lon_labels"]

        labels = [l for l, m in zip(labels, mask) if m]

        angle_normal = np.arctan2(dyy1, dxx1)
        angle_tangent = np.arctan2(dyy2, dxx2)
        mm = (dyy1 == 0) & (dxx1 == 0)  # points with degenerate normal
        angle_normal[mm] = angle_tangent[mm] + np.pi / 2

        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
        in_01 = functools.partial(
            mpl.transforms._interval_contains_close, (0, 1))

        def f1():
            """生成刻度信息的生成器

            生成每个刻度的位置、角度和标签。

            Yields:
            list: 包含位置坐标、法线角度、切线角度和标签的列表
            """
            for x, y, normal, tangent, lab \
                    in zip(xx1, yy1, angle_normal, angle_tangent, labels):
                c2 = tick_to_axes.transform((x, y))
                if in_01(c2[0]) and in_01(c2[1]):
                    yield [x, y], *np.rad2deg([normal, tangent]), lab

        return f1(), iter([])

    def get_line(self, axes):
        """获取线条路径

        根据指定的坐标轴更新限制，并返回路径对象。

        Parameters:
        axes (matplotlib.axes.Axes): Matplotlib 中的坐标轴对象

        Returns:
        matplotlib.path.Path: 表示线条路径的 Path 对象
        """
        self.update_lim(axes)
        k, v = dict(left=("lon_lines0", 0),
                    right=("lon_lines0", 1),
                    bottom=("lat_lines0", 0),
                    top=("lat_lines0", 1))[self._side]
        xx, yy = self._grid_info[k][v]
        return Path(np.column_stack([xx, yy]))
class ExtremeFinderFixed(ExtremeFinderSimple):
    # 继承的文档字符串

    def __init__(self, extremes):
        """
        这个子类始终返回相同的边界框。

        Parameters
        ----------
        extremes : (float, float, float, float)
            此辅助程序始终返回的边界框。
        """
        self._extremes = extremes

    def __call__(self, transform_xy, x1, y1, x2, y2):
        # 继承的文档字符串
        return self._extremes


class GridHelperCurveLinear(grid_helper_curvelinear.GridHelperCurveLinear):

    def __init__(self, aux_trans, extremes,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        继承的文档字符串

        Parameters
        ----------
        aux_trans : Transform
            辅助转换对象。
        extremes : (float, float, float, float)
            边界框。
        grid_locator1 : Locator, optional
            网格定位器1。
        grid_locator2 : Locator, optional
            网格定位器2。
        tick_formatter1 : Formatter, optional
            刻度标签格式化器1。
        tick_formatter2 : Formatter, optional
            刻度标签格式化器2。
        """
        super().__init__(aux_trans,
                         extreme_finder=ExtremeFinderFixed(extremes),
                         grid_locator1=grid_locator1,
                         grid_locator2=grid_locator2,
                         tick_formatter1=tick_formatter1,
                         tick_formatter2=tick_formatter2)

    @_api.deprecated("3.8")
    def get_data_boundary(self, side):
        """
        返回 v=0, nth=1。
        
        Parameters
        ----------
        side : str
            边界的方向（'left', 'right', 'bottom', 'top'）。

        Returns
        -------
        tuple
            边界的经度和数值。
        """
        lon1, lon2, lat1, lat2 = self.grid_finder.extreme_finder(*[None] * 5)
        return dict(left=(lon1, 0),
                    right=(lon2, 0),
                    bottom=(lat1, 1),
                    top=(lat2, 1))[side]

    def new_fixed_axis(
            self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        """
        创建新的固定轴对象。

        Parameters
        ----------
        loc : str
            轴的位置（'top', 'bottom', 'left', 'right'）。
        nth_coord : int, optional
            第n个坐标轴。
        axis_direction : str, optional
            轴的方向。
        offset : float, optional
            偏移量。
        axes : Axes, optional
            轴对象。

        Returns
        -------
        AxisArtist
            新创建的轴艺术家对象。
        """
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        # 这不同于 FixedAxisArtistHelper 类，在 grid_helper_curvelinear.GridHelperCurveLinear.new_fixed_axis 中使用的类！
        helper = FixedAxisArtistHelper(
            self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        # 或许应该移动到基类？
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    # new_floating_axis 将继承 grid_helper 的边界。

    # def new_floating_axis(self, nth_coord, value, axes=None, axis_direction="bottom"):
    #     axis = super(GridHelperCurveLinear,
    #                  self).new_floating_axis(nth_coord,
    #                                          value, axes=axes,
    #                                          axis_direction=axis_direction)
    #     # set extreme values of the axis helper
    #     if nth_coord == 1:
    #         axis.get_helper().set_extremes(*self._extremes[:2])
    #     elif nth_coord == 0:
    #         axis.get_helper().set_extremes(*self._extremes[2:])
    #     return axis
    # 更新网格信息，用于指定的坐标范围(x1, y1)到(x2, y2)
    def _update_grid(self, x1, y1, x2, y2):
        # 如果网格信息为空，初始化为一个空字典
        if self._grid_info is None:
            self._grid_info = dict()

        # 将self._grid_info赋值给局部变量grid_info
        grid_info = self._grid_info

        # 从self.grid_finder中获取极值查找器对象
        grid_finder = self.grid_finder
        # 使用极值查找器对象的inv_transform_xy方法获取坐标变换后的极值
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        # 将经度的最小和最大值排序后存入lon_min和lon_max，纬度同理存入lat_min和lat_max
        lon_min, lon_max = sorted(extremes[:2])
        lat_min, lat_max = sorted(extremes[2:])
        # 将极值(lon_min, lon_max, lat_min, lat_max)存入grid_info字典中的"extremes"键
        grid_info["extremes"] = lon_min, lon_max, lat_min, lat_max  # extremes

        # 使用grid_finder对象的grid_locator1方法获取经度的刻度线信息
        lon_levs, lon_n, lon_factor = \
            grid_finder.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        # 使用grid_finder对象的grid_locator2方法获取纬度的刻度线信息
        lat_levs, lat_n, lat_factor = \
            grid_finder.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)

        # 将经度和纬度的刻度线信息存入grid_info字典中的"lon_info"和"lat_info"键
        grid_info["lon_info"] = lon_levs, lon_n, lon_factor
        grid_info["lat_info"] = lat_levs, lat_n, lat_factor

        # 使用grid_finder对象的_format_ticks方法获取格式化后的经度标签信息
        grid_info["lon_labels"] = grid_finder._format_ticks(
            1, "bottom", lon_factor, lon_levs)
        # 使用grid_finder对象的_format_ticks方法获取格式化后的纬度标签信息
        grid_info["lat_labels"] = grid_finder._format_ticks(
            2, "bottom", lat_factor, lat_levs)

        # 计算经度和纬度值
        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor

        # 使用grid_finder对象的_get_raw_grid_lines方法获取经度和纬度的原始网格线
        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(
            lon_values[(lon_min < lon_values) & (lon_values < lon_max)],
            lat_values[(lat_min < lat_values) & (lat_values < lat_max)],
            lon_min, lon_max, lat_min, lat_max)

        # 将经度和纬度的原始网格线存入grid_info字典中的"lon_lines"和"lat_lines"键
        grid_info["lon_lines"] = lon_lines
        grid_info["lat_lines"] = lat_lines

        # 使用grid_finder对象的_get_raw_grid_lines方法再次获取经度和纬度的原始网格线
        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(
            extremes[:2], extremes[2:], *extremes)

        # 将再次获取的经度和纬度的原始网格线存入grid_info字典中的"lon_lines0"和"lat_lines0"键
        grid_info["lon_lines0"] = lon_lines
        grid_info["lat_lines0"] = lat_lines

    # 获取网格线，根据指定的参数which和axis返回相应的网格线
    def get_gridlines(self, which="major", axis="both"):
        # 初始化空列表grid_lines用于存放网格线
        grid_lines = []
        # 如果axis为"both"或"x"，将经度的网格线添加到grid_lines中
        if axis in ["both", "x"]:
            grid_lines.extend(self._grid_info["lon_lines"])
        # 如果axis为"both"或"y"，将纬度的网格线添加到grid_lines中
        if axis in ["both", "y"]:
            grid_lines.extend(self._grid_info["lat_lines"])
        # 返回包含经度和/或纬度网格线的列表grid_lines
        return grid_lines
class FloatingAxesBase:

    def __init__(self, *args, grid_helper, **kwargs):
        # 检查 grid_helper 参数是否为 GridHelperCurveLinear 类型的实例
        _api.check_isinstance(GridHelperCurveLinear, grid_helper=grid_helper)
        # 调用父类的初始化方法，传递所有位置参数和关键字参数，包括 grid_helper
        super().__init__(*args, grid_helper=grid_helper, **kwargs)
        # 设置该轴的纵横比为1
        self.set_aspect(1.)

    def _gen_axes_patch(self):
        # 继承文档字符串
        # 获取网格助手对象，并使用其极值查找器获取坐标轴的范围
        x0, x1, y0, y1 = self.get_grid_helper().grid_finder.extreme_finder(*[None] * 5)
        # 创建一个多边形 patch，覆盖坐标轴的范围
        patch = mpatches.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        # 设置 patch 的插值步数，以增加绘制的平滑度
        patch.get_path()._interpolation_steps = 100
        return patch

    def clear(self):
        # 调用父类的 clear 方法
        super().clear()
        # 设置当前 patch 的变换，用于绘制网格线
        self.patch.set_transform(
            self.get_grid_helper().grid_finder.get_transform()
            + self.transData)
        # 原始 patch 不在绘制树中，仅用于裁剪目的
        orig_patch = super()._gen_axes_patch()
        orig_patch.set_figure(self.figure)
        orig_patch.set_transform(self.transAxes)
        # 设置当前 patch 的裁剪路径为原始 patch
        self.patch.set_clip_path(orig_patch)
        # 设置网格线的裁剪路径为原始 patch
        self.gridlines.set_clip_path(orig_patch)
        # 调整坐标轴的限制
        self.adjust_axes_lim()

    def adjust_axes_lim(self):
        # 获取当前 patch 的路径，并获取其边界框
        bbox = self.patch.get_path().get_extents(
            # 首先转换为像素坐标，然后再转换为父级数据坐标
            self.patch.get_transform() - self.transData)
        # 扩展边界框大小，以确保所有内容都可见
        bbox = bbox.expanded(1.02, 1.02)
        # 设置坐标轴的 x 轴和 y 轴限制
        self.set_xlim(bbox.xmin, bbox.xmax)
        self.set_ylim(bbox.ymin, bbox.ymax)


floatingaxes_class_factory = cbook._make_class_factory(FloatingAxesBase, "Floating{}")
# 创建一个工厂函数，生成带有指定主轴类的 FloatingAxes 类
FloatingAxes = floatingaxes_class_factory(host_axes_class_factory(axislines.Axes))
# FloatingSubplot 与 FloatingAxes 相同，用作别名
FloatingSubplot = FloatingAxes
```