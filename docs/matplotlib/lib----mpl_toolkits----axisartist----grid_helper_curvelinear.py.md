# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\grid_helper_curvelinear.py`

```py
"""
An experimental support for curvilinear grid.
"""

import functools  # 导入 functools 模块

import numpy as np  # 导入 NumPy 库并重命名为 np

import matplotlib as mpl  # 导入 Matplotlib 库并重命名为 mpl
from matplotlib import _api  # 导入 Matplotlib 内部 API
from matplotlib.path import Path  # 导入 Path 类
from matplotlib.transforms import Affine2D, IdentityTransform  # 导入 Affine2D 和 IdentityTransform 变换
from .axislines import (
    _FixedAxisArtistHelperBase, _FloatingAxisArtistHelperBase, GridHelperBase)  # 从本地模块导入相关类
from .axis_artist import AxisArtist  # 从本地模块导入 AxisArtist 类
from .grid_finder import GridFinder  # 从本地模块导入 GridFinder 类


def _value_and_jacobian(func, xs, ys, xlims, ylims):
    """
    Compute *func* and its derivatives along x and y at positions *xs*, *ys*,
    while ensuring that finite difference calculations don't try to evaluate
    values outside of *xlims*, *ylims*.
    """
    eps = np.finfo(float).eps ** (1/2)  # 获取浮点数的最小增量
    val = func(xs, ys)  # 计算给定函数在 (xs, ys) 处的值
    # 在 x 和 y 方向上计算函数的有限差分，并确保不超出给定的 xlims 和 ylims
    xlo, xhi = sorted(xlims)
    dxlo = xs - xlo
    dxhi = xhi - xs
    xeps = (np.take([-1, 1], dxhi >= dxlo)
            * np.minimum(eps, np.maximum(dxlo, dxhi)))
    val_dx = func(xs + xeps, ys)  # x 方向上的函数值
    ylo, yhi = sorted(ylims)
    dylo = ys - ylo
    dyhi = yhi - ys
    yeps = (np.take([-1, 1], dyhi >= dylo)
            * np.minimum(eps, np.maximum(dylo, dyhi)))
    val_dy = func(xs, ys + yeps)  # y 方向上的函数值
    return (val, (val_dx - val) / xeps, (val_dy - val) / yeps)  # 返回函数值及其在 x 和 y 方向上的导数


class FixedAxisArtistHelper(_FixedAxisArtistHelperBase):
    """
    Helper class for a fixed axis.
    """

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        super().__init__(loc=side)  # 调用基类的构造方法初始化固定轴辅助类

        self.grid_helper = grid_helper  # 设置网格辅助对象
        if nth_coord_ticks is None:
            nth_coord_ticks = self.nth_coord
        self.nth_coord_ticks = nth_coord_ticks  # 设置坐标轴上的刻度位置

        self.side = side  # 设置边的位置（左、右、上、下）

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)  # 更新网格辅助对象的坐标轴限制

    def get_tick_transform(self, axes):
        return axes.transData  # 返回数据变换对象以便获取刻度

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        v1, v2 = axes.get_ylim() if self.nth_coord == 0 else axes.get_xlim()
        if v1 > v2:  # 如果限制反向（上限小于下限），调整边的位置
            side = {"left": "right", "right": "left",
                    "top": "bottom", "bottom": "top"}[self.side]
        else:
            side = self.side

        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[side]  # 根据边的位置确定刻度角度

        def iter_major():
            for nth_coord, show_labels in [
                    (self.nth_coord_ticks, True), (1 - self.nth_coord_ticks, False)]:
                gi = self.grid_helper._grid_info[["lon", "lat"][nth_coord]]  # 获取网格信息中对应坐标轴的信息
                for tick in gi["ticks"][side]:  # 遍历该边上的刻度信息
                    yield (*tick["loc"], angle_tangent,
                           (tick["label"] if show_labels else ""))  # 生成刻度位置、角度和标签（如果显示）

        return iter_major(), iter([])  # 返回主要刻度迭代器和空的次要刻度迭代器
class FloatingAxisArtistHelper(_FloatingAxisArtistHelperBase):

    def __init__(self, grid_helper, nth_coord, value, axis_direction=None):
        """
        初始化函数，用于创建FloatingAxisArtistHelper对象。

        Args:
            grid_helper: 网格帮助器对象，用于处理网格相关操作
            nth_coord: 表示坐标轴方向的索引，0表示x轴，1表示y轴
            value: 坐标轴上的特定数值
            axis_direction: 可选参数，指定坐标轴的方向
        """
        super().__init__(nth_coord, value)  # 调用父类的初始化方法
        self.value = value  # 设置坐标轴上的特定数值
        self.grid_helper = grid_helper  # 设置网格帮助器对象
        self._extremes = -np.inf, np.inf  # 设置坐标轴的极限范围，默认为负无穷到正无穷
        self._line_num_points = 100  # 创建线条所用的点的数量

    def set_extremes(self, e1, e2):
        """
        设置坐标轴的极限范围。

        Args:
            e1: 第一个极限范围的值，如果为None则默认为负无穷
            e2: 第二个极限范围的值，如果为None则默认为正无穷
        """
        if e1 is None:
            e1 = -np.inf
        if e2 is None:
            e2 = np.inf
        self._extremes = e1, e2  # 更新极限范围

    def update_lim(self, axes):
        """
        更新坐标轴的限制范围。

        Args:
            axes: matplotlib.axes.Axes对象，表示需要更新的坐标轴
        """
        self.grid_helper.update_lim(axes)  # 调用网格帮助器的更新方法

        x1, x2 = axes.get_xlim()  # 获取x轴的限制范围
        y1, y2 = axes.get_ylim()  # 获取y轴的限制范围
        grid_finder = self.grid_helper.grid_finder  # 获取网格查找器对象
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)  # 查找极限范围

        lon_min, lon_max, lat_min, lat_max = extremes  # 分解极限范围到经度和纬度的最小和最大值
        e_min, e_max = self._extremes  # 获取其他坐标轴范围

        if self.nth_coord == 0:
            lat_min = max(e_min, lat_min)  # 更新纬度的最小值
            lat_max = min(e_max, lat_max)  # 更新纬度的最大值
        elif self.nth_coord == 1:
            lon_min = max(e_min, lon_min)  # 更新经度的最小值
            lon_max = min(e_max, lon_max)  # 更新经度的最大值

        lon_levs, lon_n, lon_factor = \
            grid_finder.grid_locator1(lon_min, lon_max)  # 经度的网格定位器
        lat_levs, lat_n, lat_factor = \
            grid_finder.grid_locator2(lat_min, lat_max)  # 纬度的网格定位器

        if self.nth_coord == 0:
            xx0 = np.full(self._line_num_points, self.value)  # 在x轴上创建一条线
            yy0 = np.linspace(lat_min, lat_max, self._line_num_points)  # 在y轴上创建一条线
            xx, yy = grid_finder.transform_xy(xx0, yy0)  # 转换坐标系
        elif self.nth_coord == 1:
            xx0 = np.linspace(lon_min, lon_max, self._line_num_points)  # 在x轴上创建一条线
            yy0 = np.full(self._line_num_points, self.value)  # 在y轴上创建一条线
            xx, yy = grid_finder.transform_xy(xx0, yy0)  # 转换坐标系

        self._grid_info = {
            "extremes": (lon_min, lon_max, lat_min, lat_max),  # 极限范围信息
            "lon_info": (lon_levs, lon_n, np.asarray(lon_factor)),  # 经度信息
            "lat_info": (lat_levs, lat_n, np.asarray(lat_factor)),  # 纬度信息
            "lon_labels": grid_finder._format_ticks(
                1, "bottom", lon_factor, lon_levs),  # 经度标签
            "lat_labels": grid_finder._format_ticks(
                2, "bottom", lat_factor, lat_levs),  # 纬度标签
            "line_xy": (xx, yy),  # 线条的坐标
        }

    def get_axislabel_transform(self, axes):
        """
        获取坐标轴标签的变换对象。

        Args:
            axes: matplotlib.axes.Axes对象，表示需要获取变换的坐标轴

        Returns:
            Affine2D对象，表示坐标轴标签的变换对象
        """
        return Affine2D()  # 返回坐标轴标签的默认变换对象，即axes.transData
    def get_axislabel_pos_angle(self, axes):
        def trf_xy(x, y):
            # 获取网格变换器，结合数据坐标轴的变换器
            trf = self.grid_helper.grid_finder.get_transform() + axes.transData
            # 将输入的数据坐标 (x, y) 转换为画布坐标
            return trf.transform([x, y]).T

        # 获取坐标轴的极值
        xmin, xmax, ymin, ymax = self._grid_info["extremes"]
        if self.nth_coord == 0:
            # 根据当前坐标轴类型选择固定一个坐标轴的值，另一个值取极值的中间值
            xx0 = self.value
            yy0 = (ymin + ymax) / 2
        elif self.nth_coord == 1:
            xx0 = (xmin + xmax) / 2
            yy0 = self.value
        # 计算坐标点的值和其雅可比矩阵
        xy1, dxy1_dx, dxy1_dy = _value_and_jacobian(
            trf_xy, xx0, yy0, (xmin, xmax), (ymin, ymax))
        # 将坐标点转换为相对于坐标轴的比例位置
        p = axes.transAxes.inverted().transform(xy1)
        if 0 <= p[0] <= 1 and 0 <= p[1] <= 1:
            # 根据坐标轴类型选择对应的正交方向，计算角度并返回
            d = [dxy1_dy, dxy1_dx][self.nth_coord]
            return xy1, np.rad2deg(np.arctan2(*d[::-1]))
        else:
            # 如果坐标点不在坐标轴内，则返回空值
            return None, None

    def get_tick_transform(self, axes):
        # 返回一个恒等变换，即不进行任何变换
        return IdentityTransform()  # axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""

        # 获取经度和纬度的信息和缩放因子
        lat_levs, lat_n, lat_factor = self._grid_info["lat_info"]
        yy0 = lat_levs / lat_factor

        lon_levs, lon_n, lon_factor = self._grid_info["lon_info"]
        xx0 = lon_levs / lon_factor

        e0, e1 = self._extremes

        def trf_xy(x, y):
            # 获取网格变换器，结合数据坐标轴的变换器
            trf = self.grid_helper.grid_finder.get_transform() + axes.transData
            # 将输入的数据坐标 (x, y) 转换为画布坐标
            return trf.transform(np.column_stack(np.broadcast_arrays(x, y))).T

        # 查找角度
        if self.nth_coord == 0:
            # 根据当前坐标轴类型选择固定一个坐标轴的值，筛选出符合条件的数据点
            mask = (e0 <= yy0) & (yy0 <= e1)
            (xx1, yy1), (dxx1, dyy1), (dxx2, dyy2) = _value_and_jacobian(
                trf_xy, self.value, yy0[mask], (-np.inf, np.inf), (e0, e1))
            labels = self._grid_info["lat_labels"]

        elif self.nth_coord == 1:
            # 根据当前坐标轴类型选择固定一个坐标轴的值，筛选出符合条件的数据点
            mask = (e0 <= xx0) & (xx0 <= e1)
            (xx1, yy1), (dxx2, dyy2), (dxx1, dyy1) = _value_and_jacobian(
                trf_xy, xx0[mask], self.value, (-np.inf, np.inf), (e0, e1))
            labels = self._grid_info["lon_labels"]

        # 仅保留符合条件的标签
        labels = [l for l, m in zip(labels, mask) if m]

        # 计算法线和切线的角度
        angle_normal = np.arctan2(dyy1, dxx1)
        angle_tangent = np.arctan2(dyy2, dxx2)
        mm = (dyy1 == 0) & (dxx1 == 0)  # 带有退化法线的点
        angle_normal[mm] = angle_tangent[mm] + np.pi / 2

        # 计算刻度变换到坐标轴的变换
        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
        in_01 = functools.partial(
            mpl.transforms._interval_contains_close, (0, 1))

        def iter_major():
            for x, y, normal, tangent, lab \
                    in zip(xx1, yy1, angle_normal, angle_tangent, labels):
                c2 = tick_to_axes.transform((x, y))
                if in_01(c2[0]) and in_01(c2[1]):
                    yield [x, y], *np.rad2deg([normal, tangent]), lab

        # 返回主刻度的迭代器和空的次刻度迭代器
        return iter_major(), iter([])

    def get_line_transform(self, axes):
        # 返回数据坐标的变换
        return axes.transData
    # 定义一个方法，用于获取某个轴对象上的线条路径
    def get_line(self, axes):
        # 调用对象自身的方法，更新指定轴的限制
        self.update_lim(axes)
        # 从对象的网格信息中获取线条的 x 和 y 坐标
        x, y = self._grid_info["line_xy"]
        # 创建并返回一个路径对象，路径由 x 和 y 坐标组成的列堆叠而成
        return Path(np.column_stack([x, y]))
class GridHelperCurveLinear(GridHelperBase):
    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        Parameters
        ----------
        aux_trans : `.Transform` or tuple[Callable, Callable]
            The transform from curved coordinates to rectilinear coordinate:
            either a `.Transform` instance (which provides also its inverse),
            or a pair of callables ``(trans, inv_trans)`` that define the
            transform and its inverse.  The callables should have signature::

                x_rect, y_rect = trans(x_curved, y_curved)
                x_curved, y_curved = inv_trans(x_rect, y_rect)

        extreme_finder : callable or None
            An optional callable for finding the extremes of the grid.

        grid_locator1, grid_locator2 : Locator or None
            Grid locators for each axis.

        tick_formatter1, tick_formatter2 : Formatter or None
            Tick formatters for each axis.
        """
        super().__init__()
        self._grid_info = None
        # 创建一个 GridFinder 对象，用于处理网格相关的逻辑
        self.grid_finder = GridFinder(aux_trans,
                                      extreme_finder,
                                      grid_locator1,
                                      grid_locator2,
                                      tick_formatter1,
                                      tick_formatter2)

    def update_grid_finder(self, aux_trans=None, **kwargs):
        """
        Update the grid finder with optional new transform and keyword arguments.

        Parameters
        ----------
        aux_trans : `.Transform` or tuple[Callable, Callable], optional
            The new transform from curved to rectilinear coordinates.
        
        **kwargs
            Additional keyword arguments to update the grid finder.

        Notes
        -----
        Forces revalidation by setting `_old_limits` to `None`.
        """
        if aux_trans is not None:
            self.grid_finder.update_transform(aux_trans)
        self.grid_finder.update(**kwargs)
        self._old_limits = None  # Force revalidation.

    @_api.make_keyword_only("3.9", "nth_coord")
    def new_fixed_axis(
            self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        """
        Create a new fixed axis artist.

        Parameters
        ----------
        loc : str or int
            Location of the axis.

        nth_coord : int, optional
            Index of the coordinate axis.

        axis_direction : str, optional
            Direction of the axis.

        offset : float, optional
            Offset of the axis.

        axes : `.Axes` or None, optional
            The axes to attach the axis artist to.

        Returns
        -------
        axisline : AxisArtist
            The newly created axis artist.
        """
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        # 创建一个 FixedAxisArtistHelper 辅助对象，使用 loc 和 nth_coord
        helper = FixedAxisArtistHelper(self, loc, nth_coord_ticks=nth_coord)
        # 使用 AxisArtist 创建 axisline，使用指定的 axes 和 axis_direction
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        # 返回创建的 axisline 对象
        return axisline

    def new_floating_axis(self, nth_coord, value, axes=None, axis_direction="bottom"):
        """
        Create a new floating axis artist.

        Parameters
        ----------
        nth_coord : int
            Index of the coordinate axis.

        value : float
            Value of the axis.

        axes : `.Axes` or None, optional
            The axes to attach the axis artist to.

        axis_direction : str, optional
            Direction of the axis.

        Returns
        -------
        axisline : AxisArtist
            The newly created axis artist.
        """
        if axes is None:
            axes = self.axes
        # 创建一个 FloatingAxisArtistHelper 辅助对象，使用 nth_coord、value 和 axis_direction
        helper = FloatingAxisArtistHelper(
            self, nth_coord, value, axis_direction)
        # 使用 AxisArtist 创建 axisline，并设置 clip_on 为 True 和 clip_box
        axisline = AxisArtist(axes, helper)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        # axisline.major_ticklabels.set_visible(True)
        # axisline.minor_ticklabels.set_visible(False)
        # 返回创建的 axisline 对象
        return axisline

    def _update_grid(self, x1, y1, x2, y2):
        """
        Update the internal grid information based on the given limits.

        Parameters
        ----------
        x1, y1, x2, y2 : float
            Coordinates defining the bounds of the grid.
        """
        self._grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)
    # 获取网格线的函数，返回指定轴上的主要或所有网格线的位置
    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []  # 初始化一个空列表来存放网格线的位置信息
        if axis in ["both", "x"]:  # 如果指定获取 x 轴或者同时获取两个轴的网格线
            # 遍历经度方向（x 轴）的网格线信息，将每条线的位置信息扩展到 grid_lines 列表中
            for gl in self._grid_info["lon"]["lines"]:
                grid_lines.extend(gl)
        if axis in ["both", "y"]:  # 如果指定获取 y 轴或者同时获取两个轴的网格线
            # 遍历纬度方向（y 轴）的网格线信息，将每条线的位置信息扩展到 grid_lines 列表中
            for gl in self._grid_info["lat"]["lines"]:
                grid_lines.extend(gl)
        # 返回包含指定轴上网格线位置信息的列表
        return grid_lines

    # 标记为废弃的函数，版本 3.9 后不推荐使用
    @_api.deprecated("3.9")
    def get_tick_iterator(self, nth_coord, axis_side, minor=False):
        # 根据轴的方向确定角度的正切值
        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        # 确定是经度（lon）还是纬度（lat）方向
        lon_or_lat = ["lon", "lat"][nth_coord]
        if not minor:  # 如果获取的是主要刻度线而不是次要刻度线
            # 遍历指定轴上的主要刻度线信息，生成一个迭代器，每次返回刻度线的位置、角度、标签信息
            for tick in self._grid_info[lon_or_lat]["ticks"][axis_side]:
                yield *tick["loc"], angle_tangent, tick["label"]
        else:
            # 如果获取的是次要刻度线，生成一个迭代器，每次返回刻度线的位置、角度（无角度）、空字符串（无标签）
            for tick in self._grid_info[lon_or_lat]["ticks"][axis_side]:
                yield *tick["loc"], angle_tangent, ""
```