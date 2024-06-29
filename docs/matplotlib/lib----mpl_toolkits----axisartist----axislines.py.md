# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\axislines.py`

```py
"""
Axislines includes modified implementation of the Axes class. The
biggest difference is that the artists responsible for drawing the axis spine,
ticks, ticklabels and axis labels are separated out from Matplotlib's Axis
class. Originally, this change was motivated to support curvilinear
grid. Here are a few reasons that I came up with a new axes class:

* "top" and "bottom" x-axis (or "left" and "right" y-axis) can have
  different ticks (tick locations and labels). This is not possible
  with the current Matplotlib, although some twin axes trick can help.

* Curvilinear grid.

* angled ticks.

In the new axes class, xaxis and yaxis is set to not visible by
default, and new set of artist (AxisArtist) are defined to draw axis
line, ticks, ticklabels and axis label. Axes.axis attribute serves as
a dictionary of these artists, i.e., ax.axis["left"] is a AxisArtist
instance responsible to draw left y-axis. The default Axes.axis contains
"bottom", "left", "top" and "right".

AxisArtist can be considered as a container artist and has the following
children artists which will draw ticks, labels, etc.

* line
* major_ticks, major_ticklabels
* minor_ticks, minor_ticklabels
* offsetText
* label

Note that these are separate artists from `matplotlib.axis.Axis`, thus most
tick-related functions in Matplotlib won't work. For example, color and
markerwidth of the ``ax.axis["bottom"].major_ticks`` will follow those of
Axes.xaxis unless explicitly specified.

In addition to AxisArtist, the Axes will have *gridlines* attribute,
which obviously draws grid lines. The gridlines needs to be separated
from the axis as some gridlines can never pass any axis.
"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection


class _AxisArtistHelperBase:
    """
    Base class for axis helper.

    Subclasses should define the methods listed below.  The *axes*
    argument will be the ``.axes`` attribute of the caller artist. ::

        # Construct the spine.

        def get_line_transform(self, axes):
            return transform

        def get_line(self, axes):
            return path

        # Construct the label.

        def get_axislabel_transform(self, axes):
            return transform

        def get_axislabel_pos_angle(self, axes):
            return (x, y), angle

        # Construct the ticks.

        def get_tick_transform(self, axes):
            return transform

        def get_tick_iterators(self, axes):
            # A pair of iterables (one for major ticks, one for minor ticks)
            # that yield (tick_position, tick_angle, tick_label).
            return iter_major, iter_minor
    """

    def __init__(self, nth_coord):
        self.nth_coord = nth_coord

    def update_lim(self, axes):
        pass
    # 返回实例变量 self.nth_coord 的当前值
    def get_nth_coord(self):
        return self.nth_coord

    # 将给定的 values 数组与常量 const 组合成 (*values.shape, 2) 形状的数组，表示 (x, y) 对

    # 例如：
    # >>> self.nth_coord = 0
    # >>> self._to_xy([1, 2, 3], const=0)
    # 返回数组：
    # array([[1, 0],
    #        [2, 0],
    #        [3, 0]])

    def _to_xy(self, values, const):
        if self.nth_coord == 0:
            # 将 values 与 const 组合成 (x, const) 形状的数组
            return np.stack(np.broadcast_arrays(values, const), axis=-1)
        elif self.nth_coord == 1:
            # 将 const 与 values 组合成 (const, y) 形状的数组
            return np.stack(np.broadcast_arrays(const, values), axis=-1)
        else:
            # 如果 self.nth_coord 不是 0 或 1，抛出异常
            raise ValueError("Unexpected nth_coord")
class _FixedAxisArtistHelperBase(_AxisArtistHelperBase):
    """Helper class for a fixed (in the axes coordinate) axis."""

    @_api.delete_parameter("3.9", "nth_coord")
    def __init__(self, loc, nth_coord=None):
        """Initialize the helper with a location and optional coordinate index.

        ``nth_coord = 0``: x-axis; ``nth_coord = 1``: y-axis.
        """
        super().__init__(_api.check_getitem(
            {"bottom": 0, "top": 0, "left": 1, "right": 1}, loc=loc))
        self._loc = loc
        self._pos = {"bottom": 0, "top": 1, "left": 0, "right": 1}[loc]
        # Create a path representing the axis line in transAxes coordinates
        self._path = Path(self._to_xy((0, 1), const=self._pos))

    # LINE

    def get_line(self, axes):
        """Return the path representing the axis line in transAxes."""
        return self._path

    def get_line_transform(self, axes):
        """Return the transformation to transAxes for the axis line."""
        return axes.transAxes

    # LABEL

    def get_axislabel_transform(self, axes):
        """Return the transformation to transAxes for the axis label."""
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        """
        Return the position and angle of the axis label in transAxes coordinates.

        get_label_transform() returns a transform of (transAxes+offset)
        """
        return dict(left=((0., 0.5), 90),  # (position, angle_tangent)
                    right=((1., 0.5), 90),
                    bottom=((0.5, 0.), 0),
                    top=((0.5, 1.), 0))[self._loc]

    # TICK

    def get_tick_transform(self, axes):
        """Return the transformation to transAxes for ticks."""
        return [axes.get_xaxis_transform(), axes.get_yaxis_transform()][self.nth_coord]


class _FloatingAxisArtistHelperBase(_AxisArtistHelperBase):
    """Base class for floating axis artists."""

    def __init__(self, nth_coord, value):
        """Initialize with the coordinate index and a fixed value."""
        self._value = value
        super().__init__(nth_coord)

    def get_line(self, axes):
        """Raise an error indicating that this method should be defined by derived classes."""
        raise RuntimeError("get_line method should be defined by the derived class")


class FixedAxisArtistHelperRectilinear(_FixedAxisArtistHelperBase):
    """Helper class for fixed rectilinear axes."""

    @_api.delete_parameter("3.9", "nth_coord")
    def __init__(self, axes, loc, nth_coord=None):
        """
        Initialize the helper with axes, location, and optional coordinate index.

        nth_coord = along which coordinate value varies.
        In 2D, nth_coord = 0 -> x axis, nth_coord = 1 -> y axis
        """
        super().__init__(loc)
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    # TICK
    def get_tick_iterators(self, axes):
        """返回三个迭代器：主刻度位置和角度、次刻度位置和角度、刻度标签"""
        
        # 根据当前坐标轴方向选择角度值
        angle_normal, angle_tangent = {0: (90, 0), 1: (0, 90)}[self.nth_coord]

        # 获取主刻度的位置和标签
        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)

        # 获取次刻度的位置和标签
        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)

        # 计算刻度点到坐标轴的变换，并减去坐标轴的整体变换
        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes

        def _f(locs, labels):
            # 遍历位置和标签，生成包含刻度点、角度、标签的迭代器
            for loc, label in zip(locs, labels):
                c = self._to_xy(loc, const=self._pos)
                # 检查刻度点是否在坐标轴内部
                c2 = tick_to_axes.transform(c)
                if mpl.transforms._interval_contains_close((0, 1), c2[self.nth_coord]):
                    yield c, angle_normal, angle_tangent, label

        # 返回主刻度和次刻度的迭代器函数
        return _f(major_locs, major_labels), _f(minor_locs, minor_labels)
class FloatingAxisArtistHelperRectilinear(_FloatingAxisArtistHelperBase):
    # 继承自 _FloatingAxisArtistHelperBase 类的浮动坐标轴辅助类，用于直角坐标系

    def __init__(self, axes, nth_coord,
                 passingthrough_point, axis_direction="bottom"):
        # 初始化方法，设置坐标轴方向和通过的点
        super().__init__(nth_coord, passingthrough_point)
        self._axis_direction = axis_direction  # 设置坐标轴方向
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]  # 根据坐标轴方向选择 x 轴或 y 轴

    def get_line(self, axes):
        # 获取坐标轴线路径对象
        fixed_coord = 1 - self.nth_coord  # 固定坐标轴方向
        data_to_axes = axes.transData - axes.transAxes  # 数据到坐标轴变换
        p = data_to_axes.transform([self._value, self._value])  # 将数据点变换到坐标轴上
        return Path(self._to_xy((0, 1), const=p[fixed_coord]))  # 返回路径对象

    def get_line_transform(self, axes):
        # 获取坐标轴线的变换
        return axes.transAxes  # 返回坐标轴变换对象

    def get_axislabel_transform(self, axes):
        # 获取坐标轴标签的变换
        return axes.transAxes  # 返回坐标轴变换对象

    def get_axislabel_pos_angle(self, axes):
        """
        返回标签参考位置在 transAxes 中的位置。

        get_label_transform() 返回 (transAxes+offset) 的变换。
        """
        angle = [0, 90][self.nth_coord]  # 标签角度，水平或垂直
        fixed_coord = 1 - self.nth_coord  # 固定坐标轴方向
        data_to_axes = axes.transData - axes.transAxes  # 数据到坐标轴变换
        p = data_to_axes.transform([self._value, self._value])  # 将数据点变换到坐标轴上
        verts = self._to_xy(0.5, const=p[fixed_coord])  # 获取标签位置
        return (verts, angle) if 0 <= verts[fixed_coord] <= 1 else (None, None)  # 如果位置在有效范围内则返回，否则返回空

    def get_tick_transform(self, axes):
        # 获取刻度的变换
        return axes.transData  # 返回数据到坐标轴的变换对象

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        angle_normal, angle_tangent = {0: (90, 0), 1: (0, 90)}[self.nth_coord]  # 刻度线角度，法线和切线方向

        major = self.axis.major  # 主刻度
        major_locs = major.locator()  # 主刻度位置
        major_labels = major.formatter.format_ticks(major_locs)  # 主刻度标签

        minor = self.axis.minor  # 次刻度
        minor_locs = minor.locator()  # 次刻度位置
        minor_labels = minor.formatter.format_ticks(minor_locs)  # 次刻度标签

        data_to_axes = axes.transData - axes.transAxes  # 数据到坐标轴变换

        def _f(locs, labels):
            for loc, label in zip(locs, labels):
                c = self._to_xy(loc, const=self._value)  # 将刻度位置变换到坐标轴
                c1, c2 = data_to_axes.transform(c)  # 数据到坐标轴变换
                if 0 <= c1 <= 1 and 0 <= c2 <= 1:  # 如果位置在有效范围内
                    yield c, angle_normal, angle_tangent, label  # 生成刻度信息

        return _f(major_locs, major_labels), _f(minor_locs, minor_labels)  # 返回主刻度和次刻度的生成器


class AxisArtistHelper:  # Backcompat.
    Fixed = _FixedAxisArtistHelperBase  # 固定坐标轴辅助类
    Floating = _FloatingAxisArtistHelperBase  # 浮动坐标轴辅助类


class AxisArtistHelperRectlinear:  # Backcompat.
    Fixed = FixedAxisArtistHelperRectilinear  # 固定直角坐标轴辅助类
    Floating = FloatingAxisArtistHelperRectilinear  # 浮动直角坐标轴辅助类


class GridHelperBase:
    # 网格帮助基类

    def __init__(self):
        self._old_limits = None  # 初始化保存的旧坐标轴限制
        super().__init__()  # 调用父类初始化方法

    def update_lim(self, axes):
        # 更新坐标轴限制方法
        x1, x2 = axes.get_xlim()  # 获取 x 轴限制
        y1, y2 = axes.get_ylim()  # 获取 y 轴限制
        if self._old_limits != (x1, x2, y1, y2):  # 如果限制有变化
            self._update_grid(x1, y1, x2, y2)  # 更新网格
            self._old_limits = (x1, x2, y1, y2)  # 更新保存的旧坐标轴限制

    def _update_grid(self, x1, y1, x2, y2):
        """缓存当坐标轴限制变化时相关计算。"""
    # 定义一个方法 get_gridlines，用于获取网格线的路径列表
    def get_gridlines(self, which, axis):
        """
        Return list of grid lines as a list of paths (list of points).
    
        Parameters
        ----------
        which : {"both", "major", "minor"}
            指定返回的网格线类型，可以是 "both"（主次均包括）、"major"（仅主网格线）、"minor"（仅次网格线）
        axis : {"both", "x", "y"}
            指定返回的网格线所属的轴，可以是 "both"（两个轴都包括）、"x"（x 轴）、"y"（y 轴）
        """
        # 返回一个空列表，表示当前未实现获取网格线的具体逻辑
        return []
class GridHelperRectlinear(GridHelperBase):
    # GridHelperRectlinear 类，继承自 GridHelperBase

    def __init__(self, axes):
        # 初始化方法，接受参数 axes
        super().__init__()
        self.axes = axes
        # 设置实例变量 axes

    @_api.delete_parameter(
        "3.9", "nth_coord", addendum="'nth_coord' is now inferred from 'loc'.")
    # 标记为 API 删除参数，版本 3.9 开始，'nth_coord' 参数由 'loc' 推断

    def new_fixed_axis(
            self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        # 创建新的固定轴线方法
        if axes is None:
            _api.warn_external(
                "'new_fixed_axis' explicitly requires the axes keyword.")
            axes = self.axes
            # 如果 axes 参数未提供，发出警告并使用实例变量 axes
        if axis_direction is None:
            axis_direction = loc
            # 如果未指定 axis_direction，使用 loc 参数
        # 返回创建的 AxisArtist 对象，使用 FixedAxisArtistHelperRectilinear 辅助类
        return AxisArtist(axes, FixedAxisArtistHelperRectilinear(axes, loc),
                          offset=offset, axis_direction=axis_direction)

    def new_floating_axis(self, nth_coord, value, axis_direction="bottom", axes=None):
        # 创建新的浮动轴线方法
        if axes is None:
            _api.warn_external(
                "'new_floating_axis' explicitly requires the axes keyword.")
            axes = self.axes
            # 如果 axes 参数未提供，发出警告并使用实例变量 axes
        # 创建 FloatingAxisArtistHelperRectilinear 辅助类实例
        helper = FloatingAxisArtistHelperRectilinear(
            axes, nth_coord, value, axis_direction)
        # 创建 AxisArtist 对象，使用 axis_direction 参数
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        # 设置轴线对象的剪裁属性和剪裁框
        return axisline

    def get_gridlines(self, which="major", axis="both"):
        """
        Return list of gridline coordinates in data coordinates.

        Parameters
        ----------
        which : {"both", "major", "minor"}
            指定要获取的网格线类型
        axis : {"both", "x", "y"}
            指定要获取的轴线类型
        """
        _api.check_in_list(["both", "major", "minor"], which=which)
        _api.check_in_list(["both", "x", "y"], axis=axis)
        # 使用 API 检查参数是否在允许列表中
        gridlines = []

        if axis in ("both", "x"):
            locs = []
            y1, y2 = self.axes.get_ylim()
            # 获取当前轴的 y 轴限制范围
            if which in ("both", "major"):
                locs.extend(self.axes.xaxis.major.locator())
                # 若 which 包含 "major"，则获取 x 轴主要刻度定位器
            if which in ("both", "minor"):
                locs.extend(self.axes.xaxis.minor.locator())
                # 若 which 包含 "minor"，则获取 x 轴次要刻度定位器
            gridlines.extend([[x, x], [y1, y2]] for x in locs)
            # 生成 x 轴网格线坐标并添加到 gridlines

        if axis in ("both", "y"):
            x1, x2 = self.axes.get_xlim()
            # 获取当前轴的 x 轴限制范围
            locs = []
            if self.axes.yaxis._major_tick_kw["gridOn"]:
                locs.extend(self.axes.yaxis.major.locator())
                # 若 y 轴主要刻度网格开启，则获取 y 轴主要刻度定位器
            if self.axes.yaxis._minor_tick_kw["gridOn"]:
                locs.extend(self.axes.yaxis.minor.locator())
                # 若 y 轴次要刻度网格开启，则获取 y 轴次要刻度定位器
            gridlines.extend([[x1, x2], [y, y]] for y in locs)
            # 生成 y 轴网格线坐标并添加到 gridlines

        return gridlines
        # 返回所有网格线坐标的列表


class Axes(maxes.Axes):

    @_api.deprecated("3.8", alternative="ax.axis")
    # 标记为 API 弃用，推荐使用 alternative 参数指定的替代方法 'ax.axis'

    def __call__(self, *args, **kwargs):
        # Axes 对象的调用方法
        return maxes.Axes.axis(self.axes, *args, **kwargs)
        # 调用 maxes.Axes 的 axis 方法

    def __init__(self, *args, grid_helper=None, **kwargs):
        # 初始化方法，接受可变数量参数及关键字参数，grid_helper 参数可选
        self._axisline_on = True
        # 设置轴线开关状态为开启
        self._grid_helper = grid_helper if grid_helper else GridHelperRectlinear(self)
        # 设置实例变量 _grid_helper，如果 grid_helper 未提供，则使用 GridHelperRectlinear 实例
        super().__init__(*args, **kwargs)
        # 调用父类的初始化方法，并传入所有参数
        self.toggle_axisline(True)
        # 开启轴线显示
    # 定义一个方法用于切换坐标轴线的可见性
    def toggle_axisline(self, b=None):
        # 如果参数 b 为 None，则根据当前状态取反
        if b is None:
            b = not self._axisline_on
        # 如果 b 为 True，则显示坐标轴线
        if b:
            self._axisline_on = True
            # 隐藏所有边框线
            self.spines[:].set_visible(False)
            # 隐藏 x 轴和 y 轴
            self.xaxis.set_visible(False)
            self.yaxis.set_visible(False)
        else:
            # 如果 b 为 False，则显示坐标轴线
            self._axisline_on = False
            # 显示所有边框线
            self.spines[:].set_visible(True)
            # 显示 x 轴和 y 轴
            self.xaxis.set_visible(True)
            self.yaxis.set_visible(True)

    @property
    def axis(self):
        # 返回 _axislines 属性
        return self._axislines

    def clear(self):
        # 继承的文档字符串

        # 在调用 clear() 方法之前初始化网格线
        self.gridlines = gridlines = GridlinesCollection(
            [],
            colors=mpl.rcParams['grid.color'],
            linestyles=mpl.rcParams['grid.linestyle'],
            linewidths=mpl.rcParams['grid.linewidth'])
        self._set_artist_props(gridlines)
        gridlines.set_grid_helper(self.get_grid_helper())

        # 调用父类的 clear() 方法
        super().clear()

        # 设置剪切路径，以在创建补丁后设置
        gridlines.set_clip_path(self.axes.patch)

        # 初始化坐标轴艺术家
        self._axislines = mpl_axes.Axes.AxisDict(self)
        new_fixed_axis = self.get_grid_helper().new_fixed_axis
        # 更新 _axislines 属性
        self._axislines.update({
            loc: new_fixed_axis(loc=loc, axes=self, axis_direction=loc)
            for loc in ["bottom", "top", "left", "right"]})
        for axisline in [self._axislines["top"], self._axislines["right"]]:
            axisline.label.set_visible(False)
            axisline.major_ticklabels.set_visible(False)
            axisline.minor_ticklabels.set_visible(False)

    def get_grid_helper(self):
        # 返回 _grid_helper 属性
        return self._grid_helper

    def grid(self, visible=None, which='major', axis="both", **kwargs):
        """
        Toggle the gridlines, and optionally set the properties of the lines.
        """
        # 在 axes_grid 和 Matplotlib 之间存在 grid() 方法行为的一些差异，因为 axes_grid 明确设置了网格线的可见性
        super().grid(visible, which=which, axis=axis, **kwargs)
        # 如果 _axisline_on 为 False，则直接返回
        if not self._axisline_on:
            return
        # 如果 visible 为 None，则根据坐标轴的设置确定是否可见
        if visible is None:
            visible = (self.axes.xaxis._minor_tick_kw["gridOn"]
                       or self.axes.xaxis._major_tick_kw["gridOn"]
                       or self.axes.yaxis._minor_tick_kw["gridOn"]
                       or self.axes.yaxis._major_tick_kw["gridOn"])
        # 设置网格线的显示属性
        self.gridlines.set(which=which, axis=axis, visible=visible)
        self.gridlines.set(**kwargs)

    def get_children(self):
        # 如果 _axisline_on 为 True，则返回 _axislines 和 gridlines 的列表
        if self._axisline_on:
            children = [*self._axislines.values(), self.gridlines]
        else:
            # 否则只返回空列表
            children = []
        # 添加父类的 children 到列表中
        children.extend(super().get_children())
        return children

    def new_fixed_axis(self, loc, offset=None):
        # 返回新的固定坐标轴
        return self.get_grid_helper().new_fixed_axis(loc, offset=offset, axes=self)
    # 定义一个新的浮动坐标轴，用于在图形中添加新的浮动坐标轴
    def new_floating_axis(self, nth_coord, value, axis_direction="bottom"):
        # 调用所属对象的获取网格助手方法，创建新的浮动坐标轴
        return self.get_grid_helper().new_floating_axis(
            nth_coord,            # 第几个坐标轴（从0开始）
            value,                # 坐标轴的值
            axis_direction=axis_direction,  # 坐标轴方向，默认为底部
            axes=self             # 传递当前对象的坐标轴信息
        )
class AxesZero(Axes):
    # 定义一个自定义的 Axes 类，继承自 Axes 类

    def clear(self):
        # 清空图表的方法

        # 调用父类的 clear 方法，清空图表
        super().clear()

        # 获取网格辅助器并创建新的浮动轴
        new_floating_axis = self.get_grid_helper().new_floating_axis

        # 更新 _axislines 字典，设置新的 x 和 y 轴
        self._axislines.update(
            xzero=new_floating_axis(
                nth_coord=0, value=0., axis_direction="bottom", axes=self),
            yzero=new_floating_axis(
                nth_coord=1, value=0., axis_direction="left", axes=self),
        )

        # 遍历 ["xzero", "yzero"] 列表中的元素
        for k in ["xzero", "yzero"]:
            # 设置轴线的剪裁路径为图表的补丁路径（patch）
            self._axislines[k].line.set_clip_path(self.patch)
            # 将轴线设置为不可见状态
            self._axislines[k].set_visible(False)


Subplot = Axes
SubplotZero = AxesZero
```