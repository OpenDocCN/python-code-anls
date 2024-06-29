# `D:\src\scipysrc\matplotlib\lib\matplotlib\patches.py`

```py
"""
Patches are `.Artist`\s with a face color and an edge color.
"""

# 导入必要的模块和库
import functools  # 导入 functools 模块
import inspect  # 导入 inspect 模块
import math  # 导入 math 模块
from numbers import Number, Real  # 从 numbers 模块导入 Number 和 Real 类
import textwrap  # 导入 textwrap 模块
from types import SimpleNamespace  # 从 types 模块导入 SimpleNamespace 类
from collections import namedtuple  # 导入 namedtuple 类
from matplotlib.transforms import Affine2D  # 从 matplotlib.transforms 模块导入 Affine2D 类

import numpy as np  # 导入 numpy 库

import matplotlib as mpl  # 导入 matplotlib 库，并用 mpl 别名表示
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
               lines as mlines, transforms)  # 导入当前目录下的 _api, artist, cbook, colors, _docstring, hatch, lines, transforms 模块

from .bezier import (  # 从当前目录下的 bezier 模块导入以下函数和异常
    NonIntersectingPathException, get_cos_sin, get_intersection,
    get_parallels, inside_circle, make_wedged_bezier2,
    split_bezier_intersecting_with_closedpath, split_path_inout
)
from .path import Path  # 从当前目录下的 path 模块导入 Path 类
from ._enums import JoinStyle, CapStyle  # 导入当前目录下的 _enums 模块中的 JoinStyle 和 CapStyle 枚举


@_docstring.interpd
@_api.define_aliases({
    "antialiased": ["aa"],
    "edgecolor": ["ec"],
    "facecolor": ["fc"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
})
class Patch(artist.Artist):
    """
    A patch is a 2D artist with a face color and an edge color.

    If any of *edgecolor*, *facecolor*, *linewidth*, or *antialiased*
    are *None*, they default to their rc params setting.
    """
    zorder = 1  # 设置 zorder 属性为 1，用于决定渲染顺序

    # Whether to draw an edge by default.  Set on a
    # subclass-by-subclass basis.
    _edge_default = False  # 默认情况下不绘制边缘

    def __init__(self, *,
                 edgecolor=None,
                 facecolor=None,
                 color=None,
                 linewidth=None,
                 linestyle=None,
                 antialiased=None,
                 hatch=None,
                 fill=True,
                 capstyle=None,
                 joinstyle=None,
                 **kwargs):
        """
        The following kwarg properties are supported

        %(Patch:kwdoc)s
        """
        super().__init__()  # 调用父类的构造函数

        # 设置默认的线条风格和端点风格
        if linestyle is None:
            linestyle = "solid"
        if capstyle is None:
            capstyle = CapStyle.butt
        if joinstyle is None:
            joinstyle = JoinStyle.miter

        # 根据 rcParams 设置初始化填充颜色的 RGBA 值
        self._hatch_color = colors.to_rgba(mpl.rcParams['hatch.color'])
        self._fill = bool(fill)  # 将 fill 转换为布尔类型，用于 set_facecolor 方法调用
        if color is not None:
            # 如果指定了 color，则设置颜色并警告用户 color 属性会覆盖 edgecolor 或 facecolor 属性
            if edgecolor is not None or facecolor is not None:
                _api.warn_external(
                    "Setting the 'color' property will override "
                    "the edgecolor or facecolor properties.")
            self.set_color(color)
        else:
            # 如果未指定 color，则分别设置边缘颜色和填充颜色
            self.set_edgecolor(edgecolor)
            self.set_facecolor(facecolor)

        # 初始化线条宽度及虚线模式
        self._linewidth = 0
        self._unscaled_dash_pattern = (0, None)  # 偏移量, 虚线
        self._dash_pattern = (0, None)  # 偏移量, 虚线（按线宽缩放）

        # 设置线条风格、线宽、是否抗锯齿、图案填充、端点风格和连接风格
        self.set_linestyle(linestyle)
        self.set_linewidth(linewidth)
        self.set_antialiased(antialiased)
        self.set_hatch(hatch)
        self.set_capstyle(capstyle)
        self.set_joinstyle(joinstyle)

        # 如果有其他未处理的参数，调用 _internal_update 方法处理
        if len(kwargs):
            self._internal_update(kwargs)
    def get_verts(self):
        """
        Return a copy of the vertices used in this patch.

        If the patch contains Bézier curves, the curves will be interpolated by
        line segments.  To access the curves as curves, use `get_path`.
        """
        # 获取当前图形的变换
        trans = self.get_transform()
        # 获取当前图形的路径
        path = self.get_path()
        # 将路径转换为多边形的顶点坐标
        polygons = path.to_polygons(trans)
        # 如果存在多边形数据，则返回第一个多边形的顶点坐标
        if len(polygons):
            return polygons[0]
        # 如果不存在多边形数据，则返回空列表
        return []

    def _process_radius(self, radius):
        # 如果传入的半径不为 None，则直接返回该半径值
        if radius is not None:
            return radius
        # 如果当前对象的 picker 是数字类型，则使用该值作为半径
        if isinstance(self._picker, Number):
            _radius = self._picker
        else:
            # 如果边缘颜色的 alpha 通道为 0，则半径设为 0
            if self.get_edgecolor()[3] == 0:
                _radius = 0
            else:
                # 否则使用当前对象的线宽作为半径
                _radius = self.get_linewidth()
        return _radius

    def contains(self, mouseevent, radius=None):
        """
        Test whether the mouse event occurred in the patch.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`
            Where the user clicked.

        radius : float, optional
            Additional margin on the patch in target coordinates of
            `.Patch.get_transform`. See `.Path.contains_point` for further
            details.

            If `None`, the default value depends on the state of the object:

            - If `.Artist.get_picker` is a number, the default
              is that value.  This is so that picking works as expected.
            - Otherwise if the edge color has a non-zero alpha, the default
              is half of the linewidth.  This is so that all the colored
              pixels are "in" the patch.
            - Finally, if the edge has 0 alpha, the default is 0.  This is
              so that patches without a stroked edge do not have points
              outside of the filled region report as "in" due to an
              invisible edge.


        Returns
        -------
        (bool, empty dict)
        """
        # 如果鼠标事件发生在不同的画布上，则返回 False 和空字典
        if self._different_canvas(mouseevent):
            return False, {}
        # 处理半径参数，根据对象的状态确定默认值
        radius = self._process_radius(radius)
        # 获取当前路径的代码段
        codes = self.get_path().codes
        if codes is not None:
            # 获取当前路径的顶点坐标
            vertices = self.get_path().vertices
            # 获取所有子路径起始的 MOVETO 代码段的索引
            idxs, = np.where(codes == Path.MOVETO)
            # 不要在第一个 MOVETO 前进行分割
            idxs = idxs[1:]
            # 将路径分割成多个子路径
            subpaths = map(
                Path, np.split(vertices, idxs), np.split(codes, idxs))
        else:
            # 如果不存在代码段，则将当前路径作为唯一的子路径
            subpaths = [self.get_path()]
        # 判断鼠标事件是否发生在任意一个子路径内部
        inside = any(
            subpath.contains_point(
                (mouseevent.x, mouseevent.y), self.get_transform(), radius)
            for subpath in subpaths)
        # 返回是否在内部的布尔值和空字典
        return inside, {}
    def contains_point(self, point, radius=None):
        """
        Return whether the given point is inside the patch.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check, in target coordinates of
            ``.Patch.get_transform()``. These are display coordinates for patches
            that are added to a figure or Axes.
        radius : float, optional
            Additional margin on the patch in target coordinates of
            `.Patch.get_transform`. See `.Path.contains_point` for further
            details.

            If `None`, the default value depends on the state of the object:

            - If `.Artist.get_picker` is a number, the default
              is that value.  This is so that picking works as expected.
            - Otherwise if the edge color has a non-zero alpha, the default
              is half of the linewidth.  This is so that all the colored
              pixels are "in" the patch.
            - Finally, if the edge has 0 alpha, the default is 0.  This is
              so that patches without a stroked edge do not have points
              outside of the filled region report as "in" due to an
              invisible edge.

        Returns
        -------
        bool
            True if the point is inside the patch, False otherwise.

        Notes
        -----
        The proper use of this method depends on the transform of the patch.
        Isolated patches do not have a transform. In this case, the patch
        creation coordinates and the point coordinates match. The following
        example checks that the center of a circle is within the circle

        >>> center = 0, 0
        >>> c = Circle(center, radius=1)
        >>> c.contains_point(center)
        True

        The convention of checking against the transformed patch stems from
        the fact that this method is predominantly used to check if display
        coordinates (e.g. from mouse events) are within the patch. If you want
        to do the above check with data coordinates, you have to properly
        transform them first:

        >>> center = 0, 0
        >>> c = Circle(center, radius=3)
        >>> plt.gca().add_patch(c)
        >>> transformed_interior_point = c.get_data_transform().transform((0, 2))
        >>> c.contains_point(transformed_interior_point)
        True

        """
        # Process the radius value to ensure it's correctly formatted
        radius = self._process_radius(radius)
        # Check if the given point is inside the patch using the patch's path,
        # its transformation, and the specified or default radius
        return self.get_path().contains_point(point,
                                              self.get_transform(),
                                              radius)
    def contains_points(self, points, radius=None):
        """
        Return whether the given points are inside the patch.

        Parameters
        ----------
        points : (N, 2) array
            The points to check, in target coordinates of
            ``self.get_transform()``. These are display coordinates for patches
            that are added to a figure or Axes. Columns contain x and y values.
        radius : float, optional
            Additional margin on the patch in target coordinates of
            `.Patch.get_transform`. See `.Path.contains_point` for further
            details.

            If `None`, the default value depends on the state of the object:

            - If `.Artist.get_picker` is a number, the default
              is that value.  This is so that picking works as expected.
            - Otherwise if the edge color has a non-zero alpha, the default
              is half of the linewidth.  This is so that all the colored
              pixels are "in" the patch.
            - Finally, if the edge has 0 alpha, the default is 0.  This is
              so that patches without a stroked edge do not have points
              outside of the filled region report as "in" due to an
              invisible edge.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The proper use of this method depends on the transform of the patch.
        See the notes on `.Patch.contains_point`.
        """
        # Process the radius parameter to ensure it is correctly set
        radius = self._process_radius(radius)
        # Check if the given points are inside the patch's path
        return self.get_path().contains_points(points,
                                               self.get_transform(),
                                               radius)

    def update_from(self, other):
        """
        Update the current patch's attributes from another patch instance.

        Parameters
        ----------
        other : Patch
            The patch instance from which to copy attributes.
        """
        # Call the inherited update method to update common properties
        super().update_from(other)
        # Directly copy specific properties from the other patch instance
        self._edgecolor = other._edgecolor
        self._facecolor = other._facecolor
        self._original_edgecolor = other._original_edgecolor
        self._original_facecolor = other._original_facecolor
        self._fill = other._fill
        self._hatch = other._hatch
        self._hatch_color = other._hatch_color
        self._unscaled_dash_pattern = other._unscaled_dash_pattern
        # Set the linewidth and scaled dashes to match the other patch
        self.set_linewidth(other._linewidth)
        # Set the data transformation to match the other patch
        self.set_transform(other.get_data_transform())
        # Determine if the transform for this artist needs further initialization
        self._transformSet = other.is_transform_set()

    def get_extents(self):
        """
        Return the `Patch`'s axis-aligned extents as a `~.transforms.Bbox`.
        """
        # Retrieve and return the axis-aligned extents of the patch's path
        return self.get_path().get_extents(self.get_transform())
    def get_transform(self):
        """
        Return the `~.transforms.Transform` applied to the `Patch`.

        This method retrieves the transformation applied to the Patch object,
        combining the patch-specific transform with the general artist transform.
        """
        return self.get_patch_transform() + artist.Artist.get_transform(self)

    def get_data_transform(self):
        """
        Return the `~.transforms.Transform` mapping data coordinates to
        physical coordinates.

        This method returns the transformation that maps data coordinates to
        physical (display) coordinates for the Patch object.
        """
        return artist.Artist.get_transform(self)

    def get_patch_transform(self):
        """
        Return the `~.transforms.Transform` instance mapping patch coordinates
        to data coordinates.

        This method returns an identity transform indicating no additional
        transformation applied to patch coordinates.
        """
        return transforms.IdentityTransform()

    def get_antialiased(self):
        """
        Return whether antialiasing is used for drawing.

        This method returns the current antialiasing status for rendering the
        Patch object.
        """
        return self._antialiased

    def get_edgecolor(self):
        """
        Return the edge color.

        This method returns the current edge color used for rendering the
        Patch object.
        """
        return self._edgecolor

    def get_facecolor(self):
        """
        Return the face color.

        This method returns the current face color used for rendering the
        Patch object.
        """
        return self._facecolor

    def get_linewidth(self):
        """
        Return the line width in points.

        This method returns the current line width (in points) used for rendering
        the Patch object.
        """
        return self._linewidth

    def get_linestyle(self):
        """
        Return the linestyle.

        This method returns the current line style used for rendering the
        Patch object.
        """
        return self._linestyle

    def set_antialiased(self, aa):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        aa : bool or None
            If None, uses the default antialiasing setting from matplotlib.
        """
        if aa is None:
            aa = mpl.rcParams['patch.antialiased']
        self._antialiased = aa
        self.stale = True

    def _set_edgecolor(self, color):
        """
        Set the patch edge color.

        Parameters
        ----------
        color : :mpltype:`color` or None
            The color to set for the patch edge. If None, defaults are applied
            based on matplotlib configuration settings.
        """
        set_hatch_color = True
        if color is None:
            if (mpl.rcParams['patch.force_edgecolor'] or
                    not self._fill or self._edge_default):
                color = mpl.rcParams['patch.edgecolor']
            else:
                color = 'none'
                set_hatch_color = False

        self._edgecolor = colors.to_rgba(color, self._alpha)
        if set_hatch_color:
            self._hatch_color = self._edgecolor
        self.stale = True

    def set_edgecolor(self, color):
        """
        Set the patch edge color.

        Parameters
        ----------
        color : :mpltype:`color` or None
            The color to set for the patch edge. If None, defaults are applied
            based on matplotlib configuration settings.
        """
        self._original_edgecolor = color
        self._set_edgecolor(color)

    def _set_facecolor(self, color):
        """
        Set the patch face color.

        Parameters
        ----------
        color : :mpltype:`color` or None
            The color to set for the patch face. If None, defaults are applied
            based on matplotlib configuration settings.
        """
        if color is None:
            color = mpl.rcParams['patch.facecolor']
        alpha = self._alpha if self._fill else 0
        self._facecolor = colors.to_rgba(color, alpha)
        self.stale = True

    def set_facecolor(self, color):
        """
        Set the patch face color.

        Parameters
        ----------
        color : :mpltype:`color` or None
            The color to set for the patch face. If None, defaults are applied
            based on matplotlib configuration settings.
        """
        self._original_facecolor = color
        self._set_facecolor(color)
    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.

        Parameters
        ----------
        c : :mpltype:`color`
            The color to set for both edge and face.

        See Also
        --------
        Patch.set_facecolor, Patch.set_edgecolor
            For setting the edge or face color individually.
        """
        # 设置面和边的颜色为相同的颜色
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_alpha(self, alpha):
        """
        Set the transparency alpha level of the patch.

        Parameters
        ----------
        alpha : float
            Transparency level (0.0 transparent through 1.0 opaque).

        Notes
        -----
        This method overrides the alpha level of the patch and restores
        original face and edge colors.

        """
        # 调用父类方法设置透明度
        super().set_alpha(alpha)
        # 恢复原始的面和边颜色
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)
        # 设置标记为过时，已变为真值
        # stale is already True

    def set_linewidth(self, w):
        """
        Set the patch linewidth in points.

        Parameters
        ----------
        w : float or None
            The linewidth in points. If None, defaults to the value from
            Matplotlib's configuration.

        """
        # 如果 w 是 None，则使用 Matplotlib 配置中的默认线宽
        if w is None:
            w = mpl.rcParams['patch.linewidth']
        # 将线宽转换为浮点数并存储
        self._linewidth = float(w)
        # 根据线宽调整虚线样式
        self._dash_pattern = mlines._scale_dashes(
            *self._unscaled_dash_pattern, w)
        # 设置标记为过时
        self.stale = True

    def set_linestyle(self, ls):
        """
        Set the patch linestyle.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            The line style as a string or dash tuple.

        Notes
        -----
        This method sets the linestyle for the patch and adjusts the dash
        pattern accordingly.

        """
        # 如果 ls 是 None，则默认为实线
        if ls is None:
            ls = "solid"
        # 如果 ls 是空字符串或 'none'，则转换为 'None'
        if ls in [' ', '', 'none']:
            ls = 'None'
        # 设置线条样式
        self._linestyle = ls
        # 获取未缩放的虚线模式
        self._unscaled_dash_pattern = mlines._get_dash_pattern(ls)
        # 根据线宽调整虚线样式
        self._dash_pattern = mlines._scale_dashes(
            *self._unscaled_dash_pattern, self._linewidth)
        # 设置标记为过时
        self.stale = True

    def set_fill(self, b):
        """
        Set whether to fill the patch.

        Parameters
        ----------
        b : bool
            Whether to fill the patch (True) or not (False).

        """
        # 设置是否填充 patch
        self._fill = bool(b)
        # 恢复原始的面和边颜色
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)
        # 设置标记为过时
        self.stale = True

    def get_fill(self):
        """
        Return whether the patch is filled.

        Returns
        -------
        bool
            True if the patch is filled, False otherwise.

        """
        # 返回是否填充 patch
        return self._fill

    # Make fill a property so as to preserve the long-standing
    # but somewhat inconsistent behavior in which fill was an
    # attribute.
    fill = property(get_fill, set_fill)

    @_docstring.interpd
    # 设置箭头或图形的端点风格（CapStyle）。
    # 默认情况下，`.FancyArrowPatch` 使用 'round' 风格，而其他图形使用 'butt' 风格。
    def set_capstyle(self, s):
        # 创建一个 CapStyle 对象，用于表示端点风格
        cs = CapStyle(s)
        # 将对象的端点风格设置为指定的风格
        self._capstyle = cs
        # 设置 stale 标志为 True，表示对象状态已过时
        self.stale = True

    # 获取当前的端点风格（CapStyle）。
    def get_capstyle(self):
        """Return the capstyle."""
        # 返回当前端点风格的名称
        return self._capstyle.name

    # 设置线条的连接风格（JoinStyle）。
    # 默认情况下，`.FancyArrowPatch` 使用 'round' 风格，而其他图形使用 'miter' 风格。
    @_docstring.interpd
    def set_joinstyle(self, s):
        # 创建一个 JoinStyle 对象，用于表示连接风格
        js = JoinStyle(s)
        # 将对象的连接风格设置为指定的风格
        self._joinstyle = js
        # 设置 stale 标志为 True，表示对象状态已过时
        self.stale = True

    # 获取当前的连接风格（JoinStyle）。
    def get_joinstyle(self):
        """Return the joinstyle."""
        # 返回当前连接风格的名称
        return self._joinstyle.name

    # 设置图案填充样式（hatching pattern）。
    # 可选的填充样式包括斜线、反斜线、垂直线、水平线、交叉线、对角线交叉、小圆、大圆、点、星等。
    def set_hatch(self, hatch):
        r"""
        设置图案填充样式（hatching pattern）。

        *hatch* 可以是以下之一::

          /   - 斜线
          \   - 反斜线
          |   - 垂直线
          -   - 水平线
          +   - 交叉线
          x   - 对角线交叉
          o   - 小圆
          O   - 大圆
          .   - 点
          *   - 星号

        可以组合使用多个字母，以便同时显示多种填充样式。如果重复相同的字母，则增加该图案填充的密度。

        Parameters
        ----------
        hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
        """
        # 使用 mhatch._validate_hatch_pattern(hatch) 来验证填充样式（已弃用）。
        mhatch._validate_hatch_pattern(hatch)
        # 将对象的填充样式设置为指定的填充图案
        self._hatch = hatch
        # 设置 stale 标志为 True，表示对象状态已过时
        self.stale = True

    # 获取当前的填充样式（hatching pattern）。
    def get_hatch(self):
        """Return the hatching pattern."""
        # 返回当前填充样式的名称
        return self._hatch
    def _draw_paths_with_artist_properties(
            self, renderer, draw_path_args_list):
        """
        ``draw()`` helper factored out for sharing with `FancyArrowPatch`.

        Configure *renderer* and the associated graphics context *gc*
        from the artist properties, then repeatedly call
        ``renderer.draw_path(gc, *draw_path_args)`` for each tuple
        *draw_path_args* in *draw_path_args_list*.
        """

        # 打开一个名为 'patch' 的渲染组，使用当前对象的唯一标识符作为标识符
        renderer.open_group('patch', self.get_gid())
        
        # 创建一个新的图形上下文
        gc = renderer.new_gc()

        # 设置图形上下文的前景色为 self._edgecolor，并指定其为 RGBA 模式
        gc.set_foreground(self._edgecolor, isRGBA=True)

        # 如果边缘颜色的透明度为 0，或者线型为 'None'，则将线宽设为 0
        lw = self._linewidth
        if self._edgecolor[3] == 0 or self._linestyle == 'None':
            lw = 0
        gc.set_linewidth(lw)
        
        # 设置虚线模式
        gc.set_dashes(*self._dash_pattern)
        
        # 设置线段端点样式和连接样式
        gc.set_capstyle(self._capstyle)
        gc.set_joinstyle(self._joinstyle)

        # 设置是否使用抗锯齿
        gc.set_antialiased(self._antialiased)
        
        # 设置图形上下文的裁剪路径
        self._set_gc_clip(gc)
        
        # 设置URL
        gc.set_url(self._url)
        
        # 设置是否对齐像素网格
        gc.set_snap(self.get_snap())
        
        # 设置透明度
        gc.set_alpha(self._alpha)

        # 如果有阴影填充样式，则设置阴影填充和阴影填充颜色
        if self._hatch:
            gc.set_hatch(self._hatch)
            gc.set_hatch_color(self._hatch_color)

        # 如果有草图参数，则设置草图参数
        if self.get_sketch_params() is not None:
            gc.set_sketch_params(*self.get_sketch_params())

        # 如果有路径效果，则创建路径效果渲染器
        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        # 对每个路径参数元组调用 renderer.draw_path(gc, *draw_path_args)
        for draw_path_args in draw_path_args_list:
            renderer.draw_path(gc, *draw_path_args)

        # 恢复图形上下文的先前状态
        gc.restore()
        
        # 关闭 'patch' 渲染组
        renderer.close_group('patch')
        
        # 将标志位 stale 设置为 False，表示图形对象是最新的
        self.stale = False
class Shadow(Patch):
    # 定义一个名为 Shadow 的类，继承自 Patch 类
    def __str__(self):
        # 返回一个字符串表示形式，格式为 "Shadow(某个补丁对象)"
        return f"Shadow({self.patch})"

    @_docstring.dedent_interpd
    # 使用 dedent_interpd 进行文档字符串格式化
    def __init__(self, patch, ox, oy, *, shade=0.7, **kwargs):
        """
        创建给定 *patch* 的阴影。

        默认情况下，阴影将具有与 *patch* 相同的面部颜色，但会变暗。阴影的暗度可以通过 *shade* 参数控制。

        Parameters
        ----------
        patch : `~matplotlib.patches.Patch`
            要为其创建阴影的补丁对象。
        ox, oy : float
            阴影在数据坐标中的偏移量，以 dpi/72 为比例因子。
        shade : float, 默认值: 0.7
            阴影的暗度与原始颜色的关系。如果为 1，则阴影为黑色；如果为 0，则阴影与 *patch* 的颜色相同。

            .. versionadded:: 3.8

        **kwargs
            阴影补丁的属性。支持的键包括:

            %(Patch:kwdoc)s
        """
        super().__init__()
        # 调用父类的构造方法

        self.patch = patch
        # 设置 self.patch 属性为传入的 patch 参数

        self._ox, self._oy = ox, oy
        # 设置阴影的偏移量属性

        self._shadow_transform = transforms.Affine2D()
        # 创建一个仿射变换对象用于处理阴影的变换

        self.update_from(self.patch)
        # 从传入的 patch 更新阴影的属性

        if not 0 <= shade <= 1:
            # 检查 shade 参数的合法性
            raise ValueError("shade must be between 0 and 1.")

        # 计算阴影的颜色，通过调整原始颜色来实现变暗效果
        color = (1 - shade) * np.asarray(colors.to_rgb(self.patch.get_facecolor()))
        
        self.update({'facecolor': color, 'edgecolor': color, 'alpha': 0.5,
                     # 将阴影补丁放置在继承补丁对象的背后
                     'zorder': np.nextafter(self.patch.zorder, -np.inf),
                     **kwargs})
        # 更新阴影的属性，包括颜色、透明度等

    def _update_transform(self, renderer):
        # 更新阴影的变换
        ox = renderer.points_to_pixels(self._ox)
        oy = renderer.points_to_pixels(self._oy)
        self._shadow_transform.clear().translate(ox, oy)
        # 清除当前变换并平移阴影的变换矩阵

    def get_path(self):
        # 获取阴影补丁的路径
        return self.patch.get_path()

    def get_patch_transform(self):
        # 获取阴影补丁的变换，包括阴影变换的累积效果
        return self.patch.get_patch_transform() + self._shadow_transform

    def draw(self, renderer):
        # 绘制阴影
        self._update_transform(renderer)
        # 更新阴影的变换
        super().draw(renderer)
        # 调用父类的 draw 方法绘制阴影
    # 返回表示矩形对象的字符串表示形式
    def __str__(self):
        # 将矩形的位置、宽度、高度、角度格式化成字符串
        pars = self._x0, self._y0, self._width, self._height, self.angle
        fmt = "Rectangle(xy=(%g, %g), width=%g, height=%g, angle=%g)"
        return fmt % pars

    # 初始化矩形对象
    @_docstring.dedent_interpd
    def __init__(self, xy, width, height, *,
                 angle=0.0, rotation_point='xy', **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            锚点坐标。
        width : float
            矩形的宽度。
        height : float
            矩形的高度。
        angle : float, default: 0
            逆时针旋转的角度（单位为度）。
        rotation_point : {'xy', 'center', (number, number)}, default: 'xy'
            如果是 ``'xy'``，围绕锚点旋转。如果是 ``'center'``，围绕中心旋转。
            如果是包含两个数字的元组，则围绕这个坐标点旋转。

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` 属性
            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        # 初始化矩形的位置、宽度、高度、角度和旋转点
        self._x0 = xy[0]
        self._y0 = xy[1]
        self._width = width
        self._height = height
        self.angle = float(angle)
        self.rotation_point = rotation_point
        # 用于处理矩形选择器中坐标系纵横比不为1时的修正值
        # 补偿因数据坐标系和显示坐标系之间纵横比差异而产生的修正值
        # 通常由 Axes._get_aspect_ratio() 提供
        self._aspect_ratio_correction = 1.0
        self._convert_units()  # 验证输入值的单位转换

    # 返回矩形的路径
    def get_path(self):
        """Return the vertices of the rectangle."""
        return Path.unit_rectangle()

    # 转换矩形的边界单位
    def _convert_units(self):
        """Convert bounds of the rectangle."""
        # 将矩形的边界坐标转换为适当的单位
        x0 = self.convert_xunits(self._x0)
        y0 = self.convert_yunits(self._y0)
        x1 = self.convert_xunits(self._x0 + self._width)
        y1 = self.convert_yunits(self._y0 + self._height)
        return x0, y0, x1, y1
    def get_patch_transform(self):
        """
        Note: This cannot be called until after this has been added to
        an Axes, otherwise unit conversion will fail. This makes it very
        important to call the accessor method and not directly access the
        transformation member variable.
        """
        # 获取当前图形的边界框
        bbox = self.get_bbox()
        # 根据旋转点类型确定旋转点的坐标
        if self.rotation_point == 'center':
            width, height = bbox.x1 - bbox.x0, bbox.y1 - bbox.y0
            rotation_point = bbox.x0 + width / 2., bbox.y0 + height / 2.
        elif self.rotation_point == 'xy':
            rotation_point = bbox.x0, bbox.y0
        else:
            rotation_point = self.rotation_point
        # 返回一系列变换操作的组合，以实现图形的旋转和缩放等变换
        return transforms.BboxTransformTo(bbox) \
                + transforms.Affine2D() \
                .translate(-rotation_point[0], -rotation_point[1]) \
                .scale(1, self._aspect_ratio_correction) \
                .rotate_deg(self.angle) \
                .scale(1, 1 / self._aspect_ratio_correction) \
                .translate(*rotation_point)

    @property
    def rotation_point(self):
        """
        The rotation point of the patch.
        """
        return self._rotation_point

    @rotation_point.setter
    def rotation_point(self, value):
        """
        Setter for rotation_point attribute. Validates the value.
        """
        if value in ['center', 'xy'] or (
                isinstance(value, tuple) and len(value) == 2 and
                isinstance(value[0], Real) and isinstance(value[1], Real)
                ):
            self._rotation_point = value
        else:
            raise ValueError("`rotation_point` must be one of "
                             "{'xy', 'center', (number, number)}.")

    def get_x(self):
        """
        Return the left coordinate of the rectangle.
        """
        return self._x0

    def get_y(self):
        """
        Return the bottom coordinate of the rectangle.
        """
        return self._y0

    def get_xy(self):
        """
        Return the left and bottom coords of the rectangle as a tuple.
        """
        return self._x0, self._y0

    def get_corners(self):
        """
        Return the corners of the rectangle, moving anti-clockwise from
        (x0, y0).
        """
        # 返回矩形的四个角点的坐标，按逆时针方向排序
        return self.get_patch_transform().transform(
            [(0, 0), (1, 0), (1, 1), (0, 1)])

    def get_center(self):
        """
        Return the centre of the rectangle.
        """
        # 返回矩形的中心点的坐标
        return self.get_patch_transform().transform((0.5, 0.5))

    def get_width(self):
        """
        Return the width of the rectangle.
        """
        return self._width

    def get_height(self):
        """
        Return the height of the rectangle.
        """
        return self._height

    def get_angle(self):
        """
        Get the rotation angle in degrees.
        """
        return self.angle

    def set_x(self, x):
        """
        Set the left coordinate of the rectangle.
        """
        self._x0 = x
        self.stale = True

    def set_y(self, y):
        """
        Set the bottom coordinate of the rectangle.
        """
        self._y0 = y
        self.stale = True
    def set_angle(self, angle):
        """
        Set the rotation angle in degrees.

        The rotation is performed anti-clockwise around *xy*.
        """
        # 设置矩形对象的旋转角度
        self.angle = angle
        # 标记对象为需要更新状态
        self.stale = True

    def set_xy(self, xy):
        """
        Set the left and bottom coordinates of the rectangle.

        Parameters
        ----------
        xy : (float, float)
            Tuple containing the new left and bottom coordinates.
        """
        # 将给定的坐标元组分别赋给矩形对象的左侧和底部坐标
        self._x0, self._y0 = xy
        # 标记对象为需要更新状态
        self.stale = True

    def set_width(self, w):
        """Set the width of the rectangle."""
        # 设置矩形对象的宽度
        self._width = w
        # 标记对象为需要更新状态
        self.stale = True

    def set_height(self, h):
        """Set the height of the rectangle."""
        # 设置矩形对象的高度
        self._height = h
        # 标记对象为需要更新状态
        self.stale = True

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle as *left*, *bottom*, *width*, *height*.

        The values may be passed as separate parameters or as a tuple::

            set_bounds(left, bottom, width, height)
            set_bounds((left, bottom, width, height))

        .. ACCEPTS: (left, bottom, width, height)
        """
        # 根据参数数量不同解析并设置矩形对象的边界参数
        if len(args) == 1:
            l, b, w, h = args[0]
        else:
            l, b, w, h = args
        # 分别设置矩形对象的左侧、底部、宽度和高度
        self._x0 = l
        self._y0 = b
        self._width = w
        self._height = h
        # 标记对象为需要更新状态
        self.stale = True

    def get_bbox(self):
        """Return the `.Bbox`."""
        # 返回与矩形对象关联的边界框对象
        return transforms.Bbox.from_extents(*self._convert_units())

    xy = property(get_xy, set_xy)
class RegularPolygon(Patch):
    """A regular polygon patch."""

    def __str__(self):
        # 返回描述正多边形对象的格式化字符串
        s = "RegularPolygon((%g, %g), %d, radius=%g, orientation=%g)"
        return s % (self.xy[0], self.xy[1], self.numvertices, self.radius,
                    self.orientation)

    @_docstring.dedent_interpd
    def __init__(self, xy, numVertices, *,
                 radius=5, orientation=0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            多边形的中心位置。

        numVertices : int
            多边形的顶点数。

        radius : float
            中心到每个顶点的距离。

        orientation : float
            多边形的旋转角度（弧度）。

        **kwargs
            `Patch` 类的属性:

            %(Patch:kwdoc)s
        """
        # 初始化 RegularPolygon 对象
        self.xy = xy
        self.numvertices = numVertices
        self.orientation = orientation
        self.radius = radius
        # 创建一个单位正多边形的路径
        self._path = Path.unit_regular_polygon(numVertices)
        self._patch_transform = transforms.Affine2D()
        super().__init__(**kwargs)

    def get_path(self):
        # 返回多边形的路径对象
        return self._path

    def get_patch_transform(self):
        # 返回应用到多边形的变换对象
        return self._patch_transform.clear() \
            .scale(self.radius) \
            .rotate(self.orientation) \
            .translate(*self.xy)


class PathPatch(Patch):
    """A general polycurve path patch."""

    _edge_default = True

    def __str__(self):
        # 返回描述路径对象的格式化字符串
        s = "PathPatch%d((%g, %g) ...)"
        return s % (len(self._path.vertices), *tuple(self._path.vertices[0]))

    @_docstring.dedent_interpd
    def __init__(self, path, **kwargs):
        """
        *path* is a `.Path` object.

        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        # 初始化 PathPatch 对象
        self._path = path

    def get_path(self):
        # 返回路径对象
        return self._path

    def set_path(self, path):
        # 设置路径对象
        self._path = path


class StepPatch(PathPatch):
    """
    A path patch describing a stepwise constant function.

    By default, the path is not closed and starts and stops at
    baseline value.
    """

    _edge_default = False

    @_docstring.dedent_interpd
    def __init__(self, values, edges, *,
                 orientation='vertical', baseline=0, **kwargs):
        """
        Parameters
        ----------
        values : array-like
            The step heights.

        edges : array-like
            The edge positions, with ``len(edges) == len(vals) + 1``,
            between which the curve takes on vals values.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            The direction of the steps. Vertical means that *values* are
            along the y-axis, and edges are along the x-axis.

        baseline : float, array-like or None, default: 0
            The bottom value of the bounding edges or when
            ``fill=True``, position of lower edge. If *fill* is
            True or an array is passed to *baseline*, a closed
            path is drawn.

        **kwargs
            `Patch` properties:

            %(Patch:kwdoc)s
        """
        # 设置步骤图的方向，垂直或水平
        self.orientation = orientation
        # 将边界位置转换为NumPy数组
        self._edges = np.asarray(edges)
        # 将步骤高度转换为NumPy数组
        self._values = np.asarray(values)
        # 如果指定了基准线，则转换为NumPy数组；否则设为None
        self._baseline = np.asarray(baseline) if baseline is not None else None
        # 更新路径信息
        self._update_path()
        # 调用父类的构造函数，传递路径和其他关键字参数
        super().__init__(self._path, **kwargs)
    def _update_path(self):
        # 检查边缘数组中是否存在 NaN 值，如果有则引发异常
        if np.isnan(np.sum(self._edges)):
            raise ValueError('Nan values in "edges" are disallowed')
        # 检查边缘数组大小与值数组大小是否匹配，如果不匹配则引发异常，同时提供详细信息
        if self._edges.size - 1 != self._values.size:
            raise ValueError('Size mismatch between "values" and "edges". '
                             "Expected `len(values) + 1 == len(edges)`, but "
                             f"`len(values) = {self._values.size}` and "
                             f"`len(edges) = {self._edges.size}`.")
        # 初始化空数组以支持空的阶梯图
        verts, codes = [np.empty((0, 2))], [np.empty(0, dtype=Path.code_type)]

        # 创建一个 NaN 掩码以处理 NaN 值
        _nan_mask = np.isnan(self._values)
        # 如果基准线不为 None，则将基准线的 NaN 值也加入掩码中
        if self._baseline is not None:
            _nan_mask |= np.isnan(self._baseline)
        # 使用连续的非 NaN 区域来构建阶梯图的路径
        for idx0, idx1 in cbook.contiguous_regions(~_nan_mask):
            # 重复边缘值和数值，用于构建路径
            x = np.repeat(self._edges[idx0:idx1+1], 2)
            y = np.repeat(self._values[idx0:idx1], 2)
            # 如果没有指定基准线，则在 y 数组的前后各添加一个值，用于闭合路径
            if self._baseline is None:
                y = np.concatenate([y[:1], y, y[-1:]])
            # 如果基准线是一个标量值，则在 y 数组的前后各添加这个值，用于闭合路径
            elif self._baseline.ndim == 0:  # single baseline value
                y = np.concatenate([[self._baseline], y, [self._baseline]])
            # 如果基准线是一个数组，则在路径的两端添加基准线的反转值，并对 x 值进行相应处理
            elif self._baseline.ndim == 1:  # baseline array
                base = np.repeat(self._baseline[idx0:idx1], 2)[::-1]
                x = np.concatenate([x, x[::-1]])
                y = np.concatenate([base[-1:], y, base[:1],
                                    base[:1], base, base[-1:]])
            else:  # no baseline
                raise ValueError('Invalid `baseline` specified')
            # 根据方向确定 xy 的顺序，并将 xy 添加到 verts 中
            if self.orientation == 'vertical':
                xy = np.column_stack([x, y])
            else:
                xy = np.column_stack([y, x])
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO]*(len(xy)-1))
        # 构建路径对象并赋值给 _path 属性
        self._path = Path(np.concatenate(verts), np.concatenate(codes))

    def get_data(self):
        """获取 `.StepPatch` 的值，边缘和基准线，返回命名元组 StairData。"""
        StairData = namedtuple('StairData', 'values edges baseline')
        return StairData(self._values, self._edges, self._baseline)

    def set_data(self, values=None, edges=None, baseline=None):
        """
        设置 `.StepPatch` 的值，边缘和基准线。

        Parameters
        ----------
        values : 1D array-like or None
            如果传入 None，则不更新 values。
        edges : 1D array-like, optional
        baseline : float, 1D array-like or None
        """
        # 如果没有任何参数被设置，则引发异常
        if values is None and edges is None and baseline is None:
            raise ValueError("Must set *values*, *edges* or *baseline*.")
        # 如果 values 参数不为 None，则将其转换为 NumPy 数组并赋值给 _values 属性
        if values is not None:
            self._values = np.asarray(values)
        # 如果 edges 参数不为 None，则将其转换为 NumPy 数组并赋值给 _edges 属性
        if edges is not None:
            self._edges = np.asarray(edges)
        # 如果 baseline 参数不为 None，则将其转换为 NumPy 数组并赋值给 _baseline 属性
        if baseline is not None:
            self._baseline = np.asarray(baseline)
        # 更新路径并将 stale 属性设置为 True
        self._update_path()
        self.stale = True
# 定义一个多边形的类，继承自Patch类
class Polygon(Patch):
    """A general polygon patch."""

    # 返回多边形对象的字符串表示形式
    def __str__(self):
        # 如果多边形路径有顶点
        if len(self._path.vertices):
            s = "Polygon%d((%g, %g) ...)"
            # 返回多边形顶点的描述字符串
            return s % (len(self._path.vertices), *self._path.vertices[0])
        else:
            # 如果没有顶点，则返回空多边形的描述字符串
            return "Polygon0()"

    # 初始化方法，设置多边形的顶点坐标和其他参数
    @_docstring.dedent_interpd
    def __init__(self, xy, *, closed=True, **kwargs):
        """
        Parameters
        ----------
        xy : (N, 2) array
            多边形的顶点坐标数组，每行表示一个顶点的 (x, y) 坐标。

        closed : bool, default: True
            指示多边形是否闭合（即起始点和结束点相同）。

        **kwargs
            其他关键字参数，参考 Patch 类的文档。
        """
        super().__init__(**kwargs)
        # 是否闭合的标志
        self._closed = closed
        # 设置多边形的顶点坐标
        self.set_xy(xy)

    # 返回多边形的路径对象 `.Path`
    def get_path(self):
        """Get the `.Path` of the polygon."""
        return self._path

    # 返回多边形是否闭合的状态
    def get_closed(self):
        """Return whether the polygon is closed."""
        return self._closed

    # 设置多边形是否闭合的状态
    def set_closed(self, closed):
        """
        Set whether the polygon is closed.

        Parameters
        ----------
        closed : bool
            True if the polygon is closed
        """
        # 如果当前闭合状态与新状态相同，则直接返回
        if self._closed == bool(closed):
            return
        # 更新闭合状态，并重新设置顶点坐标以更新路径
        self._closed = bool(closed)
        self.set_xy(self.get_xy())
        # 标记对象为过时，需要重新绘制
        self.stale = True

    # 获取多边形的顶点坐标
    def get_xy(self):
        """
        Get the vertices of the path.

        Returns
        -------
        (N, 2) array
            The coordinates of the vertices.
        """
        return self._path.vertices

    # 设置多边形的顶点坐标
    def set_xy(self, xy):
        """
        Set the vertices of the polygon.

        Parameters
        ----------
        xy : (N, 2) array-like
            The coordinates of the vertices.

        Notes
        -----
        Unlike `.Path`, we do not ignore the last input vertex. If the
        polygon is meant to be closed, and the last point of the polygon is not
        equal to the first, we assume that the user has not explicitly passed a
        ``CLOSEPOLY`` vertex, and add it ourselves.
        """
        # 将输入的顶点坐标转换为 NumPy 数组
        xy = np.asarray(xy)
        nverts, _ = xy.shape
        # 如果多边形需要闭合
        if self._closed:
            # 如果顶点数为1或大于1且第一个和最后一个顶点不相同，则添加第一个顶点以闭合多边形
            if nverts == 1 or nverts > 1 and (xy[0] != xy[-1]).any():
                xy = np.concatenate([xy, [xy[0]]])
        else:
            # 如果不需要闭合，并且最后一个顶点与第一个顶点相同，则删除最后一个顶点
            if nverts > 2 and (xy[0] == xy[-1]).all():
                xy = xy[:-1]
        # 创建多边形的路径对象，根据是否闭合
        self._path = Path(xy, closed=self._closed)
        # 标记对象为过时，需要重新绘制
        self.stale = True

    # 使用 property 定义 xy 属性，使得可以通过 get_xy 和 set_xy 方法获取和设置顶点坐标
    xy = property(get_xy, set_xy,
                  doc='The vertices of the path as a (N, 2) array.')


# 定义一个楔形的类，继承自 Patch 类
class Wedge(Patch):
    """Wedge shaped patch."""
    # 返回表示当前对象的字符串形式，使用给定的格式字符串和参数来格式化输出
    def __str__(self):
        pars = (self.center[0], self.center[1], self.r,
                self.theta1, self.theta2, self.width)
        fmt = "Wedge(center=(%g, %g), r=%g, theta1=%g, theta2=%g, width=%s)"
        return fmt % pars
    
    # 初始化一个楔形图形对象，其圆心为center，半径为r，扫过角度从theta1到theta2（单位为度）。
    # 如果指定了width，则绘制从内半径r-width到外半径r的部分楔形。
    # 支持的关键字参数详见Patch类的文档说明。
    def __init__(self, center, r, theta1, theta2, *, width=None, **kwargs):
        super().__init__(**kwargs)
        # 设置楔形图形的圆心和半径，以及扫过的角度范围
        self.center = center
        self.r, self.width = r, width
        self.theta1, self.theta2 = theta1, theta2
        # 初始化楔形图形的变换为单位矩阵
        self._patch_transform = transforms.IdentityTransform()
        # 重新计算楔形图形的路径
        self._recompute_path()
    
    # 重新计算楔形图形的路径
    def _recompute_path(self):
        # 如果内外圆环连接，则扫过的角度接近360度
        if abs((self.theta2 - self.theta1) - 360) <= 1e-12:
            theta1, theta2 = 0, 360
            connector = Path.MOVETO
        else:
            theta1, theta2 = self.theta1, self.theta2
            connector = Path.LINETO
    
        # 构建外圆环的路径
        arc = Path.arc(theta1, theta2)
    
        if self.width is not None:
            # 部分环形需要绘制外圆环，然后是反向缩放的内圆环
            v1 = arc.vertices
            v2 = arc.vertices[::-1] * (self.r - self.width) / self.r
            v = np.concatenate([v1, v2, [(0, 0)]])
            c = [*arc.codes, connector, *arc.codes[1:], Path.CLOSEPOLY]
        else:
            # 楔形图形不需要内圆环
            v = np.concatenate([arc.vertices, [(0, 0), (0, 0)]])
            c = [*arc.codes, connector, Path.CLOSEPOLY]
    
        # 将楔形图形移动和缩放到最终位置
        self._path = Path(v * self.r + self.center, c)
    
    # 设置楔形图形的圆心位置
    def set_center(self, center):
        self._path = None
        self.center = center
        self.stale = True
    
    # 设置楔形图形的半径
    def set_radius(self, radius):
        self._path = None
        self.r = radius
        self.stale = True
    
    # 设置楔形图形的起始角度
    def set_theta1(self, theta1):
        self._path = None
        self.theta1 = theta1
        self.stale = True
    
    # 设置楔形图形的结束角度
    def set_theta2(self, theta2):
        self._path = None
        self.theta2 = theta2
        self.stale = True
    
    # 设置楔形图形的宽度（内外半径之差）
    def set_width(self, width):
        self._path = None
        self.width = width
        self.stale = True
    
    # 获取楔形图形的路径
    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path
# COVERAGE NOTE: Not used internally or from examples
# 定义 Arrow 类，继承自 Patch 类
class Arrow(Patch):
    """An arrow patch."""

    # 返回字符串描述 Arrow 对象
    def __str__(self):
        return "Arrow()"

    # 箭头路径的定义，表示箭头的形状
    _path = Path._create_closed([
        [0.0, 0.1], [0.0, -0.1], [0.8, -0.1], [0.8, -0.3], [1.0, 0.0],
        [0.8, 0.3], [0.8, 0.1]])

    # 初始化方法，用于创建 Arrow 对象
    @_docstring.dedent_interpd
    def __init__(self, x, y, dx, dy, *, width=1.0, **kwargs):
        """
        Draws an arrow from (*x*, *y*) to (*x* + *dx*, *y* + *dy*).
        The width of the arrow is scaled by *width*.

        Parameters
        ----------
        x : float
            x coordinate of the arrow tail.
        y : float
            y coordinate of the arrow tail.
        dx : float
            Arrow length in the x direction.
        dy : float
            Arrow length in the y direction.
        width : float, default: 1
            Scale factor for the width of the arrow. With a default value of 1,
            the tail width is 0.2 and head width is 0.6.
        **kwargs
            Keyword arguments control the `Patch` properties:

            %(Patch:kwdoc)s

        See Also
        --------
        FancyArrow
            Patch that allows independent control of the head and tail
            properties.
        """
        # 调用父类的初始化方法，设置 Patch 对象的属性
        super().__init__(**kwargs)
        # 设置 Arrow 对象的数据，包括位置和大小
        self.set_data(x, y, dx, dy, width)

    # 获取箭头的路径
    def get_path(self):
        return self._path

    # 获取 Patch 对象的变换信息
    def get_patch_transform(self):
        return self._patch_transform

    # 设置 Arrow 对象的数据，更新位置和大小信息
    def set_data(self, x=None, y=None, dx=None, dy=None, width=None):
        """
        Set `.Arrow` x, y, dx, dy and width.
        Values left as None will not be updated.

        Parameters
        ----------
        x, y : float or None, default: None
            The x and y coordinates of the arrow base.

        dx, dy : float or None, default: None
            The length of the arrow along x and y direction.

        width : float or None, default: None
            Width of full arrow tail.
        """
        # 根据传入的参数更新箭头的位置和大小信息
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y
        if dx is not None:
            self._dx = dx
        if dy is not None:
            self._dy = dy
        if width is not None:
            self._width = width
        # 设置 Patch 的变换，根据箭头的位置和方向来进行缩放、旋转和平移
        self._patch_transform = (
            transforms.Affine2D()
            .scale(np.hypot(self._dx, self._dy), self._width)
            .rotate(np.arctan2(self._dy, self._dx))
            .translate(self._x, self._y)
            .frozen())


# 定义 FancyArrow 类，继承自 Polygon 类
class FancyArrow(Polygon):
    """
    Like Arrow, but lets you set head width and head height independently.
    """

    # 返回字符串描述 FancyArrow 对象
    def __str__(self):
        return "FancyArrow()"

    # 设置默认边缘为 True
    _edge_default = True

    # 初始化方法，用于创建 FancyArrow 对象
    @_docstring.dedent_interpd
    def __init__(self, x, y, dx, dy, *,
                 width=0.001, length_includes_head=False, head_width=None,
                 head_length=None, shape='full', overhang=0,
                 head_starts_at_zero=False, **kwargs):
        """
        Parameters
        ----------
        x, y : float
            箭头基底的 x 和 y 坐标。

        dx, dy : float
            箭头沿着 x 和 y 方向的长度。

        width : float, default: 0.001
            箭尾的宽度。

        length_includes_head : bool, default: False
            如果为 True，则在计算长度时包括箭头头部。

        head_width : float or None, default: 3*width
            箭头头部的总宽度。

        head_length : float or None, default: 1.5*head_width
            箭头头部的长度。

        shape : {'full', 'left', 'right'}, default: 'full'
            绘制箭头的方式：全箭头、左半箭头、右半箭头。

        overhang : float, default: 0
            箭头向后扫过的比例（0 表示三角形形状）。可以是负数或大于 1。

        head_starts_at_zero : bool, default: False
            如果为 True，则箭头头部从坐标 0 开始绘制，而不是以坐标 0 结束。

        **kwargs
            `.Patch` 属性：

            %(Patch:kwdoc)s
        """
        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._width = width
        self._length_includes_head = length_includes_head
        self._head_width = head_width
        self._head_length = head_length
        self._shape = shape
        self._overhang = overhang
        self._head_starts_at_zero = head_starts_at_zero
        # 根据给定参数生成箭头的顶点坐标
        self._make_verts()
        # 调用父类构造函数，使用生成的顶点坐标和其他参数初始化箭头图形对象
        super().__init__(self.verts, closed=True, **kwargs)
    def set_data(self, *, x=None, y=None, dx=None, dy=None, width=None,
                 head_width=None, head_length=None):
        """
        Set `.FancyArrow` x, y, dx, dy, width, head_with, and head_length.
        Values left as None will not be updated.

        Parameters
        ----------
        x, y : float or None, default: None
            The x and y coordinates of the arrow base.

        dx, dy : float or None, default: None
            The length of the arrow along x and y direction.

        width : float or None, default: None
            Width of full arrow tail.

        head_width : float or None, default: None
            Total width of the full arrow head.

        head_length : float or None, default: None
            Length of arrow head.
        """
        # 如果参数 x 不为 None，则更新箭头基础的 x 坐标
        if x is not None:
            self._x = x
        # 如果参数 y 不为 None，则更新箭头基础的 y 坐标
        if y is not None:
            self._y = y
        # 如果参数 dx 不为 None，则更新箭头在 x 方向上的长度
        if dx is not None:
            self._dx = dx
        # 如果参数 dy 不为 None，则更新箭头在 y 方向上的长度
        if dy is not None:
            self._dy = dy
        # 如果参数 width 不为 None，则更新箭头尾部的宽度
        if width is not None:
            self._width = width
        # 如果参数 head_width 不为 None，则更新箭头头部的总宽度
        if head_width is not None:
            self._head_width = head_width
        # 如果参数 head_length 不为 None，则更新箭头头部的长度
        if head_length is not None:
            self._head_length = head_length
        # 重新生成箭头的顶点坐标
        self._make_verts()
        # 将箭头顶点坐标设置为当前对象的 xy 坐标属性
        self.set_xy(self.verts)
    # 定义一个私有方法 _make_verts，用于计算箭头的顶点坐标
    def _make_verts(self):
        # 如果箭头头部宽度未指定，设置为宽度的三倍
        if self._head_width is None:
            head_width = 3 * self._width
        else:
            head_width = self._head_width
        # 如果箭头头部长度未指定，设置为头部宽度的1.5倍
        if self._head_length is None:
            head_length = 1.5 * head_width
        else:
            head_length = self._head_length

        # 计算箭头的总长度
        distance = np.hypot(self._dx, self._dy)

        # 根据长度是否包含头部来确定箭头的实际长度
        if self._length_includes_head:
            length = distance
        else:
            length = distance + head_length
        
        # 如果长度为0，将顶点数组设为空数组，即不显示箭头
        if not length:
            self.verts = np.empty([0, 2])
        else:
            # 开始绘制水平箭头，箭头尖在 (0, 0) 处
            hw, hl = head_width, head_length
            hs, lw = self._overhang, self._width
            left_half_arrow = np.array([
                [0.0, 0.0],                 # 箭尖
                [-hl, -hw / 2],             # 最左侧
                [-hl * (1 - hs), -lw / 2],  # 与主干相交
                [-length, -lw / 2],         # 左下角
                [-length, 0],               # 底部
            ])
            
            # 如果不包含头部，向右平移头部长度
            if not self._length_includes_head:
                left_half_arrow += [head_length, 0]
            # 如果头部从0开始，再向右平移一个头部长度的一半
            if self._head_starts_at_zero:
                left_half_arrow += [head_length / 2, 0]
            
            # 根据箭头形状确定坐标
            if self._shape == 'left':
                coords = left_half_arrow
            else:
                right_half_arrow = left_half_arrow * [1, -1]
                if self._shape == 'right':
                    coords = right_half_arrow
                elif self._shape == 'full':
                    # 半箭头包含主干的中点，完整箭头应省略这一部分
                    # 包含两次可能导致 xpdf 的问题
                    coords = np.concatenate([left_half_arrow[:-1],
                                             right_half_arrow[-2::-1]])
                else:
                    raise ValueError(f"Got unknown shape: {self._shape!r}")
            
            # 计算旋转矩阵以及最终顶点坐标
            if distance != 0:
                cx = self._dx / distance
                sx = self._dy / distance
            else:
                # 处理除以零的情况
                cx, sx = 0, 1
            M = [[cx, sx], [-sx, cx]]
            self.verts = np.dot(coords, M) + [
                self._x + self._dx,
                self._y + self._dy,
            ]
# 更新文档字符串的插值字典，将“FancyArrow”条目更新为修改后的内容
_docstring.interpd.update(
    FancyArrow="\n".join(
        (inspect.getdoc(FancyArrow.__init__) or "").splitlines()[2:]))

# 定义一个继承自RegularPolygon的CirclePolygon类，用于近似表示圆形的多边形图形
class CirclePolygon(RegularPolygon):
    """A polygon-approximation of a circle patch."""

    def __str__(self):
        # 返回描述此CirclePolygon对象的字符串，包括中心坐标、半径和顶点数的信息
        s = "CirclePolygon((%g, %g), radius=%g, resolution=%d)"
        return s % (self.xy[0], self.xy[1], self.radius, self.numvertices)

    @_docstring.dedent_interpd
    def __init__(self, xy, radius=5, *,
                 resolution=20,  # 多边形的顶点数
                 **kwargs):
        """
        创建一个以xy=(x, y)为中心、给定半径的圆。

        该圆由具有指定顶点数的正多边形近似表示。对于使用样条线绘制的更平滑的圆，请参见`Circle`。

        有效的关键字参数有:

        %(Patch:kwdoc)s
        """
        # 调用父类RegularPolygon的初始化方法来初始化CirclePolygon对象
        super().__init__(
            xy, resolution, radius=radius, orientation=0, **kwargs)


# 定义一个继承自Patch的Ellipse类，表示一个无标度的椭圆形
class Ellipse(Patch):
    """A scale-free ellipse."""

    def __str__(self):
        # 返回描述此Ellipse对象的字符串，包括中心坐标、宽度、高度和角度的信息
        pars = (self._center[0], self._center[1],
                self.width, self.height, self.angle)
        fmt = "Ellipse(xy=(%s, %s), width=%s, height=%s, angle=%s)"
        return fmt % pars

    @_docstring.dedent_interpd
    def __init__(self, xy, width, height, *, angle=0, **kwargs):
        """
        参数
        ----------
        xy : (float, float)
            椭圆中心的xy坐标。
        width : float
            水平轴的总长度（直径）。
        height : float
            垂直轴的总长度（直径）。
        angle : float, 默认值: 0
            逆时针旋转的角度（以度为单位）。

        注意
        -----
        有效的关键字参数有:

        %(Patch:kwdoc)s
        """
        # 调用父类Patch的初始化方法来初始化Ellipse对象
        super().__init__(**kwargs)

        # 设置椭圆的中心坐标、宽度、高度和角度
        self._center = xy
        self._width, self._height = width, height
        self._angle = angle
        # 设置椭圆的路径为单位圆
        self._path = Path.unit_circle()
        # 用于处理Axes的长宽比不等于1时的椭圆选择器的修正
        # 补偿数据坐标和显示坐标系统之间长宽比的差异
        self._aspect_ratio_correction = 1.0
        # 注意: 这个属性在添加到Axes之前无法计算
        self._patch_transform = transforms.IdentityTransform()
    def _recompute_transform(self):
        """
        Notes
        -----
        This cannot be called until after this has been added to an Axes,
        otherwise unit conversion will fail. This makes it very important to
        call the accessor method and not directly access the transformation
        member variable.
        """
        # 计算变换前的中心点坐标，并进行单位转换
        center = (self.convert_xunits(self._center[0]),
                  self.convert_yunits(self._center[1]))
        # 计算宽度，并进行单位转换
        width = self.convert_xunits(self._width)
        # 计算高度，并进行单位转换
        height = self.convert_yunits(self._height)
        # 创建一个 Affine2D 变换对象，进行缩放、旋转和平移操作
        self._patch_transform = transforms.Affine2D() \
            .scale(width * 0.5, height * 0.5 * self._aspect_ratio_correction) \
            .rotate_deg(self.angle) \
            .scale(1, 1 / self._aspect_ratio_correction) \
            .translate(*center)

    def get_path(self):
        """Return the path of the ellipse."""
        return self._path

    def get_patch_transform(self):
        # 重新计算变换并返回变换对象
        self._recompute_transform()
        return self._patch_transform

    def set_center(self, xy):
        """
        Set the center of the ellipse.

        Parameters
        ----------
        xy : (float, float)
            Coordinates of the new center.
        """
        self._center = xy
        self.stale = True

    def get_center(self):
        """Return the center of the ellipse."""
        return self._center

    center = property(get_center, set_center)

    def set_width(self, width):
        """
        Set the width of the ellipse.

        Parameters
        ----------
        width : float
            New width value.
        """
        self._width = width
        self.stale = True

    def get_width(self):
        """
        Return the width of the ellipse.
        """
        return self._width

    width = property(get_width, set_width)

    def set_height(self, height):
        """
        Set the height of the ellipse.

        Parameters
        ----------
        height : float
            New height value.
        """
        self._height = height
        self.stale = True

    def get_height(self):
        """Return the height of the ellipse."""
        return self._height

    height = property(get_height, set_height)

    def set_angle(self, angle):
        """
        Set the angle of the ellipse.

        Parameters
        ----------
        angle : float
            New angle value in degrees.
        """
        self._angle = angle
        self.stale = True

    def get_angle(self):
        """Return the angle of the ellipse."""
        return self._angle

    angle = property(get_angle, set_angle)

    def get_corners(self):
        """
        Return the corners of the ellipse bounding box.

        The bounding box orientation is moving anti-clockwise from the
        lower left corner defined before rotation.
        """
        # 获取椭圆边界框的四个角点坐标，并根据当前变换进行转换
        return self.get_patch_transform().transform(
            [(-1, -1), (1, -1), (1, 1), (-1, 1)])
    # 返回椭圆的顶点坐标。

    # 如果椭圆的宽度小于高度，则顶点为单位正方向的坐标变换后的结果
    def get_vertices(self):
        ret = self.get_patch_transform().transform([(0, 1), (0, -1)])
        # 如果椭圆的宽度大于等于高度，则顶点为单位正方向的坐标变换后的结果
        # 使用 get_patch_transform() 方法获取图形的变换矩阵，并对指定的坐标进行变换
        return [tuple(x) for x in ret]

    # 返回椭圆的共顶点坐标。

    # 如果椭圆的宽度小于高度，则共顶点为单位正方向的坐标变换后的结果
    def get_co_vertices(self):
        ret = self.get_patch_transform().transform([(1, 0), (-1, 0)])
        # 如果椭圆的宽度大于等于高度，则共顶点为单位正方向的坐标变换后的结果
        # 使用 get_patch_transform() 方法获取图形的变换矩阵，并对指定的坐标进行变换
        return [tuple(x) for x in ret]
class Annulus(Patch):
    """
    An elliptical annulus.
    """

    # 使用修饰符 _docstring.dedent_interpd 处理文档字符串格式化
    @_docstring.dedent_interpd
    def __init__(self, xy, r, width, angle=0.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            annulus 中心的 xy 坐标。
        r : float or (float, float)
            半径或半轴：

            - 如果是 float：外圆的半径。
            - 如果是两个 float：外椭圆的半长轴和半短轴。
        width : float
            环形区域的宽度（厚度）。宽度从外椭圆向内测量，对于内椭圆，半轴由 ``r - width`` 给出。
            *width* 必须小于或等于半短轴。
        angle : float, default: 0
            旋转角度，单位为度（逆时针方向，从正 x 轴开始计算）。对于圆形环，即 *r* 是标量时忽略此参数。
        **kwargs
            关键字参数控制 `Patch` 的属性：

            %(Patch:kwdoc)s
        """
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 设置半径
        self.set_radii(r)
        # 设置中心坐标
        self.center = xy
        # 设置宽度
        self.width = width
        # 设置角度
        self.angle = angle
        # 初始化路径变量为 None
        self._path = None

    def __str__(self):
        # 如果主轴与副轴相等，r 为主轴长度，否则为 (主轴长度, 副轴长度)
        if self.a == self.b:
            r = self.a
        else:
            r = (self.a, self.b)

        # 返回 Annulus 对象的字符串表示
        return "Annulus(xy=(%s, %s), r=%s, width=%s, angle=%s)" % \
                (*self.center, r, self.width, self.angle)

    def set_center(self, xy):
        """
        设置 annulus 的中心坐标。

        Parameters
        ----------
        xy : (float, float)
        """
        # 设置 annulus 的中心坐标
        self._center = xy
        # 将路径变量置为 None，标记为需要更新
        self._path = None
        self.stale = True

    def get_center(self):
        """返回 annulus 的中心坐标。"""
        return self._center

    # 使用 property 方法定义 center 属性
    center = property(get_center, set_center)

    def set_width(self, width):
        """
        设置 annulus 环的宽度（厚度）。

        宽度从外椭圆向内测量。

        Parameters
        ----------
        width : float
        """
        # 如果宽度大于主轴与副轴中的最小值，抛出 ValueError
        if width > min(self.a, self.b):
            raise ValueError(
                'Width of annulus must be less than or equal to semi-minor axis')

        # 设置 annulus 环的宽度
        self._width = width
        # 将路径变量置为 None，标记为需要更新
        self._path = None
        self.stale = True

    def get_width(self):
        """返回 annulus 环的宽度（厚度）。"""
        return self._width

    # 使用 property 方法定义 width 属性
    width = property(get_width, set_width)

    def set_angle(self, angle):
        """
        设置 annulus 的倾斜角度。

        Parameters
        ----------
        angle : float
        """
        # 设置 annulus 的倾斜角度
        self._angle = angle
        # 将路径变量置为 None，标记为需要更新
        self._path = None
        self.stale = True

    def get_angle(self):
        """返回 annulus 的倾斜角度。"""
        return self._angle

    # 使用 property 方法定义 angle 属性
    angle = property(get_angle, set_angle)
    def set_semimajor(self, a):
        """
        Set the semi-major axis *a* of the annulus.

        Parameters
        ----------
        a : float
            The semi-major axis value to be set.
        """
        # Set the semi-major axis of the annulus
        self.a = float(a)
        # Reset the cached path to None to indicate it needs recomputation
        self._path = None
        # Mark the annulus as stale to trigger recomputation when needed
        self.stale = True

    def set_semiminor(self, b):
        """
        Set the semi-minor axis *b* of the annulus.

        Parameters
        ----------
        b : float
            The semi-minor axis value to be set.
        """
        # Set the semi-minor axis of the annulus
        self.b = float(b)
        # Reset the cached path to None to indicate it needs recomputation
        self._path = None
        # Mark the annulus as stale to trigger recomputation when needed
        self.stale = True

    def set_radii(self, r):
        """
        Set the semi-major (*a*) and semi-minor radii (*b*) of the annulus.

        Parameters
        ----------
        r : float or (float, float)
            The radius, or semi-axes:

            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        """
        # Determine and set radii based on the shape of r
        if np.shape(r) == (2,):
            self.a, self.b = r
        elif np.shape(r) == ():
            self.a = self.b = float(r)
        else:
            raise ValueError("Parameter 'r' must be one or two floats.")

        # Reset the cached path to None to indicate it needs recomputation
        self._path = None
        # Mark the annulus as stale to trigger recomputation when needed
        self.stale = True

    def get_radii(self):
        """Return the semi-major and semi-minor radii of the annulus."""
        return self.a, self.b

    radii = property(get_radii, set_radii)

    def _transform_verts(self, verts, a, b):
        """
        Transform vertices using affine transformations.

        Parameters
        ----------
        verts : array-like
            Vertices to be transformed.
        a : float
            Semi-major axis length.
        b : float
            Semi-minor axis length.

        Returns
        -------
        transformed_verts : array-like
            Transformed vertices after applying scaling, rotation, and translation.
        """
        return transforms.Affine2D() \
            .scale(*self._convert_xy_units((a, b))) \
            .rotate_deg(self.angle) \
            .translate(*self._convert_xy_units(self.center)) \
            .transform(verts)

    def _recompute_path(self):
        """
        Recompute the path representing the annulus.

        This method constructs a Path object that defines the shape of the annulus
        using the current semi-major axis (a), semi-minor axis (b), and width (w).
        """
        # Generate a circular arc from 0 to 360 degrees
        arc = Path.arc(0, 360)

        # Transform the arc vertices to define the outer and inner rings of the annulus
        a, b, w = self.a, self.b, self.width
        v1 = self._transform_verts(arc.vertices, a, b)
        v2 = self._transform_verts(arc.vertices[::-1], a - w, b - w)

        # Combine vertices and codes to form the complete Path of the annulus
        v = np.vstack([v1, v2, v1[0, :], (0, 0)])
        c = np.hstack([arc.codes, Path.MOVETO,
                       arc.codes[1:], Path.MOVETO,
                       Path.CLOSEPOLY])
        self._path = Path(v, c)

    def get_path(self):
        """
        Get the Path object representing the annulus.

        Returns
        -------
        _path : matplotlib.path.Path
            Path object defining the shape of the annulus.
        """
        if self._path is None:
            self._recompute_path()
        return self._path
class Circle(Ellipse):
    """
    A circle patch.
    """
    
    def __str__(self):
        # 定义格式化字符串和参数，返回表示圆的字符串
        pars = self.center[0], self.center[1], self.radius
        fmt = "Circle(xy=(%g, %g), radius=%g)"
        return fmt % pars

    @_docstring.dedent_interpd
    def __init__(self, xy, radius=5, **kwargs):
        """
        Create a true circle at center *xy* = (*x*, *y*) with given *radius*.

        Unlike `CirclePolygon` which is a polygonal approximation, this uses
        Bezier splines and is much closer to a scale-free circle.

        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        # 调用父类构造函数初始化椭圆形状，设置圆的半径
        super().__init__(xy, radius * 2, radius * 2, **kwargs)
        self.radius = radius

    def set_radius(self, radius):
        """
        Set the radius of the circle.

        Parameters
        ----------
        radius : float
            圆的半径
        """
        # 设置圆的宽度和高度为半径的两倍，并标记为过时
        self.width = self.height = 2 * radius
        self.stale = True

    def get_radius(self):
        """
        Return the radius of the circle.

        Returns
        -------
        float
            圆的半径
        """
        return self.width / 2.

    radius = property(get_radius, set_radius)


class Arc(Ellipse):
    """
    An elliptical arc, i.e. a segment of an ellipse.

    Due to internal optimizations, the arc cannot be filled.
    """

    def __str__(self):
        # 定义格式化字符串和参数，返回表示椭圆弧的字符串
        pars = (self.center[0], self.center[1], self.width,
                self.height, self.angle, self.theta1, self.theta2)
        fmt = ("Arc(xy=(%g, %g), width=%g, "
               "height=%g, angle=%g, theta1=%g, theta2=%g)")
        return fmt % pars

    @_docstring.dedent_interpd
    def __init__(self, xy, width, height, *,
                 angle=0.0, theta1=0.0, theta2=360.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            椭圆的中心点坐标。

        width : float
            水平轴的长度。

        height : float
            垂直轴的长度。

        angle : float
            椭圆的旋转角度（逆时针方向）。

        theta1, theta2 : float, default: 0, 360
            弧的起始角度和终止角度（以度为单位）。这些值相对于 *angle* 计算，
            例如，如果 *angle* = 45 并且 *theta1* = 90，则绝对起始角度为 135。
            默认 *theta1* = 0，*theta2* = 360，即完整的椭圆。
            弧以逆时针方向绘制。
            大于或等于 360 或小于 0 的角度通过模 360 转换为 [0, 360) 范围内的等效角度。

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties
            大多数 `.Patch` 属性支持作为关键字参数，
            除了 *fill* 和 *facecolor* 因为不支持填充而不包括。

        %(Patch:kwdoc)s
        """
        # 设置默认的 fill 属性为 False，如果用户传入了 fill 属性，则抛出 ValueError
        fill = kwargs.setdefault('fill', False)
        if fill:
            raise ValueError("Arc objects cannot be filled")

        # 调用父类的构造方法来初始化基本的椭圆属性
        super().__init__(xy, width, height, angle=angle, **kwargs)

        # 设置 theta1 和 theta2 的初始值
        self.theta1 = theta1
        self.theta2 = theta2
        # 调用 _theta_stretch 方法计算并设置椭圆的扭曲值及弧线路径
        (self._theta1, self._theta2, self._stretched_width,
         self._stretched_height) = self._theta_stretch()
        self._path = Path.arc(self._theta1, self._theta2)

    @artist.allow_rasterization
    def _update_path(self):
        # 计算新的值并更新 _path，如果有任何值发生变化
        stretched = self._theta_stretch()
        if any(a != b for a, b in zip(
                stretched, (self._theta1, self._theta2, self._stretched_width,
                            self._stretched_height))):
            # 更新扭曲值和弧线路径
            (self._theta1, self._theta2, self._stretched_width,
             self._stretched_height) = stretched
            self._path = Path.arc(self._theta1, self._theta2)
    def _theta_stretch(self):
        # 如果椭圆的宽度和高度不相等，计算绘制角度时考虑拉伸效果

        # 定义函数 theta_stretch，用于根据拉伸比例调整角度
        def theta_stretch(theta, scale):
            theta = np.deg2rad(theta)  # 将角度转换为弧度
            x = np.cos(theta)
            y = np.sin(theta)
            stheta = np.rad2deg(np.arctan2(scale * y, x))
            # arctan2 的取值范围为 [-pi, pi]，我们期望结果在 [0, 2*pi]，因此进行调整
            return (stheta + 360) % 360

        # 将宽度和高度转换为适当的单位
        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)
        if (
            # 如果需要根据椭圆的形变进行角度拉伸
            width != height
            # 并且不是完整的圆形。
            #
            # 角度为 0 和 360 在角度拉伸时不会完全保持一致
            # （由于浮点精度限制以及 arctan2 的范围 [-pi, pi] 与此方法 [0, 360] 的差异），
            # 因此如果不必要，应避免这样做。
            and not (self.theta1 != self.theta2 and
                     self.theta1 % 360 == self.theta2 % 360)
        ):
            # 根据宽度和高度的比例进行角度拉伸
            theta1 = theta_stretch(self.theta1, width / height)
            theta2 = theta_stretch(self.theta2, width / height)
            return theta1, theta2, width, height
        # 如果不需要拉伸角度，则直接返回原始的角度和尺寸
        return self.theta1, self.theta2, width, height
# 绘制包围艺术家返回的边界框的矩形框以测试艺术家是否返回正确的边界框。
# 艺术家的边界框通过 `artist.get_window_extent(renderer)` 获取。
def bbox_artist(artist, renderer, props=None, fill=True):
    """
    A debug function to draw a rectangle around the bounding
    box returned by an artist's `.Artist.get_window_extent`
    to test whether the artist is returning the correct bbox.

    *props* is a dict of rectangle props with the additional property
    'pad' that sets the padding around the bbox in points.
    """
    if props is None:
        props = {}
    # 复制 props 字典，以避免在外部改变 pad 的值
    props = props.copy()
    # 获取 'pad' 属性的值，如果不存在则默认为 4
    pad = props.pop('pad', 4)
    # 将 pad 单位从点转换为像素
    pad = renderer.points_to_pixels(pad)
    # 获取艺术家的边界框
    bbox = artist.get_window_extent(renderer)
    # 创建一个 Rectangle 对象，围绕边界框绘制矩形
    r = Rectangle(
        xy=(bbox.x0 - pad / 2, bbox.y0 - pad / 2),
        width=bbox.width + pad, height=bbox.height + pad,
        fill=fill, transform=transforms.IdentityTransform(), clip_on=False)
    # 更新矩形框的属性
    r.update(props)
    # 在渲染器上绘制矩形框
    r.draw(renderer)


# 绘制一个围绕边界框的矩形框以测试艺术家是否返回正确的边界框。
def draw_bbox(bbox, renderer, color='k', trans=None):
    """
    A debug function to draw a rectangle around the bounding
    box returned by an artist's `.Artist.get_window_extent`
    to test whether the artist is returning the correct bbox.
    """
    # 创建一个 Rectangle 对象，围绕边界框绘制矩形
    r = Rectangle(xy=bbox.p0, width=bbox.width, height=bbox.height,
                  edgecolor=color, fill=False, clip_on=False)
    # 如果提供了 trans 参数，则设置矩形框的变换
    if trans is not None:
        r.set_transform(trans)
    # 在渲染器上绘制矩形框
    r.draw(renderer)


class _Style:
    """
    A base class for the Styles. It is meant to be a container class,
    where actual styles are declared as subclass of it, and it
    provides some helper functions.
    """

    # 在子类中自动执行文档字符串插值：
    # 这允许通过以下方式列出支持的样式：
    # - %(BoxStyle:table)s
    # - %(ConnectionStyle:table)s
    # - %(ArrowStyle:table)s
    # 并额外添加 .. ACCEPTS: 块，例如：
    # - %(BoxStyle:table_and_accepts)s
    # - %(ConnectionStyle:table_and_accepts)s
    # - %(ArrowStyle:table_and_accepts)s
    def __init_subclass__(cls):
        _docstring.interpd.update({
            f"{cls.__name__}:table": cls.pprint_styles(),
            f"{cls.__name__}:table_and_accepts": (
                cls.pprint_styles()
                + "\n\n    .. ACCEPTS: ["
                + "|".join(map(" '{}' ".format, cls._style_list))
                + "]")
        })
    def __new__(cls, stylename, **kwargs):
        """Return the instance of the subclass with the given style name."""
        # 获取子类实例，根据给定的样式名称
        # “class”应该具有_style_list属性，该属性是样式名称到样式类的映射
        _list = stylename.replace(" ", "").split(",")
        _name = _list[0].lower()
        
        # 尝试从_style_list中获取对应名称的样式类
        try:
            _cls = cls._style_list[_name]
        except KeyError as err:
            raise ValueError(f"Unknown style: {stylename!r}") from err
        
        # 尝试解析样式参数，将其转换为关键字参数字典
        try:
            _args_pair = [cs.split("=") for cs in _list[1:]]
            _args = {k: float(v) for k, v in _args_pair}
        except ValueError as err:
            raise ValueError(
                f"Incorrect style argument: {stylename!r}") from err
        
        # 使用解析后的参数和额外的kwargs创建样式类实例并返回
        return _cls(**{**_args, **kwargs})

    @classmethod
    def get_styles(cls):
        """Return a dictionary of available styles."""
        # 返回一个包含可用样式的字典
        return cls._style_list

    @classmethod
    def pprint_styles(cls):
        """Return the available styles as pretty-printed string."""
        # 创建一个包含样式信息的rst格式表格，并返回格式化后的字符串
        table = [('Class', 'Name', 'Attrs'),
                 *[(cls.__name__,
                    # 添加反引号，因为在reST中 - 和 | 有特殊含义
                    f'``{name}``',
                    # [1:-1]去掉周围的括号
                    str(inspect.signature(cls))[1:-1] or 'None')
                   for name, cls in cls._style_list.items()]]
        # 计算每列的最大长度，以便格式化rst表格
        col_len = [max(len(cell) for cell in column) for column in zip(*table)]
        table_formatstr = '  '.join('=' * cl for cl in col_len)
        rst_table = '\n'.join([
            '',
            table_formatstr,
            '  '.join(cell.ljust(cl) for cell, cl in zip(table[0], col_len)),
            table_formatstr,
            *['  '.join(cell.ljust(cl) for cell, cl in zip(row, col_len))
              for row in table[1:]],
            table_formatstr,
        ])
        # 使用textwrap.indent对rst_table进行缩进处理，然后返回
        return textwrap.indent(rst_table, prefix=' ' * 4)

    @classmethod
    def register(cls, name, style):
        """Register a new style."""
        # 注册一个新的样式，确保style是cls._Base的子类
        if not issubclass(style, cls._Base):
            raise ValueError(f"{style} must be a subclass of {cls._Base}")
        cls._style_list[name] = style
# 将类装饰为样式字典的条目的装饰器函数
def _register_style(style_list, cls=None, *, name=None):
    if cls is None:
        # 如果未提供类，则返回一个带有部分参数的偏函数
        return functools.partial(_register_style, style_list, name=name)
    # 将类添加到样式字典中，键为类名的小写形式或自定义名称
    style_list[name or cls.__name__.lower()] = cls
    return cls

# 继承自 _Style 类的 BoxStyle 类，使用 _docstring.dedent_interpd 进行字符串插值处理
@_docstring.dedent_interpd
class BoxStyle(_Style):
    """
    `BoxStyle` 是一个容器类，定义了多个 boxstyle 类，用于 `FancyBboxPatch`。

    可以通过以下方式创建一个样式对象：

       BoxStyle.Round(pad=0.2)

    或者：

       BoxStyle("Round", pad=0.2)

    或者：

       BoxStyle("Round, pad=0.2")

    下面定义了几个 boxstyle 类。

    %(BoxStyle:table)s

    boxstyle 类的实例是可调用对象，具有以下签名：

       __call__(self, x0, y0, width, height, mutation_size) -> Path

    参数 *x0*, *y0*, *width* 和 *height* 指定要绘制的框的位置和大小；
    *mutation_size* 缩放边框属性，如填充。
    """

    # 存储所有注册的样式的字典
    _style_list = {}

    # 将 Square 类注册到 _style_list 中
    @_register_style(_style_list)
    class Square:
        """一个方形框。"""

        def __init__(self, pad=0.3):
            """
            参数
            ----------
            pad : float, 默认: 0.3
                原始框周围的填充量。
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            # 加入填充后的宽度和高度
            width, height = width + 2 * pad, height + 2 * pad
            # 填充框的边界
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height
            return Path._create_closed(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # 将 Circle 类注册到 _style_list 中
    @_register_style(_style_list)
    class Circle:
        """一个圆形框。"""

        def __init__(self, pad=0.3):
            """
            参数
            ----------
            pad : float, 默认: 0.3
                原始框周围的填充量。
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            width, height = width + 2 * pad, height + 2 * pad
            # 填充框的边界
            x0, y0 = x0 - pad, y0 - pad
            return Path.circle((x0 + width / 2, y0 + height / 2),
                                max(width, height) / 2)

    # 继续注册更多的样式类...
    class Ellipse:
        """
        An elliptical box.

        .. versionadded:: 3.7
        """

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # Calculate padding based on mutation size and stored pad value
            pad = mutation_size * self.pad
            # Adjust width and height by adding padding
            width, height = width + 2 * pad, height + 2 * pad
            # Calculate new coordinates for the padded box
            x0, y0 = x0 - pad, y0 - pad
            # Calculate semi-major and semi-minor axes for the ellipse
            a = width / math.sqrt(2)
            b = height / math.sqrt(2)
            # Create an affine transformation to scale and translate
            trans = Affine2D().scale(a, b).translate(x0 + width / 2,
                                                     y0 + height / 2)
            # Transform a unit circle path using the affine transformation
            return trans.transform_path(Path.unit_circle())

    @_register_style(_style_list)
    class LArrow:
        """A box in the shape of a left-pointing arrow."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # Calculate padding based on mutation size and stored pad value
            pad = mutation_size * self.pad
            # Adjust width and height by adding padding
            width, height = width + 2 * pad, height + 2 * pad
            # Calculate new coordinates for the padded box
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height

            dx = (y1 - y0) / 2
            dxx = dx / 2
            # Adjust x0 position for the arrow shape
            x0 = x0 + pad / 1.4  # adjust by ~sqrt(2)

            # Create a closed path for the left-pointing arrow
            return Path._create_closed(
                [(x0 + dxx, y0), (x1, y0), (x1, y1), (x0 + dxx, y1),
                 (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx),
                 (x0 + dxx, y0 - dxx),  # arrow
                 (x0 + dxx, y0)])

    @_register_style(_style_list)
    class RArrow(LArrow):
        """A box in the shape of a right-pointing arrow."""

        def __call__(self, x0, y0, width, height, mutation_size):
            # Call the LArrow class method to get the left-pointing arrow path
            p = BoxStyle.LArrow.__call__(
                self, x0, y0, width, height, mutation_size)
            # Reflect the vertices horizontally to get a right-pointing arrow
            p.vertices[:, 0] = 2 * x0 + width - p.vertices[:, 0]
            return p

    @_register_style(_style_list)
    class DArrow:
        """A box in the shape of a two-way arrow."""
        # Modified from LArrow to add a right arrow to the bbox.

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # 计算填充量
            pad = mutation_size * self.pad
            # 加上填充后的高度
            height = height + 2 * pad
            # 计算填充后的边界框
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height

            dx = (y1 - y0) / 2
            dxx = dx / 2
            x0 = x0 + pad / 1.4  # 根据近似公式调整

            # 创建封闭路径
            return Path._create_closed([
                (x0 + dxx, y0), (x1, y0),  # 底部线段
                (x1, y0 - dxx), (x1 + dx + dxx, y0 + dx),
                (x1, y1 + dxx),  # 右箭头
                (x1, y1), (x0 + dxx, y1),  # 顶部线段
                (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx),
                (x0 + dxx, y0 - dxx),  # 左箭头
                (x0 + dxx, y0)])

    @_register_style(_style_list)
    class Round:
        """定义一个带有圆角的框。"""

        def __init__(self, pad=0.3, rounding_size=None):
            """
            初始化方法，设置圆角框的参数。

            Parameters
            ----------
            pad : float, default: 0.3
                原始框周围的填充量。
            rounding_size : float, default: *pad*
                圆角的半径大小。
            """
            self.pad = pad  # 设置填充量
            self.rounding_size = rounding_size  # 设置圆角大小

        def __call__(self, x0, y0, width, height, mutation_size):
            """
            根据给定的参数生成圆角框的路径。

            Parameters
            ----------
            x0, y0 : float
                框的左下角的坐标。
            width : float
                框的宽度。
            height : float
                框的高度。
            mutation_size : float
                变异大小，用于计算实际的填充和圆角尺寸。

            Returns
            -------
            Path
                表示圆角框路径的对象。
            """

            # 计算填充量
            pad = mutation_size * self.pad

            # 计算圆角的大小
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad

            # 调整框的尺寸
            width, height = width + 2 * pad, height + 2 * pad

            # 计算框的四个角的坐标
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height

            # 创建圆角框的路径，使用二次贝塞尔曲线表示圆角
            cp = [(x0 + dr, y0),
                  (x1 - dr, y0),
                  (x1, y0), (x1, y0 + dr),
                  (x1, y1 - dr),
                  (x1, y1), (x1 - dr, y1),
                  (x0 + dr, y1),
                  (x0, y1), (x0, y1 - dr),
                  (x0, y0 + dr),
                  (x0, y0), (x0 + dr, y0),
                  (x0 + dr, y0)]

            # 定义路径的绘制命令
            com = [Path.MOVETO,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.CLOSEPOLY]

            # 返回圆角框的路径对象
            return Path(cp, com)

    @_register_style(_style_list)
    class Round4:
        """A box with rounded edges."""
    
        def __init__(self, pad=0.3, rounding_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            rounding_size : float, default: *pad*/2
                Rounding of edges.
            """
            # 初始化函数，设置圆角框的参数
            self.pad = pad
            self.rounding_size = rounding_size
    
        def __call__(self, x0, y0, width, height, mutation_size):
    
            # 计算填充大小
            pad = mutation_size * self.pad
    
            # 计算圆角大小，默认为填充大小的一半
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad / 2.
    
            # 计算调整后的宽度和高度
            width = width + 2 * pad - 2 * dr
            height = height + 2 * pad - 2 * dr
    
            # 计算调整后的左上角和右下角坐标
            x0, y0 = x0 - pad + dr, y0 - pad + dr,
            x1, y1 = x0 + width, y0 + height
    
            # 计算圆角矩形的控制点
            cp = [(x0, y0),
                  (x0 + dr, y0 - dr), (x1 - dr, y0 - dr), (x1, y0),
                  (x1 + dr, y0 + dr), (x1 + dr, y1 - dr), (x1, y1),
                  (x1 - dr, y1 + dr), (x0 + dr, y1 + dr), (x0, y1),
                  (x0 - dr, y1 - dr), (x0 - dr, y0 + dr), (x0, y0),
                  (x0, y0)]
    
            # 定义控制点的连接方式
            com = [Path.MOVETO,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CLOSEPOLY]
    
            # 返回一个表示圆角矩形的路径对象
            return Path(cp, com)
    
    @_register_style(_style_list)
    class Sawtooth:
        """A box with a sawtooth outline."""
    
        def __init__(self, pad=0.3, tooth_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            tooth_size : float, default: *pad*/2
                Size of the sawtooth.
            """
            self.pad = pad  # 设置对象的属性，表示原始框周围的填充量
            self.tooth_size = tooth_size  # 设置对象的属性，表示锯齿的大小
    
        def _get_sawtooth_vertices(self, x0, y0, width, height, mutation_size):
    
            # padding
            pad = mutation_size * self.pad  # 计算实际填充量，乘以变异大小
    
            # size of sawtooth
            if self.tooth_size is None:
                tooth_size = self.pad * .5 * mutation_size  # 如果未指定锯齿大小，则计算默认值
            else:
                tooth_size = self.tooth_size * mutation_size  # 使用指定的锯齿大小
    
            hsz = tooth_size / 2  # 计算锯齿的一半大小
            width = width + 2 * pad - tooth_size  # 调整宽度以适应锯齿
            height = height + 2 * pad - tooth_size  # 调整高度以适应锯齿
    
            # the sizes of the vertical and horizontal sawtooth are
            # separately adjusted to fit the given box size.
            dsx_n = round((width - tooth_size) / (tooth_size * 2)) * 2  # 计算水平方向锯齿数量
            dsy_n = round((height - tooth_size) / (tooth_size * 2)) * 2  # 计算垂直方向锯齿数量
    
            x0, y0 = x0 - pad + hsz, y0 - pad + hsz  # 调整起始点以适应填充
            x1, y1 = x0 + width, y0 + height  # 计算框的右下角坐标
    
            xs = [
                x0, *np.linspace(x0 + hsz, x1 - hsz, 2 * dsx_n + 1),  # 底部
                *([x1, x1 + hsz, x1, x1 - hsz] * dsy_n)[:2*dsy_n+2],  # 右侧
                x1, *np.linspace(x1 - hsz, x0 + hsz, 2 * dsx_n + 1),  # 顶部
                *([x0, x0 - hsz, x0, x0 + hsz] * dsy_n)[:2*dsy_n+2],  # 左侧
            ]
            ys = [
                *([y0, y0 - hsz, y0, y0 + hsz] * dsx_n)[:2*dsx_n+2],  # 底部
                y0, *np.linspace(y0 + hsz, y1 - hsz, 2 * dsy_n + 1),  # 右侧
                *([y1, y1 + hsz, y1, y1 - hsz] * dsx_n)[:2*dsx_n+2],  # 顶部
                y1, *np.linspace(y1 - hsz, y0 + hsz, 2 * dsy_n + 1),  # 左侧
            ]
    
            return [*zip(xs, ys), (xs[0], ys[0])]  # 返回构成锯齿形状的顶点坐标列表
    
        def __call__(self, x0, y0, width, height, mutation_size):
            saw_vertices = self._get_sawtooth_vertices(x0, y0, width,
                                                       height, mutation_size)
            return Path(saw_vertices, closed=True)
    class Roundtooth(Sawtooth):
        """A box with a rounded sawtooth outline."""

        def __call__(self, x0, y0, width, height, mutation_size):
            # 调用父类方法获取锯齿形状的顶点坐标
            saw_vertices = self._get_sawtooth_vertices(x0, y0,
                                                       width, height,
                                                       mutation_size)
            # 添加一个额外的顶点，以正确闭合多边形
            saw_vertices = np.concatenate([saw_vertices, [saw_vertices[0]]])
            # 定义绘制路径时的操作代码，包括移动到起始点、使用三次贝塞尔曲线、最后闭合多边形
            codes = ([Path.MOVETO] +
                     [Path.CURVE3, Path.CURVE3] * ((len(saw_vertices)-1)//2) +
                     [Path.CLOSEPOLY])
            # 返回描述锯齿形状的Path对象
            return Path(saw_vertices, codes)
# 定义一个名为 ConnectionStyle 的类，它是 _Style 类的子类
@_docstring.dedent_interpd
class ConnectionStyle(_Style):
    """
    `ConnectionStyle` is a container class which defines
    several connectionstyle classes, which is used to create a path
    between two points.  These are mainly used with `FancyArrowPatch`.

    A connectionstyle object can be either created as::

           ConnectionStyle.Arc3(rad=0.2)

    or::

           ConnectionStyle("Arc3", rad=0.2)

    or::

           ConnectionStyle("Arc3, rad=0.2")

    The following classes are defined

    %(ConnectionStyle:table)s

    An instance of any connection style class is a callable object,
    whose call signature is::

        __call__(self, posA, posB,
                 patchA=None, patchB=None,
                 shrinkA=2., shrinkB=2.)

    and it returns a `.Path` instance. *posA* and *posB* are
    tuples of (x, y) coordinates of the two points to be
    connected. *patchA* (or *patchB*) is given, the returned path is
    clipped so that it start (or end) from the boundary of the
    patch. The path is further shrunk by *shrinkA* (or *shrinkB*)
    which is given in points.
    """

    # 一个静态变量，用于存储不同连接样式的字典，初始化为空字典
    _style_list = {}
    # 定义一个基类 _Base，用于连接样式类的基础功能。子类需要实现 connect 方法，其调用签名为：
    #
    # connect(posA, posB)
    #
    # 其中 posA 和 posB 是要连接的 x、y 坐标的元组。该方法需要返回连接两点的路径。这个基类定义了 __call__ 方法和一些辅助方法。
    class _Base:
        """
        A base class for connectionstyle classes. The subclass needs
        to implement a *connect* method whose call signature is::

          connect(posA, posB)

        where posA and posB are tuples of x, y coordinates to be
        connected.  The method needs to return a path connecting two
        points. This base class defines a __call__ method, and a few
        helper methods.
        """
        
        # 返回一个谓词函数，测试点 *xy* 是否包含在 *patch* 中。
        def _in_patch(self, patch):
            """
            Return a predicate function testing whether a point *xy* is
            contained in *patch*.
            """
            return lambda xy: patch.contains(
                SimpleNamespace(x=xy[0], y=xy[1]))[0]

        # 在 *in_start* 和 *in_stop* 区域处剪切 *path*。
        # 假定原始路径从 *in_start* 区域开始，并在 *in_stop* 区域结束。
        def _clip(self, path, in_start, in_stop):
            """
            Clip *path* at its start by the region where *in_start* returns
            True, and at its stop by the region where *in_stop* returns True.

            The original path is assumed to start in the *in_start* region and
            to stop in the *in_stop* region.
            """
            if in_start:
                try:
                    _, path = split_path_inout(path, in_start)
                except ValueError:
                    pass
            if in_stop:
                try:
                    path, _ = split_path_inout(path, in_stop)
                except ValueError:
                    pass
            return path

        # 调用 *connect* 方法创建 *posA* 和 *posB* 之间的路径，然后剪切和收缩路径。
        def __call__(self, posA, posB,
                     shrinkA=2., shrinkB=2., patchA=None, patchB=None):
            """
            Call the *connect* method to create a path between *posA* and
            *posB*; then clip and shrink the path.
            """
            path = self.connect(posA, posB)
            # 根据 patchA 和 patchB 剪切路径
            path = self._clip(
                path,
                self._in_patch(patchA) if patchA else None,
                self._in_patch(patchB) if patchB else None,
            )
            # 根据 shrinkA 和 shrinkB 收缩路径
            path = self._clip(
                path,
                inside_circle(*path.vertices[0], shrinkA) if shrinkA else None,
                inside_circle(*path.vertices[-1], shrinkB) if shrinkB else None
            )
            return path

    @_register_style(_style_list)
    class Arc3(_Base):
        """
        创建一个简单的二次贝塞尔曲线，连接两个点。
        曲线被创建为中间控制点(C1)位于起点(C0)和终点(C2)的中间，
        并且控制点C1到连接C0-C2线段的距离是*rad*乘以C0-C2的距离。
        """

        def __init__(self, rad=0.):
            """
            参数
            ----------
            rad : float
              曲线的曲率。
            """
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
            dx, dy = x2 - x1, y2 - y1

            f = self.rad

            cx, cy = x12 + f * dy, y12 - f * dx

            vertices = [(x1, y1),
                        (cx, cy),
                        (x2, y2)]
            codes = [Path.MOVETO,
                     Path.CURVE3,
                     Path.CURVE3]

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle3(_Base):
        """
        创建一个简单的二次贝塞尔曲线，连接两点。
        中间控制点位于从起点和终点出发，斜率分别为*angleA*和*angleB*的两条直线的交点。
        """

        def __init__(self, angleA=90, angleB=0):
            """
            参数
            ----------
            angleA : float
              路径的起始角度。

            angleB : float
              路径的结束角度。
            """

            self.angleA = angleA
            self.angleB = angleB

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            return Path(vertices, codes)
    class Angle(_Base):
        """
        Creates a piecewise continuous quadratic Bézier path between two
        points. The path has a one passing-through point placed at the
        intersecting point of two lines which cross the start and end point,
        and have a slope of *angleA* and *angleB*, respectively.
        The connecting edges are rounded with *rad*.
        """

        def __init__(self, angleA=90, angleB=0, rad=0.):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.

            rad : float
              Rounding radius of the edge.
            """

            self.angleA = angleA  # 初始化起始角度
            self.angleB = angleB  # 初始化结束角度

            self.rad = rad  # 初始化圆角半径

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA = math.cos(math.radians(self.angleA))  # 计算起始角度的余弦值
            sinA = math.sin(math.radians(self.angleA))  # 计算起始角度的正弦值
            cosB = math.cos(math.radians(self.angleB))  # 计算结束角度的余弦值
            sinB = math.sin(math.radians(self.angleB))  # 计算结束角度的正弦值

            # 计算两条线的交点坐标
            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1)]  # 起始点坐标
            codes = [Path.MOVETO]  # 路径起始码

            if self.rad == 0.:
                vertices.append((cx, cy))  # 如果圆角半径为0，则直接连接到交点
                codes.append(Path.LINETO)  # 添加直线连接码
            else:
                dx1, dy1 = x1 - cx, y1 - cy  # 计算起始点到交点的向量
                d1 = np.hypot(dx1, dy1)  # 计算起始点到交点的距离
                f1 = self.rad / d1  # 计算起始端圆弧的比例因子
                dx2, dy2 = x2 - cx, y2 - cy  # 计算结束点到交点的向量
                d2 = np.hypot(dx2, dy2)  # 计算结束点到交点的距离
                f2 = self.rad / d2  # 计算结束端圆弧的比例因子
                vertices.extend([(cx + dx1 * f1, cy + dy1 * f1),
                                 (cx, cy),
                                 (cx + dx2 * f2, cy + dy2 * f2)])  # 添加圆弧控制点和结束点
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])  # 添加路径码

            vertices.append((x2, y2))  # 结束点坐标
            codes.append(Path.LINETO)  # 添加直线连接码

            return Path(vertices, codes)  # 返回路径对象

    @_register_style(_style_list)
    @_register_style(_style_list)
    class Bar(_Base):
        """
        A line with *angle* between A and B with *armA* and *armB*. One of the
        arms is extended so that they are connected in a right angle. The
        length of *armA* is determined by (*armA* + *fraction* x AB distance).
        Same for *armB*.
        """

        def __init__(self, armA=0., armB=0., fraction=0.3, angle=None):
            """
            Parameters
            ----------
            armA : float
                Minimum length of armA.

            armB : float
                Minimum length of armB.

            fraction : float
                A fraction of the distance between two points that will be
                added to armA and armB.

            angle : float or None
                Angle of the connecting line (if None, parallel to A and B).
            """
            # 初始化函数，设定实例的初始属性值
            self.armA = armA  # 设定实例属性 armA，代表 A 端臂长
            self.armB = armB  # 设定实例属性 armB，代表 B 端臂长
            self.fraction = fraction  # 设定实例属性 fraction，代表距离比例系数
            self.angle = angle  # 设定实例属性 angle，代表连接线角度

        def connect(self, posA, posB):
            # 解构 posA 和 posB 的坐标
            x1, y1 = posA
            x20, y20 = x2, y2 = posB

            # 计算两点间直线的角度 theta1
            theta1 = math.atan2(y2 - y1, x2 - x1)
            dx, dy = x2 - x1, y2 - y1
            dd = (dx * dx + dy * dy) ** .5
            ddx, ddy = dx / dd, dy / dd

            # 获取实例属性 armA 和 armB 的值
            armA, armB = self.armA, self.armB

            # 如果实例属性 angle 不为 None，则调整连接线角度
            if self.angle is not None:
                theta0 = np.deg2rad(self.angle)
                dtheta = theta1 - theta0
                dl = dd * math.sin(dtheta)
                dL = dd * math.cos(dtheta)
                x2, y2 = x1 + dL * math.cos(theta0), y1 + dL * math.sin(theta0)
                armB = armB - dl

                # 更新连接线参数
                dx, dy = x2 - x1, y2 - y1
                dd2 = (dx * dx + dy * dy) ** .5
                ddx, ddy = dx / dd2, dy / dd2

            # 计算连接点的坐标
            arm = max(armA, armB)
            f = self.fraction * dd + arm
            cx1, cy1 = x1 + f * ddy, y1 - f * ddx
            cx2, cy2 = x2 + f * ddy, y2 - f * ddx

            # 定义路径的顶点和代码
            vertices = [(x1, y1),
                        (cx1, cy1),
                        (cx2, cy2),
                        (x20, y20)]
            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO]

            # 返回路径对象
            return Path(vertices, codes)
def _point_along_a_line(x0, y0, x1, y1, d):
    """
    Return the point on the line connecting (*x0*, *y0*) -- (*x1*, *y1*) whose
    distance from (*x0*, *y0*) is *d*.
    """
    # 计算线段的方向向量
    dx, dy = x0 - x1, y0 - y1
    # 计算距离比例因子
    ff = d / (dx * dx + dy * dy) ** .5
    # 计算目标点的坐标
    x2, y2 = x0 - ff * dx, y0 - ff * dy

    return x2, y2


@_docstring.dedent_interpd
class ArrowStyle(_Style):
    """
    `ArrowStyle` is a container class which defines several
    arrowstyle classes, which is used to create an arrow path along a
    given path.  These are mainly used with `FancyArrowPatch`.

    An arrowstyle object can be either created as::

           ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy", head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy, head_length=.4, head_width=.4, tail_width=.4")

    The following classes are defined

    %(ArrowStyle:table)s

    For an overview of the visual appearance, see
    :doc:`/gallery/text_labels_and_annotations/fancyarrow_demo`.

    An instance of any arrow style class is a callable object,
    whose call signature is::

        __call__(self, path, mutation_size, linewidth, aspect_ratio=1.)

    and it returns a tuple of a `.Path` instance and a boolean
    value. *path* is a `.Path` instance along which the arrow
    will be drawn. *mutation_size* and *aspect_ratio* have the same
    meaning as in `BoxStyle`. *linewidth* is a line width to be
    stroked. This is meant to be used to correct the location of the
    head so that it does not overshoot the destination point, but not all
    classes support it.

    Notes
    -----
    *angleA* and *angleB* specify the orientation of the bracket, as either a
    clockwise or counterclockwise angle depending on the arrow type. 0 degrees
    means perpendicular to the line connecting the arrow's head and tail.

    .. plot:: gallery/text_labels_and_annotations/angles_on_bracket_arrows.py
    """

    _style_list = {}

    @_register_style(_style_list, name="-")
    class Curve(_Curve):
        """A simple curve without any arrow head."""

        def __init__(self):  # hide head_length, head_width
            # These attributes (whose values come from backcompat) only matter
            # if someone modifies beginarrow/etc. on an ArrowStyle instance.
            super().__init__(head_length=.2, head_width=.1)

    @_register_style(_style_list, name="<-")
    class CurveA(_Curve):
        """An arrow with a head at its start point."""
        arrow = "<-"

    @_register_style(_style_list, name="->")
    class CurveB(_Curve):
        """An arrow with a head at its end point."""
        arrow = "->"

    @_register_style(_style_list, name="<->")
    class CurveAB(_Curve):
        """An arrow with heads both at the start and the end point."""
        arrow = "<->"

    @_register_style(_style_list, name="<|-")
    class BarAB(_Bar):
        """A bar with heads at both ends."""
        pass
    class CurveFilledA(_Curve):
        """An arrow with filled triangle head at the start."""
        arrow = "<|-"  # 定义一个箭头，箭头头部是填充的三角形

    @_register_style(_style_list, name="-|>")
    class CurveFilledB(_Curve):
        """An arrow with filled triangle head at the end."""
        arrow = "-|>"  # 定义一个箭头，箭头尾部是填充的三角形

    @_register_style(_style_list, name="<|-|>")
    class CurveFilledAB(_Curve):
        """An arrow with filled triangle heads at both ends."""
        arrow = "<|-|>"  # 定义一个箭头，箭头两端都是填充的三角形

    @_register_style(_style_list, name="]-")
    class BracketA(_Curve):
        """An arrow with an outward square bracket at its start."""
        arrow = "]-"  # 定义一个箭头，箭头起始端是外向的方括号

        def __init__(self, widthA=1., lengthA=0.2, angleA=0):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.
            lengthA : float, default: 0.2
                Length of the bracket.
            angleA : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)
            # 初始化函数，设置箭头起始端的外向方括号的宽度、长度和角度

    @_register_style(_style_list, name="-[")
    class BracketB(_Curve):
        """An arrow with an outward square bracket at its end."""
        arrow = "-["  # 定义一个箭头，箭头末端是外向的方括号

        def __init__(self, widthB=1., lengthB=0.2, angleB=0):
            """
            Parameters
            ----------
            widthB : float, default: 1.0
                Width of the bracket.
            lengthB : float, default: 0.2
                Length of the bracket.
            angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthB=widthB, lengthB=lengthB, angleB=angleB)
            # 初始化函数，设置箭头末端的外向方括号的宽度、长度和角度

    @_register_style(_style_list, name="]-[")
    class BracketAB(_Curve):
        """An arrow with outward square brackets at both ends."""
        arrow = "]-["  # 定义一个箭头，箭头两端都是外向的方括号

        def __init__(self,
                     widthA=1., lengthA=0.2, angleA=0,
                     widthB=1., lengthB=0.2, angleB=0):
            """
            Parameters
            ----------
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            lengthA, lengthB : float, default: 0.2
                Length of the bracket.
            angleA, angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA,
                             widthB=widthB, lengthB=lengthB, angleB=angleB)
            # 初始化函数，设置箭头两端的外向方括号的宽度、长度和角度
    @_register_style(_style_list, name=']->')
    class BracketCurve(_Curve):
        """
        注册箭头样式到样式列表，并定义箭头为一个朝外的方括号和一个箭头头部。
        """
        arrow = "]->"

        def __init__(self, widthA=1., lengthA=0.2, angleA=None):
            """
            构造函数，初始化箭头对象。

            Parameters
            ----------
            widthA : float, 默认值: 1.0
                方括号的宽度。
            lengthA : float, 默认值: 0.2
                方括号的长度。
            angleA : float, 默认值: 0 度
                方向角度，逆时针方向。0 度表示垂直于线。

            调用父类构造函数进行初始化。
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)

    @_register_style(_style_list, name='<-[')
    class CurveBracket(_Curve):
        """
        注册箭头样式到样式列表，并定义箭头为一个朝外的箭头头部和一个方括号。
        """
        arrow = "<-["

        def __init__(self, widthB=1., lengthB=0.2, angleB=None):
            """
            构造函数，初始化箭头对象。

            Parameters
            ----------
            widthB : float, 默认值: 1.0
                方括号的宽度。
            lengthB : float, 默认值: 0.2
                方括号的长度。
            angleB : float, 默认值: 0 度
                方向角度，逆时针方向。0 度表示垂直于线。

            调用父类构造函数进行初始化。
            """
            super().__init__(widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list)
    @_register_style(_style_list)
    @_register_style(_style_list)
    class Wedge(_Base):
        """
        Wedge(?) shape. Only works with a quadratic Bézier curve.  The
        start point has a width of the *tail_width* and the end point has a
        width of 0. At the middle, the width is *shrink_factor*x*tail_width*.
        """

        def __init__(self, tail_width=.3, shrink_factor=0.5):
            """
            Parameters
            ----------
            tail_width : float, default: 0.3
                Width of the tail.

            shrink_factor : float, default: 0.5
                Fraction of the arrow width at the middle point.
            """
            # 初始化 Wedge 类的实例，设置尾部宽度和收缩因子
            self.tail_width = tail_width
            self.shrink_factor = shrink_factor
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            # docstring inherited
            # 确保路径是二次贝塞尔曲线，获取曲线的控制点坐标
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            # 创建具有楔形末端的贝塞尔曲线
            b_plus, b_minus = make_wedged_bezier2(
                                    arrow_path,
                                    self.tail_width * mutation_size / 2.,
                                    wm=self.shrink_factor)

            # 构建路径描述符列表，定义箭头的形状
            patch_path = [(Path.MOVETO, b_plus[0]),
                          (Path.CURVE3, b_plus[1]),
                          (Path.CURVE3, b_plus[2]),
                          (Path.LINETO, b_minus[2]),
                          (Path.CURVE3, b_minus[1]),
                          (Path.CURVE3, b_minus[0]),
                          (Path.CLOSEPOLY, b_minus[0]),
                          ]
            # 根据路径描述符创建路径对象
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True
class FancyBboxPatch(Patch):
    """
    A fancy box around a rectangle with lower left at *xy* = (*x*, *y*)
    with specified width and height.

    `.FancyBboxPatch` is similar to `.Rectangle`, but it draws a fancy box
    around the rectangle. The transformation of the rectangle box to the
    fancy box is delegated to the style classes defined in `.BoxStyle`.
    """

    # 默认使用边缘特性
    _edge_default = True

    def __str__(self):
        # 返回描述对象的字符串表示，包括位置和尺寸信息
        s = self.__class__.__name__ + "((%g, %g), width=%g, height=%g)"
        return s % (self._x, self._y, self._width, self._height)

    @_docstring.dedent_interpd
    def __init__(self, xy, width, height, boxstyle="round", *,
                 mutation_scale=1, mutation_aspect=1, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
          The lower left corner of the box.

        width : float
            The width of the box.

        height : float
            The height of the box.

        boxstyle : str or `~matplotlib.patches.BoxStyle`
            The style of the fancy box. This can either be a `.BoxStyle`
            instance or a string of the style name and optionally comma
            separated attributes (e.g. "Round, pad=0.2"). This string is
            passed to `.BoxStyle` to construct a `.BoxStyle` object. See
            there for a full documentation.

            The following box styles are available:

            %(BoxStyle:table)s

        mutation_scale : float, default: 1
            Scaling factor applied to the attributes of the box style
            (e.g. pad or rounding_size).

        mutation_aspect : float, default: 1
            The height of the rectangle will be squeezed by this value before
            the mutation and the mutated box will be stretched by the inverse
            of it. For example, this allows different horizontal and vertical
            padding.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties

        %(Patch:kwdoc)s
        """

        # 调用父类构造函数初始化
        super().__init__(**kwargs)
        # 设置矩形框左下角坐标
        self._x, self._y = xy
        # 设置矩形框宽度和高度
        self._width = width
        self._height = height
        # 设置矩形框的样式
        self.set_boxstyle(boxstyle)
        # 设置变换规模和纵横比
        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect
        # 设为过时状态，需要更新
        self.stale = True

    @_docstring.dedent_interpd
    # 设置文本框样式，可选地带有额外属性。

    # 如果没有参数（或者 boxstyle=None），返回可用的文本框样式作为人类可读的字符串。

    # Parameters 参数：
    # ----------
    # boxstyle : str or `~matplotlib.patches.BoxStyle`
    #     文本框的样式：可以是 `.BoxStyle` 实例，也可以是一个字符串，包括样式名称和可选的逗号分隔的属性
    #     （例如 "Round,pad=0.2"）。这样的字符串用于构建一个 `.BoxStyle` 对象，详见该类的文档。

    #     可用的文本框样式列于下面的表中：
    #     %(BoxStyle:table_and_accepts)s

    # **kwargs
    #     文本框样式的额外属性。支持的参数请参考上述表格。

    # Examples 示例：
    # --------
    # ::
    # 
    #     set_boxstyle("Round,pad=0.2")
    #     set_boxstyle("round", pad=0.2)
    def set_boxstyle(self, boxstyle=None, **kwargs):
        if boxstyle is None:
            return BoxStyle.pprint_styles()  # 如果未指定 boxstyle，则打印可用的样式
        self._bbox_transmuter = (
            BoxStyle(boxstyle, **kwargs)  # 创建一个新的 BoxStyle 对象
            if isinstance(boxstyle, str) else boxstyle)  # 如果 boxstyle 是字符串，则解析为 BoxStyle 对象
        self.stale = True  # 标记为需要更新

    # 返回当前的文本框样式对象。
    def get_boxstyle(self):
        return self._bbox_transmuter

    # 设置变换比例尺。
    def set_mutation_scale(self, scale):
        self._mutation_scale = scale  # 设置变换比例尺
        self.stale = True  # 标记为需要更新

    # 返回当前的变换比例尺。
    def get_mutation_scale(self):
        return self._mutation_scale

    # 设置文本框变换的纵横比。
    def set_mutation_aspect(self, aspect):
        self._mutation_aspect = aspect  # 设置文本框变换的纵横比
        self.stale = True  # 标记为需要更新

    # 返回文本框变换的纵横比。
    def get_mutation_aspect(self):
        return (self._mutation_aspect if self._mutation_aspect is not None
                else 1)  # 返回文本框变换的纵横比，如果未设置则默认为 1（向后兼容）

    # 返回矩形变换后的路径。
    def get_path(self):
        boxstyle = self.get_boxstyle()
        m_aspect = self.get_mutation_aspect()
        # 使用文本框样式创建路径，y 和 height 按照纵横比进行压缩。
        path = boxstyle(self._x, self._y / m_aspect,
                        self._width, self._height / m_aspect,
                        self.get_mutation_scale())
        return Path(path.vertices * [1, m_aspect], path.codes)  # 将 y 解压缩。

    # 返回矩形的左边坐标。
    def get_x(self):
        return self._x  # 返回矩形的左边坐标

    # 返回矩形的底部坐标。
    def get_y(self):
        return self._y  # 返回矩形的底部坐标
    # 返回矩形的宽度
    def get_width(self):
        """Return the width of the rectangle."""
        return self._width

    # 返回矩形的高度
    def get_height(self):
        """Return the height of the rectangle."""
        return self._height

    # 设置矩形左边界的坐标
    def set_x(self, x):
        """
        Set the left coord of the rectangle.

        Parameters
        ----------
        x : float
            新的左边界坐标值
        """
        self._x = x
        self.stale = True

    # 设置矩形底边界的坐标
    def set_y(self, y):
        """
        Set the bottom coord of the rectangle.

        Parameters
        ----------
        y : float
            新的底边界坐标值
        """
        self._y = y
        self.stale = True

    # 设置矩形的宽度
    def set_width(self, w):
        """
        Set the rectangle width.

        Parameters
        ----------
        w : float
            新的矩形宽度值
        """
        self._width = w
        self.stale = True

    # 设置矩形的高度
    def set_height(self, h):
        """
        Set the rectangle height.

        Parameters
        ----------
        h : float
            新的矩形高度值
        """
        self._height = h
        self.stale = True

    # 设置矩形的边界
    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle.

        Call signatures::

            set_bounds(left, bottom, width, height)
            set_bounds((left, bottom, width, height))

        Parameters
        ----------
        left, bottom : float
            矩形左下角的坐标
        width, height : float
            矩形的宽度和高度
        """
        if len(args) == 1:
            l, b, w, h = args[0]
        else:
            l, b, w, h = args
        self._x = l
        self._y = b
        self._width = w
        self._height = h
        self.stale = True

    # 返回矩形的边界框对象 `.Bbox`
    def get_bbox(self):
        """Return the `.Bbox`."""
        return transforms.Bbox.from_bounds(self._x, self._y,
                                           self._width, self._height)
class FancyArrowPatch(Patch):
    """
    A fancy arrow patch.

    It draws an arrow using the `ArrowStyle`. It is primarily used by the
    `~.axes.Axes.annotate` method.  For most purposes, use the annotate method for
    drawing arrows.

    The head and tail positions are fixed at the specified start and end points
    of the arrow, but the size and shape (in display coordinates) of the arrow
    does not change when the axis is moved or zoomed.
    """
    _edge_default = True  # 默认边缘为真

    def __str__(self):
        if self._posA_posB is not None:
            (x1, y1), (x2, y2) = self._posA_posB
            return f"{type(self).__name__}(({x1:g}, {y1:g})->({x2:g}, {y2:g}))"
        else:
            return f"{type(self).__name__}({self._path_original})"

    @_docstring.dedent_interpd
    def __init__(self, posA=None, posB=None, *,
                 path=None, arrowstyle="simple", connectionstyle="arc3",
                 patchA=None, patchB=None, shrinkA=2, shrinkB=2,
                 mutation_scale=1, mutation_aspect=1, **kwargs):
        """
        There are two ways for defining an arrow:

        - If *posA* and *posB* are given, a path connecting two points is
          created according to *connectionstyle*. The path will be
          clipped with *patchA* and *patchB* and further shrunken by
          *shrinkA* and *shrinkB*. An arrow is drawn along this
          resulting path using the *arrowstyle* parameter.

        - Alternatively if *path* is provided, an arrow is drawn along this
          path and *patchA*, *patchB*, *shrinkA*, and *shrinkB* are ignored.

        Parameters
        ----------
        posA, posB : (float, float), default: None
            (x, y) coordinates of arrow tail and arrow head respectively.

        path : `~matplotlib.path.Path`, default: None
            If provided, an arrow is drawn along this path and *patchA*,
            *patchB*, *shrinkA*, and *shrinkB* are ignored.

        arrowstyle : str or `.ArrowStyle`, default: 'simple'
            The `.ArrowStyle` with which the fancy arrow is drawn.  If a
            string, it should be one of the available arrowstyle names, with
            optional comma-separated attributes.  The optional attributes are
            meant to be scaled with the *mutation_scale*.  The following arrow
            styles are available:

            %(ArrowStyle:table)s

        connectionstyle : str or `.ConnectionStyle` or None, optional, \
            # 箭头连接样式，可以是字符串或 `.ConnectionStyle` 对象，可选，默认为 'arc3'
        default: 'arc3'
            # 连接样式，用于连接 posA 和 posB 的 `.ConnectionStyle`。
            # 如果是字符串，应为可用的连接样式名称之一，可以有逗号分隔的可选属性。
            # 可用的连接样式如下：
            #
            # %(ConnectionStyle:table)s

        patchA, patchB : `~matplotlib.patches.Patch`, default: None
            # 箭头的头部和尾部补丁，分别对应 patchA 和 patchB。

        shrinkA, shrinkB : float, default: 2
            # 箭头的尾部和头部的缩小量，以点为单位。

        mutation_scale : float, default: 1
            # 用于缩放箭头样式属性（例如 *head_length*）的值。

        mutation_aspect : None or float, default: None
            # 在变异之前，矩形的高度将被该值压缩，变异后，变异的框将被其倒数拉伸。

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties, optional
            # 这里是可用的 `.Patch` 属性列表：
            #
            # %(Patch:kwdoc)s
            #
            # 与其他补丁不同，`FancyArrowPatch` 的默认 ``capstyle`` 和 ``joinstyle`` 设置为 ``"round"``。
        """
        # 传统上，`FancyArrowPatch` 的 capstyle 和 joinstyle 是圆形的
        kwargs.setdefault("joinstyle", JoinStyle.round)
        kwargs.setdefault("capstyle", CapStyle.round)

        super().__init__(**kwargs)

        if posA is not None and posB is not None and path is None:
            self._posA_posB = [posA, posB]

            if connectionstyle is None:
                connectionstyle = "arc3"
            self.set_connectionstyle(connectionstyle)

        elif posA is None and posB is None and path is not None:
            self._posA_posB = None
        else:
            raise ValueError("Either posA and posB, or path need to provided")

        self.patchA = patchA
        self.patchB = patchB
        self.shrinkA = shrinkA
        self.shrinkB = shrinkB

        self._path_original = path

        self.set_arrowstyle(arrowstyle)

        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect

        self._dpi_cor = 1.0

    def set_positions(self, posA, posB):
        """
        Set the start and end positions of the connecting path.

        Parameters
        ----------
        posA, posB : None, tuple
            # 连接路径的起始和结束位置的坐标 (x, y)。如果为 `None`，则使用当前值。
        """
        if posA is not None:
            self._posA_posB[0] = posA
        if posB is not None:
            self._posA_posB[1] = posB
        self.stale = True
    # 设置尾部补丁。
    def set_patchA(self, patchA):
        """
        Set the tail patch.

        Parameters
        ----------
        patchA : `.patches.Patch`
            尾部补丁对象。
        """
        self.patchA = patchA
        self.stale = True

    # 设置头部补丁。
    def set_patchB(self, patchB):
        """
        Set the head patch.

        Parameters
        ----------
        patchB : `.patches.Patch`
            头部补丁对象。
        """
        self.patchB = patchB
        self.stale = True

    # 设置连接风格，可能包含更多属性。
    @_docstring.dedent_interpd
    def set_connectionstyle(self, connectionstyle=None, **kwargs):
        """
        Set the connection style, possibly with further attributes.

        Attributes from the previous connection style are not reused.

        Without argument (or with ``connectionstyle=None``), the available box
        styles are returned as a human-readable string.

        Parameters
        ----------
        connectionstyle : str or `~matplotlib.patches.ConnectionStyle`
            The style of the connection: either a `.ConnectionStyle` instance,
            or a string, which is the style name and optionally comma separated
            attributes (e.g. "Arc,armA=30,rad=10"). Such a string is used to
            construct a `.ConnectionStyle` object, as documented in that class.

            The following connection styles are available:

            %(ConnectionStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the connection style. See the table above
            for supported parameters.

        Examples
        --------
        ::

            set_connectionstyle("Arc,armA=30,rad=10")
            set_connectionstyle("arc", armA=30, rad=10)
        """
        if connectionstyle is None:
            # 如果未指定连接风格，则返回可用的连接风格列表的人类可读字符串。
            return ConnectionStyle.pprint_styles()
        # 根据给定的连接风格字符串或对象，设置连接器对象。
        self._connector = (
            ConnectionStyle(connectionstyle, **kwargs)
            if isinstance(connectionstyle, str) else connectionstyle)
        self.stale = True

    # 返回当前使用的连接风格对象。
    def get_connectionstyle(self):
        """Return the `ConnectionStyle` used."""
        return self._connector
    def set_arrowstyle(self, arrowstyle=None, **kwargs):
        """
        Set the arrow style, possibly with further attributes.

        Attributes from the previous arrow style are not reused.

        Without argument (or with ``arrowstyle=None``), the available box
        styles are returned as a human-readable string.

        Parameters
        ----------
        arrowstyle : str or `~matplotlib.patches.ArrowStyle`
            The style of the arrow: either a `.ArrowStyle` instance, or a
            string, which is the style name and optionally comma separated
            attributes (e.g. "Fancy,head_length=0.2"). Such a string is used to
            construct a `.ArrowStyle` object, as documented in that class.

            The following arrow styles are available:

            %(ArrowStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the arrow style. See the table above for
            supported parameters.

        Examples
        --------
        ::

            set_arrowstyle("Fancy,head_length=0.2")
            set_arrowstyle("fancy", head_length=0.2)
        """
        # 如果 arrowstyle 为 None，则返回可用的箭头样式列表的人类可读字符串
        if arrowstyle is None:
            return ArrowStyle.pprint_styles()
        
        # 根据传入的 arrowstyle 和 kwargs 创建一个新的 ArrowStyle 对象
        self._arrow_transmuter = (
            ArrowStyle(arrowstyle, **kwargs)
            if isinstance(arrowstyle, str) else arrowstyle)
        
        # 将 stale 属性设置为 True，表示对象需要重新绘制
        self.stale = True

    def get_arrowstyle(self):
        """Return the arrowstyle object."""
        # 返回当前箭头样式对象 _arrow_transmuter
        return self._arrow_transmuter

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        Parameters
        ----------
        scale : float
            The scale factor for the mutation.
        """
        # 设置 bbox 变换的缩放比例
        self._mutation_scale = scale
        
        # 将 stale 属性设置为 True，表示对象需要重新绘制
        self.stale = True

    def get_mutation_scale(self):
        """
        Return the mutation scale.

        Returns
        -------
        scalar
            The current scale factor for the mutation.
        """
        # 返回当前的 bbox 变换缩放比例 _mutation_scale
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        Parameters
        ----------
        aspect : float
            The aspect ratio to set.
        """
        # 设置 bbox 变换的长宽比例
        self._mutation_aspect = aspect
        
        # 将 stale 属性设置为 True，表示对象需要重新绘制
        self.stale = True

    def get_mutation_aspect(self):
        """
        Return the aspect ratio of the bbox mutation.

        Returns
        -------
        float
            The current aspect ratio of the bbox mutation.
        """
        # 返回当前的 bbox 变换长宽比例 _mutation_aspect，若为 None，则返回默认值 1
        return (self._mutation_aspect if self._mutation_aspect is not None
                else 1)  # backcompat.

    def get_path(self):
        """
        Return the path of the arrow in the data coordinates.

        Returns
        -------
        `matplotlib.path.Path`
            The path of the arrow in data coordinates.
        """
        # 获取箭头在显示坐标系中的路径 _path，以及一个是否可填充的标志 fillable
        _path, fillable = self._get_path_in_displaycoord()
        
        # 如果 fillable 是可迭代的，则将 _path 转换为复合路径
        if np.iterable(fillable):
            _path = Path.make_compound_path(*_path)
        
        # 将显示坐标系中的路径 _path 转换为数据坐标系中的路径，并返回
        return self.get_transform().inverted().transform_path(_path)
    def _get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        # 获取当前对象的 DPI 相关校正系数
        dpi_cor = self._dpi_cor

        # 如果存在起始点和结束点的坐标
        if self._posA_posB is not None:
            # 将起始点和结束点的坐标转换为适当的单位
            posA = self._convert_xy_units(self._posA_posB[0])
            posB = self._convert_xy_units(self._posA_posB[1])
            # 使用对象的变换函数将坐标转换为显示坐标系下的坐标
            (posA, posB) = self.get_transform().transform((posA, posB))
            # 使用连接样式函数创建路径对象，考虑到相关属性如补丁和收缩比例
            _path = self.get_connectionstyle()(posA, posB,
                                               patchA=self.patchA,
                                               patchB=self.patchB,
                                               shrinkA=self.shrinkA * dpi_cor,
                                               shrinkB=self.shrinkB * dpi_cor
                                               )
        else:
            # 否则，使用变换函数将原始路径转换为显示坐标系下的路径
            _path = self.get_transform().transform_path(self._path_original)

        # 根据箭头样式和 DPI 相关校正系数来调整路径和可填充性
        _path, fillable = self.get_arrowstyle()(
            _path,
            self.get_mutation_scale() * dpi_cor,
            self.get_linewidth() * dpi_cor,
            self.get_mutation_aspect())

        # 返回调整后的路径和可填充性
        return _path, fillable

    def draw(self, renderer):
        if not self.get_visible():
            return

        # FIXME: dpi_cor 是为了线条宽度的 DPI 依赖性。可能还有改进的空间。
        # 可能 _get_path_in_displaycoord 函数可以接受一个 renderer 参数，
        # 但是 get_path 也需要相应调整。
        # 计算当前渲染器下的 DPI 相关校正系数
        self._dpi_cor = renderer.points_to_pixels(1.)
        # 获取箭头的路径和可填充性
        path, fillable = self._get_path_in_displaycoord()

        # 如果填充性不可迭代，则将路径转换为列表
        if not np.iterable(fillable):
            path = [path]
            fillable = [fillable]

        # 使用恒等变换对象
        affine = transforms.IdentityTransform()

        # 使用当前对象的属性绘制路径
        self._draw_paths_with_artist_properties(
            renderer,
            [(p, affine, self._facecolor if f and self._facecolor[3] else None)
             for p, f in zip(path, fillable)])
# 定义一个自定义连接箭头的类，继承自FancyArrowPatch类
class ConnectionPatch(FancyArrowPatch):
    
    # 返回连接箭头的字符串表示
    def __str__(self):
        return "ConnectionPatch((%g, %g), (%g, %g))" % \
               (self.xy1[0], self.xy1[1], self.xy2[0], self.xy2[1])

    # 装饰器：用于处理文档字符串的缩进问题
    @_docstring.dedent_interpd
    # 计算给定点的像素位置
    def _get_xy(self, xy, s, axes=None):
        """Calculate the pixel position of given point."""
        s0 = s  # 保存原始坐标系统名称，以便在需要时作为错误消息的一部分使用
        
        # 如果未提供Axes对象，则使用self.axes
        if axes is None:
            axes = self.axes
        
        # 将坐标转换为NumPy数组
        xy = np.array(xy)
        
        # 根据坐标系统进行转换
        if s in ["figure points", "axes points"]:
            # 将坐标转换为像素单位
            xy *= self.figure.dpi / 72
            s = s.replace("points", "pixels")
        elif s == "figure fraction":
            s = self.figure.transFigure
        elif s == "subfigure fraction":
            s = self.figure.transSubfigure
        elif s == "axes fraction":
            s = axes.transAxes
        
        # 获取坐标点的x和y值
        x, y = xy

        # 根据不同的坐标系统进行坐标转换
        if s == 'data':
            # 使用Axes对象的数据转换进行坐标转换
            trans = axes.transData
            x = float(self.convert_xunits(x))
            y = float(self.convert_yunits(y))
            return trans.transform((x, y))
        elif s == 'offset points':
            # 如果xycoords也是offset points，则避免递归调用，直接返回对应的数据点
            if self.xycoords == 'offset points':
                return self._get_xy(self.xy, 'data')
            # 返回转换后的数据点加上转换后的偏移量
            return (
                self._get_xy(self.xy, self.xycoords)  # 转换后的数据点
                + xy * self.figure.dpi / 72)  # 转换后的偏移量
        elif s == 'polar':
            # 如果坐标系统是极坐标，则将极坐标转换为笛卡尔坐标
            theta, r = x, y
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            trans = axes.transData
            return trans.transform((x, y))
        elif s == 'figure pixels':
            # 基于图形左下角的像素坐标系
            bb = self.figure.figbbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif s == 'subfigure pixels':
            # 基于子图左下角的像素坐标系
            bb = self.figure.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif s == 'axes pixels':
            # 基于Axes左下角的像素坐标系
            bb = axes.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif isinstance(s, transforms.Transform):
            # 如果坐标系统是一个Transform对象，则使用其进行坐标转换
            return s.transform(xy)
        else:
            # 如果坐标系统不在预定义的类型中，则抛出错误
            raise ValueError(f"{s0} is not a valid coordinate transformation")
    def set_annotation_clip(self, b):
        """
        设置注释对象的裁剪行为。

        Parameters
        ----------
        b : bool or None
            - True: 当 `self.xy` 在 Axes 外部时，注释将被裁剪。
            - False: 注释将始终被绘制。
            - None: 当 `self.xy` 在 Axes 外部且 `self.xycoords == "data"` 时，注释将被裁剪。
        """
        # 设置注释的裁剪行为为参数 b
        self._annotation_clip = b
        # 设置对象状态为过期，需要重新绘制
        self.stale = True

    def get_annotation_clip(self):
        """
        返回裁剪行为。

        查看 `.set_annotation_clip` 获取返回值的含义。
        """
        # 返回注释的裁剪行为
        return self._annotation_clip

    def _get_path_in_displaycoord(self):
        """返回箭头在显示坐标系中的变异路径。"""
        dpi_cor = self._dpi_cor
        # 获取起始点 posA 和结束点 posB 的坐标
        posA = self._get_xy(self.xy1, self.coords1, self.axesA)
        posB = self._get_xy(self.xy2, self.coords2, self.axesB)
        # 根据连接样式获取路径
        path = self.get_connectionstyle()(
            posA, posB,
            patchA=self.patchA, patchB=self.patchB,
            shrinkA=self.shrinkA * dpi_cor, shrinkB=self.shrinkB * dpi_cor,
        )
        # 根据箭头样式获取路径和填充属性
        path, fillable = self.get_arrowstyle()(
            path,
            self.get_mutation_scale() * dpi_cor,
            self.get_linewidth() * dpi_cor,
            self.get_mutation_aspect()
        )
        return path, fillable

    def _check_xy(self, renderer):
        """检查是否需要绘制注释。"""

        b = self.get_annotation_clip()

        # 如果 b 为 True 或者 b 为 None 且 self.coords1 为 "data"，则进行下列操作
        if b or (b is None and self.coords1 == "data"):
            # 获取第一个坐标点的像素位置
            xy_pixel = self._get_xy(self.xy1, self.coords1, self.axesA)
            if self.axesA is None:
                axes = self.axes
            else:
                axes = self.axesA
            # 如果 axes 不包含 xy_pixel 点，则返回 False
            if not axes.contains_point(xy_pixel):
                return False

        # 如果 b 为 True 或者 b 为 None 且 self.coords2 为 "data"，则进行下列操作
        if b or (b is None and self.coords2 == "data"):
            # 获取第二个坐标点的像素位置
            xy_pixel = self._get_xy(self.xy2, self.coords2, self.axesB)
            if self.axesB is None:
                axes = self.axes
            else:
                axes = self.axesB
            # 如果 axes 不包含 xy_pixel 点，则返回 False
            if not axes.contains_point(xy_pixel):
                return False

        # 如果以上条件都不满足，则返回 True
        return True

    def draw(self, renderer):
        # 如果对象不可见或者不需要绘制注释，则直接返回
        if not self.get_visible() or not self._check_xy(renderer):
            return
        # 调用父类的 draw 方法进行绘制
        super().draw(renderer)
```