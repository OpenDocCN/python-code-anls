# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\axes3d.py`

```py
"""
axes3d.py, original mplot3d version by John Porter
Created: 23 Sep 2005

Parts fixed by Reinier Heeres <reinier@heeres.eu>
Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>

Module containing Axes3D, an object which can plot 3D objects on a
2D matplotlib figure.
"""

# 导入必要的模块和库
from collections import defaultdict  # 导入 defaultdict 类
import itertools  # 导入 itertools 模块
import math  # 导入 math 数学库
import textwrap  # 导入 textwrap 文本包装模块
import warnings  # 导入 warnings 警告模块

import numpy as np  # 导入 NumPy 数学计算库

import matplotlib as mpl  # 导入 matplotlib 绘图库
from matplotlib import _api, cbook, _docstring, _preprocess_data  # 导入 matplotlib 内部模块
import matplotlib.artist as martist  # 导入 matplotlib.artist 模块
import matplotlib.collections as mcoll  # 导入 matplotlib.collections 模块
import matplotlib.colors as mcolors  # 导入 matplotlib.colors 模块
import matplotlib.image as mimage  # 导入 matplotlib.image 模块
import matplotlib.lines as mlines  # 导入 matplotlib.lines 模块
import matplotlib.patches as mpatches  # 导入 matplotlib.patches 模块
import matplotlib.container as mcontainer  # 导入 matplotlib.container 模块
import matplotlib.transforms as mtransforms  # 导入 matplotlib.transforms 模块
from matplotlib.axes import Axes  # 从 matplotlib.axes 模块导入 Axes 类
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format  # 导入 Axes 相关内部函数
from matplotlib.transforms import Bbox  # 导入 Bbox 类
from matplotlib.tri._triangulation import Triangulation  # 导入 Triangulation 类

from . import art3d  # 从当前包导入 art3d 模块
from . import proj3d  # 从当前包导入 proj3d 模块
from . import axis3d  # 从当前包导入 axis3d 模块


@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):
    """
    3D Axes object.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.
    """
    name = '3d'  # 设定对象名称为 '3d'

    _axis_names = ("x", "y", "z")  # 设置坐标轴名称元组
    Axes._shared_axes["z"] = cbook.Grouper()  # 将 z 轴添加到共享坐标轴组中
    Axes._shared_axes["view"] = cbook.Grouper()  # 将 view 视图添加到共享坐标轴组中

    def __init__(
        self, fig, rect=None, *args,
        elev=30, azim=-60, roll=0, shareview=None, sharez=None,
        proj_type='persp', focal_length=None,
        box_aspect=None,
        computed_zorder=True,
        **kwargs,
    ):
        """
        Initialize 3D Axes.

        Parameters:
        - fig : Figure
            Parent figure.
        - rect : tuple or None, optional
            [left, bottom, width, height] of the new axes. None defaults to
            subplot.
        - *args
            Variable length argument list.
        - elev : float, default: 30
            Elevation viewing angle.
        - azim : float, default: -60
            Azimuthal viewing angle.
        - roll : float, default: 0
            Rotation viewing angle.
        - shareview : Axes3D, optional
            Share view configuration with another Axes3D object.
        - sharez : Axes3D, optional
            Share z-axis configuration with another Axes3D object.
        - proj_type : {'persp', 'ortho'}, default: 'persp'
            Projection type.
        - focal_length : float, optional
            Focal length of the projection. Defaults to None.
        - box_aspect : tuple or None, optional
            Aspect ratio of the axes.
        - computed_zorder : bool, default: True
            Whether to compute z-order of artists.
        - **kwargs
            Other keyword arguments.

        """
        super().__init__(fig, rect, *args, **kwargs)
        self.dist = 10  # 设置初始距离为 10
        self.elev = elev  # 设置俯仰角
        self.azim = azim  # 设置方位角
        self.roll = roll  # 设置旋转角度
        self.sharex = shareview  # 设置共享 x 轴
        self.sharey = shareview  # 设置共享 y 轴
        self.sharez = sharez  # 设置共享 z 轴
        self.proj_type = proj_type  # 设置投影类型
        self.focal_length = focal_length  # 设置焦距
        self.box_aspect = box_aspect  # 设置盒子纵横比
        self.computed_zorder = computed_zorder  # 设置是否计算 Z 轴顺序

    def set_axis_off(self):
        """Turn off 3D axes."""
        self._axis3don = False  # 设置 3D 坐标轴为关闭状态
        self.stale = True  # 标记为需要刷新

    def set_axis_on(self):
        """Turn on 3D axes."""
        self._axis3don = True  # 设置 3D 坐标轴为开启状态
        self.stale = True  # 标记为需要刷新

    def convert_zunits(self, z):
        """
        Convert z-axis units for artists in the Axes.

        Parameters:
        - z : scalar
            Z-axis value to convert.

        Returns:
        - scalar
            Converted z-axis value.

        """
        return self.zaxis.convert_units(z)  # 调用 z 轴的单位转换方法

    def set_top_view(self):
        """
        Set top view for the 3D plot.

        This adjusts the viewing pane to fit labels and axes properly.

        """
        # Calculate viewport adjustments
        xdwl = 0.95 / self.dist
        xdw = 0.9 / self.dist
        ydwl = 0.95 / self.dist
        ydw = 0.9 / self.dist
        # Set the viewing pane.
        self.viewLim.intervalx = (-xdwl, xdw)
        self.viewLim.intervaly = (-ydwl, ydw)
        self.stale = True  # 标记为需要刷新

    def _init_axis(self):
        """Initialize 3D Axes; overrides creation of regular X/Y Axes."""
        self.xaxis = axis3d.XAxis(self)  # 创建自定义的 X 轴对象
        self.yaxis = axis3d.YAxis(self)  # 创建自定义的 Y 轴对象
        self.zaxis = axis3d.ZAxis(self)  # 创建自定义的 Z 轴对象
    # 返回与 Z 轴相关联的 `ZAxis` (`~.axis3d.Axis`) 实例
    def get_zaxis(self):
        return self.zaxis

    # 返回与 Z 轴相关联的网格线
    get_zgridlines = _axis_method_wrapper("zaxis", "get_gridlines")

    # 返回与 Z 轴相关联的刻度线
    get_zticklines = _axis_method_wrapper("zaxis", "get_ticklines")

    # 根据给定的值 *vals*，通过 self.M 进行转换，并返回限制范围内的立方体
    def _transformed_cube(self, vals):
        minx, maxx, miny, maxy, minz, maxz = vals
        # 定义立方体的八个顶点坐标
        xyzs = [(minx, miny, minz),
                (maxx, miny, minz),
                (maxx, maxy, minz),
                (minx, maxy, minz),
                (minx, miny, maxz),
                (maxx, miny, maxz),
                (maxx, maxy, maxz),
                (minx, maxy, maxz)]
        # 使用 self.M 对立方体的顶点进行投影，并返回结果
        return proj3d._proj_points(xyzs, self.M)

    # 根据给定的 `aspect` 参数确定哪些轴需要保持相等的纵横比
    def _equal_aspect_axis_indices(self, aspect):
        ax_indices = []  # 默认情况下，不设置任何约束
        if aspect == 'equal':
            ax_indices = [0, 1, 2]  # x, y, z 轴都保持相等的纵横比
        elif aspect == 'equalxy':
            ax_indices = [0, 1]  # 仅 x 和 y 轴保持相等的纵横比
        elif aspect == 'equalxz':
            ax_indices = [0, 2]  # 仅 x 和 z 轴保持相等的纵横比
        elif aspect == 'equalyz':
            ax_indices = [1, 2]  # 仅 y 和 z 轴保持相等的纵横比
        return ax_indices
    def set_box_aspect(self, aspect, *, zoom=1):
        """
        Set the Axes box aspect.

        The box aspect is the ratio of height to width in display
        units for each face of the box when viewed perpendicular to
        that face.  This is not to be confused with the data aspect (see
        `~.Axes3D.set_aspect`). The default ratios are 4:4:3 (x:y:z).

        To simulate having equal aspect in data space, set the box
        aspect to match your data range in each dimension.

        *zoom* controls the overall size of the Axes3D in the figure.

        Parameters
        ----------
        aspect : 3-tuple of floats or None
            Changes the physical dimensions of the Axes3D, such that the ratio
            of the axis lengths in display units is x:y:z.
            If None, defaults to (4, 4, 3).

        zoom : float, default: 1
            Control overall size of the Axes3D in the figure. Must be > 0.
        """
        # 检查 zoom 参数是否合法，必须大于 0
        if zoom <= 0:
            raise ValueError(f'Argument zoom = {zoom} must be > 0')

        # 如果 aspect 参数为 None，则设置默认的比例 (4, 4, 3)
        if aspect is None:
            aspect = np.asarray((4, 4, 3), dtype=float)
        else:
            # 将 aspect 转换为 numpy 数组，并检查其形状是否为 (3,)
            aspect = np.asarray(aspect, dtype=float)
            _api.check_shape((3,), aspect=aspect)
        
        # 调整比例以匹配所需的视觉效果
        # 根据经验值调整比例，以匹配 matplotlib 3.2 的外观
        # 使用 zoom 参数和当前比例的归一化值来调整比例
        aspect *= 1.8294640721620434 * 25/24 * zoom / np.linalg.norm(aspect)

        # 将调整后的比例保存到 _box_aspect 属性中
        self._box_aspect = self._roll_to_vertical(aspect, reverse=True)
        # 设置为需要刷新状态
        self.stale = True

    def apply_aspect(self, position=None):
        # 如果 position 为 None，则获取当前位置
        if position is None:
            position = self.get_position(original=True)

        # 获取图形的变换矩阵
        trans = self.get_figure().transSubfigure
        # 计算子图或图形的物理比例
        bb = mtransforms.Bbox.unit().transformed(trans)
        fig_aspect = bb.height / bb.width  # 这是面板（或图形）的物理比例

        # 设置 box_aspect 为 1（即正方形）
        box_aspect = 1
        pb = position.frozen()
        # 调整位置以匹配给定的 box_aspect 和图形的物理比例
        pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
        # 设置新的位置，并指定为 'active'
        self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')

    @martist.allow_rasterization
    def draw(self, renderer):
        # 如果图形不可见，则直接返回
        if not self.get_visible():
            return
        # 更新视图限制
        self._unstale_viewLim()

        # 绘制背景补丁
        self.patch.draw(renderer)
        self._frameon = False

        # 首先设置图形的比例
        # 这部分是从 `axes._base._AxesBase.draw` 复制过来的
        # 必须在绘制任何艺术家之前调用，因为它调整视图限制和边界框的大小
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)

        # 将投影矩阵添加到渲染器中
        self.M = self.get_proj()
        self.invM = np.linalg.inv(self.M)

        # 获取所有集合和补丁，仅包括可见的部分
        collections_and_patches = (
            artist for artist in self._children
            if isinstance(artist, (mcoll.Collection, mpatches.Patch))
            and artist.get_visible())
        if self.computed_zorder:
            # 如果计算了 zorder，则计算集合和补丁的投影并排序
            # 确保它们在网格之上绘制
            zorder_offset = max(axis.get_zorder()
                                for axis in self._axis_map.values()) + 1
            collection_zorder = patch_zorder = zorder_offset

            for artist in sorted(collections_and_patches,
                                 key=lambda artist: artist.do_3d_projection(),
                                 reverse=True):
                if isinstance(artist, mcoll.Collection):
                    artist.zorder = collection_zorder
                    collection_zorder += 1
                elif isinstance(artist, mpatches.Patch):
                    artist.zorder = patch_zorder
                    patch_zorder += 1
        else:
            # 否则，只做3D投影变换
            for artist in collections_and_patches:
                artist.do_3d_projection()

        if self._axis3don:
            # 先绘制平面
            for axis in self._axis_map.values():
                axis.draw_pane(renderer)
            # 然后绘制网格线
            for axis in self._axis_map.values():
                axis.draw_grid(renderer)
            # 最后绘制坐标轴、标签、文本和刻度
            for axis in self._axis_map.values():
                axis.draw(renderer)

        # 最后绘制剩余部分
        super().draw(renderer)

    def get_axis_position(self):
        # 获取坐标轴位置信息
        tc = self._transformed_cube(self.get_w_lims())
        xhigh = tc[1][2] > tc[2][2]
        yhigh = tc[3][2] > tc[2][2]
        zhigh = tc[0][2] > tc[2][2]
        return xhigh, yhigh, zhigh

    def update_datalim(self, xys, **kwargs):
        """
        `~mpl_toolkits.mplot3d.axes3d.Axes3D` 中未实现的方法。
        """
        # 更新数据限制，但在 `Axes3D` 中未实现

    get_autoscalez_on = _axis_method_wrapper("zaxis", "_get_autoscale_on")
    set_autoscalez_on = _axis_method_wrapper("zaxis", "_set_autoscale_on")
    def get_zmargin(self):
        """
        Retrieve autoscaling margin of the z-axis.

        .. versionadded:: 3.9

        Returns
        -------
        zmargin : float
            Current margin value for autoscaling the z-axis.

        See Also
        --------
        mpl_toolkits.mplot3d.axes3d.Axes3D.set_zmargin
            Method for setting the z-axis margin.
        """
        return self._zmargin

    def set_zmargin(self, m):
        """
        Set padding of Z data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        Parameters
        ----------
        m : float greater than -0.5
            Margin multiplier to adjust the z-axis limits.

        Raises
        ------
        ValueError
            If the margin value is less than or equal to -0.5.
        """
        if m <= -0.5:
            raise ValueError("margin must be greater than -0.5")
        self._zmargin = m
        self._request_autoscale_view("z")
        self.stale = True

    def margins(self, *margins, x=None, y=None, z=None, tight=True):
        """
        Set or retrieve autoscaling margins for x, y, and z axes.

        Parameters
        ----------
        *margins : tuple of floats
            If passed as a tuple, sets equal margins for x, y, and z axes.
        x, y, z : float or None
            Individual margins for x, y, and z axes respectively.
        tight : bool, optional
            Whether to adjust view tightly around data.

        Returns
        -------
        xmargin, ymargin, zmargin : float
            Current margins for x, y, and z axes respectively.

        Raises
        ------
        TypeError
            If both positional and keyword arguments for x, y, or z are provided.
            If an incorrect number of positional arguments are provided for margins.

        Notes
        -----
        If no arguments are provided for x, y, and z, returns current margins.
        """
        if margins and (x is not None or y is not None or z is not None):
            raise TypeError('Cannot pass both positional and keyword '
                            'arguments for x, y, and/or z.')
        elif len(margins) == 1:
            x = y = z = margins[0]
        elif len(margins) == 3:
            x, y, z = margins
        elif margins:
            raise TypeError('Must pass a single positional argument for all '
                            'margins, or one for each margin (x, y, z).')

        if x is None and y is None and z is None:
            if tight is not True:
                _api.warn_external(f'ignoring tight={tight!r} in get mode')
            return self._xmargin, self._ymargin, self._zmargin

        if x is not None:
            self.set_xmargin(x)
        if y is not None:
            self.set_ymargin(y)
        if z is not None:
            self.set_zmargin(z)

        self.autoscale_view(
            tight=tight, scalex=(x is not None), scaley=(y is not None),
            scalez=(z is not None)
        )
    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Convenience method for simple axis view autoscaling.

        See `.Axes.autoscale` for full documentation.  Because this function
        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
        to 'both' autoscales all three axes.
        """
        # 如果 enable 为 None，则默认全部启用自动缩放
        if enable is None:
            scalex = True
            scaley = True
            scalez = True
        else:
            # 根据 axis 参数设置各个轴的自动缩放状态
            if axis in ['x', 'both']:
                self.set_autoscalex_on(enable)
                scalex = self.get_autoscalex_on()
            else:
                scalex = False
            if axis in ['y', 'both']:
                self.set_autoscaley_on(enable)
                scaley = self.get_autoscaley_on()
            else:
                scaley = False
            if axis in ['z', 'both']:
                self.set_autoscalez_on(enable)
                scalez = self.get_autoscalez_on()
            else:
                scalez = False
        # 根据各个轴的自动缩放状态，请求自动缩放视图
        if scalex:
            self._request_autoscale_view("x", tight=tight)
        if scaley:
            self._request_autoscale_view("y", tight=tight)
        if scalez:
            self._request_autoscale_view("z", tight=tight)

    def auto_scale_xyz(self, X, Y, Z=None, had_data=None):
        # 更新数据的边界框，以保持记录最小尺寸的矩形体积包含数据
        if np.shape(X) == np.shape(Y):
            self.xy_dataLim.update_from_data_xy(
                np.column_stack([np.ravel(X), np.ravel(Y)]), not had_data)
        else:
            self.xy_dataLim.update_from_data_x(X, not had_data)
            self.xy_dataLim.update_from_data_y(Y, not had_data)
        if Z is not None:
            self.zz_dataLim.update_from_data_x(Z, not had_data)
        # 让 autoscale_view 决定如何使用这些数据进行视图自动缩放
        self.autoscale_view()
    def autoscale_view(self, tight=None,
                       scalex=True, scaley=True, scalez=True):
        """
        Autoscale the view limits using the data limits.

        See `.Axes.autoscale_view` for full documentation.  Because this
        function applies to 3D Axes, it also takes a *scalez* argument.
        """
        # 根据数据限制自动调整视图限制

        # 如果 tight 参数为 None，则使用 self._tight 的值
        if tight is None:
            _tight = self._tight
            # 如果 self._tight 为 False，则根据子元素类型决定是否设置 _tight 为 True
            if not _tight:
                for artist in self._children:
                    if isinstance(artist, mimage.AxesImage):
                        _tight = True
                    elif isinstance(artist, (mlines.Line2D, mpatches.Patch)):
                        _tight = False
                        break
        else:
            # 否则，将 tight 赋值给 _tight，并设置 self._tight 为 tight 的布尔值
            _tight = self._tight = bool(tight)

        # 如果 scalex 为 True 且 x 轴自动缩放开启
        if scalex and self.get_autoscalex_on():
            # 获取 x 轴的数据限制
            x0, x1 = self.xy_dataLim.intervalx
            xlocator = self.xaxis.get_major_locator()
            # 对 x0 和 x1 进行非奇异性处理
            x0, x1 = xlocator.nonsingular(x0, x1)
            # 如果 self._xmargin 大于 0，则应用视图边界修正
            if self._xmargin > 0:
                delta = (x1 - x0) * self._xmargin
                x0 -= delta
                x1 += delta
            # 如果不是 tight 模式，则调用 xlocator 的 view_limits 方法
            if not _tight:
                x0, x1 = xlocator.view_limits(x0, x1)
            # 设置 x 轴的界限
            self.set_xbound(x0, x1, self._view_margin)

        # 如果 scaley 为 True 且 y 轴自动缩放开启
        if scaley and self.get_autoscaley_on():
            # 获取 y 轴的数据限制
            y0, y1 = self.xy_dataLim.intervaly
            ylocator = self.yaxis.get_major_locator()
            # 对 y0 和 y1 进行非奇异性处理
            y0, y1 = ylocator.nonsingular(y0, y1)
            # 如果 self._ymargin 大于 0，则应用视图边界修正
            if self._ymargin > 0:
                delta = (y1 - y0) * self._ymargin
                y0 -= delta
                y1 += delta
            # 如果不是 tight 模式，则调用 ylocator 的 view_limits 方法
            if not _tight:
                y0, y1 = ylocator.view_limits(y0, y1)
            # 设置 y 轴的界限
            self.set_ybound(y0, y1, self._view_margin)

        # 如果 scalez 为 True 且 z 轴自动缩放开启
        if scalez and self.get_autoscalez_on():
            # 获取 z 轴的数据限制
            z0, z1 = self.zz_dataLim.intervalx
            zlocator = self.zaxis.get_major_locator()
            # 对 z0 和 z1 进行非奇异性处理
            z0, z1 = zlocator.nonsingular(z0, z1)
            # 如果 self._zmargin 大于 0，则应用视图边界修正
            if self._zmargin > 0:
                delta = (z1 - z0) * self._zmargin
                z0 -= delta
                z1 += delta
            # 如果不是 tight 模式，则调用 zlocator 的 view_limits 方法
            if not _tight:
                z0, z1 = zlocator.view_limits(z0, z1)
            # 设置 z 轴的界限
            self.set_zbound(z0, z1, self._view_margin)

    def get_w_lims(self):
        """Get 3D world limits."""
        # 获取 3D 世界的限制，并返回 minx, maxx, miny, maxy, minz, maxz
        minx, maxx = self.get_xlim3d()
        miny, maxy = self.get_ylim3d()
        minz, maxz = self.get_zlim3d()
        return minx, maxx, miny, maxy, minz, maxz
    def _set_bound3d(self, get_bound, set_lim, axis_inverted,
                     lower=None, upper=None, view_margin=None):
        """
        Set 3D axis bounds.
        """
        # 如果 `upper` 为 None 并且 `lower` 是可迭代对象，解包并设置 `lower` 和 `upper`
        if upper is None and np.iterable(lower):
            lower, upper = lower

        # 获取当前的轴界限并保存为 `old_lower` 和 `old_upper`
        old_lower, old_upper = get_bound()
        
        # 如果 `lower` 为 None，则使用旧的 lower 边界
        if lower is None:
            lower = old_lower
        # 如果 `upper` 为 None，则使用旧的 upper 边界
        if upper is None:
            upper = old_upper

        # 根据轴反转的情况，排序并设置新的轴界限，不启用自动缩放，视图边距为 `view_margin`
        set_lim(sorted((lower, upper), reverse=bool(axis_inverted())),
                auto=None, view_margin=view_margin)

    def set_xbound(self, lower=None, upper=None, view_margin=None):
        """
        Set the lower and upper numerical bounds of the x-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalex_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.
        view_margin : float or None
            The margin to apply to the bounds. If *None*, the margin is handled
            by `.set_xlim`.

        See Also
        --------
        get_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
        # 使用 `_set_bound3d` 方法设置 x 轴界限
        self._set_bound3d(self.get_xbound, self.set_xlim, self.xaxis_inverted,
                          lower, upper, view_margin)

    def set_ybound(self, lower=None, upper=None, view_margin=None):
        """
        Set the lower and upper numerical bounds of the y-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.
        view_margin : float or None
            The margin to apply to the bounds. If *None*, the margin is handled
            by `.set_ylim`.

        See Also
        --------
        get_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
        # 使用 `_set_bound3d` 方法设置 y 轴界限
        self._set_bound3d(self.get_ybound, self.set_ylim, self.yaxis_inverted,
                          lower, upper, view_margin)
    def set_zbound(self, lower=None, upper=None, view_margin=None):
        """
        Set the lower and upper numerical bounds of the z-axis.
        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.
        view_margin : float or None
            The margin to apply to the bounds. If *None*, the margin is handled
            by `.set_zlim`.

        See Also
        --------
        get_zbound
        get_zlim, set_zlim
        invert_zaxis, zaxis_inverted
        """
        # 调用私有方法 `_set_bound3d` 来设置 3D 坐标轴的界限
        self._set_bound3d(self.get_zbound, self.set_zlim, self.zaxis_inverted,
                          lower, upper, view_margin)

    def _set_lim3d(self, axis, lower=None, upper=None, *, emit=True,
                   auto=False, view_margin=None, axmin=None, axmax=None):
        """
        Set 3D axis limits.
        """
        # 如果 `upper` 为 None
        if upper is None:
            # 如果 `lower` 是可迭代的，将其解包为 `lower` 和 `upper`
            if np.iterable(lower):
                lower, upper = lower
            # 否则如果 `axmax` 也为 None，则 `upper` 设为当前视图间隔的上限
            elif axmax is None:
                upper = axis.get_view_interval()[1]
        # 如果 `lower` 为 None 且 `axmin` 也为 None，则 `lower` 设为当前视图间隔的下限
        if lower is None and axmin is None:
            lower = axis.get_view_interval()[0]
        # 如果指定了 `axmin`
        if axmin is not None:
            # 如果 `lower` 已经有值，抛出 TypeError
            if lower is not None:
                raise TypeError("Cannot pass both 'lower' and 'min'")
            lower = axmin
        # 如果指定了 `axmax`
        if axmax is not None:
            # 如果 `upper` 已经有值，抛出 TypeError
            if upper is not None:
                raise TypeError("Cannot pass both 'upper' and 'max'")
            upper = axmax
        # 如果 `lower` 或 `upper` 为无穷大，抛出 ValueError
        if np.isinf(lower) or np.isinf(upper):
            raise ValueError(f"Axis limits {lower}, {upper} cannot be infinite")
        # 如果 `view_margin` 为 None
        if view_margin is None:
            # 如果 `mpl.rcParams['axes3d.automargin']` 开启，则使用类属性 `_view_margin`
            if mpl.rcParams['axes3d.automargin']:
                view_margin = self._view_margin
            # 否则视图间隔为 0
            else:
                view_margin = 0
        # 计算边界增量，并调整 `lower` 和 `upper`
        delta = (upper - lower) * view_margin
        lower -= delta
        upper += delta
        # 调用 axis 对象的 `_set_lim` 方法来设置坐标轴的上下界限，并返回结果
        return axis._set_lim(lower, upper, emit=emit, auto=auto)
    def set_xlim(self, left=None, right=None, *, emit=True, auto=False,
                 view_margin=None, xmin=None, xmax=None):
        """
        Set the 3D x-axis view limits.

        Parameters
        ----------
        left : float, optional
            The left xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The left and right xlims may also be passed as the tuple
            (*left*, *right*) as the first positional argument (or as
            the *left* keyword argument).

            .. ACCEPTS: (left: float, right: float)

        right : float, optional
            The right xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the x-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        view_margin : float, optional
            The additional margin to apply to the limits.

        xmin, xmax : float, optional
            They are equivalent to left and right respectively, and it is an
            error to pass both *xmin* and *left* or *xmax* and *right*.

        Returns
        -------
        left, right : (float, float)
            The new x-axis limits in data coordinates.

        See Also
        --------
        get_xlim
        set_xbound, get_xbound
        invert_xaxis, xaxis_inverted

        Notes
        -----
        The *left* value may be greater than the *right* value, in which
        case the x-axis values will decrease from *left* to *right*.

        Examples
        --------
        >>> set_xlim(left, right)
        >>> set_xlim((left, right))
        >>> left, right = set_xlim(left, right)

        One limit may be left unchanged.

        >>> set_xlim(right=right_lim)

        Limits may be passed in reverse order to flip the direction of
        the x-axis. For example, suppose ``x`` represents depth of the
        ocean in m. The x-axis limits might be set like the following
        so 5000 m depth is at the left of the plot and the surface,
        0 m, is at the right.

        >>> set_xlim(5000, 0)
        """
        # 调用内部方法 `_set_lim3d` 设置三维坐标系的 x 轴限制
        return self._set_lim3d(self.xaxis, left, right, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=xmin, axmax=xmax)
    def set_ylim(self, bottom=None, top=None, *, emit=True, auto=False,
                 view_margin=None, ymin=None, ymax=None):
        """
        Set the 3D y-axis view limits.

        Parameters
        ----------
        bottom : float, optional
            The bottom ylim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The bottom and top ylims may also be passed as the tuple
            (*bottom*, *top*) as the first positional argument (or as
            the *bottom* keyword argument).

            .. ACCEPTS: (bottom: float, top: float)

        top : float, optional
            The top ylim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the y-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        view_margin : float, optional
            The additional margin to apply to the limits.

        ymin, ymax : float, optional
            They are equivalent to bottom and top respectively, and it is an
            error to pass both *ymin* and *bottom* or *ymax* and *top*.

        Returns
        -------
        bottom, top : (float, float)
            The new y-axis limits in data coordinates.

        See Also
        --------
        get_ylim
        set_ybound, get_ybound
        invert_yaxis, yaxis_inverted

        Notes
        -----
        The *bottom* value may be greater than the *top* value, in which
        case the y-axis values will decrease from *bottom* to *top*.

        Examples
        --------
        >>> set_ylim(bottom, top)
        >>> set_ylim((bottom, top))
        >>> bottom, top = set_ylim(bottom, top)

        One limit may be left unchanged.

        >>> set_ylim(top=top_lim)

        Limits may be passed in reverse order to flip the direction of
        the y-axis. For example, suppose ``y`` represents depth of the
        ocean in m. The y-axis limits might be set like the following
        so 5000 m depth is at the bottom of the plot and the surface,
        0 m, is at the top.

        >>> set_ylim(5000, 0)
        """
        # 调用内部方法 _set_lim3d 设置 3D y 轴的限制
        return self._set_lim3d(self.yaxis, bottom, top, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=ymin, axmax=ymax)
    def set_zlim(self, bottom=None, top=None, *, emit=True, auto=False,
                 view_margin=None, zmin=None, zmax=None):
        """
        Set the 3D z-axis view limits.

        Parameters
        ----------
        bottom : float, optional
            The bottom zlim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The bottom and top zlims may also be passed as the tuple
            (*bottom*, *top*) as the first positional argument (or as
            the *bottom* keyword argument).

            .. ACCEPTS: (bottom: float, top: float)

        top : float, optional
            The top zlim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the z-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        view_margin : float, optional
            The additional margin to apply to the limits.

        zmin, zmax : float, optional
            They are equivalent to bottom and top respectively, and it is an
            error to pass both *zmin* and *bottom* or *zmax* and *top*.

        Returns
        -------
        bottom, top : (float, float)
            The new z-axis limits in data coordinates.

        See Also
        --------
        get_zlim
        set_zbound, get_zbound
        invert_zaxis, zaxis_inverted

        Notes
        -----
        The *bottom* value may be greater than the *top* value, in which
        case the z-axis values will decrease from *bottom* to *top*.

        Examples
        --------
        >>> set_zlim(bottom, top)
        >>> set_zlim((bottom, top))
        >>> bottom, top = set_zlim(bottom, top)

        One limit may be left unchanged.

        >>> set_zlim(top=top_lim)

        Limits may be passed in reverse order to flip the direction of
        the z-axis. For example, suppose ``z`` represents depth of the
        ocean in m. The z-axis limits might be set like the following
        so 5000 m depth is at the bottom of the plot and the surface,
        0 m, is at the top.

        >>> set_zlim(5000, 0)
        """
        # 调用内部方法 _set_lim3d 来设置 z 轴的限制
        return self._set_lim3d(self.zaxis, bottom, top, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=zmin, axmax=zmax)

    # 将 set_xlim 与 set_ylim 方法别名为 set_xlim3d 与 set_ylim3d
    set_xlim3d = set_xlim
    set_ylim3d = set_ylim
    set_zlim3d = set_zlim

    def get_xlim(self):
        # 继承文档字符串，返回 x 轴的视图限制区间的元组
        return tuple(self.xy_viewLim.intervalx)

    def get_ylim(self):
        # 继承文档字符串，返回 y 轴的视图限制区间的元组
        return tuple(self.xy_viewLim.intervaly)
    def get_zlim(self):
        """
        Return the 3D z-axis view limits.

        Returns
        -------
        left, right : (float, float)
            The current z-axis limits in data coordinates.

        See Also
        --------
        set_zlim
        set_zbound, get_zbound
        invert_zaxis, zaxis_inverted

        Notes
        -----
        The z-axis may be inverted, in which case the *left* value will
        be greater than the *right* value.
        """
        return tuple(self.zz_viewLim.intervalx)



    get_zscale = _axis_method_wrapper("zaxis", "get_scale")



    # Redefine all three methods to overwrite their docstrings.
    set_xscale = _axis_method_wrapper("xaxis", "_set_axes_scale")
    set_yscale = _axis_method_wrapper("yaxis", "_set_axes_scale")
    set_zscale = _axis_method_wrapper("zaxis", "_set_axes_scale")
    set_xscale.__doc__, set_yscale.__doc__, set_zscale.__doc__ = map(
        """
        Set the {}-axis scale.

        Parameters
        ----------
        value : {{"linear"}}
            The axis scale type to apply.  3D Axes currently only support
            linear scales; other scales yield nonsensical results.

        **kwargs
            Keyword arguments are nominally forwarded to the scale class, but
            none of them is applicable for linear scales.
        """.format,
        ["x", "y", "z"])



    get_zticks = _axis_method_wrapper("zaxis", "get_ticklocs")
    set_zticks = _axis_method_wrapper("zaxis", "set_ticks")
    get_zmajorticklabels = _axis_method_wrapper("zaxis", "get_majorticklabels")
    get_zminorticklabels = _axis_method_wrapper("zaxis", "get_minorticklabels")
    get_zticklabels = _axis_method_wrapper("zaxis", "get_ticklabels")
    set_zticklabels = _axis_method_wrapper(
        "zaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes3D.set_zticks"})



    zaxis_date = _axis_method_wrapper("zaxis", "axis_date")
    if zaxis_date.__doc__:
        zaxis_date.__doc__ += textwrap.dedent("""

        Notes
        -----
        This function is merely provided for completeness, but 3D Axes do not
        support dates for ticks, and so this may not work as expected.
        """)



    def clabel(self, *args, **kwargs):
        """Currently not implemented for 3D Axes, and returns *None*."""
        return None
    # 设置投影类型的方法，可以是透视或正交投影
    def set_proj_type(self, proj_type, focal_length=None):
        """
        Set the projection type.

        Parameters
        ----------
        proj_type : {'persp', 'ortho'}
            The projection type.
        focal_length : float, default: None
            For a projection type of 'persp', the focal length of the virtual
            camera. Must be > 0. If None, defaults to 1.
            The focal length can be computed from a desired Field Of View via
            the equation: focal_length = 1/tan(FOV/2)
        """
        # 检查投影类型是否在支持的列表中
        _api.check_in_list(['persp', 'ortho'], proj_type=proj_type)
        # 如果是透视投影
        if proj_type == 'persp':
            # 如果未指定焦距，默认为1
            if focal_length is None:
                focal_length = 1
            # 如果指定的焦距不合法（小于等于0），抛出异常
            elif focal_length <= 0:
                raise ValueError(f"focal_length = {focal_length} must be "
                                 "greater than 0")
            # 将有效的焦距值设置到对象的内部变量中
            self._focal_length = focal_length
        else:  # 如果是正交投影
            # 正交投影的焦距必须为None或无穷大
            if focal_length not in (None, np.inf):
                raise ValueError(f"focal_length = {focal_length} must be "
                                 f"None for proj_type = {proj_type}")
            # 将正交投影的特殊焦距值设置到对象的内部变量中
            self._focal_length = np.inf

    # 将数组滚动以匹配不同的垂直轴
    def _roll_to_vertical(
        self, arr: "np.typing.ArrayLike", reverse: bool = False
    ) -> np.ndarray:
        """
        Roll arrays to match the different vertical axis.

        Parameters
        ----------
        arr : ArrayLike
            Array to roll.
        reverse : bool, default: False
            Reverse the direction of the roll.
        """
        # 如果需要反向滚动数组
        if reverse:
            # 根据当前对象的垂直轴调整滚动方向并返回结果数组
            return np.roll(arr, (self._vertical_axis - 2) * -1)
        else:
            # 根据当前对象的垂直轴进行滚动并返回结果数组
            return np.roll(arr, (self._vertical_axis - 2))
    def get_proj(self):
        """从当前视角位置创建投影矩阵。"""

        # 将视角转换为统一的世界坐标系 0-1, 0-1, 0-1
        box_aspect = self._roll_to_vertical(self._box_aspect)
        # 计算世界变换矩阵，将当前的 x, y, z 轴限制应用到世界变换
        worldM = proj3d.world_transformation(
            *self.get_xlim3d(),
            *self.get_ylim3d(),
            *self.get_zlim3d(),
            pb_aspect=box_aspect,
        )

        # 看向世界坐标系的中心点：
        R = 0.5 * box_aspect

        # elev: z平面的仰角。
        # azim: xy平面的方位角。
        # 为围绕数据框旋转的点计算坐标。
        # p0, p1 对应仅围绕垂直轴旋转数据框。
        # p2 对应仅围绕水平轴旋转数据框。
        elev_rad = np.deg2rad(self.elev)
        azim_rad = np.deg2rad(self.azim)
        p0 = np.cos(elev_rad) * np.cos(azim_rad)
        p1 = np.cos(elev_rad) * np.sin(azim_rad)
        p2 = np.sin(elev_rad)

        # 当改变垂直轴时，坐标也随之变化。
        # 将值滚动以获得与默认行为相同的行为：
        ps = self._roll_to_vertical([p0, p1, p2])

        # 眼睛观察点的坐标。眼睛从一定距离看向数据框的中心：
        eye = R + self._dist * ps

        # 计算眼睛位置的观察轴
        u, v, w = self._calc_view_axes(eye)
        self._view_u = u  # _view_u 朝屏幕右侧
        self._view_v = v  # _view_v 朝屏幕顶部
        self._view_w = w  # _view_w 指向屏幕外侧

        # 生成视图和投影变换矩阵
        if self._focal_length == np.inf:
            # 正交投影
            viewM = proj3d._view_transformation_uvw(u, v, w, eye)
            projM = proj3d._ortho_transformation(-self._dist, self._dist)
        else:
            # 透视投影
            # 缩放眼睛距离以补偿焦距变焦效应
            eye_focal = R + self._dist * ps * self._focal_length
            viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
            projM = proj3d._persp_transformation(-self._dist,
                                                 self._dist,
                                                 self._focal_length)

        # 合并所有变换矩阵以获得最终投影
        M0 = np.dot(viewM, worldM)
        M = np.dot(projM, M0)
        return M
    def mouse_init(self, rotate_btn=1, pan_btn=2, zoom_btn=3):
        """
        Set the mouse buttons for 3D rotation and zooming.

        Parameters
        ----------
        rotate_btn : int or list of int, default: 1
            The mouse button or buttons to use for 3D rotation of the Axes.
        pan_btn : int or list of int, default: 2
            The mouse button or buttons to use to pan the 3D Axes.
        zoom_btn : int or list of int, default: 3
            The mouse button or buttons to use to zoom the 3D Axes.
        """
        # 初始化鼠标按钮状态为 None
        self.button_pressed = None
        # 将输入的旋转按钮转化为至少包含一个元素的列表，以避免与 None 比较的问题
        self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
        # 将输入的平移按钮转化为至少包含一个元素的列表，以避免与 None 比较的问题
        self._pan_btn = np.atleast_1d(pan_btn).tolist()
        # 将输入的缩放按钮转化为至少包含一个元素的列表，以避免与 None 比较的问题
        self._zoom_btn = np.atleast_1d(zoom_btn).tolist()

    def disable_mouse_rotation(self):
        """Disable mouse buttons for 3D rotation, panning, and zooming."""
        # 禁用鼠标按钮以防止3D旋转、平移和缩放
        self.mouse_init(rotate_btn=[], pan_btn=[], zoom_btn=[])

    def can_zoom(self):
        # doc-string inherited
        # 可以缩放，始终返回 True
        return True

    def can_pan(self):
        # doc-string inherited
        # 可以平移，始终返回 True
        return True

    def sharez(self, other):
        """
        Share the z-axis with *other*.

        This is equivalent to passing ``sharez=other`` when constructing the
        Axes, and cannot be used if the z-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
        # 检查 other 是否为 Axes3D 类型的对象
        _api.check_isinstance(Axes3D, other=other)
        # 如果已经与其他 Axes 共享 z 轴，则引发 ValueError
        if self._sharez is not None and other is not self._sharez:
            raise ValueError("z-axis is already shared")
        # 将自身与 other 加入共享 z 轴的 Axes 列表
        self._shared_axes["z"].join(self, other)
        # 记录当前与之共享 z 轴的 Axes
        self._sharez = other
        # 设置 z 轴的刻度及相关参数与 other 保持一致
        self.zaxis.major = other.zaxis.major  # 包含刻度和格式化信息的 Ticker 实例
        self.zaxis.minor = other.zaxis.minor  # 包含刻度和格式化信息的 Ticker 实例
        # 获取 other 的 z 轴限制并设置自身的 z 轴限制
        z0, z1 = other.get_zlim()
        self.set_zlim(z0, z1, emit=False, auto=other.get_autoscalez_on())
        # 设置自身 z 轴的缩放参数与 other 一致
        self.zaxis._scale = other.zaxis._scale

    def shareview(self, other):
        """
        Share the view angles with *other*.

        This is equivalent to passing ``shareview=other`` when constructing the
        Axes, and cannot be used if the view angles are already being shared
        with another Axes.  Note that it is not possible to unshare axes.
        """
        # 检查 other 是否为 Axes3D 类型的对象
        _api.check_isinstance(Axes3D, other=other)
        # 如果已经与其他 Axes 共享视角，则引发 ValueError
        if self._shareview is not None and other is not self._shareview:
            raise ValueError("view angles are already shared")
        # 将自身与 other 加入共享视角的 Axes 列表
        self._shared_axes["view"].join(self, other)
        # 记录当前与之共享视角的 Axes
        self._shareview = other
        # 根据 other 的垂直轴信息设置自身的视角
        vertical_axis = self._axis_names[other._vertical_axis]
        self.view_init(elev=other.elev, azim=other.azim, roll=other.roll,
                       vertical_axis=vertical_axis, share=True)
    # 继承的文档字符串
    def clear(self):
        # 调用父类的 clear 方法
        super().clear()
        # 如果焦距为无穷大，则使用默认的 z 边距；否则设置为 0
        if self._focal_length == np.inf:
            self._zmargin = mpl.rcParams['axes.zmargin']
        else:
            self._zmargin = 0.

        # 设置 xy 数据的限制框，以匹配 mpl3.8 的外观
        xymargin = 0.05 * 10/11
        self.xy_dataLim = Bbox([[xymargin, xymargin],
                                [1 - xymargin, 1 - xymargin]])
        # z 限制编码在 Bbox 的 x 组件中，y 组件未使用
        self.zz_dataLim = Bbox.unit()
        # 设置视图边距，默认为匹配 mpl3.8 的值
        self._view_margin = 1/48
        # 自动调整视图
        self.autoscale_view()

        # 根据 mpl.rcParams 的设置绘制网格
        self.grid(mpl.rcParams['axes3d.grid'])

    def _button_press(self, event):
        # 如果事件发生在当前对象的坐标轴内
        if event.inaxes == self:
            # 记录按下的按钮和坐标位置
            self.button_pressed = event.button
            self._sx, self._sy = event.xdata, event.ydata
            # 获取当前图形的工具栏
            toolbar = self.figure.canvas.toolbar
            # 如果工具栏存在且导航栈为空，则推送当前状态
            if toolbar and toolbar._nav_stack() is None:
                toolbar.push_current()
            # 如果工具栏存在，则设置消息为鼠标事件的信息
            if toolbar:
                toolbar.set_message(toolbar._mouse_event_to_message(event))

    def _button_release(self, event):
        # 释放按钮后重置按钮状态为 None
        self.button_pressed = None
        # 获取当前图形的工具栏
        toolbar = self.figure.canvas.toolbar
        # 检查导航模式，避免重复调用 push_current
        if toolbar and self.get_navigate_mode() is None:
            toolbar.push_current()
        # 如果工具栏存在，则设置消息为鼠标事件的信息
        if toolbar:
            toolbar.set_message(toolbar._mouse_event_to_message(event))

    def _get_view(self):
        # 继承的文档字符串
        # 返回当前视图的设置，包括 x, y, z 轴的限制和自动缩放状态
        return {
            "xlim": self.get_xlim(), "autoscalex_on": self.get_autoscalex_on(),
            "ylim": self.get_ylim(), "autoscaley_on": self.get_autoscaley_on(),
            "zlim": self.get_zlim(), "autoscalez_on": self.get_autoscalez_on(),
        }, (self.elev, self.azim, self.roll)

    def _set_view(self, view):
        # 继承的文档字符串
        # 设置视图的属性，包括 x, y, z 轴的限制和自动缩放状态
        props, (elev, azim, roll) = view
        self.set(**props)
        self.elev = elev
        self.azim = azim
        self.roll = roll

    def format_zdata(self, z):
        """
        Return *z* string formatted.  This function will use the
        :attr:`fmt_zdata` attribute if it is callable, else will fall
        back on the zaxis major formatter
        """
        # 格式化 z 数据的字符串表示
        try:
            return self.fmt_zdata(z)
        except (AttributeError, TypeError):
            # 如果 fmt_zdata 不可调用，则使用 z 轴的主要格式化器进行格式化
            func = self.zaxis.get_major_formatter().format_data_short
            val = func(z)
            return val
    def format_coord(self, xv, yv, renderer=None):
        """
        Return a string giving the current view rotation angles, or the x, y, z
        coordinates of the point on the nearest axis pane underneath the mouse
        cursor, depending on the mouse button pressed.
        """
        coords = ''  # 初始化一个空字符串用于存储坐标信息

        if self.button_pressed in self._rotate_btn:
            # 如果鼠标按下的按钮在旋转按钮集合中，则返回旋转角度信息
            coords = self._rotation_coords()

        elif self.M is not None:
            # 如果存在 M 属性，则返回鼠标下最近的坐标轴平面上的点的坐标信息
            coords = self._location_coords(xv, yv, renderer)

        return coords  # 返回坐标信息字符串

    def _rotation_coords(self):
        """
        Return the rotation angles as a string.
        """
        # 计算并返回旋转角度信息的字符串表示
        norm_elev = art3d._norm_angle(self.elev)
        norm_azim = art3d._norm_angle(self.azim)
        norm_roll = art3d._norm_angle(self.roll)
        coords = (f"elevation={norm_elev:.0f}\N{DEGREE SIGN}, "
                  f"azimuth={norm_azim:.0f}\N{DEGREE SIGN}, "
                  f"roll={norm_roll:.0f}\N{DEGREE SIGN}"
                  ).replace("-", "\N{MINUS SIGN}")
        return coords

    def _location_coords(self, xv, yv, renderer):
        """
        Return the location on the axis pane underneath the cursor as a string.
        """
        # 计算并返回鼠标下坐标轴平面上点的坐标信息字符串
        p1, pane_idx = self._calc_coord(xv, yv, renderer)
        xs = self.format_xdata(p1[0])
        ys = self.format_ydata(p1[1])
        zs = self.format_zdata(p1[2])
        if pane_idx == 0:
            coords = f'x pane={xs}, y={ys}, z={zs}'
        elif pane_idx == 1:
            coords = f'x={xs}, y pane={ys}, z={zs}'
        elif pane_idx == 2:
            coords = f'x={xs}, y={ys}, z pane={zs}'
        return coords  # 返回坐标信息字符串

    def _get_camera_loc(self):
        """
        Returns the current camera location in data coordinates.
        """
        # 计算并返回当前相机位置的数据坐标
        cx, cy, cz, dx, dy, dz = self._get_w_centers_ranges()
        c = np.array([cx, cy, cz])
        r = np.array([dx, dy, dz])

        if self._focal_length == np.inf:  # 如果是正交投影
            focal_length = 1e9  # 设定一个足够大的值表示无限远
        else:  # 透视投影
            focal_length = self._focal_length
        eye = c + self._view_w * self._dist * r / self._box_aspect * focal_length
        return eye  # 返回相机位置的数据坐标
    def _calc_coord(self, xv, yv, renderer=None):
        """
        Given the 2D view coordinates, find the point on the nearest axis pane
        that lies directly below those coordinates. Returns a 3D point in data
        coordinates.
        """
        if self._focal_length == np.inf:  # 如果焦距为无穷大，则为正交投影
            zv = 1
        else:  # 如果焦距不为无穷大，则为透视投影
            zv = -1 / self._focal_length

        # 将视图平面上的点转换为数据坐标
        p1 = np.array(proj3d.inv_transform(xv, yv, zv, self.invM)).ravel()

        # 获取从相机到视图平面上点的向量
        vec = self._get_camera_loc() - p1

        # 获取每个轴的平面位置
        pane_locs = []
        for axis in self._axis_map.values():
            xys, loc = axis.active_pane()
            pane_locs.append(loc)

        # 通过投影视图向量来找到最近平面的距离
        scales = np.zeros(3)
        for i in range(3):
            if vec[i] == 0:
                scales[i] = np.inf
            else:
                scales[i] = (p1[i] - pane_locs[i]) / vec[i]
        pane_idx = np.argmin(abs(scales))
        scale = scales[pane_idx]

        # 计算最近平面上的点
        p2 = p1 - scale * vec
        return p2, pane_idx

    def _arcball(self, x: float, y: float) -> np.ndarray:
        """
        Convert a point (x, y) to a point on a virtual trackball
        This is Ken Shoemake's arcball
        See: Ken Shoemake, "ARCBALL: A user interface for specifying
        three-dimensional rotation using a mouse." in
        Proceedings of Graphics Interface '92, 1992, pp. 151-156,
        https://doi.org/10.20380/GI1992.18
        """
        x *= 2
        y *= 2
        r2 = x * x + y * y
        if r2 > 1:
            p = np.array([0, x / math.sqrt(r2), y / math.sqrt(r2)])
        else:
            p = np.array([math.sqrt(1 - r2), x, y])
        return p
    def _on_move(self, event):
        """
        Mouse moving.

        By default, button-1 rotates, button-2 pans, and button-3 zooms;
        these buttons can be modified via `mouse_init`.
        """

        # 如果没有按钮按下，退出函数
        if not self.button_pressed:
            return

        # 如果当前处于导航模式，不执行旋转操作
        if self.get_navigate_mode() is not None:
            # 来自工具栏的缩放/平移操作，不执行旋转
            return

        # 如果当前矩阵为空，退出函数
        if self.M is None:
            return

        # 获取鼠标当前的数据点坐标
        x, y = event.xdata, event.ydata
        # 如果鼠标坐标为None或者不在当前Axes内，退出函数
        if x is None or event.inaxes != self:
            return

        # 计算鼠标移动的增量
        dx, dy = x - self._sx, y - self._sy
        w = self._pseudo_w
        h = self._pseudo_h

        # 旋转视角
        if self.button_pressed in self._rotate_btn:
            # 旋转视点
            # 获取x和y像素坐标
            if dx == 0 and dy == 0:
                return

            # 将角度转换为四元数
            elev = np.deg2rad(self.elev)
            azim = np.deg2rad(self.azim)
            roll = np.deg2rad(self.roll)
            q = _Quaternion.from_cardan_angles(elev, azim, roll)

            # 更新四元数 - 基于Ken Shoemake的ARCBALL的一种变体
            current_vec = self._arcball(self._sx/w, self._sy/h)
            new_vec = self._arcball(x/w, y/h)
            dq = _Quaternion.rotate_from_to(current_vec, new_vec)
            q = dq * q

            # 将四元数转换为角度
            elev, azim, roll = q.as_cardan_angles()
            azim = np.rad2deg(azim)
            elev = np.rad2deg(elev)
            roll = np.rad2deg(roll)
            vertical_axis = self._axis_names[self._vertical_axis]
            # 更新视角设置
            self.view_init(
                elev=elev,
                azim=azim,
                roll=roll,
                vertical_axis=vertical_axis,
                share=True,
            )
            # 设置为需要重新绘制
            self.stale = True

        # 平移
        elif self.button_pressed in self._pan_btn:
            # 使用像素坐标开始平移事件
            px, py = self.transData.transform([self._sx, self._sy])
            self.start_pan(px, py, 2)
            # 执行平移视图（输入像素坐标）
            self.drag_pan(2, None, event.x, event.y)
            self.end_pan()

        # 缩放
        elif self.button_pressed in self._zoom_btn:
            # 缩放视图（向下拖动缩小）
            scale = h/(h - dy)
            self._scale_axis_limits(scale, scale, scale)

        # 存储事件坐标以备下次使用
        self._sx, self._sy = x, y
        # 总是在交互结束时请求重绘
        self.figure.canvas.draw_idle()
    def drag_pan(self, button, key, x, y):
        """
        Perform panning operation based on mouse drag event.

        Parameters:
        - button : int
            Mouse button identifier.
        - key : str
            Key identifier indicating axis ('x', 'y') for constrained pan.
        - x, y : float
            Coordinates of the mouse cursor during the drag event.
        """

        # Get the coordinates from the move event
        p = self._pan_start
        (xdata, ydata), (xdata_start, ydata_start) = p.trans_inverse.transform(
            [(x, y), (p.x, p.y)])
        self._sx, self._sy = xdata, ydata

        # Calling start_pan() to set the x/y of this event as the starting
        # move location for the next event
        self.start_pan(x, y, button)

        du, dv = xdata - xdata_start, ydata - ydata_start
        dw = 0
        if key == 'x':
            dv = 0
        elif key == 'y':
            du = 0

        if du == 0 and dv == 0:
            return

        # Transform the pan from the view axes to the data axes
        R = np.array([self._view_u, self._view_v, self._view_w])
        R = -R / self._box_aspect * self._dist
        duvw_projected = R.T @ np.array([du, dv, dw])

        # Calculate pan distance
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
        dx = (maxx - minx) * duvw_projected[0]
        dy = (maxy - miny) * duvw_projected[1]
        dz = (maxz - minz) * duvw_projected[2]

        # Set the new axis limits
        self.set_xlim3d(minx + dx, maxx + dx, auto=None)
        self.set_ylim3d(miny + dy, maxy + dy, auto=None)
        self.set_zlim3d(minz + dz, maxz + dz, auto=None)

    def _calc_view_axes(self, eye):
        """
        Calculate the unit vectors for the viewing axes in data coordinates.

        Parameters:
        - eye : array-like
            Position of the viewer's eye.

        Returns:
        - u, v, w : array-like
            Unit vectors representing the viewing axes in data coordinates.
            `u` is towards the right of the screen
            `v` is towards the top of the screen
            `w` is out of the screen
        """
        elev_rad = np.deg2rad(art3d._norm_angle(self.elev))
        roll_rad = np.deg2rad(art3d._norm_angle(self.roll))

        # Look into the middle of the world coordinates
        R = 0.5 * self._roll_to_vertical(self._box_aspect)

        # Define which axis should be vertical. A negative value
        # indicates the plot is upside down and therefore the values
        # have been reversed:
        V = np.zeros(3)
        V[self._vertical_axis] = -1 if abs(elev_rad) > np.pi/2 else 1

        u, v, w = proj3d._view_axes(eye, R, V, roll_rad)
        return u, v, w
    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        """
        Zoom in or out of the bounding box.

        Will center the view in the center of the bounding box, and zoom by
        the ratio of the size of the bounding box to the size of the Axes3D.
        """
        # 解构边界框的坐标信息
        (start_x, start_y, stop_x, stop_y) = bbox
        # 根据指定的模式调整边界框的起始和停止位置
        if mode == 'x':
            start_y = self.bbox.min[1]
            stop_y = self.bbox.max[1]
        elif mode == 'y':
            start_x = self.bbox.min[0]
            stop_x = self.bbox.max[0]

        # 将起始和停止位置限制在边界框的范围内
        start_x, stop_x = np.clip(sorted([start_x, stop_x]),
                                  self.bbox.min[0], self.bbox.max[0])
        start_y, stop_y = np.clip(sorted([start_y, stop_y]),
                                  self.bbox.min[1], self.bbox.max[1])

        # 将视图的中心移动到边界框的中心
        zoom_center_x = (start_x + stop_x)/2
        zoom_center_y = (start_y + stop_y)/2

        # 计算当前坐标轴的中心
        ax_center_x = (self.bbox.max[0] + self.bbox.min[0])/2
        ax_center_y = (self.bbox.max[1] + self.bbox.min[1])/2

        # 开始平移视图到指定的中心点
        self.start_pan(zoom_center_x, zoom_center_y, 2)
        # 拖拽平移视图到指定的中心点
        self.drag_pan(2, None, ax_center_x, ax_center_y)
        # 结束视图的平移操作
        self.end_pan()

        # 计算缩放级别
        dx = abs(start_x - stop_x)
        dy = abs(start_y - stop_y)
        scale_u = dx / (self.bbox.max[0] - self.bbox.min[0])
        scale_v = dy / (self.bbox.max[1] - self.bbox.min[1])

        # 保持宽高比相等
        scale = max(scale_u, scale_v)

        # 根据指定的方向进行缩放
        if direction == 'out':
            scale = 1 / scale

        # 应用缩放到数据的限制范围
        self._zoom_data_limits(scale, scale, scale)
    def _zoom_data_limits(self, scale_u, scale_v, scale_w):
        """
        Zoom in or out of a 3D plot.

        Will scale the data limits by the scale factors. These will be
        transformed to the x, y, z data axes based on the current view angles.
        A scale factor > 1 zooms out and a scale factor < 1 zooms in.

        For an Axes that has had its aspect ratio set to 'equal', 'equalxy',
        'equalyz', or 'equalxz', the relevant axes are constrained to zoom
        equally.

        Parameters
        ----------
        scale_u : float
            Scale factor for the u view axis (view screen horizontal).
        scale_v : float
            Scale factor for the v view axis (view screen vertical).
        scale_w : float
            Scale factor for the w view axis (view screen depth).
        """
        scale = np.array([scale_u, scale_v, scale_w])

        # Only perform frame conversion if unequal scale factors
        if not np.allclose(scale, scale_u):
            # Convert the scale factors from the view frame to the data frame
            R = np.array([self._view_u, self._view_v, self._view_w])
            S = scale * np.eye(3)
            scale = np.linalg.norm(R.T @ S, axis=1)

            # Set the constrained scale factors to the factor closest to 1
            if self._aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
                ax_idxs = self._equal_aspect_axis_indices(self._aspect)
                min_ax_idxs = np.argmin(np.abs(scale[ax_idxs] - 1))
                scale[ax_idxs] = scale[ax_idxs][min_ax_idxs]

        self._scale_axis_limits(scale[0], scale[1], scale[2])

    def _scale_axis_limits(self, scale_x, scale_y, scale_z):
        """
        Keeping the center of the x, y, and z data axes fixed, scale their
        limits by scale factors. A scale factor > 1 zooms out and a scale
        factor < 1 zooms in.

        Parameters
        ----------
        scale_x : float
            Scale factor for the x data axis.
        scale_y : float
            Scale factor for the y data axis.
        scale_z : float
            Scale factor for the z data axis.
        """
        # Get the axis centers and ranges
        cx, cy, cz, dx, dy, dz = self._get_w_centers_ranges()

        # Set the scaled axis limits
        self.set_xlim3d(cx - dx*scale_x/2, cx + dx*scale_x/2, auto=None)
        self.set_ylim3d(cy - dy*scale_y/2, cy + dy*scale_y/2, auto=None)
        self.set_zlim3d(cz - dz*scale_z/2, cz + dz*scale_z/2, auto=None)

    def _get_w_centers_ranges(self):
        """Get 3D world centers and axis ranges."""
        # Calculate center of axis limits
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
        cx = (maxx + minx)/2
        cy = (maxy + miny)/2
        cz = (maxz + minz)/2

        # Calculate range of axis limits
        dx = (maxx - minx)
        dy = (maxy - miny)
        dz = (maxz - minz)
        return cx, cy, cz, dx, dy, dz
    def set_zlabel(self, zlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set zlabel.  See doc for `.set_ylabel` for description.
        """
        if labelpad is not None:
            # 如果提供了labelpad参数，则设置z轴标签的内边距
            self.zaxis.labelpad = labelpad
        # 调用z轴对象的set_label_text方法设置z轴标签文本，并返回结果
        return self.zaxis.set_label_text(zlabel, fontdict, **kwargs)

    def get_zlabel(self):
        """
        Get the z-label text string.
        """
        # 获取z轴标签对象，并返回其文本内容
        label = self.zaxis.get_label()
        return label.get_text()

    # Axes rectangle characteristics

    # The frame_on methods are not available for 3D axes.
    # Python will raise a TypeError if they are called.
    # 对于3D坐标轴，frame_on方法不可用，如果调用将会引发TypeError异常
    get_frame_on = None
    set_frame_on = None

    def grid(self, visible=True, **kwargs):
        """
        Set / unset 3D grid.

        .. note::

            Currently, this function does not behave the same as
            `.axes.Axes.grid`, but it is intended to eventually support that
            behavior.
        """
        # TODO: Operate on each axes separately
        # 如果kwargs中有参数，强制设置visible为True
        if len(kwargs):
            visible = True
        # 设置_draw_grid属性为visible，并标记为需要更新
        self._draw_grid = visible
        self.stale = True

    def tick_params(self, axis='both', **kwargs):
        """
        Convenience method for changing the appearance of ticks and
        tick labels.

        See `.Axes.tick_params` for full documentation.  Because this function
        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
        to 'both' autoscales all three axes.

        Also, because of how Axes3D objects are drawn very differently
        from regular 2D Axes, some of these settings may have
        ambiguous meaning.  For simplicity, the 'z' axis will
        accept settings as if it was like the 'y' axis.

        .. note::
           Axes3D currently ignores some of these settings.
        """
        # 检查axis参数的合法性，只允许'x', 'y', 'z', 'both'
        _api.check_in_list(['x', 'y', 'z', 'both'], axis=axis)
        if axis in ['x', 'y', 'both']:
            # 对'x', 'y'或者'both'轴应用tick_params方法的设置
            super().tick_params(axis, **kwargs)
        if axis in ['z', 'both']:
            # 对'z'轴应用tick_params方法的设置，同时清除不适用于z轴的特定参数
            zkw = dict(kwargs)
            zkw.pop('top', None)
            zkw.pop('bottom', None)
            zkw.pop('labeltop', None)
            zkw.pop('labelbottom', None)
            self.zaxis.set_tick_params(**zkw)

    # data limits, ticks, tick labels, and formatting

    def invert_zaxis(self):
        """
        Invert the z-axis.

        See Also
        --------
        zaxis_inverted
        get_zlim, set_zlim
        get_zbound, set_zbound
        """
        # 获取当前z轴的上下限，并反转z轴
        bottom, top = self.get_zlim()
        self.set_zlim(top, bottom, auto=None)

    # 将指定方法封装成属性的方法
    zaxis_inverted = _axis_method_wrapper("zaxis", "get_inverted")
    def get_zbound(self):
        """
        Return the lower and upper z-axis bounds, in increasing order.

        See Also
        --------
        set_zbound
        get_zlim, set_zlim
        invert_zaxis, zaxis_inverted
        """
        # 调用 get_zlim 方法获取当前图形的 z 轴界限
        lower, upper = self.get_zlim()
        # 检查界限的顺序并确保返回的是从小到大的顺序
        if lower < upper:
            return lower, upper
        else:
            return upper, lower

    def text(self, x, y, z, s, zdir=None, **kwargs):
        """
        Add the text *s* to the 3D Axes at location *x*, *y*, *z* in data coordinates.

        Parameters
        ----------
        x, y, z : float
            The position to place the text.
        s : str
            The text.
        zdir : {'x', 'y', 'z', 3-tuple}, optional
            The direction to be used as the z-direction. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.text`.

        Returns
        -------
        `.Text3D`
            The created `.Text3D` instance.
        """
        # 调用基类 Axes 的 text 方法添加文本到 3D 坐标轴
        text = super().text(x, y, s, **kwargs)
        # 将二维文本转换为三维文本，并设置其位置和方向
        art3d.text_2d_to_3d(text, z, zdir)
        return text

    text3D = text
    text2D = Axes.text

    def plot(self, xs, ys, *args, zdir='z', **kwargs):
        """
        Plot 2D or 3D data.

        Parameters
        ----------
        xs : 1D array-like
            x coordinates of vertices.
        ys : 1D array-like
            y coordinates of vertices.
        zs : float or 1D array-like
            z coordinates of vertices; either one for all points or one for
            each point.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z.
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.plot`.
        """
        # 检查是否已经存在数据
        had_data = self.has_data()

        # 处理是否有传入 zs 参数，并且根据参数类型处理 xs, ys, zs
        if args and not isinstance(args[0], str):
            zs, *args = args
            # 检查是否同时传入了 'zs' 参数
            if 'zs' in kwargs:
                raise TypeError("plot() for multiple values for argument 'zs'")
        else:
            zs = kwargs.pop('zs', 0)

        # 使用 cbook._broadcast_with_masks 处理 xs, ys, zs，保证它们对齐
        xs, ys, zs = cbook._broadcast_with_masks(xs, ys, zs)

        # 调用基类 Axes 的 plot 方法进行绘图
        lines = super().plot(xs, ys, *args, **kwargs)

        # 将二维线条转换为三维线条，设置其 z 值和方向
        for line in lines:
            art3d.line_2d_to_3d(line, zs=zs, zdir=zdir)

        # 调整坐标轴的范围以适应新数据
        xs, ys, zs = art3d.juggle_axes(xs, ys, zs, zdir)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return lines

    plot3D = plot
    def _3d_extend_contour(self, cset, stride=5):
        """
        在三维空间中扩展轮廓，通过创建
        """

        # 计算轮廓间的高度差，用于确定顶部和底部的位置
        dz = (cset.levels[1] - cset.levels[0]) / 2
        # 存储多边形顶点的列表
        polyverts = []
        # 存储颜色的列表
        colors = []
        # 遍历轮廓集合中的每一个轮廓
        for idx, level in enumerate(cset.levels):
            # 获取当前轮廓的路径
            path = cset.get_paths()[idx]
            # 将路径分解成连接的子路径
            subpaths = [*path._iter_connected_components()]
            # 获取当前轮廓的边缘颜色
            color = cset.get_edgecolor()[idx]
            # 将当前轮廓的顶部和底部转换为三维线段
            top = art3d._paths_to_3d_segments(subpaths, level - dz)
            bot = art3d._paths_to_3d_segments(subpaths, level + dz)
            # 如果顶部没有内容，则跳过当前轮廓
            if not len(top[0]):
                continue
            # 确定用于插值的步数
            nsteps = max(round(len(top[0]) / stride), 2)
            stepsize = (len(top[0]) - 1) / (nsteps - 1)
            # 将顶部和底部的多边形顶点扩展到polyverts中
            polyverts.extend([
                (top[0][round(i * stepsize)], top[0][round((i + 1) * stepsize)],
                 bot[0][round((i + 1) * stepsize)], bot[0][round(i * stepsize)])
                for i in range(round(nsteps) - 1)])
            # 将当前轮廓的颜色添加到colors列表中
            colors.extend([color] * (round(nsteps) - 1))
        # 将生成的三维多边形集合添加到当前对象中
        self.add_collection3d(art3d.Poly3DCollection(
            np.array(polyverts),  # 所有多边形都有4个顶点，因此向量化处理。
            facecolors=colors, edgecolors=colors, shade=True))
        # 从轮廓集合中移除当前轮廓
        cset.remove()

    def add_contour_set(
            self, cset, extend3d=False, stride=5, zdir='z', offset=None):
        # 根据参数zdir设置z轴的方向
        zdir = '-' + zdir
        # 如果extend3d为True，则调用_3d_extend_contour方法扩展轮廓
        if extend3d:
            self._3d_extend_contour(cset, stride)
        else:
            # 否则将二维集合转换为三维集合
            art3d.collection_2d_to_3d(
                cset, zs=offset if offset is not None else cset.levels, zdir=zdir)

    def add_contourf_set(self, cset, zdir='z', offset=None):
        # 调用私有方法_add_contourf_set，用于处理填充轮廓集合
        self._add_contourf_set(cset, zdir=zdir, offset=offset)

    def _add_contourf_set(self, cset, zdir='z', offset=None):
        """
        返回
        -------
        levels : `numpy.ndarray`
            添加填充轮廓的级别。
        """
        # 根据参数zdir设置z轴的方向
        zdir = '-' + zdir

        # 计算每个填充轮廓的中点位置
        midpoints = cset.levels[:-1] + np.diff(cset.levels) / 2
        # 线性插值以获取任何扩展的级别
        if cset._extend_min:
            min_level = cset.levels[0] - np.diff(cset.levels[:2]) / 2
            midpoints = np.insert(midpoints, 0, min_level)
        if cset._extend_max:
            max_level = cset.levels[-1] + np.diff(cset.levels[-2:]) / 2
            midpoints = np.append(midpoints, max_level)

        # 将二维填充轮廓集合转换为三维集合
        art3d.collection_2d_to_3d(
            cset, zs=offset if offset is not None else midpoints, zdir=zdir)
        # 返回所有填充轮廓的级别
        return midpoints

    @_preprocess_data()
    def contour(self, X, Y, Z, *args,
                extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        """
        Create a 3D contour plot.

        Parameters
        ----------
        X, Y, Z : array-like,
            Input data. See `.Axes.contour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int, default: 5
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contour`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            A set of contour lines representing the data on the plot.
        """
        # Check if the plot had previous data
        had_data = self.has_data()

        # Rotate the input data arrays X, Y, Z based on the specified zdir
        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)

        # Create a contour plot using the rotated data
        cset = super().contour(jX, jY, jZ, *args, **kwargs)

        # Add the created contour set to the plot with specified parameters
        self.add_contour_set(cset, extend3d, stride, zdir, offset)

        # Adjust the plot limits based on the original data X, Y, Z
        self.auto_scale_xyz(X, Y, Z, had_data)

        # Return the created contour set
        return cset
    def tricontour(self, *args,
                   extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        """
        Create a 3D contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.tricontour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int, default: 5
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.tricontour`.

        Returns
        -------
        matplotlib.tri._tricontour.TriContourSet
        """
        # Check if the plot has data already
        had_data = self.has_data()

        # Extract triangulation and adjusted arguments and keyword arguments
        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(
                *args, **kwargs)
        X = tri.x
        Y = tri.y

        # Extract or pop 'Z' from kwargs, or use first positional argument for Z
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            Z, *args = args  # Ensure Z is not passed to Axes.tricontour as an argument

        # Rotate axes according to specified direction 'zdir'
        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)

        # Perform tricontour plot using adjusted data
        cset = super().tricontour(tri, jZ, *args, **kwargs)

        # Add contour set to the plot, considering 3D extension, stride, and offset
        self.add_contour_set(cset, extend3d, stride, zdir, offset)

        # Automatically scale the plot's X, Y, Z axes based on the data
        self.auto_scale_xyz(X, Y, Z, had_data)

        # Return the TriContourSet object created by tricontour
        return cset
    def contourf(self, X, Y, Z, *args, zdir='z', offset=None, **kwargs):
        """
        Create a 3D filled contour plot.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.contourf` for supported data shapes.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contourf`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        # Check if the plot already has data
        had_data = self.has_data()

        # Rotate input data to match the direction specified by zdir
        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        
        # Create a filled contour plot using the rotated data
        cset = super().contourf(jX, jY, jZ, *args, **kwargs)
        
        # Add the contour set to the plot with specified direction and offset
        levels = self._add_contourf_set(cset, zdir, offset)

        # Automatically scale the plot based on input data
        self._auto_scale_contourf(X, Y, Z, zdir, levels, had_data)
        
        # Return the created contour set
        return cset

    contourf3D = contourf

    @_preprocess_data()
    def tricontourf(self, *args, zdir='z', offset=None, **kwargs):
        """
        Create a 3D filled contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.tricontourf` for supported data shapes.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to zdir.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to
            `matplotlib.axes.Axes.tricontourf`.

        Returns
        -------
        matplotlib.tri._tricontour.TriContourSet
        """
        # Check if the plot already has data
        had_data = self.has_data()

        # Extract triangulation from input arguments and keyword arguments
        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(
                *args, **kwargs)
        X = tri.x
        Y = tri.y
        
        # Extract Z from kwargs or from positional arguments
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            # We do this so Z doesn't get passed as an arg to Axes.tricontourf
            Z, *args = args

        # Rotate input data to match the direction specified by zdir
        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        
        # Create a filled contour plot using the rotated data and triangulation
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)
        cset = super().tricontourf(tri, jZ, *args, **kwargs)
        
        # Add the contour set to the plot with specified direction and offset
        levels = self._add_contourf_set(cset, zdir, offset)

        # Automatically scale the plot based on input data
        self._auto_scale_contourf(X, Y, Z, zdir, levels, had_data)
        
        # Return the created contour set
        return cset
    def add_collection3d(self, col, zs=0, zdir='z', autolim=True):
        """
        Add a 3D collection object to the plot.

        2D collection types are converted to a 3D version by
        modifying the object and adding z coordinate information,
        *zs* and *zdir*.

        Supported 2D collection types are:

        - `.PolyCollection`
        - `.LineCollection`
        - `.PatchCollection` (currently not supporting *autolim*)

        Parameters
        ----------
        col : `.Collection`
            A 2D collection object.
        zs : float or array-like, default: 0
            The z-positions to be used for the 2D objects.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use for the z-positions.
        autolim : bool, default: True
            Whether to update the data limits.
        """
        # Check if the plot already has data
        had_data = self.has_data()

        # Ensure zs is an array
        zvals = np.atleast_1d(zs)
        
        # Determine the minimum z value; set to 0 if zs is empty
        zsortval = (np.min(zvals) if zvals.size else 0)  # FIXME: arbitrary default

        # Convert supported 2D collection types to 3D and set z-sort values
        if type(col) is mcoll.PolyCollection:
            # Convert PolyCollection to 3D
            art3d.poly_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.LineCollection:
            # Convert LineCollection to 3D
            art3d.line_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.PatchCollection:
            # Convert PatchCollection to 3D
            art3d.patch_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)

        # Automatically adjust plot limits if autolim is enabled
        if autolim:
            if isinstance(col, art3d.Line3DCollection):
                # Autoscale based on Line3DCollection segments
                self.auto_scale_xyz(*np.array(col._segments3d).transpose(),
                                    had_data=had_data)
            elif isinstance(col, art3d.Poly3DCollection):
                # Autoscale based on Poly3DCollection vertices
                self.auto_scale_xyz(*col._vec[:-1], had_data=had_data)
            elif isinstance(col, art3d.Patch3DCollection):
                # Currently no autoscaling function implemented for Patch3DCollection
                pass  # FIXME: Implement auto-scaling function for Patch3DCollection

        # Add the collection to the plot and return it
        collection = super().add_collection(col)
        return collection
    def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True,
                *args, **kwargs):
        """
        Create a scatter plot.

        Parameters
        ----------
        xs, ys : array-like
            The data positions.
        zs : float or array-like, default: 0
            The z-positions. Either an array of the same length as *xs* and
            *ys* or a single value to place all points in the same plane.
        zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, default: 'z'
            The axis direction for the *zs*. This is useful when plotting 2D
            data on a 3D Axes. The data must be passed as *xs*, *ys*. Setting
            *zdir* to 'y' then plots the data to the x-z-plane.

            See also :doc:`/gallery/mplot3d/2dcollections3d`.

        s : float or array-like, default: 20
            The marker size in points**2. Either an array of the same length
            as *xs* and *ys* or a single value to make all markers the same
            size.
        c : :mpltype:`color`, sequence, or sequence of colors, optional
            The marker color. Possible values:

            - A single color format string.
            - A sequence of colors of length n.
            - A sequence of n numbers to be mapped to colors using *cmap* and
              *norm*.
            - A 2D array in which the rows are RGB or RGBA.

            For more details see the *c* argument of `~.axes.Axes.scatter`.
        depthshade : bool, default: True
            Whether to shade the scatter markers to give the appearance of
            depth. Each call to ``scatter()`` will perform its depthshading
            independently.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs
            All other keyword arguments are passed on to `~.axes.Axes.scatter`.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`
        """

        # Check if there was any existing data on the plot
        had_data = self.has_data()
        
        # Store the original zs value
        zs_orig = zs
        
        # Ensure xs, ys, zs are broadcasted to compatible shapes
        xs, ys, zs = cbook._broadcast_with_masks(xs, ys, zs)
        
        # Ensure s (marker size) is raveled to match xs, ys in size
        s = np.ma.ravel(s)  # This doesn't have to match x, y in size.
        
        # Remove masked points from xs, ys, zs, s, c
        xs, ys, zs, s, c, color = cbook.delete_masked_points(
            xs, ys, zs, s, c, kwargs.get('color', None)
            )
        
        # Update kwargs['color'] if 'color' is in kwargs
        if kwargs.get("color") is not None:
            kwargs['color'] = color

        # Avoid unnecessary copies of zs if possible
        if np.may_share_memory(zs_orig, zs):
            zs = zs.copy()

        # Create the scatter plot patches
        patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
        
        # Convert 2D patch collection to 3D if necessary
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir,
                                        depthshade=depthshade)

        # Adjust z-margin if necessary
        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)

        # Automatically scale axes based on data
        self.auto_scale_xyz(xs, ys, zs, had_data)

        # Return the patches representing the scatter plot
        return patches

    # Alias scatter3D to scatter
    scatter3D = scatter

    # Decorate scatter function with _preprocess_data() decorator
    @_preprocess_data()
    def bar(self, left, height, zs=0, zdir='z', *args, **kwargs):
        """
        Add 2D bar(s) to the 3D plot.

        Parameters
        ----------
        left : 1D array-like
            The x coordinates of the left sides of the bars.
        height : 1D array-like
            The height of the bars.
        zs : float or 1D array-like, default: 0
            Z coordinate of bars; if a single value is specified, it will be
            used for all bars.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z ('x', 'y' or 'z').
        *args, **kwargs
            Additional positional and keyword arguments forwarded to
            `matplotlib.axes.Axes.bar`.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Patch3DCollection
            Collection of bar patches added to the plot.
        """
        had_data = self.has_data()

        # Call the bar method of the superclass to add bars to the plot
        patches = super().bar(left, height, *args, **kwargs)

        # Ensure zs is broadcasted to match the length of left
        zs = np.broadcast_to(zs, len(left), subok=True)

        verts = []
        verts_zs = []

        # Iterate through patches to extract vertices and adjust for 3D plotting
        for p, z in zip(patches, zs):
            vs = art3d._get_patch_verts(p)  # Get vertices of the bar patch
            verts += vs.tolist()  # Append vertices to the list
            verts_zs += [z] * len(vs)  # Append z coordinate for each vertex
            art3d.patch_2d_to_3d(p, z, zdir)  # Convert 2D bar patch to 3D
            if 'alpha' in kwargs:
                p.set_alpha(kwargs['alpha'])  # Set alpha value if provided in kwargs

        # If vertices are extracted, adjust axes and scale
        if len(verts) > 0:
            xs, ys = zip(*verts)
        else:
            xs, ys = [], []

        # Adjust axes and scaling based on the data
        xs, ys, verts_zs = art3d.juggle_axes(xs, ys, verts_zs, zdir)
        self.auto_scale_xyz(xs, ys, verts_zs, had_data)  # Autoscale XYZ based on data

        return patches

    @_preprocess_data()
    def set_title(self, label, fontdict=None, loc='center', **kwargs):
        """
        Set title for the 3D plot.

        Parameters
        ----------
        label : str
            The title text.
        fontdict : dict, optional
            A dictionary controlling the appearance of the title font.
        loc : {'center', 'left', 'right'}, default: 'center'
            The location of the title on the plot.
        **kwargs
            Additional keyword arguments forwarded to `matplotlib.axes.Axes.set_title`.

        Returns
        -------
        None
        """
        ret = super().set_title(label, fontdict=fontdict, loc=loc, **kwargs)
        (x, y) = self.title.get_position()
        self.title.set_y(0.92 * y)  # Adjust the vertical position of the title
        return ret

    @_preprocess_data()
    quiver3D = quiver

    def voxels(self, *args, facecolors=None, edgecolors=None, shade=True,
               lightsource=None, **kwargs):
        """
        Plot 3D voxel data on the axes.

        Parameters
        ----------
        *args
            Arguments passed to `matplotlib.axes.Axes.voxels`.
        facecolors : array-like, optional
            Face colors of the voxels.
        edgecolors : array-like, optional
            Edge colors of the voxels.
        shade : bool, default: True
            Whether to shade the voxels.
        lightsource : `matplotlib.colors.LightSource`, optional
            The light source for shading the voxels.
        **kwargs
            Additional keyword arguments forwarded to `matplotlib.axes.Axes.voxels`.

        Returns
        -------
        None
        """
        pass  # Placeholder for the method, actual implementation is omitted
    # 调用父类的方法获取紧凑边界框，可以通过参数自定义行为
    def get_tightbbox(self, renderer=None, call_axes_locator=True,
                      bbox_extra_artists=None, *, for_layout_only=False):
        # 调用父类方法获取紧凑边界框
        ret = super().get_tightbbox(renderer,
                                    call_axes_locator=call_axes_locator,
                                    bbox_extra_artists=bbox_extra_artists,
                                    for_layout_only=for_layout_only)
        
        # 初始化批处理列表，将父类方法返回的边界框对象添加到列表中
        batch = [ret]
        
        # 如果存在 3D 坐标轴
        if self._axis3don:
            # 遍历存储在 _axis_map 中的所有坐标轴对象
            for axis in self._axis_map.values():
                # 如果坐标轴可见
                if axis.get_visible():
                    # 调用 _get_tightbbox_for_layout_only 方法获取仅用于布局的紧凑边界框
                    axis_bb = martist._get_tightbbox_for_layout_only(
                        axis, renderer)
                    # 如果获取到了坐标轴的紧凑边界框，则将其添加到批处理列表中
                    if axis_bb:
                        batch.append(axis_bb)
        
        # 返回批处理列表中所有边界框对象的联合边界框
        return mtransforms.Bbox.union(batch)

    # 在数据预处理后，将 stem3D 属性设置为 stem 方法的装饰版本
    @_preprocess_data()
    stem3D = stem
def get_test_data(delta=0.05):
    """返回一个测试数据集的元组 X, Y, Z。"""
    # 生成从 -3.0 到 3.0 步长为 delta 的一维数组 x 和 y
    x = y = np.arange(-3.0, 3.0, delta)
    # 创建二维数组 X 和 Y，分别是 x 和 y 的网格
    X, Y = np.meshgrid(x, y)

    # 计算高斯分布函数 Z1 和 Z2，并相减得到 Z
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = Z2 - Z1

    # 对 X, Y, Z 进行放大倍数处理
    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z


class _Quaternion:
    """
    四元数类，由标量和向量组成，向量有 i, j, k 三个分量。
    """

    def __init__(self, scalar, vector):
        # 初始化四元数，接受标量和向量作为参数
        self.scalar = scalar
        self.vector = np.array(vector)

    def __neg__(self):
        # 返回当前四元数的负数
        return self.__class__(-self.scalar, -self.vector)

    def __mul__(self, other):
        """
        四元数乘法，根据标量和向量部分计算乘积。
        参考文献：<https://en.wikipedia.org/wiki/Quaternion#Scalar_and_vector_parts>
        """
        return self.__class__(
            self.scalar*other.scalar - np.dot(self.vector, other.vector),
            self.scalar*other.vector + self.vector*other.scalar
            + np.cross(self.vector, other.vector))

    def conjugate(self):
        """返回共轭四元数 -(1/2)*(q+i*q*i+j*q*j+k*q*k)"""
        return self.__class__(self.scalar, -self.vector)

    @property
    def norm(self):
        """返回四元数的 2-范数，即标量部分平方加上向量部分的点积"""
        return self.scalar*self.scalar + np.dot(self.vector, self.vector)

    def normalize(self):
        """将四元数标准化，使其 2-范数为 1"""
        n = np.sqrt(self.norm)
        return self.__class__(self.scalar/n, self.vector/n)

    def reciprocal(self):
        """返回四元数的倒数，即 q' / norm(q)"""
        n = self.norm
        return self.__class__(self.scalar/n, -self.vector/n)

    def __div__(self, other):
        # 实现四元数的除法运算，等同于乘以其倒数
        return self*other.reciprocal()

    __truediv__ = __div__

    def rotate(self, v):
        # 将向量 v 绕四元数 q 旋转，计算 q*v/q 的向量部分
        v = self.__class__(0, v)
        v = self*v/self
        return v.vector

    def __eq__(self, other):
        # 比较两个四元数是否相等
        return (self.scalar == other.scalar) and (self.vector == other.vector).all

    def __repr__(self):
        # 返回四元数的字符串表示形式
        return "_Quaternion({}, {})".format(repr(self.scalar), repr(self.vector))

    @classmethod
    @classmethod
    def rotate_from_to(cls, r1, r2):
        """
        The quaternion for the shortest rotation from vector r1 to vector r2
        i.e., q = sqrt(r2*r1'), normalized.
        If r1 and r2 are antiparallel, then the result is ambiguous;
        a normal vector will be returned, and a warning will be issued.
        """
        # Calculate the cross product between r1 and r2
        k = np.cross(r1, r2)
        # Compute the norm of the cross product
        nk = np.linalg.norm(k)
        # Compute the angle using arctan2 and dot product of r1 and r2
        th = np.arctan2(nk, np.dot(r1, r2))
        th = th/2
        # Check if r1 and r2 are parallel or anti-parallel
        if nk == 0:
            # r1 and r2 are anti-parallel
            if np.dot(r1, r2) < 0:
                # Issue a warning about the ambiguity of rotation
                warnings.warn("Rotation defined by anti-parallel vectors is ambiguous")
                # Define a basis vector most perpendicular to r1-r2
                k = np.zeros(3)
                k[np.argmin(r1*r1)] = 1
                k = np.cross(r1, k)
                k = k / np.linalg.norm(k)  # Normalize the vector
                q = cls(0, k)
            else:
                # r1 and r2 are parallel, no rotation
                q = cls(1, [0, 0, 0])
        else:
            # Compute the quaternion based on the calculated angle and axis
            q = cls(math.cos(th), k*math.sin(th)/nk)
        return q

    @classmethod
    def from_cardan_angles(cls, elev, azim, roll):
        """
        Converts the angles to a quaternion
            q = exp((roll/2)*e_x)*exp((elev/2)*e_y)*exp((-azim/2)*e_z)
        i.e., the angles are a kind of Tait-Bryan angles, -z,y',x".
        The angles should be given in radians, not degrees.
        """
        # Compute cosine and sine of each angle divided by 2
        ca, sa = np.cos(azim/2), np.sin(azim/2)
        ce, se = np.cos(elev/2), np.sin(elev/2)
        cr, sr = np.cos(roll/2), np.sin(roll/2)

        # Compute the components of the quaternion
        qw = ca*ce*cr + sa*se*sr
        qx = ca*ce*sr - sa*se*cr
        qy = ca*se*cr + sa*ce*sr
        qz = ca*se*sr - sa*ce*cr
        return cls(qw, [qx, qy, qz])

    def as_cardan_angles(self):
        """
        The inverse of `from_cardan_angles()`.
        Note that the angles returned are in radians, not degrees.
        """
        # Extract quaternion components
        qw = self.scalar
        qx, qy, qz = self.vector[..., :]
        # Compute Euler angles using quaternion components
        azim = np.arctan2(2*(-qw*qz+qx*qy), qw*qw+qx*qx-qy*qy-qz*qz)
        elev = np.arcsin(2*(qw*qy+qz*qx)/(qw*qw+qx*qx+qy*qy+qz*qz))  # noqa E201
        roll = np.arctan2(2*(qw*qx-qy*qz), qw*qw-qx*qx-qy*qy+qz*qz)   # noqa E201
        return elev, azim, roll
```