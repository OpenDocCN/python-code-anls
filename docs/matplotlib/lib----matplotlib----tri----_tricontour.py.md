# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_tricontour.py`

```py
# 导入NumPy库，并使用别名np
import numpy as np

# 从matplotlib库中导入_docstring模块
from matplotlib import _docstring

# 从matplotlib.contour模块中导入ContourSet类
from matplotlib.contour import ContourSet

# 从matplotlib.tri._triangulation模块中导入Triangulation类
from matplotlib.tri._triangulation import Triangulation

# 使用_docstring.dedent_interpd修饰器处理文档字符串的缩进
@_docstring.dedent_interpd
# 定义TriContourSet类，继承自ContourSet类
class TriContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions for
    a triangular grid.

    This class is typically not instantiated directly by the user but by
    `~.Axes.tricontour` and `~.Axes.tricontourf`.

    %(contour_set_attributes)s
    """

    # 构造函数，初始化TriContourSet对象
    def __init__(self, ax, *args, **kwargs):
        """
        Draw triangular grid contour lines or filled regions,
        depending on whether keyword arg *filled* is False
        (default) or True.

        The first argument of the initializer must be an `~.axes.Axes`
        object.  The remaining arguments and keyword arguments
        are described in the docstring of `~.Axes.tricontour`.
        """
        # 调用父类ContourSet的构造函数进行初始化
        super().__init__(ax, *args, **kwargs)

    # 内部方法，用于处理参数
    def _process_args(self, *args, **kwargs):
        """
        Process args and kwargs.
        """
        # 如果第一个参数是TriContourSet对象
        if isinstance(args[0], TriContourSet):
            # 从args[0]获取_contour_generator对象，并赋值给C
            C = args[0]._contour_generator
            # 如果self.levels为None，则从args[0]继承levels属性
            if self.levels is None:
                self.levels = args[0].levels
            # 继承args[0]的zmin和zmax属性
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
        else:
            # 导入matplotlib._tri模块
            from matplotlib import _tri
            # 使用_contour_args方法处理args和kwargs，获取tri和z
            tri, z = self._contour_args(args, kwargs)
            # 创建_tri.TriContourGenerator对象C
            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
            # 设置self._mins和self._maxs属性
            self._mins = [tri.x.min(), tri.y.min()]
            self._maxs = [tri.x.max(), tri.y.max()]

        # 将C赋值给self._contour_generator属性，并返回kwargs
        self._contour_generator = C
        return kwargs
    # 定义一个方法 _contour_args，接收参数 args 和 kwargs
    def _contour_args(self, args, kwargs):
        # 调用 Triangulation 类的 get_from_args_and_kwargs 方法，获取三角化对象 tri 和更新后的 args 和 kwargs
        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
                                                                   **kwargs)
        # 从 args 中获取 z，并将剩余的参数重新赋值给 args
        z, *args = args
        # 将 z 转换为 MaskedArray 类型
        z = np.ma.asarray(z)
        # 检查 z 的形状是否与三角化对象 tri 的 x 和 y 数组长度相同，如果不同则引发 ValueError 异常
        if z.shape != tri.x.shape:
            raise ValueError('z array must have same length as triangulation x'
                             ' and y arrays')

        # 只需检查包含在三角化中的点的 z 值是否有限
        z_check = z[np.unique(tri.get_masked_triangles())]
        # 如果 z_check 包含掩码值，则引发 ValueError 异常
        if np.ma.is_masked(z_check):
            raise ValueError('z must not contain masked points within the '
                             'triangulation')
        # 如果 z_check 中存在非有限值，则引发 ValueError 异常
        if not np.isfinite(z_check).all():
            raise ValueError('z array must not contain non-finite values '
                             'within the triangulation')

        # 将 z 中的无效数据掩码化
        z = np.ma.masked_invalid(z, copy=False)
        # 计算 z_check 中的最大值，并赋值给 self.zmax
        self.zmax = float(z_check.max())
        # 计算 z_check 中的最小值，并赋值给 self.zmin
        self.zmin = float(z_check.min())
        # 如果使用对数尺度并且 self.zmin 小于等于 0，则根据填充状态选择对应的错误信息函数，并引发 ValueError 异常
        if self.logscale and self.zmin <= 0:
            func = 'contourf' if self.filled else 'contour'
            raise ValueError(f'Cannot {func} log of negative values.')
        # 处理轮廓等级的参数，并根据 z 的数据类型进行处理
        self._process_contour_level_args(args, z.dtype)
        # 返回包含 tri 和 z 的元组
        return (tri, z)
_docstring.interpd.update(_tricontour_doc="""
Draw contour %(type)s on an unstructured triangular grid.

Call signatures::

    %(func)s(triangulation, z, [levels], ...)
    %(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)

The triangular grid can be specified either by passing a `.Triangulation`
object as the first parameter, or by passing the points *x*, *y* and
optionally the *triangles* and a *mask*. See `.Triangulation` for an
explanation of these parameters. If neither of *triangulation* or
*triangles* are given, the triangulation is calculated on the fly.

It is possible to pass *triangles* positionally, i.e.
``%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more
clarity, pass *triangles* via keyword argument.

Parameters
----------
triangulation : `.Triangulation`, optional
    An already created triangular grid.

x, y, triangles, mask
    Parameters defining the triangular grid. See `.Triangulation`.
    This is mutually exclusive with specifying *triangulation*.

z : array-like
    The height values over which the contour is drawn.  Color-mapping is
    controlled by *cmap*, *norm*, *vmin*, and *vmax*.

    .. note::
        All values in *z* must be finite. Hence, nan and inf values must
        either be removed or `~.Triangulation.set_mask` be used.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
    automatically choose no more than *n+1* "nice" contour levels between
    between minimum and maximum numeric values of *Z*.

    If array-like, draw contour lines at the specified levels.  The values must
    be in increasing order.

Returns
-------
`~matplotlib.tri.TriContourSet`

Other Parameters
----------------
colors : :mpltype:`color` or list of :mpltype:`color`, optional
    The colors of the levels, i.e., the contour %(type)s.

    The sequence is cycled for the levels in ascending order. If the sequence
    is shorter than the number of levels, it is repeated.

    As a shortcut, single color strings may be used in place of one-element
    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
    same color. This shortcut does only work for color strings, not for other
    ways of specifying colors.

    By default (value *None*), the colormap specified by *cmap* will be used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

%(cmap_doc)s

    This parameter is ignored if *colors* is set.

%(norm_doc)s

    This parameter is ignored if *colors* is set.

%(vmin_vmax_doc)s

    If *vmin* or *vmax* are not given, the default color scaling is based on
    *levels*.

    This parameter is ignored if *colors* is set.

origin : {*None*, 'upper', 'lower', 'image'}, default: None
    Determines the orientation and exact position of *z* by specifying the
"""
    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.

    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.
    - 'lower': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
    - 'upper': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
    - 'image': Use the value from :rc:`image.origin`.
# extent : (x0, x1, y0, y1), optional
# 如果 origin 不是 None，则 extent 被解释为 `.imshow` 中的内容：它给出外部像素边界。
# 在这种情况下，z[0, 0] 的位置是像素中心，而不是角落。
# 如果 origin 是 None，则 (x0, y0) 是 z[0, 0] 的位置，(x1, y1) 是 z[-1, -1] 的位置。

# locator : ticker.Locator subclass, optional
# 如果没有显式指定 levels，locator 用于确定等高线的级别。
# 默认为 `~.ticker.MaxNLocator`。

# extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
# 确定超出 levels 范围的值的着色方式。

# 如果是 'neither'，超出 levels 范围的值不着色。
# 如果是 'min'、'max' 或 'both'，分别着色低于、高于或低于和高于 levels 范围的值。
# 小于 ``min(levels)`` 和大于 ``max(levels)`` 的值被映射到 `.Colormap` 的 under/over 值。
# 注意，大多数 colormap 默认情况下不具有这些专门的颜色，因此 over 和 under 值是 colormap 的边缘值。
# 可以使用 `.Colormap.set_under` 和 `.Colormap.set_over` 显式设置这些值。

# .. note::
#    如果更改其 colormap 的属性，现有的 `.TriContourSet` 不会收到通知。
#    因此，在修改 colormap 后需要显式调用 `.ContourSet.changed()`。
#    如果将颜色条分配给 `.TriContourSet`，则可以省略显式调用，因为它在内部调用 `.ContourSet.changed()`。

# xunits, yunits : registered units, optional
# 通过指定 `matplotlib.units.ConversionInterface` 的实例来覆盖轴单位。

# antialiased : bool, optional
# 启用抗锯齿功能，覆盖默认设置。
# 对于填充的等高线，默认为 True。
# 对于线条等高线，它从 :rc:`lines.antialiased` 获取。
    # 设置默认的线型样式为None，可选值包括'solid', 'dashed', 'dashdot', 'dotted'，如果未指定则使用'solid'，
    # 除非所有线条都是单色的，此时负轮廓将使用rc配置中的'contour.negative_linestyle'设置。
    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
        If *linestyles* is *None*, the default is 'solid' unless the lines are
        monochrome.  In that case, negative contours will take their linestyle
        from :rc:`contour.negative_linestyle` setting.

        *linestyles* can also be an iterable of the above strings specifying a
        set of linestyles to be used. If this iterable is shorter than the
        number of contour levels it will be repeated as necessary.
    """
    # 将'filled'关键字参数设为False
    kwargs['filled'] = False
    # 使用TriContourSet类创建等高线图对象，传入参数ax, *args和**kwargs
    return TriContourSet(ax, *args, **kwargs)
# 应用函数装饰器，将 tricontourf 函数的文档字符串中的 %(_tricontour_doc)s 替换为 tricontourf 类型的文档内容
@_docstring.Substitution(func='tricontourf', type='regions')
# 应用函数装饰器，用于处理文档字符串的缩进
@_docstring.dedent_interpd
def tricontourf(ax, *args, **kwargs):
    """
    %(_tricontour_doc)s
    描述：用于在三角形网格上填充等高线图。

    参数：
    ax : Axes
        绘图所用的轴对象。
    *args
        传递给 TriContourSet 的位置参数。
    **kwargs
        传递给 TriContourSet 的关键字参数。

    可选参数：
    hatches : list[str], optional
        用于填充区域的交叉图案列表。
        如果为 None，则填充区域不添加任何图案。

    注意事项
    -----
    `.tricontourf` 方法用于填充闭合于顶部的间隔；也就是说，对于边界 *z1* 和 *z2*，填充的区域是::

        z1 < Z <= z2

    最低的间隔是两侧都闭合的（即包括最低值）。
    """
    # 设置 filled 参数为 True，表示要进行填充操作
    kwargs['filled'] = True
    # 返回 TriContourSet 对象，用于绘制填充的等高线图
    return TriContourSet(ax, *args, **kwargs)
```