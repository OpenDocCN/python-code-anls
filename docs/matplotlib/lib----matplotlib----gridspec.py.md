# `D:\src\scipysrc\matplotlib\lib\matplotlib\gridspec.py`

```
r"""
:mod:`~matplotlib.gridspec` contains classes that help to layout multiple
`~.axes.Axes` in a grid-like pattern within a figure.

The `GridSpec` specifies the overall grid structure. Individual cells within
the grid are referenced by `SubplotSpec`\s.

Often, users need not access this module directly, and can use higher-level
methods like `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` and
`~.Figure.subfigures`. See the tutorial :ref:`arranging_axes` for a guide.
"""

import copy  # 导入用于复制对象的标准库
import logging  # 导入用于日志记录的标准库
from numbers import Integral  # 从标准库导入用于数值类型检查的 Integral 类

import numpy as np  # 导入数值计算库 NumPy

import matplotlib as mpl  # 导入绘图库 Matplotlib
from matplotlib import _api, _pylab_helpers, _tight_layout  # 导入 Matplotlib 内部模块
from matplotlib.transforms import Bbox  # 从 Matplotlib 导入边界框变换类 Bbox

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.
        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.
        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        if not isinstance(nrows, Integral) or nrows <= 0:
            raise ValueError(
                f"Number of rows must be a positive integer, not {nrows!r}")  # 抛出值错误异常，如果行数不是正整数
        if not isinstance(ncols, Integral) or ncols <= 0:
            raise ValueError(
                f"Number of columns must be a positive integer, not {ncols!r}")  # 抛出值错误异常，如果列数不是正整数
        self._nrows, self._ncols = nrows, ncols  # 设置对象的行数和列数属性
        self.set_height_ratios(height_ratios)  # 调用设置行高比例的方法
        self.set_width_ratios(width_ratios)  # 调用设置列宽比例的方法

    def __repr__(self):
        height_arg = (f', height_ratios={self._row_height_ratios!r}'
                      if len(set(self._row_height_ratios)) != 1 else '')  # 按需设置行高比例参数字符串
        width_arg = (f', width_ratios={self._col_width_ratios!r}'
                     if len(set(self._col_width_ratios)) != 1 else '')  # 按需设置列宽比例参数字符串
        return '{clsname}({nrows}, {ncols}{optionals})'.format(
            clsname=self.__class__.__name__,  # 获取类名
            nrows=self._nrows,  # 获取行数
            ncols=self._ncols,  # 获取列数
            optionals=height_arg + width_arg,  # 组合高度和宽度比例参数字符串
            )

    nrows = property(lambda self: self._nrows,
                     doc="The number of rows in the grid.")  # 定义属性 nrows，表示网格中的行数
    ncols = property(lambda self: self._ncols,
                     doc="The number of columns in the grid.")  # 定义属性 ncols，表示网格中的列数
    def get_geometry(self):
        """
        返回包含网格行数和列数的元组。
        """
        return self._nrows, self._ncols

    def get_subplot_params(self, figure=None):
        """
        在子类中必须实现，用于获取子图的参数。
        """
        pass

    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        """
        创建并返回一个 `.SubplotSpec` 实例。

        Parameters
        ----------
        loc : (int, int)
            子图在网格中的位置，格式为 ``(行索引, 列索引)``。
        rowspan, colspan : int, 默认为 1
            子图在网格中跨越的行数和列数。
        """
        loc1, loc2 = loc
        subplotspec = self[loc1:loc1+rowspan, loc2:loc2+colspan]
        return subplotspec

    def set_width_ratios(self, width_ratios):
        """
        设置列的相对宽度。

        *width_ratios* 的长度必须为 *ncols*。每列的相对宽度为 ``width_ratios[i] / sum(width_ratios)``。
        """
        if width_ratios is None:
            width_ratios = [1] * self._ncols
        elif len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to '
                             'match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def get_width_ratios(self):
        """
        返回列的宽度比例。

        如果没有明确设置列的宽度比例，则返回 *None*。
        """
        return self._col_width_ratios

    def set_height_ratios(self, height_ratios):
        """
        设置行的相对高度。

        *height_ratios* 的长度必须为 *nrows*。每行的相对高度为 ``height_ratios[i] / sum(height_ratios)``。
        """
        if height_ratios is None:
            height_ratios = [1] * self._nrows
        elif len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to '
                             'match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_height_ratios(self):
        """
        返回行的高度比例。

        如果没有明确设置行的高度比例，则返回 *None*。
        """
        return self._row_height_ratios
    def get_grid_positions(self, fig):
        """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
        # 获取网格的行数和列数
        nrows, ncols = self.get_geometry()
        # 获取子图参数（边距和子图之间的间距）从传入的图形对象中
        subplot_params = self.get_subplot_params(fig)
        # 提取左侧、右侧、底部和顶部的位置信息
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace  # 子图之间的水平间距
        hspace = subplot_params.hspace  # 子图之间的垂直间距
        tot_width = right - left  # 总宽度
        tot_height = top - bottom  # 总高度

        # 计算列的累积高度
        cell_h = tot_height / (nrows + hspace*(nrows-1))  # 每个单元格的高度
        sep_h = hspace * cell_h  # 分隔线的高度
        norm = cell_h * nrows / sum(self._row_height_ratios)  # 标准化系数
        cell_heights = [r * norm for r in self._row_height_ratios]  # 每行单元格的高度
        sep_heights = [0] + ([sep_h] * (nrows-1))  # 分隔线的高度列表
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)  # 累积高度

        # 计算行的累积宽度
        cell_w = tot_width / (ncols + wspace*(ncols-1))  # 每个单元格的宽度
        sep_w = wspace * cell_w  # 分隔线的宽度
        norm = cell_w * ncols / sum(self._col_width_ratios)  # 标准化系数
        cell_widths = [r * norm for r in self._col_width_ratios]  # 每列单元格的宽度
        sep_widths = [0] + ([sep_w] * (ncols-1))  # 分隔线的宽度列表
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)  # 累积宽度

        # 计算每个单元格的位置边界
        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T  # 计算顶部和底部边界
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T  # 计算左侧和右侧边界
        return fig_bottoms, fig_tops, fig_lefts, fig_rights

    @staticmethod
    def _check_gridspec_exists(figure, nrows, ncols):
        """
        Check if the figure already has a gridspec with these dimensions,
        or create a new one
        """
        # 遍历图形中的所有轴，检查是否存在指定尺寸的网格规格
        for ax in figure.get_axes():
            gs = ax.get_gridspec()
            if gs is not None:
                if hasattr(gs, 'get_topmost_subplotspec'):
                    # 对于颜色条的网格规格布局，需要这个特定的检查
                    gs = gs.get_topmost_subplotspec().get_gridspec()
                if gs.get_geometry() == (nrows, ncols):
                    return gs  # 如果找到匹配的网格规格，则返回该网格规格对象
        # 如果未找到匹配的网格规格，则创建一个新的网格规格对象并返回
        return GridSpec(nrows, ncols, figure=figure)
    # 定义特殊方法 `__getitem__`，用于获取对象中指定键的值，通常用于索引操作
    def __getitem__(self, key):
        """Create and return a `.SubplotSpec` instance."""
        
        # 获取图表布局的行数和列数
        nrows, ncols = self.get_geometry()

        # 定义内部函数 `_normalize`，用于规范化索引
        def _normalize(key, size, axis):  # Includes last index.
            orig_key = key
            if isinstance(key, slice):
                # 处理切片类型的索引
                start, stop, _ = key.indices(size)
                if stop > start:
                    return start, stop - 1
                # 如果切片范围为空，则抛出索引错误
                raise IndexError("GridSpec slice would result in no space "
                                 "allocated for subplot")
            else:
                # 处理单个索引
                if key < 0:
                    key = key + size
                if 0 <= key < size:
                    return key, key
                elif axis is not None:
                    # 如果索引超出范围，针对指定的轴抛出索引错误
                    raise IndexError(f"index {orig_key} is out of bounds for "
                                     f"axis {axis} with size {size}")
                else:  # flat index
                    # 如果索引超出范围，针对整体布局抛出索引错误
                    raise IndexError(f"index {orig_key} is out of bounds for "
                                     f"GridSpec with size {size}")

        # 如果键是元组类型，表示多个索引
        if isinstance(key, tuple):
            try:
                k1, k2 = key
            except ValueError as err:
                # 如果解析元组失败，抛出值错误
                raise ValueError("Unrecognized subplot spec") from err
            # 根据行列索引规范化后，将二维索引转换为一维索引
            num1, num2 = np.ravel_multi_index(
                [_normalize(k1, nrows, 0), _normalize(k2, ncols, 1)],
                (nrows, ncols))
        else:  # 单个索引
            # 对单个索引进行规范化，并转换为一维索引
            num1, num2 = _normalize(key, nrows * ncols, None)

        # 返回由当前对象、行索引和列索引构成的 SubplotSpec 实例
        return SubplotSpec(self, num1, num2)
    def subplots(self, *, sharex=False, sharey=False, squeeze=True,
                 subplot_kw=None):
        """
        Add all subplots specified by this `GridSpec` to its parent figure.

        See `.Figure.subplots` for detailed documentation.
        """

        # 获取 GridSpec 对应的父图对象
        figure = self.figure

        # 如果父图对象为空，则抛出数值错误异常
        if figure is None:
            raise ValueError("GridSpec.subplots() only works for GridSpecs "
                             "created with a parent figure")

        # 如果 sharex 不是字符串类型，则根据其布尔值设置为 "all" 或 "none"
        if not isinstance(sharex, str):
            sharex = "all" if sharex else "none"
        # 如果 sharey 不是字符串类型，则根据其布尔值设置为 "all" 或 "none"
        if not isinstance(sharey, str):
            sharey = "all" if sharey else "none"

        # 检查 sharex 和 sharey 是否在指定的列表中
        _api.check_in_list(["all", "row", "col", "none", False, True],
                           sharex=sharex, sharey=sharey)

        # 如果 subplot_kw 为 None，则初始化为空字典
        if subplot_kw is None:
            subplot_kw = {}
        # 不改变用户传入的关键字参数，复制一份给 subplot_kw
        subplot_kw = subplot_kw.copy()

        # 创建一个数组来保存所有的 Axes 对象
        axarr = np.empty((self._nrows, self._ncols), dtype=object)
        for row in range(self._nrows):
            for col in range(self._ncols):
                # 根据 sharex 和 sharey 的设置，确定共享的 Axes 对象
                shared_with = {"none": None, "all": axarr[0, 0],
                               "row": axarr[row, 0], "col": axarr[0, col]}
                subplot_kw["sharex"] = shared_with[sharex]
                subplot_kw["sharey"] = shared_with[sharey]
                # 在父图中添加子图，并使用指定的 subplot_kw 参数
                axarr[row, col] = figure.add_subplot(
                    self[row, col], **subplot_kw)

        # 关闭冗余的刻度标签
        if sharex in ["col", "all"]:
            for ax in axarr.flat:
                ax._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ["row", "all"]:
            for ax in axarr.flat:
                ax._label_outer_yaxis(skip_non_rectangular_axes=True)

        # 如果 squeeze 为 True，则丢弃等于1的不必要维度；如果只有一个子图，直接返回它而不是一个长度为1的数组
        if squeeze:
            return axarr.item() if axarr.size == 1 else axarr.squeeze()
        else:
            # 返回的轴数组始终是二维的，即使 nrows=ncols=1 也是如此
            return axarr
class GridSpec(GridSpecBase):
    """
    A grid layout to place subplots within a figure.

    The location of the grid cells is determined in a similar way to
    `.SubplotParams` using *left*, *right*, *top*, *bottom*, *wspace*
    and *hspace*.

    Indexing a GridSpec instance returns a `.SubplotSpec`.
    """

    def __init__(self, nrows, ncols, figure=None,
                 left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None,
                 width_ratios=None, height_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.

        figure : `.Figure`, optional
            Only used for constrained layout to create a proper layout grid.

        left, right, top, bottom : float, optional
            Extent of the subplots as a fraction of figure width or height.
            Left cannot be larger than right, and bottom cannot be larger than
            top. If not given, the values will be inferred from a figure or
            rcParams at draw time. See also `GridSpec.get_subplot_params`.

        wspace : float, optional
            The amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        hspace : float, optional
            The amount of height reserved for space between subplots,
            expressed as a fraction of the average axis height.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        # 设置子图布局的位置参数
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        # 设置子图之间的宽度和高度空间
        self.wspace = wspace
        self.hspace = hspace
        # 关联的图形对象
        self.figure = figure

        # 调用父类构造函数，初始化网格布局
        super().__init__(nrows, ncols,
                         width_ratios=width_ratios,
                         height_ratios=height_ratios)

    # 定义允许设置的键列表
    _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]
    def update(self, **kwargs):
        """
        Update the subplot parameters of the grid.

        Parameters that are not explicitly given are not changed. Setting a
        parameter to *None* resets it to :rc:`figure.subplot.*`.

        Parameters
        ----------
        left, right, top, bottom : float or None, optional
            Extent of the subplots as a fraction of figure width or height.
        wspace, hspace : float, optional
            Spacing between the subplots as a fraction of the average subplot
            width / height.
        """
        # 遍历所有传入的关键字参数
        for k, v in kwargs.items():
            # 检查是否在允许的参数列表中
            if k in self._AllowedKeys:
                # 设置对象的属性值为传入的值
                setattr(self, k, v)
            else:
                # 如果关键字不在允许的参数列表中，则引发异常
                raise AttributeError(f"{k} is an unknown keyword")

        # 遍历所有当前活动的图形管理器中的所有轴
        for figmanager in _pylab_helpers.Gcf.figs.values():
            for ax in figmanager.canvas.figure.axes:
                # 检查轴是否具有子图规范
                if ax.get_subplotspec() is not None:
                    # 获取顶层的子图规范
                    ss = ax.get_subplotspec().get_topmost_subplotspec()
                    # 如果子图规范属于当前的网格规范对象，则设置轴的位置
                    if ss.get_gridspec() == self:
                        ax._set_position(
                            ax.get_subplotspec().get_position(ax.figure))

    def get_subplot_params(self, figure=None):
        """
        Return the `.SubplotParams` for the GridSpec.

        In order of precedence the values are taken from

        - non-*None* attributes of the GridSpec
        - the provided *figure*
        - :rc:`figure.subplot.*`

        Note that the ``figure`` attribute of the GridSpec is always ignored.
        """
        # 如果未提供 figure 参数，则从全局配置中获取参数值
        if figure is None:
            kw = {k: mpl.rcParams["figure.subplot."+k]
                  for k in self._AllowedKeys}
            subplotpars = SubplotParams(**kw)
        else:
            # 否则，从提供的 figure 对象中复制 subplot 参数
            subplotpars = copy.copy(figure.subplotpars)

        # 更新 subplot 参数对象的属性值
        subplotpars.update(**{k: getattr(self, k) for k in self._AllowedKeys})

        return subplotpars

    def locally_modified_subplot_params(self):
        """
        Return a list of the names of the subplot parameters explicitly set
        in the GridSpec.

        This is a subset of the attributes of `.SubplotParams`.
        """
        # 返回当前对象中已显式设置的子图参数的名称列表
        return [k for k in self._AllowedKeys if getattr(self, k)]
    def tight_layout(self, figure, renderer=None,
                     pad=1.08, h_pad=None, w_pad=None, rect=None):
        """
        Adjust subplot parameters to give specified padding.

        Parameters
        ----------
        figure : `.Figure`
            The figure object to adjust subplot parameters for.
        renderer :  `.RendererBase` subclass, optional
            The renderer to be used for rendering the figure. If not provided,
            it defaults to the renderer obtained from the figure.
        pad : float
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font size.
        h_pad, w_pad : float, optional
            Padding (height/width) between edges of adjacent subplots. If not
            specified, defaults to the value of `pad`.
        rect : tuple (left, bottom, right, top), default: None
            Specifies the normalized figure coordinates of the subplot area,
            including labels. If not provided, defaults to the entire figure.
        """
        # 如果没有指定渲染器，则使用 figure 对象的方法获取渲染器
        if renderer is None:
            renderer = figure._get_renderer()
        # 调用 _tight_layout 模块的方法，获取调整 subplot 参数的关键字参数
        kwargs = _tight_layout.get_tight_layout_figure(
            figure, figure.axes,
            _tight_layout.get_subplotspec_list(figure.axes, grid_spec=self),
            renderer, pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        # 如果返回了关键字参数，则更新当前对象的属性
        if kwargs:
            self.update(**kwargs)
class GridSpecFromSubplotSpec(GridSpecBase):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """
    def __init__(self, nrows, ncols,
                 subplot_spec,
                 wspace=None, hspace=None,
                 height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            Number of rows and number of columns of the grid.
        subplot_spec : SubplotSpec
            Spec from which the layout parameters are inherited.
        wspace, hspace : float, optional
            See `GridSpec` for more details. If not specified default values
            (from the figure or rcParams) are used.
        height_ratios : array-like of length *nrows*, optional
            See `GridSpecBase` for details.
        width_ratios : array-like of length *ncols*, optional
            See `GridSpecBase` for details.
        """
        # 设置水平和垂直间距
        self._wspace = wspace
        self._hspace = hspace
        # 如果 subplot_spec 是 SubplotSpec 类型，则使用给定的 subplot_spec
        if isinstance(subplot_spec, SubplotSpec):
            self._subplot_spec = subplot_spec
        else:
            # 如果不是 SubplotSpec 类型，则抛出类型错误
            raise TypeError(
                            "subplot_spec must be type SubplotSpec, "
                            "usually from GridSpec, or axes.get_subplotspec.")
        # 获取 subplot_spec 所属的图形对象
        self.figure = self._subplot_spec.get_gridspec().figure
        # 调用父类的初始化方法
        super().__init__(nrows, ncols,
                         width_ratios=width_ratios,
                         height_ratios=height_ratios)

    def get_subplot_params(self, figure=None):
        """Return a dictionary of subplot layout parameters."""
        # 获取水平和垂直间距，优先使用传入的参数，其次使用 figure 的 subplot 参数，最后使用默认配置
        hspace = (self._hspace if self._hspace is not None
                  else figure.subplotpars.hspace if figure is not None
                  else mpl.rcParams["figure.subplot.hspace"])
        wspace = (self._wspace if self._wspace is not None
                  else figure.subplotpars.wspace if figure is not None
                  else mpl.rcParams["figure.subplot.wspace"])

        # 获取 subplot_spec 所属的图形对象的位置信息
        figbox = self._subplot_spec.get_position(figure)
        left, bottom, right, top = figbox.extents

        # 返回包含布局参数的 SubplotParams 对象
        return SubplotParams(left=left, right=right,
                             bottom=bottom, top=top,
                             wspace=wspace, hspace=hspace)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `.SubplotSpec` instance associated with the subplot.
        """
        # 返回与此 subplot 相关联的最顶层的 SubplotSpec 实例
        return self._subplot_spec.get_topmost_subplotspec()


class SubplotSpec:
    """
    The location of a subplot in a `GridSpec`.

    .. note::

        Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
        you will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    """
    num1, num2 : int
        The subplot will occupy the *num1*-th cell of the given
        *gridspec*.  If *num2* is provided, the subplot will span between
        *num1*-th cell and *num2*-th cell **inclusive**.

        The index starts from 0.
    """
    def __init__(self, gridspec, num1, num2=None):
        # 初始化方法，保存所属的 gridspec 对象及两个整数参数 num1 和 num2
        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2

    def __repr__(self):
        # 返回对象的字符串表示形式，显示所属 gridspec 的行列范围
        return (f"{self.get_gridspec()}["
                f"{self.rowspan.start}:{self.rowspan.stop}, "
                f"{self.colspan.start}:{self.colspan.stop}]")

    @staticmethod
    def _from_subplot_args(figure, args):
        """
        Construct a `.SubplotSpec` from a parent `.Figure` and either

        - a `.SubplotSpec` -- returned as is;
        - one or three numbers -- a MATLAB-style subplot specifier.
        """
        if len(args) == 1:
            arg, = args
            if isinstance(arg, SubplotSpec):
                return arg
            elif not isinstance(arg, Integral):
                # 如果参数不是整数，则引发 ValueError 异常
                raise ValueError(
                    f"Single argument to subplot must be a three-digit "
                    f"integer, not {arg!r}")
            try:
                # 尝试解析参数为三位数（行、列、编号）
                rows, cols, num = map(int, str(arg))
            except ValueError:
                # 解析失败则引发 ValueError 异常
                raise ValueError(
                    f"Single argument to subplot must be a three-digit "
                    f"integer, not {arg!r}") from None
        elif len(args) == 3:
            # 如果参数为三个数，则依次赋值给行数、列数、编号
            rows, cols, num = args
        else:
            # 参数数量不符合要求时，引发参数错误异常
            raise _api.nargs_error("subplot", takes="1 or 3", given=len(args))

        # 检查并获取或创建 gridspec 对象
        gs = GridSpec._check_gridspec_exists(figure, rows, cols)
        if gs is None:
            gs = GridSpec(rows, cols, figure=figure)
        
        # 如果编号是一个包含两个整数的元组，则解析为起始和结束编号
        if isinstance(num, tuple) and len(num) == 2:
            if not all(isinstance(n, Integral) for n in num):
                # 如果元组内不全为整数，则引发 ValueError 异常
                raise ValueError(
                    f"Subplot specifier tuple must contain integers, not {num}"
                )
            i, j = num
        else:
            # 否则，将编号直接视为行列合并的编号，验证其有效性
            if not isinstance(num, Integral) or num < 1 or num > rows*cols:
                raise ValueError(
                    f"num must be an integer with 1 <= num <= {rows*cols}, "
                    f"not {num!r}"
                )
            i = j = num
        
        # 返回从 gridspec 中获取的 SubplotSpec 对象
        return gs[i-1:j]

    # num2 is a property only to handle the case where it is None and someone
    # mutates num1.

    @property
    def num2(self):
        # 返回 num2 属性值，如果 _num2 为 None，则返回 num1
        return self.num1 if self._num2 is None else self._num2

    @num2.setter
    def num2(self, value):
        # 设置 num2 属性值
        self._num2 = value

    def get_gridspec(self):
        # 返回所属的 gridspec 对象
        return self._gridspec
    def get_geometry(self):
        """
        Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.

        The indices *start* and *stop* define the range of the subplot within
        the `GridSpec`. *stop* is inclusive (i.e. for a single cell
        ``start == stop``).
        """
        # 获取当前子图的网格布局对象
        gridspec = self.get_gridspec()
        # 获取网格布局的行数和列数
        rows, cols = gridspec.get_geometry()
        # 返回子图的行数、列数、起始索引和结束索引
        return rows, cols, self.num1, self.num2

    @property
    def rowspan(self):
        """The rows spanned by this subplot, as a `range` object."""
        # 获取网格布局的列数
        ncols = self.get_gridspec().ncols
        # 计算当前子图跨越的行范围
        return range(self.num1 // ncols, self.num2 // ncols + 1)

    @property
    def colspan(self):
        """The columns spanned by this subplot, as a `range` object."""
        # 获取网格布局的列数
        ncols = self.get_gridspec().ncols
        # 确保列索引的顺序正确，使得范围的定义是合理的
        c1, c2 = sorted([self.num1 % ncols, self.num2 % ncols])
        # 计算当前子图跨越的列范围
        return range(c1, c2 + 1)

    def is_first_row(self):
        # 判断当前子图是否位于第一行
        return self.rowspan.start == 0

    def is_last_row(self):
        # 判断当前子图是否位于最后一行
        return self.rowspan.stop == self.get_gridspec().nrows

    def is_first_col(self):
        # 判断当前子图是否位于第一列
        return self.colspan.start == 0

    def is_last_col(self):
        # 判断当前子图是否位于最后一列
        return self.colspan.stop == self.get_gridspec().ncols

    def get_position(self, figure):
        """
        Update the subplot position from ``figure.subplotpars``.
        """
        # 获取当前子图的网格布局对象
        gridspec = self.get_gridspec()
        # 获取网格布局的行数和列数
        nrows, ncols = gridspec.get_geometry()
        # 根据子图编号计算其在网格中的行列索引
        rows, cols = np.unravel_index([self.num1, self.num2], (nrows, ncols))
        # 获取子图在 figure 中的位置信息
        fig_bottoms, fig_tops, fig_lefts, fig_rights = \
            gridspec.get_grid_positions(figure)
        # 计算子图在 figure 中的四个边界位置
        fig_bottom = fig_bottoms[rows].min()
        fig_top = fig_tops[rows].max()
        fig_left = fig_lefts[cols].min()
        fig_right = fig_rights[cols].max()
        # 返回子图在 figure 中的位置信息的边界框对象
        return Bbox.from_extents(fig_left, fig_bottom, fig_right, fig_top)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
        # 获取当前子图的网格布局对象
        gridspec = self.get_gridspec()
        # 若网格布局对象具有获取最顶层子图规格的方法，则调用该方法
        if hasattr(gridspec, "get_topmost_subplotspec"):
            return gridspec.get_topmost_subplotspec()
        else:
            # 否则返回当前子图规格对象本身
            return self

    def __eq__(self, other):
        """
        Two SubplotSpecs are considered equal if they refer to the same
        position(s) in the same `GridSpec`.
        """
        # 比较两个子图规格对象是否相等，要求它们具有相同的网格布局和子图位置索引
        return ((self._gridspec, self.num1, self.num2)
                == (getattr(other, "_gridspec", object()),
                    getattr(other, "num1", object()),
                    getattr(other, "num2", object())))

    def __hash__(self):
        # 返回子图规格对象的哈希值，用于在集合中进行比较和查找
        return hash((self._gridspec, self.num1, self.num2))
    # 创建一个在当前 subplot 内的 GridSpec

    """
    Create a GridSpec within this subplot.

    The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as
    a parent.

    Parameters
    ----------
    nrows : int
        Number of rows in grid.

    ncols : int
        Number of columns in grid.

    Returns
    -------
    `.GridSpecFromSubplotSpec`

    Other Parameters
    ----------------
    **kwargs
        All other parameters are passed to `.GridSpecFromSubplotSpec`.

    See Also
    --------
    matplotlib.pyplot.subplots

    Examples
    --------
    Adding three subplots in the space occupied by a single subplot::

        fig = plt.figure()
        gs0 = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs0[0])
        ax2 = fig.add_subplot(gs0[1])
        gssub = gs0[2].subgridspec(1, 3)
        for i in range(3):
            fig.add_subplot(gssub[0, i])
    """
    # 返回一个由当前 subplot 派生的 GridSpecFromSubplotSpec 对象
    return GridSpecFromSubplotSpec(nrows, ncols, self, **kwargs)
class SubplotParams:
    """
    Parameters defining the positioning of a subplots grid in a figure.
    """

    def __init__(self, left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None):
        """
        Defaults are given by :rc:`figure.subplot.[name]`.

        Parameters
        ----------
        left : float
            The position of the left edge of the subplots,
            as a fraction of the figure width.
        right : float
            The position of the right edge of the subplots,
            as a fraction of the figure width.
        bottom : float
            The position of the bottom edge of the subplots,
            as a fraction of the figure height.
        top : float
            The position of the top edge of the subplots,
            as a fraction of the figure height.
        wspace : float
            The width of the padding between subplots,
            as a fraction of the average Axes width.
        hspace : float
            The height of the padding between subplots,
            as a fraction of the average Axes height.
        """
        # 使用 matplotlib 的默认配置参数来初始化实例变量
        for key in ["left", "bottom", "right", "top", "wspace", "hspace"]:
            setattr(self, key, mpl.rcParams[f"figure.subplot.{key}"])
        # 调用 update 方法来更新参数
        self.update(left, bottom, right, top, wspace, hspace)

    def update(self, left=None, bottom=None, right=None, top=None,
               wspace=None, hspace=None):
        """
        Update the dimensions of the passed parameters. *None* means unchanged.
        """
        # 检查左边界不能大于等于右边界，若条件不满足则抛出 ValueError 异常
        if ((left if left is not None else self.left)
                >= (right if right is not None else self.right)):
            raise ValueError('left cannot be >= right')
        # 检查底边界不能大于等于顶边界，若条件不满足则抛出 ValueError 异常
        if ((bottom if bottom is not None else self.bottom)
                >= (top if top is not None else self.top)):
            raise ValueError('bottom cannot be >= top')
        # 更新实例变量的参数值，若参数非 None 则更新对应的实例变量
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if bottom is not None:
            self.bottom = bottom
        if top is not None:
            self.top = top
        if wspace is not None:
            self.wspace = wspace
        if hspace is not None:
            self.hspace = hspace
```