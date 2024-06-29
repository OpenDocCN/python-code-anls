# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\axes_divider.py`

```py
        """
        Helper classes to adjust the positions of multiple axes at drawing time.
        """

        import functools  # 导入 functools 模块

        import numpy as np  # 导入 NumPy 库，使用 np 别名

        import matplotlib as mpl  # 导入 Matplotlib 库并使用 mpl 别名
        from matplotlib import _api  # 从 Matplotlib 中导入 _api
        from matplotlib.gridspec import SubplotSpec  # 从 Matplotlib 中导入 SubplotSpec 类
        import matplotlib.transforms as mtransforms  # 导入 Matplotlib 的 transforms 模块
        from . import axes_size as Size  # 从当前包中导入 axes_size 模块，并使用 Size 别名

        
        class Divider:
            """
            An Axes positioning class.

            The divider is initialized with lists of horizontal and vertical sizes
            (:mod:`mpl_toolkits.axes_grid1.axes_size`) based on which a given
            rectangular area will be divided.

            The `new_locator` method then creates a callable object
            that can be used as the *axes_locator* of the axes.
            """

            def __init__(self, fig, pos, horizontal, vertical,
                         aspect=None, anchor="C"):
                """
                Parameters
                ----------
                fig : Figure
                    The figure object.
                pos : tuple of 4 floats
                    Position of the rectangle that will be divided.
                horizontal : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
                    Sizes for horizontal division.
                vertical : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
                    Sizes for vertical division.
                aspect : bool, optional
                    Whether overall rectangular area is reduced so that the relative
                    part of the horizontal and vertical scales have the same scale.
                anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', \
    'NW', 'W'}, default: 'C'
                    Placement of the reduced rectangle, when *aspect* is True.
                """
                self._fig = fig  # 初始化 figure 对象
                self._pos = pos  # 初始化矩形区域的位置
                self._horizontal = horizontal  # 初始化水平方向的尺寸列表
                self._vertical = vertical  # 初始化垂直方向的尺寸列表
                self._anchor = anchor  # 初始化锚点位置
                self.set_anchor(anchor)  # 调用 set_anchor 方法设置锚点位置
                self._aspect = aspect  # 初始化是否保持纵横比
                self._xrefindex = 0  # 初始化水平参考索引
                self._yrefindex = 0  # 初始化垂直参考索引
                self._locator = None  # 初始化定位器为 None

            def get_horizontal_sizes(self, renderer):
                """
                Returns the sizes of horizontal divisions.

                Parameters
                ----------
                renderer : RendererBase
                    The renderer instance.

                Returns
                -------
                ndarray
                    Array of sizes of horizontal divisions.
                """
                return np.array([s.get_size(renderer) for s in self.get_horizontal()])

            def get_vertical_sizes(self, renderer):
                """
                Returns the sizes of vertical divisions.

                Parameters
                ----------
                renderer : RendererBase
                    The renderer instance.

                Returns
                -------
                ndarray
                    Array of sizes of vertical divisions.
                """
                return np.array([s.get_size(renderer) for s in self.get_vertical()])

            def set_position(self, pos):
                """
                Set the position of the rectangle.

                Parameters
                ----------
                pos : tuple of 4 floats
                    position of the rectangle that will be divided
                """
                self._pos = pos  # 设置矩形的位置

            def get_position(self):
                """
                Returns the position of the rectangle.

                Returns
                -------
                tuple
                    Position of the rectangle.
                """
                return self._pos  # 返回矩形的位置

            def set_anchor(self, anchor):
                """
                Sets the anchor point for reduced rectangle positioning.

                Parameters
                ----------
                anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', \
    'NW', 'W'}
                    The anchor point specification.
                """
                self._anchor = anchor  # 设置锚点位置
        ```
    @staticmethod
    def _calc_k(sizes, total):
        # sizes 是一个 (n, 2) 的数组，包含了 (rel_size, abs_size)；该方法计算 k 因子，
        # 使得 sum(rel_size * k + abs_size) == total。
        rel_sum, abs_sum = sizes.sum(0)
        return (total - abs_sum) / rel_sum if rel_sum else 0

    @staticmethod
    def _calc_offsets(sizes, k):
        # 对 (n, 2) 大小的数组 sizes 应用 k 因子，其中包含了 (rel_size, abs_size)；
        # 返回累积偏移位置的结果。
        return np.cumsum([0, *(sizes @ [k, 1])])
    def new_locator(self, nx, ny, nx1=None, ny1=None):
        """
        Return an axes locator callable for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise, location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        """
        # 如果 nx1 未指定，则默认为 nx+1
        if nx1 is None:
            nx1 = nx + 1
        # 如果 ny1 未指定，则默认为 ny+1
        if ny1 is None:
            ny1 = ny + 1
        
        # append_size("left") 在水平尺寸列表的开头添加一个新的尺寸；
        # 这种移动将例如 new_locator(nx=2, ...) 转换为实际上的 new_locator(nx=3, ...)。
        # 考虑到这一点，我们记录 nx-self._xrefindex，其中 _xrefindex 每次
        # append_size("left") 都会加1。在 _locate 函数中，当计算实际轴位置时，
        # 我们将 self._xrefindex 重新加回到 nx 中。y 坐标也是同理。
        xref = self._xrefindex
        yref = self._yrefindex
        
        # 创建一个 functools.partial 对象，部分应用 _locate 方法，
        # 用调整后的坐标参数调用它
        locator = functools.partial(
            self._locate, nx - xref, ny - yref, nx1 - xref, ny1 - yref)
        
        # 将 get_subplotspec 方法赋值给 locator 对象，以便后续调用
        locator.get_subplotspec = self.get_subplotspec
        
        # 返回创建的定位器对象
        return locator

    @_api.deprecated(
        "3.8", alternative="divider.new_locator(...)(ax, renderer)")
    def locate(self, nx, ny, nx1=None, ny1=None, axes=None, renderer=None):
        """
        Implementation of ``divider.new_locator().__call__``.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the cell. When *nx1* is
            None, a single *nx*-th column is specified. Otherwise, the
            location of columns spanning between *nx* to *nx1* (but excluding
            *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        axes
        renderer
        """
        # 获取当前对象的 x 和 y 参考索引
        xref = self._xrefindex
        yref = self._yrefindex
        
        # 调用 _locate 方法，传入调整后的坐标参数，及其他可能的参数
        return self._locate(
            nx - xref, (nx + 1 if nx1 is None else nx1) - xref,
            ny - yref, (ny + 1 if ny1 is None else ny1) - yref,
            axes, renderer)
    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        """
        Implementation of ``divider.new_locator().__call__``.

        The axes locator callable returned by ``new_locator()`` is created as
        a `functools.partial` of this method with *nx*, *ny*, *nx1*, and *ny1*
        specifying the requested cell.
        """
        # 将索引偏移量加到所请求的行和列上
        nx += self._xrefindex
        nx1 += self._xrefindex
        ny += self._yrefindex
        ny1 += self._yrefindex

        # 获取图形的宽度和高度（单位为英寸），并计算相对位置和大小
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)

        # 获取水平和垂直方向的尺寸信息，并计算比例因子 k_h 和 k_v
        hsizes = self.get_horizontal_sizes(renderer)
        vsizes = self.get_vertical_sizes(renderer)
        k_h = self._calc_k(hsizes, fig_w * w)
        k_v = self._calc_k(vsizes, fig_h * h)

        # 如果设置了固定比例，则根据最小比例因子 k 计算偏移量和新的位置和大小
        if self.get_aspect():
            k = min(k_h, k_v)
            ox = self._calc_offsets(hsizes, k)
            oy = self._calc_offsets(vsizes, k)

            # 计算基于固定比例调整后的新边界框
            ww = (ox[-1] - ox[0]) / fig_w
            hh = (oy[-1] - oy[0]) / fig_h
            pb = mtransforms.Bbox.from_bounds(x, y, w, h)
            pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
            x0, y0 = pb1.anchored(self.get_anchor(), pb).p0

        else:
            # 如果没有固定比例，直接使用计算出的位置
            ox = self._calc_offsets(hsizes, k_h)
            oy = self._calc_offsets(vsizes, k_v)
            x0, y0 = x, y

        # 处理可能为 None 的索引，将其设为默认值 -1
        if nx1 is None:
            nx1 = -1
        if ny1 is None:
            ny1 = -1

        # 根据计算出的位置偏移和尺寸，生成最终的边界框对象
        x1, w1 = x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w
        y1, h1 = y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h

        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)

    def append_size(self, position, size):
        """
        Append a size value to either horizontal or vertical size list,
        depending on the given position.

        Parameters
        ----------
        position : str
            One of "left", "right", "bottom", "top" indicating where to append the size.
        size : float
            The size value to append.

        Raises
        ------
        ValueError
            If the position is not one of the specified strings.
        """
        _api.check_in_list(["left", "right", "bottom", "top"],
                           position=position)
        if position == "left":
            self._horizontal.insert(0, size)
            self._xrefindex += 1
        elif position == "right":
            self._horizontal.append(size)
        elif position == "bottom":
            self._vertical.insert(0, size)
            self._yrefindex += 1
        else:  # 'top'
            self._vertical.append(size)

    def add_auto_adjustable_area(self, use_axes, pad=0.1, adjust_dirs=None):
        """
        Add auto-adjustable padding around *use_axes* to take their decorations
        (title, labels, ticks, ticklabels) into account during layout.

        Parameters
        ----------
        use_axes : `~matplotlib.axes.Axes` or list of `~matplotlib.axes.Axes`
            The Axes whose decorations are taken into account.
        pad : float, default: 0.1
            Additional padding in inches.
        adjust_dirs : list of {"left", "right", "bottom", "top"}, optional
            The sides where padding is added; defaults to all four sides.
        """
        # 如果没有指定调整方向，默认为所有四个方向
        if adjust_dirs is None:
            adjust_dirs = ["left", "right", "bottom", "top"]
        
        # 遍历指定的调整方向列表，为每个方向添加尺寸信息
        for d in adjust_dirs:
            self.append_size(d, Size._AxesDecorationsSize(use_axes, d) + pad)
# 使用 @_api.deprecated 装饰器标记此类已经在版本 3.8 中被废弃
@_api.deprecated("3.8")
class AxesLocator:
    """
    A callable object which returns the position and size of a given
    `.AxesDivider` cell.
    """

    def __init__(self, axes_divider, nx, ny, nx1=None, ny1=None):
        """
        Parameters
        ----------
        axes_divider : `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
            实例化时传入的 AxesDivider 对象，用于定位子图单元格的位置和大小
        nx, nx1 : int
            表示单元格列位置的整数。当 nx1 为 None 时，指定单个 nx 列位置；
            否则，指定从 nx 到 nx1（不包括 nx1 ）的列位置范围。
        ny, ny1 : int
            与 nx 和 nx1 相同，但用于行位置。
        """
        self._axes_divider = axes_divider

        _xrefindex = axes_divider._xrefindex
        _yrefindex = axes_divider._yrefindex

        # 计算实际使用的列和行位置，考虑到 xrefindex 和 yrefindex 的偏移
        self._nx, self._ny = nx - _xrefindex, ny - _yrefindex

        if nx1 is None:
            nx1 = len(self._axes_divider)
        if ny1 is None:
            ny1 = len(self._axes_divider[0])

        # 计算实际使用的结束列和行位置，同样考虑到 xrefindex 和 yrefindex 的偏移
        self._nx1 = nx1 - _xrefindex
        self._ny1 = ny1 - _yrefindex

    def __call__(self, axes, renderer):
        """
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            要定位的子图对象
        renderer : matplotlib.backend_bases.RendererBase
            渲染器对象，用于渲染子图

        Returns
        -------
        tuple
            返回定位的子图单元格的位置和大小信息
        """
        _xrefindex = self._axes_divider._xrefindex
        _yrefindex = self._axes_divider._yrefindex

        # 调用 AxesDivider 对象的 locate 方法，定位子图单元格的位置和大小
        return self._axes_divider.locate(self._nx + _xrefindex,
                                         self._ny + _yrefindex,
                                         self._nx1 + _xrefindex,
                                         self._ny1 + _yrefindex,
                                         axes,
                                         renderer)

    def get_subplotspec(self):
        """
        Returns
        -------
        matplotlib.gridspec.SubplotSpec
            返回与此 AxesLocator 对象关联的子图规格对象
        """
        return self._axes_divider.get_subplotspec()


class SubplotDivider(Divider):
    """
    The Divider class whose rectangle area is specified as a subplot geometry.
    """
    def __init__(self, fig, *args, horizontal=None, vertical=None,
                 aspect=None, anchor='C'):
        """
        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            用于创建子图的 matplotlib 图形对象

        *args : tuple (*nrows*, *ncols*, *index*) or int
            子图数组在图形中的维度是 ``(nrows, ncols)``, *index* 是正在创建的子图的索引。
            *index* 从左上角开始为1逐渐增加到右边。

            如果 *nrows*, *ncols* 和 *index* 都是单个数字，那么可以将 *args* 作为一个3位数传递（例如234表示(2, 3, 4))。

        horizontal : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`, optional
            水平方向分割的大小列表。

        vertical : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`, optional
            垂直方向分割的大小列表。

        aspect : bool, optional
            是否减少整体矩形区域，使水平和垂直比例的相对部分具有相同的比例。

        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', \
            'NW', 'W'}
            子图的锚定位置，可以是具体的浮点数坐标或预定义的位置字符。
        """
    """
    Divider based on the preexisting axes.
    """

    def __init__(self, axes, xref=None, yref=None):
        """
        Parameters
        ----------
        axes : :class:`~matplotlib.axes.Axes`
            Existing matplotlib axes object to divide.
        xref
            Reference size along the x-axis.
        yref
            Reference size along the y-axis.
        """
        self._axes = axes
        # Determine the reference size along the x-axis based on the provided xref or default to AxesX.
        if xref is None:
            self._xref = Size.AxesX(axes)
        else:
            self._xref = xref
        # Determine the reference size along the y-axis based on the provided yref or default to AxesY.
        if yref is None:
            self._yref = Size.AxesY(axes)
        else:
            self._yref = yref

        # Initialize the Divider superclass with appropriate parameters.
        super().__init__(fig=axes.get_figure(), pos=None,
                         horizontal=[self._xref], vertical=[self._yref],
                         aspect=None, anchor="C")

    def _get_new_axes(self, *, axes_class=None, **kwargs):
        """
        Create a new axes instance based on the current axes.

        Parameters
        ----------
        axes_class : class, optional
            Class of the new axes to create.
        **kwargs
            Additional keyword arguments to pass to the new axes constructor.

        Returns
        -------
        axes_class
            New axes instance based on the current axes.
        """
        axes = self._axes
        # If axes_class is None, use the same class as the current axes.
        if axes_class is None:
            axes_class = type(axes)
        # Create and return a new axes instance with the same figure and position.
        return axes_class(axes.get_figure(), axes.get_position(original=True),
                          **kwargs)

    def new_horizontal(self, size, pad=None, pack_start=False, **kwargs):
        """
        Helper method to create a new horizontal axes.

        Parameters
        ----------
        size
            Size of the new axes.
        pad
            Padding space between the new axes and existing axes.
        pack_start : bool, optional
            If True, the new axes will be packed to the left; otherwise, to the right.
        **kwargs
            Additional keyword arguments passed to the new axes creation.

        Returns
        -------
        ax
            Newly created axes instance.
        """
        # Determine the padding based on default figure settings or the provided pad value.
        if pad is None:
            pad = mpl.rcParams["figure.subplot.wspace"] * self._xref
        # Determine the position for appending based on pack_start.
        pos = "left" if pack_start else "right"
        # Convert pad and size to Size objects if they are not already.
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._xref)
            self.append_size(pos, pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._xref)
        self.append_size(pos, size)
        # Determine the locator for the new axes and create it using _get_new_axes method.
        locator = self.new_locator(
            nx=0 if pack_start else len(self._horizontal) - 1,
            ny=self._yrefindex)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax
    def new_vertical(self, size, pad=None, pack_start=False, **kwargs):
        """
        Helper method for ``append_axes("top")`` and ``append_axes("bottom")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
        # 如果 pad 参数未指定，则使用默认的间距设置
        if pad is None:
            pad = mpl.rcParams["figure.subplot.hspace"] * self._yref
        # 根据 pack_start 参数确定位置是 "bottom" 还是 "top"
        pos = "bottom" if pack_start else "top"
        # 如果 pad 存在，将其转换为相应的 Size 对象，并添加到指定位置
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._yref)
            self.append_size(pos, pad)
        # 将 size 转换为相应的 Size 对象，并添加到指定位置
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._yref)
        self.append_size(pos, size)
        # 根据当前对象的状态和参数创建一个新的定位器
        locator = self.new_locator(
            nx=self._xrefindex,
            ny=0 if pack_start else len(self._vertical) - 1)
        # 获取一个新的子图对象，并设置其定位器
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        # 返回创建的新子图对象
        return ax

    def append_axes(self, position, size, pad=None, *, axes_class=None,
                    **kwargs):
        """
        Add a new axes on a given side of the main axes.

        Parameters
        ----------
        position : {"left", "right", "bottom", "top"}
            Where the new axes is positioned relative to the main axes.
        size : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            The axes width or height.  float or str arguments are interpreted
            as ``axes_size.from_any(size, AxesX(<main_axes>))`` for left or
            right axes, and likewise with ``AxesY`` for bottom or top axes.
        pad : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            Padding between the axes.  float or str arguments are interpreted
            as for *size*.  Defaults to :rc:`figure.subplot.wspace` times the
            main Axes width (left or right axes) or :rc:`figure.subplot.hspace`
            times the main Axes height (bottom or top axes).
        axes_class : subclass type of `~.axes.Axes`, optional
            The type of the new axes.  Defaults to the type of the main axes.
        **kwargs
            All extra keywords arguments are passed to the created axes.
        """
        # 根据 position 参数选择创建子图的方法和起始位置
        create_axes, pack_start = _api.check_getitem({
            "left": (self.new_horizontal, True),
            "right": (self.new_horizontal, False),
            "bottom": (self.new_vertical, True),
            "top": (self.new_vertical, False),
        }, position=position)
        # 使用选定的方法创建子图对象
        ax = create_axes(
            size, pad, pack_start=pack_start, axes_class=axes_class, **kwargs)
        # 将创建的子图对象添加到主图对象中
        self._fig.add_axes(ax)
        # 返回创建的子图对象
        return ax

    def get_aspect(self):
        # 如果没有显式设置 aspect，则根据当前子图对象的状态获取其 aspect
        if self._aspect is None:
            aspect = self._axes.get_aspect()
            # 如果获取的 aspect 为 "auto"，返回 False；否则返回 True
            if aspect == "auto":
                return False
            else:
                return True
        else:
            # 如果已经显式设置了 aspect，则直接返回其值
            return self._aspect
    # 获取对象的位置信息
    def get_position(self):
        # 如果对象的位置信息未设置 (_pos 是 None)，则获取原始坐标轴的位置边界框
        if self._pos is None:
            bbox = self._axes.get_position(original=True)
            return bbox.bounds
        else:
            # 否则直接返回已设置的位置信息
            return self._pos

    # 获取对象的锚点信息
    def get_anchor(self):
        # 如果对象的锚点信息未设置 (_anchor 是 None)，则获取所属坐标轴的锚点信息
        if self._anchor is None:
            return self._axes.get_anchor()
        else:
            # 否则直接返回已设置的锚点信息
            return self._anchor

    # 获取对象所属的子图规格对象
    def get_subplotspec(self):
        # 直接返回对象所属坐标轴的子图规格对象
        return self._axes.get_subplotspec()
# Helper for HBoxDivider/VBoxDivider.
# The variable names are written for a horizontal layout, but the calculations
# work identically for vertical layouts.
# 用于 HBoxDivider/VBoxDivider 的辅助函数。
# 变量名是为水平布局编写的，但计算对垂直布局同样适用。

def _locate(x, y, w, h, summed_widths, equal_heights, fig_w, fig_h, anchor):
    # 定位函数，计算布局位置和大小

    total_width = fig_w * w
    max_height = fig_h * h

    # Determine the k factors.
    n = len(equal_heights)
    eq_rels, eq_abss = equal_heights.T
    sm_rels, sm_abss = summed_widths.T
    A = np.diag([*eq_rels, 0])
    A[:n, -1] = -1
    A[-1, :-1] = sm_rels
    B = [*(-eq_abss), total_width - sm_abss.sum()]
    # A @ K = B: This finds factors {k_0, ..., k_{N-1}, H} so that
    #   eq_rel_i * k_i + eq_abs_i = H for all i: all axes have the same height
    #   sum(sm_rel_i * k_i + sm_abs_i) = total_width: fixed total width
    # (foo_rel_i * k_i + foo_abs_i will end up being the size of foo.)
    *karray, height = np.linalg.solve(A, B)
    if height > max_height:  # Additionally, upper-bound the height.
        karray = (max_height - eq_abss) / eq_rels

    # Compute the offsets corresponding to these factors.
    ox = np.cumsum([0, *(sm_rels * karray + sm_abss)])
    ww = (ox[-1] - ox[0]) / fig_w
    h0_rel, h0_abs = equal_heights[0]
    hh = (karray[0]*h0_rel + h0_abs) / fig_h
    pb = mtransforms.Bbox.from_bounds(x, y, w, h)
    pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
    x0, y0 = pb1.anchored(anchor, pb).p0

    return x0, y0, ox, hh

# Class for laying out axes horizontally with equal heights
class HBoxDivider(SubplotDivider):
    """
    A `.SubplotDivider` for laying out axes horizontally, while ensuring that
    they have equal heights.

    Examples
    --------
    .. plot:: gallery/axes_grid1/demo_axes_hbox_divider.py
    """
    
    # Create a new locator for the specified cell
    def new_locator(self, nx, nx1=None):
        """
        Create an axes locator callable for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise, location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        """
        return super().new_locator(nx, 0, nx1, 0)

    # Locate the position and size of the axes
    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        # docstring inherited
        nx += self._xrefindex
        nx1 += self._xrefindex
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_ws = self.get_horizontal_sizes(renderer)
        equal_hs = self.get_vertical_sizes(renderer)
        x0, y0, ox, hh = _locate(
            x, y, w, h, summed_ws, equal_hs, fig_w, fig_h, self.get_anchor())
        if nx1 is None:
            nx1 = -1
        x1, w1 = x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w
        y1, h1 = y0, hh
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)

# Class for laying out axes vertically with equal widths
class VBoxDivider(SubplotDivider):
    """
    A `.SubplotDivider` for laying out axes vertically, while ensuring that
    they have equal widths.
    """
    """
    定义一个新的定位器方法，用于创建指定单元格的坐标定位器。

    Parameters
    ----------
    ny, ny1 : int
        整数，指定单元格的行位置。当 *ny1* 为 None 时，指定第 *ny* 行。否则，指定从 *ny* 到 *ny1*（但不包括 *ny1* 行）的行的位置。
    """
    return super().new_locator(0, ny, 0, ny1)

def _locate(self, nx, ny, nx1, ny1, axes, renderer):
    # 继承的文档字符串
    ny += self._yrefindex  # 将 ny 增加 _yrefindex，调整参考的行索引
    ny1 += self._yrefindex  # 将 ny1 增加 _yrefindex，调整参考的行索引
    fig_w, fig_h = self._fig.bbox.size / self._fig.dpi  # 获取图形的宽度和高度
    x, y, w, h = self.get_position_runtime(axes, renderer)  # 获取运行时位置信息
    summed_hs = self.get_vertical_sizes(renderer)  # 获取垂直尺寸的总和
    equal_ws = self.get_horizontal_sizes(renderer)  # 获取水平尺寸的相等值
    y0, x0, oy, ww = _locate(
        y, x, h, w, summed_hs, equal_ws, fig_h, fig_w, self.get_anchor())
    if ny1 is None:
        ny1 = -1  # 如果 ny1 为 None，设置为 -1
    x1, w1 = x0, ww  # 设置 x1 和 w1 为计算得到的 x0 和 ww
    y1, h1 = y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h  # 计算 y1 和 h1
    return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)
def make_axes_locatable(axes):
    # 创建一个AxesDivider对象，用于处理给定axes的位置分隔
    divider = AxesDivider(axes)
    # 创建一个新的定位器，将其放置在axes的左下角
    locator = divider.new_locator(nx=0, ny=0)
    # 将新的定位器应用到给定的axes上
    axes.set_axes_locator(locator)

    # 返回创建的divider对象，用于后续的自动调整区域布局
    return divider


def make_axes_area_auto_adjustable(
        ax, use_axes=None, pad=0.1, adjust_dirs=None):
    """
    Add auto-adjustable padding around *ax* to take its decorations (title,
    labels, ticks, ticklabels) into account during layout, using
    `.Divider.add_auto_adjustable_area`.

    By default, padding is determined from the decorations of *ax*.
    Pass *use_axes* to consider the decorations of other Axes instead.
    """
    # 如果adjust_dirs为None，则默认调整left, right, bottom, top四个方向
    if adjust_dirs is None:
        adjust_dirs = ["left", "right", "bottom", "top"]
    
    # 使用make_axes_locatable函数创建一个divider对象
    divider = make_axes_locatable(ax)
    
    # 如果use_axes为None，则默认使用传入的ax作为use_axes
    if use_axes is None:
        use_axes = ax
    
    # 向divider对象添加一个自动可调整区域，考虑ax或者use_axes的装饰物（标题、标签、刻度、刻度标签）
    divider.add_auto_adjustable_area(use_axes=use_axes, pad=pad,
                                     adjust_dirs=adjust_dirs)
```