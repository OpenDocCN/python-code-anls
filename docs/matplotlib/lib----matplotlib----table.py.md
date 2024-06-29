# `D:\src\scipysrc\matplotlib\lib\matplotlib\table.py`

```
# Original code by:
#    John Gill <jng@europe.renre.com>
#    Copyright 2004 John Gill and John Hunter
#
# Subsequent changes:
#    The Matplotlib development team
#    Copyright The Matplotlib development team

"""
Tables drawing.

.. note::
    The table implementation in Matplotlib is lightly maintained. For a more
    featureful table implementation, you may wish to try `blume
    <https://github.com/swfiua/blume>`_.

Use the factory function `~matplotlib.table.table` to create a ready-made
table from texts. If you need more control, use the `.Table` class and its
methods.

The table consists of a grid of cells, which are indexed by (row, column).
The cell (0, 0) is positioned at the top left.

Thanks to John Gill for providing the class and table.
"""

import numpy as np

from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path


class Cell(Rectangle):
    """
    A cell is a `.Rectangle` with some associated `.Text`.

    As a user, you'll most likely not creates cells yourself. Instead, you
    should use either the `~matplotlib.table.table` factory function or
    `.Table.add_cell`.
    """

    PAD = 0.1
    """Padding between text and rectangle."""

    _edges = 'BRTL'
    _edge_aliases = {'open':         '',
                     'closed':       _edges,  # default
                     'horizontal':   'BT',
                     'vertical':     'RL'
                     }

    def __init__(self, xy, width, height, *,
                 edgecolor='k', facecolor='w',
                 fill=True,
                 text='',
                 loc='right',
                 fontproperties=None,
                 visible_edges='closed',
                 ):
        """
        Parameters
        ----------
        xy : 2-tuple
            The position of the bottom left corner of the cell.
        width : float
            The cell width.
        height : float
            The cell height.
        edgecolor : :mpltype:`color`, default: 'k'
            The color of the cell border.
        facecolor : :mpltype:`color`, default: 'w'
            The cell facecolor.
        fill : bool, default: True
            Whether the cell background is filled.
        text : str, optional
            The cell text.
        loc : {'right', 'center', 'left'}
            The alignment of the text within the cell.
        fontproperties : dict, optional
            A dict defining the font properties of the text. Supported keys and
            values are the keyword arguments accepted by `.FontProperties`.
        visible_edges : {'closed', 'open', 'horizontal', 'vertical'} or \
                        'B', 'R', 'T', 'L' or any combination thereof
            Determines which edges of the cell rectangle are visible.
        """
        # Initialize the Cell object as a Rectangle with associated Text
        super().__init__(xy, width, height, edgecolor=edgecolor, facecolor=facecolor, fill=fill)
        # Store the text associated with the cell
        self.text = text
        # Text alignment within the cell
        self.loc = loc
        # Font properties for the text
        self.fontproperties = fontproperties
        # Determine the visible edges of the cell rectangle
        self._visible_edges = self._edge_aliases.get(visible_edges, visible_edges)

        # Set the padding between text and rectangle
        self.PAD = 0.1

        # Validate and assign the visible edges based on the input
        if any(edge not in self._edges for edge in self._visible_edges):
            raise ValueError(f"Invalid visible_edges specified: {visible_edges}. Must be one or more of 'closed', 'open', 'horizontal', 'vertical' or 'B', 'R', 'T', 'L'.")

        # Initialize the associated Text object
        self._text = Text()

    def set_visible_edges(self, edges):
        """
        Set which edges of the cell rectangle are visible.

        Parameters
        ----------
        edges : {'closed', 'open', 'horizontal', 'vertical'} or \
                'B', 'R', 'T', 'L' or any combination thereof
            Determines which edges of the cell rectangle are visible.
        """
        self._visible_edges = self._edge_aliases.get(edges, edges)

        # Validate the input edges
        if any(edge not in self._edges for edge in self._visible_edges):
            raise ValueError(f"Invalid edges specified: {edges}. Must be one or more of 'closed', 'open', 'horizontal', 'vertical' or 'B', 'R', 'T', 'L'.")
    # 设置表格单元格的边缘显示方式，可选项包括子集 'BRTL'（底部、右侧、顶部、左侧的边缘将显示），或者 'open'（不绘制边缘），'closed'（绘制所有边缘），'horizontal'（仅绘制底部和顶部边缘），'vertical'（仅绘制右侧和左侧边缘）。
    # 这是一个类级别的注释，描述了参数 visible_edges 的功能和可选取值。
    """
    The cell edges to be drawn with a line: a substring of 'BRTL'
    (bottom, right, top, left), or one of 'open' (no edges drawn),
    'closed' (all edges drawn), 'horizontal' (bottom and top),
    'vertical' (right and left).
    """

    # 调用父类的初始化方法，设置单元格的基本属性
    super().__init__(xy, width=width, height=height, fill=fill,
                     edgecolor=edgecolor, facecolor=facecolor)
    # 设置是否裁剪，默认为不裁剪
    self.set_clip_on(False)
    # 设置单元格的可见边缘属性
    self.visible_edges = visible_edges

    # 创建文本对象，用于在单元格中显示文字
    self._loc = loc
    self._text = Text(x=xy[0], y=xy[1], clip_on=False,
                      text=text, fontproperties=fontproperties,
                      horizontalalignment=loc, verticalalignment='center')

# 以下两个方法是方法级别的注释，描述了它们的功能和作用
# 设置变换（transform）对象，用于控制单元格的变换
# 在版本 3.8 中，参数 'trans' 被重命名为 't'
"""
    @_api.rename_parameter("3.8", "trans", "t")
    def set_transform(self, t):
        super().set_transform(t)
        # the text does not get the transform!
        self.stale = True
"""

# 设置所属的图形对象
"""
    def set_figure(self, fig):
        super().set_figure(fig)
        self._text.set_figure(fig)
"""

# 获取单元格中的文本对象
"""
    def get_text(self):
        """Return the cell `.Text` instance."""
        return self._text
"""

# 设置文本的字体大小
"""
    def set_fontsize(self, size):
        """Set the text fontsize."""
        self._text.set_fontsize(size)
        self.stale = True
"""

# 获取文本的字体大小
"""
    def get_fontsize(self):
        """Return the cell fontsize."""
        return self._text.get_fontsize()
"""

# 自动调整字体大小，使文本适应单元格宽度
"""
    def auto_set_font_size(self, renderer):
        """Shrink font size until the text fits into the cell width."""
        fontsize = self.get_fontsize()
        required = self.get_required_width(renderer)
        while fontsize > 1 and required > self.get_width():
            fontsize -= 1
            self.set_fontsize(fontsize)
            required = self.get_required_width(renderer)

        return fontsize
"""

# 绘制单元格及其文本内容
"""
    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        # draw the rectangle
        super().draw(renderer)
        # position the text
        self._set_text_position(renderer)
        self._text.draw(renderer)
        self.stale = False
"""

# 设置文本的位置，确保文本在单元格中正确显示
"""
    def _set_text_position(self, renderer):
        """Set text up so it is drawn in the right place."""
        bbox = self.get_window_extent(renderer)
        # center vertically
        y = bbox.y0 + bbox.height / 2
        # position horizontally
        loc = self._text.get_horizontalalignment()
        if loc == 'center':
            x = bbox.x0 + bbox.width / 2
        elif loc == 'left':
            x = bbox.x0 + bbox.width * self.PAD
        else:  # right.
            x = bbox.x0 + bbox.width * (1 - self.PAD)
        self._text.set_position((x, y))
"""

# 获取文本在表格坐标系中的边界信息
"""
    def get_text_bounds(self, renderer):
        """
        Return the text bounds as *(x, y, width, height)* in table coordinates.
        """
        return (self._text.get_window_extent(renderer)
                .transformed(self.get_data_transform().inverted())
                .bounds)
    def get_required_width(self, renderer):
        """
        Return the minimal required width for the cell.

        Calculate the minimal required width for the cell based on text bounds
        obtained from the renderer.

        Parameters:
        - renderer: The renderer object used to obtain text bounds.

        Returns:
        - The minimal required width for the cell.
        """
        l, b, w, h = self.get_text_bounds(renderer)
        return w * (1.0 + (2.0 * self.PAD))

    @_docstring.dedent_interpd
    def set_text_props(self, **kwargs):
        """
        Update the text properties.

        Update the internal text properties of the object using provided keyword
        arguments.

        Valid keyword arguments are:
        %(Text:kwdoc)s
        """
        self._text._internal_update(kwargs)
        self.stale = True

    @property
    def visible_edges(self):
        """
        The cell edges to be drawn with a line.

        Property that defines the edges of the cell to be drawn with a line.
        Reading this property returns a substring of 'BRTL' (bottom, right,
        top, left').

        Returns:
        - A substring of 'BRTL' representing visible edges.
        """
        return self._visible_edges

    @visible_edges.setter
    def visible_edges(self, value):
        """
        Set the visible edges of the cell.

        Set the visible edges of the cell to the specified value. This can be a
        substring of 'BRTL' or one of {'open', 'closed', 'horizontal', 'vertical'}.

        Parameters:
        - value: The value specifying which edges to make visible.

        Raises:
        - ValueError: If the provided value is invalid or contains edges not in
          the allowed set.
        """
        if value is None:
            self._visible_edges = self._edges
        elif value in self._edge_aliases:
            self._visible_edges = self._edge_aliases[value]
        else:
            if any(edge not in self._edges for edge in value):
                raise ValueError('Invalid edge param {}, must only be one of '
                                 '{} or string of {}'.format(
                                     value,
                                     ", ".join(self._edge_aliases),
                                     ", ".join(self._edges)))
            self._visible_edges = value
        self.stale = True

    def get_path(self):
        """
        Return a `.Path` for the `.visible_edges`.

        Generate a `.Path` object based on the visible edges of the cell.

        Returns:
        - A `.Path` object representing the edges of the cell.
        """
        codes = [Path.MOVETO]
        codes.extend(
            Path.LINETO if edge in self._visible_edges else Path.MOVETO
            for edge in self._edges)
        if Path.MOVETO not in codes[1:]:  # All sides are visible
            codes[-1] = Path.CLOSEPOLY
        return Path(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            codes,
            readonly=True
            )
CustomCell = Cell  # Backcompat. alias.


class Table(Artist):
    """
    A table of cells.

    The table consists of a grid of cells, which are indexed by (row, column).

    For a simple table, you'll have a full grid of cells with indices from
    (0, 0) to (num_rows-1, num_cols-1), in which the cell (0, 0) is positioned
    at the top left. However, you can also add cells with negative indices.
    You don't have to add a cell to every grid position, so you can create
    tables that have holes.

    *Note*: You'll usually not create an empty table from scratch. Instead use
    `~matplotlib.table.table` to create a table from data.
    """
    codes = {'best': 0,
             'upper right':  1,  # default
             'upper left':   2,
             'lower left':   3,
             'lower right':  4,
             'center left':  5,
             'center right': 6,
             'lower center': 7,
             'upper center': 8,
             'center':       9,
             'top right':    10,
             'top left':     11,
             'bottom left':  12,
             'bottom right': 13,
             'right':        14,
             'left':         15,
             'top':          16,
             'bottom':       17,
             }
    """Possible values where to place the table relative to the Axes."""

    FONTSIZE = 10

    AXESPAD = 0.02
    """The border between the Axes and the table edge in Axes units."""

    def __init__(self, ax, loc=None, bbox=None, **kwargs):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` to plot the table into.
        loc : str, optional
            The position of the cell with respect to *ax*. This must be one of
            the `~.Table.codes`.
        bbox : `.Bbox` or [xmin, ymin, width, height], optional
            A bounding box to draw the table into. If this is not *None*, this
            overrides *loc*.

        Other Parameters
        ----------------
        **kwargs
            `.Artist` properties.
        """

        super().__init__()  # 调用父类的构造函数，初始化基类的状态

        if isinstance(loc, str):
            if loc not in self.codes:
                raise ValueError(
                    "Unrecognized location {!r}. Valid locations are\n\t{}"
                    .format(loc, '\n\t'.join(self.codes)))
            loc = self.codes[loc]  # 将位置字符串映射为对应的数字代码
        self.set_figure(ax.figure)  # 设置表格所属的图形对象
        self._axes = ax  # 存储传入的Axes对象
        self._loc = loc  # 存储位置信息
        self._bbox = bbox  # 存储边界框信息，用于绘制表格

        # use axes coords
        ax._unstale_viewLim()  # 刷新视图界限，确保最新的视图范围
        self.set_transform(ax.transAxes)  # 设置坐标变换为Axes的坐标变换

        self._cells = {}  # 初始化存储单元格的字典
        self._edges = None  # 边缘信息，暂时设为None
        self._autoColumns = []  # 自动列宽的列表
        self._autoFontsize = True  # 是否自动调整字体大小，默认为True
        self._internal_update(kwargs)  # 更新内部参数

        self.set_clip_on(False)  # 关闭裁剪功能，允许在表格外部绘制内容
    def add_cell(self, row, col, *args, **kwargs):
        """
        Create a cell and add it to the table.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        *args, **kwargs
            All other parameters are passed on to `Cell`.

        Returns
        -------
        `.Cell`
            The created cell.

        """
        # 定义单元格左上角位置为原点(0, 0)
        xy = (0, 0)
        # 创建一个新的单元格对象，并设置可见边缘和其他传递的参数
        cell = Cell(xy, visible_edges=self.edges, *args, **kwargs)
        # 将创建的单元格添加到指定的行和列位置
        self[row, col] = cell
        # 返回创建的单元格对象
        return cell

    def __setitem__(self, position, cell):
        """
        Set a custom cell in a given position.
        """
        # 检查传入的单元格是否为 Cell 类型的实例
        _api.check_isinstance(Cell, cell=cell)
        try:
            # 解析位置元组，获取行和列索引
            row, col = position[0], position[1]
        except Exception as err:
            # 如果位置不是长度为2的元组，抛出 Key 错误异常
            raise KeyError('Only tuples length 2 are accepted as '
                           'coordinates') from err
        # 设置单元格的图形属性、变换和裁剪状态
        cell.set_figure(self.figure)
        cell.set_transform(self.get_transform())
        cell.set_clip_on(False)
        # 在内部数据结构中保存单元格对象
        self._cells[row, col] = cell
        # 设置表格为过时状态，需要重新绘制
        self.stale = True

    def __getitem__(self, position):
        """Retrieve a custom cell from a given position."""
        # 返回指定位置的单元格对象
        return self._cells[position]

    @property
    def edges(self):
        """
        The default value of `~.Cell.visible_edges` for newly added
        cells using `.add_cell`.

        Notes
        -----
        This setting does currently only affect newly created cells using
        `.add_cell`.

        To change existing cells, you have to set their edges explicitly::

            for c in tab.get_celld().values():
                c.visible_edges = 'horizontal'

        """
        # 返回当前表格对象的可见边缘设置
        return self._edges

    @edges.setter
    def edges(self, value):
        # 设置表格对象的可见边缘设置，并标记表格为过时状态
        self._edges = value
        self.stale = True

    def _approx_text_height(self):
        # 计算大致的文本高度，基于字体大小和绘图区域的高度
        return (self.FONTSIZE / 72.0 * self.figure.dpi /
                self._axes.bbox.height * 1.2)

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        # 需要渲染器以便在鼠标事件中进行命中测试；假设使用最后一个渲染器
        if renderer is None:
            renderer = self.figure._get_renderer()
        if renderer is None:
            raise RuntimeError('No renderer defined')

        # 如果表格不可见，则直接返回
        if not self.get_visible():
            return
        # 打开一个表格渲染组
        renderer.open_group('table', gid=self.get_gid())
        # 更新单元格的位置信息
        self._update_positions(renderer)

        # 遍历并绘制所有单元格
        for key in sorted(self._cells):
            self._cells[key].draw(renderer)

        # 关闭表格渲染组
        renderer.close_group('table')
        # 标记表格为不需要重新绘制的状态
        self.stale = False
    def _get_grid_bbox(self, renderer):
        """
        Get a bbox, in axes coordinates for the cells.

        Only include those in the range (0, 0) to (maxRow, maxCol).
        """
        # Collect bounding boxes for cells within the specified grid range
        boxes = [cell.get_window_extent(renderer)
                 for (row, col), cell in self._cells.items()
                 if row >= 0 and col >= 0]
        # Calculate the union of all collected bounding boxes
        bbox = Bbox.union(boxes)
        # Transform the bbox to axes coordinates
        return bbox.transformed(self.get_transform().inverted())

    def contains(self, mouseevent):
        """
        Check if the mouse event is contained within any table cell.

        Returns a tuple indicating whether the mouse is inside the table and an empty dictionary.
        """
        # Check if the mouse event occurs on a different canvas
        if self._different_canvas(mouseevent):
            return False, {}
        # TODO: Implement functionality to find the cell index containing the mouse cursor
        # Get the renderer for the figure
        renderer = self.figure._get_renderer()
        if renderer is not None:
            # Collect bounding boxes for cells within the specified grid range
            boxes = [cell.get_window_extent(renderer)
                     for (row, col), cell in self._cells.items()
                     if row >= 0 and col >= 0]
            # Calculate the union of all collected bounding boxes
            bbox = Bbox.union(boxes)
            # Check if the mouse event coordinates are within the calculated bbox
            return bbox.contains(mouseevent.x, mouseevent.y), {}
        else:
            return False, {}

    def get_children(self):
        """Return the Artists contained by the table."""
        # Return a list of all cells contained in the table
        return list(self._cells.values())

    def get_window_extent(self, renderer=None):
        """
        Return the bounding box of the table in window (pixel) coordinates.

        Uses the provided renderer; if none is given, retrieves the renderer from the figure.
        """
        # If renderer is not provided, use the figure's renderer
        if renderer is None:
            renderer = self.figure._get_renderer()
        # Update the positions of all cells based on the renderer
        self._update_positions(renderer)
        # Collect bounding boxes for all cells in the table
        boxes = [cell.get_window_extent(renderer)
                 for cell in self._cells.values()]
        # Calculate the union of all collected bounding boxes
        return Bbox.union(boxes)

    def _do_cell_alignment(self):
        """
        Align cells within the table by calculating row heights and column widths.

        Position each cell accordingly based on the calculated dimensions.
        """
        # Calculate the maximum height for each row and the maximum width for each column
        widths = {}
        heights = {}
        for (row, col), cell in self._cells.items():
            height = heights.setdefault(row, 0.0)
            heights[row] = max(height, cell.get_height())
            width = widths.setdefault(col, 0.0)
            widths[col] = max(width, cell.get_width())

        # Calculate left position for each column
        xpos = 0
        lefts = {}
        for col in sorted(widths):
            lefts[col] = xpos
            xpos += widths[col]

        # Calculate bottom position for each row
        ypos = 0
        bottoms = {}
        for row in sorted(heights, reverse=True):
            bottoms[row] = ypos
            ypos += heights[row]

        # Set the position (x, y) for each cell based on calculated lefts and bottoms
        for (row, col), cell in self._cells.items():
            cell.set_x(lefts[col])
            cell.set_y(bottoms[row])
    def auto_set_column_width(self, col):
        """
        Automatically set the widths of given columns to optimal sizes.

        Parameters
        ----------
        col : int or sequence of ints
            The indices of the columns to auto-scale.
        """
        # 将输入的列索引转换为至少是一维的 NumPy 数组
        col1d = np.atleast_1d(col)
        # 如果列索引数组的数据类型不是整数类型，则发出弃用警告并返回
        if not np.issubdtype(col1d.dtype, np.integer):
            _api.warn_deprecated("3.8", name="col",
                                 message="%(name)r must be an int or sequence of ints. "
                                 "Passing other types is deprecated since %(since)s "
                                 "and will be removed %(removal)s.")
            return
        # 将列索引添加到自动列宽列表中
        for cell in col1d:
            self._autoColumns.append(cell)

        # 设置标记为需要更新
        self.stale = True

    def _auto_set_column_width(self, col, renderer):
        """Automatically set width for column."""
        # 获取指定列的单元格列表
        cells = [cell for key, cell in self._cells.items() if key[1] == col]
        # 计算该列中单元格所需的最大宽度
        max_width = max((cell.get_required_width(renderer) for cell in cells),
                        default=0)
        # 将该列中所有单元格的宽度设置为最大宽度
        for cell in cells:
            cell.set_width(max_width)

    def auto_set_font_size(self, value=True):
        """Automatically set font size."""
        # 设置是否自动设置字体大小的标志位
        self._autoFontsize = value
        # 设置标记为需要更新
        self.stale = True

    def _auto_set_font_size(self, renderer):
        # 如果单元格字典为空，则直接返回
        if len(self._cells) == 0:
            return
        # 获取第一个单元格的字体大小作为初始值
        fontsize = next(iter(self._cells.values())).get_fontsize()
        cells = []
        # 遍历单元格字典，自动调整字体大小
        for key, cell in self._cells.items():
            # 忽略已经自动设置字体大小的列
            if key[1] in self._autoColumns:
                continue
            size = cell.auto_set_font_size(renderer)
            fontsize = min(fontsize, size)
            cells.append(cell)

        # 将所有单元格设置为相同的字体大小
        for cell in self._cells.values():
            cell.set_fontsize(fontsize)

    def scale(self, xscale, yscale):
        """Scale column widths by *xscale* and row heights by *yscale*."""
        # 按照指定比例缩放列宽和行高
        for c in self._cells.values():
            c.set_width(c.get_width() * xscale)
            c.set_height(c.get_height() * yscale)

    def set_fontsize(self, size):
        """
        Set the font size, in points, of the cell text.

        Parameters
        ----------
        size : float
            The font size to set for all cells.

        Notes
        -----
        As long as auto font size has not been disabled, the value will be
        clipped such that the text fits horizontally into the cell.

        You can disable this behavior using `.auto_set_font_size`.

        >>> the_table.auto_set_font_size(False)
        >>> the_table.set_fontsize(20)

        However, there is no automatic scaling of the row height so that the
        text may exceed the cell boundary.
        """
        # 设置所有单元格的字体大小
        for cell in self._cells.values():
            cell.set_fontsize(size)
        # 设置标记为需要更新
        self.stale = True
    # 定义一个方法 `_offset`，用于移动所有艺术对象的位置，按照指定的偏移量 (ox, oy)，这些偏移量是相对于坐标轴的位置。
    def _offset(self, ox, oy):
        """Move all the artists by ox, oy (axes coords)."""
        # 遍历所有的单元格对象
        for c in self._cells.values():
            # 获取当前单元格的原始位置
            x, y = c.get_x(), c.get_y()
            # 更新单元格的 x 和 y 坐标，以实现移动效果
            c.set_x(x + ox)
            c.set_y(y + oy)

    # 定义一个方法 `_update_positions`，该方法在渲染器中被调用，允许使用 `get_window_extent` 更精确地估计宽度和高度。
    def _update_positions(self, renderer):
        # 对于所有自动设置宽度的列，调用 `_auto_set_column_width` 方法
        for col in self._autoColumns:
            self._auto_set_column_width(col, renderer)

        # 如果设置了自动字体大小，调用 `_auto_set_font_size` 方法
        if self._autoFontsize:
            self._auto_set_font_size(renderer)

        # 对齐所有单元格
        self._do_cell_alignment()

        # 获取表格的边界框（bounding box）
        bbox = self._get_grid_bbox(renderer)
        l, b, w, h = bbox.bounds

        # 如果指定了特定的边界框 `_bbox`
        if self._bbox is not None:
            # 根据边界框的尺寸调整表格大小
            if isinstance(self._bbox, Bbox):
                rl, rb, rw, rh = self._bbox.bounds
            else:
                rl, rb, rw, rh = self._bbox
            # 根据宽度和高度的比例缩放表格
            self.scale(rw / w, rh / h)
            # 计算偏移量以使表格对齐到边界框
            ox = rl - l
            oy = rb - b
            # 再次对齐所有单元格
            self._do_cell_alignment()
        else:
            # 根据位置参数 `_loc` 来确定表格的位置

            # 定义位置编码
            (BEST, UR, UL, LL, LR, CL, CR, LC, UC, C,
             TR, TL, BL, BR, R, L, T, B) = range(len(self.codes))

            # 默认为居中位置
            ox = (0.5 - w / 2) - l
            oy = (0.5 - h / 2) - b

            # 根据 `_loc` 参数调整偏移量
            if self._loc in (UL, LL, CL):   # 左侧
                ox = self.AXESPAD - l
            if self._loc in (BEST, UR, LR, R, CR):  # 右侧
                ox = 1 - (l + w + self.AXESPAD)
            if self._loc in (BEST, UR, UL, UC):     # 上方
                oy = 1 - (b + h + self.AXESPAD)
            if self._loc in (LL, LR, LC):           # 下方
                oy = self.AXESPAD - b
            if self._loc in (LC, UC, C):            # 水平居中
                ox = (0.5 - w / 2) - l
            if self._loc in (CL, CR, C):            # 垂直居中
                oy = (0.5 - h / 2) - b

            if self._loc in (TL, BL, L):            # 左侧外部
                ox = - (l + w)
            if self._loc in (TR, BR, R):            # 右侧外部
                ox = 1.0 - l
            if self._loc in (TR, TL, T):            # 顶部外部
                oy = 1.0 - b
            if self._loc in (BL, BR, B):            # 底部外部
                oy = - (b + h)

        # 使用 `_offset` 方法来应用计算出的偏移量
        self._offset(ox, oy)

    # 返回表格中的所有单元格，以字典形式映射 *(行, 列)* 到 `.Cell` 对象。
    def get_celld(self):
        r"""
        Return a dict of cells in the table mapping *(row, column)* to
        `.Cell`\s.

        Notes
        -----
        You can also directly index into the Table object to access individual
        cells::

            cell = table[row, col]

        """
        return self._cells
# 导入 _docstring 模块中的 dedent_interpd 函数，用于处理文档字符串的缩进
@_docstring.dedent_interpd
# 定义名为 table 的函数，用于在给定的 Axes 对象 ax 上添加表格
def table(ax,
          # 表格的内容数据，必须是一个二维列表，表示表格的行和列
          cellText=None, 
          # 表格单元格的背景颜色，也是一个二维列表
          cellColours=None,
          # 文字在单元格中的水平对齐方式，默认为右对齐
          cellLoc='right', 
          # 每列的宽度，如果未指定，则所有列宽度相等
          colWidths=None,
          # 行标签，显示在表格左侧
          rowLabels=None, 
          # 行标签的背景颜色
          rowColours=None, 
          # 行标签文字的水平对齐方式，默认为左对齐
          rowLoc='left',
          # 列标签，显示在表格顶部
          colLabels=None, 
          # 列标签的背景颜色
          colColours=None, 
          # 列标签文字的水平对齐方式，默认为居中对齐
          colLoc='center',
          # 表格的位置，相对于 Axes 对象的位置，默认在底部
          loc='bottom', 
          # 表格的边界框，用于指定表格的具体位置和大小
          bbox=None, 
          # 表格的边缘绘制方式，默认为闭合边界
          edges='closed',
          # 其他参数通过 **kwargs 传递，用于配置表格的其他属性
          **kwargs):
    """
    Add a table to an `~.axes.Axes`.

    At least one of *cellText* or *cellColours* must be specified. These
    parameters must be 2D lists, in which the outer lists define the rows and
    the inner list define the column values per row. Each row must have the
    same number of elements.

    The table can optionally have row and column headers, which are configured
    using *rowLabels*, *rowColours*, *rowLoc* and *colLabels*, *colColours*,
    *colLoc* respectively.

    For finer grained control over tables, use the `.Table` class and add it to
    the Axes with `.Axes.add_table`.

    Parameters
    ----------
    cellText : 2D list of str, optional
        The texts to place into the table cells.

        *Note*: Line breaks in the strings are currently not accounted for and
        will result in the text exceeding the cell boundaries.

    cellColours : 2D list of :mpltype:`color`, optional
        The background colors of the cells.

    cellLoc : {'right', 'center', 'left'}
        The alignment of the text within the cells.

    colWidths : list of float, optional
        The column widths in units of the axes. If not given, all columns will
        have a width of *1 / ncols*.

    rowLabels : list of str, optional
        The text of the row header cells.

    rowColours : list of :mpltype:`color`, optional
        The colors of the row header cells.

    rowLoc : {'left', 'center', 'right'}
        The text alignment of the row header cells.

    colLabels : list of str, optional
        The text of the column header cells.

    colColours : list of :mpltype:`color`, optional
        The colors of the column header cells.

    colLoc : {'center', 'left', 'right'}
        The text alignment of the column header cells.

    loc : str, default: 'bottom'
        The position of the cell with respect to *ax*. This must be one of
        the `~.Table.codes`.

    bbox : `.Bbox` or [xmin, ymin, width, height], optional
        A bounding box to draw the table into. If this is not *None*, this
        overrides *loc*.

    edges : {'closed', 'open', 'horizontal', 'vertical'} or substring of 'BRTL'
        The cell edges to be drawn with a line. See also
        `~.Cell.visible_edges`.

    Returns
    -------
    `~matplotlib.table.Table`
        The created table.

    Other Parameters
    ----------------
    **kwargs
        `.Table` properties.

    %(Table:kwdoc)s
    """

    # 如果 cellColours 和 cellText 都为 None，则抛出 ValueError
    if cellColours is None and cellText is None:
        raise ValueError('At least one argument from "cellColours" or '
                         '"cellText" must be provided to create a table.')

    # 检查是否提供了至少一些 cellText
    if cellText is None:
        # 如果 cellText 为 None，则假设只需要颜色信息
        # 获取行数和列数
        rows = len(cellColours)
        cols = len(cellColours[0])
        # 创建一个空的二维列表作为 cellText
        cellText = [[''] * cols] * rows

    # 获取实际行数和列数
    rows = len(cellText)
    cols = len(cellText[0])

    # 检查每行的列数是否与第一行相同，如果不同则抛出异常
    for row in cellText:
        if len(row) != cols:
            raise ValueError(f"Each row in 'cellText' must have {cols} columns")

    # 如果 cellColours 不为 None，则进行进一步检查
    if cellColours is not None:
        # 检查 cellColours 的行数是否与 cellText 的行数相同，如果不同则抛出异常
        if len(cellColours) != rows:
            raise ValueError(f"'cellColours' must have {rows} rows")
        # 检查每行的列数是否与第一行相同，如果不同则抛出异常
        for row in cellColours:
            if len(row) != cols:
                raise ValueError(f"Each row in 'cellColours' must have {cols} columns")
    else:
        # 如果 cellColours 为 None，则创建一个默认的颜色列表
        cellColours = ['w' * cols] * rows

    # 如果 colWidths 为 None，则设置默认的列宽度
    if colWidths is None:
        colWidths = [1.0 / cols] * cols

    # 填充缺失的行和列标签信息
    rowLabelWidth = 0
    if rowLabels is None:
        # 如果 rowLabels 为 None，并且存在 rowColours，则创建空的标签列表，并设置 rowLabelWidth
        if rowColours is not None:
            rowLabels = [''] * rows
            rowLabelWidth = colWidths[0]
    elif rowColours is None:
        # 如果 rowLabels 不为 None，但 rowColours 为 None，则设置默认的行颜色
        rowColours = 'w' * rows

    # 如果 rowLabels 不为 None，则检查其长度是否与行数相同，如果不同则抛出异常
    if rowLabels is not None:
        if len(rowLabels) != rows:
            raise ValueError(f"'rowLabels' must be of length {rows}")

    # 如果 colLabels 为 None，则设置默认的列标签
    # 如果 colColours 不为 None，则创建一个空的标签列表，并设置偏移量为1
    offset = 1
    if colLabels is None:
        if colColours is not None:
            colLabels = [''] * cols
        else:
            offset = 0
    elif colColours is None:
        # 如果 colLabels 不为 None，但 colColours 为 None，则设置默认的列颜色
        colColours = 'w' * cols

    # 如果 cellColours 为 None，则创建一个默认的颜色列表
    if cellColours is None:
        cellColours = ['w' * cols] * rows

    # 创建 Table 对象
    table = Table(ax, loc, bbox, **kwargs)
    table.edges = edges
    height = table._approx_text_height()

    # 添加单元格数据
    for row in range(rows):
        for col in range(cols):
            table.add_cell(row + offset, col,
                           width=colWidths[col], height=height,
                           text=cellText[row][col],
                           facecolor=cellColours[row][col],
                           loc=cellLoc)

    # 添加列标签
    if colLabels is not None:
        for col in range(cols):
            table.add_cell(0, col,
                           width=colWidths[col], height=height,
                           text=colLabels[col], facecolor=colColours[col],
                           loc=colLoc)

    # 添加行标签
    if rowLabels is not None:
        for row in range(rows):
            table.add_cell(row + offset, -1,
                           width=rowLabelWidth or 1e-15, height=height,
                           text=rowLabels[row], facecolor=rowColours[row],
                           loc=rowLoc)
        # 自动设置行标签列的宽度
        if rowLabelWidth == 0:
            table.auto_set_column_width(-1)

    # 将表格添加到坐标轴中
    ax.add_table(table)
    return table
```