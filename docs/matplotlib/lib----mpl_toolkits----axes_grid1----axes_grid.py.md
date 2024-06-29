# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\axes_grid.py`

```py
from numbers import Number
import functools
from types import MethodType

import numpy as np

from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec

from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist


class CbarAxesBase:
    def __init__(self, *args, orientation, **kwargs):
        # 初始化方法，接受任意位置参数和指定的方向关键字参数
        self.orientation = orientation
        super().__init__(*args, **kwargs)

    def colorbar(self, mappable, **kwargs):
        # 创建颜色条的方法，使用当前对象所在图形的颜色条方法，在指定位置和其他参数的情况下绘制颜色条
        return self.figure.colorbar(
            mappable, cax=self, location=self.orientation, **kwargs)

    @_api.deprecated("3.8", alternative="ax.tick_params and colorbar.set_label")
    def toggle_label(self, b):
        # 标记方法已经废弃，提供替代方法建议
        axis = self.axis[self.orientation]
        axis.toggle(ticklabels=b, label=b)


_cbaraxes_class_factory = cbook._make_class_factory(CbarAxesBase, "Cbar{}")


class Grid:
    """
    A grid of Axes.

    In Matplotlib, the Axes location (and size) is specified in normalized
    figure coordinates. This may not be ideal for images that needs to be
    displayed with a given aspect ratio; for example, it is difficult to
    display multiple images of a same size with some fixed padding between
    them.  AxesGrid can be used in such case.

    Attributes
    ----------
    axes_all : list of Axes
        A flat list of Axes. Note that you can also access this directly
        from the grid. The following is equivalent ::

            grid[i] == grid.axes_all[i]
            len(grid) == len(grid.axes_all)

    axes_column : list of list of Axes
        A 2D list of Axes where the first index is the column. This results
        in the usage pattern ``grid.axes_column[col][row]``.
    axes_row : list of list of Axes
        A 2D list of Axes where the first index is the row. This results
        in the usage pattern ``grid.axes_row[row][col]``.
    axes_llc : Axes
        The Axes in the lower left corner.
    ngrids : int
        Number of Axes in the grid.
    """

    _defaultAxesClass = Axes

    def __init__(self, fig,
                 rect,
                 nrows_ncols,
                 ngrids=None,
                 direction="row",
                 axes_pad=0.02,
                 *,
                 share_all=False,
                 share_x=True,
                 share_y=True,
                 label_mode="L",
                 axes_class=None,
                 aspect=False,
                 ):
        """
        Parameters
        ----------
        fig : `.Figure`
            The parent figure.
        rect : (float, float, float, float), (int, int, int), int, or \
            sequence of float
            The location of the grid in the figure (left, bottom, width,
            height) or in pixels. The conventions of the units are the same
            as the `.Figure.add_subplot` itself.
        nrows_ncols : tuple of int
            Number of rows and number of columns.
        ngrids : int, optional
            Total number of grids to create. If not provided, calculated
            from the product of `nrows` and `ncols`.
        direction : {'row', 'column'}, default: 'row'
            Direction of grid creation: either row-wise or column-wise.
        axes_pad : float, default: 0.02
            Padding between axes in figure coordinates.
        share_all : bool, default: False
            If True, each subplot x or y axis will be shared among all
            subplots.
        share_x : bool, default: True
            If True, the x axis will be shared among all subplots.
        share_y : bool, default: True
            If True, the y axis will be shared among all subplots.
        label_mode : {'L', '1', 'all'}, default: 'L'
            Controls the labeling of axes. 'L' assigns labels only to the
            outer axes, '1' assigns labels only to the outer x- and
            y-axes, and 'all' assigns labels to all axes.
        axes_class : type, default: None
            The class to instantiate for each axes. If None, defaults to
            `.Axes`.
        aspect : bool, default: False
            If True, the grid will have an aspect ratio of 1.

        Notes
        -----
        In the 2D case, the aspect ratio of the axes is controlled by the
        relative lengths of the width and height. If the aspect ratio is
        not fixed, then the relative size of the axes will reflect the
        aspect ratio.
        """
        # 初始化方法，接受多个参数来配置网格的属性和布局
        pass  # Placeholder for additional code
    # 初始化定位器设置，用于设置子图之间的水平和垂直间距
    def _init_locators(self):
        # 设置水平方向的分隔符，使用比例大小和水平填充大小交替排列
        self._divider.set_horizontal(
            [Size.Scaled(1), self._horiz_pad_size] * (self._ncols-1) + [Size.Scaled(1)])
        # 设置垂直方向的分隔符，使用比例大小和垂直填充大小交替排列
        self._divider.set_vertical(
            [Size.Scaled(1), self._vert_pad_size] * (self._nrows-1) + [Size.Scaled(1)])
        # 遍历每个子图格子，设置其坐标定位器
        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            # 根据列和行计算新的定位器并应用到对应的子图上
            self.axes_all[i].set_axes_locator(
                self._divider.new_locator(nx=2 * col, ny=2 * (self._nrows - 1 - row)))

    # 根据索引计算并返回列和行号
    def _get_col_row(self, n):
        if self._direction == "column":
            col, row = divmod(n, self._nrows)
        else:
            row, col = divmod(n, self._ncols)

        return col, row

    # 如果已经实现了 __getitem__，则推荐实现 __len__ 方法
    def __len__(self):
        return len(self.axes_all)

    # 实现获取子图的方法，通过索引获取对应的子图对象
    def __getitem__(self, i):
        return self.axes_all[i]

    # 返回当前子图布局的行数和列数
    def get_geometry(self):
        """
        返回网格的行数和列数作为元组 (nrows, ncols)。
        """
        return self._nrows, self._ncols

    # 设置子图之间的填充大小
    def set_axes_pad(self, axes_pad):
        """
        设置子图之间的填充大小。

        Parameters
        ----------
        axes_pad : (float, float)
            填充大小（水平填充大小，垂直填充大小），单位为英寸。
        """
        self._horiz_pad_size.fixed_size = axes_pad[0]
        self._vert_pad_size.fixed_size = axes_pad[1]

    # 获取当前子图之间的填充大小
    def get_axes_pad(self):
        """
        返回子图之间的填充大小。

        Returns
        -------
        hpad, vpad
            填充大小（水平填充大小，垂直填充大小），单位为英寸。
        """
        return (self._horiz_pad_size.fixed_size,
                self._vert_pad_size.fixed_size)

    # 设置子图之间的纵横比
    def set_aspect(self, aspect):
        """设置子图分隔符的纵横比。"""
        self._divider.set_aspect(aspect)

    # 获取子图之间的纵横比
    def get_aspect(self):
        """返回子图分隔符的纵横比。"""
        return self._divider.get_aspect()
    def set_label_mode(self, mode):
        """
        Define which axes have tick labels.

        Parameters
        ----------
        mode : {"L", "1", "all", "keep"}
            The label mode:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": All axes are labelled.
            - "keep": Do not do anything.
        """
        # 检查模式是否在允许的列表中
        _api.check_in_list(["all", "L", "1", "keep"], mode=mode)
        
        # 判断是否为最后一行和第一列的布尔数组
        is_last_row, is_first_col = (
            np.mgrid[:self._nrows, :self._ncols] == [[[self._nrows - 1]], [[0]]])
        
        # 根据不同的模式设置底部和左侧标签的显示情况
        if mode == "all":
            bottom = left = np.full((self._nrows, self._ncols), True)
        elif mode == "L":
            bottom = is_last_row
            left = is_first_col
        elif mode == "1":
            bottom = left = is_last_row & is_first_col
        else:
            return
        
        # 遍历每个子图的行和列，根据类型设置底部和左侧轴的标签显示
        for i in range(self._nrows):
            for j in range(self._ncols):
                ax = self.axes_row[i][j]
                if isinstance(ax.axis, MethodType):
                    # 如果轴是方法类型，创建简单的轴艺术家对象
                    bottom_axis = SimpleAxisArtist(ax.xaxis, 1, ax.spines["bottom"])
                    left_axis = SimpleAxisArtist(ax.yaxis, 1, ax.spines["left"])
                else:
                    # 否则直接使用底部和左侧轴
                    bottom_axis = ax.axis["bottom"]
                    left_axis = ax.axis["left"]
                
                # 根据布尔值开关底部和左侧轴的标签
                bottom_axis.toggle(ticklabels=bottom[i, j], label=bottom[i, j])
                left_axis.toggle(ticklabels=left[i, j], label=left[i, j])

    def get_divider(self):
        """
        获取当前对象的分隔器对象。
        """
        return self._divider

    def set_axes_locator(self, locator):
        """
        设置分隔器的定位器。
        
        Parameters
        ----------
        locator : object
            分隔器的定位器对象。
        """
        self._divider.set_locator(locator)

    def get_axes_locator(self):
        """
        获取分隔器的定位器对象。
        """
        return self._divider.get_locator()
# 定义一个名为 ImageGrid 的类，继承自 Grid 类
class ImageGrid(Grid):
    """
    A grid of Axes for Image display.

    This class is a specialization of `~.axes_grid1.axes_grid.Grid` for displaying a
    grid of images.  In particular, it forces all axes in a column to share their x-axis
    and all axes in a row to share their y-axis.  It further provides helpers to add
    colorbars to some or all axes.
    """

# 将 ImageGrid 类赋值给 AxesGrid 变量，用作别名
AxesGrid = ImageGrid
```