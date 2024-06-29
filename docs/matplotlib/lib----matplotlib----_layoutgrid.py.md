# `D:\src\scipysrc\matplotlib\lib\matplotlib\_layoutgrid.py`

```py
"""
A layoutgrid is a nrows by ncols set of boxes, meant to be used by
`._constrained_layout`, each box is analogous to a subplotspec element of
a gridspec.

Each box is defined by left[ncols], right[ncols], bottom[nrows] and top[nrows],
and by two editable margins for each side.  The main margin gets its value
set by the size of ticklabels, titles, etc on each Axes that is in the figure.
The outer margin is the padding around the Axes, and space for any
colorbars.

The "inner" widths and heights of these boxes are then constrained to be the
same (relative the values of `width_ratios[ncols]` and `height_ratios[nrows]`).

The layoutgrid is then constrained to be contained within a parent layoutgrid,
its column(s) and row(s) specified when it is created.
"""

import itertools
import kiwisolver as kiwi  # 导入kiwisolver库，用作布局求解器
import logging  # 导入日志记录模块
import numpy as np  # 导入NumPy库

import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayoutGrid:
    """
    Analogous to a gridspec, and contained in another LayoutGrid.
    """
    # 初始化函数，用于创建一个布局网格对象
    def __init__(self, parent=None, parent_pos=(0, 0),
                 parent_inner=False, name='', ncols=1, nrows=1,
                 h_pad=None, w_pad=None, width_ratios=None,
                 height_ratios=None):
        # 导入 kiwi 模块中的 Variable 类
        Variable = kiwi.Variable
        # 设置父对象的位置
        self.parent_pos = parent_pos
        # 确定是否为内部父对象
        self.parent_inner = parent_inner
        # 设置对象的名称，并加上唯一的序列 ID
        self.name = name + seq_id()
        # 如果父对象是 LayoutGrid 类型，则在其名称前加上父对象的名称
        if isinstance(parent, LayoutGrid):
            self.name = f'{parent.name}.{self.name}'
        # 设置网格行数和列数
        self.nrows = nrows
        self.ncols = ncols
        # 将传入的高度比例转换为至少为 1 维的 numpy 数组
        self.height_ratios = np.atleast_1d(height_ratios)
        # 如果未指定高度比例，则将其默认设置为每行高度为 1
        if height_ratios is None:
            self.height_ratios = np.ones(nrows)
        # 将传入的宽度比例转换为至少为 1 维的 numpy 数组
        self.width_ratios = np.atleast_1d(width_ratios)
        # 如果未指定宽度比例，则将其默认设置为每列宽度为 1
        if width_ratios is None:
            self.width_ratios = np.ones(ncols)

        # 创建带有下划线的名称变量
        sn = self.name + '_'
        # 如果父对象不是 LayoutGrid 类型，则创建一个新的求解器 kiwi.Solver()
        if not isinstance(parent, LayoutGrid):
            # parent 如果不是 LayoutGrid 类型，则可以是一个矩形，用来容纳布局
            # 允许指定一个矩形来包含布局
            self.solver = kiwi.Solver()
        else:
            # 如果父对象是 LayoutGrid 类型，则将当前对象添加为其子对象，并使用其求解器
            parent.add_child(self, *parent_pos)
            self.solver = parent.solver
        # 用于跟踪与该布局关联的图形对象，初始为空数组
        self.artists = np.empty((nrows, ncols), dtype=object)
        # 用于跟踪布局中的子对象，初始为空数组
        self.children = np.empty((nrows, ncols), dtype=object)

        # 初始化边距字典和边距值字典
        self.margins = {}
        self.margin_vals = {}

        # 所有列中的所有框共享相同的左/右边距：
        for todo in ['left', 'right', 'leftcb', 'rightcb']:
            # 跟踪值，以便只有当边距大于当前值时才更改
            self.margin_vals[todo] = np.zeros(ncols)

        # 获取当前对象的求解器
        sol = self.solver

        # 创建每列的左边界变量列表和右边界变量列表
        self.lefts = [Variable(f'{sn}lefts[{i}]') for i in range(ncols)]
        self.rights = [Variable(f'{sn}rights[{i}]') for i in range(ncols)]
        # 对于每个边距类型 ['left', 'right', 'leftcb', 'rightcb']，创建相应的边距变量列表
        for todo in ['left', 'right', 'leftcb', 'rightcb']:
            self.margins[todo] = [Variable(f'{sn}margins[{todo}][{i}]')
                                  for i in range(ncols)]
            # 将每个边距变量设置为强约束
            for i in range(ncols):
                sol.addEditVariable(self.margins[todo][i], 'strong')

        # 对于每个边距类型 ['bottom', 'top', 'bottomcb', 'topcb']，创建相应的边距变量列表和值数组
        for todo in ['bottom', 'top', 'bottomcb', 'topcb']:
            self.margins[todo] = np.empty((nrows), dtype=object)
            self.margin_vals[todo] = np.zeros(nrows)

        # 创建每行的底边界变量列表和顶边界变量列表
        self.bottoms = [Variable(f'{sn}bottoms[{i}]') for i in range(nrows)]
        self.tops = [Variable(f'{sn}tops[{i}]') for i in range(nrows)]
        # 对于每个边距类型 ['bottom', 'top', 'bottomcb', 'topcb']，创建相应的边距变量列表
        for todo in ['bottom', 'top', 'bottomcb', 'topcb']:
            self.margins[todo] = [Variable(f'{sn}margins[{todo}][{i}]')
                                  for i in range(nrows)]
            # 将每个边距变量设置为强约束
            for i in range(nrows):
                sol.addEditVariable(self.margins[todo][i], 'strong')

        # 将默认情况下这些边距设置为零。随着子对象的填充，它们将被编辑。
        self.reset_margins()
        # 添加与父对象相关的约束
        self.add_constraints(parent)

        # 设置水平和垂直间距
        self.h_pad = h_pad
        self.w_pad = w_pad
    def __repr__(self):
        # 返回对象的字符串表示，包括布局框的名称、行数、列数
        str = f'LayoutBox: {self.name:25s} {self.nrows}x{self.ncols},\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                # 每个单元格的详细布局信息，包括边界值和边距
                str += f'{i}, {j}: ' \
                       f'L{self.lefts[j].value():1.3f}, ' \
                       f'B{self.bottoms[i].value():1.3f}, ' \
                       f'R{self.rights[j].value():1.3f}, ' \
                       f'T{self.tops[i].value():1.3f}, ' \
                       f'ML{self.margins["left"][j].value():1.3f}, ' \
                       f'MR{self.margins["right"][j].value():1.3f}, ' \
                       f'MB{self.margins["bottom"][i].value():1.3f}, ' \
                       f'MT{self.margins["top"][i].value():1.3f}, \n'
        return str

    def reset_margins(self):
        """
        重置所有边距为零。例如在更改图形大小后必须执行此操作，因为轴标签的相对大小会改变。
        """
        for todo in ['left', 'right', 'bottom', 'top',
                     'leftcb', 'rightcb', 'bottomcb', 'topcb']:
            self.edit_margins(todo, 0.0)

    def add_constraints(self, parent):
        # 定义自洽的约束条件
        self.hard_constraints()
        # 与父布局网格的关系定义：
        self.parent_constraints(parent)
        # 定义网格单元之间的相对宽度，并水平和垂直堆叠。
        self.grid_constraints()

    def hard_constraints(self):
        """
        这些是冗余约束条件，以及使代码其余部分更容易的约束条件。
        """
        for i in range(self.ncols):
            # 定义水平约束条件
            hc = [self.rights[i] >= self.lefts[i],
                  (self.rights[i] - self.margins['right'][i] -
                    self.margins['rightcb'][i] >=
                    self.lefts[i] - self.margins['left'][i] -
                    self.margins['leftcb'][i])
                  ]
            for c in hc:
                # 将约束条件添加到求解器中，标记为必需的
                self.solver.addConstraint(c | 'required')

        for i in range(self.nrows):
            # 定义垂直约束条件
            hc = [self.tops[i] >= self.bottoms[i],
                  (self.tops[i] - self.margins['top'][i] -
                    self.margins['topcb'][i] >=
                    self.bottoms[i] - self.margins['bottom'][i] -
                    self.margins['bottomcb'][i])
                  ]
            for c in hc:
                # 将约束条件添加到求解器中，标记为必需的
                self.solver.addConstraint(c | 'required')

    def add_child(self, child, i=0, j=0):
        # np.ix_ 返回 i 和 j 索引的交叉产品
        self.children[np.ix_(np.atleast_1d(i), np.atleast_1d(j))] = child
    def parent_constraints(self, parent):
        # 父级约束条件...
        # 即第一列的左边等于父级的左边，最后一列的右边等于父级的右边...
        
        if not isinstance(parent, LayoutGrid):
            # 如果父级不是 LayoutGrid 类型
            
            # 定义一个与图形坐标相关的矩形
            hc = [self.lefts[0] == parent[0],
                  self.rights[-1] == parent[0] + parent[2],
                  # top 和 bottom 的顺序颠倒了...
                  self.tops[0] == parent[1] + parent[3],
                  self.bottoms[-1] == parent[1]]
        else:
            # 如果父级是 LayoutGrid 类型
            rows, cols = self.parent_pos
            rows = np.atleast_1d(rows)
            cols = np.atleast_1d(cols)
            
            # 获取父级格子的左边界、右边界、上边界和下边界
            left = parent.lefts[cols[0]]
            right = parent.rights[cols[-1]]
            top = parent.tops[rows[0]]
            bottom = parent.bottoms[rows[-1]]
            
            if self.parent_inner:
                # 布局网格位于父级内部的情况下
                
                # 调整左边界和右边界
                left += parent.margins['left'][cols[0]]
                left += parent.margins['leftcb'][cols[0]]
                right -= parent.margins['right'][cols[-1]]
                right -= parent.margins['rightcb'][cols[-1]]
                
                # 调整上边界和下边界
                top -= parent.margins['top'][rows[0]]
                top -= parent.margins['topcb'][rows[0]]
                bottom += parent.margins['bottom'][rows[-1]]
                bottom += parent.margins['bottomcb'][rows[-1]]
            
            # 构建水平和垂直约束列表
            hc = [self.lefts[0] == left,
                  self.rights[-1] == right,
                  # 从上到下的顺序
                  self.tops[0] == top,
                  self.bottoms[-1] == bottom]
        
        # 将约束添加到求解器中
        for c in hc:
            self.solver.addConstraint(c | 'required')
    # 网格约束函数：用于约束网格内部部分的比例相同（相对于width_ratios）

    # 约束宽度：
    w = (self.rights[0] - self.margins['right'][0] -
         self.margins['rightcb'][0])
    w = (w - self.lefts[0] - self.margins['left'][0] -
         self.margins['leftcb'][0])
    w0 = w / self.width_ratios[0]
    # 从左到右遍历
    for i in range(1, self.ncols):
        w = (self.rights[i] - self.margins['right'][i] -
             self.margins['rightcb'][i])
        w = (w - self.lefts[i] - self.margins['left'][i] -
             self.margins['leftcb'][i])
        c = (w == w0 * self.width_ratios[i])
        # 添加宽度约束条件到求解器
        self.solver.addConstraint(c | 'strong')
        # 约束网格单元格直接相邻
        c = (self.rights[i - 1] == self.lefts[i])
        # 添加相邻约束条件到求解器
        self.solver.addConstraint(c | 'strong')

    # 约束高度：
    h = self.tops[0] - self.margins['top'][0] - self.margins['topcb'][0]
    h = (h - self.bottoms[0] - self.margins['bottom'][0] -
         self.margins['bottomcb'][0])
    h0 = h / self.height_ratios[0]
    # 从上到下遍历：
    for i in range(1, self.nrows):
        h = (self.tops[i] - self.margins['top'][i] -
             self.margins['topcb'][i])
        h = (h - self.bottoms[i] - self.margins['bottom'][i] -
             self.margins['bottomcb'][i])
        c = (h == h0 * self.height_ratios[i])
        # 添加高度约束条件到求解器
        self.solver.addConstraint(c | 'strong')
        # 约束网格单元格直接垂直相邻
        c = (self.bottoms[i - 1] == self.tops[i])
        # 添加相邻约束条件到求解器
        self.solver.addConstraint(c | 'strong')

# 边距编辑函数：边距是可变的，旨在包含诸如坐标轴标签、刻度标签、标题等固定大小的内容
def edit_margin(self, todo, size, cell):
    """
    Change the size of the margin for one cell.

    Parameters
    ----------
    todo : string (one of 'left', 'right', 'bottom', 'top')
        margin to alter.

    size : float
        Size of the margin.  If it is larger than the existing minimum it
        updates the margin size. Fraction of figure size.

    cell : int
        Cell column or row to edit.
    """
    # 建议求解器设置边距的大小
    self.solver.suggestValue(self.margins[todo][cell], size)
    # 更新边距值列表中的值
    self.margin_vals[todo][cell] = size
    def edit_margin_min(self, todo, size, cell=0):
        """
        Change the minimum size of the margin for one cell.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            Margin to alter.

        size : float
            Minimum size of the margin. If larger than the existing minimum,
            it updates the margin size. Fraction of figure size.

        cell : int, optional
            Cell column or row to edit. Default is 0.
        """

        if size > self.margin_vals[todo][cell]:
            # Call the edit_margin method to update the margin size
            self.edit_margin(todo, size, cell)

    def edit_margins(self, todo, size):
        """
        Change the size of all the margins of all the cells in the layout grid.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            Margin to alter.

        size : float
            Size to set the margins. Fraction of figure size.
        """

        for i in range(len(self.margin_vals[todo])):
            # Call edit_margin method for each cell to update the margin size
            self.edit_margin(todo, size, i)

    def edit_all_margins_min(self, todo, size):
        """
        Change the minimum size of all the margins of all the cells in the layout grid.

        Parameters
        ----------
        todo : {'left', 'right', 'bottom', 'top'}
            The margin to alter.

        size : float
            Minimum size of the margin. If larger than the existing minimum,
            it updates the margin size. Fraction of figure size.
        """

        for i in range(len(self.margin_vals[todo])):
            # Call edit_margin_min method for each cell to update the minimum margin size
            self.edit_margin_min(todo, size, i)

    def edit_outer_margin_mins(self, margin, ss):
        """
        Edit all four margin minimums in one statement.

        Parameters
        ----------
        margin : dict
            Size of margins in a dictionary with keys 'left', 'right', 'bottom', 'top'.

        ss : SubplotSpec
            Defines the subplotspec these margins should be applied to.
        """

        # Update the minimum margin sizes for specific cells based on subplot spec
        self.edit_margin_min('left', margin['left'], ss.colspan.start)
        self.edit_margin_min('leftcb', margin['leftcb'], ss.colspan.start)
        self.edit_margin_min('right', margin['right'], ss.colspan.stop - 1)
        self.edit_margin_min('rightcb', margin['rightcb'], ss.colspan.stop - 1)
        # Rows are from the top down:
        self.edit_margin_min('top', margin['top'], ss.rowspan.start)
        self.edit_margin_min('topcb', margin['topcb'], ss.rowspan.start)
        self.edit_margin_min('bottom', margin['bottom'], ss.rowspan.stop - 1)
        self.edit_margin_min('bottomcb', margin['bottomcb'],
                             ss.rowspan.stop - 1)

    def get_margins(self, todo, col):
        """Return the margin at this position."""
        return self.margin_vals[todo][col]
    def get_outer_bbox(self, rows=0, cols=0):
        """
        Return the outer bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        # 将行数转换为至少是一维数组
        rows = np.atleast_1d(rows)
        # 将列数转换为至少是一维数组
        cols = np.atleast_1d(cols)

        # 根据给定的行和列计算外部边界框的坐标范围
        bbox = Bbox.from_extents(
            self.lefts[cols[0]].value(),  # 左边界
            self.bottoms[rows[-1]].value(),  # 底边界
            self.rights[cols[-1]].value(),  # 右边界
            self.tops[rows[0]].value()  # 顶边界
        )
        return bbox

    def get_inner_bbox(self, rows=0, cols=0):
        """
        Return the inner bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        # 将行数转换为至少是一维数组
        rows = np.atleast_1d(rows)
        # 将列数转换为至少是一维数组
        cols = np.atleast_1d(cols)

        # 根据给定的行和列计算内部边界框的坐标范围
        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['left'][cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),  # 左边界
            (self.bottoms[rows[-1]].value() +
                self.margins['bottom'][rows[-1]].value() +
                self.margins['bottomcb'][rows[-1]].value()),  # 底边界
            (self.rights[cols[-1]].value() -
                self.margins['right'][cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),  # 右边界
            (self.tops[rows[0]].value() -
                self.margins['top'][rows[0]].value() -
                self.margins['topcb'][rows[0]].value())  # 顶边界
        )
        return bbox

    def get_bbox_for_cb(self, rows=0, cols=0):
        """
        Return the bounding box that includes the
        decorations but, *not* the colorbar...
        """
        # 将行数转换为至少是一维数组
        rows = np.atleast_1d(rows)
        # 将列数转换为至少是一维数组
        cols = np.atleast_1d(cols)

        # 根据给定的行和列计算包含装饰但不包括颜色条的边界框的坐标范围
        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),  # 左边界
            (self.bottoms[rows[-1]].value() +
                self.margins['bottomcb'][rows[-1]].value()),  # 底边界
            (self.rights[cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),  # 右边界
            (self.tops[rows[0]].value() -
                self.margins['topcb'][rows[0]].value())  # 顶边界
        )
        return bbox

    def get_left_margin_bbox(self, rows=0, cols=0):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        # 将行数转换为至少是一维数组
        rows = np.atleast_1d(rows)
        # 将列数转换为至少是一维数组
        cols = np.atleast_1d(cols)

        # 根据给定的行和列计算左边距边界框的坐标范围
        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),  # 左边界
            (self.bottoms[rows[-1]].value()),  # 底边界
            (self.lefts[cols[0]].value() +
                self.margins['leftcb'][cols[0]].value() +
                self.margins['left'][cols[0]].value()),  # 右边界
            (self.tops[rows[0]].value())  # 顶边界
        )
        return bbox
    def get_bottom_margin_bbox(self, rows=0, cols=0):
        """
        Return the bounding box of the bottom margin of the subplot specified by rows and cols.
        Rows and cols can be spans.
        """
        # Ensure rows and cols are arrays with at least one dimension
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        # Calculate the bounding box based on subplot specifications
        bbox = Bbox.from_extents(
            # Left extent of the subplot
            (self.lefts[cols[0]].value()),
            # Bottom extent including bottom and bottomcb margins
            (self.bottoms[rows[-1]].value() +
             self.margins['bottomcb'][rows[-1]].value()),
            # Right extent of the subplot
            (self.rights[cols[-1]].value()),
            # Adjusted bottom extent including all bottom margins
            (self.bottoms[rows[-1]].value() +
                self.margins['bottom'][rows[-1]].value() +
             self.margins['bottomcb'][rows[-1]].value()
             ))
        return bbox

    def get_right_margin_bbox(self, rows=0, cols=0):
        """
        Return the bounding box of the right margin of the subplot specified by rows and cols.
        Rows and cols can be spans.
        """
        # Ensure rows and cols are arrays with at least one dimension
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        # Calculate the bounding box based on subplot specifications
        bbox = Bbox.from_extents(
            # Right extent adjusted by right and rightcb margins
            (self.rights[cols[-1]].value() -
                self.margins['right'][cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            # Bottom extent of the subplot
            (self.bottoms[rows[-1]].value()),
            # Right extent adjusted by rightcb margin
            (self.rights[cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            # Top extent of the subplot
            (self.tops[rows[0]].value()))
        return bbox

    def get_top_margin_bbox(self, rows=0, cols=0):
        """
        Return the bounding box of the top margin of the subplot specified by rows and cols.
        Rows and cols can be spans.
        """
        # Ensure rows and cols are arrays with at least one dimension
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        # Calculate the bounding box based on subplot specifications
        bbox = Bbox.from_extents(
            # Left extent of the subplot
            (self.lefts[cols[0]].value()),
            # Top extent adjusted by topcb margin
            (self.tops[rows[0]].value() -
                self.margins['topcb'][rows[0]].value()),
            # Right extent of the subplot
            (self.rights[cols[-1]].value()),
            # Adjusted top extent including top and topcb margins
            (self.tops[rows[0]].value() -
                self.margins['topcb'][rows[0]].value() -
                self.margins['top'][rows[0]].value()))
        return bbox

    def update_variables(self):
        """
        Update the variables for the solver attached to this layoutgrid.
        """
        # Invoke the solver's updateVariables method to update variables
        self.solver.updateVariables()
# 使用 itertools 模块创建一个无限递增的计数器对象，用于为布局盒子对象生成顺序ID
_layoutboxobjnum = itertools.count()


def seq_id():
    """生成布局盒子对象的短顺序ID."""
    # 返回格式化后的下一个计数器值作为字符串，用于标识布局盒子对象
    return '%06d' % next(_layoutboxobjnum)


def plot_children(fig, lg=None, level=0):
    """简单的绘图函数，用于展示盒子的位置。"""
    # 如果未提供布局网格对象 lg，则通过图形对象的布局引擎执行获取
    if lg is None:
        _layoutgrids = fig.get_layout_engine().execute(fig)
        lg = _layoutgrids[fig]
    
    # 获取当前绘图所需的颜色列表
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    # 根据层级选择当前绘图所需的颜色
    col = colors[level]
    
    # 遍历布局网格对象的行和列
    for i in range(lg.nrows):
        for j in range(lg.ncols):
            # 获取当前单元格的外部边界框并添加矩形图元以展示其外观
            bb = lg.get_outer_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bb.p0, bb.width, bb.height, linewidth=1,
                                   edgecolor='0.7', facecolor='0.7',
                                   alpha=0.2, transform=fig.transFigure,
                                   zorder=-3))
            
            # 获取当前单元格的内部边界框并添加矩形图元以展示其边框
            bbi = lg.get_inner_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=2,
                                   edgecolor=col, facecolor='none',
                                   transform=fig.transFigure, zorder=-2))

            # 获取当前单元格的左边距边界框并添加矩形图元以展示其边距
            bbi = lg.get_left_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.7, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            
            # 获取当前单元格的右边距边界框并添加矩形图元以展示其边距
            bbi = lg.get_right_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.5, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            
            # 获取当前单元格的底部边距边界框并添加矩形图元以展示其边距
            bbi = lg.get_bottom_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.5, 0.7],
                                   transform=fig.transFigure, zorder=-2))
            
            # 获取当前单元格的顶部边距边界框并添加矩形图元以展示其边距
            bbi = lg.get_top_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.2, 0.7],
                                   transform=fig.transFigure, zorder=-2))
    
    # 递归绘制当前布局网格对象的所有子对象
    for ch in lg.children.flat:
        if ch is not None:
            plot_children(fig, ch, level=level+1)
```