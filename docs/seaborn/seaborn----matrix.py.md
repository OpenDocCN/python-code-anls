# `D:\src\scipysrc\seaborn\seaborn\matrix.py`

```
# 导入警告模块
import warnings

# 导入 Matplotlib 库及其子模块
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 导入 NumPy 和 Pandas 库
import numpy as np
import pandas as pd

# 尝试导入 SciPy 层次聚类模块
try:
    from scipy.cluster import hierarchy
    _no_scipy = False  # 设置变量 _no_scipy 表示成功导入 SciPy
except ImportError:
    _no_scipy = True  # 如果导入失败，设置 _no_scipy 变量为 True

# 从内部模块导入颜色映射、轴网格和兼容性支持
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
    despine,
    axis_ticklabels_overlap,
    relative_luminance,
    to_utf8,
    _draw_figure,
)

# 导出的模块列表
__all__ = ["heatmap", "clustermap"]

# 将 Pandas 索引或多重索引转换为坐标轴标签
def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))  # 返回多重索引的名称拼接结果
    else:
        return index.name  # 返回索引的名称

# 将 Pandas 索引或多重索引转换为刻度标签
def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]  # 返回多重索引值的拼接结果
    else:
        return index.values  # 返回索引的值

# 将颜色列表或嵌套颜色列表转换为 RGB 形式
def _convert_colors(colors):
    """Convert either a list of colors or nested lists of colors to RGB."""
    to_rgb = mpl.colors.to_rgb

    try:
        to_rgb(colors[0])
        # If this works, there is only one level of colors
        return list(map(to_rgb, colors))  # 转换为 RGB 形式的颜色列表
    except ValueError:
        # If we get here, we have nested lists
        return [list(map(to_rgb, color_list)) for color_list in colors]  # 转换嵌套颜色列表为 RGB 形式

# 确保数据和掩码兼容，并添加缺失值处理
def _matrix_mask(data, mask):
    """Ensure that data and mask are compatible and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, bool)  # 如果掩码为空，则创建与数据形状相同的全 False 数组

    if isinstance(mask, np.ndarray):
        # 对于数组形式的掩码，确保形状与数据一致，然后转换为 DataFrame
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")
        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=bool)  # 将数组掩码转换为 DataFrame 格式

    elif isinstance(mask, pd.DataFrame):
        # 对于 DataFrame 形式的掩码，确保索引和列标签与数据一致
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # 将缺失数据的单元格添加到掩码中
    # 这解决了 `plt.pcolormesh` 不能正确表示缺失数据的问题
    mask = mask | pd.isnull(data)  # 将空值位置添加到掩码中

    return mask  # 返回处理后的掩码

# 绘制带有标签和颜色映射的矩阵热图的类
class _HeatMapper:
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""
    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        
        # 将 plot_data 转换为 float 类型的 np.ma.array，并将 masked 值填充为 np.nan
        calc_data = plot_data.astype(float).filled(np.nan)
        
        # 如果未提供 vmin，则根据 robust 参数设置
        if vmin is None:
            if robust:
                vmin = np.nanpercentile(calc_data, 2)
            else:
                vmin = np.nanmin(calc_data)
        
        # 如果未提供 vmax，则根据 robust 参数设置
        if vmax is None:
            if robust:
                vmax = np.nanpercentile(calc_data, 98)
            else:
                vmax = np.nanmax(calc_data)
        
        # 设置对象的 vmin 和 vmax 属性
        self.vmin, self.vmax = vmin, vmax

        # 如果未提供 cmap，则根据 center 参数选择默认的 colormap
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, str):
            # 如果 cmap 是字符串，则获取对应的 colormap
            self.cmap = get_colormap(cmap)
        elif isinstance(cmap, list):
            # 如果 cmap 是列表，则创建一个 ListedColormap
            self.cmap = mpl.colors.ListedColormap(cmap)
        else:
            # 否则直接使用提供的 colormap
            self.cmap = cmap

        # 如果提供了 center 参数，则重新调整 colormap
        if center is not None:
            
            # 复制坏值颜色设置
            bad = self.cmap(np.ma.masked_invalid([np.nan]))[0]

            # 在 mpl<3.2 中，只有屏蔽的值会受到 "bad" 颜色规范的影响
            # 参见 https://github.com/matplotlib/matplotlib/pull/14257
            
            # 当 colormap 的极值不映射到与 +-inf 相同颜色时，确定 under 和 over 的值
            under = self.cmap(-np.inf)
            over = self.cmap(np.inf)
            under_set = under != self.cmap(0)
            over_set = over != self.cmap(self.cmap.N - 1)

            # 计算 colormap 的新范围
            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            
            # 根据新的范围创建新的 ListedColormap
            self.cmap = mpl.colors.ListedColormap(self.cmap(cc))
            self.cmap.set_bad(bad)
            
            # 如果设置了 under 和 over 的值，则设置相应的颜色
            if under_set:
                self.cmap.set_under(under)
            if over_set:
                self.cmap.set_over(over)
    # 在热图上添加每个单元格中的数值文本标签
    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        # 更新网格的映射信息
        mesh.update_scalarmappable()
        # 获取注释数据的形状（高度和宽度）
        height, width = self.annot_data.shape
        # 创建网格，确定每个单元格中心的坐标
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        # 遍历每个单元格，添加文本标签
        for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                       mesh.get_array().flat, mesh.get_facecolors(),
                                       self.annot_data.flat):
            # 如果单元格不是遮罩，则计算亮度和文本颜色
            if m is not np.ma.masked:
                lum = relative_luminance(color)
                text_color = ".15" if lum > .408 else "w"
                # 格式化数值并创建注释文本
                annotation = ("{:" + self.fmt + "}").format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kws)
                # 在指定位置添加文本标签
                ax.text(x, y, annotation, **text_kwargs)

    # 根据标签的间隔确定要显示的刻度和标签
    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        # 标签的数量
        n = len(labels)
        # 如果 tickevery 为 0，则不返回任何刻度和标签
        if tickevery == 0:
            ticks, labels = [], []
        # 如果 tickevery 为 1，则每个标签都显示
        elif tickevery == 1:
            ticks, labels = np.arange(n) + .5, labels
        else:
            # 否则按照 tickevery 的间隔返回刻度和标签
            start, end, step = 0, n, tickevery
            ticks = np.arange(start, end, step) + .5
            labels = labels[start:end:step]
        return ticks, labels

    # 自动确定刻度和标签，以便最小化重叠
    def _auto_ticks(self, ax, labels, axis):
        """Determine ticks and ticklabels that minimize overlap."""
        # 获取图表的 DPI 转换
        transform = ax.figure.dpi_scale_trans.inverted()
        # 获取轴的窗口范围
        bbox = ax.get_window_extent().transformed(transform)
        # 根据轴和窗口大小确定字体大小
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        # 设置一个虚拟的刻度，并获取其标签的字体大小
        tick, = axis.set_ticks([0])
        fontsize = tick.label1.get_size()
        # 计算允许的最大刻度数
        max_ticks = int(size // (fontsize / 72))
        # 如果最大刻度数小于 1，则返回空刻度和标签
        if max_ticks < 1:
            return [], []
        # 根据最大刻度数确定每个标签的显示间隔
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        # 调用 _skip_ticks 方法返回最终的刻度和标签
        ticks, labels = self._skip_ticks(labels, tick_every)
        return ticks, labels
    def plot(self, ax, cax, kws):
        """Draw the heatmap on the provided Axes."""
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # setting vmin/vmax in addition to norm is deprecated
        # so avoid setting if norm is set
        if kws.get("norm") is None:
            # 设置默认的最小值和最大值，如果没有设置norm的话
            kws.setdefault("vmin", self.vmin)
            kws.setdefault("vmax", self.vmax)

        # Draw the heatmap using pcolormesh with specified colormap and other keyword arguments
        mesh = ax.pcolormesh(self.plot_data, cmap=self.cmap, **kws)

        # Set the axis limits based on the dimensions of the data
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))

        # Invert the y axis to display the plot in matrix form
        ax.invert_yaxis()

        # Add a colorbar if self.cbar is True
        if self.cbar:
            # Create a colorbar associated with the heatmap
            cb = ax.figure.colorbar(mesh, cax, ax, **self.cbar_kws)
            # Remove the outline of the colorbar
            cb.outline.set_linewidth(0)
            # If 'rasterized' is set to True in kws, rasterize the colorbar
            if kws.get('rasterized', False):
                cb.solids.set_rasterized(True)

        # Add row and column labels based on self.xticks and self.yticks
        if isinstance(self.xticks, str) and self.xticks == "auto":
            # Automatically determine xticks and xticklabels
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, 0)
        else:
            xticks, xticklabels = self.xticks, self.xticklabels

        if isinstance(self.yticks, str) and self.yticks == "auto":
            # Automatically determine yticks and yticklabels
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, 1)
        else:
            yticks, yticklabels = self.yticks, self.yticklabels

        # Set xticks and yticks
        ax.set(xticks=xticks, yticks=yticks)
        # Set xticklabels and yticklabels with optional vertical rotation
        xtl = ax.set_xticklabels(xticklabels)
        ytl = ax.set_yticklabels(yticklabels, rotation="vertical")
        # Adjust the vertical alignment of yticklabels
        plt.setp(ytl, va="center")  # GH2484

        # Adjust the rotation of xticklabels and yticklabels if they overlap
        _draw_figure(ax.figure)
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Set the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)

        # Annotate the cells of the heatmap with formatted values if self.annot is True
        if self.annot:
            self._annotate_heatmap(ax, mesh)
# 绘制矩形数据作为色彩编码矩阵的热图。

# 这是一个基于 Axes 的函数，如果没有提供 `ax` 参数，则会将热图绘制到当前活动的 Axes 中。
# 除非 `cbar` 设为 False 或者提供了单独的 Axes 给 `cbar_ax` 参数，否则将使用部分 Axes 空间来绘制颜色映射条。

Parameters
----------
data : rectangular dataset
    可转换为 ndarray 的二维数据集。如果提供了 Pandas DataFrame，将使用索引/列信息来标记行和列。
vmin, vmax : floats, optional
    确定颜色映射的值范围，否则将根据数据和其他关键字参数推断。
cmap : matplotlib colormap name or object, or list of colors, optional
    将数据值映射到颜色空间的方式。如果未提供，则默认值取决于是否设置了 `center`。
center : float, optional
    绘制发散数据时，用于居中颜色映射的值。如果未指定 `cmap`，使用此参数将更改默认的颜色映射。
robust : bool, optional
    如果为 True，并且 `vmin` 或 `vmax` 不存在，则使用强健的分位数计算颜色映射范围，而不是极值。
annot : bool or rectangular dataset, optional
    如果为 True，在每个单元格中写入数据值。如果形状与 `data` 相同的类数组，则使用此参数替代数据来注释热图。
    注意，对于 DataFrame，匹配的是位置而不是索引。
fmt : str, optional
    添加注释时使用的字符串格式化代码。
annot_kws : dict of key, value mappings, optional
    当 `annot` 设为 True 时，用于 :meth:`matplotlib.axes.Axes.text` 的关键字参数。
linewidths : float, optional
    分隔每个单元格的线宽度。
linecolor : color, optional
    分隔每个单元格的线颜色。
cbar : bool, optional
    是否绘制颜色条。
cbar_kws : dict of key, value mappings, optional
    用于 :meth:`matplotlib.figure.Figure.colorbar` 的关键字参数。
cbar_ax : matplotlib Axes, optional
    绘制颜色条的 Axes，否则从主 Axes 中占用空间。
square : bool, optional
    如果为 True，则将 Axes 的 aspect 设置为 "equal"，使每个单元格都是正方形。
    # xticklabels, yticklabels：可选参数，控制横纵坐标轴的标签显示方式。
    # 如果为 True，则绘制数据框的列名作为标签；如果为 False，则不绘制列名；如果是类列表，则使用这些替代标签作为横坐标轴标签。
    # 如果是整数，则使用列名，但只绘制每个 n 个标签；如果为 "auto"，尝试密集绘制不重叠的标签。
    
    # mask：可选参数，布尔数组或数据框，指示哪些单元格不显示数据。
    # 当 mask 的值为 True 时，对应的单元格将被遮蔽。自动屏蔽缺失值的单元格。
    
    # ax：matplotlib Axes 对象，可选参数，用于绘制图形的坐标轴。
    # 如果未提供，则使用当前活动的 Axes。
    
    # kwargs：其他关键字参数，传递给 matplotlib.axes.Axes.pcolormesh 方法。
    
    """
    Draw a heatmap of the given data.
    
    Parameters
    ----------
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        控制横纵坐标轴的标签显示方式。
    mask : bool array or DataFrame, optional
        指示哪些单元格不显示数据。
    ax : matplotlib Axes, optional
        用于绘制图形的坐标轴。
    kwargs : other keyword arguments
        传递给 matplotlib.axes.Axes.pcolormesh 的其他关键字参数。
    
    Returns
    -------
    ax : matplotlib Axes
        带有热图的 Axes 对象。
    
    See Also
    --------
    clustermap : 使用层次聚类排列行和列来绘制矩阵。
    
    Examples
    --------
    
    .. include:: ../docstrings/heatmap.rst
    """
    
    # 初始化 HeatMapper 对象
    plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                          annot_kws, cbar, cbar_kws, xticklabels,
                          yticklabels, mask)
    
    # 添加 pcolormesh 方法的 kwargs 参数
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor
    
    # 绘制图形并返回 Axes 对象
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax
class _DendrogramPlotter:
    """用于绘制数据行/列之间相似性树的对象"""

    def __init__(self, data, linkage, metric, method, axis, label, rotate):
        """绘制数据列之间关系的树状图

        Parameters
        ----------
        data : pandas.DataFrame
            矩形数据
        """
        self.axis = axis
        # 如果 axis 为 1，则对数据进行转置
        if self.axis == 1:
            data = data.T

        # 如果 data 是 pandas.DataFrame，则将其转换为 numpy 数组
        if isinstance(data, pd.DataFrame):
            array = data.values
        else:
            array = np.asarray(data)
            data = pd.DataFrame(array)

        self.array = array
        self.data = data

        self.shape = self.data.shape
        self.metric = metric
        self.method = method
        self.axis = axis
        self.label = label
        self.rotate = rotate

        # 如果没有提供 linkage，则计算基于数组的 linkage
        if linkage is None:
            self.linkage = self._calculate_linkage_scipy
        else:
            self.linkage = linkage
        # 计算并获取树状图数据
        self.dendrogram = self.calculate_dendrogram()

        # 树状图的端点总是在 5 的倍数上，具体原因未知
        ticks = 10 * np.arange(self.data.shape[0]) + 5

        # 如果需要标签
        if self.label:
            # 获取标签对应的刻度标签，并根据重新排序的索引进行排序
            ticklabels = _index_to_ticklabels(self.data.index)
            ticklabels = [ticklabels[i] for i in self.reordered_ind]
            # 根据 rotate 参数确定坐标轴及其标签的设置
            if self.rotate:
                self.xticks = []
                self.yticks = ticks
                self.xticklabels = []
                self.yticklabels = ticklabels
                self.ylabel = _index_to_label(self.data.index)
                self.xlabel = ''
            else:
                self.xticks = ticks
                self.yticks = []
                self.xticklabels = ticklabels
                self.yticklabels = []
                self.ylabel = ''
                self.xlabel = _index_to_label(self.data.index)
        else:
            # 如果不需要标签，则清空所有标签和标签列表
            self.xticks, self.yticks = [], []
            self.yticklabels, self.xticklabels = [], []
            self.xlabel, self.ylabel = '', ''

        # 获取树状图的依赖坐标和独立坐标
        self.dependent_coord = self.dendrogram['dcoord']
        self.independent_coord = self.dendrogram['icoord']

    def _calculate_linkage_scipy(self):
        """使用 SciPy 计算数据的 linkage"""
        linkage = hierarchy.linkage(self.array, method=self.method,
                                    metric=self.metric)
        return linkage
    def _calculate_linkage_fastcluster(self):
        import fastcluster
        # 引入 fastcluster 库
        # Fastcluster 库提供了一个节省内存的向量化版本，但仅适用于特定的链接方法，主要是欧氏距离度量
        # vector_methods = ('single', 'centroid', 'median', 'ward')
        euclidean_methods = ('centroid', 'median', 'ward')
        # 判断当前距离度量是否为欧氏距离，并且链接方法在支持的欧氏距离方法列表中
        euclidean = self.metric == 'euclidean' and self.method in \
            euclidean_methods
        # 如果满足欧氏距离条件或者链接方法为 'single'，则使用 fastcluster 库的向量化函数进行计算
        if euclidean or self.method == 'single':
            return fastcluster.linkage_vector(self.array,
                                              method=self.method,
                                              metric=self.metric)
        else:
            # 否则，使用 fastcluster 库的普通链接函数进行计算
            linkage = fastcluster.linkage(self.array, method=self.method,
                                          metric=self.metric)
            return linkage

    @property
    def calculated_linkage(self):
        try:
            # 尝试使用快速的 fastcluster 计算链接
            return self._calculate_linkage_fastcluster()
        except ImportError:
            # 如果 fastcluster 库导入失败，则进行处理
            if np.prod(self.shape) >= 10000:
                msg = ("Clustering large matrix with scipy. Installing "
                       "`fastcluster` may give better performance.")
                # 发出警告信息
                warnings.warn(msg)

        # 如果 fastcluster 计算失败或者未安装，则使用 scipy 库计算链接
        return self._calculate_linkage_scipy()

    def calculate_dendrogram(self):
        """Calculates a dendrogram based on the linkage matrix

        Made a separate function, not a property because don't want to
        recalculate the dendrogram every time it is accessed.

        Returns
        -------
        dendrogram : dict
            Dendrogram dictionary as returned by scipy.cluster.hierarchy
            .dendrogram. The important key-value pairing is
            "reordered_ind" which indicates the re-ordering of the matrix
        """
        # 计算基于链接矩阵的树状图
        return hierarchy.dendrogram(self.linkage, no_plot=True,
                                    color_threshold=-np.inf)

    @property
    def reordered_ind(self):
        """Indices of the matrix, reordered by the dendrogram"""
        # 返回根据树状图重新排序的矩阵的索引
        return self.dendrogram['leaves']
    def plot(self, ax, tree_kws):
        """Plots a dendrogram of the similarities between data on the axes

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted

        """
        # 如果 tree_kws 为 None，则设置为空字典；否则复制 tree_kws
        tree_kws = {} if tree_kws is None else tree_kws.copy()
        # 设置 dendrogram 线条的默认线宽
        tree_kws.setdefault("linewidths", .5)
        # 设置 dendrogram 线条的颜色，默认为灰色 (.2, .2, .2)
        tree_kws.setdefault("colors", tree_kws.pop("color", (.2, .2, .2)))

        # 根据对象的旋转状态和轴的方向选择坐标顺序
        if self.rotate and self.axis == 0:
            coords = zip(self.dependent_coord, self.independent_coord)
        else:
            coords = zip(self.independent_coord, self.dependent_coord)
        # 创建 LineCollection 对象，用于绘制 dendrogram
        lines = LineCollection([list(zip(x, y)) for x, y in coords],
                               **tree_kws)

        # 将 LineCollection 添加到指定的 Axes 对象中
        ax.add_collection(lines)
        # 计算重新排序后的索引的数量
        number_of_leaves = len(self.reordered_ind)
        # 计算依赖坐标的最大值
        max_dependent_coord = max(map(max, self.dependent_coord))

        # 如果对象旋转，则调整 y 轴刻度的位置到右侧
        if self.rotate:
            ax.yaxis.set_ticks_position('right')

            # 根据常量设置 y 轴的显示范围
            ax.set_ylim(0, number_of_leaves * 10)
            # 根据常量设置 x 轴的显示范围
            ax.set_xlim(0, max_dependent_coord * 1.05)

            # 反转 x 轴和 y 轴的显示顺序
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:
            # 根据常量设置 x 轴的显示范围
            ax.set_xlim(0, number_of_leaves * 10)
            # 根据常量设置 y 轴的显示范围
            ax.set_ylim(0, max_dependent_coord * 1.05)

        # 调用 despine 函数，移除 Axes 对象的底部和左侧边框
        despine(ax=ax, bottom=True, left=True)

        # 设置 Axes 对象的标签和刻度
        ax.set(xticks=self.xticks, yticks=self.yticks,
               xlabel=self.xlabel, ylabel=self.ylabel)
        # 设置 x 轴刻度标签和 y 轴刻度标签
        xtl = ax.set_xticklabels(self.xticklabels)
        ytl = ax.set_yticklabels(self.yticklabels, rotation='vertical')

        # 强制绘制图形以避免 matplotlib 窗口错误
        _draw_figure(ax.figure)

        # 如果 y 轴刻度标签存在且存在重叠，则水平旋转标签
        if len(ytl) > 0 and axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")
        # 如果 x 轴刻度标签存在且存在重叠，则垂直旋转标签
        if len(xtl) > 0 and axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")

        # 返回对象本身，用于方法链
        return self
# 定义一个函数用于绘制数据矩阵的树状图
def dendrogram(
    data, *,
    linkage=None, axis=1, label=True, metric='euclidean',
    method='average', rotate=False, tree_kws=None, ax=None
):
    """Draw a tree diagram of relationships within a matrix

    Parameters
    ----------
    data : pandas.DataFrame
        Rectangular data  # 输入的矩阵数据，应为 pandas 的 DataFrame 格式
    linkage : numpy.array, optional
        Linkage matrix  # 可选参数，用于指定链接矩阵
    axis : int, optional
        Which axis to use to calculate linkage. 0 is rows, 1 is columns.
        # 指定用于计算链接的轴向，0 表示行，1 表示列
    label : bool, optional
        If True, label the dendrogram at leaves with column or row names
        # 是否在叶子节点上标注列或行的名称
    metric : str, optional
        Distance metric. Anything valid for scipy.spatial.distance.pdist
        # 距离度量标准，可使用 scipy.spatial.distance.pdist 支持的任何有效度量方法
    method : str, optional
        Linkage method to use. Anything valid for
        scipy.cluster.hierarchy.linkage
        # 使用的链接方法，可使用 scipy.cluster.hierarchy.linkage 支持的任何有效方法
    rotate : bool, optional
        When plotting the matrix, whether to rotate it 90 degrees
        counter-clockwise, so the leaves face right
        # 绘制矩阵时，是否将其逆时针旋转90度，使叶子朝向右侧
    tree_kws : dict, optional
        Keyword arguments for the ``matplotlib.collections.LineCollection``
        that is used for plotting the lines of the dendrogram tree.
        # 绘制树状图线条时使用的关键字参数，用于 ``matplotlib.collections.LineCollection``
    ax : matplotlib axis, optional
        Axis to plot on, otherwise uses current axis
        # 绘图所用的 matplotlib 坐标轴，如果未指定则使用当前轴

    Returns
    -------
    dendrogramplotter : _DendrogramPlotter
        A Dendrogram plotter object.
        # 返回一个树状图绘制器对象

    Notes
    -----
    Access the reordered dendrogram indices with
    dendrogramplotter.reordered_ind
    # 可通过 dendrogramplotter.reordered_ind 访问重新排序后的树状图索引

    """
    # 如果没有安装 scipy，则抛出运行时错误
    if _no_scipy:
        raise RuntimeError("dendrogram requires scipy to be installed")

    # 创建一个 _DendrogramPlotter 对象，用于绘制树状图
    plotter = _DendrogramPlotter(data, linkage=linkage, axis=axis,
                                 metric=metric, method=method,
                                 label=label, rotate=rotate)
    # 如果未指定绘图的坐标轴，则使用当前的 matplotlib 坐标轴
    if ax is None:
        ax = plt.gca()

    # 调用 plotter 对象的 plot 方法进行绘图，返回绘图结果
    return plotter.plot(ax=ax, tree_kws=tree_kws)
    def _preprocess_colors(self, data, colors, axis):
        """Preprocess {row/col}_colors to extract labels and convert colors."""
        labels = None  # 初始化标签为空

        if colors is not None:  # 如果颜色参数不为空
            if isinstance(colors, (pd.DataFrame, pd.Series)):  # 如果颜色参数是DataFrame或Series类型

                # 如果数据没有索引，且axis为0或1，则引发TypeError异常
                if (not hasattr(data, "index") and axis == 0) or (
                    not hasattr(data, "columns") and axis == 1
                ):
                    axis_name = "col" if axis else "row"  # 根据axis的值确定axis_name
                    msg = (f"{axis_name}_colors indices can't be matched with data "
                           f"indices. Provide {axis_name}_colors as a non-indexed "
                           "datatype, e.g. by using `.to_numpy()``")  # 准备错误消息
                    raise TypeError(msg)  # 抛出类型错误异常

                # 确保颜色索引与数据索引匹配
                if axis == 0:
                    colors = colors.reindex(data.index)  # 根据数据的行索引重新索引颜色
                else:
                    colors = colors.reindex(data.columns)  # 根据数据的列索引重新索引颜色

                # 将缺失值NaN替换为白色
                # TODO 这里应当将缺失值设置为透明而非白色
                colors = colors.astype(object).fillna('white')  # 将颜色数据类型转换为对象并填充NaN为白色

                # 从DataFrame或Series中提取颜色值和标签
                if isinstance(colors, pd.DataFrame):
                    labels = list(colors.columns)  # 如果是DataFrame，标签为列名列表
                    colors = colors.T.values  # 转置后取值为颜色数据
                else:
                    if colors.name is None:
                        labels = [""]  # 如果没有列名，标签为空字符串列表
                    else:
                        labels = [colors.name]  # 否则标签为颜色Series的名称
                    colors = colors.values  # 颜色为Series的值数组

            colors = _convert_colors(colors)  # 调用_convert_colors函数转换颜色格式

        return colors, labels  # 返回处理后的颜色数据和标签

    def format_data(self, data, pivot_kws, z_score=None,
                    standard_scale=None):
        """Extract variables from data or use directly."""

        # 数据要么已经是二维矩阵格式，要么需要进行透视
        if pivot_kws is not None:
            data2d = data.pivot(**pivot_kws)  # 如果有透视参数，对数据进行透视处理
        else:
            data2d = data  # 否则数据直接使用

        if z_score is not None and standard_scale is not None:
            raise ValueError(
                'Cannot perform both z-scoring and standard-scaling on data')  # 如果同时指定z-score和标准化，抛出值错误异常

        if z_score is not None:
            data2d = self.z_score(data2d, z_score)  # 如果指定z-score，对数据进行z-score标准化处理
        if standard_scale is not None:
            data2d = self.standard_scale(data2d, standard_scale)  # 如果指定标准化，对数据进行标准化处理
        return data2d  # 返回处理后的数据
    def z_score(data2d, axis=1):
        """Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.
        """
        # If axis is 1, work directly with data2d; otherwise, transpose data2d
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        # Calculate z-score normalization across the specified axis
        z_scored = (z_scored - z_scored.mean()) / z_scored.std()

        # If axis is 1, return z_scored; otherwise, transpose z_scored before returning
        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def standard_scale(data2d, axis=1):
        """Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        # Normalize the data to range from 0 to 1
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        # Calculate min-max normalization across the specified axis
        subtract = standardized.min()
        standardized = (standardized - subtract) / (standardized.max() - standardized.min())

        # If axis is 1, return standardized; otherwise, transpose standardized before returning
        if axis == 1:
            return standardized
        else:
            return standardized.T

    def dim_ratios(self, colors, dendrogram_ratio, colors_ratio):
        """Get the proportions of the figure taken up by each axes."""
        ratios = [dendrogram_ratio]

        if colors is not None:
            # Check if colors have more than 2 dimensions to determine the count of colors
            if np.ndim(colors) > 2:
                n_colors = len(colors)
            else:
                n_colors = 1

            ratios += [n_colors * colors_ratio]

        # Calculate the ratio for the heatmap itself
        ratios.append(1 - sum(ratios))

        return ratios
    def plot_dendrograms(self, row_cluster, col_cluster, metric, method,
                         row_linkage, col_linkage, tree_kws):
        # 绘制行谱系图
        if row_cluster:
            # 如果需要绘制行谱系图，则调用 dendrogram 绘制，并设置相关参数
            self.dendrogram_row = dendrogram(
                self.data2d, metric=metric, method=method, label=False, axis=0,
                ax=self.ax_row_dendrogram, rotate=True, linkage=row_linkage,
                tree_kws=tree_kws
            )
        else:
            # 如果不需要绘制行谱系图，则清空行谱系图的坐标轴刻度
            self.ax_row_dendrogram.set_xticks([])
            self.ax_row_dendrogram.set_yticks([])
        
        # 绘制列谱系图
        if col_cluster:
            # 如果需要绘制列谱系图，则调用 dendrogram 绘制，并设置相关参数
            self.dendrogram_col = dendrogram(
                self.data2d, metric=metric, method=method, label=False,
                axis=1, ax=self.ax_col_dendrogram, linkage=col_linkage,
                tree_kws=tree_kws
            )
        else:
            # 如果不需要绘制列谱系图，则清空列谱系图的坐标轴刻度
            self.ax_col_dendrogram.set_xticks([])
            self.ax_col_dendrogram.set_yticks([])
        
        # 调整行谱系图和列谱系图的外边框
        despine(ax=self.ax_row_dendrogram, bottom=True, left=True)
        despine(ax=self.ax_col_dendrogram, bottom=True, left=True)
    def plot_colors(self, xind, yind, **kws):
        """Plots color labels between the dendrogram and the heatmap
        
        Parameters
        ----------
        xind : int
            Index for columns in the heatmap
        yind : int
            Index for rows in the heatmap
        **kws : dict
            Additional keyword arguments for customization
        
        """
        
        # Remove any custom colormap and centering
        # TODO this code has consistently caused problems when we
        # have missed kwargs that need to be excluded that it might
        # be better to rewrite *in*clusively.
        kws = kws.copy()
        kws.pop('cmap', None)       # Remove 'cmap' from kws
        kws.pop('norm', None)       # Remove 'norm' from kws
        kws.pop('center', None)     # Remove 'center' from kws
        kws.pop('annot', None)      # Remove 'annot' from kws
        kws.pop('vmin', None)       # Remove 'vmin' from kws
        kws.pop('vmax', None)       # Remove 'vmax' from kws
        kws.pop('robust', None)     # Remove 'robust' from kws
        kws.pop('xticklabels', None) # Remove 'xticklabels' from kws
        kws.pop('yticklabels', None) # Remove 'yticklabels' from kws
        
        # Plot the row colors
        if self.row_colors is not None:
            # Convert row_colors to a matrix and determine the colormap
            matrix, cmap = self.color_list_to_matrix_and_cmap(
                self.row_colors, yind, axis=0)

            # Get row_color labels
            if self.row_color_labels is not None:
                row_color_labels = self.row_color_labels
            else:
                row_color_labels = False

            # Plot the heatmap for row colors
            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_row_colors,
                    xticklabels=row_color_labels, yticklabels=False, **kws)

            # Adjust rotation of labels if row_color_labels is not False
            if row_color_labels is not False:
                plt.setp(self.ax_row_colors.get_xticklabels(), rotation=90)
        else:
            # Remove left and bottom spines if no row colors
            despine(self.ax_row_colors, left=True, bottom=True)

        # Plot the column colors
        if self.col_colors is not None:
            # Convert col_colors to a matrix and determine the colormap
            matrix, cmap = self.color_list_to_matrix_and_cmap(
                self.col_colors, xind, axis=1)

            # Get col_color labels
            if self.col_color_labels is not None:
                col_color_labels = self.col_color_labels
            else:
                col_color_labels = False

            # Plot the heatmap for column colors
            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_col_colors,
                    xticklabels=False, yticklabels=col_color_labels, **kws)

            # Adjust rotation of labels and place on the right side
            if col_color_labels is not False:
                self.ax_col_colors.yaxis.tick_right()
                plt.setp(self.ax_col_colors.get_yticklabels(), rotation=0)
        else:
            # Remove left and bottom spines if no column colors
            despine(self.ax_col_colors, left=True, bottom=True)
    def plot_matrix(self, colorbar_kws, xind, yind, **kws):
        # 将数据和遮罩根据给定的行列索引重新组织
        self.data2d = self.data2d.iloc[yind, xind]
        self.mask = self.mask.iloc[yind, xind]

        # 尝试重新组织指定的刻度标签（如果提供了的话）
        xtl = kws.pop("xticklabels", "auto")
        try:
            xtl = np.asarray(xtl)[xind]
        except (TypeError, IndexError):
            pass
        ytl = kws.pop("yticklabels", "auto")
        try:
            ytl = np.asarray(ytl)[yind]
        except (TypeError, IndexError):
            pass

        # 根据热图重新组织注释以匹配数据
        annot = kws.pop("annot", None)
        if annot is None or annot is False:
            pass
        else:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != self.data2d.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
                annot_data = annot_data[yind][:, xind]
            annot = annot_data

        # 在集群图中设置 ax_cbar=None 表示不显示颜色条
        kws.setdefault("cbar", self.ax_cbar is not None)
        # 调用 heatmap 函数绘制热图，并设置参数
        heatmap(self.data2d, ax=self.ax_heatmap, cbar_ax=self.ax_cbar,
                cbar_kws=colorbar_kws, mask=self.mask,
                xticklabels=xtl, yticklabels=ytl, annot=annot, **kws)

        # 获取热图的 y 轴刻度标签，并根据需要设置旋转
        ytl = self.ax_heatmap.get_yticklabels()
        ytl_rot = None if not ytl else ytl[0].get_rotation()
        self.ax_heatmap.yaxis.set_ticks_position('right')
        self.ax_heatmap.yaxis.set_label_position('right')
        if ytl_rot is not None:
            ytl = self.ax_heatmap.get_yticklabels()
            plt.setp(ytl, rotation=ytl_rot)

        # 设置紧凑布局参数，确保图形布局合理
        tight_params = dict(h_pad=.02, w_pad=.02)
        if self.ax_cbar is None:
            # 如果没有颜色条，使用紧凑布局参数调整图形布局
            self._figure.tight_layout(**tight_params)
        else:
            # 如果有颜色条，先关闭颜色条的轴，以避免其刻度干扰布局，
            # 然后使用紧凑布局参数调整整体布局，并最后设置颜色条的位置
            self.ax_cbar.set_axis_off()
            self._figure.tight_layout(**tight_params)
            self.ax_cbar.set_axis_on()
            self.ax_cbar.set_position(self.cbar_pos)
    def plot(self, metric, method, colorbar_kws, row_cluster, col_cluster,
             row_linkage, col_linkage, tree_kws, **kws):
        # 如果参数中包含 square=True，则在 clustergrid 的多轴布局中不兼容此设置
        # 因此在 clustermap 中忽略 square=True，并发出警告信息
        if kws.get("square", False):
            msg = "``square=True`` ignored in clustermap"
            warnings.warn(msg)
            kws.pop("square")  # 移除参数中的 square=True

        # 如果 colorbar_kws 为 None，则将其设为一个空字典
        colorbar_kws = {} if colorbar_kws is None else colorbar_kws

        # 绘制行和列的树状图
        self.plot_dendrograms(row_cluster, col_cluster, metric, method,
                              row_linkage=row_linkage, col_linkage=col_linkage,
                              tree_kws=tree_kws)

        # 尝试获取列的重排序索引
        try:
            xind = self.dendrogram_col.reordered_ind
        except AttributeError:
            xind = np.arange(self.data2d.shape[1])  # 若不存在，则使用默认的索引

        # 尝试获取行的重排序索引
        try:
            yind = self.dendrogram_row.reordered_ind
        except AttributeError:
            yind = np.arange(self.data2d.shape[0])  # 若不存在，则使用默认的索引

        # 绘制颜色相关的部分
        self.plot_colors(xind, yind, **kws)

        # 绘制矩阵部分，包括颜色条
        self.plot_matrix(colorbar_kws, xind, yind, **kws)

        # 返回 self，即 ClusterGrid 对象本身
        return self
# 定义一个函数用于绘制基于层次聚类的热力图，需要依赖于 scipy 库

def clustermap(
    data, *,
    pivot_kws=None,  # 如果 data 是一个整洁的数据框架，可以提供用于 pivot 的关键字参数
    method='average',  # 计算聚类时使用的链接方法，默认为 'average'
    metric='euclidean',  # 计算数据距离时使用的度量方式，默认为欧几里得距离
    z_score=None,  # 是否对行或列进行 z-score 标准化，0 表示行，1 表示列，默认为 None
    standard_scale=None,  # 是否对行或列进行标准化，0 表示行，1 表示列，默认为 None
    figsize=(10, 10),  # 图像的总体尺寸，默认为 (10, 10)
    cbar_kws=None,  # 传递给 heatmap 中 cbar_kws 的关键字参数，例如用于添加颜色条的标签
    row_cluster=True,  # 是否对行进行聚类，默认为 True
    col_cluster=True,  # 是否对列进行聚类，默认为 True
    row_linkage=None,  # 预先计算的行链接矩阵，用于聚类行，默认为 None
    col_linkage=None,  # 预先计算的列链接矩阵，用于聚类列，默认为 None
    row_colors=None,  # 行颜色标记，用于可视化行分组，默认为 None
    col_colors=None,  # 列颜色标记，用于可视化列分组，默认为 None
    mask=None,  # 用于遮蔽矩阵中的特定区域，默认为 None
    dendrogram_ratio=.2,  # 用于设置树形图（树状图）高度的比率，默认为 0.2
    colors_ratio=0.03,  # 用于设置颜色条高度的比率，默认为 0.03
    cbar_pos=(.02, .8, .05, .18),  # 颜色条位置的参数 (left, bottom, width, height)，默认为 (.02, .8, .05, .18)
    tree_kws=None,  # 传递给树形图绘制函数的关键字参数，默认为 None
    **kwargs  # 其他未指定的关键字参数，用于接收额外的参数
):
    """
    绘制作为层次聚类热力图的矩阵数据。

    此函数依赖于 scipy 库。

    Parameters
    ----------
    data : 2D array-like
        用于聚类的矩形数据。不能包含缺失值。
    pivot_kws : dict, optional
        如果 `data` 是一个整洁的数据框架，可以提供用于 pivot 的关键字参数。
    method : str, optional
        用于计算聚类的链接方法。参见 :func:`scipy.cluster.hierarchy.linkage` 文档获取更多信息。
    metric : str, optional
        用于数据的距离度量。参见 :func:`scipy.spatial.distance.pdist` 文档获取更多选项。
        可以为行和列使用不同的度量（或方法），可以分别构造每个链接矩阵并提供它们作为 `{row,col}_linkage`。
    z_score : int or None, optional
        要么为 0（行）或 1（列）。是否计算行或列的 z-score 标准化。
        z-score 为 z = (x - mean) / std，因此每行（列）的值将减去行（列）的平均值，然后除以行（列）的标准差。
        这确保每行（列）的均值为 0，方差为 1。
    standard_scale : int or None, optional
        要么为 0（行）或 1（列）。是否对该维度进行标准化，即对每行或每列，减去最小值并除以最大值。
    figsize : tuple of (width, height), optional
        图像的总体尺寸。
    cbar_kws : dict, optional
        传递给 `heatmap` 中 `cbar_kws` 的关键字参数，例如用于添加颜色条的标签。
    {row,col}_cluster : bool, optional
        如果 ``True``，对 {行, 列} 进行聚类。
    {row,col}_linkage : :class:`numpy.ndarray`, optional
        预先计算的 {行, 列} 链接矩阵。参见 :func:`scipy.cluster.hierarchy.linkage` 获取具体格式。
    ```
    {row,col}_colors : list-like or pandas DataFrame/Series, optional
        # 用于指定行或列的颜色列表，可以是列表形式或者 pandas 的 DataFrame/Series。用于标记是否样本在同一组内聚集。
        # 可以使用嵌套列表或者 DataFrame 来进行多级标签的颜色标记。如果传入 pandas.DataFrame 或 pandas.Series，
        # 颜色标签将从DataFrame的列名或Series的名称中提取。DataFrame/Series 的颜色也会根据它们的索引与数据匹配，确保颜色以正确的顺序绘制。
    mask : bool array or DataFrame, optional
        # 如果传入，则标记数据中 mask 为 True 的单元格不显示。自动屏蔽缺失值的单元格。仅用于可视化，不用于计算。
    {dendrogram,colors}_ratio : float, or pair of floats, optional
        # 图形大小中用于两个边缘元素的比例。如果给出一对值，则分别对应行和列的比例。
    cbar_pos : tuple of (left, bottom, width, height), optional
        # 颜色条轴在图中的位置。设置为 ``None`` 将禁用颜色条。
    tree_kws : dict, optional
        # 传递给用于绘制树状图的 :class:`matplotlib.collections.LineCollection` 的参数。
    kwargs : other keyword arguments
        # 所有其他关键字参数都会传递给 :func:`heatmap` 函数。

    Returns
    -------
    :class:`ClusterGrid`
        # 返回一个 :class:`ClusterGrid` 实例。

    See Also
    --------
    heatmap : Plot rectangular data as a color-encoded matrix.
        # 参见：heatmap：将矩形数据绘制为颜色编码矩阵。

    Notes
    -----
    The returned object has a ``savefig`` method that should be used if you
    want to save the figure object without clipping the dendrograms.
        # 返回的对象具有 ``savefig`` 方法，如果要保存图形对象而不剪裁树状图，则应使用该方法。

    To access the reordered row indices, use:
    ``clustergrid.dendrogram_row.reordered_ind``
        # 要访问重新排序后的行索引，请使用：``clustergrid.dendrogram_row.reordered_ind``

    Column indices, use:
    ``clustergrid.dendrogram_col.reordered_ind``
        # 要访问重新排序后的列索引，请使用：``clustergrid.dendrogram_col.reordered_ind``

    Examples
    --------

    .. include:: ../docstrings/clustermap.rst
        # 查看示例和更多细节，请参考clustermap的文档。
```