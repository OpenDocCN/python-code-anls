# `D:\src\scipysrc\seaborn\seaborn\axisgrid.py`

```
from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
    adjust_legend_subtitles,
    set_hls_values,
    _check_argument,
    _draw_figure,
    _disable_autolayout
)
from .palettes import color_palette, blend_palette
from ._docstrings import (
    DocstringComponents,
    _core_docs,
)

__all__ = ["FacetGrid", "PairGrid", "JointGrid", "pairplot", "jointplot"]


_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
)

# 定义一个基类用于子图网格
class _BaseGrid:
    """Base class for grids of subplots."""

    # 设置子图网格中每个 Axes 的属性
    def set(self, **kwargs):
        """Set attributes on each subplot Axes."""
        for ax in self.axes.flat:
            if ax is not None:  # 处理已移除的 Axes
                ax.set(**kwargs)
        return self

    @property
    def fig(self):
        """DEPRECATED: prefer the `figure` property."""
        # Grid.figure 是首选，因为它与 Axes 属性名称匹配。
        # 但由于维护此属性的负担很小，
        # 我们暂时慢慢地将其标记为废弃。目前只在文档字符串中标注其废弃；
        # 在版本 0.13 中添加警告，并最终移除它。
        return self._figure

    @property
    def figure(self):
        """Access the :class:`matplotlib.figure.Figure` object underlying the grid."""
        return self._figure

    def apply(self, func, *args, **kwargs):
        """
        Pass the grid to a user-supplied function and return self.

        The `func` must accept an object of this type for its first
        positional argument. Additional arguments are passed through.
        The return value of `func` is ignored; this method returns self.
        See the `pipe` method if you want the return value.

        Added in v0.12.0.

        """
        func(self, *args, **kwargs)
        return self

    def pipe(self, func, *args, **kwargs):
        """
        Pass the grid to a user-supplied function and return its value.

        The `func` must accept an object of this type for its first
        positional argument. Additional arguments are passed through.
        The return value of `func` becomes the return value of this method.
        See the `apply` method if you want to return self instead.

        Added in v0.12.0.

        """
        return func(self, *args, **kwargs)
    # 定义一个方法用于保存绘图的图像
    def savefig(self, *args, **kwargs):
        """
        Save an image of the plot.

        This wraps :meth:`matplotlib.figure.Figure.savefig`, using bbox_inches="tight"
        by default. Parameters are passed through to the matplotlib function.

        """
        # 复制传入的关键字参数
        kwargs = kwargs.copy()
        # 如果未指定，则设置bbox_inches为"tight"，这是保存图像时的默认设置
        kwargs.setdefault("bbox_inches", "tight")
        # 调用matplotlib的Figure.savefig方法保存图像，传入所有位置参数和关键字参数
        self.figure.savefig(*args, **kwargs)
class Grid(_BaseGrid):
    """A grid that can have multiple subplots and an external legend."""
    _margin_titles = False  # 设置边缘标题为 False
    _legend_out = True  # 设置图例位置在外部

    def __init__(self):
        self._tight_layout_rect = [0, 0, 1, 1]  # 初始化紧凑布局的矩形区域为整个图像
        self._tight_layout_pad = None  # 初始化紧凑布局的填充为 None

        # 下面这个属性是外部设置的，用于处理不再在 Axes 上添加代理艺术家的新函数，需要更干净的方法来处理这个情况。
        self._extract_legend_handles = False  # 初始化为 False，用于从 Axes 中提取图例句柄

    def tight_layout(self, *args, **kwargs):
        """Call fig.tight_layout within rect that exclude the legend."""
        kwargs = kwargs.copy()
        kwargs.setdefault("rect", self._tight_layout_rect)  # 设置默认的矩形区域为紧凑布局的矩形区域
        if self._tight_layout_pad is not None:
            kwargs.setdefault("pad", self._tight_layout_pad)  # 如果紧凑布局的填充不为 None，则设置默认填充
        self._figure.tight_layout(*args, **kwargs)  # 调用 Figure 对象的 tight_layout 方法
        return self

    def _update_legend_data(self, ax):
        """Extract the legend data from an axes object and save it."""
        data = {}

        # 从图例直接获取数据，适用于不再添加标记代理艺术家的新函数
        if ax.legend_ is not None and self._extract_legend_handles:
            handles = get_legend_handles(ax.legend_)  # 获取图例句柄
            labels = [t.get_text() for t in ax.legend_.texts]  # 获取图例文本
            data.update({label: handle for handle, label in zip(handles, labels)})  # 更新数据字典

        handles, labels = ax.get_legend_handles_labels()  # 获取 Axes 的图例句柄和标签
        data.update({label: handle for handle, label in zip(handles, labels)})  # 更新数据字典

        self._legend_data.update(data)  # 更新 legend_data 属性的数据

        # 现在清除图例
        ax.legend_ = None  # 清除 Axes 的图例对象

    def _get_palette(self, data, hue, hue_order, palette):
        """Get a list of colors for the hue variable."""
        if hue is None:
            palette = color_palette(n_colors=1)  # 如果没有指定 hue，设置一个颜色列表，长度为 1
        else:
            hue_names = categorical_order(data[hue], hue_order)  # 获取分类变量的排序列表
            n_colors = len(hue_names)  # 获取分类变量的数量

            # 默认使用当前的颜色列表或者 HUSL 颜色空间
            if palette is None:
                current_palette = utils.get_color_cycle()  # 获取当前的颜色列表
                if n_colors > len(current_palette):
                    colors = color_palette("husl", n_colors)  # 如果颜色数量大于当前列表，使用 HUSL 颜色空间
                else:
                    colors = color_palette(n_colors=n_colors)  # 否则使用默认颜色列表

            # 允许通过字典映射 hue 变量名到颜色
            elif isinstance(palette, dict):
                color_names = [palette[h] for h in hue_names]  # 根据映射字典获取颜色名称
                colors = color_palette(color_names, n_colors)  # 创建颜色列表

            # 否则，假设 palette 是一个颜色列表
            else:
                colors = color_palette(palette, n_colors)  # 使用指定的颜色列表

            palette = color_palette(colors, n_colors)  # 创建颜色列表

        return palette  # 返回颜色列表

    @property
    def legend(self):
        """The :class:`matplotlib.legend.Legend` object, if present."""
        try:
            return self._legend  # 返回 legend 属性
        except AttributeError:
            return None  # 如果属性不存在，返回 None
    def tick_params(self, axis='both', **kwargs):
        """Modify the ticks, tick labels, and gridlines.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}
            The axis on which to apply the formatting.
            默认为 'both'，表示同时应用于 x 和 y 轴
        kwargs : keyword arguments
            Additional keyword arguments to pass to
            :meth:`matplotlib.axes.Axes.tick_params`.
            传递给 matplotlib.axes.Axes.tick_params 方法的额外关键字参数

        Returns
        -------
        self : Grid instance
            Returns self for easy chaining.
            返回当前的 Grid 实例，支持方法链操作

        """
        # 遍历图表中的所有坐标轴对象
        for ax in self.figure.axes:
            # 调用每个坐标轴对象的 tick_params 方法，应用指定的轴和其他参数
            ax.tick_params(axis=axis, **kwargs)
        # 返回当前的 Grid 实例，以支持方法链操作
        return self
_facet_docs = dict(

    data=dedent("""\
    data : DataFrame
        Tidy ("long-form") dataframe where each column is a variable and each
        row is an observation.\
    """),

    rowcol=dedent("""\
    row, col : vectors or keys in ``data``
        Variables that define subsets to plot on different facets.\
    """),

    rowcol_order=dedent("""\
    {row,col}_order : vector of strings
        Specify the order in which levels of the ``row`` and/or ``col`` variables
        appear in the grid of subplots.\
    """),

    col_wrap=dedent("""\
    col_wrap : int
        "Wrap" the column variable at this width, so that the column facets
        span multiple rows. Incompatible with a ``row`` facet.\
    """),

    share_xy=dedent("""\
    share{x,y} : bool, 'col', or 'row' optional
        If true, the facets will share y axes across columns and/or x axes
        across rows.\
    """),

    height=dedent("""\
    height : scalar
        Height (in inches) of each facet. See also: ``aspect``.\
    """),

    aspect=dedent("""\
    aspect : scalar
        Aspect ratio of each facet, so that ``aspect * height`` gives the width
        of each facet in inches.\
    """),

    palette=dedent("""\
    palette : palette name, list, or dict
        Colors to use for the different levels of the ``hue`` variable. Should
        be something that can be interpreted by :func:`color_palette`, or a
        dictionary mapping hue levels to matplotlib colors.\
    """),

    legend_out=dedent("""\
    legend_out : bool
        If ``True``, the figure size will be extended, and the legend will be
        drawn outside the plot on the center right.\
    """),

    margin_titles=dedent("""\
    margin_titles : bool
        If ``True``, the titles for the row variable are drawn to the right of
        the last column. This option is experimental and may not work in all
        cases.\
    """),

    facet_kws=dedent("""\
    facet_kws : dict
        Additional parameters passed to :class:`FacetGrid`.
    """),
)


class FacetGrid(Grid):
    """Multi-plot grid for plotting conditional relationships."""

    def __init__(
        self, data, *,
        row=None, col=None, hue=None, col_wrap=None,
        sharex=True, sharey=True, height=3, aspect=1, palette=None,
        row_order=None, col_order=None, hue_order=None, hue_kws=None,
        dropna=False, legend_out=True, despine=True,
        margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
        gridspec_kws=None,
    ):
        # 调用父类 Grid 的构造函数来初始化网格布局
        super().__init__(
            data=data, 
            row=row, 
            col=col, 
            hue=hue, 
            col_wrap=col_wrap, 
            sharex=sharex, 
            sharey=sharey, 
            height=height, 
            aspect=aspect, 
            palette=palette, 
            row_order=row_order, 
            col_order=col_order, 
            hue_order=hue_order, 
            hue_kws=hue_kws, 
            dropna=dropna, 
            legend_out=legend_out, 
            despine=despine, 
            margin_titles=margin_titles, 
            xlim=xlim, 
            ylim=ylim, 
            subplot_kws=subplot_kws, 
            gridspec_kws=gridspec_kws,
        )
    def facet_data(self):
        """Generator for name indices and data subsets for each facet.

        Yields
        ------
        (i, j, k), data_ijk : tuple of ints, DataFrame
            The ints provide an index into the {row, col, hue}_names attribute,
            and the dataframe contains a subset of the full data corresponding
            to each facet. The generator yields subsets that correspond with
            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
            is None.

        """
        # 获取数据集
        data = self.data

        # 构建行变量的掩码
        if self.row_names:
            row_masks = [data[self._row_var] == n for n in self.row_names]
        else:
            row_masks = [np.repeat(True, len(self.data))]  # 如果没有行变量名，所有行都为True

        # 构建列变量的掩码
        if self.col_names:
            col_masks = [data[self._col_var] == n for n in self.col_names]
        else:
            col_masks = [np.repeat(True, len(self.data))]  # 如果没有列变量名，所有列都为True

        # 构建色调变量的掩码
        if self.hue_names:
            hue_masks = [data[self._hue_var] == n for n in self.hue_names]
        else:
            hue_masks = [np.repeat(True, len(self.data))]  # 如果没有色调变量名，所有色调都为True

        # 主生成器循环
        for (i, row), (j, col), (k, hue) in product(enumerate(row_masks),
                                                    enumerate(col_masks),
                                                    enumerate(hue_masks)):
            # 组合所有掩码并过滤数据集中的非空值行
            data_ijk = data[row & col & hue & self._not_na]
            yield (i, j, k), data_ijk  # 生成器产出，包括索引元组和相应的数据子集
    def map_dataframe(self, func, *args, **kwargs):
        """
        类似于 `.map` 方法，但将 args 作为字符串传递，并插入 kwargs 中的数据。

        该方法适用于使用接受长格式 DataFrame 作为 `data` 关键字参数，并使用字符串变量名访问该 DataFrame 中数据的绘图函数。

        Parameters
        ----------
        func : callable
            接受数据和关键字参数的绘图函数。与 `map` 方法不同，此处使用的函数必须“理解”Pandas对象。它还必须绘制到当前活动的 matplotlib Axes，并接受 `color` 关键字参数。如果在 `hue` 维度上进行分面处理，则还必须接受 `label` 关键字参数。
        args : strings
            self.data 中标识包含要绘制数据的变量的列名。每个变量的数据按调用中指定的顺序传递给 `func`。
        kwargs : keyword arguments
            所有关键字参数都传递给绘图函数。

        Returns
        -------
        self : object
            返回 self。

        """

        # 如果 `color` 是关键字参数，在这里获取它
        kw_color = kwargs.pop("color", None)

        # 遍历数据子集
        for (row_i, col_j, hue_k), data_ijk in self.facet_data():

            # 如果此子集为空，继续下一个
            if not data_ijk.values.size:
                continue

            # 获取当前轴
            modify_state = not str(func.__module__).startswith("seaborn")
            ax = self.facet_axis(row_i, col_j, modify_state)

            # 决定要绘制的颜色
            kwargs["color"] = self._facet_color(hue_k, kw_color)

            # 如果适用，插入其他 hue 美学
            for kw, val_list in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]

            # 在关键字参数中插入一个标签用于图例
            if self._hue_var is not None:
                kwargs["label"] = self.hue_names[hue_k]

            # 将分面数据框插入 kwargs
            if self._dropna:
                data_ijk = data_ijk.dropna()
            kwargs["data"] = data_ijk

            # 绘制图形
            self._facet_plot(func, ax, args, kwargs)

        # 对于坐标轴标签，首选使用位置参数以保持向后兼容性，
        # 但也提取 x/y kwargs，并在没有相应参数时使用它们
        axis_labels = [kwargs.get("x", None), kwargs.get("y", None)]
        for i, val in enumerate(args[:2]):
            axis_labels[i] = val
        self._finalize_grid(axis_labels)

        return self

    def _facet_color(self, hue_index, kw_color):
        """
        根据 hue 索引选择颜色。

        Parameters
        ----------
        hue_index : int
            hue 的索引值。
        kw_color : str or None
            可选的颜色字符串。

        Returns
        -------
        color : str
            返回用于绘制的颜色字符串。
        """

        color = self._colors[hue_index]
        if kw_color is not None:
            return kw_color
        elif color is not None:
            return color
    # 绘制图形
    if str(func.__module__).startswith("seaborn"):
        # 如果函数来自 seaborn 模块，复制关键字参数并设置语义标签
        plot_kwargs = plot_kwargs.copy()
        semantics = ["x", "y", "hue", "size", "style"]
        for key, val in zip(semantics, plot_args):
            plot_kwargs[key] = val
        plot_args = []
        plot_kwargs["ax"] = ax
    # 调用函数并传入参数和关键字参数
    func(*plot_args, **plot_kwargs)

    # 更新图例数据
    self._update_legend_data(ax)

# 根据行列索引获取对应的轴对象，并将其设为活动状态，返回该轴对象
def facet_axis(self, row_i, col_j, modify_state=True):
    # 计算要绘制的轴对象的实际索引
    if self._col_wrap is not None:
        ax = self.axes.flat[col_j]
    else:
        ax = self.axes[row_i, col_j]

    # 如果需要修改状态，则将此轴设为当前活动轴
    if modify_state:
        plt.sca(ax)
    return ax

# 移除子图中的坐标轴边框
def despine(self, **kwargs):
    utils.despine(self._figure, **kwargs)
    return self

# 设置轴标签，左列和底部行
def set_axis_labels(self, x_var=None, y_var=None, clear_inner=True, **kwargs):
    if x_var is not None:
        self._x_var = x_var
        # 设置底部行的 x 轴标签
        self.set_xlabels(x_var, clear_inner=clear_inner, **kwargs)
    if y_var is not None:
        self._y_var = y_var
        # 设置左列的 y 轴标签
        self.set_ylabels(y_var, clear_inner=clear_inner, **kwargs)

    return self

# 设置底部行的 x 轴标签
def set_xlabels(self, label=None, clear_inner=True, **kwargs):
    if label is None:
        label = self._x_var
    # 设置底部行每个轴对象的 x 轴标签
    for ax in self._bottom_axes:
        ax.set_xlabel(label, **kwargs)
    if clear_inner:
        # 清除非底部行的轴对象的 x 轴标签
        for ax in self._not_bottom_axes:
            ax.set_xlabel("")
    return self

# 设置左列的 y 轴标签
def set_ylabels(self, label=None, clear_inner=True, **kwargs):
    if label is None:
        label = self._y_var
    # 设置左列每个轴对象的 y 轴标签
    for ax in self._left_axes:
        ax.set_ylabel(label, **kwargs)
    if clear_inner:
        # 清除非左列的轴对象的 y 轴标签
        for ax in self._not_left_axes:
            ax.set_ylabel("")
    return self
    def set_xticklabels(self, labels=None, step=None, **kwargs):
        """Set x axis tick labels of the grid."""
        # 遍历每个子图的 x 轴，设置刻度标签
        for ax in self.axes.flat:
            # 获取当前子图的 x 轴刻度位置
            curr_ticks = ax.get_xticks()
            # 设置 x 轴刻度位置
            ax.set_xticks(curr_ticks)
            if labels is None:
                # 如果未提供自定义标签，获取当前刻度的标签文本
                curr_labels = [label.get_text() for label in ax.get_xticklabels()]
                if step is not None:
                    # 根据步长获取部分刻度位置和对应的标签文本
                    xticks = ax.get_xticks()[::step]
                    curr_labels = curr_labels[::step]
                    ax.set_xticks(xticks)
                # 设置 x 轴刻度标签
                ax.set_xticklabels(curr_labels, **kwargs)
            else:
                # 使用提供的标签设置 x 轴刻度标签
                ax.set_xticklabels(labels, **kwargs)
        # 返回当前对象以支持方法链式调用
        return self

    def set_yticklabels(self, labels=None, **kwargs):
        """Set y axis tick labels on the left column of the grid."""
        # 遍历每个子图的 y 轴，设置刻度标签
        for ax in self.axes.flat:
            # 获取当前子图的 y 轴刻度位置
            curr_ticks = ax.get_yticks()
            # 设置 y 轴刻度位置
            ax.set_yticks(curr_ticks)
            if labels is None:
                # 如果未提供自定义标签，获取当前刻度的标签文本
                curr_labels = [label.get_text() for label in ax.get_yticklabels()]
                # 设置 y 轴刻度标签
                ax.set_yticklabels(curr_labels, **kwargs)
            else:
                # 使用提供的标签设置 y 轴刻度标签
                ax.set_yticklabels(labels, **kwargs)
        # 返回当前对象以支持方法链式调用
        return self

    def refline(self, *, x=None, y=None, color='.5', linestyle='--', **line_kws):
        """Add a reference line(s) to each facet.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s). Pass ``color=None`` to
            use ``hue`` mapping.
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`FacetGrid` instance
            Returns ``self`` for easy method chaining.

        """
        # 设置参考线的颜色和线型
        line_kws['color'] = color
        line_kws['linestyle'] = linestyle

        # 如果指定了 x 值，调用 matplotlib 的 axvline 在每个子图上添加垂直参考线
        if x is not None:
            self.map(plt.axvline, x=x, **line_kws)

        # 如果指定了 y 值，调用 matplotlib 的 axhline 在每个子图上添加水平参考线
        if y is not None:
            self.map(plt.axhline, y=y, **line_kws)

        # 返回当前对象以支持方法链式调用
        return self

    # ------ Properties that are part of the public API and documented by Sphinx

    @property
    def axes(self):
        """An array of the :class:`matplotlib.axes.Axes` objects in the grid."""
        # 返回网格中的所有子图对象数组
        return self._axes

    @property
    def ax(self):
        """The :class:`matplotlib.axes.Axes` when no faceting variables are assigned."""
        # 当未指定分面变量时返回网格的主轴对象
        if self.axes.shape == (1, 1):
            return self.axes[0, 0]
        else:
            err = (
                "Use the `.axes` attribute when facet variables are assigned."
            )
            # 如果存在分面变量，则抛出属性错误异常
            raise AttributeError(err)

    @property
    def axes_dict(self):
        """Return the mapping of facet names to corresponding :class:`matplotlib.axes.Axes`.

        If only one of ``row`` or ``col`` is assigned, each key is a string
        representing a level of that variable. If both facet dimensions are
        assigned, each key is a ``({row_level}, {col_level})`` tuple.

        """
        return self._axes_dict



    # ------ Private properties, that require some computation to get

    @property
    def _inner_axes(self):
        """Return a flat array of the inner axes."""
        if self._col_wrap is None:
            # If col_wrap is not set, return all axes except the last one in each row
            return self.axes[:-1, 1:].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (
                    i % self._ncol
                    and i < (self._ncol * (self._nrow - 1))
                    and i < (self._ncol * (self._nrow - 1) - n_empty)
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat



    @property
    def _left_axes(self):
        """Return a flat array of the left column of axes."""
        if self._col_wrap is None:
            # If col_wrap is not set, return the first column of axes
            return self.axes[:, 0].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if not i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat



    @property
    def _not_left_axes(self):
        """Return a flat array of axes that aren't on the left column."""
        if self._col_wrap is None:
            # If col_wrap is not set, return all axes except the first column
            return self.axes[:, 1:].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat



    @property
    def _bottom_axes(self):
        """Return a flat array of the bottom row of axes."""
        if self._col_wrap is None:
            # If col_wrap is not set, return the last row of axes
            return self.axes[-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (
                    i >= (self._ncol * (self._nrow - 1))
                    or i >= (self._ncol * (self._nrow - 1) - n_empty)
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat



    @property
    def _right_axes(self):
        """Return a flat array of the right column of axes."""
        if self._col_wrap is None:
            # If col_wrap is not set, return the last column of axes
            return self.axes[:, -1].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if (i + 1) % self._ncol == 0:
                    axes.append(ax)
            return np.array(axes, object).flat



    @property
    def _not_bottom_axes(self):
        """Return a flat array of axes that aren't on the bottom row."""
        if self._col_wrap is None:
            # If col_wrap is not set, return all axes except the last row
            return self.axes[:-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (
                    i < (self._ncol * (self._nrow - 1))
                    or i < (self._ncol * (self._nrow - 1) - n_empty)
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat
    # 返回不位于底部行的所有轴的扁平数组
    def _not_bottom_axes(self):
        """Return a flat array of axes that aren't on the bottom row."""
        
        # 如果未指定列包裹（col_wrap），则直接返回除了最后一行以外的所有轴
        if self._col_wrap is None:
            return self.axes[:-1, :].flat
        
        # 如果指定了列包裹（col_wrap）
        else:
            axes = []
            # 计算空轴的数量
            n_empty = self._nrow * self._ncol - self._n_facets
            
            # 遍历所有轴
            for i, ax in enumerate(self.axes):
                # 判断当前轴是否应该添加到结果中
                append = (
                    i < (self._ncol * (self._nrow - 1))  # 不位于最后一行
                    and i < (self._ncol * (self._nrow - 1) - n_empty)  # 不是空轴
                )
                if append:
                    axes.append(ax)
            
            # 将结果作为 NumPy 数组返回，并扁平化处理
            return np.array(axes, object).flat
class PairGrid(Grid):
    """Subplot grid for plotting pairwise relationships in a dataset.

    This object maps each variable in a dataset onto a column and row in a
    grid of multiple axes. Different axes-level plotting functions can be
    used to draw bivariate plots in the upper and lower triangles, and the
    marginal distribution of each variable can be shown on the diagonal.

    Several different common plots can be generated in a single line using
    :func:`pairplot`. Use :class:`PairGrid` when you need more flexibility.

    See the :ref:`tutorial <grid_tutorial>` for more information.

    """

    def __init__(
        self, data, *, hue=None, vars=None, x_vars=None, y_vars=None,
        hue_order=None, palette=None, hue_kws=None, corner=False, diag_sharey=True,
        height=2.5, aspect=1, layout_pad=.5, despine=True, dropna=False,
    ):
        """Initialize PairGrid object.

        Parameters
        ----------
        data : DataFrame
            Input data for plotting.
        hue : string (variable name), optional
            Variable in data to map plot aspects to different colors.
        vars : list of variable names, optional
            Variables within data to use, otherwise use every column with
            a numeric datatype.
        x_vars, y_vars : lists of variable names, optional
            Variables within ``data`` to use separately for the rows and columns
            of the grid.
        hue_order : list of strings
            Order for the levels of the hue variable in the palette.
        palette : dict or seaborn color palette
            Set of colors for mapping the hue variable.
        hue_kws : dictionary of param -> list of values mapping
            Additional keyword arguments to insert into the plotting call to
            let other plot attributes vary across levels of the hue variable
        corner : boolean, optional
            If True, don't add axes to the upper (off-diagonal) triangle of the grid,
            making this a "corner" plot.
        diag_sharey : boolean, optional
            If True, the diagonal elements are shared between the grids.
        height : scalar, optional
            Size of each facet.
        aspect : scalar, optional
            Aspect * height gives the width (in inches) of each facet.
        layout_pad : scalar, optional
            Padding between axes, in inches.
        despine : boolean, optional
            Remove the top and right spines from the plots.
        dropna : boolean, optional
            Drop missing values from the data before plotting.

        """

    def map(self, func, **kwargs):
        """Plot with the same function in every subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        Returns
        -------
        self : object
            Returns self.

        """
        row_indices, col_indices = np.indices(self.axes.shape)
        indices = zip(row_indices.flat, col_indices.flat)
        self._map_bivariate(func, indices, **kwargs)

        return self

    def map_lower(self, func, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        Returns
        -------
        self : object
            Returns self.

        """
        indices = zip(*np.tril_indices_from(self.axes, -1))
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_upper(self, func, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        Returns
        -------
        self : object
            Returns self.

        """
        indices = zip(*np.triu_indices_from(self.axes, 1))
        self._map_bivariate(func, indices, **kwargs)
        return self
    def map_offdiag(self, func, **kwargs):
        """Plot with a bivariate function on the off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        # 如果设置为使用方形网格
        if self.square_grid:
            # 在下对角线的子图上调用给定的绘图函数
            self.map_lower(func, **kwargs)
            # 如果不是角落图，还在上对角线的子图上调用给定的绘图函数
            if not self._corner:
                self.map_upper(func, **kwargs)
        else:
            # 遍历所有的y变量和x变量组合，找出非对角线位置的索引
            indices = []
            for i, (y_var) in enumerate(self.y_vars):
                for j, (x_var) in enumerate(self.x_vars):
                    if x_var != y_var:
                        indices.append((i, j))
            # 使用给定的绘图函数在非对角线位置上绘制双变量图
            self._map_bivariate(func, indices, **kwargs)
        return self

    def _map_diag_iter_hue(self, func, **kwargs):
        """Put marginal plot on each diagonal axes, iterating over hue."""
        # 从关键字参数中弹出固定的颜色值（如果有）
        fixed_color = kwargs.pop("color", None)

        # 遍历对角线变量和对应的轴对象
        for var, ax in zip(self.diag_vars, self.diag_axes):
            # 根据色调值对数据进行分组
            hue_grouped = self.data[var].groupby(self.hue_vals, observed=True)

            # 复制关键字参数为绘图时的参数
            plot_kwargs = kwargs.copy()
            # 如果绘图函数的模块以"seaborn"开头，将绘图的轴指定为当前轴
            if str(func.__module__).startswith("seaborn"):
                plot_kwargs["ax"] = ax
            else:
                plt.sca(ax)

            # 遍历色调的顺序和标签
            for k, label_k in enumerate(self._hue_order):

                # 尝试获取当前色调水平的数据，如果不存在则创建一个空的序列
                try:
                    data_k = hue_grouped.get_group(label_k)
                except KeyError:
                    data_k = pd.Series([], dtype=float)

                # 如果未指定固定颜色，则使用预定义的调色板颜色
                if fixed_color is None:
                    color = self.palette[k]
                else:
                    color = fixed_color

                # 如果设置了丢弃NaN值的选项，则在数据中移除NaN
                if self._dropna:
                    data_k = utils.remove_na(data_k)

                # 根据绘图函数的模块进行调用，传递数据、标签、颜色及其他参数
                if str(func.__module__).startswith("seaborn"):
                    func(x=data_k, label=label_k, color=color, **plot_kwargs)
                else:
                    func(data_k, label=label_k, color=color, **plot_kwargs)

        # 添加轴标签
        self._add_axis_labels()

        return self
    def _map_bivariate(self, func, indices, **kwargs):
        """Draw a bivariate plot on the indicated axes."""
        # 处理新分布图形绘制不会自动添加艺术元素到轴的情况的临时解决方案。
        # 这通常是更好的做法，但在 axisgrid 函数中需要一个更好的处理方式。
        from .distributions import histplot, kdeplot
        # 如果 func 是 histplot 或 kdeplot，则设置标志以提取图例句柄
        if func is histplot or func is kdeplot:
            self._extract_legend_handles = True

        kws = kwargs.copy()  # 复制 kwargs，因为我们会插入其他的 kwargs
        # 遍历索引列表中的每一对 (i, j)
        for i, j in indices:
            x_var = self.x_vars[j]
            y_var = self.y_vars[i]
            ax = self.axes[i, j]
            if ax is None:  # 如果 ax 是 None，说明我们处于角落模式
                continue
            # 在指定的 axes 上绘制双变量图
            self._plot_bivariate(x_var, y_var, ax, func, **kws)
        # 添加坐标轴标签
        self._add_axis_labels()

        # 如果 func 的参数中有 "hue"
        if "hue" in signature(func).parameters:
            # 将图例数据名称列表设为当前图例数据的键列表
            self.hue_names = list(self._legend_data)

    def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot on the specified axes."""
        # 如果 func 的参数中没有 "hue"，则调用 _plot_bivariate_iter_hue 方法绘制图形，并返回
        if "hue" not in signature(func).parameters:
            self._plot_bivariate_iter_hue(x_var, y_var, ax, func, **kwargs)
            return

        kwargs = kwargs.copy()
        # 如果 func 的模块名以 "seaborn" 开头，则将 kwargs 中的 "ax" 键设为 ax
        if str(func.__module__).startswith("seaborn"):
            kwargs["ax"] = ax
        else:
            plt.sca(ax)

        # 如果 x_var 等于 y_var，则 axes_vars 为 [x_var]，否则为 [x_var, y_var]
        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]

        # 如果 _hue_var 不为 None 且不在 axes_vars 中，则将其添加到 axes_vars 中
        if self._hue_var is not None and self._hue_var not in axes_vars:
            axes_vars.append(self._hue_var)

        # 从 self.data 中选取 axes_vars 对应的数据
        data = self.data[axes_vars]
        # 如果 _dropna 标志为 True，则删除包含 NaN 值的行
        if self._dropna:
            data = data.dropna()

        # 提取 x 和 y 数据
        x = data[x_var]
        y = data[y_var]
        # 如果 _hue_var 为 None，则 hue 为 None，否则为 data 中 _hue_var 对应的数据
        if self._hue_var is None:
            hue = None
        else:
            hue = data.get(self._hue_var)

        # 如果 kwargs 中不包含 "hue"，则更新 kwargs
        if "hue" not in kwargs:
            kwargs.update({
                "hue": hue, "hue_order": self._hue_order, "palette": self._orig_palette,
            })
        # 调用 func 绘制双变量图形
        func(x=x, y=y, **kwargs)

        # 更新图例数据
        self._update_legend_data(ax)
    def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot while iterating over hue subsets."""
        # 复制 kwargs，以确保不修改原始输入参数
        kwargs = kwargs.copy()
        
        # 根据函数模块名判断是否为 seaborn 函数，设置绘图参数
        if str(func.__module__).startswith("seaborn"):
            kwargs["ax"] = ax
        else:
            plt.sca(ax)  # 设置当前的坐标轴为 ax
        
        # 根据 x_var 和 y_var 是否相同，确定绘图的变量
        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]

        # 按照 hue 值分组数据
        hue_grouped = self.data.groupby(self.hue_vals, observed=True)
        
        # 遍历每个 hue 值及其对应的标签
        for k, label_k in enumerate(self._hue_order):

            kws = kwargs.copy()  # 复制绘图参数

            # 尝试获取当前 hue 值对应的数据，允许数据为空的情况
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.DataFrame(columns=axes_vars, dtype=float)

            # 如果设置了 _dropna 标志，则去除数据中的空值
            if self._dropna:
                data_k = data_k[axes_vars].dropna()

            x = data_k[x_var]
            y = data_k[y_var]

            # 设置每个 hue 值的其他关键字参数
            for kw, val_list in self.hue_kws.items():
                kws[kw] = val_list[k]

            kws.setdefault("color", self.palette[k])  # 设置默认颜色为预定义的调色板中的颜色
            if self._hue_var is not None:
                kws["label"] = label_k  # 如果存在 hue 变量，则设置标签为当前 hue 值的标签

            # 根据函数模块名选择调用不同的绘图函数
            if str(func.__module__).startswith("seaborn"):
                func(x=x, y=y, **kws)
            else:
                func(x, y, **kws)

        # 更新图例数据
        self._update_legend_data(ax)

    def _add_axis_labels(self):
        """Add labels to the left and bottom Axes."""
        # 分别为左侧和底部的坐标轴添加标签
        for ax, label in zip(self.axes[-1, :], self.x_vars):
            ax.set_xlabel(label)
        for ax, label in zip(self.axes[:, 0], self.y_vars):
            ax.set_ylabel(label)

    def _find_numeric_cols(self, data):
        """Find which variables in a DataFrame are numeric."""
        numeric_cols = []
        # 遍历数据框中的每一列，判断是否为数值类型
        for col in data:
            if variable_type(data[col]) == "numeric":
                numeric_cols.append(col)
        return numeric_cols
class JointGrid(_BaseGrid):
    """Grid for drawing a bivariate plot with marginal univariate plots.

    Many plots can be drawn by using the figure-level interface :func:`jointplot`.
    Use this class directly when you need more flexibility.

    """

    def __init__(
        self, data=None, *,
        x=None, y=None, hue=None,
        height=6, ratio=5, space=.2,
        palette=None, hue_order=None, hue_norm=None,
        dropna=False, xlim=None, ylim=None, marginal_ticks=False,



        # 初始化函数，用于创建一个联合网格对象，用于绘制双变量图和边缘单变量图
        # 继承自 _BaseGrid 类
        # 可以通过使用顶层接口 :func:`jointplot` 来绘制多种图形
        # 在需要更多灵活性时直接使用此类

        # 初始化函数的参数列表：
        # - data: 数据集，可以是 DataFrame 或数组形式，默认为 None
        # - x: x 轴数据，用于绘制联合图，默认为 None
        # - y: y 轴数据，用于绘制联合图，默认为 None
        # - hue: 可选参数，用于分组数据的变量，默认为 None
        # - height: 图形的高度，默认为 6
        # - ratio: 比例参数，默认为 5
        # - space: 子图之间的间距，默认为 0.2
        # - palette: 调色板，用于设置颜色，默认为 None
        # - hue_order: hue 变量的排序列表，默认为 None
        # - hue_norm: hue 变量的归一化对象，默认为 None
        # - dropna: 是否丢弃 NaN 值，默认为 False
        # - xlim: x 轴的限制范围，默认为 None
        # - ylim: y 轴的限制范围，默认为 None
        # - marginal_ticks: 是否显示边缘图的刻度，默认为 False
    ):

        # 设置子图网格
        f = plt.figure(figsize=(height, height))  # 创建一个指定大小的画布
        gs = plt.GridSpec(ratio + 1, ratio + 1)  # 创建网格布局对象

        ax_joint = f.add_subplot(gs[1:, :-1])  # 在网格中添加主要联合轴
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)  # 添加X边缘轴，与主轴共享X轴
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)  # 添加Y边缘轴，与主轴共享Y轴

        self._figure = f  # 将图形对象存储在实例变量中
        self.ax_joint = ax_joint  # 存储主要联合轴对象
        self.ax_marg_x = ax_marg_x  # 存储X边缘轴对象
        self.ax_marg_y = ax_marg_y  # 存储Y边缘轴对象

        # 在边缘图上关闭测量轴的刻度可见性
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)  # X边缘轴的主刻度标签不可见
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)  # Y边缘轴的主刻度标签不可见
        plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)  # X边缘轴的次刻度标签不可见
        plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)  # Y边缘轴的次刻度标签不可见

        # 如果不需要边缘刻度，则关闭边缘图上的密度轴刻度
        if not marginal_ticks:
            plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)  # X边缘轴的主刻度线不可见
            plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)  # X边缘轴的次刻度线不可见
            plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)  # Y边缘轴的主刻度线不可见
            plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)  # Y边缘轴的次刻度线不可见
            plt.setp(ax_marg_x.get_yticklabels(), visible=False)  # X边缘轴的Y刻度标签不可见
            plt.setp(ax_marg_y.get_xticklabels(), visible=False)  # Y边缘轴的X刻度标签不可见
            plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)  # X边缘轴的次Y刻度标签不可见
            plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)  # Y边缘轴的次X刻度标签不可见
            ax_marg_x.yaxis.grid(False)  # 关闭X边缘轴的网格线
            ax_marg_y.xaxis.grid(False)  # 关闭Y边缘轴的网格线

        # 处理输入变量
        p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))  # 创建VectorPlotter对象并传入数据和变量
        plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]  # 从处理后的数据中选择非空列

        # 可能删除缺失值
        if dropna:
            plot_data = plot_data.dropna()  # 删除包含NaN的行

        def get_var(var):
            vector = plot_data.get(var, None)  # 获取指定变量的向量数据
            if vector is not None:
                vector = vector.rename(p.variables.get(var, None))  # 根据变量名称重命名向量
            return vector

        self.x = get_var("x")  # 存储X变量的向量数据
        self.y = get_var("y")  # 存储Y变量的向量数据
        self.hue = get_var("hue")  # 存储色调变量的向量数据

        for axis in "xy":
            name = p.variables.get(axis, None)  # 获取变量名称
            if name is not None:
                getattr(ax_joint, f"set_{axis}label")(name)  # 根据变量名称设置主联合轴的标签

        if xlim is not None:
            ax_joint.set_xlim(xlim)  # 设置主联合轴的X轴限制
        if ylim is not None:
            ax_joint.set_ylim(ylim)  # 设置主联合轴的Y轴限制

        # 存储用于轴级函数的语义映射参数
        self._hue_params = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm)

        # 美化网格外观
        utils.despine(f)  # 使用工具函数去除图形的轴线和边框
        if not marginal_ticks:
            utils.despine(ax=ax_marg_x, left=True)  # 去除X边缘轴的左边轴线和边框
            utils.despine(ax=ax_marg_y, bottom=True)  # 去除Y边缘轴的底部轴线和边框
        for axes in [ax_marg_x, ax_marg_y]:
            for axis in [axes.xaxis, axes.yaxis]:
                axis.label.set_visible(False)  # 将边缘轴的标签设置为不可见
        f.tight_layout()  # 调整子图布局以防止重叠
        f.subplots_adjust(hspace=space, wspace=space)  # 调整子图之间的垂直和水平间距
    def _inject_kwargs(self, func, kws, params):
        """Add params to kws if they are accepted by func."""
        # 获取函数 func 的参数签名信息
        func_params = signature(func).parameters
        # 遍历参数字典 params
        for key, val in params.items():
            # 如果参数 key 存在于 func 的参数列表中，则将其添加到 kws 中
            if key in func_params:
                kws.setdefault(key, val)

    def plot(self, joint_func, marginal_func, **kwargs):
        """Draw the plot by passing functions for joint and marginal axes.

        This method passes the ``kwargs`` dictionary to both functions. If you
        need more control, call :meth:`JointGrid.plot_joint` and
        :meth:`JointGrid.plot_marginals` directly with specific parameters.

        Parameters
        ----------
        joint_func, marginal_func : callables
            Functions to draw the bivariate and univariate plots. See methods
            referenced above for information about the required characteristics
            of these functions.
        kwargs
            Additional keyword arguments are passed to both functions.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        # 调用 plot_marginals 方法，传递 marginal_func 和 kwargs
        self.plot_marginals(marginal_func, **kwargs)
        # 调用 plot_joint 方法，传递 joint_func 和 kwargs，然后返回 self
        self.plot_joint(joint_func, **kwargs)
        return self

    def plot_joint(self, func, **kwargs):
        """Draw a bivariate plot on the joint axes of the grid.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should accept ``x`` and ``y``. Otherwise,
            it must accept ``x`` and ``y`` vectors of data as the first two
            positional arguments, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, the function must
            accept ``hue`` as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        # 复制 kwargs，以免修改原始输入
        kwargs = kwargs.copy()
        # 如果 func 是 seaborn 的函数，将 ax 设置为 self.ax_joint
        if str(func.__module__).startswith("seaborn"):
            kwargs["ax"] = self.ax_joint
        else:
            # 否则使用 plt.sca 设置当前的绘图坐标轴为 self.ax_joint
            plt.sca(self.ax_joint)
        # 如果定义了 hue，将其添加到 kwargs 中，并注入额外的参数到 func 中
        if self.hue is not None:
            kwargs["hue"] = self.hue
            self._inject_kwargs(func, kwargs, self._hue_params)

        # 根据 func 的模块判断调用方式，并传递 x、y 和 kwargs
        if str(func.__module__).startswith("seaborn"):
            func(x=self.x, y=self.y, **kwargs)
        else:
            func(self.x, self.y, **kwargs)

        # 返回 self 以支持方法链式调用
        return self
    def plot_marginals(self, func, **kwargs):
        """Draw univariate plots on each marginal axes.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should  accept ``x`` and ``y`` and plot
            when only one of them is defined. Otherwise, it must accept a vector
            of data as the first positional argument and determine its orientation
            using the ``vertical`` parameter, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, it must accept ``hue``
            as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        seaborn_func = (
            str(func.__module__).startswith("seaborn")
            # deprecated distplot has a legacy API, special case it
            and not func.__name__ == "distplot"
        )
        func_params = signature(func).parameters
        kwargs = kwargs.copy()
        if self.hue is not None:
            # 将对象构造函数中定义的 hue 参数注入到 kwargs 中，供绘图函数使用
            kwargs["hue"] = self.hue
            self._inject_kwargs(func, kwargs, self._hue_params)

        if "legend" in func_params:
            # 如果函数签名包含 "legend" 参数，则在 kwargs 中设置默认值为 False
            kwargs.setdefault("legend", False)

        if "orientation" in func_params:
            # 如果函数签名包含 "orientation" 参数，则根据不同情况设置绘图方向参数
            orient_kw_x = {"orientation": "vertical"}
            orient_kw_y = {"orientation": "horizontal"}
        elif "vertical" in func_params:
            # 如果函数签名包含 "vertical" 参数，则根据不同情况设置绘图方向参数
            orient_kw_x = {"vertical": False}
            orient_kw_y = {"vertical": True}

        if seaborn_func:
            # 如果是 seaborn 函数，则使用给定的参数绘制 x 轴的边缘图
            func(x=self.x, ax=self.ax_marg_x, **kwargs)
        else:
            # 否则，使用 matplotlib 绘制 x 轴的边缘图，根据参数设置方向
            plt.sca(self.ax_marg_x)
            func(self.x, **orient_kw_x, **kwargs)

        if seaborn_func:
            # 如果是 seaborn 函数，则使用给定的参数绘制 y 轴的边缘图
            func(y=self.y, ax=self.ax_marg_y, **kwargs)
        else:
            # 否则，使用 matplotlib 绘制 y 轴的边缘图，根据参数设置方向
            plt.sca(self.ax_marg_y)
            func(self.y, **orient_kw_y, **kwargs)

        # 隐藏 x 轴和 y 轴边缘图的标签
        self.ax_marg_x.yaxis.get_label().set_visible(False)
        self.ax_marg_y.xaxis.get_label().set_visible(False)

        # 返回当前对象，支持方法链式调用
        return self
    ):
        """
        Add a reference line(s) to joint and/or marginal axes.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        joint, marginal : bools
            Whether to add the reference line(s) to the joint/marginal axes.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s).
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        # 设置参考线的颜色和线型
        line_kws['color'] = color
        line_kws['linestyle'] = linestyle

        # 如果指定了 x 值，则根据 joint 和 marginal 参数添加垂直参考线
        if x is not None:
            if joint:
                self.ax_joint.axvline(x, **line_kws)  # 在联合轴上添加垂直参考线
            if marginal:
                self.ax_marg_x.axvline(x, **line_kws)  # 在边缘 x 轴上添加垂直参考线

        # 如果指定了 y 值，则根据 joint 和 marginal 参数添加水平参考线
        if y is not None:
            if joint:
                self.ax_joint.axhline(y, **line_kws)  # 在联合轴上添加水平参考线
            if marginal:
                self.ax_marg_y.axhline(y, **line_kws)  # 在边缘 y 轴上添加水平参考线

        return self

    def set_axis_labels(self, xlabel="", ylabel="", **kwargs):
        """
        Set axis labels on the bivariate axes.

        Parameters
        ----------
        xlabel, ylabel : strings
            Label names for the x and y variables.
        kwargs : key, value mappings
            Other keyword arguments are passed to the following functions:

            - :meth:`matplotlib.axes.Axes.set_xlabel`
            - :meth:`matplotlib.axes.Axes.set_ylabel`

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        self.ax_joint.set_xlabel(xlabel, **kwargs)  # 设置联合轴的 x 轴标签
        self.ax_joint.set_ylabel(ylabel, **kwargs)  # 设置联合轴的 y 轴标签
        return self
# 将 JointGrid 类的 __init__ 方法的文档字符串设置为特定格式，用于设置子图网格和存储数据以便于简便绘图。
JointGrid.__init__.__doc__ = """\
Set up the grid of subplots and store data internally for easy plotting.

Parameters
----------
{params.core.data}
{params.core.xy}
height : number
    Size of each side of the figure in inches (it will be square).
ratio : number
    Ratio of joint axes height to marginal axes height.
space : number
    Space between the joint and marginal axes
dropna : bool
    If True, remove missing observations before plotting.
{{x, y}}lim : pairs of numbers
    Set axis limits to these values before plotting.
marginal_ticks : bool
    If False, suppress ticks on the count/density axis of the marginal plots.
{params.core.hue}
    Note: unlike in :class:`FacetGrid` or :class:`PairGrid`, the axes-level
    functions must support ``hue`` to use it in :class:`JointGrid`.
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}

See Also
--------
{seealso.jointplot}
{seealso.pairgrid}
{seealso.pairplot}

Examples
--------

.. include:: ../docstrings/JointGrid.rst

""".format(
    params=_param_docs,  # 参数文档，从 _param_docs 中获取相关信息
    seealso=_core_docs["seealso"],  # 参见部分文档，从 _core_docs["seealso"] 中获取相关信息
)


def pairplot(
    data, *,
    hue=None, hue_order=None, palette=None,
    vars=None, x_vars=None, y_vars=None,
    kind="scatter", diag_kind="auto", markers=None,
    height=2.5, aspect=1, corner=False, dropna=False,
    plot_kws=None, diag_kws=None, grid_kws=None, size=None,
):
    """Plot pairwise relationships in a dataset.

    By default, this function will create a grid of Axes such that each numeric
    variable in ``data`` will by shared across the y-axes across a single row and
    the x-axes across a single column. The diagonal plots are treated
    differently: a univariate distribution plot is drawn to show the marginal
    distribution of the data in each column.

    It is also possible to show a subset of variables or plot different
    variables on the rows and columns.

    This is a high-level interface for :class:`PairGrid` that is intended to
    make it easy to draw a few common styles. You should use :class:`PairGrid`
    directly if you need more flexibility.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : name of variable in ``data``
        Variable in ``data`` to map plot aspects to different colors.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    vars : list of variable names
        Variables within ``data`` to use, otherwise use every column with
        a numeric datatype.
    {x, y}_vars : lists of variable names
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    # 可视化图类型，可以是'scatter'（散点图）、'kde'（核密度估计图）、'hist'（直方图）、'reg'（回归图）之一
    kind : {'scatter', 'kde', 'hist', 'reg'}
        Kind of plot to make.
    # 对角线子图的类型，可以是'auto'（根据是否有'hue'参数自动选择）、'hist'（直方图）、'kde'（核密度估计图）、None（无）
    diag_kind : {'auto', 'hist', 'kde', None}
        Kind of plot for the diagonal subplots. If 'auto', choose based on
        whether or not ``hue`` is used.
    # 散点图中使用的标记，可以是单个matplotlib标记代码或与'hue'变量水平数相同长度的标记列表，以便不同颜色的点有不同的标记
    markers : single matplotlib marker code or list
        Either the marker to use for all scatterplot points or a list of markers
        with a length the same as the number of levels in the hue variable so that
        differently colored points will also have different scatterplot
        markers.
    # 每个子图的高度（单位为英寸）
    height : scalar
        Height (in inches) of each facet.
    # 子图宽高比，aspect * height 给出每个子图的宽度（单位为英寸）
    aspect : scalar
        Aspect * height gives the width (in inches) of each facet.
    # 是否为“角落”图，如果为True，则不向网格的上（非对角线）三角形添加坐标轴
    corner : bool
        If True, don't add axes to the upper (off-diagonal) triangle of the
        grid, making this a "corner" plot.
    # 是否在绘图前删除数据中的缺失值
    dropna : boolean
        Drop missing values from the data before plotting.
    # 各种参数的字典，'plot_kws'传递给双变量绘图函数，'diag_kws'传递给单变量绘图函数，'grid_kws'传递给PairGrid构造函数
    {plot, diag, grid}_kws : dicts
        Dictionaries of keyword arguments. ``plot_kws`` are passed to the
        bivariate plotting function, ``diag_kws`` are passed to the univariate
        plotting function, and ``grid_kws`` are passed to the :class:`PairGrid`
        constructor.

    # 返回PairGrid的实例，以便进一步调整
    Returns
    -------
    grid : :class:`PairGrid`
        Returns the underlying :class:`PairGrid` instance for further tweaking.

    # 参见
    # --------
    # PairGrid：用于更灵活地绘制成对关系的子图网格
    # JointGrid：用于绘制两个变量的联合和边缘分布的网格
    See Also
    --------
    PairGrid : Subplot grid for more flexible plotting of pairwise relationships.
    JointGrid : Grid for plotting joint and marginal distributions of two variables.

    # 示例
    # --------
    # 包含在../docstrings/pairplot.rst中的文档
    Examples
    --------
    .. include:: ../docstrings/pairplot.rst

    """
    # 避免循环导入
    from .distributions import histplot, kdeplot

    # 处理已弃用的功能
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    # 如果'data'不是pd.DataFrame对象，则引发类型错误
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"'data' must be pandas DataFrame object, not: {type(data)}")

    # 如果'plot_kws', 'diag_kws', 'grid_kws'为None，则将其设置为空字典
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    diag_kws = {} if diag_kws is None else diag_kws.copy()
    grid_kws = {} if grid_kws is None else grid_kws.copy()

    # 解析"auto"对角线类型
    if diag_kind == "auto":
        if hue is None:
            diag_kind = "kde" if kind == "kde" else "hist"
        else:
            diag_kind = "hist" if kind == "hist" else "kde"

    # 设置PairGrid
    grid_kws.setdefault("diag_sharey", diag_kind == "hist")
    grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue,
                    hue_order=hue_order, palette=palette, corner=corner,
                    height=height, aspect=aspect, dropna=dropna, **grid_kws)

    # 在这里添加标记，因为PairGrid已经确定了'hue'变量需要多少级别，我们不希望重复该过程
    # 如果 markers 参数不为 None，则进入条件判断
    if markers is not None:
        # 如果 kind 为 "reg"，执行以下逻辑
        if kind == "reg":
            # regplot 暂时不支持样式，这里是一个临时需求
            # 如果 grid.hue_names 为 None，则设置 n_markers 为 1
            if grid.hue_names is None:
                n_markers = 1
            else:
                # 否则，n_markers 设置为 grid.hue_names 的长度
                n_markers = len(grid.hue_names)
            
            # 如果 markers 不是列表，将其转换为包含 n_markers 个元素的列表
            if not isinstance(markers, list):
                markers = [markers] * n_markers
            
            # 如果 markers 列表长度不等于 n_markers，抛出数值错误
            if len(markers) != n_markers:
                raise ValueError("markers must be a singleton or a list of "
                                 "markers for each level of the hue variable")
            
            # 设置 grid.hue_kws 字典，指定 "marker" 键对应 markers 参数
            grid.hue_kws = {"marker": markers}
        
        # 如果 kind 为 "scatter"，执行以下逻辑
        elif kind == "scatter":
            # 如果 markers 是字符串，将 plot_kws 字典中的 "marker" 键设置为 markers
            if isinstance(markers, str):
                plot_kws["marker"] = markers
            # 如果 hue 不为 None，设置 plot_kws 中的 "style" 键为 data[hue]，"markers" 键为 markers
            elif hue is not None:
                plot_kws["style"] = data[hue]
                plot_kws["markers"] = markers

    # 在对角线上绘制边际图
    # 复制 diag_kws 字典并设置默认 "legend" 键为 False
    diag_kws = diag_kws.copy()
    diag_kws.setdefault("legend", False)
    
    # 如果 diag_kind 为 "hist"，在网格上调用 histplot 函数，传入 diag_kws 参数
    if diag_kind == "hist":
        grid.map_diag(histplot, **diag_kws)
    
    # 如果 diag_kind 为 "kde"，设置 diag_kws 的默认 "fill" 键为 True，"warn_singular" 键为 False
    # 在网格上调用 kdeplot 函数，传入 diag_kws 参数
    elif diag_kind == "kde":
        diag_kws.setdefault("fill", True)
        diag_kws.setdefault("warn_singular", False)
        grid.map_diag(kdeplot, **diag_kws)

    # 可能在非对角线位置上绘制图形
    # 如果 diag_kind 不为 None，plotter 设置为 grid.map_offdiag，否则设置为 grid.map
    if diag_kind is not None:
        plotter = grid.map_offdiag
    else:
        plotter = grid.map

    # 根据 kind 的值选择合适的绘图函数并在网格上调用
    if kind == "scatter":
        from .relational import scatterplot  # 避免循环导入
        plotter(scatterplot, **plot_kws)
    elif kind == "reg":
        from .regression import regplot  # 避免循环导入
        plotter(regplot, **plot_kws)
    elif kind == "kde":
        from .distributions import kdeplot  # 避免循环导入
        plot_kws.setdefault("warn_singular", False)
        plotter(kdeplot, **plot_kws)
    elif kind == "hist":
        from .distributions import histplot  # 避免循环导入
        plotter(histplot, **plot_kws)

    # 如果 hue 不为 None，为网格添加图例
    if hue is not None:
        grid.add_legend()

    # 调整网格布局以适应图像
    grid.tight_layout()

    # 返回绘制完成的网格对象
    return grid
# 导入所需的函数模块，避免循环导入
def jointplot(
    data=None, *, x=None, y=None, hue=None, kind="scatter",
    height=6, ratio=5, space=.2, dropna=False, xlim=None, ylim=None,
    color=None, palette=None, hue_order=None, hue_norm=None, marginal_ticks=False,
    joint_kws=None, marginal_kws=None,
    **kwargs
):
    # 忽略掉 "ax" 关键字参数，因为 jointplot 是一个图级别的函数
    if kwargs.pop("ax", None) is not None:
        msg = "Ignoring `ax`; jointplot is a figure-level function."
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 设置空的默认关键字参数字典
    joint_kws = {} if joint_kws is None else joint_kws.copy()
    joint_kws.update(kwargs)
    marginal_kws = {} if marginal_kws is None else marginal_kws.copy()

    # 处理 distplot 特定的关键字参数不推荐使用的情况
    distplot_keys = [
        "rug", "fit", "hist_kws", "norm_hist" "hist_kws", "rug_kws",
    ]
    unused_keys = []
    for key in distplot_keys:
        if key in marginal_kws:
            unused_keys.append(key)
            marginal_kws.pop(key)
    if unused_keys and kind != "kde":
        msg = (
            "The marginal plotting function has changed to `histplot`,"
            " which does not accept the following argument(s): {}."
        ).format(", ".join(unused_keys))
        warnings.warn(msg, UserWarning)

    # 验证绘图类型的合法性
    plot_kinds = ["scatter", "hist", "hex", "kde", "reg", "resid"]
    _check_argument("kind", plot_kinds, kind)

    # 如果使用了不支持的 kind 类型和 hue 参数，则提前抛出 ValueError
    if hue is not None and kind in ["hex", "reg", "resid"]:
        msg = f"Use of `hue` with `kind='{kind}'` is not currently supported."
        raise ValueError(msg)

    # 根据绘图颜色生成一个 colormap
    # （目前仅用于 kind="hex"）
    if color is None:
        color = "C0"
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [set_hls_values(color_rgb, l=val) for val in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)

    # 当 kind 为 "hex" 时，确保 dropna 参数为 True
    if kind == "hex":
        dropna = True

    # 初始化 JointGrid 对象
    grid = JointGrid(
        data=data, x=x, y=y, hue=hue,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        dropna=dropna, height=height, ratio=ratio, space=space,
        xlim=xlim, ylim=ylim, marginal_ticks=marginal_ticks,
    )

    # 如果 JointGrid 对象有 hue 参数，则默认设置 marginal_kws 中的 "legend" 为 False
    if grid.hue is not None:
        marginal_kws.setdefault("legend", False)

    # 使用 grid 对象绘制数据
    # 如果图表类型以 "scatter" 开头
    if kind.startswith("scatter"):

        # 将颜色设置为默认颜色
        joint_kws.setdefault("color", color)
        # 绘制联合分布的散点图
        grid.plot_joint(scatterplot, **joint_kws)

        # 如果没有设置色调（hue）
        if grid.hue is None:
            # 使用直方图绘制边际分布
            marg_func = histplot
        else:
            # 使用核密度估计绘制边际分布
            marg_func = kdeplot
            # 设置边际图的参数：警告奇异值（warn_singular）为假，填充为真
            marginal_kws.setdefault("warn_singular", False)
            marginal_kws.setdefault("fill", True)

        # 将颜色设置为默认颜色
        marginal_kws.setdefault("color", color)
        # 绘制边际分布图
        grid.plot_marginals(marg_func, **marginal_kws)

    # 如果图表类型以 "hist" 开头
    elif kind.startswith("hist"):

        # TODO 处理对参数（bins 等）的处理并传递给联合和边际图

        # 将颜色设置为默认颜色
        joint_kws.setdefault("color", color)
        # 使用直方图绘制联合分布图
        grid.plot_joint(histplot, **joint_kws)

        # 设置边际图的参数：不使用核密度估计，颜色设置为默认颜色
        marginal_kws.setdefault("kde", False)
        marginal_kws.setdefault("color", color)

        # 复制边际图参数用于 x 和 y 方向
        marg_x_kws = marginal_kws.copy()
        marg_y_kws = marginal_kws.copy()

        # 针对一对键（bins, binwidth, binrange），如果联合图中存在元组，则将值分配给对应的边际图参数
        pair_keys = "bins", "binwidth", "binrange"
        for key in pair_keys:
            if isinstance(joint_kws.get(key), tuple):
                x_val, y_val = joint_kws[key]
                marg_x_kws.setdefault(key, x_val)
                marg_y_kws.setdefault(key, y_val)

        # 使用直方图绘制 x 方向的边际分布图
        histplot(data=data, x=x, hue=hue, **marg_x_kws, ax=grid.ax_marg_x)
        # 使用直方图绘制 y 方向的边际分布图
        histplot(data=data, y=y, hue=hue, **marg_y_kws, ax=grid.ax_marg_y)

    # 如果图表类型以 "kde" 开头
    elif kind.startswith("kde"):

        # 将颜色设置为默认颜色
        joint_kws.setdefault("color", color)
        # 绘制联合分布的核密度估计图
        grid.plot_joint(kdeplot, **joint_kws)

        # 设置边际图的参数：颜色设置为默认颜色，如果在联合图中设置了 "fill" 参数，则在边际图中也设置
        marginal_kws.setdefault("color", color)
        if "fill" in joint_kws:
            marginal_kws.setdefault("fill", joint_kws["fill"])

        # 绘制边际分布图
        grid.plot_marginals(kdeplot, **marginal_kws)

    # 如果图表类型以 "hex" 开头
    elif kind.startswith("hex"):

        # 计算 x 和 y 方向的网格数量
        x_bins = min(_freedman_diaconis_bins(grid.x), 50)
        y_bins = min(_freedman_diaconis_bins(grid.y), 50)
        gridsize = int(np.mean([x_bins, y_bins]))

        # 设置联合图的参数：网格大小和颜色映射
        joint_kws.setdefault("gridsize", gridsize)
        joint_kws.setdefault("cmap", cmap)
        # 绘制联合分布的六边形图
        grid.plot_joint(plt.hexbin, **joint_kws)

        # 设置边际图的参数：不使用核密度估计，颜色设置为默认颜色
        marginal_kws.setdefault("kde", False)
        marginal_kws.setdefault("color", color)
        # 绘制边际分布图，使用直方图
        grid.plot_marginals(histplot, **marginal_kws)

    # 如果图表类型以 "reg" 开头
    elif kind.startswith("reg"):

        # 设置边际图的参数：颜色设置为默认颜色，使用核密度估计
        marginal_kws.setdefault("color", color)
        marginal_kws.setdefault("kde", True)
        # 绘制边际分布图，使用直方图
        grid.plot_marginals(histplot, **marginal_kws)

        # 将颜色设置为默认颜色
        joint_kws.setdefault("color", color)
        # 绘制联合回归图
        grid.plot_joint(regplot, **joint_kws)

    # 如果图表类型以 "resid" 开头
    elif kind.startswith("resid"):

        # 将颜色设置为默认颜色
        joint_kws.setdefault("color", color)
        # 绘制联合残差图
        grid.plot_joint(residplot, **joint_kws)

        # 获取联合轴上的点集的 x 和 y 坐标
        x, y = grid.ax_joint.collections[0].get_offsets().T
        # 设置边际图的参数：颜色设置为默认颜色
        marginal_kws.setdefault("color", color)
        # 使用直方图绘制 x 方向的边际分布图
        histplot(x=x, hue=hue, ax=grid.ax_marg_x, **marginal_kws)
        # 使用直方图绘制 y 方向的边际分布图
        histplot(y=y, hue=hue, ax=grid.ax_marg_y, **marginal_kws)

    # 将主轴设置为 matplotlib 状态机中的活动轴
    plt.sca(grid.ax_joint)

    # 返回绘制的图形对象
    return grid
# 将文档字符串赋值给 `jointplot` 函数的 `__doc__` 属性，用于描述绘制两个变量的联合图和边缘图的方法

jointplot.__doc__ = """\
Draw a plot of two variables with bivariate and univariate graphs.

This function provides a convenient interface to the :class:`JointGrid`
class, with several canned plot kinds. This is intended to be a fairly
lightweight wrapper; if you need more flexibility, you should use
:class:`JointGrid` directly.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
kind : {{ "scatter" | "kde" | "hist" | "hex" | "reg" | "resid" }}
    Kind of plot to draw. See the examples for references to the underlying functions.
height : numeric
    Size of the figure (it will be square).
ratio : numeric
    Ratio of joint axes height to marginal axes height.
space : numeric
    Space between the joint and marginal axes
dropna : bool
    If True, remove observations that are missing from ``x`` and ``y``.
{{x, y}}lim : pairs of numbers
    Axis limits to set before plotting.
{params.core.color}
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
marginal_ticks : bool
    If False, suppress ticks on the count/density axis of the marginal plots.
{{joint, marginal}}_kws : dicts
    Additional keyword arguments for the plot components.
kwargs
    Additional keyword arguments are passed to the function used to
    draw the plot on the joint Axes, superseding items in the
    ``joint_kws`` dictionary.

Returns
-------
{returns.jointgrid}

See Also
--------
{seealso.jointgrid}
{seealso.pairgrid}
{seealso.pairplot}

Examples
--------

.. include:: ../docstrings/jointplot.rst

""".format(
    params=_param_docs,  # 使用 `_param_docs` 提供的参数文档部分填充参数段落
    returns=_core_docs["returns"],  # 使用 `_core_docs` 中的返回值文档部分填充返回值段落
    seealso=_core_docs["seealso"],  # 使用 `_core_docs` 中的相关函数文档部分填充相关函数段落
)
```