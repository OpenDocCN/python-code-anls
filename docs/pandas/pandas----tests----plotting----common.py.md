# `D:\src\scipysrc\pandas\pandas\tests\plotting\common.py`

```
"""
Module consolidating common testing functions for checking plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.api import is_list_like

import pandas as pd
from pandas import Series
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _check_legend_labels(axes, labels=None, visible=True):
    """
    Check each axes has expected legend labels

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
        Matplotlib Axes object or a list of such objects to check
    labels : list-like
        Expected legend labels to match against axes
    visible : bool
        Expected legend visibility. Labels are checked only when visible is True
    """
    if visible and (labels is None):
        raise ValueError("labels must be specified when visible is True")
    axes = _flatten_visible(axes)
    for ax in axes:
        if visible:
            assert ax.get_legend() is not None
            _check_text_labels(ax.get_legend().get_texts(), labels)
        else:
            assert ax.get_legend() is None


def _check_legend_marker(ax, expected_markers=None, visible=True):
    """
    Check ax has expected legend markers

    Parameters
    ----------
    ax : matplotlib Axes object
        Matplotlib Axes object to check for legend markers
    expected_markers : list-like
        Expected legend markers to match against ax
    visible : bool
        Expected legend visibility. Markers are checked only when visible is True
    """
    if visible and (expected_markers is None):
        raise ValueError("Markers must be specified when visible is True")
    if visible:
        handles, _ = ax.get_legend_handles_labels()
        markers = [handle.get_marker() for handle in handles]
        assert markers == expected_markers
    else:
        assert ax.get_legend() is None


def _check_data(xp, rs):
    """
    Check each axes has identical lines

    Parameters
    ----------
    xp : matplotlib Axes object
        Matplotlib Axes object to compare lines
    rs : matplotlib Axes object
        Matplotlib Axes object to compare lines with xp
    """
    xp_lines = xp.get_lines()
    rs_lines = rs.get_lines()

    assert len(xp_lines) == len(rs_lines)
    for xpl, rsl in zip(xp_lines, rs_lines):
        xpdata = xpl.get_xydata()
        rsdata = rsl.get_xydata()
        tm.assert_almost_equal(xpdata, rsdata)


def _check_visible(collections, visible=True):
    """
    Check each artist is visible or not

    Parameters
    ----------
    collections : matplotlib Artist or its list-like
        Target Artist or its list or collection to check visibility
    visible : bool
        Expected visibility state for each artist
    """
    from matplotlib.collections import Collection

    if not isinstance(collections, Collection) and not is_list_like(collections):
        collections = [collections]

    for patch in collections:
        assert patch.get_visible() == visible


def _check_patches_all_filled(axes: Axes | Sequence[Axes], filled: bool = True) -> None:
    """
    Check for each artist whether it is filled or not

    Parameters
    ----------
    axes : matplotlib Axes object or its Sequence
        Matplotlib Axes object or a sequence of such objects to check
    filled : bool, optional
        Expected filled state for each artist, defaults to True
    """
    # 获取展开后的 matplotlib Axes 对象列表
    axes = _flatten_visible(axes)
    
    # 遍历每个 Axes 对象
    for ax in axes:
        # 遍历当前 Axes 对象的所有图形补丁（patches）
        for patch in ax.patches:
            # 断言当前补丁（patch）的填充状态是否与给定的 filled 参数相符
            assert patch.fill == filled
# 返回一个字典，将 series 中的唯一值映射到 colors 中对应的颜色
def _get_colors_mapped(series, colors):
    # 获取 series 中的唯一值
    unique = series.unique()
    # 使用唯一值和 colors 创建映射字典
    mapped = dict(zip(unique, colors))
    # 返回根据 series 的值映射到 colors 的结果列表
    return [mapped[v] for v in series.values]


# 检查每个图形对象是否具有预期的线条颜色和填充颜色
def _check_colors(collections, linecolors=None, facecolors=None, mapping=None):
    """
    Check each artist has expected line colors and face colors

    Parameters
    ----------
    collections : list-like
        list or collection of target artist
    linecolors : list-like which has the same length as collections
        list of expected line colors
    facecolors : list-like which has the same length as collections
        list of expected face colors
    mapping : Series
        Series used for color grouping key
        used for andrew_curves, parallel_coordinates, radviz test
    """
    # 导入需要的模块
    from matplotlib import colors
    from matplotlib.collections import (
        Collection,
        LineCollection,
        PolyCollection,
    )
    from matplotlib.lines import Line2D

    # 颜色转换器
    conv = colors.ColorConverter
    # 检查线条颜色
    if linecolors is not None:
        # 如果提供了 mapping，则将 linecolors 根据 mapping 进行映射
        if mapping is not None:
            linecolors = _get_colors_mapped(mapping, linecolors)
            linecolors = linecolors[: len(collections)]

        # 断言 collections 和 linecolors 的长度相同
        assert len(collections) == len(linecolors)
        # 遍历每个图形对象及其对应的预期线条颜色
        for patch, color in zip(collections, linecolors):
            if isinstance(patch, Line2D):
                # 对于 Line2D 对象，获取其颜色并转换为 RGBA 格式
                result = patch.get_color()
                result = conv.to_rgba(result)
            elif isinstance(patch, (PolyCollection, LineCollection)):
                # 对于 PolyCollection 或 LineCollection，获取其边缘颜色并转换为元组格式
                result = tuple(patch.get_edgecolor()[0])
            else:
                # 其他情况，获取其边缘颜色
                result = patch.get_edgecolor()

            # 将预期颜色转换为 RGBA 格式
            expected = conv.to_rgba(color)
            # 断言实际结果与预期结果相同
            assert result == expected

    # 检查填充颜色
    if facecolors is not None:
        # 如果提供了 mapping，则将 facecolors 根据 mapping 进行映射
        if mapping is not None:
            facecolors = _get_colors_mapped(mapping, facecolors)
            facecolors = facecolors[: len(collections)]

        # 断言 collections 和 facecolors 的长度相同
        assert len(collections) == len(facecolors)
        # 遍历每个图形对象及其对应的预期填充颜色
        for patch, color in zip(collections, facecolors):
            if isinstance(patch, Collection):
                # 对于 Collection 对象，获取其填充颜色并转换为元组格式
                result = patch.get_facecolor()[0]
            else:
                # 其他情况，获取其填充颜色
                result = patch.get_facecolor()

            # 如果结果是 ndarray，则转换为元组格式
            if isinstance(result, np.ndarray):
                result = tuple(result)

            # 将预期颜色转换为 RGBA 格式
            expected = conv.to_rgba(color)
            # 断言实际结果与预期结果相同
            assert result == expected


# 检查每个文本对象是否具有预期的标签文本
def _check_text_labels(texts, expected):
    """
    Check each text has expected labels

    Parameters
    ----------
    texts : matplotlib Text object, or its list-like
        target text, or its list
    expected : str or list-like which has the same length as texts
        expected text label, or its list
    """
    # 如果 texts 不是列表形式，则断言其文本内容与预期相同
    if not is_list_like(texts):
        assert texts.get_text() == expected
    else:
        # 从文本列表中提取每个元素的文本内容，形成标签列表
        labels = [t.get_text() for t in texts]
        # 断言标签列表的长度与预期长度相等，确保数据一致性
        assert len(labels) == len(expected)
        # 遍历标签列表和预期列表，逐一比较每个标签与对应的预期值是否相等
        for label, e in zip(labels, expected):
            assert label == e
# 导入 NullFormatter 类以进行刻度格式化
from matplotlib.ticker import NullFormatter

# 对输入的 axes 对象进行扁平化处理，以便处理单个或多个 Axes 对象
axes = _flatten_visible(axes)

# 遍历每一个 Axes 对象
for ax in axes:
    # 检查 x 轴的 minor ticks 是否使用 NullFormatter
    if xlabelsize is not None or xrot is not None:
        if isinstance(ax.xaxis.get_minor_formatter(), NullFormatter):
            # 如果 minor ticks 使用 NullFormatter，则获取主要刻度的标签
            labels = ax.get_xticklabels()
        else:
            # 如果没有使用 NullFormatter，则获取所有刻度（主要和次要）的标签
            labels = ax.get_xticklabels() + ax.get_xticklabels(minor=True)

        # 遍历 x 轴的标签
        for label in labels:
            # 检查 x 轴标签的字体大小是否符合预期
            if xlabelsize is not None:
                tm.assert_almost_equal(label.get_fontsize(), xlabelsize)
            # 检查 x 轴标签的旋转角度是否符合预期
            if xrot is not None:
                tm.assert_almost_equal(label.get_rotation(), xrot)

    # 检查 y 轴的 minor ticks 是否使用 NullFormatter
    if ylabelsize is not None or yrot is not None:
        if isinstance(ax.yaxis.get_minor_formatter(), NullFormatter):
            # 如果 minor ticks 使用 NullFormatter，则获取主要刻度的标签
            labels = ax.get_yticklabels()
        else:
            # 如果没有使用 NullFormatter，则获取所有刻度（主要和次要）的标签
            labels = ax.get_yticklabels() + ax.get_yticklabels(minor=True)

        # 遍历 y 轴的标签
        for label in labels:
            # 检查 y 轴标签的字体大小是否符合预期
            if ylabelsize is not None:
                tm.assert_almost_equal(label.get_fontsize(), ylabelsize)
            # 检查 y 轴标签的旋转角度是否符合预期
            if yrot is not None:
                tm.assert_almost_equal(label.get_rotation(), yrot)
    # 如果指定了轴数，进行以下检查
    if axes_num is not None:
        # 断言可见轴列表的长度与指定轴数相同
        assert len(visible_axes) == axes_num
        # 对于每个可见轴进行检查，确保有绘制的内容
        for ax in visible_axes:
            # 检查可见轴上的子元素数量大于0
            assert len(ax.get_children()) > 0

    # 如果指定了布局参数，进行以下检查
    if layout is not None:
        # 初始化x和y坐标集合
        x_set = set()
        y_set = set()
        # 遍历展平后的所有轴
        for ax in flatten_axes(axes):
            # 获取当前轴的位置坐标点
            points = ax.get_position().get_points()
            # 将x坐标的起始点和y坐标的起始点加入集合
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        # 计算布局的结果，即x和y坐标集合的长度
        result = (len(y_set), len(x_set))
        # 断言计算得到的布局结果与预期布局参数相同
        assert result == layout

    # 检查第一个可见轴的图形尺寸是否与指定的figsize参数相等
    tm.assert_numpy_array_equal(
        visible_axes[0].figure.get_size_inches(),
        np.array(figsize, dtype=np.float64),
    )
# 将传入的 axes 参数展平，仅保留可见的 matplotlib Axes 对象
def _flatten_visible(axes: Axes | Sequence[Axes]) -> Sequence[Axes]:
    """
    Flatten axes, and filter only visible

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like

    """
    from pandas.plotting._matplotlib.tools import flatten_axes

    # 调用 flatten_axes 函数将 axes 展平成 ndarray
    axes_ndarray = flatten_axes(axes)
    # 过滤出所有可见的 Axes 对象
    axes = [ax for ax in axes_ndarray if ax.get_visible()]
    return axes


# 检查 axes 中是否包含预期数量的误差条
def _check_has_errorbars(axes, xerr=0, yerr=0):
    """
    Check axes has expected number of errorbars

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xerr : number
        expected number of x errorbar
    yerr : number
        expected number of y errorbar
    """
    # 将 axes 展平并过滤出可见的 Axes 对象
    axes = _flatten_visible(axes)
    for ax in axes:
        # 获取 Axes 对象中的容器
        containers = ax.containers
        xerr_count = 0
        yerr_count = 0
        for c in containers:
            # 检查容器对象是否有 xerr 和 yerr 属性
            has_xerr = getattr(c, "has_xerr", False)
            has_yerr = getattr(c, "has_yerr", False)
            if has_xerr:
                xerr_count += 1
            if has_yerr:
                yerr_count += 1
        # 断言实际的 xerr_count 和 yerr_count 分别与预期的 xerr 和 yerr 相等
        assert xerr == xerr_count
        assert yerr == yerr_count


# 检查 boxplot 返回的类型是否正确
def _check_box_return_type(
    returned, return_type, expected_keys=None, check_ax_title=True
):
    """
    Check box returned type is correct

    Parameters
    ----------
    returned : object to be tested, returned from boxplot
    return_type : str
        return_type passed to boxplot
    expected_keys : list-like, optional
        group labels in subplot case. If not passed,
        the function checks assuming boxplot uses single ax
    check_ax_title : bool
        Whether to check the ax.title is the same as expected_key
        Intended to be checked by calling from ``boxplot``.
        Normal ``plot`` doesn't attach ``ax.title``, it must be disabled.
    """
    from matplotlib.axes import Axes

    # 定义类型映射关系，用于检查返回类型是否符合预期
    types = {"dict": dict, "axes": Axes, "both": tuple}
    if expected_keys is None:
        # 如果 expected_keys 未指定，根据 return_type 进行断言
        # 如果 return_type 是 None，则默认为 "dict"
        if return_type is None:
            return_type = "dict"

        # 断言返回对象的类型符合预期的类型
        assert isinstance(returned, types[return_type])
        if return_type == "both":
            # 如果 return_type 是 "both"，则同时断言 ax 和 lines 的类型
            assert isinstance(returned.ax, Axes)
            assert isinstance(returned.lines, dict)
    else:
        # 如果不是 None 类型的返回值，则进行类型和内容的断言检查
        # 当返回值类型为 None 时，该部分代码需要修复
        if return_type is None:
            # 检查所有展开后的返回值是否为 Axes 对象
            for r in _flatten_visible(returned):
                assert isinstance(r, Axes)
            # 返回空，因为 return_type 为 None
            return

        # 断言返回值为 Series 类型
        assert isinstance(returned, Series)

        # 断言返回值的键与期望键相同（无序）
        assert sorted(returned.keys()) == sorted(expected_keys)
        
        # 遍历返回值的键值对，进行类型和内容的断言检查
        for key, value in returned.items():
            assert isinstance(value, types[return_type])

            # 根据 return_type 进行进一步的检查
            # 当 return_type 为 "axes" 时
            if return_type == "axes":
                if check_ax_title:
                    # 检查 Axes 对象的标题是否与键值相符
                    assert value.get_title() == key
            
            # 当 return_type 为 "both" 时
            elif return_type == "both":
                if check_ax_title:
                    # 检查 Axes 对象的标题是否与键值相符
                    assert value.ax.get_title() == key
                # 进一步断言 value.ax 是 Axes 类型，value.lines 是字典类型
                assert isinstance(value.ax, Axes)
                assert isinstance(value.lines, dict)
            
            # 当 return_type 为 "dict" 时
            elif return_type == "dict":
                # 获取 medians 列表的第一个元素作为 line
                line = value["medians"][0]
                # 获取 line 的 axes 对象
                axes = line.axes
                if check_ax_title:
                    # 检查 axes 对象的标题是否与键值相符
                    assert axes.get_title() == key
            
            # 如果 return_type 不在预期范围内，则抛出 AssertionError
            else:
                raise AssertionError
# 检查网格设置是否符合预期，确保绘图默认使用 rcParams['axes.grid'] 设置，GH 9792
def _check_grid_settings(obj, kinds, kws=None):
    import matplotlib as mpl  # 导入 matplotlib 库

    # 判断当前绘图是否显示网格
    def is_grid_on():
        xticks = mpl.pyplot.gca().xaxis.get_major_ticks()  # 获取当前绘图的 x 轴主要刻度
        yticks = mpl.pyplot.gca().yaxis.get_major_ticks()  # 获取当前绘图的 y 轴主要刻度
        xoff = all(not g.gridline.get_visible() for g in xticks)  # 检查 x 轴是否所有网格线均不可见
        yoff = all(not g.gridline.get_visible() for g in yticks)  # 检查 y 轴是否所有网格线均不可见

        return not (xoff and yoff)  # 返回是否存在可见的网格线

    if kws is None:
        kws = {}  # 如果 kws 为空，则初始化为空字典

    spndx = 1  # 设置子图索引起始值
    for kind in kinds:  # 遍历图表类型列表
        # 创建子图并设置绘图风格
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc("axes", grid=False)  # 设置绘图对象的网格线为不可见
        obj.plot(kind=kind, **kws)  # 调用对象的绘图方法并传入参数
        assert not is_grid_on()  # 断言当前绘图不应该显示网格
        mpl.pyplot.clf()  # 清空当前绘图

        # 创建子图并设置绘图风格
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc("axes", grid=True)  # 设置绘图对象的网格线为可见
        obj.plot(kind=kind, grid=False, **kws)  # 调用对象的绘图方法并传入参数，但不显示网格
        assert not is_grid_on()  # 断言当前绘图不应该显示网格
        mpl.pyplot.clf()  # 清空当前绘图

        if kind not in ["pie", "hexbin", "scatter"]:  # 如果图表类型不在指定列表中
            # 创建子图并设置绘图风格
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc("axes", grid=True)  # 设置绘图对象的网格线为可见
            obj.plot(kind=kind, **kws)  # 调用对象的绘图方法并传入参数
            assert is_grid_on()  # 断言当前绘图应该显示网格
            mpl.pyplot.clf()  # 清空当前绘图

            # 创建子图并设置绘图风格
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc("axes", grid=False)  # 设置绘图对象的网格线为不可见
            obj.plot(kind=kind, grid=True, **kws)  # 调用对象的绘图方法并传入参数，但显示网格
            assert is_grid_on()  # 断言当前绘图应该显示网格
            mpl.pyplot.clf()  # 清空当前绘图
    default_axes : bool, optional
        如果为 False（默认行为）：
            - 如果 `ax` 不在 `kwargs` 中，则创建 subplot(211) 并在其上绘图
            - 同样创建新的 subplot(212) 并在其上绘图
            - 特别注意 bootstrap_plot 的特殊情况（见 `_gen_two_subplots`）
        如果为 True：
            - 直接运行绘图函数并传递提供的 kwargs
            - 所有必要的 axes 实例将自动创建
            - 建议在绘图函数自身创建多个 axes 时使用此选项，有助于避免警告信息，如
              'UserWarning: To output multiple subplots,
              the figure containing the passed axes is being cleared'

    **kwargs
        传递给绘图函数的关键字参数。

    Returns
    -------
    最后一个绘图函数返回的绘图对象。
    """
    import matplotlib.pyplot as plt

    # 根据 default_axes 的值选择要使用的绘图生成函数
    if default_axes:
        gen_plots = _gen_default_plot
    else:
        gen_plots = _gen_two_subplots

    ret = None
    # 获取关键字参数中的 figure，如果没有则使用当前的图形对象
    fig = kwargs.get("figure", plt.gcf())
    # 清除当前图形对象的内容
    fig.clf()

    # 对于 gen_plots 函数生成的每个 ret 进行迭代，并确保其为有效的绘图对象
    for ret in gen_plots(f, fig, **kwargs):
        assert_is_valid_plot_return_object(ret)

    # 返回最后一个 ret，即最后一个绘图函数返回的对象
    return ret
# 定义一个生成器函数，用于生成默认方式的绘图
def _gen_default_plot(f, fig, **kwargs):
    # 使用给定的函数 f 和参数 kwargs 生成一个图形
    yield f(**kwargs)


# 定义一个生成器函数，强制在两个子图上创建绘图
def _gen_two_subplots(f, fig, **kwargs):
    # 如果参数 kwargs 中不包含 "ax"，则在图形 fig 上添加一个子图(211)
    if "ax" not in kwargs:
        fig.add_subplot(211)
    # 使用给定的函数 f 和参数 kwargs 在当前图形上生成一个图形
    yield f(**kwargs)

    # 如果函数 f 是 pd.plotting.bootstrap_plot，则断言参数 kwargs 中不应包含 "ax"
    if f is pd.plotting.bootstrap_plot:
        assert "ax" not in kwargs
    else:
        # 否则，将参数 "ax" 设置为 fig 上添加的子图(212)
        kwargs["ax"] = fig.add_subplot(212)
    # 使用更新后的参数 kwargs 在当前图形上生成另一个图形
    yield f(**kwargs)
```