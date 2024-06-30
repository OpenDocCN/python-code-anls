# `D:\src\scipysrc\seaborn\seaborn\utils.py`

```
"""Utility functions, mostly for internal use."""

# 导入必要的库和模块
import os  # 操作系统相关功能
import inspect  # 检查对象的内部信息
import warnings  # 警告处理
import colorsys  # 颜色系统转换
from contextlib import contextmanager  # 上下文管理器
from urllib.request import urlopen, urlretrieve  # 网络请求相关
from types import ModuleType  # Python类型相关操作

import numpy as np  # 数值计算库
import pandas as pd  # 数据分析库
import matplotlib as mpl  # 绘图库
from matplotlib.colors import to_rgb  # 颜色转换
import matplotlib.pyplot as plt  # 绘图库的绘图接口
from matplotlib.cbook import normalize_kwargs  # 参数标准化

from seaborn._core.typing import deprecated  # 引入 deprecated 类型
from seaborn.external.version import Version  # 引入 Version 类
from seaborn.external.appdirs import user_cache_dir  # 用户缓存目录

__all__ = ["desaturate", "saturate", "set_hls_values", "move_legend",
           "despine", "get_dataset_names", "get_data_home", "load_dataset"]

# 数据集来源的基础链接
DATASET_SOURCE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master"
# 数据集名称列表的URL
DATASET_NAMES_URL = f"{DATASET_SOURCE}/dataset_names.txt"


def ci_to_errsize(cis, heights):
    """Convert intervals to error arguments relative to plot heights.

    Parameters
    ----------
    cis : 2 x n sequence
        sequence of confidence interval limits
    heights : n sequence
        sequence of plot heights

    Returns
    -------
    errsize : 2 x n array
        sequence of error size relative to height values in correct
        format as argument for plt.bar

    """
    # 将置信区间转换为相对于绘图高度的误差参数
    cis = np.atleast_2d(cis).reshape(2, -1)
    heights = np.atleast_1d(heights)
    errsize = []
    for i, (low, high) in enumerate(np.transpose(cis)):
        h = heights[i]
        elow = h - low
        ehigh = high - h
        errsize.append([elow, ehigh])

    errsize = np.asarray(errsize).T
    return errsize


def _draw_figure(fig):
    """Force draw of a matplotlib figure, accounting for back-compat."""
    # 强制绘制 matplotlib 图形，考虑向后兼容性
    # 查看 https://github.com/matplotlib/matplotlib/issues/19197 获取更多背景信息
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass


def _default_color(method, hue, color, kws, saturation=1):
    """If needed, get a default color by using the matplotlib property cycle."""

    if hue is not None:
        # 如果有需要，通过使用 matplotlib 属性循环获取默认颜色
        # 此警告可能对用户更友好，但当前在 FacetGrid 上下文中触发，暂不修改逻辑
        # if color is not None:
        #     msg = "`color` is ignored when `hue` is assigned."
        #     warnings.warn(msg)
        return None

    kws = kws.copy()
    kws.pop("label", None)

    if color is not None:
        if saturation < 1:
            color = desaturate(color, saturation)
        return color

    elif method.__name__ == "plot":

        color = normalize_kwargs(kws, mpl.lines.Line2D).get("color")
        scout, = method([], [], scalex=False, scaley=False, color=color)
        color = scout.get_color()
        scout.remove()
    elif method.__name__ == "scatter":

        # 检查散点图的大小设置，确保 x/y 的大小与 s/c 匹配，后者可能在 kws 字典中
        scout_size = max(
            np.atleast_1d(kws.get(key, [])).shape[0]
            for key in ["s", "c", "fc", "facecolor", "facecolors"]
        )
        # 创建一个大小为 scout_size 的 NaN 数组作为散点的 x 和 y 坐标
        scout_x = scout_y = np.full(scout_size, np.nan)

        # 调用散点图方法，并传入相应的参数
        scout = method(scout_x, scout_y, **kws)
        # 获取散点图的面颜色
        facecolors = scout.get_facecolors()

        if not len(facecolors):
            # 处理 matplotlib <= 3.2 的 bug
            # 在存在 bug 的版本中，如果未指定颜色参数，可能无法正确识别颜色
            single_color = False
        else:
            # 判断散点图中的颜色是否只有一种
            single_color = np.unique(facecolors, axis=0).shape[0] == 1

        # 如果用户未指定颜色并且只有一种颜色，则使用散点图的第一种颜色
        if "c" not in kws and single_color:
            color = to_rgb(facecolors[0])

        # 移除生成的散点图
        scout.remove()

    elif method.__name__ == "bar":

        # bar() 方法需要使用掩码数据而不是空数据来生成一个图形
        scout, = method([np.nan], [np.nan], **kws)
        # 获取柱状图的颜色并转换为 RGB 格式
        color = to_rgb(scout.get_facecolor())
        # 移除生成的柱状图
        scout.remove()
        # 从容器列表中移除最后一个容器，因为 Axes.bar 方法会添加图形和容器
        method.__self__.containers.pop(-1)

    elif method.__name__ == "fill_between":

        # 标准化 kws 字典，确保其适用于 PolyCollection
        kws = normalize_kwargs(kws, mpl.collections.PolyCollection)
        # 调用 fill_between 方法，并传入空的 x 和 y 数据
        scout = method([], [], **kws)
        # 获取填充区域的面颜色
        facecolor = scout.get_facecolor()
        # 将面颜色转换为 RGB 格式
        color = to_rgb(facecolor[0])
        # 移除生成的填充区域
        scout.remove()

    # 如果饱和度小于 1，则对颜色进行去饱和处理
    if saturation < 1:
        color = desaturate(color, saturation)

    # 返回最终确定的颜色值
    return color
# 减少颜色的饱和度通道一定百分比。

def desaturate(color, prop):
    """Decrease the saturation channel of a color by some percent.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    prop : float
        saturation channel of color will be multiplied by this value

    Returns
    -------
    new_color : rgb tuple
        desaturated color code in RGB tuple representation

    """
    # 检查输入是否在合法范围内
    if not 0 <= prop <= 1:
        raise ValueError("prop must be between 0 and 1")

    # 获取颜色的 RGB 元组表示
    rgb = to_rgb(color)

    # 避免浮点数问题的短路处理
    if prop == 1:
        return rgb

    # 将颜色转换为 HLS（色相、亮度、饱和度）空间
    h, l, s = colorsys.rgb_to_hls(*rgb)

    # 减少饱和度通道的值
    s *= prop

    # 转换回 RGB
    new_color = colorsys.hls_to_rgb(h, l, s)

    return new_color


def saturate(color):
    """Return a fully saturated color with the same hue.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name

    Returns
    -------
    new_color : rgb tuple
        saturated color code in RGB tuple representation

    """
    # 返回具有相同色调的完全饱和的颜色
    return set_hls_values(color, s=1)


def set_hls_values(color, h=None, l=None, s=None):  # noqa
    """Independently manipulate the h, l, or s channels of a color.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    h, l, s : floats between 0 and 1, or None
        new values for each channel in hls space

    Returns
    -------
    new_color : rgb tuple
        new color code in RGB tuple representation

    """
    # 获取颜色的 RGB 元组表示
    rgb = to_rgb(color)
    
    # 将颜色转换为 HLS（色相、亮度、饱和度）空间，并修改各通道值
    vals = list(colorsys.rgb_to_hls(*rgb))
    for i, val in enumerate([h, l, s]):
        if val is not None:
            vals[i] = val

    # 转换回 RGB
    rgb = colorsys.hls_to_rgb(*vals)
    return rgb


def axlabel(xlabel, ylabel, **kwargs):
    """Grab current axis and label it.

    DEPRECATED: will be removed in a future version.

    """
    # 警告用户此函数将在未来版本中移除
    msg = "This function is deprecated and will be removed in a future version"
    warnings.warn(msg, FutureWarning)
    
    # 获取当前图表的坐标轴并设置标签
    ax = plt.gca()
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)


def remove_na(vector):
    """Helper method for removing null values from data vectors.

    Parameters
    ----------
    vector : vector object
        Must implement boolean masking with [] subscript syntax.

    Returns
    -------
    clean_clean : same type as ``vector``
        Vector of data with null values removed. May be a copy or a view.

    """
    # 帮助函数，用于从数据向量中移除空值
    return vector[pd.notnull(vector)]


def get_color_cycle():
    """Return the list of colors in the current matplotlib color cycle

    Parameters
    ----------
    None

    Returns
    -------
    colors : list
        List of matplotlib colors in the current cycle, or dark gray if
        the current color cycle is empty.
    """
    # 返回当前 matplotlib 颜色循环中的颜色列表
    cycler = mpl.rcParams['axes.prop_cycle']
    # 如果 'color' 是 cycler.by_key() 返回的字典的键之一，则返回其对应的值；否则返回包含字符串 ".15" 的列表。
    return cycler.by_key()['color'] if 'color' in cycler.keys else [".15"]
# 移除绘图中的顶部和右侧边框线
def despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False, offset=None, trim=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
        用于去除所有轴的图形对象，默认为当前图形。
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
        指定要去除轴线的特定轴对象。如果提供了 fig 参数，则忽略此参数。
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
        如果为 True，则移除对应的轴线。
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
        距离轴线的绝对距离（以点为单位），负值向内移动轴线。可以是单个值（应用于所有轴线），也可以是字典（每条边上设置不同的偏移值）。
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.
        如果为 True，则将轴线限制在每个非去轴轴上的最小和最大主刻度之间。

    Returns
    -------
    None
    """
    # 获取需要处理的轴对象的引用
    if fig is None and ax is None:
        axes = plt.gcf().axes  # 如果未提供 fig 和 ax 参数，则获取当前图形的所有轴对象
    elif fig is not None:
        axes = fig.axes  # 如果提供了 fig 参数，则获取该图形的所有轴对象
    elif ax is not None:
        axes = [ax]  # 如果提供了 ax 参数，则使用该特定的轴对象
    # 对每个子图对象中的轴进行操作
    for ax_i in axes:
        # 遍历四个轴脊（上、右、左、下）
        for side in ["top", "right", "left", "bottom"]:
            # 切换脊的可见性
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            # 如果设置了偏移量并且脊是可见的，则尝试获取偏移量值并设置脊的位置
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(('outward', val))

        # 可能需要移动刻度线的位置
        if left and not right:
            # 检查 y 轴主要刻度线和次要刻度线的可见性
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.minorTicks
            )
            # 将 y 轴刻度线位置设置为右侧
            ax_i.yaxis.set_ticks_position("right")
            # 设置 y 轴主要刻度线的次要刻度线的可见性
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            # 检查 x 轴主要刻度线和次要刻度线的可见性
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.minorTicks
            )
            # 将 x 轴刻度线位置设置为顶部
            ax_i.xaxis.set_ticks_position("top")
            # 设置 x 轴主要刻度线的次要刻度线的可见性
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if trim:
            # 裁剪掉超出主要刻度线范围的脊的部分
            xticks = np.asarray(ax_i.get_xticks())
            if xticks.size:
                # 获取在 x 轴范围内的第一个和最后一个刻度线
                firsttick = np.compress(xticks >= min(ax_i.get_xlim()),
                                        xticks)[0]
                lasttick = np.compress(xticks <= max(ax_i.get_xlim()),
                                       xticks)[-1]
                # 设置底部和顶部脊的边界为刻度线的范围
                ax_i.spines['bottom'].set_bounds(firsttick, lasttick)
                ax_i.spines['top'].set_bounds(firsttick, lasttick)
                # 筛选出在新的刻度线范围内的刻度值
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                # 获取在 y 轴范围内的第一个和最后一个刻度线
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()),
                                        yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()),
                                       yticks)[-1]
                # 设置左侧和右侧脊的边界为刻度线的范围
                ax_i.spines['left'].set_bounds(firsttick, lasttick)
                ax_i.spines['right'].set_bounds(firsttick, lasttick)
                # 筛选出在新的刻度线范围内的刻度值
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)
# 将图例移动到新位置的函数。

"""
Recreate a plot's legend at a new location.

The name is a slight misnomer. Matplotlib legends do not expose public
control over their position parameters. So this function creates a new legend,
copying over the data from the original object, which is then removed.

Parameters
----------
obj : the object with the plot
    This argument can be either a seaborn or matplotlib object:

    - :class:`seaborn.FacetGrid` or :class:`seaborn.PairGrid`
    - :class:`matplotlib.axes.Axes` or :class:`matplotlib.figure.Figure`

loc : str or int
    Location argument, as in :meth:`matplotlib.axes.Axes.legend`.

kwargs
    Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.legend`.

Examples
--------

.. include:: ../docstrings/move_legend.rst

"""

# This is a somewhat hackish solution that will hopefully be obviated by
# upstream improvements to matplotlib legends that make them easier to
# modify after creation.

from seaborn.axisgrid import Grid  # Avoid circular import

# Locate the legend object and a method to recreate the legend
if isinstance(obj, Grid):
    old_legend = obj.legend
    legend_func = obj.figure.legend
elif isinstance(obj, mpl.axes.Axes):
    old_legend = obj.legend_
    legend_func = obj.legend
elif isinstance(obj, mpl.figure.Figure):
    if obj.legends:
        old_legend = obj.legends[-1]
    else:
        old_legend = None
    legend_func = obj.legend
else:
    err = "`obj` must be a seaborn Grid or matplotlib Axes or Figure instance."
    raise TypeError(err)

if old_legend is None:
    err = f"{obj} has no legend attached."
    raise ValueError(err)

# Extract the components of the legend we need to reuse
# Import here to avoid a circular import
from seaborn._compat import get_legend_handles
handles = get_legend_handles(old_legend)
labels = [t.get_text() for t in old_legend.get_texts()]

# Handle the case where the user is trying to override the labels
if (new_labels := kwargs.pop("labels", None)) is not None:
    if len(new_labels) != len(labels):
        err = "Length of new labels does not match existing legend."
        raise ValueError(err)
    labels = new_labels

# Extract legend properties that can be passed to the recreation method
# (Vexingly, these don't all round-trip)
legend_kws = inspect.signature(mpl.legend.Legend).parameters
props = {k: v for k, v in old_legend.properties().items() if k in legend_kws}

# Delegate default bbox_to_anchor rules to matplotlib
props.pop("bbox_to_anchor")

# Try to propagate the existing title and font properties; respect new ones too
title = props.pop("title")
if "title" in kwargs:
    title.set_text(kwargs.pop("title"))
    # 从参数 kwargs 中筛选出所有以 "title_" 开头的键值对，组成一个新的字典 title_kwargs
    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith("title_")}
    
    # 遍历 title_kwargs 字典中的每个键值对
    for key, val in title_kwargs.items():
        # 设置标题对象的属性，去掉键名中的 "title_" 部分，设置对应的属性值为 val
        title.set(**{key[6:]: val})
    
        # 从原始的 kwargs 中移除已处理的键值对
        kwargs.pop(key)
    
    # 尝试根据旧的图例对象的可见性设置参数 "frameon" 的默认值
    kwargs.setdefault("frameon", old_legend.legendPatch.get_visible())
    
    # 更新属性字典 props，将 kwargs 中的内容更新到 props 中
    props.update(kwargs)
    
    # 移除旧的图例对象
    old_legend.remove()
    
    # 调用 legend_func 函数创建新的图例对象 new_legend，并传入指定的参数
    new_legend = legend_func(handles, labels, loc=loc, **props)
    
    # 设置新图例对象的标题文本和字体属性
    new_legend.set_title(title.get_text(), title.get_fontproperties())
    
    # 如果 obj 是 Grid 类型的对象，则更新其内部记录的图例对象为 new_legend
    if isinstance(obj, Grid):
        obj._legend = new_legend
# 计算核密度估计的支持范围
def _kde_support(data, bw, gridsize, cut, clip):
    # 计算支持范围的最小值，确保不低于指定的剪切范围下限
    support_min = max(data.min() - bw * cut, clip[0])
    # 计算支持范围的最大值，确保不高于指定的剪切范围上限
    support_max = min(data.max() + bw * cut, clip[1])
    # 在支持范围内生成均匀间隔的网格点
    support = np.linspace(support_min, support_max, gridsize)

    return support


# 计算给定数组的置信区间
def ci(a, which=95, axis=None):
    # 根据指定的置信水平计算百分位数范围
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


# 获取可用示例数据集的名称列表
def get_dataset_names():
    """Report available example datasets, useful for reporting issues.

    Requires an internet connection.

    """
    # 通过URL获取可用数据集的名称列表
    with urlopen(DATASET_NAMES_URL) as resp:
        txt = resp.read()

    # 将获取的文本解码并按行分割，去除每行两端的空白字符
    dataset_names = [name.strip() for name in txt.decode().split("\n")]
    # 返回非空的数据集名称列表
    return list(filter(None, dataset_names))


# 返回用于示例数据集缓存的目录路径
def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.

    This directory is used by :func:`load_dataset`.

    If the ``data_home`` argument is not provided, it will use a directory
    specified by the `SEABORN_DATA` environment variable (if it exists)
    or otherwise default to an OS-appropriate user cache location.

    """
    # 如果未提供 data_home 参数，则使用环境变量 SEABORN_DATA 指定的路径，否则使用用户缓存目录
    if data_home is None:
        data_home = os.environ.get("SEABORN_DATA", user_cache_dir("seaborn"))
    # 将路径中的波浪号展开为用户目录
    data_home = os.path.expanduser(data_home)
    # 如果目录不存在，则创建该目录
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    # 返回数据集缓存目录的路径
    return data_home


# 从在线仓库加载示例数据集（需要网络连接）
def load_dataset(name, cache=True, data_home=None, **kws):
    """Load an example dataset from the online repository (requires internet).

    This function provides quick access to a small number of example datasets
    that are useful for documenting seaborn or generating reproducible examples
    for bug reports. It is not necessary for normal usage.

    Note that some of the datasets have a small amount of preprocessing applied
    to define a proper ordering for categorical variables.

    Use :func:`get_dataset_names` to see a list of available datasets.

    Parameters
    ----------
    name : str
        Name of the dataset (``{name}.csv`` on
        https://github.com/mwaskom/seaborn-data).
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.

    """
    # 提示用户不要误以为需要将个人数据通过此函数传递，以便与 seaborn 一起使用
    # 提供比通常的错误信息更有帮助的错误提示
    # 检查传入的数据集名称是否为 pandas DataFrame 类型，如果是则抛出类型错误
    if isinstance(name, pd.DataFrame):
        err = (
            "This function accepts only strings (the name of an example dataset). "
            "You passed a pandas DataFrame. If you have your own dataset, "
            "it is not necessary to use this function before plotting."
        )
        raise TypeError(err)
    
    # 构造数据集的 URL 地址，格式为 DATASET_SOURCE/name.csv
    url = f"{DATASET_SOURCE}/{name}.csv"

    # 如果 cache 为 True，则缓存数据集到本地
    if cache:
        # 计算缓存文件的路径，使用数据集 URL 的基本名称
        cache_path = os.path.join(get_data_home(data_home), os.path.basename(url))
        # 如果缓存文件不存在，则下载数据集到缓存路径
        if not os.path.exists(cache_path):
            # 如果数据集名称不在可用数据集列表中，则抛出数值错误
            if name not in get_dataset_names():
                raise ValueError(f"'{name}' is not one of the example datasets.")
            urlretrieve(url, cache_path)
        # 设置完整路径为缓存路径
        full_path = cache_path
    else:
        # 如果不缓存数据集，则使用数据集的 URL 作为完整路径
        full_path = url

    # 使用 pandas 读取 CSV 文件，传入额外的关键字参数 kws
    df = pd.read_csv(full_path, **kws)

    # 如果数据集最后一行全部为空值，则移除该行
    if df.iloc[-1].isnull().all():
        df = df.iloc[:-1]

    # 根据数据集名称对特定列进行类型设置为有序分类
    
    # 如果数据集名称为 "tips"
    if name == "tips":
        # 设置 "day" 列为有序分类，指定顺序为 ["Thur", "Fri", "Sat", "Sun"]
        df["day"] = pd.Categorical(df["day"], ["Thur", "Fri", "Sat", "Sun"])
        # 设置 "sex" 列为有序分类，指定顺序为 ["Male", "Female"]
        df["sex"] = pd.Categorical(df["sex"], ["Male", "Female"])
        # 设置 "time" 列为有序分类，指定顺序为 ["Lunch", "Dinner"]
        df["time"] = pd.Categorical(df["time"], ["Lunch", "Dinner"])
        # 设置 "smoker" 列为有序分类，指定顺序为 ["Yes", "No"]
        df["smoker"] = pd.Categorical(df["smoker"], ["Yes", "No"])

    # 如果数据集名称为 "flights"
    elif name == "flights":
        # 提取月份的前三个字母，将 "month" 列设置为有序分类，使用唯一值作为顺序
        months = df["month"].str[:3]
        df["month"] = pd.Categorical(months, months.unique())

    # 如果数据集名称为 "exercise"
    elif name == "exercise":
        # 设置 "time" 列为有序分类，指定顺序为 ["1 min", "15 min", "30 min"]
        df["time"] = pd.Categorical(df["time"], ["1 min", "15 min", "30 min"])
        # 设置 "kind" 列为有序分类，指定顺序为 ["rest", "walking", "running"]
        df["kind"] = pd.Categorical(df["kind"], ["rest", "walking", "running"])
        # 设置 "diet" 列为有序分类，指定顺序为 ["no fat", "low fat"]
        df["diet"] = pd.Categorical(df["diet"], ["no fat", "low fat"])

    # 如果数据集名称为 "titanic"
    elif name == "titanic":
        # 设置 "class" 列为有序分类，指定顺序为 ["First", "Second", "Third"]
        df["class"] = pd.Categorical(df["class"], ["First", "Second", "Third"])
        # 设置 "deck" 列为有序分类，使用列表 "ABCDEFG" 作为顺序
        df["deck"] = pd.Categorical(df["deck"], list("ABCDEFG"))

    # 如果数据集名称为 "penguins"
    elif name == "penguins":
        # 将 "sex" 列的字符串首字母大写，使其成为标题格式
        df["sex"] = df["sex"].str.title()

    # 如果数据集名称为 "diamonds"
    elif name == "diamonds":
        # 设置 "color" 列为有序分类，指定顺序为 ["D", "E", "F", "G", "H", "I", "J"]
        df["color"] = pd.Categorical(
            df["color"], ["D", "E", "F", "G", "H", "I", "J"],
        )
        # 设置 "clarity" 列为有序分类，指定顺序为 ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
        df["clarity"] = pd.Categorical(
            df["clarity"], ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
        )
        # 设置 "cut" 列为有序分类，指定顺序为 ["Ideal", "Premium", "Very Good", "Good", "Fair"]
        df["cut"] = pd.Categorical(
            df["cut"], ["Ideal", "Premium", "Very Good", "Good", "Fair"],
        )

    # 如果数据集名称为 "taxis"
    elif name == "taxis":
        # 将 "pickup" 和 "dropoff" 列转换为 pandas 的 datetime 类型
        df["pickup"] = pd.to_datetime(df["pickup"])
        df["dropoff"] = pd.to_datetime(df["dropoff"])

    # 如果数据集名称为 "seaice" 或 "dowjones"
    elif name == "seaice" or name == "dowjones":
        # 将 "Date" 列转换为 pandas 的 datetime 类型
        df["Date"] = pd.to_datetime(df["Date"])

    # 返回处理后的数据集 df
    return df
# 检查给定的刻度标签列表是否存在重叠，返回布尔值

def axis_ticklabels_overlap(labels):
    """Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels
        刻度标签的列表

    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.
        如果任何标签重叠则返回True，否则返回False

    """
    if not labels:
        return False
    try:
        # 获取每个标签的绘图区域
        bboxes = [l.get_window_extent() for l in labels]
        # 检查每个标签的重叠情况
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1  # 如果有任何标签重叠次数大于1，则返回True
    except RuntimeError:
        # 处理在macOS后端上可能引发的错误
        return False


# 检查给定Axes上的x和y轴刻度标签是否重叠，返回布尔值对

def axes_ticklabels_overlap(ax):
    """Return booleans for whether the x and y ticklabels on an Axes overlap.

    Parameters
    ----------
    ax : matplotlib Axes
        Matplotlib绘图中的Axes对象

    Returns
    -------
    x_overlap, y_overlap : booleans
        True when the labels on that axis overlap.
        如果对应轴上的标签重叠，则返回True

    """
    return (axis_ticklabels_overlap(ax.get_xticklabels()),  # 检查x轴刻度标签是否重叠
            axis_ticklabels_overlap(ax.get_yticklabels()))  # 检查y轴刻度标签是否重叠


# 将定位器返回的刻度转换为图例条目的级别和格式化级别

def locator_to_legend_entries(locator, limits, dtype):
    """Return levels and formatted levels for brief numeric legends."""
    # 获取原始刻度值，并确保它们在指定的限制范围内
    raw_levels = locator.tick_values(*limits).astype(dtype)

    # 剪裁超出限制范围的刻度值
    raw_levels = [l for l in raw_levels if l >= limits[0] and l <= limits[1]]

    # 创建一个虚拟的Axes对象以设置格式化器
    class dummy_axis:
        def get_view_interval(self):
            return limits

    # 根据定位器类型选择适当的格式化器
    if isinstance(locator, mpl.ticker.LogLocator):
        formatter = mpl.ticker.LogFormatter()
    else:
        formatter = mpl.ticker.ScalarFormatter()
        # 避免在图例中使用科学计数法或偏移量
        formatter.set_useOffset(False)
        formatter.set_scientific(False)
    formatter.axis = dummy_axis()

    # 使用格式化器对原始刻度值进行格式化
    formatted_levels = formatter.format_ticks(raw_levels)

    return raw_levels, formatted_levels  # 返回原始刻度值和格式化后的刻度值


# 计算颜色相对亮度，根据W3C标准

def relative_luminance(color):
    """Calculate the relative luminance of a color according to W3C standards

    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.

    Returns
    -------
    luminance : float(s) between 0 and 1
        相对亮度，范围在0到1之间的浮点数

    """
    # 将颜色转换为RGB数组，并应用相对亮度计算公式
    rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])  # 计算加权平均亮度
    try:
        return lum.item()  # 尝试返回亮度的标量值
    except ValueError:
        return lum  # 处理异常情况下的返回亮度值


# 返回表示Python对象的字符串

def to_utf8(obj):
    """Return a string representing a Python object.

    Strings (i.e. type ``str``) are returned unchanged.

    Byte strings (i.e. type ``bytes``) are returned as UTF-8-decoded strings.

    For other objects, the method ``__str__()`` is called, and the result is
    returned as a string.

    Parameters
    ----------
    obj : object
        Any Python object

    Returns
    -------
    s : str
        UTF-8-decoded string representation of ``obj``

    """
    if isinstance(obj, str):
        return obj  # 如果是字符串，则直接返回
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')  # 如果是字节串，则解码为UTF-8编码的字符串
    else:
        return str(obj)  # 对于其他对象，调用其__str__()方法并返回结果字符串化
    """
    如果输入的 obj 是字符串类型，直接返回该字符串
    如果不是字符串类型，尝试将其解码成 UTF-8 编码的字符串
    如果无法解码（即 obj 不是类字节型对象），则将 obj 转换成字符串返回
    """
    if isinstance(obj, str):
        return obj
    try:
        return obj.decode(encoding="utf-8")
    except AttributeError:  # 如果 obj 不是类字节型对象，则捕获 AttributeError 异常
        return str(obj)
# 检查参数值是否在指定选项列表中，如果不在则抛出 ValueError 异常。
def _check_argument(param, options, value, prefix=False):
    if prefix and value is not None:
        # 如果 prefix 为真且值不为 None，则检查值是否以选项列表中任意字符串开头
        failure = not any(value.startswith(p) for p in options if isinstance(p, str))
    else:
        # 否则直接检查值是否在选项列表中
        failure = value not in options
    if failure:
        # 如果检查失败，则抛出 ValueError 异常
        raise ValueError(
            f"The value for `{param}` must be one of {options}, "
            f"but {repr(value)} was passed."
        )
    return value


# 从源函数的默认参数中为调用函数的 kwargs 分配默认值
def _assign_default_kwargs(kws, call_func, source_func):
    # 获取调用函数的参数签名
    needed = inspect.signature(call_func).parameters
    # 获取源函数的参数签名及其默认值
    defaults = inspect.signature(source_func).parameters

    # 遍历调用函数的参数，如果该参数在源函数中有默认值且在 kwargs 中不存在，则将其默认值分配给 kwargs
    for param in needed:
        if param in defaults and param not in kws:
            kws[param] = defaults[param].default

    return kws


# 调整图例中不可见句柄的 "子标题"，使其更像标题
def adjust_legend_subtitles(legend):
    # 获取图例标题的字体大小，如果未设置则为 None
    font_size = plt.rcParams.get("legend.title_fontsize", None)
    # 获取图例中 VPacker 的子元素列表
    hpackers = legend.findobj(mpl.offsetbox.VPacker)[0].get_children()
    for hpack in hpackers:
        # 获取每个 VPacker 中的绘制区域和文本区域
        draw_area, text_area = hpack.get_children()
        # 获取绘制区域的句柄列表
        handles = draw_area.get_children()
        # 如果所有句柄都不可见，则将绘制区域的宽度设置为 0
        if not all(artist.get_visible() for artist in handles):
            draw_area.set_width(0)
            # 遍历文本区域的子元素，如果设置了字体大小则应用于文本
            for text in text_area.get_children():
                if font_size is not None:
                    text.set_size(font_size)


# 警告：ci= 参数已弃用，转换为适当的 errorbar= 参数
def _deprecate_ci(errorbar, ci):
    if ci is not deprecated and ci != "deprecated":
        # 如果 ci 参数不是 deprecated 且不为 None，则根据不同的值设置 errorbar
        if ci is None:
            errorbar = None
        elif ci == "sd":
            errorbar = "sd"
        else:
            errorbar = ("ci", ci)
        # 发出警告信息，提醒使用者更新代码
        msg = (
            "\n\nThe `ci` parameter is deprecated. "
            f"Use `errorbar={repr(errorbar)}` for the same effect.\n"
        )
        warnings.warn(msg, FutureWarning, stacklevel=3)

    return errorbar


# 返回给定轴的前向和逆向变换函数
def _get_transform_functions(ax, axis):
    axis_obj = getattr(ax, f"{axis}axis")
    # 获取 axis_obj 的坐标变换对象
    transform = axis_obj.get_transform()
    # 返回坐标变换对象的 transform 方法和其反转后的 transform 方法
    return transform.transform, transform.inverted().transform
# 创建一个上下文管理器，用于禁用 matplotlib 中由 rc 参数控制的自动布局行为
@contextmanager
def _disable_autolayout():
    """Context manager for preventing rc-controlled auto-layout behavior."""
    # 保存当前 figure.autolayout 的原始设置值
    orig_val = mpl.rcParams["figure.autolayout"]
    try:
        # 将 figure.autolayout 设置为 False，禁用自动布局
        mpl.rcParams["figure.autolayout"] = False
        # 返回控制权，允许执行上下文中的代码
        yield
    finally:
        # 恢复原始的 figure.autolayout 设置值
        mpl.rcParams["figure.autolayout"] = orig_val


def _version_predates(lib: ModuleType, version: str) -> bool:
    """Helper function for checking version compatibility."""
    # 检查给定库的版本是否早于指定的版本号
    return Version(lib.__version__) < Version(version)


def _scatter_legend_artist(**kws):
    # 根据关键字参数创建散点图例子图元
    kws = normalize_kwargs(kws, mpl.collections.PathCollection)

    # 提取 edgecolor 参数，准备设置线条的颜色
    edgecolor = kws.pop("edgecolor", None)
    rc = mpl.rcParams

    # 准备线条的样式参数
    line_kws = {
        "linestyle": "",
        "marker": kws.pop("marker", "o"),
        "markersize": np.sqrt(kws.pop("s", rc["lines.markersize"] ** 2)),
        "markerfacecolor": kws.pop("facecolor", kws.get("color")),
        "markeredgewidth": kws.pop("linewidth", 0),
        **kws,
    }

    # 根据 edgecolor 参数设置 markeredgecolor
    if edgecolor is not None:
        if edgecolor == "face":
            line_kws["markeredgecolor"] = line_kws["markerfacecolor"]
        else:
            line_kws["markeredgecolor"] = edgecolor

    # 返回一个 matplotlib 的 Line2D 对象，代表散点图例子
    return mpl.lines.Line2D([], [], **line_kws)


def _get_patch_legend_artist(fill):
    # 创建一个函数，用于生成填充或非填充的图例图元

    def legend_artist(**kws):
        # 提取 color 参数，并根据 fill 参数设置 facecolor 或 edgecolor
        color = kws.pop("color", None)
        if color is not None:
            if fill:
                kws["facecolor"] = color
            else:
                kws["edgecolor"] = color
                kws["facecolor"] = "none"

        # 返回一个 matplotlib 的 Rectangle 对象，代表图例图元
        return mpl.patches.Rectangle((0, 0), 0, 0, **kws)

    # 返回内部定义的 legend_artist 函数
    return legend_artist
```