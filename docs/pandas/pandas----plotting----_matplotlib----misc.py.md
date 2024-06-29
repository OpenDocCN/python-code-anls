# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\misc.py`

```
from __future__ import annotations

# 导入必要的库
import random
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

# 导入 pandas 相关模块
from pandas.core.dtypes.missing import notna
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    do_adjust_figure,
    maybe_adjust_figure,
    set_ticks_props,
)

# 如果支持类型检查，则导入必要的类型
if TYPE_CHECKING:
    from collections.abc import Hashable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index, Series


def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax=None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds=None,
    hist_kwds=None,
    range_padding: float = 0.05,
    **kwds,
):
    # 获取 DataFrame 中的数值数据
    df = frame._get_numeric_data()

    # 计算 DataFrame 中的列数
    n = df.columns.size
    # 计算子图的数量
    naxes = n * n
    # 创建子图
    fig, axes = create_subplots(naxes=naxes, figsize=figsize, ax=ax, squeeze=False)

    # 调整图形，消除子图之间的间隔
    maybe_adjust_figure(fig, wspace=0, hspace=0)

    # 创建用于标记缺失值的掩码
    mask = notna(df)

    # 获取兼容的标记器
    marker = _get_marker_compat(marker)

    # 初始化直方图和密度估计的参数
    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # 设置关键字参数的默认值，用于绘制散点图
    kwds.setdefault("edgecolors", "none")

    # 计算每列的数据范围
    boundaries_list = []
    for a in df.columns:
        values = df[a].values[mask[a].values]
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

    # 遍历 DataFrame 的列
    for i, a in enumerate(df.columns):
        for j, b in enumerate(df.columns):
            ax = axes[i, j]

            # 如果是对角线上的子图
            if i == j:
                values = df[a].values[mask[a].values]

                # 根据 diagonal 参数选择绘制直方图或密度估计图
                if diagonal == "hist":
                    ax.hist(values, **hist_kwds)
                elif diagonal in ("kde", "density"):
                    from scipy.stats import gaussian_kde

                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)

                # 设置 x 轴的范围
                ax.set_xlim(boundaries_list[i])

            else:
                # 获取公共元素的掩码
                common = (mask[a] & mask[b]).values

                # 绘制散点图
                ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)

                # 设置 x 和 y 轴的范围
                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])

            # 设置 x 和 y 轴的标签
            ax.set_xlabel(b)
            ax.set_ylabel(a)

            # 如果不是第一列，隐藏 y 轴标签
            if j != 0:
                ax.yaxis.set_visible(False)
            # 如果不是最后一行，隐藏 x 轴标签
            if i != n - 1:
                ax.xaxis.set_visible(False)
    `
        # 检查 DataFrame 的列数是否大于 1
        if len(df.columns) > 1:
            # 获取边界列表中的第一个边界值
            lim1 = boundaries_list[0]
            # 获取 y 轴主刻度的位置
            locs = axes[0][1].yaxis.get_majorticklocs()
            # 筛选出在 lim1 范围内的刻度位置
            locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
            # 计算调整后的刻度位置，归一化到边界范围内
            adj = (locs - lim1[0]) / (lim1[1] - lim1[0])
    
            # 获取第一个子图的 y 轴的 y 轴范围
            lim0 = axes[0][0].get_ylim()
            # 根据调整后的刻度位置调整 y 轴的刻度范围
            adj = adj * (lim0[1] - lim0[0]) + lim0[0]
            # 设置第一个子图 y 轴的刻度位置
            axes[0][0].yaxis.set_ticks(adj)
    
            # 检查所有刻度是否为整数
            if np.all(locs == locs.astype(int)):
                # 如果所有刻度都是整数，将刻度转换为整数类型
                locs = locs.astype(int)
            # 设置第一个子图 y 轴的刻度标签
            axes[0][0].yaxis.set_ticklabels(locs)
    
        # 设置坐标轴的属性，包括标签字体大小、标签旋转角度等
        set_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)
    
        # 返回 axes 对象
        return axes
# 根据给定的标记选择合适的标记，如果标记不在 mpl.lines.lineMarkers 中，默认返回 "o"
def _get_marker_compat(marker):
    if marker not in mpl.lines.lineMarkers:
        return "o"
    return marker


# 绘制基于 RadViz 算法的可视化图形，展示多维数据在二维空间的分布情况
def radviz(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    # 数据归一化函数，将数据线性归一化到 [0, 1] 区间
    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    n = len(frame)  # 数据帧中的样本数量
    classes = frame[class_column].drop_duplicates()  # 获取类别列中的唯一值
    class_col = frame[class_column]  # 提取类别列
    df = frame.drop(class_column, axis=1).apply(normalize)  # 对除类别列外的所有列应用归一化函数

    if ax is None:
        ax = plt.gca()  # 获取当前坐标轴
        ax.set_xlim(-1, 1)  # 设置 x 轴范围
        ax.set_ylim(-1, 1)  # 设置 y 轴范围

    to_plot: dict[Hashable, list[list]] = {}  # 用于存储每个类别的数据点坐标
    # 获取用于绘图的标准颜色列表
    colors = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )

    # 初始化每个类别的数据点列表
    for kls in classes:
        to_plot[kls] = [[], []]

    m = len(frame.columns) - 1  # 数据帧中除类别列外的特征数量
    # 计算在单位圆周上均匀分布的 m 个点的坐标
    s = np.array(
        [(np.cos(t), np.sin(t)) for t in [2 * np.pi * (i / m) for i in range(m)]]
    )

    # 遍历每个样本
    for i in range(n):
        row = df.iloc[i].values  # 获取当前样本的特征值数组
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)  # 将特征值数组扩展成二维数组
        y = (s * row_).sum(axis=0) / row.sum()  # 计算当前样本在二维空间中的坐标
        kls = class_col.iat[i]  # 获取当前样本的类别
        to_plot[kls][0].append(y[0])  # 将当前样本的 x 坐标添加到对应类别的列表中
        to_plot[kls][1].append(y[1])  # 将当前样本的 y 坐标添加到对应类别的列表中

    # 遍历每个类别，并在坐标轴上绘制对应的散点图
    for i, kls in enumerate(classes):
        ax.scatter(
            to_plot[kls][0],
            to_plot[kls][1],
            color=colors[i],
            label=pprint_thing(kls),  # 使用 pprint_thing 函数获取可打印的类别名称
            **kwds,
        )
    ax.legend()  # 添加图例

    # 在坐标轴上添加单位圆周的圆形边界
    ax.add_patch(mpl.patches.Circle((0.0, 0.0), radius=1.0, facecolor="none"))

    # 在坐标轴上添加每个特征点的小圆点及其名称标签
    for xy, name in zip(s, df.columns):
        ax.add_patch(mpl.patches.Circle(xy, radius=0.025, facecolor="gray"))

        # 根据特征点的位置决定标签的相对位置及对齐方式
        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(
                xy[0] - 0.025, xy[1] - 0.025, name, ha="right", va="top", size="small"
            )
        elif xy[0] < 0.0 <= xy[1]:
            ax.text(
                xy[0] - 0.025,
                xy[1] + 0.025,
                name,
                ha="right",
                va="bottom",
                size="small",
            )
        elif xy[1] < 0.0 <= xy[0]:
            ax.text(
                xy[0] + 0.025, xy[1] - 0.025, name, ha="left", va="top", size="small"
            )
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(
                xy[0] + 0.025, xy[1] + 0.025, name, ha="left", va="bottom", size="small"
            )

    ax.axis("equal")  # 设置坐标轴比例为等比例
    return ax  # 返回绘制好的坐标轴对象


def andrews_curves(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    samples: int = 200,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt
    # 定义一个函数 function，接受振幅数组作为参数
    def function(amplitudes):
        # 定义内部函数 f，接受时间参数 t
        def f(t):
            # 获取振幅数组中的第一个元素并计算其除以根号2的结果
            x1 = amplitudes[0]
            result = x1 / np.sqrt(2.0)

            # 复制振幅数组并删除第一个元素，以避免影响原始数组
            coeffs = np.delete(np.copy(amplitudes), 0)
            # 调整 coeffs 数组的形状为 (n/2, 2)，其中 n 是 coeffs 的大小
            coeffs = np.resize(coeffs, (int((coeffs.size + 1) / 2), 2))

            # 生成谐波和用于正弦和余弦函数的参数
            harmonics = np.arange(0, coeffs.shape[0]) + 1
            trig_args = np.outer(harmonics, t)

            # 计算函数的值，使用正弦和余弦函数以及对应的系数
            result += np.sum(
                coeffs[:, 0, np.newaxis] * np.sin(trig_args)
                + coeffs[:, 1, np.newaxis] * np.cos(trig_args),
                axis=0,
            )
            return result

        return f

    # 获取数据帧中的行数
    n = len(frame)
    # 获取数据帧中指定列（class_column）的内容
    class_col = frame[class_column]
    # 获取指定列（class_column）中的唯一值列表
    classes = frame[class_column].drop_duplicates()
    # 从数据帧中删除指定列（class_column），并得到新的数据帧 df
    df = frame.drop(class_column, axis=1)
    # 在 [-π, π] 范围内生成 samples 个等间隔的时间点 t
    t = np.linspace(-np.pi, np.pi, samples)
    # 初始化一个空集合，用于存储已使用的图例标签
    used_legends: set[str] = set()

    # 调用函数获取标准颜色值，并根据类别数目和其他参数进行配置
    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )
    # 将类别与对应的颜色值进行映射
    colors = dict(zip(classes, color_values))
    # 如果未提供绘图轴对象 ax，则获取当前轴对象
    if ax is None:
        ax = plt.gca()
        # 设置绘图的 x 轴范围为 [-π, π]
        ax.set_xlim(-np.pi, np.pi)
    
    # 遍历数据帧中的每一行
    for i in range(n):
        # 获取数据帧中第 i 行的数据并转换为数组格式
        row = df.iloc[i].values
        # 使用行数据创建函数 f
        f = function(row)
        # 计算函数 f 在时间点 t 处的值
        y = f(t)
        # 获取当前行的类别标签
        kls = class_col.iat[i]
        # 将类别标签格式化为可打印的字符串形式
        label = pprint_thing(kls)
        # 如果该标签尚未被使用过，则将其加入已使用的图例标签集合中，并绘制带图例的曲线
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(t, y, color=colors[kls], label=label, **kwds)
        else:
            # 如果该标签已经被使用过，则绘制不带图例的曲线
            ax.plot(t, y, color=colors[kls], **kwds)

    # 添加图例到绘图对象的右上角位置
    ax.legend(loc="upper right")
    # 在绘图对象上显示网格
    ax.grid()
    # 返回绘图对象 ax
    return ax
def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds,
) -> Figure:
    import matplotlib.pyplot as plt

    # TODO: is the failure mentioned below still relevant?
    # random.sample(ndarray, int) fails on python 3.3, sigh
    # 将输入的 series 转换为列表形式
    data = list(series.values)
    # 对数据进行多次随机抽样，每次抽样大小为 size
    samplings = [random.sample(data, size) for _ in range(samples)]

    # 计算每次抽样的均值、中位数和中间范围
    means = np.array([np.mean(sampling) for sampling in samplings])
    medians = np.array([np.median(sampling) for sampling in samplings])
    midranges = np.array(
        [(min(sampling) + max(sampling)) * 0.5 for sampling in samplings]
    )
    
    # 如果未提供图形对象，则创建一个新的图形
    if fig is None:
        fig = plt.figure()
    # x 轴为样本编号
    x = list(range(samples))
    axes = []
    
    # 添加子图1：均值与样本关系
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlabel("Sample")
    axes.append(ax1)
    ax1.plot(x, means, **kwds)
    
    # 添加子图2：中位数与样本关系
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlabel("Sample")
    axes.append(ax2)
    ax2.plot(x, medians, **kwds)
    
    # 添加子图3：中间范围与样本关系
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlabel("Sample")
    axes.append(ax3)
    ax3.plot(x, midranges, **kwds)
    
    # 添加子图4：均值的直方图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_xlabel("Mean")
    axes.append(ax4)
    ax4.hist(means, **kwds)
    
    # 添加子图5：中位数的直方图
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_xlabel("Median")
    axes.append(ax5)
    ax5.hist(medians, **kwds)
    
    # 添加子图6：中间范围的直方图
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlabel("Midrange")
    axes.append(ax6)
    ax6.hist(midranges, **kwds)
    
    # 调整子图的标签字体大小
    for axis in axes:
        plt.setp(axis.get_xticklabels(), fontsize=8)
        plt.setp(axis.get_yticklabels(), fontsize=8)
    
    # 调整图形布局，使得子图紧凑排列
    if do_adjust_figure(fig):
        plt.tight_layout()
    
    # 返回绘制好的图形对象
    return fig


def parallel_coordinates(
    frame: DataFrame,
    class_column,
    cols=None,
    ax: Axes | None = None,
    color=None,
    use_columns: bool = False,
    xticks=None,
    colormap=None,
    axvlines: bool = True,
    axvlines_kwds=None,
    sort_labels: bool = False,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    # 如果未指定 axvlines_kwds，则使用默认参数设置
    if axvlines_kwds is None:
        axvlines_kwds = {"linewidth": 1, "color": "black"}

    n = len(frame)
    # 从数据帧中获取类别列的唯一值
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]

    # 根据传入的列名列表或默认使用所有非类别列构建数据子集 df
    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    # 初始化用于图例的集合
    used_legends: set[str] = set()

    ncols = len(df.columns)

    # 确定用于 x 轴刻度的值
    x: list[int] | Index
    if use_columns is True:
        # 如果使用列名作为 x 轴刻度，确保所有列名均为数值型
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError("Columns must be numeric to be used as xticks")
        x = df.columns
    elif xticks is not None:
        # 如果手动指定了 x 轴刻度，确保刻度值为数值型，并且数量与列数相同
        if not np.all(np.isreal(xticks)):
            raise ValueError("xticks specified must be numeric")
        if len(xticks) != ncols:
            raise ValueError("Length of xticks must match number of columns")
        x = xticks
    else:
        # 默认情况下，使用列索引作为 x 轴刻度
        x = list(range(ncols))

    # 如果未提供坐标轴对象，则使用当前的坐标轴
    if ax is None:
        ax = plt.gca()
    # 根据指定参数生成标准颜色值列表
    color_values = get_standard_colors(
        num_colors=len(classes),  # 使用类别数量确定颜色数量
        colormap=colormap,         # 使用指定的颜色映射
        color_type="random",       # 使用随机颜色类型
        color=color                # 使用指定的颜色
    )
    
    # 如果需要对类别进行排序，则按照字母顺序对类别和颜色值进行排序
    if sort_labels:
        classes = sorted(classes)          # 对类别按照字母顺序排序
        color_values = sorted(color_values)  # 对颜色值按照字母顺序排序
    
    # 将类别和对应的颜色值以字典形式进行关联
    colors = dict(zip(classes, color_values))
    
    # 遍历数据框中的每一行
    for i in range(n):
        y = df.iloc[i].values      # 获取第 i 行的数据作为 y 值
        kls = class_col.iat[i]     # 获取第 i 行的类别值
        label = pprint_thing(kls)  # 调用 pprint_thing 函数处理类别值，得到标签
        # 如果标签尚未被使用过，则添加到已使用标签集合中，并以指定颜色和标签绘制图像
        if label not in used_legends:
            used_legends.add(label)  # 将标签添加到已使用标签集合中
            ax.plot(x, y, color=colors[kls], label=label, **kwds)  # 绘制带有标签的线图
        else:
            ax.plot(x, y, color=colors[kls], **kwds)  # 否则，仅绘制线图，不显示标签
    
    # 如果需要绘制竖直线
    if axvlines:
        for i in x:
            ax.axvline(i, **axvlines_kwds)  # 在指定位置绘制竖直线，使用额外关键字参数
    
    # 设置 x 轴刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])  # 设置 x 轴显示范围为数据的第一个到最后一个值
    ax.legend(loc="upper right")  # 添加图例，并指定位置为右上角
    ax.grid()  # 添加网格线
    return ax  # 返回绘制好的轴对象
# 绘制 lag 图，显示时间序列 series 在 lag 时刻的散点图
def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
    # 导入 matplotlib.pyplot 库
    import matplotlib.pyplot as plt

    # 如果未指定颜色参数，则使用 matplotlib 默认的 patch.facecolor
    kwds.setdefault("c", plt.rcParams["patch.facecolor"])

    # 提取时间序列数据
    data = series.values
    # 计算 lag 时刻前后的数据
    y1 = data[:-lag]
    y2 = data[lag:]
    # 如果未提供绘图对象 ax，则获取当前的 Axes 对象
    if ax is None:
        ax = plt.gca()
    # 设置 x 轴标签
    ax.set_xlabel("y(t)")
    # 设置 y 轴标签，包括 lag 的信息
    ax.set_ylabel(f"y(t + {lag})")
    # 绘制散点图
    ax.scatter(y1, y2, **kwds)
    # 返回绘图对象
    return ax


# 绘制自相关图，显示时间序列 series 的自相关性
def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    # 导入 matplotlib.pyplot 库
    import matplotlib.pyplot as plt
    # 导入 numpy 库
    import numpy as np

    # 获取时间序列的长度
    n = len(series)
    # 将时间序列转换为 numpy 数组
    data = np.asarray(series)
    # 如果未提供绘图对象 ax，则获取当前的 Axes 对象，并设置 x、y 轴的范围
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    # 计算时间序列的均值
    mean = np.mean(data)
    # 计算 lag 为 0 时的自相关系数
    c0 = np.sum((data - mean) ** 2) / n

    # 定义函数 r(h)，计算 lag 为 h 时的自相关系数
    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    # 构建 lag 的序列
    x = np.arange(n) + 1
    # 计算所有 lag 对应的自相关系数
    y = [r(loc) for loc in x]
    # 设置置信区间的 Z 值
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    # 绘制置信区间的水平线，以及自相关系数曲线
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / np.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")
    # 设置 x、y 轴标签
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    # 绘制自相关系数曲线
    ax.plot(x, y, **kwds)
    # 如果提供了标签参数，则显示图例
    if "label" in kwds:
        ax.legend()
    # 显示网格线
    ax.grid()
    # 返回绘图对象
    return ax


# 如果输入的 keys 是长度为 1 的列表，则解包为其包含的单个字符串
def unpack_single_str_list(keys):
    # GH 42795：解决 GitHub issue 42795
    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]
    return keys
```