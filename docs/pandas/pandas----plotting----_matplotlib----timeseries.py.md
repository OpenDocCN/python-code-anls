# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\timeseries.py`

```
# TODO: Use the fact that axis can have units to simplify the process

from __future__ import annotations  # 允许在类型注释中使用类名作为字符串

import functools  # 提供高阶函数的工具，如部分应用函数
from typing import (  # 引入类型提示相关模块
    TYPE_CHECKING,  # 用于类型检查时避免循环导入
    Any,  # 表示任意类型
    cast,  # 类型转换工具
)
import warnings  # 警告模块

import numpy as np  # 导入NumPy库

from pandas._libs.tslibs import (  # 导入时间序列相关的底层库
    BaseOffset,  # 时间偏移基类
    Period,  # 表示时间段的类
    to_offset,  # 将对象转换为时间偏移
)
from pandas._libs.tslibs.dtypes import (  # 导入时间序列数据类型相关的底层库
    OFFSET_TO_PERIOD_FREQSTR,  # 偏移量到周期频率字符串的映射
    FreqGroup,  # 频率分组
)

from pandas.core.dtypes.generic import (  # 导入Pandas通用数据类型相关库
    ABCDatetimeIndex,  # Pandas日期时间索引的抽象基类
    ABCPeriodIndex,  # Pandas周期索引的抽象基类
    ABCTimedeltaIndex,  # Pandas时间差索引的抽象基类
)

from pandas.io.formats.printing import pprint_thing  # 用于打印任意对象的函数
from pandas.plotting._matplotlib.converter import (  # 导入Pandas与Matplotlib之间的转换器
    TimeSeries_DateFormatter,  # 时间序列日期格式化器
    TimeSeries_DateLocator,  # 时间序列日期定位器
    TimeSeries_TimedeltaFormatter,  # 时间序列时间差格式化器
)
from pandas.tseries.frequencies import (  # 导入时间序列频率相关库
    get_period_alias,  # 获取周期别名
    is_subperiod,  # 检查是否为子周期
    is_superperiod,  # 检查是否为父周期
)

if TYPE_CHECKING:
    from datetime import timedelta  # 导入时间差类型

    from matplotlib.axes import Axes  # 导入Matplotlib中的坐标轴类型

    from pandas._typing import NDFrameT  # Pandas中的通用数据框架类型

    from pandas import (  # 导入Pandas库的相关类型
        DataFrame,  # 数据帧类型
        DatetimeIndex,  # 日期时间索引类型
        Index,  # 索引类型
        PeriodIndex,  # 周期索引类型
        Series,  # 系列类型
    )

# ---------------------------------------------------------------------
# Plotting functions and monkey patches


def maybe_resample(series: Series, ax: Axes, kwargs: dict[str, Any]):
    # 如果需要，根据轴的频率重新采样序列

    if "how" in kwargs:
        raise ValueError(
            "'how' is not a valid keyword for plotting functions. If plotting "
            "multiple objects on shared axes, resample manually first."
        )

    freq, ax_freq = _get_freq(ax, series)  # 获取序列和轴的频率信息

    if freq is None:  # pragma: no cover
        raise ValueError("Cannot use dynamic axis without frequency info")

    # 将DatetimeIndex转换为PeriodIndex
    if isinstance(series.index, ABCDatetimeIndex):
        series = series.to_period(freq=freq)

    if ax_freq is not None and freq != ax_freq:
        if is_superperiod(freq, ax_freq):  # 如果序列频率是轴频率的父周期，则上采样输入
            series = series.copy()
            # error: "Index" has no attribute "asfreq"
            series.index = series.index.asfreq(  # type: ignore[attr-defined]
                ax_freq, how="s"
            )
            freq = ax_freq
        elif _is_sup(freq, ax_freq):  # 如果一个是周的周期
            # 由于使用PeriodDtype进行重采样已被弃用，因此我们转换为DatetimeIndex，进行重采样，然后再转换回来。
            ser_ts = series.to_timestamp()
            ser_d = ser_ts.resample("D").last().dropna()
            ser_freq = ser_d.resample(ax_freq).last().dropna()
            series = ser_freq.to_period(ax_freq)
            freq = ax_freq
        elif is_subperiod(freq, ax_freq) or _is_sub(freq, ax_freq):
            _upsample_others(ax, freq, kwargs)  # 对其他序列进行上采样
        else:  # pragma: no cover
            raise ValueError("Incompatible frequency conversion")
    return freq, series


def _is_sub(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_subperiod("D", f2)) or (
        f2.startswith("W") and is_subperiod(f1, "D")
    )
# 检查两个文件名是否符合“W”开头及超期要求的条件，返回布尔值
def _is_sup(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_superperiod("D", f2)) or (
        f2.startswith("W") and is_superperiod(f1, "D")
    )


# 在给定的Axes对象上重新绘制图形，并根据频率进行设置
def _upsample_others(ax: Axes, freq: BaseOffset, kwargs: dict[str, Any]) -> None:
    # 获取图例对象
    legend = ax.get_legend()
    # 调用_replot_ax函数重新绘制当前Axes对象，并获取返回的线和标签
    lines, labels = _replot_ax(ax, freq)
    # 再次调用_replot_ax函数以确保更新图形数据
    _replot_ax(ax, freq)

    other_ax = None
    # 检查是否存在左侧辅助轴
    if hasattr(ax, "left_ax"):
        other_ax = ax.left_ax
    # 检查是否存在右侧辅助轴
    if hasattr(ax, "right_ax"):
        other_ax = ax.right_ax

    # 如果存在其他辅助轴，则重新绘制其图形并扩展线和标签列表
    if other_ax is not None:
        rlines, rlabels = _replot_ax(other_ax, freq)
        lines.extend(rlines)
        labels.extend(rlabels)

    # 如果图例对象不为None，并且kwargs中设置为显示图例并且至少有一条线被绘制，则设置图例的标题并显示
    if legend is not None and kwargs.get("legend", True) and len(lines) > 0:
        title: str | None = legend.get_title().get_text()
        if title == "None":
            title = None
        ax.legend(lines, labels, loc="best", title=title)


# 在给定的Axes对象上重新绘制图形，并返回线和标签列表
def _replot_ax(ax: Axes, freq: BaseOffset):
    data = getattr(ax, "_plot_data", None)

    # 清空当前Axes对象的图形数据
    # TODO #54485
    ax._plot_data = []  # type: ignore[attr-defined]
    ax.clear()

    # 装饰Axes对象，设置频率相关属性
    decorate_axes(ax, freq)

    lines = []
    labels = []
    if data is not None:
        for series, plotf, kwds in data:
            series = series.copy()
            # 将时间序列数据的索引重新采样为指定频率
            idx = series.index.asfreq(freq, how="S")
            series.index = idx
            # TODO #54485
            ax._plot_data.append((series, plotf, kwds))  # type: ignore[attr-defined]

            # 对于tsplot，根据plotf字符串类型选择合适的绘图函数
            if isinstance(plotf, str):
                from pandas.plotting._matplotlib import PLOT_CLASSES
                plotf = PLOT_CLASSES[plotf]._plot

            # 绘制并获取线对象，并将标签添加到标签列表中
            lines.append(plotf(ax, series.index._mpl_repr(), series.values, **kwds)[0])
            labels.append(pprint_thing(series.name))

    return lines, labels


# 为Axes对象初始化时间序列绘图所需的属性
def decorate_axes(ax: Axes, freq: BaseOffset) -> None:
    if not hasattr(ax, "_plot_data"):
        # TODO #54485
        ax._plot_data = []  # type: ignore[attr-defined]

    # TODO #54485
    ax.freq = freq  # type: ignore[attr-defined]
    xaxis = ax.get_xaxis()
    # TODO #54485
    xaxis.freq = freq  # type: ignore[attr-defined]


# 获取Axes对象的频率属性
def _get_ax_freq(ax: Axes):
    """
    获取Axes对象的频率属性，如果未设置则检查共享轴（例如使用secondary yaxis时，sharex=True或twinx）
    """
    ax_freq = getattr(ax, "freq", None)
    if ax_freq is None:
        # 在使用辅助y轴时检查左右轴
        if hasattr(ax, "left_ax"):
            ax_freq = getattr(ax.left_ax, "freq", None)
        elif hasattr(ax, "right_ax"):
            ax_freq = getattr(ax.right_ax, "freq", None)
    # 如果 ax_freq 参数为 None，则执行以下逻辑
    # 检查是否有共享的 x 轴 (例如 sharex 或 twinx) 已经设置了频率
    shared_axes = ax.get_shared_x_axes().get_siblings(ax)
    # 如果找到了多个共享的轴
    if len(shared_axes) > 1:
        # 遍历每一个共享的轴
        for shared_ax in shared_axes:
            # 获取当前共享轴的 freq 属性，如果存在则赋值给 ax_freq
            ax_freq = getattr(shared_ax, "freq", None)
            # 如果找到了频率信息，就跳出循环
            if ax_freq is not None:
                break
    # 返回确定的 ax_freq 值，可能是 None 或者找到的共享轴的频率
    return ax_freq
def _get_period_alias(freq: timedelta | BaseOffset | str) -> str | None:
    if isinstance(freq, BaseOffset):
        # 如果频率是 BaseOffset 类型，则使用其名称作为频率字符串
        freqstr = freq.name
    else:
        # 否则，将频率转换为偏移量对象，并获取其规则代码作为频率字符串
        freqstr = to_offset(freq, is_period=True).rule_code

    # 返回频率字符串的别名
    return get_period_alias(freqstr)


def _get_freq(ax: Axes, series: Series):
    # 从数据中获取频率信息
    freq = getattr(series.index, "freq", None)
    if freq is None:
        # 如果没有明确的频率信息，则尝试获取推断的频率信息并转换为偏移量对象
        freq = getattr(series.index, "inferred_freq", None)
        freq = to_offset(freq, is_period=True)

    # 获取图表的频率信息
    ax_freq = _get_ax_freq(ax)

    # 如果数据中没有频率信息，则使用图表的频率信息
    if freq is None:
        freq = ax_freq

    # 获取周期频率的别名
    freq = _get_period_alias(freq)
    return freq, ax_freq


def use_dynamic_x(ax: Axes, data: DataFrame | Series) -> bool:
    # 获取数据索引的频率
    freq = _get_index_freq(data.index)
    ax_freq = _get_ax_freq(ax)

    # 如果数据索引没有频率信息，则尝试使用图表的频率信息
    if freq is None:
        # 如果图表没有频率信息，并且已经有绘制的线条，则返回 False
        if (ax_freq is None) and (len(ax.get_lines()) > 0):
            return False

    # 如果仍然没有获取到有效的频率信息，则返回 False
    if freq is None:
        return False

    # 获取频率的别名
    freq_str = _get_period_alias(freq)

    # 如果频率别名为 None，则返回 False
    if freq_str is None:
        return False

    # FIXME: hack this for 0.10.1, creating more technical debt...sigh
    if isinstance(data.index, ABCDatetimeIndex):
        # 如果数据索引是时间类型，则进行特定处理
        freq_str = OFFSET_TO_PERIOD_FREQSTR.get(freq_str, freq_str)
        # 获取频率对应的基础偏移量代码
        base = to_offset(freq_str, is_period=True)._period_dtype_code  # type: ignore[attr-defined]
        x = data.index
        if base <= FreqGroup.FR_DAY.value:
            # 检查时间戳是否已规范化
            return x[:1].is_normalized
        # 创建周期对象，并检查周期是否等于原始时间戳的本地化版本
        period = Period(x[0], freq_str)
        assert isinstance(period, Period)
        return period.to_timestamp().tz_localize(x.tz) == x[0]
    return True


def _get_index_freq(index: Index) -> BaseOffset | None:
    # 获取索引对象的频率信息
    freq = getattr(index, "freq", None)
    if freq is None:
        # 如果没有明确的频率信息，则尝试获取推断的频率信息
        freq = getattr(index, "inferred_freq", None)
        # 如果推断的频率为 "B"（工作日），则进一步检查是否包含周末
        if freq == "B":
            weekdays = np.unique(index.dayofweek)  # type: ignore[attr-defined]
            if (5 in weekdays) or (6 in weekdays):
                freq = None

    # 将频率转换为偏移量对象
    freq = to_offset(freq)
    return freq


def maybe_convert_index(ax: Axes, data: NDFrameT) -> NDFrameT:
    # tsplot 自动转换索引，但不希望对 DataFrames 反复转换索引
    # 此处略去不影响逻辑的代码注释
    # 检查数据的索引是否属于ABCDatetimeIndex或ABCPeriodIndex之一
    if isinstance(data.index, (ABCDatetimeIndex, ABCPeriodIndex)):
        # 获取数据索引的频率信息，可以是字符串、BaseOffset对象或None
        freq: str | BaseOffset | None = data.index.freq

        # 如果频率为None，说明数据索引为DatetimeIndex类型
        if freq is None:
            # 将数据索引强制转换为DatetimeIndex类型
            data.index = cast("DatetimeIndex", data.index)
            # 推断数据索引的频率
            freq = data.index.inferred_freq
            # 将推断得到的频率转换为BaseOffset对象
            freq = to_offset(freq)

        # 如果仍然无法获取频率信息，则调用_get_ax_freq函数获取频率
        if freq is None:
            freq = _get_ax_freq(ax)

        # 如果最终仍然没有有效的频率信息，则抛出值错误异常
        if freq is None:
            raise ValueError("Could not get frequency alias for plotting")

        # 根据频率获取相应的周期别名字符串
        freq_str = _get_period_alias(freq)

        # 忽略关于Period[B]的废弃警告，目前没有找到在废弃前的替代方法
        with warnings.catch_warnings():
            # 设置警告过滤器，忽略特定类别的FutureWarning警告
            warnings.filterwarnings(
                "ignore",
                r"PeriodDtype\[B\] is deprecated",
                category=FutureWarning,
            )

            # 如果数据索引是ABCDatetimeIndex类型，则转换为周期索引，使用给定的频率字符串
            if isinstance(data.index, ABCDatetimeIndex):
                data = data.tz_localize(None).to_period(freq=freq_str)
            # 如果数据索引是ABCPeriodIndex类型，则重新采样索引，使用给定的频率字符串和"start"方式
            elif isinstance(data.index, ABCPeriodIndex):
                data.index = data.index.asfreq(freq=freq_str, how="start")

    # 返回处理后的数据
    return data
# 定义一个函数用于格式化坐标信息，返回格式化后的字符串
def _format_coord(freq, t, y) -> str:
    # 创建一个 Period 对象，表示时间段，用于显示时间信息
    time_period = Period(ordinal=int(t), freq=freq)
    return f"t = {time_period}  y = {y:8f}"

# 定义一个函数，用于美化日期轴（x轴）
def format_dateaxis(
    subplot, freq: BaseOffset, index: DatetimeIndex | PeriodIndex
) -> None:
    """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the
    current underlying series.  As the dynamic mode is activated by
    default, changing the limits of the x axis will intelligently change
    the positions of the ticks.
    """
    import matplotlib.pyplot as plt

    # 处理不同类型的索引格式
    # 注意：DatetimeIndex 不使用这个接口，而是直接使用 matplotlib.date
    if isinstance(index, ABCPeriodIndex):
        # 设置主要定位器和次要定位器，根据频率设置，动态模式下自动调整位置
        majlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_locator(majlocator)
        subplot.xaxis.set_minor_locator(minlocator)

        # 设置主要格式化器和次要格式化器，根据频率设置，动态模式下自动调整格式
        majformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_formatter(majformatter)
        subplot.xaxis.set_minor_formatter(minformatter)

        # 设置坐标信息的格式化函数
        subplot.format_coord = functools.partial(_format_coord, freq)

    elif isinstance(index, ABCTimedeltaIndex):
        # 如果索引是时间间隔类型，使用专门的时间间隔格式化器
        subplot.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter())
    else:
        # 如果索引类型不支持，抛出类型错误异常
        raise TypeError("index type not supported")

    # 根据交互模式选择是否重新绘制图表
    plt.draw_if_interactive()
```