# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_datetimelike.py`

```
"""Test cases for time series specific (freq conversion, etc)"""

from datetime import (
    date,  # 导入 date 类
    datetime,  # 导入 datetime 类
    time,  # 导入 time 类
    timedelta,  # 导入 timedelta 类
)
import pickle  # 导入 pickle 库

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs import (
    BaseOffset,  # 导入 BaseOffset 类
    to_offset,  # 导入 to_offset 函数
)

from pandas.core.dtypes.dtypes import PeriodDtype  # 导入 PeriodDtype 类

from pandas import (
    DataFrame,  # 导入 DataFrame 类
    Index,  # 导入 Index 类
    NaT,  # 导入 NaT 常量
    Series,  # 导入 Series 类
    concat,  # 导入 concat 函数
    isna,  # 导入 isna 函数
    to_datetime,  # 导入 to_datetime 函数
)
import pandas._testing as tm  # 导入 pandas 测试模块
from pandas.core.indexes.datetimes import (
    DatetimeIndex,  # 导入 DatetimeIndex 类
    bdate_range,  # 导入 bdate_range 函数
    date_range,  # 导入 date_range 函数
)
from pandas.core.indexes.period import (
    Period,  # 导入 Period 类
    PeriodIndex,  # 导入 PeriodIndex 类
    period_range,  # 导入 period_range 函数
)
from pandas.core.indexes.timedeltas import timedelta_range  # 导入 timedelta_range 函数
from pandas.tests.plotting.common import _check_ticks_props  # 导入 _check_ticks_props 函数

from pandas.tseries.offsets import WeekOfMonth  # 导入 WeekOfMonth 类

mpl = pytest.importorskip("matplotlib")  # 导入并检查 matplotlib 库
plt = pytest.importorskip("matplotlib.pyplot")  # 导入并检查 matplotlib.pyplot 库

import pandas.plotting._matplotlib.converter as conv  # 导入 pandas.plotting._matplotlib.converter 模块


class TestTSPlot:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ts_plot_with_tz(self, tz_aware_fixture):
        # GH2877, GH17173, GH31205, GH31580
        tz = tz_aware_fixture  # 设置时区
        index = date_range("1/1/2011", periods=2, freq="h", tz=tz)  # 创建日期时间索引对象，频率为每小时
        ts = Series([188.5, 328.25], index=index)  # 创建时间序列对象
        _check_plot_works(ts.plot)  # 检查绘图功能正常
        ax = ts.plot()  # 绘制时间序列图，并返回 Axes 对象
        xdata = next(iter(ax.get_lines())).get_xdata()  # 获取第一条线的 X 轴数据
        # 检查第一个和最后一个点的标签是否正确
        assert (xdata[0].hour, xdata[0].minute) == (0, 0)
        assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)

    def test_fontsize_set_correctly(self):
        # For issue #8765
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)), index=range(10)
        )  # 创建包含随机数据的 DataFrame 对象，索引为 0 到 9
        _, ax = mpl.pyplot.subplots()  # 创建子图对象
        df.plot(fontsize=2, ax=ax)  # 在指定的 Axes 对象上绘制 DataFrame
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            assert label.get_fontsize() == 2  # 检查标签的字体大小是否为 2

    def test_frame_inferred(self):
        # inferred freq
        idx = date_range("1/1/1987", freq="MS", periods=10)  # 创建日期时间索引对象，频率为每月开始，共 10 个数据点
        idx = DatetimeIndex(idx.values, freq=None)  # 创建 DatetimeIndex 对象，频率设为 None

        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )  # 创建包含随机数据的 DataFrame 对象，索引为 idx
        _check_plot_works(df.plot)  # 检查绘图功能正常

        # axes freq
        idx = idx[0:4].union(idx[6:])  # 合并索引的前四个和后两个数据点
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )  # 创建包含随机数据的 DataFrame 对象，索引为 idx
        _check_plot_works(df2.plot)  # 检查绘图功能正常

    def test_frame_inferred_n_gt_1(self):
        # N > 1
        idx = date_range("2008-1-1 00:15:00", freq="15min", periods=10)  # 创建日期时间索引对象，频率为每 15 分钟，共 10 个数据点
        idx = DatetimeIndex(idx.values, freq=None)  # 创建 DatetimeIndex 对象，频率设为 None
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )  # 创建包含随机数据的 DataFrame 对象，索引为 idx
        _check_plot_works(df.plot)  # 检查绘图功能正常

    def test_is_error_nozeroindex(self):
        # GH11858
        i = np.array([1, 2, 3])  # 创建 NumPy 数组 i
        a = DataFrame(i, index=i)  # 创建包含数据 i 的 DataFrame 对象，索引为 i
        _check_plot_works(a.plot, xerr=a)  # 检查绘图功能正常，同时传入 x 轴误差数据
        _check_plot_works(a.plot, yerr=a)  # 检查绘图功能正常，同时传入 y 轴误差数据
    # 测试排除非数字列的情况
    def test_nonnumeric_exclude(self):
        # 创建日期范围索引
        idx = date_range("1/1/1987", freq="YE", periods=3)
        # 创建包含"A"和"B"列的数据帧
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)

        # 创建一个新的图形和轴对象
        fig, ax = mpl.pyplot.subplots()
        # 在给定的轴上绘制数据帧
        df.plot(ax=ax)  # 这行代码有效
        # 断言：检查图中的线条数是否为1，以确认"B"列被绘制了
        assert len(ax.get_lines()) == 1  # 绘制了"B"列

    # 测试排除非数字数据列时的错误情况
    def test_nonnumeric_exclude_error(self):
        # 创建日期范围索引
        idx = date_range("1/1/1987", freq="YE", periods=3)
        # 创建包含"A"和"B"列的数据帧
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)
        # 预期的错误消息
        msg = "no numeric data to plot"
        # 使用 pytest 检查是否会抛出指定类型的错误，并匹配特定消息
        with pytest.raises(TypeError, match=msg):
            # 尝试对非数值列"A"进行绘图操作
            df["A"].plot()

    # 使用不同频率参数来测试时间序列绘图
    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_tsplot_period(self, freq):
        # 创建周期范围索引
        idx = period_range("12/31/1999", freq=freq, periods=10)
        # 创建时间序列，使用随机生成的标准正态分布值
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 调用辅助函数检查绘图是否正常工作
        _check_plot_works(ser.plot, ax=ax)

    # 使用不同日期频率参数测试时间序列绘图
    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_tsplot_datetime(self, freq):
        # 创建日期范围索引
        idx = date_range("12/31/1999", freq=freq, periods=10)
        # 创建时间序列，使用随机生成的标准正态分布值
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 调用辅助函数检查绘图是否正常工作
        _check_plot_works(ser.plot, ax=ax)

    # 测试时间序列绘图
    def test_tsplot(self):
        # 创建具有浮点数值的时间序列，使用日期范围索引
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 在给定的轴上绘制时间序列，使用黑色样式
        ts.plot(style="k", ax=ax)
        # 断言：检查第一条线的颜色是否为黑色
        color = (0.0, 0.0, 0.0, 1)
        assert color == ax.get_lines()[0].get_color()

    # 测试同时设置样式和颜色时的错误情况
    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=10)])
    def test_both_style_and_color(self, index):
        # 创建具有浮点数值的时间序列，使用指定的索引
        ts = Series(np.arange(10, dtype=np.float64), index=index)
        # 预期的错误消息
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' "
            "keyword argument. Please use one or the other or pass 'style' "
            "without a color symbol"
        )
        # 使用 pytest 检查是否会抛出指定类型的错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            # 尝试使用同时设置样式和颜色来绘制时间序列
            ts.plot(style="b-", color="#000099")

    # 使用高频率参数测试时间序列绘图
    @pytest.mark.parametrize("freq", ["ms", "us"])
    def test_high_freq(self, freq):
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 创建日期范围索引，使用指定的频率和周期数
        rng = date_range("1/1/2012", periods=10, freq=freq)
        # 创建时间序列，使用随机生成的标准正态分布值
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        # 调用辅助函数检查绘图是否正常工作
        _check_plot_works(ser.plot, ax=ax)

    # 测试获取日期值的辅助函数
    def test_get_datevalue(self):
        # 断言：检查对于 None 类型的输入，函数返回 None
        assert conv.get_datevalue(None, "D") is None
        # 断言：检查对于整数输入，函数返回相同的整数值
        assert conv.get_datevalue(1987, "Y") == 1987
        # 断言：检查对于 Period 类型的输入，函数返回其 ordinal 值
        assert (
            conv.get_datevalue(Period(1987, "Y"), "M") == Period("1987-12", "M").ordinal
        )
        # 断言：检查对于日期字符串输入，函数返回对应的 ordinal 值
        assert conv.get_datevalue("1/1/1987", "D") == Period("1987-1-1", "D").ordinal

    # 使用参数化测试来测试时间频率和预期字符串
    @pytest.mark.parametrize(
        "freq, expected_string",
        [["YE-DEC", "t = 2014  y = 1.000000"], ["D", "t = 2014-01-01  y = 1.000000"]],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 标记：忽略特定警告，这里是针对期间数据类型的未来警告

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    # 使用 pytest 的 parametrize 标记，定义频率参数，用于多次运行测试函数

    def test_line_plot_datetime_series(self, freq):
        # 测试函数：测试日期时间序列的折线图
        idx = date_range("12/31/1999", freq=freq, periods=10)
        # 创建日期时间索引，使用给定的频率和周期数

        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        # 创建随机数值的时间序列，使用标准正态分布，长度与索引相同

        _check_plot_works(ser.plot, ser.index.freq.rule_code)
        # 调用辅助函数 _check_plot_works，测试时间序列的绘图功能，使用频率规则码

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "ME", "QE", "YE"])
    # 使用 pytest 的 parametrize 标记，定义频率参数，用于多次运行测试函数

    def test_line_plot_period_frame(self, freq):
        # 测试函数：测试周期数据框的折线图
        idx = date_range("12/31/1999", freq=freq, periods=10)
        # 创建日期时间索引，使用给定的频率和周期数

        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        # 创建具有随机数值的数据框，使用标准正态分布，行索引为日期时间索引，列为 A、B、C

        _check_plot_works(df.plot, df.index.freq)
        # 调用辅助函数 _check_plot_works，测试数据框的绘图功能，使用索引的频率

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    # 使用 pytest 的 parametrize 标记，定义频率参数，用于多次运行测试函数

    def test_line_plot_period_mlt_frame(self, frqncy):
        # 测试函数：测试带有倍数频率的周期数据框的折线图
        idx = period_range("12/31/1999", freq=frqncy, periods=10)
        # 创建周期索引，使用给定的倍数频率和周期数

        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        # 创建具有随机数值的数据框，使用标准正态分布，行索引为周期索引，列为 A、B、C

        freq = df.index.freq.rule_code
        # 获取数据框索引的频率规则码

        _check_plot_works(df.plot, freq)
        # 调用辅助函数 _check_plot_works，测试数据框的绘图功能，使用频率规则码

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    # 使用 pytest 的 parametrize 标记，定义频率参数，用于多次运行测试函数

    def test_line_plot_period_mlt_series(self, frqncy):
        # 测试函数：测试带有倍数频率的周期时间序列的折线图
        idx = period_range("12/31/1999", freq=frqncy, periods=10)
        # 创建周期索引，使用给定的倍数频率和周期数

        s = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        # 创建随机数值的时间序列，使用标准正态分布，长度与索引相同

        _check_plot_works(s.plot, s.index.freq.rule_code)
        # 调用辅助函数 _check_plot_works，测试时间序列的绘图功能，使用频率规则码

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    # 使用 pytest 的 parametrize 标记，定义频率参数，用于多次运行测试函数

    def test_ts_plot_format_coord(self, freq, expected_string):
        # 测试函数：测试时间序列绘图坐标格式化
        ser = Series(1, index=date_range("2014-01-01", periods=3, freq=freq))
        # 创建具有固定值的时间序列，使用给定的频率和周期数

        _, ax = mpl.pyplot.subplots()
        # 创建图形和轴对象

        ser.plot(ax=ax)
        # 在指定的轴上绘制时间序列

        first_line = ax.get_lines()[0]
        # 获取绘图对象的第一条线

        first_x = first_line.get_xdata()[0].ordinal
        # 获取第一条线的第一个点的 X 坐标，转换为日期的序数表示

        first_y = first_line.get_ydata()[0]
        # 获取第一条线的第一个点的 Y 坐标

        assert expected_string == ax.format_coord(first_x, first_y)
        # 断言：期望的坐标格式化字符串等于指定坐标点的格式化结果
    # 测试绘制线性图的函数，使用日期范围作为索引
    def test_line_plot_datetime_frame(self, freq):
        # 创建一个日期范围，从"12/31/1999"开始，频率为freq，包含10个时间点
        idx = date_range("12/31/1999", freq=freq, periods=10)
        # 创建一个DataFrame，包含随机生成的标准正态分布数据，索引为idx，列名为["A", "B", "C"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        # 获取DataFrame索引的频率字符串
        freq = PeriodDtype(df.index.freq)._freqstr
        # 将DataFrame的索引转换为周期对象，并获取其频率
        freq = df.index.to_period(freq).freq
        # 调用_check_plot_works函数，验证df.plot的绘图功能正常
        _check_plot_works(df.plot, freq)

    # 使用参数化测试，测试推断频率下的线性图绘制
    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_inferred_freq(self, freq):
        # 创建一个日期范围，从"12/31/1999"开始，频率为freq，包含10个时间点
        idx = date_range("12/31/1999", freq=freq, periods=10)
        # 创建一个Series，包含随机生成的标准正态分布数据，索引为idx
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        # 将Series的索引类型转换为Index对象
        ser = Series(ser.values, Index(np.asarray(ser.index)))
        # 调用_check_plot_works函数，验证ser.plot的绘图功能正常
        _check_plot_works(ser.plot, ser.index.inferred_freq)

        # 选择ser的部分数据点进行绘图测试
        ser = ser.iloc[[0, 3, 5, 6]]
        _check_plot_works(ser.plot)

    # 测试推断不是工作日的情况下的绘图
    def test_fake_inferred_business(self):
        # 创建一个matplotlib子图
        _, ax = mpl.pyplot.subplots()
        # 创建一个日期范围，从"2001-1-1"到"2001-1-10"
        rng = date_range("2001-1-1", "2001-1-10")
        # 创建一个Series，包含索引为rng的整数序列
        ts = Series(range(len(rng)), index=rng)
        # 将ts拆分为两部分并连接起来
        ts = concat([ts[:3], ts[5:]])
        # 在ax上绘制ts的图形
        ts.plot(ax=ax)
        # 验证ax对象没有属性"freq"
        assert not hasattr(ax, "freq")

    # 测试偏移频率的绘图
    def test_plot_offset_freq(self):
        # 创建一个Series，包含从"2020-01-01"开始的10个浮点数
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 调用_check_plot_works函数，验证ser.plot的绘图功能正常
        _check_plot_works(ser.plot)

    # 测试偏移频率为工作日的绘图
    def test_plot_offset_freq_business(self):
        # 创建一个日期范围，从"2023-01-01"开始，频率为工作季度开始日，包含10个时间点
        dr = date_range("2023-01-01", freq="BQS", periods=10)
        # 创建一个Series，包含随机生成的标准正态分布数据，索引为dr
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        # 调用_check_plot_works函数，验证ser.plot的绘图功能正常
        _check_plot_works(ser.plot)

    # 测试多个推断频率下的绘图
    def test_plot_multiple_inferred_freq(self):
        # 创建一个Index对象，包含三个datetime对象
        dr = Index([datetime(2000, 1, 1), datetime(2000, 1, 6), datetime(2000, 1, 11)])
        # 创建一个Series，包含随机生成的标准正态分布数据，索引为dr
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        # 调用_check_plot_works函数，验证ser.plot的绘图功能正常
        _check_plot_works(ser.plot)

    # 测试不规则高频率的绘图
    def test_irreg_hf(self):
        # 创建一个日期范围，从"2012-6-22 21:59:51"开始，频率为秒，包含10个时间点
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        # 创建一个DataFrame，包含随机生成的标准正态分布数据，索引为idx，包含两列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )

        # 选择df的部分数据点进行绘图测试
        irreg = df.iloc[[0, 1, 3, 4]]
        # 创建一个matplotlib子图
        _, ax = mpl.pyplot.subplots()
        # 在ax上绘制irreg的图形
        irreg.plot(ax=ax)
        # 获取图形中第一条线的x坐标数据，并计算相邻数据点的时间差
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()

        # 定义秒的时间差
        sec = 1.0 / 24 / 60 / 60
        # 断言相邻数据点的时间差与预期秒数非常接近
        assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-8).all()

    # 测试对象类型索引的不规则高频率绘图
    def test_irreg_hf_object(self):
        # 创建一个日期范围，从"2012-6-22 21:59:51"开始，频率为秒，包含10个时间点
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        # 创建一个DataFrame，包含随机生成的标准正态分布数据，索引为idx，包含两列
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )
        # 创建一个matplotlib子图
        _, ax = mpl.pyplot.subplots()
        # 将df2的索引类型转换为对象类型
        df2.index = df2.index.astype(object)
        # 在ax上绘制df2的图形
        df2.plot(ax=ax)
        # 获取图形中第一条线的x坐标数据，并计算相邻数据点的时间差
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        # 定义秒的时间差
        sec = 1.0 / 24 / 60 / 60
        # 断言相邻数据点的时间差与预期秒数非常接近
        assert (np.fabs(diffs[1:] - sec) < 1e-8).all()
    # 测试修复日期时间64位表示的不规则问题
    def test_irregular_datetime64_repr_bug(self):
        # 创建一个Series，包含从2020-01-01开始的10个浮点数，用日期索引
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 选择索引为0, 1, 2, 7的部分数据
        ser = ser.iloc[[0, 1, 2, 7]]

        # 创建一个新的图形和坐标轴
        _, ax = mpl.pyplot.subplots()

        # 绘制Series的数据到图形上，并返回绘制对象
        ret = ser.plot(ax=ax)
        # 断言绘制对象不为None
        assert ret is not None

        # 检查绘制的线条的数据点是否与Series的索引相匹配
        for rs, xp in zip(ax.get_lines()[0].get_xdata(), ser.index):
            assert rs == xp

    # 测试业务频率
    def test_business_freq(self):
        # 创建一个Series，包含0到4的整数，使用期间范围作为索引
        bts = Series(range(5), period_range("2020-01-01", periods=5))
        # 设置警告消息
        msg = r"PeriodDtype\[B\] is deprecated"
        # 将第一个索引转换为时间戳
        dt = bts.index[0].to_timestamp()
        # 使用上下文确保产生特定警告
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 将索引重新设置为以工作日为频率的期间范围
            bts.index = period_range(start=dt, periods=len(bts), freq="B")
        # 创建一个新的图形和坐标轴
        _, ax = mpl.pyplot.subplots()
        # 绘制Series的数据到图形上
        bts.plot(ax=ax)
        # 断言绘制的第一条线的x坐标数据的第一个点与Series的第一个索引的序数相等
        assert ax.get_lines()[0].get_xydata()[0, 0] == bts.index[0].ordinal
        # 获取绘制线条的x坐标数据
        idx = ax.get_lines()[0].get_xdata()
        # 使用上下文确保产生特定警告
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 断言PeriodIndex对象的频率字符串为"B"
            assert PeriodIndex(data=idx).freqstr == "B"

    # 测试业务频率转换
    def test_business_freq_convert(self):
        # 创建一个Series，包含从2020-01-01开始的50个浮点数，使用工作日频率
        bts = Series(
            np.arange(50, dtype=np.float64),
            index=date_range("2020-01-01", periods=50, freq="B"),
        ).asfreq("BME")
        # 将Series转换为月份的期间索引
        ts = bts.to_period("M")
        # 创建一个新的图形和坐标轴
        _, ax = mpl.pyplot.subplots()
        # 绘制Series的数据到图形上
        bts.plot(ax=ax)
        # 断言绘制的第一条线的x坐标数据的第一个点与期间索引的第一个序数相等
        assert ax.get_lines()[0].get_xydata()[0, 0] == ts.index[0].ordinal
        # 获取绘制线条的x坐标数据
        idx = ax.get_lines()[0].get_xdata()
        # 断言PeriodIndex对象的频率字符串为"M"
        assert PeriodIndex(data=idx).freqstr == "M"

    # 测试没有期间别名的频率
    def test_freq_with_no_period_alias(self):
        # GH34487
        # 创建一个周内的频率对象
        freq = WeekOfMonth()
        # 创建一个Series，包含从2020-01-01开始的10个浮点数，使用日期范围作为索引
        bts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        ).asfreq(freq)
        # 创建一个新的图形和坐标轴
        _, ax = mpl.pyplot.subplots()
        # 绘制Series的数据到图形上
        bts.plot(ax=ax)

        # 获取绘制线条的x坐标数据
        idx = ax.get_lines()[0].get_xdata()
        # 设置错误消息
        msg = "freq not specified and cannot be inferred"
        # 使用上下文确保引发特定异常
        with pytest.raises(ValueError, match=msg):
            # 创建一个期间索引对象
            PeriodIndex(data=idx)

    # 测试非零基数
    def test_nonzero_base(self):
        # GH2571
        # 创建一个日期范围，从2012-12-20开始，24个小时的频率，并加上30分钟
        idx = date_range("2012-12-20", periods=24, freq="h") + timedelta(minutes=30)
        # 创建一个DataFrame，包含0到23的整数，使用上述日期范围作为索引
        df = DataFrame(np.arange(24), index=idx)
        # 创建一个新的图形和坐标轴
        _, ax = mpl.pyplot.subplots()
        # 绘制DataFrame的数据到图形上
        df.plot(ax=ax)
        # 获取绘制线条的x坐标数据
        rs = ax.get_lines()[0].get_xdata()
        # 断言索引对象不是规范化的
        assert not Index(rs).is_normalized

    # 测试DataFrame
    def test_dataframe(self):
        # 创建一个DataFrame，包含一个名为"a"的Series，该Series包含从2020-01-01开始的10个浮点数
        bts = DataFrame(
            {
                "a": Series(
                    np.arange(10, dtype=np.float64),
                    index=date_range("2020-01-01", periods=10),
                )
            }
        )
        # 创建一个新的图形和坐标轴
        _, ax = mpl.pyplot.subplots()
        # 绘制DataFrame的数据到图形上
        bts.plot(ax=ax)
        # 获取绘制线条的x坐标数据
        idx = ax.get_lines()[0].get_xdata()
        # 断言索引对象的期间版本与期间索引相等
        tm.assert_index_equal(bts.index.to_period(), PeriodIndex(idx))

    # 使用pytest标记，忽略特定警告
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize(
        "obj",
        [
            Series(
                np.arange(10, dtype=np.float64),
                index=date_range("2020-01-01", periods=10),
            ),  # 创建一个包含10个浮点数的Series对象，以2020-01-01为起始日期的索引
            DataFrame(
                {
                    "a": Series(
                        np.arange(10, dtype=np.float64),
                        index=date_range("2020-01-01", periods=10),
                    ),  # 创建一个包含10个浮点数的Series对象，作为DataFrame的'a'列，索引与前面相同
                    "b": Series(
                        np.arange(10, dtype=np.float64),
                        index=date_range("2020-01-01", periods=10),
                    )
                    + 1,  # 创建一个包含10个浮点数的Series对象，作为DataFrame的'b'列，索引与前面相同，每个值加1
                }
            ),
        ],
    )
    def test_axis_limits(self, obj):
        _, ax = mpl.pyplot.subplots()  # 创建一个新的Figure和Axes对象
        obj.plot(ax=ax)  # 在Axes对象上绘制obj的图表
        xlim = ax.get_xlim()  # 获取当前Axes对象的x轴限制范围
        ax.set_xlim(xlim[0] - 5, xlim[1] + 10)  # 设置x轴限制范围为当前范围左移5，右移10
        result = ax.get_xlim()  # 再次获取设置后的x轴限制范围
        assert result[0] == xlim[0] - 5  # 断言设置后的左限与预期相符
        assert result[1] == xlim[1] + 10  # 断言设置后的右限与预期相符

        # string
        expected = (Period("1/1/2000", ax.freq), Period("4/1/2000", ax.freq))  # 设置期望的日期范围
        ax.set_xlim("1/1/2000", "4/1/2000")  # 设置x轴限制范围为指定日期字符串
        result = ax.get_xlim()  # 获取设置后的x轴限制范围
        assert int(result[0]) == expected[0].ordinal  # 断言左限的日期序数与预期相符
        assert int(result[1]) == expected[1].ordinal  # 断言右限的日期序数与预期相符

        # datetime
        expected = (Period("1/1/2000", ax.freq), Period("4/1/2000", ax.freq))  # 设置期望的日期范围
        ax.set_xlim(datetime(2000, 1, 1), datetime(2000, 4, 1))  # 设置x轴限制范围为指定的datetime对象
        result = ax.get_xlim()  # 获取设置后的x轴限制范围
        assert int(result[0]) == expected[0].ordinal  # 断言左限的日期序数与预期相符
        assert int(result[1]) == expected[1].ordinal  # 断言右限的日期序数与预期相符

    def test_get_finder(self):
        assert conv.get_finder(to_offset("B")) == conv._daily_finder  # 断言获取'B'频率对应的查找器
        assert conv.get_finder(to_offset("D")) == conv._daily_finder  # 断言获取'D'频率对应的查找器
        assert conv.get_finder(to_offset("ME")) == conv._monthly_finder  # 断言获取'ME'频率对应的查找器
        assert conv.get_finder(to_offset("QE")) == conv._quarterly_finder  # 断言获取'QE'频率对应的查找器
        assert conv.get_finder(to_offset("YE")) == conv._annual_finder  # 断言获取'YE'频率对应的查找器
        assert conv.get_finder(to_offset("W")) == conv._daily_finder  # 断言获取'W'频率对应的查找器

    def test_finder_daily(self):
        day_lst = [10, 40, 252, 400, 950, 2750, 10000]  # 创建一组日历天数列表

        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):  # 断言未来警告与消息匹配
            xpl1 = xpl2 = [Period("1999-1-1", freq="B").ordinal] * len(day_lst)  # 创建指定频率的日期序数列表
        rs1 = []
        rs2 = []
        for n in day_lst:
            rng = bdate_range("1999-1-1", periods=n)  # 创建指定天数的工作日范围
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)  # 创建指定索引的随机数Series对象
            _, ax = mpl.pyplot.subplots()  # 创建一个新的Figure和Axes对象
            ser.plot(ax=ax)  # 在Axes对象上绘制ser的图表
            xaxis = ax.get_xaxis()  # 获取Axes对象的x轴信息
            rs1.append(xaxis.get_majorticklocs()[0])  # 获取主要刻度位置并添加到rs1列表

            vmin, vmax = ax.get_xlim()  # 获取当前Axes对象的x轴限制范围的最小值和最大值
            ax.set_xlim(vmin + 0.9, vmax)  # 设置x轴限制范围为当前范围左移0.9
            rs2.append(xaxis.get_majorticklocs()[0])  # 获取更新后的主要刻度位置并添加到rs2列表
            mpl.pyplot.close(ax.get_figure())  # 关闭当前Axes对象所属的Figure

        assert rs1 == xpl1  # 断言rs1与预期结果xpl1相等
        assert rs2 == xpl2  # 断言rs2与预期结果xpl2相等
    # 定义一个测试函数，用于验证季度频率的数据查找器
    def test_finder_quarterly(self):
        # 设定一组年数
        yrs = [3.5, 11]

        # 初始化两个期望值列表，均为1988年第一季度的序数
        xpl1 = xpl2 = [Period("1988Q1").ordinal] * len(yrs)
        rs1 = []  # 存储第一个检查点结果的列表
        rs2 = []  # 存储第二个检查点结果的列表

        # 遍历年数列表
        for n in yrs:
            # 生成对应长度和频率的时间范围
            rng = period_range("1987Q2", periods=int(n * 4), freq="Q")
            # 生成随机数据序列
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            # 创建子图并绘制数据
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            # 获取子图的 x 轴对象
            xaxis = ax.get_xaxis()
            # 获取主刻度的位置并添加到第一个检查点结果列表中
            rs1.append(xaxis.get_majorticklocs()[0])

            # 获取当前子图 x 轴的限制范围
            (vmin, vmax) = ax.get_xlim()
            # 调整 x 轴的限制范围
            ax.set_xlim(vmin + 0.9, vmax)
            # 获取调整后的主刻度位置并添加到第二个检查点结果列表中
            rs2.append(xaxis.get_majorticklocs()[0])
            # 关闭当前子图
            mpl.pyplot.close(ax.get_figure())

        # 断言第一个检查点的结果等于期望值列表
        assert rs1 == xpl1
        # 断言第二个检查点的结果等于期望值列表
        assert rs2 == xpl2

    # 定义一个测试函数，用于验证月度频率的数据查找器
    def test_finder_monthly(self):
        # 设定一组年数
        yrs = [1.15, 2.5, 4, 11]

        # 初始化两个期望值列表，均为1988年1月的序数
        xpl1 = xpl2 = [Period("Jan 1988").ordinal] * len(yrs)
        rs1 = []  # 存储第一个检查点结果的列表
        rs2 = []  # 存储第二个检查点结果的列表

        # 遍历年数列表
        for n in yrs:
            # 生成对应长度和频率的时间范围
            rng = period_range("1987Q2", periods=int(n * 12), freq="M")
            # 生成随机数据序列
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            # 创建子图并绘制数据
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            # 获取子图的 x 轴对象
            xaxis = ax.get_xaxis()
            # 获取主刻度的位置并添加到第一个检查点结果列表中
            rs1.append(xaxis.get_majorticklocs()[0])

            # 获取当前子图 x 轴的限制范围
            vmin, vmax = ax.get_xlim()
            # 调整 x 轴的限制范围
            ax.set_xlim(vmin + 0.9, vmax)
            # 获取调整后的主刻度位置并添加到第二个检查点结果列表中
            rs2.append(xaxis.get_majorticklocs()[0])
            # 关闭当前子图
            mpl.pyplot.close(ax.get_figure())

        # 断言第一个检查点的结果等于期望值列表
        assert rs1 == xpl1
        # 断言第二个检查点的结果等于期望值列表
        assert rs2 == xpl2

    # 定义一个测试函数，用于验证长期月度频率的数据查找器
    def test_finder_monthly_long(self):
        # 生成时间范围，从1988年第一季度开始，连续24*12个月
        rng = period_range("1988Q1", periods=24 * 12, freq="M")
        # 生成随机数据序列
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        # 创建子图并绘制数据
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        # 获取子图的 x 轴对象
        xaxis = ax.get_xaxis()
        # 获取主刻度的位置
        rs = xaxis.get_majorticklocs()[0]
        # 期望值为1989年第一季度的序数，断言获取的主刻度位置等于期望值
        xp = Period("1989Q1", "M").ordinal
        assert rs == xp

    # 定义一个测试函数，用于验证年度频率的数据查找器
    def test_finder_annual(self):
        # 期望值列表，将一组年份转换为对应的序数
        xp = [1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170]
        xp = [Period(x, freq="Y").ordinal for x in xp]
        rs = []  # 存储检查点结果的列表

        # 遍历不同的年数
        for nyears in [5, 10, 19, 49, 99, 199, 599, 1001]:
            # 生成对应长度和频率的时间范围
            rng = period_range("1987", periods=nyears, freq="Y")
            # 生成随机数据序列
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            # 创建子图并绘制数据
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            # 获取子图的 x 轴对象
            xaxis = ax.get_xaxis()
            # 获取主刻度的位置并添加到检查点结果列表中
            rs.append(xaxis.get_majorticklocs()[0])
            # 关闭当前子图
            mpl.pyplot.close(ax.get_figure())

        # 断言检查点的结果列表等于期望值列表
        assert rs == xp

    # 使用 pytest 标记，定义一个慢速测试函数，用于验证分钟级频率的数据查找器
    @pytest.mark.slow
    def test_finder_minutely(self):
        # 设定总分钟数
        nminutes = 1 * 24 * 60
        # 生成分钟级的时间范围
        rng = date_range("1/1/1999", freq="Min", periods=nminutes)
        # 生成随机数据序列
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        # 创建子图并绘制数据
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        # 获取子图的 x 轴对象
        xaxis = ax.get_xaxis()
        # 获取主刻度的位置
        rs = xaxis.get_majorticklocs()[0]
        # 期望值为1999年1月1日的序数，断言获取的主刻度位置等于期望值
        xp = Period("1/1/1999", freq="Min").ordinal

        assert rs == xp
    def test_finder_hourly(self):
        nhours = 23  # 设置小时数为23
        rng = date_range("1/1/1999", freq="h", periods=nhours)  # 生成一个日期时间索引，频率为每小时，包含23个时间点
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)  # 创建一个时间序列，使用标准正态分布随机数填充，索引为rng
        _, ax = mpl.pyplot.subplots()  # 创建一个新的图形和一个子图轴对象ax
        ser.plot(ax=ax)  # 在ax上绘制时间序列ser的图形
        xaxis = ax.get_xaxis()  # 获取ax的x轴对象
        rs = xaxis.get_majorticklocs()[0]  # 获取x轴的主要刻度位置的第一个值
        xp = Period("1/1/1999", freq="h").ordinal  # 获取日期时间"1/1/1999"对应的序数表示

        assert rs == xp  # 断言rs与xp相等

    def test_gaps(self):
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )  # 创建一个包含10个浮点数的时间序列，索引为从"2020-01-01"开始的10个时间点
        ts.iloc[5:7] = np.nan  # 将索引为5和6的位置设为NaN（缺失值）
        _, ax = mpl.pyplot.subplots()  # 创建一个新的图形和一个子图轴对象ax
        ts.plot(ax=ax)  # 在ax上绘制时间序列ts的图形
        lines = ax.get_lines()  # 获取ax上的所有线条对象
        assert len(lines) == 1  # 断言线条数量为1
        line = lines[0]  # 获取第一个线条对象
        data = line.get_xydata()  # 获取线条对象的数据点坐标

        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)  # 创建一个掩码数组，将缺失值设为NaN

        assert isinstance(data, np.ma.core.MaskedArray)  # 断言data是MaskedArray类型
        mask = data.mask  # 获取掩码
        assert mask[5:7, 1].all()  # 断言第6到第7行、第2列的掩码值全部为True

    def test_gaps_irregular(self):
        # irregular
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )  # 创建一个包含30个浮点数的时间序列，索引为从"2020-01-01"开始的30个时间点
        ts = ts.iloc[[0, 1, 2, 5, 7, 9, 12, 15, 20]]  # 根据索引位置选择部分数据，重置时间序列ts
        ts.iloc[2:5] = np.nan  # 将索引为2到4的位置设为NaN（缺失值）
        _, ax = mpl.pyplot.subplots()  # 创建一个新的图形和一个子图轴对象ax
        ax = ts.plot(ax=ax)  # 在ax上绘制时间序列ts的图形
        lines = ax.get_lines()  # 获取ax上的所有线条对象
        assert len(lines) == 1  # 断言线条数量为1
        line = lines[0]  # 获取第一个线条对象
        data = line.get_xydata()  # 获取线条对象的数据点坐标

        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)  # 创建一个掩码数组，将缺失值设为NaN

        assert isinstance(data, np.ma.core.MaskedArray)  # 断言data是MaskedArray类型
        mask = data.mask  # 获取掩码
        assert mask[2:5, 1].all()  # 断言第3到第5行、第2列的掩码值全部为True

    def test_gaps_non_ts(self):
        # non-ts
        idx = [0, 1, 2, 5, 7, 9, 12, 15, 20]  # 创建一个索引列表
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)  # 创建一个时间序列，使用标准正态分布随机数填充，索引为idx
        ser.iloc[2:5] = np.nan  # 将索引为2到4的位置设为NaN（缺失值）
        _, ax = mpl.pyplot.subplots()  # 创建一个新的图形和一个子图轴对象ax
        ser.plot(ax=ax)  # 在ax上绘制时间序列ser的图形
        lines = ax.get_lines()  # 获取ax上的所有线条对象
        assert len(lines) == 1  # 断言线条数量为1
        line = lines[0]  # 获取第一个线条对象
        data = line.get_xydata()  # 获取线条对象的数据点坐标
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)  # 创建一个掩码数组，将缺失值设为NaN

        assert isinstance(data, np.ma.core.MaskedArray)  # 断言data是MaskedArray类型
        mask = data.mask  # 获取掩码
        assert mask[2:5, 1].all()  # 断言第3到第5行、第2列的掩码值全部为True

    def test_gap_upsample(self):
        low = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )  # 创建一个包含10个浮点数的时间序列，索引为从"2020-01-01"开始的10个时间点
        low.iloc[5:7] = np.nan  # 将索引为5和6的位置设为NaN（缺失值）
        _, ax = mpl.pyplot.subplots()  # 创建一个新的图形和一个子图轴对象ax
        low.plot(ax=ax)  # 在ax上绘制时间序列low的图形

        idxh = date_range(low.index[0], low.index[-1], freq="12h")  # 生成一个新的日期时间索引，频率为每12小时
        s = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)  # 创建一个时间序列，使用标准正态分布随机数填充，索引为idxh
        s.plot(secondary_y=True)  # 在右轴上绘制时间序列s的图形
        lines = ax.get_lines()  # 获取ax上的所有线条对象
        assert len(lines) == 1  # 断言ax上线条数量为1
        assert len(ax.right_ax.get_lines()) == 1  # 断言右轴上线条数量为1

        line = lines[0]  # 获取第一个线条对象
        data = line.get_xydata()  # 获取线条对象的数据点坐标
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)  # 创建一个掩码数组，将缺失值设为NaN

        assert isinstance(data, np.ma.core.MaskedArray)  # 断言data是MaskedArray类型
        mask = data.mask  # 获取掩码
        assert mask[5:7, 1].all()  # 断言第6到第7行、第2列的掩码值全部为True
    def test_secondary_y(self):
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象
        ser = Series(np.random.default_rng(2).standard_normal(10))
        # 创建一个新的图形对象和子图对象
        fig, _ = mpl.pyplot.subplots()
        # 在子图对象上绘制 ser 对象，设置 secondary_y=True 以显示次要 Y 轴
        ax = ser.plot(secondary_y=True)
        # 断言 ax 对象具有属性 "left_ax"
        assert hasattr(ax, "left_ax")
        # 断言 ax 对象不具有属性 "right_ax"
        assert not hasattr(ax, "right_ax")
        # 获取图形对象的所有子图对象
        axes = fig.get_axes()
        # 获取 ax 对象上的第一个线条对象
        line = ax.get_lines()[0]
        # 根据线条对象的数据创建一个新的 Series 对象 xp
        xp = Series(line.get_ydata(), line.get_xdata())
        # 断言 ser 和 xp 对象的内容相等
        tm.assert_series_equal(ser, xp)
        # 断言 ax 对象的 Y 轴刻度位置为 "right"
        assert ax.get_yaxis().get_ticks_position() == "right"
        # 断言 axes[0] 对象的 Y 轴不可见
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_yaxis(self):
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象，但未使用该对象
        Series(np.random.default_rng(2).standard_normal(10))
        # 创建一个新的图形对象和子图对象 ax2
        ser2 = Series(np.random.default_rng(2).standard_normal(10))
        _, ax2 = mpl.pyplot.subplots()
        # 在 ax2 对象上绘制 ser2 对象
        ser2.plot(ax=ax2)
        # 断言 ax2 对象的 Y 轴刻度位置为 "left"
        assert ax2.get_yaxis().get_ticks_position() == "left"

    def test_secondary_both(self):
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象 ser
        ser = Series(np.random.default_rng(2).standard_normal(10))
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象 ser2
        ser2 = Series(np.random.default_rng(2).standard_normal(10))
        # 在 ser2 对象上创建一个新的子图对象 ax
        ax = ser2.plot()
        # 在 ser 对象上创建一个新的子图对象 ax2，设置 secondary_y=True 以显示次要 Y 轴
        ax2 = ser.plot(secondary_y=True)
        # 断言 ax 对象的 Y 轴可见
        assert ax.get_yaxis().get_visible()
        # 断言 ax 对象不具有属性 "left_ax"
        assert not hasattr(ax, "left_ax")
        # 断言 ax 对象具有属性 "right_ax"
        assert hasattr(ax, "right_ax")
        # 断言 ax2 对象具有属性 "left_ax"
        assert hasattr(ax2, "left_ax")
        # 断言 ax2 对象不具有属性 "right_ax"
        assert not hasattr(ax2, "right_ax")

    def test_secondary_y_ts(self):
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象 ser，并使用日期范围作为索引 idx
        idx = date_range("1/1/2000", periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(10), idx)
        # 创建一个新的图形对象和子图对象
        fig, _ = mpl.pyplot.subplots()
        # 在子图对象上绘制 ser 对象，设置 secondary_y=True 以显示次要 Y 轴
        ax = ser.plot(secondary_y=True)
        # 断言 ax 对象具有属性 "left_ax"
        assert hasattr(ax, "left_ax")
        # 断言 ax 对象不具有属性 "right_ax"
        assert not hasattr(ax, "right_ax")
        # 获取图形对象的所有子图对象
        axes = fig.get_axes()
        # 获取 ax 对象上的第一个线条对象
        line = ax.get_lines()[0]
        # 根据线条对象的数据创建一个新的 Series 对象 xp，并转换为时间戳
        xp = Series(line.get_ydata(), line.get_xdata()).to_timestamp()
        # 断言 ser 和 xp 对象的内容相等
        tm.assert_series_equal(ser, xp)
        # 断言 ax 对象的 Y 轴刻度位置为 "right"
        assert ax.get_yaxis().get_ticks_position() == "right"
        # 断言 axes[0] 对象的 Y 轴不可见
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_ts_yaxis(self):
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象 ser2，并使用日期范围作为索引 idx
        idx = date_range("1/1/2000", periods=10)
        ser2 = Series(np.random.default_rng(2).standard_normal(10), idx)
        # 创建一个新的图形对象和子图对象 ax2
        _, ax2 = mpl.pyplot.subplots()
        # 在 ax2 对象上绘制 ser2 对象
        ser2.plot(ax=ax2)
        # 断言 ax2 对象的 Y 轴刻度位置为 "left"
        assert ax2.get_yaxis().get_ticks_position() == "left"

    def test_secondary_y_ts_visible(self):
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象 ser2
        idx = date_range("1/1/2000", periods=10)
        ser2 = Series(np.random.default_rng(2).standard_normal(10), idx)
        # 在 ser2 对象上创建一个新的子图对象 ax
        ax = ser2.plot()
        # 断言 ax 对象的 Y 轴可见
        assert ax.get_yaxis().get_visible()

    def test_secondary_kde(self):
        # 如果没有安装 scipy，则跳过当前测试
        pytest.importorskip("scipy")
        # 创建一个包含 10 个随机标准正态分布值的 Series 对象 ser
        ser = Series(np.random.default_rng(2).standard_normal(10))
        # 创建一个新的图形对象 fig 和子图对象 ax
        fig, ax = mpl.pyplot.subplots()
        # 在 ax 对象上绘制 ser 对象的密度图（核密度估计），设置 secondary_y=True 以显示次要 Y 轴
        ax = ser.plot(secondary_y=True, kind="density", ax=ax)
        # 断言 ax 对象具有属性 "left_ax"
        assert hasattr(ax, "left_ax")
        # 断言 ax 对象不具有属性 "right_ax"
        assert not hasattr(ax, "right_ax")
        # 获取图形对象的所有子图对象
        axes = fig.get_axes()
        # 断言 axes[1] 对象的 Y 轴刻度位置为 "right"
        assert axes[1].get_yaxis().get_ticks_position() == "right"
    # 测试在绘图中使用次要 Y 轴的功能
    def test_secondary_bar(self):
        # 创建包含随机数据的 Series 对象
        ser = Series(np.random.default_rng(2).standard_normal(10))
        # 创建新的图形和坐标轴对象
        fig, ax = mpl.pyplot.subplots()
        # 绘制条形图，并将其放在次要 Y 轴上
        ser.plot(secondary_y=True, kind="bar", ax=ax)
        # 获取图形对象中的所有坐标轴
        axes = fig.get_axes()
        # 断言第二个坐标轴的 Y 轴刻度位置是否在右侧
        assert axes[1].get_yaxis().get_ticks_position() == "right"

    # 测试在 DataFrame 中使用次要 Y 轴的功能
    def test_secondary_frame(self):
        # 创建包含随机数据的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["a", "b", "c"]
        )
        # 绘制 DataFrame 中指定列的折线图，并生成子图
        axes = df.plot(secondary_y=["a", "c"], subplots=True)
        # 断言每个子图中 Y 轴刻度位置是否正确
        assert axes[0].get_yaxis().get_ticks_position() == "right"
        assert axes[1].get_yaxis().get_ticks_position() == "left"
        assert axes[2].get_yaxis().get_ticks_position() == "right"

    # 测试在 DataFrame 中使用次要 Y 轴绘制条形图的功能
    def test_secondary_bar_frame(self):
        # 创建包含随机数据的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["a", "b", "c"]
        )
        # 绘制 DataFrame 中指定列的条形图，并生成子图
        axes = df.plot(kind="bar", secondary_y=["a", "c"], subplots=True)
        # 断言每个子图中 Y 轴刻度位置是否正确
        assert axes[0].get_yaxis().get_ticks_position() == "right"
        assert axes[1].get_yaxis().get_ticks_position() == "left"
        assert axes[2].get_yaxis().get_ticks_position() == "right"

    # 测试不同频率的时间序列数据的绘制功能（定期频率在前）
    def test_mixed_freq_regular_first(self):
        # 创建包含定期频率时间索引的 Series 对象
        s1 = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, freq="B"),
        )
        # 根据索引选择部分数据，创建新的 Series 对象
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]

        # 创建新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制 s1 的折线图
        s1.plot(ax=ax)

        # 在同一坐标轴上绘制 s2 的折线图，并获取折线对象
        ax2 = s2.plot(style="g", ax=ax)
        lines = ax2.get_lines()

        # 断言警告是否被正确触发
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 获取折线对象的 X 轴数据并转换为 PeriodIndex 对象
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())

            # 断言折线对象的 X 轴数据与原始 Series 的时间索引相匹配
            tm.assert_index_equal(idx1, s1.index.to_period("B"))
            tm.assert_index_equal(idx2, s2.index.to_period("B"))

            # 获取当前坐标轴的 X 轴限制，并验证与 s1 的时间索引一致
            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    # 测试不同频率的时间序列数据的绘制功能（非定期频率在前）
    def test_mixed_freq_irregular_first(self):
        # 创建包含非定期频率时间索引的 Series 对象
        s1 = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        # 根据索引选择部分数据，创建新的 Series 对象
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]

        # 创建新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制 s2 的折线图
        s2.plot(style="g", ax=ax)
        # 绘制 s1 的折线图（覆盖在 s2 上）
        s1.plot(ax=ax)

        # 断言坐标轴对象不包含属性 "freq"
        assert not hasattr(ax, "freq")

        # 获取折线对象并验证其 X 轴数据与时间索引相匹配
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)
    def test_mixed_freq_regular_first_df(self):
        # GH 9852
        # 创建一个 Series 对象 s1，其中包含从 0 到 19 的浮点数，索引为工作日频率的日期范围
        s1 = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, freq="B"),
        ).to_frame()
        # 从 s1 中选择特定行组成 s2
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        # 创建一个新的图表和轴对象 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制 s1 的图表
        s1.plot(ax=ax)
        # 在 ax 上绘制 s2 的图表，并设定样式为绿色
        ax2 = s2.plot(style="g", ax=ax)
        # 获取 ax2 上的所有线条对象
        lines = ax2.get_lines()
        # 设置警告消息内容
        msg = r"PeriodDtype\[B\] is deprecated"
        # 断言会产生 FutureWarning 警告，并匹配给定的消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 创建 PeriodIndex 对象 idx1 和 idx2，分别使用 ax2 上两条线的 x 轴数据
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())
            # 断言 idx1 和 s1 的索引转换为工作日频率的 PeriodIndex 相等
            assert idx1.equals(s1.index.to_period("B"))
            # 断言 idx2 和 s2 的索引转换为工作日频率的 PeriodIndex 相等
            assert idx2.equals(s2.index.to_period("B"))
            # 获取 ax2 的 x 轴限制范围的左右边界
            left, right = ax2.get_xlim()
            # 将 s1 的索引转换为 PeriodIndex 类型，并赋值给 pidx
            pidx = s1.index.to_period()
        # 断言左边界小于等于 pidx 的第一个元素的 ordinal 值
        assert left <= pidx[0].ordinal
        # 断言右边界大于等于 pidx 的最后一个元素的 ordinal 值

        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first_df(self):
        # GH 9852
        # 创建一个 Series 对象 s1，其中包含从 0 到 19 的浮点数，索引为日期范围的默认频率
        s1 = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        ).to_frame()
        # 从 s1 中选择特定行组成 s2
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        # 创建一个新的图表和轴对象 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制 s2 的图表，并设定样式为绿色
        s2.plot(style="g", ax=ax)
        # 在 ax 上绘制 s1 的图表
        s1.plot(ax=ax)
        # 断言 ax 没有 freq 属性
        assert not hasattr(ax, "freq")
        # 获取 ax 上的所有线条对象
        lines = ax.get_lines()
        # 获取第一条线条的 x 轴数据 x1
        x1 = lines[0].get_xdata()
        # 断言 x1 与 s2 的索引转换为对象类型后的值相等
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        # 获取第二条线条的 x 轴数据 x2
        x2 = lines[1].get_xdata()
        # 断言 x2 与 s1 的索引转换为对象类型后的值相等
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_hf_first(self):
        # 创建一个高频率的日期范围 idxh 和低频率的日期范围 idxl
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        # 使用标准正态分布生成随机数填充高频率和低频率的 Series 对象 high 和 low
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # 创建一个新的图表和轴对象 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制 high 的图表
        high.plot(ax=ax)
        # 在 ax 上绘制 low 的图表
        low.plot(ax=ax)
        # 遍历 ax 上的每条线条对象
        for line in ax.get_lines():
            # 断言每条线条的 x 轴数据为 PeriodIndex 对象，其频率为 "D"
            assert PeriodIndex(data=line.get_xdata()).freq == "D"

    def test_mixed_freq_alignment(self):
        # 创建一个小时频率的时间序列 ts_ind 和对应的随机数据 ts_data
        ts_ind = date_range("2012-01-01 13:00", "2012-01-02", freq="h")
        ts_data = np.random.default_rng(2).standard_normal(12)

        # 创建 Series 对象 ts 和 ts2，分别使用 ts_data 和 ts_ind
        ts = Series(ts_data, index=ts_ind)
        ts2 = ts.asfreq("min").interpolate()

        # 创建一个新的图表和轴对象 ax，并在 ax 上绘制 ts 和 ts2 的图表
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax)
        ts2.plot(style="r", ax=ax)

        # 断言 ax 上的第一条线条和第二条线条的第一个 x 轴数据相等
        assert ax.lines[0].get_xdata()[0] == ax.lines[1].get_xdata()[0]
    def test_mixed_freq_lf_first(self):
        # 创建一个日期范围，从1999年1月1日开始，365天，频率为每天一次
        idxh = date_range("1/1/1999", periods=365, freq="D")
        # 创建另一个日期范围，从1999年1月1日开始，12个月，频率为每月结束
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        # 使用随机生成的标准正态分布数据创建一个时间序列，数据长度与idxh相同，索引为idxh
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 使用随机生成的标准正态分布数据创建一个时间序列，数据长度与idxl相同，索引为idxl
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # 创建一个新的图形和一个子图对象
        _, ax = mpl.pyplot.subplots()
        # 在子图ax上绘制低频时间序列，显示图例
        low.plot(legend=True, ax=ax)
        # 在同一子图ax上绘制高频时间序列，添加到已有图例中
        high.plot(legend=True, ax=ax)
        # 遍历图中所有线条
        for line in ax.get_lines():
            # 断言每条线的x数据的索引频率为每日一次
            assert PeriodIndex(data=line.get_xdata()).freq == "D"
        # 获取图例对象
        leg = ax.get_legend()
        # 断言图例文本数量为2
        assert len(leg.texts) == 2
        # 关闭当前图形
        mpl.pyplot.close(ax.get_figure())

    def test_mixed_freq_lf_first_hourly(self):
        # 创建一个日期范围，从1999年1月1日开始，240个时间点，频率为每分钟
        idxh = date_range("1/1/1999", periods=240, freq="min")
        # 创建另一个日期范围，从1999年1月1日开始，4个时间点，频率为每小时
        idxl = date_range("1/1/1999", periods=4, freq="h")
        # 使用随机生成的标准正态分布数据创建一个时间序列，数据长度与idxh相同，索引为idxh
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 使用随机生成的标准正态分布数据创建一个时间序列，数据长度与idxl相同，索引为idxl
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # 创建一个新的图形和一个子图对象
        _, ax = mpl.pyplot.subplots()
        # 在子图ax上绘制低频时间序列
        low.plot(ax=ax)
        # 在同一子图ax上绘制高频时间序列
        high.plot(ax=ax)
        # 遍历图中所有线条
        for line in ax.get_lines():
            # 断言每条线的x数据的索引频率为每分钟
            assert PeriodIndex(data=line.get_xdata()).freq == "min"

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_mixed_freq_irreg_period(self):
        # 创建一个包含30个浮点数的时间序列，从2020年1月1日开始，共30天
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        # 从ts中选择不规则的子集
        irreg = ts.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 29]]
        # 设置警告消息内容
        msg = r"PeriodDtype\[B\] is deprecated"
        # 使用期间范围函数创建一个时间范围，从2000年1月3日开始，共30个时间点，频率为工作日
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rng = period_range("1/3/2000", periods=30, freq="B")
        # 使用随机生成的标准正态分布数据创建一个时间序列，数据长度与rng相同，索引为rng
        ps = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        # 创建一个新的图形和一个子图对象
        _, ax = mpl.pyplot.subplots()
        # 在子图ax上绘制不规则时间序列
        irreg.plot(ax=ax)
        # 在同一子图ax上绘制ps时间序列
        ps.plot(ax=ax)

    def test_mixed_freq_shared_ax(self):
        # GH13341, 使用sharex=True创建两个子图对象
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        # 使用idx1创建一个时间序列，索引为idx1
        s1 = Series(range(len(idx1)), idx1)
        # 使用idx2创建一个时间序列，索引为idx2
        s2 = Series(range(len(idx2)), idx2)

        # 创建一个新的图形和两个共享x轴的子图对象
        _, (ax1, ax2) = mpl.pyplot.subplots(nrows=2, sharex=True)
        # 在第一个子图ax1上绘制s1时间序列
        s1.plot(ax=ax1)
        # 在第二个子图ax2上绘制s2时间序列
        s2.plot(ax=ax2)

        # 断言第一个子图ax1的频率为每月一次
        assert ax1.freq == "M"
        # 断言第二个子图ax2的频率为每月一次
        assert ax2.freq == "M"
        # 断言两个子图第一条线的第一个x数据相等
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_mixed_freq_shared_ax_twin_x(self):
        # GH13341, 使用sharex=True创建一个子图对象和其副本
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        # 使用idx1创建一个时间序列，索引为idx1
        s1 = Series(range(len(idx1)), idx1)
        # 使用idx2创建一个时间序列，索引为idx2
        s2 = Series(range(len(idx2)), idx2)
        # 使用twinx方法创建一个与ax1共享x轴的新的子图对象ax2
        _, ax1 = mpl.pyplot.subplots()
        ax2 = ax1.twinx()
        # 在第一个子图ax1上绘制s1时间序列
        s1.plot(ax=ax1)
        # 在第二个子图ax2上绘制s2时间序列
        s2.plot(ax=ax2)

        # 断言两个子图第一条线的第一个x数据相等
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    @pytest.mark.xfail(reason="TODO (GH14330, GH14322)")
    # 测试函数，验证在共享 X 轴且第一个数据集频率不规则的情况下的行为
    def test_mixed_freq_shared_ax_twin_x_irregular_first(self):
        # GH13341, 使用 sharex=True
        # 创建一个时间索引，包含三个月末频率的日期范围
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        # 创建另一个索引，从第一个索引中选择第一个和最后一个日期
        idx2 = idx1[:1].union(idx1[2:])
        # 创建两个 Series 对象，分别用 idx1 和 idx2 作为索引
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)
        # 创建一个新的图形和一个子图 ax1
        _, ax1 = mpl.pyplot.subplots()
        # 创建一个与 ax1 共享 X 轴的子图 ax2
        ax2 = ax1.twinx()
        # 在 ax1 上绘制 s2 的数据
        s2.plot(ax=ax1)
        # 在 ax2 上绘制 s1 的数据
        s1.plot(ax=ax2)
        # 断言两条线的第一个数据点的 X 坐标相等
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    # 测试函数，验证处理 NaT（Not a Time）时间戳的行为
    def test_nat_handling(self):
        # 创建一个新的图形和子图 ax
        _, ax = mpl.pyplot.subplots()
        # 创建一个包含 NaT 的 DatetimeIndex
        dti = DatetimeIndex(["2015-01-01", NaT, "2015-01-03"])
        # 创建一个 Series 对象，使用 dti 作为索引
        s = Series(range(len(dti)), dti)
        # 在 ax 上绘制 s 的数据
        s.plot(ax=ax)
        # 获取第一条线的 X 数据
        xdata = ax.get_lines()[0].get_xdata()
        # 断言绘制的 X 数据在索引值的范围内
        assert s.index.min() <= Series(xdata).min()
        assert Series(xdata).max() <= s.index.max()

    # 测试函数，验证周重采样时不允许使用 how 关键字的行为
    def test_to_weekly_resampling_disallow_how_kwd(self):
        # 创建一个周频率的日期范围
        idxh = date_range("1/1/1999", periods=52, freq="W")
        # 创建一个月末频率的日期范围
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        # 创建一个在高频率日期范围上标准正态分布随机数的 Series 对象
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 创建一个在低频率日期范围上标准正态分布随机数的 Series 对象
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # 创建一个新的图形和子图 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制高频率数据
        high.plot(ax=ax)
        # 准备错误消息，当使用了不允许的 how 关键字时抛出 ValueError 异常
        msg = (
            "'how' is not a valid keyword for plotting functions. If plotting "
            "multiple objects on shared axes, resample manually first."
        )
        # 使用 pytest 检查是否抛出了预期的 ValueError 异常，且错误消息匹配
        with pytest.raises(ValueError, match=msg):
            low.plot(ax=ax, how="foo")

    # 测试函数，验证周重采样的行为
    def test_to_weekly_resampling(self):
        # 创建一个周频率的日期范围
        idxh = date_range("1/1/1999", periods=52, freq="W")
        # 创建一个月末频率的日期范围
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        # 创建一个在高频率日期范围上标准正态分布随机数的 Series 对象
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 创建一个在低频率日期范围上标准正态分布随机数的 Series 对象
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # 创建一个新的图形和子图 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制高频率数据
        high.plot(ax=ax)
        # 在 ax 上绘制低频率数据
        low.plot(ax=ax)
        # 遍历 ax 中的每一条线
        for line in ax.get_lines():
            # 断言每条线的数据索引频率与高频率索引的频率相同
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
    # 定义一个测试函数，用于测试周频率重采样的功能
    def test_from_weekly_resampling(self):
        # 创建一个包含从 1999 年开始的 52 个周期的日期索引
        idxh = date_range("1/1/1999", periods=52, freq="W")
        # 创建一个包含从 1999 年开始的 12 个月末周期的日期索引
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        # 创建一个包含高频率数据的随机序列，与 idxh 对应
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 创建一个包含低频率数据的随机序列，与 idxl 对应
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # 创建一个图形及其轴对象
        _, ax = mpl.pyplot.subplots()
        # 在同一个轴上绘制低频率数据的图形
        low.plot(ax=ax)
        # 在同一个轴上绘制高频率数据的图形
        high.plot(ax=ax)

        # 创建预期的高频率数据，转换为周期并转换为 np.float64 类型
        expected_h = idxh.to_period().asi8.astype(np.float64)
        # 创建预期的低频率数据，使用 np.array 指定的固定值和数据类型 np.float64
        expected_l = np.array(
            [1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562],
            dtype=np.float64,
        )
        # 遍历图形对象 ax 中的每一条线
        for line in ax.get_lines():
            # 断言每条线的 x 轴数据是 PeriodIndex 类型且频率与 idxh 相同
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            # 获取线条的非原始 x 数据
            xdata = line.get_xdata(orig=False)
            # 如果 x 数据长度为 12，则为 idxl 对应的线条
            if len(xdata) == 12:  # idxl lines
                # 断言 x 数据与预期的低频率数据相等
                tm.assert_numpy_array_equal(xdata, expected_l)
            else:
                # 断言 x 数据与预期的高频率数据相等
                tm.assert_numpy_array_equal(xdata, expected_h)

    # 使用 pytest 的参数化装饰器来标记测试用例，参数为 kind1 和 kind2 的不同组合
    @pytest.mark.parametrize("kind1, kind2", [("line", "area"), ("area", "line")])
    # 定义一个测试方法，用于测试混合类型图表的生成
    def test_from_resampling_area_line_mixed(self, kind1, kind2):
        # 创建一个日期范围索引，频率为每周一次，从"1/1/1999"开始，共52个周期
        idxh = date_range("1/1/1999", periods=52, freq="W")
        # 创建一个日期范围索引，频率为每月末一次，从"1/1/1999"开始，共12个周期
        idxl = date_range("1/1/1999", periods=12, freq="M")
        
        # 创建一个高维度的数据框，使用随机数填充，行索引为idxh，列为[0, 1, 2]
        high = DataFrame(
            np.random.default_rng(2).random((len(idxh), 3)),
            index=idxh,
            columns=[0, 1, 2],
        )
        # 创建一个低维度的数据框，使用随机数填充，行索引为idxl，列为[0, 1, 2]
        low = DataFrame(
            np.random.default_rng(2).random((len(idxl), 3)),
            index=idxl,
            columns=[0, 1, 2],
        )

        # 创建一个子图并返回其对象
        _, ax = mpl.pyplot.subplots()
        
        # 在子图ax上绘制低维度数据框的图形，图形类型为kind1，堆叠方式为True
        low.plot(kind=kind1, stacked=True, ax=ax)
        
        # 在子图ax上绘制高维度数据框的图形，图形类型为kind2，堆叠方式为True
        high.plot(kind=kind2, stacked=True, ax=ax)

        # 检查低维度数据框的结果
        # 设置预期的x轴数据为特定的数值数组，数据类型为浮点数
        expected_x = np.array(
            [
                1514,
                1519,
                1523,
                1527,
                1531,
                1536,
                1540,
                1544,
                1549,
                1553,
                1558,
                1562,
            ],
            dtype=np.float64,
        )
        # 设置预期的y轴数据为全零数组，数据类型为浮点数
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        
        # 遍历3次，分别对每条线进行断言检查
        for i in range(3):
            # 获取子图ax上的第i条线的对象
            line = ax.lines[i]
            
            # 断言当前线的x轴数据的周期索引频率与idxh的频率相同
            assert PeriodIndex(line.get_xdata()).freq == idxh.freq
            
            # 断言当前线的x轴数据的值与预期的x轴数据相等
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            
            # 逐步累加低维度数据框第i列的数值，更新预期的y轴数据
            expected_y += low[i].values
            
            # 断言当前线的y轴数据的值与预期的y轴数据相等
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

        # 检查高维度数据框的结果
        # 将高维度数据框的索引转换为周期性的整数，再转换为浮点数数组作为预期的x轴数据
        expected_x = idxh.to_period().asi8.astype(np.float64)
        
        # 设置预期的y轴数据为全零数组，数据类型为浮点数
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        
        # 遍历3次，分别对每条线进行断言检查
        for i in range(3):
            # 获取子图ax上的第(3+i)条线的对象
            line = ax.lines[3 + i]
            
            # 断言当前线的x轴数据的数据索引频率与idxh的频率相同
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            
            # 断言当前线的x轴数据的值与预期的x轴数据相等
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            
            # 逐步累加高维度数据框第i列的数值，更新预期的y轴数据
            expected_y += high[i].values
            
            # 断言当前线的y轴数据的值与预期的y轴数据相等
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

    @pytest.mark.parametrize("kind1, kind2", [("line", "area"), ("area", "line")])
    def test_from_resampling_area_line_mixed_high_to_low(self, kind1, kind2):
        # 创建一个高频率日期索引，每周频率，共52个时间点
        idxh = date_range("1/1/1999", periods=52, freq="W")
        # 创建一个低频率日期索引，每季末频率，共12个时间点
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        
        # 生成一个高频率的数据帧，包含随机生成的数据，索引为idxh，列为[0, 1, 2]
        high = DataFrame(
            np.random.default_rng(2).random((len(idxh), 3)),
            index=idxh,
            columns=[0, 1, 2],
        )
        
        # 生成一个低频率的数据帧，包含随机生成的数据，索引为idxl，列为[0, 1, 2]
        low = DataFrame(
            np.random.default_rng(2).random((len(idxl), 3)),
            index=idxl,
            columns=[0, 1, 2],
        )
        
        # 创建一个新的图形并返回图形对象和子图对象
        _, ax = mpl.pyplot.subplots()
        
        # 使用高频率数据帧绘制图形，图形类型为kind1，堆叠模式为True，绘制在ax对象上
        high.plot(kind=kind1, stacked=True, ax=ax)
        
        # 使用低频率数据帧绘制图形，图形类型为kind2，堆叠模式为True，绘制在ax对象上
        low.plot(kind=kind2, stacked=True, ax=ax)

        # 检查高频率数据帧的结果
        expected_x = idxh.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        
        # 对每一条线进行验证
        for i in range(3):
            line = ax.lines[i]
            # 断言线的数据索引为周期索引，并且频率与idxh的频率相同
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            # 断言线的x数据与预期的x数据一致
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            # 更新预期的y数据并断言线的y数据与预期的y数据一致
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

        # 检查低频率数据帧的结果
        expected_x = np.array(
            [
                1514,
                1519,
                1523,
                1527,
                1531,
                1536,
                1540,
                1544,
                1549,
                1553,
                1558,
                1562,
            ],
            dtype=np.float64,
        )
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        
        # 对每一条线进行验证
        for i in range(3):
            lines = ax.lines[3 + i]
            # 断言线的数据索引为周期索引，并且频率与idxh的频率相同
            assert PeriodIndex(data=lines.get_xdata()).freq == idxh.freq
            # 断言线的x数据与预期的x数据一致
            tm.assert_numpy_array_equal(lines.get_xdata(orig=False), expected_x)
            # 更新预期的y数据并断言线的y数据与预期的y数据一致
            expected_y += low[i].values
            tm.assert_numpy_array_equal(lines.get_ydata(orig=False), expected_y)

    def test_mixed_freq_second_millisecond(self):
        # GH 7772, GH 7760
        # 创建一个秒级频率的日期索引，共5个时间点
        idxh = date_range("2014-07-01 09:00", freq="s", periods=5)
        # 创建一个百毫秒级频率的日期索引，共50个时间点
        idxl = date_range("2014-07-01 09:00", freq="100ms", periods=50)
        
        # 生成一个高频率时间序列，包含随机生成的标准正态分布数据，索引为idxh
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 生成一个低频率时间序列，包含随机生成的标准正态分布数据，索引为idxl
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        
        # 创建一个新的图形并返回图形对象和子图对象
        _, ax = mpl.pyplot.subplots()
        
        # 在同一个子图上绘制高频率和低频率时间序列的折线图
        high.plot(ax=ax)
        low.plot(ax=ax)
        
        # 断言图形中线条的数量为2
        assert len(ax.get_lines()) == 2
        
        # 对每一条线进行验证
        for line in ax.get_lines():
            # 断言线的数据索引为周期索引，并且频率为毫秒级
            assert PeriodIndex(data=line.get_xdata()).freq == "ms"
    def test_mixed_freq_second_millisecond_low_to_high(self):
        # GH 7772, GH 7760
        # 创建包含秒级频率的时间索引
        idxh = date_range("2014-07-01 09:00", freq="s", periods=5)
        # 创建包含100毫秒级频率的时间索引
        idxl = date_range("2014-07-01 09:00", freq="100ms", periods=50)
        # 使用随机数据创建高频率数据Series
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        # 使用随机数据创建低频率数据Series
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        
        # 创建图表并绘制低频率和高频率数据
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        
        # 断言图表中的线条数为2
        assert len(ax.get_lines()) == 2
        # 对每条线条进行断言，验证其时间索引的频率为毫秒级
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "ms"

    def test_irreg_dtypes(self):
        # date
        # 创建日期索引
        idx = [date(2000, 1, 1), date(2000, 1, 5), date(2000, 1, 20)]
        # 使用随机数据创建DataFrame，指定索引数据类型为对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            Index(idx, dtype=object),
        )
        # 检查绘图函数是否正常工作
        _check_plot_works(df.plot)

    def test_irreg_dtypes_dt64(self):
        # np.datetime64
        # 创建日期时间索引
        idx = date_range("1/1/2000", periods=10)
        # 将部分日期转换为对象类型
        idx = idx[[0, 2, 5, 9]].astype(object)
        # 使用随机数据创建DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
        # 创建图表并检查绘图函数是否正常工作，指定轴对象
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(df.plot, ax=ax)

    def test_time(self):
        # 创建特定时间点
        t = datetime(1, 1, 1, 3, 30, 0)
        # 创建随机增量时间
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        # 根据增量时间生成时间序列
        ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        # 使用随机数据创建DataFrame，以时间序列为索引
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        # 创建图表并绘制DataFrame数据
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)

        # 验证图表的刻度标签
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                # 根据刻度值生成时间字符串表示，并与标签内容进行断言比较
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs
    def test_time_change_xlim(self):
        # 创建一个 datetime 对象，表示时间为 3:30 AM
        t = datetime(1, 1, 1, 3, 30, 0)
        
        # 使用随机数生成器生成一个长度为 3 的随机整数数组，并进行累加得到时间增量
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        
        # 根据时间增量生成对应的时间戳数组
        ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        
        # 创建一个 DataFrame 对象，包含两列 'a' 和 'b'，用随机数填充，时间索引为 ts
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        
        # 创建一个新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        
        # 将 DataFrame df 的内容绘制到 ax 坐标轴上
        df.plot(ax=ax)

        # 验证 x 轴的刻度标签
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            # 将刻度值转换为小时、分钟、秒
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                # 根据秒数长度选择合适的时间格式化字符串
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                # 断言刻度标签的格式与预期相同
                assert xp == rs

        # 修改 x 轴的显示范围
        ax.set_xlim("1:30", "5:00")

        # 再次检查修改后的刻度标签
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            # 将刻度值转换为小时、分钟、秒
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                # 根据秒数长度选择合适的时间格式化字符串
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                # 断言刻度标签的格式与预期相同
                assert xp == rs

    def test_time_musec(self):
        # 创建一个 datetime 对象，表示时间为 3:30 AM
        t = datetime(1, 1, 1, 3, 30, 0)
        
        # 使用随机数生成器生成一个长度为 3 的随机整数数组，并进行累加得到微秒级时间增量
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        
        # 根据微秒级时间增量生成对应的时间戳数组
        ts = np.array([(t + timedelta(microseconds=int(x))).time() for x in deltas])
        
        # 创建一个 DataFrame 对象，包含两列 'a' 和 'b'，用随机数填充，时间索引为 ts
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        
        # 创建一个新的图形和坐标轴对象，并将 DataFrame df 的内容绘制到 ax 坐标轴上
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)

        # 验证 x 轴的刻度标签
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            # 将刻度值转换为小时、分钟、秒、微秒
            m, s = divmod(int(_tick), 60)
            us = round((_tick - int(_tick)) * 1e6)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                # 根据微秒数长度选择合适的时间格式化字符串
                if (us % 1000) != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S.%f")
                elif (us // 1000) != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S.%f")[:-3]
                elif s != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s, us).strftime("%H:%M")
                # 断言刻度标签的格式与预期相同
                assert xp == rs
    def test_secondary_upsample(self):
        # 创建两个不同频率的日期索引
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        
        # 创建两个随机数据系列，一个按天频率，一个按月末频率
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        
        # 创建一个新的图形和子图对象
        _, ax = mpl.pyplot.subplots()
        
        # 在子图上绘制低频率数据
        low.plot(ax=ax)
        
        # 在同一子图上绘制高频率数据，使用辅助y轴
        ax = high.plot(secondary_y=True, ax=ax)
        
        # 检查每条线的数据频率是否为天
        for line in ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == "D"
        
        # 断言子图对象具有属性 left_ax
        assert hasattr(ax, "left_ax")
        
        # 断言子图对象没有属性 right_ax
        assert not hasattr(ax, "right_ax")
        
        # 对左侧轴的每条线检查数据频率是否为天
        for line in ax.left_ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == "D"

    def test_secondary_legend(self):
        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        
        # 在图形上添加一个2x1的子图网格，并获取第一个子图
        ax = fig.add_subplot(211)

        # 创建一个包含随机数据的DataFrame，列名为 ABCD，行索引为工作日频率
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        
        # 在子图上绘制DataFrame的线图，指定A和B列使用辅助y轴
        df.plot(secondary_y=["A", "B"], ax=ax)
        
        # 获取子图上的图例对象
        leg = ax.get_legend()
        
        # 断言图例中有四条线
        assert len(leg.get_lines()) == 4
        
        # 断言图例的第一条文本是 "A (right)"
        assert leg.get_texts()[0].get_text() == "A (right)"
        
        # 断言图例的第二条文本是 "B (right)"
        assert leg.get_texts()[1].get_text() == "B (right)"
        
        # 断言图例的第三条文本是 "C"
        assert leg.get_texts()[2].get_text() == "C"
        
        # 断言图例的第四条文本是 "D"
        assert leg.get_texts()[3].get_text() == "D"
        
        # 断言右侧轴没有图例
        assert ax.right_ax.get_legend() is None
        
        # 检查图例中线的颜色，断言颜色集合中有四种颜色
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_right(self):
        # 创建一个包含随机数据的DataFrame，列名为 ABCD，行索引为工作日频率
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        
        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        
        # 在图形上添加一个2x1的子图网格，并获取第一个子图
        ax = fig.add_subplot(211)
        
        # 在子图上绘制DataFrame的线图，指定A和C列使用辅助y轴，右侧不标记
        df.plot(secondary_y=["A", "C"], mark_right=False, ax=ax)
        
        # 获取子图上的图例对象
        leg = ax.get_legend()
        
        # 断言图例中有四条线
        assert len(leg.get_lines()) == 4
        
        # 断言图例的第一条文本是 "A"
        assert leg.get_texts()[0].get_text() == "A"
        
        # 断言图例的第二条文本是 "B"
        assert leg.get_texts()[1].get_text() == "B"
        
        # 断言图例的第三条文本是 "C"
        assert leg.get_texts()[2].get_text() == "C"
        
        # 断言图例的第四条文本是 "D"
        assert leg.get_texts()[3].get_text() == "D"

    def test_secondary_legend_bar(self):
        # 创建一个包含随机数据的DataFrame，列名为 ABCD，行索引为工作日频率
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        
        # 创建一个新的图形对象和子图对象
        fig, ax = mpl.pyplot.subplots()
        
        # 在子图上绘制DataFrame的条形图，指定A列使用辅助y轴
        df.plot(kind="bar", secondary_y=["A"], ax=ax)
        
        # 获取子图上的图例对象
        leg = ax.get_legend()
        
        # 断言图例的第一条文本是 "A (right)"
        assert leg.get_texts()[0].get_text() == "A (right)"
        
        # 断言图例的第二条文本是 "B"
        assert leg.get_texts()[1].get_text() == "B"
    # 测试绘制带次要 y 轴的条形图，主要检验图例设置是否正确
    def test_secondary_legend_bar_right(self):
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 创建一个图形和轴对象
        fig, ax = mpl.pyplot.subplots()
        # 绘制 DataFrame 的条形图，将 A 列设为次要 y 轴，不在右边标记
        df.plot(kind="bar", secondary_y=["A"], mark_right=False, ax=ax)
        # 获取轴对象的图例
        leg = ax.get_legend()
        # 断言第一个图例文本是否为 "A"
        assert leg.get_texts()[0].get_text() == "A"
        # 断言第二个图例文本是否为 "B"
        assert leg.get_texts()[1].get_text() == "B"

    # 测试绘制带次要 y 轴的多列条形图
    def test_secondary_legend_multi_col(self):
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        # 添加一个子图
        ax = fig.add_subplot(211)
        # 绘制 DataFrame 的条形图，将 C 和 D 列设为次要 y 轴，使用已有的 ax 对象
        ax = df.plot(secondary_y=["C", "D"], ax=ax)
        # 获取轴对象的图例
        leg = ax.get_legend()
        # 断言图例中线条的数量是否为 4
        assert len(leg.get_lines()) == 4
        # 断言右侧轴对象没有图例
        assert ax.right_ax.get_legend() is None
        # 检查颜色集合，以确保颜色循环正确
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        # TODO: color cycle problems
        # 断言颜色数量为 4
        assert len(colors) == 4

    # 测试绘制带次要 y 轴的非时间序列数据条形图
    def test_secondary_legend_nonts(self):
        # 创建一个包含非时间序列数据的 DataFrame
        df = DataFrame(
            1.1 * np.arange(40).reshape((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(10)], dtype=object),
        )
        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        # 添加一个子图
        ax = fig.add_subplot(211)
        # 绘制 DataFrame 的条形图，将 A 和 B 列设为次要 y 轴，使用已有的 ax 对象
        ax = df.plot(secondary_y=["A", "B"], ax=ax)
        # 获取轴对象的图例
        leg = ax.get_legend()
        # 断言图例中线条的数量是否为 4
        assert len(leg.get_lines()) == 4
        # 断言右侧轴对象没有图例
        assert ax.right_ax.get_legend() is None
        # 检查颜色集合，以确保颜色循环正确
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        # TODO: color cycle problems
        # 断言颜色数量为 4
        assert len(colors) == 4

    # 测试绘制带次要 y 轴的非时间序列数据的多列条形图
    def test_secondary_legend_nonts_multi_col(self):
        # 创建一个包含非时间序列数据的 DataFrame
        df = DataFrame(
            1.1 * np.arange(40).reshape((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(10)], dtype=object),
        )
        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        # 添加一个子图
        ax = fig.add_subplot(211)
        # 绘制 DataFrame 的条形图，将 C 和 D 列设为次要 y 轴，使用已有的 ax 对象
        ax = df.plot(secondary_y=["C", "D"], ax=ax)
        # 获取轴对象的图例
        leg = ax.get_legend()
        # 断言图例中线条的数量是否为 4
        assert len(leg.get_lines()) == 4
        # 断言右侧轴对象没有图例
        assert ax.right_ax.get_legend() is None
        # 检查颜色集合，以确保颜色循环正确
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        # TODO: color cycle problems
        # 断言颜色数量为 4
        assert len(colors) == 4

    # 标记此测试为预期失败，原因是 API 在 3.6.0 版本中发生了更改
    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    # 测试日期轴的格式化功能
    def test_format_date_axis(self):
        # 创建一个日期范围
        rng = date_range("1/1/2012", periods=12, freq="ME")
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        # 创建子图并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 在轴上绘制 DataFrame 的数据，并返回轴对象
        ax = df.plot(ax=ax)
        # 获取轴对象的 X 轴
        xaxis = ax.get_xaxis()
        # 遍历 X 轴上的刻度标签
        for line in xaxis.get_ticklabels():
            # 如果标签文本长度大于零，断言其旋转角度为 30 度
            if len(line.get_text()) > 0:
                assert line.get_rotation() == 30

    # 测试在轴上绘制数据
    def test_ax_plot(self):
        # 创建一个日期范围
        x = date_range(start="2012-01-02", periods=10, freq="D")
        # 创建一个与日期范围长度相同的整数列表
        y = list(range(len(x)))
        # 创建子图并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 在轴上绘制 X 和 Y 数据，并返回线条对象列表
        lines = ax.plot(x, y, label="Y")
        # 断言线条对象的 X 数据索引与原始日期范围相同
        tm.assert_index_equal(DatetimeIndex(lines[0].get_xdata()), x)

    # 测试 matplotlib 绘图，不使用 pandas
    def test_mpl_nopandas(self):
        # 创建日期列表和数值数组
        dates = [date(2008, 12, 31), date(2009, 1, 31)]
        values1 = np.arange(10.0, 11.0, 0.5)
        values2 = np.arange(11.0, 12.0, 0.5)

        # 创建子图并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 在轴上绘制两条线条，每条线条由日期和对应数值组成
        (
            line1,
            line2,
        ) = ax.plot(
            [x.toordinal() for x in dates],
            values1,
            "-",
            [x.toordinal() for x in dates],
            values2,
            "-",
            linewidth=4,
        )

        # 创建期望的日期索引数组
        exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
        # 断言第一条线条的 X 数据索引与期望数组相同
        tm.assert_numpy_array_equal(line1.get_xydata()[:, 0], exp)
        # 断言第二条线条的 X 数据索引与期望数组相同
        tm.assert_numpy_array_equal(line2.get_xydata()[:, 0], exp)

    # 测试不规则时间序列共享轴的 X 轴限制
    def test_irregular_ts_shared_ax_xlim(self):
        # GH 2960
        # 创建包含不规则时间序列的 Series
        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]

        # 创建子图并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 在轴上绘制不规则时间序列的前后两个部分
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(ax=ax)

        # 检查轴的限制是否正确
        left, right = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), "", ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), "", ax)

    # 测试具有次要 Y 轴的非时间序列 X 轴限制
    def test_secondary_y_non_ts_xlim(self):
        # GH 3490 - 使用次要 Y 轴的非时间序列
        index_1 = [1, 2, 3, 4]
        index_2 = [5, 6, 7, 8]
        s1 = Series(1, index=index_1)
        s2 = Series(2, index=index_2)

        # 创建子图并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 在轴上绘制第一条数据系列
        s1.plot(ax=ax)
        # 获取绘制第一条数据系列前后的 X 轴限制
        left_before, right_before = ax.get_xlim()
        # 在次要 Y 轴上绘制第二条数据系列
        s2.plot(secondary_y=True, ax=ax)
        # 获取绘制第二条数据系列前后的 X 轴限制
        left_after, right_after = ax.get_xlim()

        # 断言绘制第一条数据系列后的左侧 X 轴限制不大于绘制第二条数据系列后的左侧 X 轴限制
        assert left_before >= left_after
        # 断言绘制第一条数据系列后的右侧 X 轴限制小于绘制第二条数据系列后的右侧 X 轴限制
        assert right_before < right_after
    def test_secondary_y_regular_ts_xlim(self):
        # 测试用例：正常时间序列与次要 y 轴限制
        # 创建两个日期范围，分别为四天的间隔
        index_1 = date_range(start="2000-01-01", periods=4, freq="D")
        index_2 = date_range(start="2000-01-05", periods=4, freq="D")
        # 创建两个 Series，每个 Series 的索引分别是 index_1 和 index_2
        s1 = Series(1, index=index_1)
        s2 = Series(2, index=index_2)

        # 创建一个新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 在主轴上绘制 s1 的数据
        s1.plot(ax=ax)
        # 获取绘制前的 x 轴限制
        left_before, right_before = ax.get_xlim()
        # 在次要 y 轴上绘制 s2 的数据，并在同一个坐标轴上绘制
        s2.plot(secondary_y=True, ax=ax)
        # 获取绘制后的 x 轴限制
        left_after, right_after = ax.get_xlim()

        # 断言：绘制 s2 后，左边界应该小于或等于绘制前的左边界
        assert left_before >= left_after
        # 断言：绘制 s2 后，右边界应该大于绘制前的右边界
        assert right_before < right_after

    def test_secondary_y_mixed_freq_ts_xlim(self):
        # 测试用例：混合频率时间序列与次要 y 轴限制
        # 创建一个包含分钟级间隔的时间序列
        rng = date_range("2000-01-01", periods=10, freq="min")
        ts = Series(1, index=rng)

        # 创建一个新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 在主轴上绘制 ts 的数据
        ts.plot(ax=ax)
        # 获取绘制前的 x 轴限制
        left_before, right_before = ax.get_xlim()
        # 对 ts 进行按日重采样后，在次要 y 轴上绘制平均值，并在同一个坐标轴上绘制
        ts.resample("D").mean().plot(secondary_y=True, ax=ax)
        # 获取绘制后的 x 轴限制
        left_after, right_after = ax.get_xlim()

        # 断言：重采样后，x 轴的限制应该保持不变
        assert left_before == left_after
        assert right_before == right_after

    def test_secondary_y_irregular_ts_xlim(self):
        # 测试用例：不规则时间序列与次要 y 轴限制
        # 创建一个包含浮点数的不规则时间序列
        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        # 从 ts 中选择不规则的索引子集
        ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]

        # 创建一个新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 在主轴上绘制 ts_irregular 的前五个数据点
        ts_irregular[:5].plot(ax=ax)
        # 在次要 y 轴上绘制 ts_irregular 的后面数据点，并在同一个坐标轴上绘制
        ts_irregular[5:].plot(secondary_y=True, ax=ax)
        # 再次在主轴上绘制 ts_irregular 的前五个数据点，以确保次要轴的限制没有被主轴的绘制改变

        # 获取最终的 x 轴限制
        left, right = ax.get_xlim()
        # 确保次要 y 轴的限制不会被主轴的绘制操作改变
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), "", ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), "", ax)

    def test_plot_outofbounds_datetime(self):
        # 测试用例：检查绘制超出范围的日期时间不会引发异常
        # 创建包含日期对象的列表
        values = [date(1677, 1, 1), date(1677, 1, 2)]
        # 创建一个新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 在坐标轴上绘制日期对象列表
        ax.plot(values)

        # 创建包含日期时间对象的列表
        values = [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]
        # 再次在同一个坐标轴上绘制日期时间对象列表
        ax.plot(values)

    def test_format_timedelta_ticks_narrow(self):
        # 测试用例：检查狭窄时间间隔的时间差格式化标签
        # 期望的标签列表，包含带有毫秒级时间的格式化字符串
        expected_labels = [f"00:00:00.0000000{i:0>2d}" for i in np.arange(10)]

        # 创建一个包含随机数据的 DataFrame，并使用纳秒为频率的时间范围作为索引
        rng = timedelta_range("0", periods=10, freq="ns")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        # 创建一个新的图形和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 在坐标轴上绘制 DataFrame 的数据，设置字体大小为 2
        df.plot(fontsize=2, ax=ax)
        # 执行图形绘制
        mpl.pyplot.draw()
        # 获取 x 轴刻度标签
        labels = ax.get_xticklabels()

        # 获取实际的刻度标签文本
        result_labels = [x.get_text() for x in labels]
        # 断言：实际的刻度标签应该与期望的标签列表相匹配
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels
    `
        def test_format_timedelta_ticks_wide(self):
            expected_labels = [
                "00:00:00",
                "1 days 03:46:40",
                "2 days 07:33:20",
                "3 days 11:20:00",
                "4 days 15:06:40",
                "5 days 18:53:20",
                "6 days 22:40:00",
                "8 days 02:26:40",
                "9 days 06:13:20",
            ]
    
            # 生成时间间隔序列，每天一个数据点，共10个数据点
            rng = timedelta_range("0", periods=10, freq="1 d")
            # 生成一个DataFrame，数据为随机生成的正态分布数据，时间索引为rng
            df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
            # 生成图形和轴对象
            _, ax = mpl.pyplot.subplots()
            # 在轴上绘制DataFrame的内容，字体大小为2
            ax = df.plot(fontsize=2, ax=ax)
            # 绘制图形
            mpl.pyplot.draw()
            # 获取轴上的x轴刻度标签
            labels = ax.get_xticklabels()
    
            # 获取标签文本内容
            result_labels = [x.get_text() for x in labels]
            # 断言刻度标签的数量和预期标签数量相同
            assert len(result_labels) == len(expected_labels)
            # 断言实际标签内容和预期标签内容完全相同
            assert result_labels == expected_labels
    
        def test_timedelta_plot(self):
            # 测试问题 #8711
            # 创建一个时间序列，每天一个数据点，共5个数据点
            s = Series(range(5), timedelta_range("1day", periods=5))
            # 生成图形和轴对象
            _, ax = mpl.pyplot.subplots()
            # 检查绘图函数正常工作
            _check_plot_works(s.plot, ax=ax)
    
        def test_timedelta_long_period(self):
            # 测试长时间段
            # 创建一个时间索引，从指定的时间点开始，每天一个数据点，共10个数据点
            index = timedelta_range("1 day 2 hr 30 min 10 s", periods=10, freq="1 d")
            # 创建一个随机生成的标准正态分布数据Series，索引为index
            s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
            # 生成图形和轴对象
            _, ax = mpl.pyplot.subplots()
            # 检查绘图函数正常工作
            _check_plot_works(s.plot, ax=ax)
    
        def test_timedelta_short_period(self):
            # 测试短时间段
            # 创建一个时间索引，从指定的时间点开始，每纳秒一个数据点，共10个数据点
            index = timedelta_range("1 day 2 hr 30 min 10 s", periods=10, freq="1 ns")
            # 创建一个随机生成的标准正态分布数据Series，索引为index
            s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
            # 生成图形和轴对象
            _, ax = mpl.pyplot.subplots()
            # 检查绘图函数正常工作
            _check_plot_works(s.plot, ax=ax)
    
        def test_hist(self):
            # https://github.com/matplotlib/matplotlib/issues/8459
            # 创建一个日期范围，从指定的日期开始，每小时一个数据点，共10个数据点
            rng = date_range("1/1/2011", periods=10, freq="h")
            x = rng
            # 创建两个权重数组
            w1 = np.arange(0, 1, 0.1)
            w2 = np.arange(0, 1, 0.1)[::-1]
            # 生成图形和轴对象
            _, ax = mpl.pyplot.subplots()
            # 绘制直方图，使用两个数据数组x和权重数组w1和w2
            ax.hist([x, x], weights=[w1, w2])
    
        def test_overlapping_datetime(self):
            # GB 6608
            # 创建第一个Series，包含三个数据点，每个数据点具有特定的日期时间索引
            s1 = Series(
                [1, 2, 3],
                index=[
                    datetime(1995, 12, 31),
                    datetime(2000, 12, 31),
                    datetime(2005, 12, 31),
                ],
            )
            # 创建第二个Series，包含三个数据点，每个数据点具有特定的日期时间索引
            s2 = Series(
                [1, 2, 3],
                index=[
                    datetime(1997, 12, 31),
                    datetime(2003, 12, 31),
                    datetime(2008, 12, 31),
                ],
            )
    
            # 生成图形和轴对象
            _, ax = mpl.pyplot.subplots()
            # 绘制第一个Series的内容到轴上
            s1.plot(ax=ax)
            # 添加第二个Series的内容到相同的轴上
            s2.plot(ax=ax)
            # 再次绘制第一个Series的内容到相同的轴上
            s1.plot(ax=ax)
    
        @pytest.mark.xfail(reason="GH9053 matplotlib does not use ax.xaxis.converter")
    def test_add_matplotlib_datetime64(self):
        # 测试函数：test_add_matplotlib_datetime64
        # GH9053 - 确保具有 PeriodConverter 的绘图仍能理解 datetime64 数据。
        # 这个测试仍然失败，因为 matplotlib 用 DatetimeConverter 覆盖了 ax.xaxis.converter
        s = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1970-01-02", periods=10),
        )
        # 绘制 Series s 的图形
        ax = s.plot()
        with tm.assert_produces_warning(DeprecationWarning):
            # 多维索引
            ax.plot(s.index, s.values, color="g")
        # 获取绘制的两条线
        l1, l2 = ax.lines
        # 断言两条线的数据相等
        tm.assert_numpy_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_matplotlib_scatter_datetime64(self):
        # 测试函数：test_matplotlib_scatter_datetime64
        # https://github.com/matplotlib/matplotlib/issues/11391
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=["x", "y"])
        df["time"] = date_range("2018-01-01", periods=10, freq="D")
        _, ax = mpl.pyplot.subplots()
        # 绘制散点图，x 轴为时间 "time"，y 轴为 "y"
        ax.scatter(x="time", y="y", data=df)
        # 绘制图形
        mpl.pyplot.draw()
        # 获取 x 轴的第一个刻度标签
        label = ax.get_xticklabels()[0]
        expected = "2018-01-01"
        # 断言标签文本与预期相符
        assert label.get_text() == expected

    def test_check_xticks_rot(self):
        # 测试函数：test_check_xticks_rot
        # https://github.com/pandas-dev/pandas/issues/29460
        # 常规时间序列
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # 按照 x 列绘制 df 的线图
        axes = df.plot(x="x", y="y")
        # 检查 x 轴刻度属性，旋转角度为 0
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_irregular(self):
        # 测试函数：test_check_xticks_rot_irregular
        # 不规则时间序列
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # 按照 x 列绘制 df 的线图
        axes = df.plot(x="x", y="y")
        # 检查 x 轴刻度属性，旋转角度为 30
        _check_ticks_props(axes, xrot=30)

    def test_check_xticks_rot_use_idx(self):
        # 测试函数：test_check_xticks_rot_use_idx
        # 不规则时间序列
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # 使用时间序列索引或不使用
        axes = df.set_index("x").plot(y="y", use_index=True)
        # 检查 x 轴刻度属性，旋转角度为 30
        _check_ticks_props(axes, xrot=30)
        axes = df.set_index("x").plot(y="y", use_index=False)
        # 检查 x 轴刻度属性，旋转角度为 0
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_sharex(self):
        # 测试函数：test_check_xticks_rot_sharex
        # 不规则时间序列
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # 单独的子图，共享 x 轴
        axes = df.plot(x="x", y="y", subplots=True, sharex=True)
        # 检查 x 轴刻度属性，旋转角度为 30
        _check_ticks_props(axes, xrot=30)
        axes = df.plot(x="x", y="y", subplots=True, sharex=False)
        # 检查 x 轴刻度属性，旋转角度为 0
        _check_ticks_props(axes, xrot=0)
    @pytest.mark.parametrize(
        "idx",
        [
            date_range("2020-01-01", periods=5),  # 生成一个日期范围，从 '2020-01-01' 开始，包含 5 个日期
            date_range("2020-01-01", periods=5, tz="UTC"),  # 生成一个带有时区的日期范围，从 '2020-01-01' 开始，包含 5 个日期
            timedelta_range("1 day", periods=5, freq="D"),  # 生成一个时间增量范围，每天增加 1 天，包含 5 个时间增量
            period_range("2020-01-01", periods=5, freq="D"),  # 生成一个时期范围，从 '2020-01-01' 开始，每天增加 1 天，包含 5 个时期
            Index([date(2000, 1, i) for i in [1, 3, 6, 20, 22]], dtype=object),  # 创建一个日期对象索引，包含指定的日期
            range(5),  # 创建一个整数范围，从 0 到 4
        ],
    )
    def test_pickle_fig(self, temp_file, frame_or_series, idx):
        # GH18439, GH#24088, statsmodels#4772
        # 使用给定的 frame_or_series 函数和索引 idx 创建 DataFrame 或 Series 对象
        df = frame_or_series(range(5), index=idx)
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots(1, 1)
        # 将 DataFrame 或 Series 对象绘制在指定的轴上
        df.plot(ax=ax)
        # 将图形对象 fig 序列化并存储到临时文件中
        with temp_file.open(mode="wb") as path:
            pickle.dump(fig, path)
# 定义一个名为 _check_plot_works 的函数，用于检查绘图是否正常工作
def _check_plot_works(f, freq=None, series=None, *args, **kwargs):
    # 获取当前的图形对象
    fig = plt.gcf()

    # 清空当前图形
    fig.clf()

    # 在图形中添加一个子图，并赋给 ax 变量
    ax = fig.add_subplot(211)

    # 从 kwargs 中获取 "ax" 参数的值，默认为 plt.gca() 返回的对象
    orig_ax = kwargs.pop("ax", plt.gca())

    # 获取 orig_ax 对象的 freq 属性，如果没有则为 None
    orig_axfreq = getattr(orig_ax, "freq", None)

    # 调用函数 f，并传入 *args 和 **kwargs，返回结果赋给 ret 变量
    ret = f(*args, **kwargs)

    # 断言 ret 不为 None，可以在此处进行更复杂的处理
    assert ret is not None  # do something more intelligent

    # 再次获取当前轴对象，并赋给 ax 变量
    ax = kwargs.pop("ax", plt.gca())

    # 如果 series 不为 None，则继续进行以下条件判断
    if series is not None:
        # 获取 series 对象的索引频率属性，并赋给 dfreq 变量
        dfreq = series.index.freq

        # 如果 dfreq 是 BaseOffset 的实例，则将其规则代码赋给 dfreq
        if isinstance(dfreq, BaseOffset):
            dfreq = dfreq.rule_code

        # 如果 orig_axfreq 为 None，则断言 ax 的频率与 dfreq 相等
        if orig_axfreq is None:
            assert ax.freq == dfreq

    # 如果 freq 不为 None 并且 orig_axfreq 为 None，则继续以下条件判断
    if freq is not None and orig_axfreq is None:
        # 断言 ax 的频率转换为偏移量后与 freq 相等
        assert to_offset(ax.freq, is_period=True) == freq

    # 在图形中添加第二个子图，并赋给 ax 变量
    ax = fig.add_subplot(212)

    # 将 ax 参数添加到 kwargs 中
    kwargs["ax"] = ax

    # 再次调用函数 f，并传入 *args 和更新后的 **kwargs，返回结果赋给 ret 变量
    ret = f(*args, **kwargs)

    # 断言 ret 不为 None，可以在此处进行更复杂的处理
    assert ret is not None  # TODO: do something more intelligent
```