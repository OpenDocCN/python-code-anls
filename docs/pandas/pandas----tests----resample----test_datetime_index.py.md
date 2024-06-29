# `D:\src\scipysrc\pandas\pandas\tests\resample\test_datetime_index.py`

```
from datetime import datetime
from functools import partial  # 导入 partial 函数用于创建偏函数
import zoneinfo  # 导入 zoneinfo 模块

import numpy as np  # 导入 NumPy 库，并使用 np 作为别名
import pytest  # 导入 pytest 测试框架

from pandas._libs import lib  # 导入 pandas 内部库
from pandas._typing import DatetimeNaTType  # 导入 pandas 时间类型的 NaT
from pandas.compat import is_platform_windows  # 导入用于判断平台是否为 Windows 的兼容性函数
import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器模块

import pandas as pd  # 导入 pandas 库，并使用 pd 作为别名
from pandas import (  # 从 pandas 中导入多个类和函数
    DataFrame,
    Index,
    Series,
    Timedelta,
    Timestamp,
    isna,
    notna,
)
import pandas._testing as tm  # 导入 pandas 测试模块
from pandas.core.groupby.grouper import Grouper  # 从 pandas 核心模块导入 Grouper 类
from pandas.core.indexes.datetimes import date_range  # 从 pandas 日期索引模块导入 date_range 函数
from pandas.core.indexes.period import (  # 从 pandas 期间索引模块导入 Period 和 period_range 函数
    Period,
    period_range,
)
from pandas.core.resample import (  # 从 pandas 重新采样模块导入 DatetimeIndex 和 _get_timestamp_range_edges 函数
    DatetimeIndex,
    _get_timestamp_range_edges,
)

from pandas.tseries import offsets  # 从 pandas 时间序列模块导入 offsets
from pandas.tseries.offsets import Minute  # 从 pandas 时间序列偏移模块导入 Minute 偏移量类


@pytest.fixture
def simple_date_range_series():
    """
    Series with date range index and random data for test purposes.
    """

    def _simple_date_range_series(start, end, freq="D"):
        rng = date_range(start, end, freq=freq)  # 生成日期范围
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)  # 返回一个随机数据的 Series 对象

    return _simple_date_range_series  # 返回内部函数 _simple_date_range_series


def test_custom_grouper(unit):
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="Min")  # 创建分钟频率的日期范围索引
    dti = index.as_unit(unit)  # 将索引转换为指定的时间单位
    s = Series(np.array([1] * len(dti)), index=dti, dtype="int64")  # 创建一个带有指定数据和索引的 Series 对象

    b = Grouper(freq=Minute(5))  # 创建一个按 5 分钟频率分组的 Grouper 对象
    g = s.groupby(b)  # 对 Series 对象按照 Grouper 对象进行分组

    # check all cython functions work
    g.ohlc()  # 调用 ohlc 方法进行开、高、低、收计算，不使用 _cython_agg_general
    funcs = ["sum", "mean", "prod", "min", "max", "var"]
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)  # 循环调用 _cython_agg_general 方法进行各种聚合计算

    b = Grouper(freq=Minute(5), closed="right", label="right")  # 创建一个按 5 分钟频率、右闭合、右标签的 Grouper 对象
    g = s.groupby(b)  # 对 Series 对象按照新的 Grouper 对象进行分组
    # check all cython functions work
    g.ohlc()  # 调用 ohlc 方法进行开、高、低、收计算，不使用 _cython_agg_general
    funcs = ["sum", "mean", "prod", "min", "max", "var"]
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)  # 循环调用 _cython_agg_general 方法进行各种聚合计算

    assert g.ngroups == 2593  # 断言分组后的组数为 2593
    assert notna(g.mean()).all()  # 断言分组后的均值不含 NaN

    # construct expected val
    arr = [1] + [5] * 2592
    idx = dti[0:-1:5]
    idx = idx.append(dti[-1:])
    idx = DatetimeIndex(idx, freq="5min").as_unit(unit)  # 将索引转换为指定的时间单位
    expect = Series(arr, index=idx)  # 创建预期的 Series 对象

    # GH2763 - return input dtype if we can
    result = g.agg("sum")  # 对分组后的数据进行求和聚合
    tm.assert_series_equal(result, expect)  # 使用测试模块中的函数进行结果比较


def test_custom_grouper_df(unit):
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")  # 创建日频率的日期范围索引
    b = Grouper(freq=Minute(5), closed="right", label="right")  # 创建一个按 5 分钟频率、右闭合、右标签的 Grouper 对象
    dti = index.as_unit(unit)  # 将索引转换为指定的时间单位
    df = DataFrame(  # 创建一个 DataFrame 对象
        np.random.default_rng(2).random((len(dti), 10)), index=dti, dtype="float64"
    )
    r = df.groupby(b).agg("sum")  # 对 DataFrame 对象按照 Grouper 对象进行分组并进行求和聚合

    assert len(r.columns) == 10  # 断言聚合后的 DataFrame 列数为 10
    assert len(r.index) == 2593  # 断言聚合后的 DataFrame 行数为 2593


@pytest.mark.parametrize(
    "closed, expected",
    [
        (
            "right",
            # 创建一个 Series 对象，包含四个元素：第一个元素是 s 的第一个元素，后面三个元素是 s 的不同区间的均值
            lambda s: Series(
                [s.iloc[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()],
                index=date_range("1/1/2000", periods=4, freq="5min", name="index"),  # 指定索引为时间序列，频率为每5分钟
            ),
        ),
        (
            "left",
            # 创建一个 Series 对象，包含三个元素：分别是 s 的前五个元素的均值，接着 s 的中间五个元素的均值，最后是 s 剩余元素的均值
            lambda s: Series(
                [s[:5].mean(), s[5:10].mean(), s[10:].mean()],
                index=date_range(
                    "1/1/2000 00:05", periods=3, freq="5min", name="index"  # 指定索引为时间序列，从 "1/1/2000 00:05" 开始，每5分钟一个周期
                ),
            ),
        ),
    ],
def test_resample_basic(closed, expected, unit):
    # 创建一个日期范围，从 "1/1/2000 00:00:00" 到 "1/1/2000 00:13:00"，频率为每分钟
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    # 创建一个序列，以索引为日期范围的时间戳，值为索引值
    s = Series(range(len(index)), index=index)
    # 设置序列的索引名称为 "index"，并将索引转换为指定单位
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    # 调用期望的函数来生成预期的结果
    expected = expected(s)
    # 将预期结果的索引转换为指定单位
    expected.index = expected.index.as_unit(unit)
    # 对序列进行重采样，时间间隔为 "5min"，使用指定的关闭方式和标签
    result = s.resample("5min", closed=closed, label="right").mean()
    # 使用测试框架检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


def test_resample_integerarray(unit):
    # GH 25580, 在整数数组上进行重采样
    # 创建一个时间序列，索引为指定时间范围，频率为每分钟，数据类型为 Int64
    ts = Series(
        range(9),
        index=date_range("1/1/2000", periods=9, freq="min").as_unit(unit),
        dtype="Int64",
    )
    # 对时间序列进行 "3min" 时间间隔的重采样，计算总和
    result = ts.resample("3min").sum()
    # 创建预期结果的序列，指定时间范围和频率，数据类型为 Int64
    expected = Series(
        [3, 12, 21],
        index=date_range("1/1/2000", periods=3, freq="3min").as_unit(unit),
        dtype="Int64",
    )
    # 使用测试框架检查结果是否与预期相等
    tm.assert_series_equal(result, expected)

    # 对时间序列进行 "3min" 时间间隔的重采样，计算均值
    result = ts.resample("3min").mean()
    # 创建预期结果的序列，指定时间范围和频率，数据类型为 Float64
    expected = Series(
        [1, 4, 7],
        index=date_range("1/1/2000", periods=3, freq="3min").as_unit(unit),
        dtype="Float64",
    )
    # 使用测试框架检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


def test_resample_basic_grouper(unit):
    # 创建一个日期范围，从 "1/1/2000 00:00:00" 到 "1/1/2000 00:13:00"，频率为每分钟
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    # 创建一个序列，以索引为日期范围的时间戳，值为索引值
    s = Series(range(len(index)), index=index)
    # 设置序列的索引名称为 "index"，并将索引转换为指定单位
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    # 对序列进行 "5Min" 时间间隔的重采样，取最后一个值
    result = s.resample("5Min").last()
    # 创建分组器对象，频率为 5分钟，关闭方式为 "left"，标签为 "left"
    grouper = Grouper(freq=Minute(5), closed="left", label="left")
    # 使用分组器对序列进行分组，并对每组应用取最后一个值的函数来生成预期结果
    expected = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    # 使用测试框架检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:The 'convention' keyword in Series.resample:FutureWarning"
)
@pytest.mark.parametrize(
    "keyword,value",
    [("label", "righttt"), ("closed", "righttt"), ("convention", "starttt")],
)
def test_resample_string_kwargs(keyword, value, unit):
    # 查看 gh-19303
    # 检查错误的关键字参数字符串是否引发错误
    # 创建一个日期范围，从 "1/1/2000 00:00:00" 到 "1/1/2000 00:13:00"，频率为每分钟
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    # 创建一个序列，以索引为日期范围的时间戳，值为索引值
    series = Series(range(len(index)), index=index)
    # 设置序列的索引名称为 "index"，并将索引转换为指定单位
    series.index.name = "index"
    series.index = series.index.as_unit(unit)
    # 准备一个错误信息字符串
    msg = f"Unsupported value {value} for `{keyword}`"
    # 使用 pytest 框架来确保在设置错误的关键字参数时会引发 ValueError 错误，并且错误信息与预期的错误信息匹配
    with pytest.raises(ValueError, match=msg):
        series.resample("5min", **({keyword: value}))


def test_resample_how(downsample_method, unit):
    # 如果 downsample_method 是 "ohlc"，则跳过该测试
    if downsample_method == "ohlc":
        pytest.skip("covered by test_resample_how_ohlc")
    # 创建一个日期范围，从 "1/1/2000 00:00:00" 到 "1/1/2000 00:13:00"，频率为每分钟
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    # 创建一个序列，以索引为日期范围的时间戳，值为索引值
    s = Series(range(len(index)), index=index)
    # 设置序列的索引名称为 "index"，并将索引转换为指定单位
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    # 创建一个数组，所有元素为 1，但第一个元素为 0
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3
    # 使用分组列表对序列进行分组，并使用指定的下采样方法对每组进行聚合，生成预期结果
    expected = s.groupby(grouplist).agg(downsample_method)
    # 将预期结果的索引转换为指定单位
    expected.index = date_range(
        "1/1/2000", periods=4, freq="5min", name="index"
    ).as_unit(unit)
    # 调用getattr函数获取对象s的resample方法的结果，使用"5min"频率重新采样，右闭合，右标签
    # 并根据变量downsample_method指定的方法进行下采样处理，结果存储在result变量中
    result = getattr(
        s.resample("5min", closed="right", label="right"), downsample_method
    )()
    # 使用tm.assert_series_equal函数比较result和expected两个Series对象是否相等
    tm.assert_series_equal(result, expected)
def test_resample_how_ohlc(unit):
    # 创建时间索引，从 "1/1/2000 00:00:00" 到 "1/1/2000 00:13:00"，频率为每分钟
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    # 创建一个序列，索引为时间索引，值为索引的位置
    s = Series(range(len(index)), index=index)
    # 设置序列的索引名称为 "index"，并将索引转换为指定的时间单位（unit）
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    # 创建一个与序列长度相同的全为1的数组
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3

    def _ohlc(group):
        # 如果分组中所有值都是缺失值，则返回包含四个NaN值的数组
        if isna(group).all():
            return np.repeat(np.nan, 4)
        # 否则，返回包含分组的开盘价、最高价、最低价和收盘价的数组
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]

    # 期望的DataFrame，由分组后的序列数据计算得出
    expected = DataFrame(
        s.groupby(grouplist).agg(_ohlc).values.tolist(),
        # 创建时间范围索引，从 "1/1/2000" 开始，每4个时间点为一组，频率为5分钟，单位为指定的时间单位（unit）
        index=date_range("1/1/2000", periods=4, freq="5min", name="index").as_unit(
            unit
        ),
        columns=["open", "high", "low", "close"],
    )

    # 使用序列的resample方法，按照5分钟间隔重新采样，并计算开盘价、最高价、最低价和收盘价
    result = s.resample("5min", closed="right", label="right").ohlc()
    # 断言结果与期望的DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_resample_how_callables(unit):
    # GH#7929
    # 创建一个包含5个整数的数组
    data = np.arange(5, dtype=np.int64)
    # 创建一个日期索引，从 "2014-01-01" 开始，长度与数据数组相同，频率为每天，并转换为指定的时间单位（unit）
    ind = date_range(start="2014-01-01", periods=len(data), freq="d").as_unit(unit)
    # 创建一个DataFrame，包含两列，列名为"A"和"B"，索引为日期索引
    df = DataFrame({"A": data, "B": data}, index=ind)

    def fn(x, a=1):
        # 返回输入对象的类型的字符串表示
        return str(type(x))

    class FnClass:
        def __call__(self, x):
            # 返回输入对象的类型的字符串表示
            return str(type(x))

    # 使用resample方法，按照"ME"（每月结束）重新采样，并应用函数fn
    df_standard = df.resample("ME").apply(fn)
    # 使用resample方法，按照"ME"重新采样，并应用lambda函数
    df_lambda = df.resample("ME").apply(lambda x: str(type(x)))
    # 使用resample方法，按照"ME"重新采样，并应用partial函数，传递给fn的参数为默认值
    df_partial = df.resample("ME").apply(partial(fn))
    # 使用resample方法，按照"ME"重新采样，并应用partial函数，传递给fn的参数为2
    df_partial2 = df.resample("ME").apply(partial(fn, a=2))
    # 使用resample方法，按照"ME"重新采样，并应用FnClass类的实例
    df_class = df.resample("ME").apply(FnClass())

    # 断言各个DataFrame的结果与df_standard相等
    tm.assert_frame_equal(df_standard, df_lambda)
    tm.assert_frame_equal(df_standard, df_partial)
    tm.assert_frame_equal(df_standard, df_partial2)
    tm.assert_frame_equal(df_standard, df_class)


def test_resample_rounding(unit):
    # GH 8371
    # 当需要进行舍入时出现奇怪的结果

    # 创建一个时间戳列表
    ts = [
        "2014-11-08 00:00:01",
        "2014-11-08 00:00:02",
        "2014-11-08 00:00:02",
        "2014-11-08 00:00:03",
        "2014-11-08 00:00:07",
        "2014-11-08 00:00:07",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:09",
        "2014-11-08 00:00:10",
        "2014-11-08 00:00:11",
        "2014-11-08 00:00:11",
        "2014-11-08 00:00:13",
        "2014-11-08 00:00:14",
        "2014-11-08 00:00:15",
        "2014-11-08 00:00:17",
        "2014-11-08 00:00:20",
        "2014-11-08 00:00:21",
    ]
    # 创建一个DataFrame，包含一个名为"value"的列，所有值为1，索引为时间戳列表，并转换为指定的时间单位（unit）
    df = DataFrame({"value": [1] * 19}, index=pd.to_datetime(ts))
    df.index = df.index.as_unit(unit)

    # 使用resample方法，按照6秒间隔重新采样，并计算和
    result = df.resample("6s").sum()
    # 创建一个期望的DataFrame，包含"value"列，值为预期的和，索引为6秒间隔的时间范围，并转换为指定的时间单位（unit）
    expected = DataFrame(
        {"value": [4, 9, 4, 2]},
        index=date_range("2014-11-08", freq="6s", periods=4).as_unit(unit),
    )
    # 断言结果与期望的DataFrame相等
    tm.assert_frame_equal(result, expected)

    # 使用resample方法，按照7秒间隔重新采样，并计算和
    result = df.resample("7s").sum()
    # 创建一个期望的DataFrame，包含"value"列，值为预期的和，索引为7秒间隔的时间范围，并转换为指定的时间单位（unit）
    expected = DataFrame(
        {"value": [4, 10, 4, 1]},
        index=date_range("2014-11-08", freq="7s", periods=4).as_unit(unit),
    )
    # 断言结果与期望的DataFrame相等
    tm.assert_frame_equal(result, expected)
    # 对 DataFrame 进行时间重采样，按照每11秒的间隔对数据进行求和
    result = df.resample("11s").sum()
    
    # 创建预期的 DataFrame，包含指定数值和索引，索引是从指定日期开始，每11秒增加一个时间单位，共2个时间点
    expected = DataFrame(
        {"value": [11, 8]},
        index=date_range("2014-11-08", freq="11s", periods=2).as_unit(unit),
    )
    # 使用断言检查重采样后的结果与预期的DataFrame是否相等
    tm.assert_frame_equal(result, expected)
    
    # 对 DataFrame 进行时间重采样，按照每13秒的间隔对数据进行求和
    result = df.resample("13s").sum()
    
    # 创建预期的 DataFrame，包含指定数值和索引，索引是从指定日期开始，每13秒增加一个时间单位，共2个时间点
    expected = DataFrame(
        {"value": [13, 6]},
        index=date_range("2014-11-08", freq="13s", periods=2).as_unit(unit),
    )
    # 使用断言检查重采样后的结果与预期的DataFrame是否相等
    tm.assert_frame_equal(result, expected)
    
    # 对 DataFrame 进行时间重采样，按照每17秒的间隔对数据进行求和
    result = df.resample("17s").sum()
    
    # 创建预期的 DataFrame，包含指定数值和索引，索引是从指定日期开始，每17秒增加一个时间单位，共2个时间点
    expected = DataFrame(
        {"value": [16, 3]},
        index=date_range("2014-11-08", freq="17s", periods=2).as_unit(unit),
    )
    # 使用断言检查重采样后的结果与预期的DataFrame是否相等
    tm.assert_frame_equal(result, expected)
def test_resample_basic_from_daily(unit):
    # 根据日频率生成日期索引
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D", name="index"
    ).as_unit(unit)

    # 使用随机数生成一个与日期索引长度相同的Series
    s = Series(np.random.default_rng(2).random(len(dti)), dti)

    # 将日频率数据重采样为周日结束的频率，并取最后一个值
    result = s.resample("w-sun").last()

    # 断言结果长度为3
    assert len(result) == 3
    # 断言结果索引的星期几为周日
    assert (result.index.dayofweek == [6, 6, 6]).all()
    # 断言结果的第一个值等于原始Series中日期为"1/2/2005"的值
    assert result.iloc[0] == s["1/2/2005"]
    # 断言结果的第二个值等于原始Series中日期为"1/9/2005"的值
    assert result.iloc[1] == s["1/9/2005"]
    # 断言结果的最后一个值等于原始Series的最后一个值
    assert result.iloc[2] == s.iloc[-1]

    # 将日频率数据重采样为周一结束的频率，并取最后一个值
    result = s.resample("W-MON").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [0, 0]).all()
    assert result.iloc[0] == s["1/3/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # 将日频率数据重采样为周二结束的频率，并取最后一个值
    result = s.resample("W-TUE").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [1, 1]).all()
    assert result.iloc[0] == s["1/4/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # 将日频率数据重采样为周三结束的频率，并取最后一个值
    result = s.resample("W-WED").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [2, 2]).all()
    assert result.iloc[0] == s["1/5/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # 将日频率数据重采样为周四结束的频率，并取最后一个值
    result = s.resample("W-THU").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [3, 3]).all()
    assert result.iloc[0] == s["1/6/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # 将日频率数据重采样为周五结束的频率，并取最后一个值
    result = s.resample("W-FRI").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [4, 4]).all()
    assert result.iloc[0] == s["1/7/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # 将日频率数据重采样为工作日的频率，并取最后一个值
    result = s.resample("B").last()
    assert len(result) == 7
    # 断言结果索引的星期几分别为[4, 0, 1, 2, 3, 4, 0]，即周五至下周一
    assert (result.index.dayofweek == [4, 0, 1, 2, 3, 4, 0]).all()

    assert result.iloc[0] == s["1/2/2005"]
    assert result.iloc[1] == s["1/3/2005"]
    assert result.iloc[5] == s["1/9/2005"]
    # 断言结果的索引名称为 "index"
    assert result.index.name == "index"


def test_resample_upsampling_picked_but_not_correct(unit):
    # 测试问题 #3020
    # 创建日期范围为 "01-Jan-2014" 至 "05-Jan-2014" 的日期索引
    dates = date_range("01-Jan-2014", "05-Jan-2014", freq="D").as_unit(unit)
    # 创建索引为 dates 的Series，值均为1
    series = Series(1, index=dates)

    # 将数据重采样为每日频率，并计算均值
    result = series.resample("D").mean()
    # 断言结果的第一个索引值等于 dates 的第一个值
    assert result.index[0] == dates[0]

    # GH 5955
    # 当轴的频率与重采样的频率匹配时，不正确地决定上采样

    # 创建索引为特定日期的Series
    s = Series(
        np.arange(1.0, 6), index=[datetime(1975, 1, i, 12, 0) for i in range(1, 6)]
    )
    s.index = s.index.as_unit(unit)
    # 创建预期的Series，其索引与日期范围 "19750101" 至 "19750105" 一致
    expected = Series(
        np.arange(1.0, 6),
        index=date_range("19750101", periods=5, freq="D").as_unit(unit),
    )

    # 将数据重采样为每日频率，并计算计数
    result = s.resample("D").count()
    # 使用断言验证结果与预期相符
    tm.assert_series_equal(result, Series(1, index=expected.index))

    # 将数据重采样为每日频率，并计算总和
    result1 = s.resample("D").sum()
    # 将数据重采样为每日频率，并计算均值
    result2 = s.resample("D").mean()
    # 使用断言验证结果与预期相符
    tm.assert_series_equal(result1, expected)
    tm.assert_series_equal(result2, expected)
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )

创建一个 DataFrame 对象 `df`，其中包含一个 50 行 4 列的随机数矩阵，这些随机数来自于标准正态分布。DataFrame 的列名为 ['A', 'B', 'C', 'D']，行索引为从 "2000-01-01" 开始的 50 个工作日的日期范围。


    df.index = df.index.as_unit(unit)

将 `df` 的索引转换为指定的时间单位 `unit`，假设 `unit` 是一个有效的时间单位字符串。


    b = Grouper(freq="ME")
    g = df.groupby(b)

创建一个 `Grouper` 对象 `b`，用于分组 `df` 的时间序列数据，频率为每月末 ("ME")。然后根据这个 `Grouper` 对象 `b` 对 `df` 进行分组，生成一个分组后的 `DataFrameGroupBy` 对象 `g`。


    # check all cython functions work
    g._cython_agg_general(f, alt=None, numeric_only=True)

调用分组后的 `DataFrameGroupBy` 对象 `g` 的 `_cython_agg_general` 方法，传递参数 `f` 以及额外的参数 `alt=None` 和 `numeric_only=True`。这个步骤用于检查所有 Cython 函数是否正常工作。
@pytest.mark.parametrize("freq", ["YE", "ME"])
# 使用 pytest 的 parametrize 装饰器，为 test_resample_frame_basic_M_A 函数定义了两个参数化的测试用例：freq 分别为 "YE" 和 "ME"
def test_resample_frame_basic_M_A(freq, unit):
    # 创建一个 DataFrame 对象，包含 50 行 4 列的随机标准正态分布数据
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    # 将 DataFrame 的索引转换为指定的单位（由外部传入的 unit 参数）
    df.index = df.index.as_unit(unit)
    # 对 DataFrame 进行频率重采样，并计算各列的均值
    result = df.resample(freq).mean()
    # 使用测试框架的 assert_series_equal 函数比较 resample 后的结果中的列"A"与原始数据 resample 后"A"列的均值
    tm.assert_series_equal(result["A"], df["A"].resample(freq).mean())


def test_resample_upsample(unit):
    # 生成一个时间索引，从 2005-01-01 到 2005-01-10，频率为每天，作为单位（由外部传入的 unit 参数）
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D", name="index"
    ).as_unit(unit)

    # 生成一个 Series 对象，包含与时间索引相同长度的随机数据
    s = Series(np.random.default_rng(2).random(len(dti)), dti)

    # 将 Series 对象按分钟重新采样，并使用前向填充方法
    result = s.resample("Min").ffill()
    # 断言重新采样后的结果长度为 12961
    assert len(result) == 12961
    # 断言重新采样后的结果的第一个元素与原始 Series 的第一个元素相等
    assert result.iloc[0] == s.iloc[0]
    # 断言重新采样后的结果的最后一个元素与原始 Series 的最后一个元素相等
    assert result.iloc[-1] == s.iloc[-1]
    # 断言重新采样后的结果的索引名称为 "index"
    assert result.index.name == "index"


def test_resample_how_method(unit):
    # 创建一个 Series 对象，包含两个元素，具有指定的时间戳索引（由外部传入的 unit 参数）
    s = Series(
        [11, 22],
        index=[
            Timestamp("2015-03-31 21:48:52.672000"),
            Timestamp("2015-03-31 21:49:52.739000"),
        ],
    )
    # 将 Series 的索引转换为指定的单位
    s.index = s.index.as_unit(unit)
    # 创建一个期望的 Series 对象，包含指定时间间隔的时间戳索引，并转换为指定的单位
    expected = Series(
        [11, np.nan, np.nan, np.nan, np.nan, np.nan, 22],
        index=DatetimeIndex(
            [
                Timestamp("2015-03-31 21:48:50"),
                Timestamp("2015-03-31 21:49:00"),
                Timestamp("2015-03-31 21:49:10"),
                Timestamp("2015-03-31 21:49:20"),
                Timestamp("2015-03-31 21:49:30"),
                Timestamp("2015-03-31 21:49:40"),
                Timestamp("2015-03-31 21:49:50"),
            ],
            freq="10s",
        ),
    )
    # 将期望的 Series 对象的索引转换为指定的单位
    expected.index = expected.index.as_unit(unit)
    # 使用测试框架的 assert_series_equal 函数比较重采样后的结果与期望的结果
    tm.assert_series_equal(s.resample("10s").mean(), expected)


def test_resample_extra_index_point(unit):
    # 创建一个日期范围的时间索引，从 20150101 到 20150331，频率为每月末的工作日，将其转换为指定的单位（由外部传入的 unit 参数）
    index = date_range(start="20150101", end="20150331", freq="BME").as_unit(unit)
    # 创建一个期望的 DataFrame 对象，包含指定索引的 Series 对象 "A"
    expected = DataFrame({"A": Series([21, 41, 63], index=index)})

    # 创建一个日期范围的时间索引，从 20150101 到 20150331，频率为每天，将其转换为指定的单位
    index = date_range(start="20150101", end="20150331", freq="B").as_unit(unit)
    # 创建一个 DataFrame 对象，包含指定索引的 Series 对象 "A"
    df = DataFrame({"A": Series(range(len(index)), index=index)}, dtype="int64")
    # 对 DataFrame 对象进行重采样，每月末，选择最后一个工作日
    result = df.resample("BME").last()
    # 使用测试框架的 assert_frame_equal 函数比较重采样后的结果与期望的结果
    tm.assert_frame_equal(result, expected)


def test_upsample_with_limit(unit):
    # 创建一个时间索引，从 "2000-01-01" 开始，频率为每 5 分钟，将其转换为指定的单位（由外部传入的 unit 参数）
    rng = date_range("1/1/2000", periods=3, freq="5min").as_unit(unit)
    # 创建一个 Series 对象，包含与时间索引相同长度的标准正态分布的随机数据
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

    # 对 Series 对象进行频率重采样，每分钟，并使用前向填充方法，限制为最多 2 个填充
    result = ts.resample("min").ffill(limit=2)
    # 重新索引原始 Series，并使用前向填充方法，限制为最多 2 个填充
    expected = ts.reindex(result.index, method="ffill", limit=2)
    # 使用测试框架的 assert_series_equal 函数比较重采样后的结果与期望的结果
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("freq", ["1D", "10h", "5Min", "10s"])
@pytest.mark.parametrize("rule", ["YE", "3ME", "15D", "30h", "15Min", "30s"])
# 使用 pytest 的 parametrize 装饰器，为 test_nearest_upsample_with_limit 函数定义了两组参数化的测试用例：freq 和 rule
def test_nearest_upsample_with_limit(tz_aware_fixture, freq, rule, unit):
    # 创建一个日期范围的时间索引，从 "2000-01-01" 开始，频率由外部传入的 freq 参数指定，并将其转换为指定的单位（由外部传入的 unit 参数）
    rng = date_range("1/1/2000", periods=3, freq=freq, tz=tz_aware_fixture).as_unit(
        unit
    )
    # 使用 NumPy 中的随机数生成器创建一个 Series，数据为长度与给定的时间序列 rng 相同的标准正态分布随机数
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    
    # 对时间序列 ts 进行重采样，使用 rule 规则，选择最接近的值，限制为每个时间点最多两个值
    result = ts.resample(rule).nearest(limit=2)
    
    # 使用最接近方法（nearest）重新索引时间序列 ts，使其索引与 result 相同，限制每个时间点最多两个值
    expected = ts.reindex(result.index, method="nearest", limit=2)
    
    # 使用断言检查 result 和 expected 是否相等，如果不相等则引发 AssertionError
    tm.assert_series_equal(result, expected)
def test_resample_ohlc(unit):
    # 创建一个时间索引，从2005年1月1日到2005年1月10日，每分钟一次
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="Min")
    # 创建一个时间序列，索引为上述时间索引，值为索引的位置
    s = Series(range(len(index)), index=index)
    # 设置时间索引的名称为 "index"
    s.index.name = "index"
    # 将时间索引重新采样为指定单位（分钟、小时等）
    s.index = s.index.as_unit(unit)

    # 创建一个时间间隔为5分钟的分组器对象
    grouper = Grouper(freq=Minute(5))
    # 期望的结果，按照5分钟分组并取每组的最后一个值
    expect = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    # 使用ohlc方法对时间序列进行5分钟重采样
    result = s.resample("5Min").ohlc()

    # 断言结果的长度与期望的长度相同
    assert len(result) == len(expect)
    # 断言结果的列数为4
    assert len(result.columns) == 4

    # 验证倒数第二行的数据
    xs = result.iloc[-2]
    assert xs["open"] == s.iloc[-6]
    assert xs["high"] == s[-6:-1].max()
    assert xs["low"] == s[-6:-1].min()
    assert xs["close"] == s.iloc[-2]

    # 验证第一行的数据
    xs = result.iloc[0]
    assert xs["open"] == s.iloc[0]
    assert xs["high"] == s[:5].max()
    assert xs["low"] == s[:5].min()
    assert xs["close"] == s.iloc[4]


def test_resample_ohlc_result(unit):
    # GH 12332
    # 创建一个时间索引，从"1-1-2000"到"2-15-2000"，并合并到"4-15-2000"到"5-15-2000"
    index = date_range("1-1-2000", "2-15-2000", freq="h").as_unit(unit)
    index = index.union(date_range("4-15-2000", "5-15-2000", freq="h").as_unit(unit))
    # 创建一个时间序列，索引为上述时间索引，值为索引的位置
    s = Series(range(len(index)), index=index)

    # 对截止到"4-15-2000"的时间序列按照30分钟间隔重采样并应用ohlc方法
    a = s.loc[:"4-15-2000"].resample("30min").ohlc()
    # 断言结果是DataFrame类型
    assert isinstance(a, DataFrame)

    # 对截止到"4-14-2000"的时间序列按照30分钟间隔重采样并应用ohlc方法
    b = s.loc[:"4-14-2000"].resample("30min").ohlc()
    # 断言结果是DataFrame类型
    assert isinstance(b, DataFrame)


def test_resample_ohlc_result_odd_period(unit):
    # GH12348
    # 抛出异常，因为周期是奇数
    # 创建一个时间范围，从"2013-12-30"到"2014-01-07"，按照指定单位重新索引
    rng = date_range("2013-12-30", "2014-01-07").as_unit(unit)
    # 从时间范围中移除指定日期的索引
    index = rng.drop(
        [
            Timestamp("2014-01-01"),
            Timestamp("2013-12-31"),
            Timestamp("2014-01-04"),
            Timestamp("2014-01-05"),
        ]
    )
    # 创建一个DataFrame，索引为上述索引，数据为索引位置
    df = DataFrame(data=np.arange(len(index)), index=index)
    # 对DataFrame进行工作日频率的重采样，并计算均值
    result = df.resample("B").mean()
    # 创建一个期望的DataFrame，重新索引为指定范围的工作日频率
    expected = df.reindex(index=date_range(rng[0], rng[-1], freq="B").as_unit(unit))
    # 断言两个DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_resample_ohlc_dataframe(unit):
    # 创建一个DataFrame，包含价格和成交量数据
    df = (
        DataFrame(
            {
                "PRICE": {
                    Timestamp("2011-01-06 10:59:05", tz=None): 24990,
                    Timestamp("2011-01-06 12:43:33", tz=None): 25499,
                    Timestamp("2011-01-06 12:54:09", tz=None): 25499,
                },
                "VOLUME": {
                    Timestamp("2011-01-06 10:59:05", tz=None): 1500000000,
                    Timestamp("2011-01-06 12:43:33", tz=None): 5000000000,
                    Timestamp("2011-01-06 12:54:09", tz=None): 100000000,
                },
            }
        )
    ).reindex(["VOLUME", "PRICE"], axis=1)
    # 将DataFrame的索引重新索引为指定单位
    df.index = df.index.as_unit(unit)
    # 设置列的名称为 "Cols"
    df.columns.name = "Cols"
    # 对DataFrame进行小时频率的重采样，并应用ohlc方法
    res = df.resample("h").ohlc()
    # 创建期望的DataFrame，分别对价格和成交量进行小时频率的ohlc重采样
    exp = pd.concat(
        [df["VOLUME"].resample("h").ohlc(), df["PRICE"].resample("h").ohlc()],
        axis=1,
        keys=df.columns,
    )
    # 断言期望结果的列名第一级是 "Cols"
    assert exp.columns.names[0] == "Cols"
    # 断言两个DataFrame相等
    tm.assert_frame_equal(exp, res)

    # 修改DataFrame的列名
    df.columns = [["a", "b"], ["c", "d"]]
    # 对修改后的DataFrame进行小时频率的ohlc重采样
    res = df.resample("h").ohlc()
    # 使用元组列表创建多级列索引，定义了DataFrame的列结构
    exp.columns = pd.MultiIndex.from_tuples(
        [
            ("a", "c", "open"),
            ("a", "c", "high"),
            ("a", "c", "low"),
            ("a", "c", "close"),
            ("b", "d", "open"),
            ("b", "d", "high"),
            ("b", "d", "low"),
            ("b", "d", "close"),
        ]
    )
    # 使用测试工具验证DataFrame exp和res是否相等
    tm.assert_frame_equal(exp, res)

    # 当前存在重复的列名，导致测试失败
    # df.columns = ['PRICE', 'PRICE']
def test_resample_reresample(unit):
    # 创建一个日期时间索引，从2005年1月1日到2005年1月10日，根据给定的单位调整频率
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D"
    ).as_unit(unit)
    # 创建一个时间序列，使用随机生成的数据，索引为dti
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    # 将时间序列按照工作日（周一到周五）重新取样，取每个时间段右闭右标签的平均值
    bs = s.resample("B", closed="right", label="right").mean()
    # 对重新取样后的时间序列再次按照8小时间隔取平均值
    result = bs.resample("8h").mean()
    # 断言结果的长度为25
    assert len(result) == 25
    # 断言结果的索引频率是DateOffset类型的小时（8小时）
    assert isinstance(result.index.freq, offsets.DateOffset)
    assert result.index.freq == offsets.Hour(8)


@pytest.mark.parametrize(
    "freq, expected_kwargs",
    [
        ["YE-DEC", {"start": "1990", "end": "2000", "freq": "Y-DEC"}],
        ["YE-JUN", {"start": "1990", "end": "2000", "freq": "Y-JUN"}],
        ["ME", {"start": "1990-01", "end": "2000-01", "freq": "M"}],
    ],
)
def test_resample_timestamp_to_period(
    simple_date_range_series, freq, expected_kwargs, unit
):
    # 创建一个简单的日期范围时间序列
    ts = simple_date_range_series("1/1/1990", "1/1/2000")
    # 将时间序列的索引转换为指定的时间单位
    ts.index = ts.index.as_unit(unit)

    # 对时间序列按照给定频率重新取样，并计算平均值，然后转换为周期（Period）
    result = ts.resample(freq).mean().to_period()
    # 对期望的时间序列按照给定的关键字参数进行重新取样，并计算平均值
    expected = ts.resample(freq).mean()
    expected.index = period_range(**expected_kwargs)
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)


def test_ohlc_5min(unit):
    def _ohlc(group):
        # 如果分组中所有值都是缺失值，则返回包含四个NaN值的数组
        if isna(group).all():
            return np.repeat(np.nan, 4)
        # 否则，返回包含开盘价、最高价、最低价和收盘价的数组
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]

    # 创建一个日期时间索引，从2000年1月1日00:00:00到2000年1月1日05:59:50，每10秒一个时间戳
    rng = date_range("1/1/2000 00:00:00", "1/1/2000 5:59:50", freq="10s").as_unit(unit)
    # 创建一个时间序列，使用随机生成的标准正态分布数据，索引为rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # 对时间序列按照5分钟间隔重新取样，右闭右标签，并生成开高低收四个数据点
    resampled = ts.resample("5min", closed="right", label="right").ohlc()

    # 断言结果的第一个时间戳的值等于时间序列的第一个值
    assert (resampled.loc["1/1/2000 00:00"] == ts.iloc[0]).all()

    # 用_ohlc函数计算时间序列的第1到第31个时间戳的开高低收数据
    exp = _ohlc(ts[1:31])
    assert (resampled.loc["1/1/2000 00:05"] == exp).all()

    # 用_ohlc函数计算时间序列从"1/1/2000 5:55:01"开始到最后一个时间戳的开高低收数据
    exp = _ohlc(ts["1/1/2000 5:55:01":])
    assert (resampled.loc["1/1/2000 6:00:00"] == exp).all()


def test_downsample_non_unique(unit):
    # 创建一个日期范围时间索引，从2000年1月1日到2000年2月29日，按给定单位转换
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    # 将日期范围时间索引重复5次，并生成随机标准正态分布数据，索引为rng2
    rng2 = rng.repeat(5).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)

    # 对时间序列按照每个月（ME）重新取样，并计算平均值
    result = ts.resample("ME").mean()

    # 用groupby函数对时间序列按照月份分组，并计算每个组的平均值
    expected = ts.groupby(lambda x: x.month).mean()
    # 断言结果的长度为2
    assert len(result) == 2
    # 使用近似函数断言结果的第一个值接近于期望结果的第一个月的平均值
    tm.assert_almost_equal(result.iloc[0], expected[1])
    # 使用近似函数断言结果的第二个值接近于期望结果的第二个月的平均值
    tm.assert_almost_equal(result.iloc[1], expected[2])


def test_asfreq_non_unique(unit):
    # 创建一个日期范围时间索引，从2000年1月1日到2000年2月29日，按给定单位转换
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    # 将日期范围时间索引重复2次，并生成随机标准正态分布数据，索引为rng2
    rng2 = rng.repeat(2).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)

    # 尝试对时间序列按照工作日（B）重新索引，应该会引发值错误异常
    msg = "cannot reindex on an axis with duplicate labels"
    with pytest.raises(ValueError, match=msg):
        ts.asfreq("B")


@pytest.mark.parametrize("freq", ["min", "5min", "15min", "30min", "4h", "12h"])
def test_resample_anchored_ticks(freq, unit):
    # 如果固定的时间间隔（如5分钟，4小时）能够整除一天，我们应该在午夜时分“锚定”原点，
    # 这样可以获得规律的时间间隔，而不是从第一个时间戳开始，可能会在期望的时间间隔中间开始
    # （这段代码主要是一段注释，没有实际代码）
    # 生成一个日期范围，从 "1/1/2000 04:00:00" 开始，包含 86400 个时间点（每秒一个），按给定的单位进行分段
    rng = date_range("1/1/2000 04:00:00", periods=86400, freq="s").as_unit(unit)
    
    # 使用长度与日期范围相同的随机正态分布数据创建时间序列
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    
    # 将时间序列的前两个值设置为 NaN，确保结果一致性
    ts[:2] = np.nan  # so results are the same
    
    # 对时间序列进行重采样，按照指定的频率（freq），左闭合，左标签，计算每个时间段的均值
    result = ts[2:].resample(freq, closed="left", label="left").mean()
    
    # 对原始时间序列进行相同的重采样操作，作为预期结果
    expected = ts.resample(freq, closed="left", label="left").mean()
    
    # 使用测试框架验证 result 和 expected 时间序列是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("end", [1, 2])
# 使用 pytest 的参数化装饰器，测试参数 'end' 分别为 1 和 2
def test_resample_single_group(end, unit):
    # 定义一个 lambda 函数 mysum，用于计算数组的和
    mysum = lambda x: x.sum()

    # 创建一个日期范围 rng，从 '2000-1-1' 到 '2000-{end}-10'，频率为每天，并根据 unit 转换单位
    rng = date_range("2000-1-1", f"2000-{end}-10", freq="D").as_unit(unit)
    
    # 创建一个时间序列 ts，其值为随机正态分布数组，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    
    # 比较按 'ME' 月末频率重采样后的和与应用 mysum 函数后的结果是否相等
    tm.assert_series_equal(ts.resample("ME").sum(), ts.resample("ME").apply(mysum))


def test_resample_single_group_std(unit):
    # GH 3849
    # 创建一个时间序列 s，包含值 [30.1, 31.6]，索引为指定的时间戳
    s = Series(
        [30.1, 31.6],
        index=[Timestamp("20070915 15:30:00"), Timestamp("20070915 15:40:00")],
    )
    
    # 将时间戳索引转换为指定单位的时间单位
    s.index = s.index.as_unit(unit)
    
    # 创建期望的时间序列 expected，包含值 [0.75]，索引为指定的日期时间索引
    expected = Series(
        [0.75], index=DatetimeIndex([Timestamp("20070915")], freq="D").as_unit(unit)
    )
    
    # 对时间序列 s 按日重采样，并应用 lambda 函数计算标准差
    result = s.resample("D").apply(lambda x: np.std(x))
    
    # 断言重采样后的结果与期望值 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_resample_offset(unit):
    # GH 31809
    # 创建一个日期范围 rng，从 '1/1/2000 00:00:00' 到 '1/1/2000 02:00'，频率为每秒，并根据 unit 转换单位
    rng = date_range("1/1/2000 00:00:00", "1/1/2000 02:00", freq="s").as_unit(unit)
    
    # 创建一个时间序列 ts，其值为随机正态分布数组，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    
    # 对时间序列 ts 进行重采样，采用均值，设置偏移量为 "2min"，结果为 resampled
    resampled = ts.resample("5min", offset="2min").mean()
    
    # 创建期望的日期范围 exp_rng，从 '12/31/1999 23:57:00' 到 '1/1/2000 01:57'，频率为每5分钟，并根据 unit 转换单位
    exp_rng = date_range("12/31/1999 23:57:00", "1/1/2000 01:57", freq="5min").as_unit(
        unit
    )
    
    # 断言重采样后的结果 resampled 的索引与期望的日期范围 exp_rng 是否相等
    tm.assert_index_equal(resampled.index, exp_rng)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin": "1999-12-31 23:57:00"},
        {"origin": Timestamp("1970-01-01 00:02:00")},
        {"origin": "epoch", "offset": "2m"},
        # '1999-31-12 12:02:00' 的起始时间在这种情况下应该等同于 '1999-12-31 12:02:00'
        {"origin": "1999-12-31 12:02:00"},
        {"offset": "-3m"},
    ],
)
# 使用 pytest 的参数化装饰器，测试参数 'kwargs' 的多种情况
def test_resample_origin(kwargs, unit):
    # GH 31809
    # 创建一个日期范围 rng，从 '2000-01-01 00:00:00' 到 '2000-01-01 02:00'，频率为每秒，并根据 unit 转换单位
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    
    # 创建一个时间序列 ts，其值为随机正态分布数组，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    
    # 创建期望的日期范围 exp_rng，从 '1999-12-31 23:57:00' 到 '2000-01-01 01:57'，频率为每5分钟，并根据 unit 转换单位
    exp_rng = date_range(
        "1999-12-31 23:57:00", "2000-01-01 01:57", freq="5min"
    ).as_unit(unit)
    
    # 对时间序列 ts 进行重采样，采用均值，并传入参数 kwargs，结果为 resampled
    resampled = ts.resample("5min", **kwargs).mean()
    
    # 断言重采样后的结果 resampled 的索引与期望的日期范围 exp_rng 是否相等
    tm.assert_index_equal(resampled.index, exp_rng)


@pytest.mark.parametrize(
    "origin", ["invalid_value", "epch", "startday", "startt", "2000-30-30", object()]
)
# 使用 pytest 的参数化装饰器，测试参数 'origin' 的多种情况
def test_resample_bad_origin(origin, unit):
    # 创建一个日期范围 rng，从 '2000-01-01 00:00:00' 到 '2000-01-01 02:00'，频率为每秒，并根据 unit 转换单位
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    
    # 创建一个时间序列 ts，其值为随机正态分布数组，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    
    # 定义异常消息 msg
    msg = (
        "'origin' should be equal to 'epoch', 'start', 'start_day', "
        "'end', 'end_day' or should be a Timestamp convertible type. Got "
        f"'{origin}' instead."
    )
    
    # 使用 pytest.raises 检查重采样时传入无效的 origin 值是否引发 ValueError 异常，并匹配异常消息 msg
    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", origin=origin)


@pytest.mark.parametrize("offset", ["invalid_value", "12dayys", "2000-30-30", object()])
# 使用 pytest 的参数化装饰器，测试参数 'offset' 的多种情况
def test_resample_bad_offset(offset, unit):
    # 创建一个日期范围 rng，从 '2000-01-01 00:00:00' 到 '2000-01-01 02:00'，频率为每秒，并根据 unit 转换单位
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    
    # 创建一个时间序列 ts，其值为随机正态分布数组，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    
    # 定义异常消息 msg
    msg = f"'offset' should be a Timedelta convertible type. Got '{offset}' instead."
    
    # 使用 pytest.raises 检查重采样时传入无效的 offset 值是否引发 ValueError 异常，并匹配异常消息 msg
    with pytest.raises(ValueError, match=msg):
        ts.resample("
    # 使用 pytest 的上下文管理器来测试是否会抛出 ValueError 异常，并匹配异常信息是否符合指定的消息字符串（msg）。
    with pytest.raises(ValueError, match=msg):
        # 对时间序列（ts）进行重新采样，使用时间间隔 "5min"，并设置偏移量为指定的 offset。
        ts.resample("5min", offset=offset)
def test_resample_origin_prime_freq(unit):
    # GH 31809
    # 定义开始时间和结束时间字符串
    start, end = "2000-10-01 23:30:00", "2000-10-02 00:30:00"
    # 生成时间范围对象，频率为每7分钟，并根据指定的单位进行调整
    rng = date_range(start, end, freq="7min").as_unit(unit)
    # 创建时间序列，使用随机正态分布的数据，索引为生成的时间范围对象
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # 期望的时间范围对象，频率为每17分钟，并根据指定的单位进行调整
    exp_rng = date_range(
        "2000-10-01 23:14:00", "2000-10-02 00:22:00", freq="17min"
    ).as_unit(unit)
    # 对时间序列进行重新采样，采样频率为每17分钟，计算均值
    resampled = ts.resample("17min").mean()
    # 断言重新采样后的时间索引与期望的时间范围对象相等
    tm.assert_index_equal(resampled.index, exp_rng)
    
    # 使用不同的起始原点进行重新采样
    resampled = ts.resample("17min", origin="start_day").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    exp_rng = date_range(
        "2000-10-01 23:30:00", "2000-10-02 00:21:00", freq="17min"
    ).as_unit(unit)
    resampled = ts.resample("17min", origin="start").mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    
    # 使用时间偏移量作为起始原点进行重新采样
    resampled = ts.resample("17min", offset="23h30min").mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample("17min", origin="start_day", offset="23h30min").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    exp_rng = date_range(
        "2000-10-01 23:18:00", "2000-10-02 00:26:00", freq="17min"
    ).as_unit(unit)
    # 使用 Epoch 作为起始原点进行重新采样
    resampled = ts.resample("17min", origin="epoch").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    exp_rng = date_range(
        "2000-10-01 23:24:00", "2000-10-02 00:15:00", freq="17min"
    ).as_unit(unit)
    # 使用指定日期作为起始原点进行重新采样
    resampled = ts.resample("17min", origin="2000-01-01").mean()
    tm.assert_index_equal(resampled.index, exp_rng)


def test_resample_origin_with_tz(unit):
    # GH 31809
    msg = "The origin must have the same timezone as the index."

    # 设置时区
    tz = "Europe/Paris"
    # 生成带时区信息的时间范围对象，频率为每秒，并根据指定的单位进行调整
    rng = date_range(
        "2000-01-01 00:00:00", "2000-01-01 02:00", freq="s", tz=tz
    ).as_unit(unit)
    # 创建时间序列，使用随机正态分布的数据，索引为生成的时间范围对象
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # 期望的带时区信息的时间范围对象，频率为每5分钟，并根据指定的单位进行调整
    exp_rng = date_range(
        "1999-12-31 23:57:00", "2000-01-01 01:57", freq="5min", tz=tz
    ).as_unit(unit)
    # 使用指定的起始原点进行重新采样，采样频率为每5分钟，计算均值
    resampled = ts.resample("5min", origin="1999-12-31 23:57:00+00:00").mean()
    # 断言重新采样后的时间索引与期望的时间范围对象相等
    tm.assert_index_equal(resampled.index, exp_rng)

    # 对于特定情况，使用带时区信息的起始原点应该等效
    resampled = ts.resample("5min", origin="1999-12-31 12:02:00+03:00").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    # 使用 Epoch 作为起始原点，并加上时间偏移量进行重新采样
    resampled = ts.resample("5min", origin="epoch", offset="2m").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    # 使用不合法的起始原点，应抛出值错误异常
    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", origin="12/31/1999 23:57:00").mean()

    # 如果时间序列不带时区信息，起始原点也不应带时区信息
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", origin="12/31/1999 23:57:00+03:00").mean()


def test_resample_origin_epoch_with_tz_day_vs_24h(unit):
    # GH 34474
    # 这个测试函数还未完全实现
    # 定义起始时间和结束时间，格式为带时区的日期时间字符串
    start, end = "2000-10-01 23:30:00+0500", "2000-12-02 00:30:00+0500"
    # 使用给定的起始时间、结束时间和频率生成日期时间范围，并将其转换为指定单位的时间序列
    rng = date_range(start, end, freq="7min").as_unit(unit)
    # 生成指定长度的标准正态分布随机数列
    random_values = np.random.default_rng(2).standard_normal(len(rng))
    # 根据生成的随机数列创建时间序列，使用生成的时间索引
    ts_1 = Series(random_values, index=rng)
    
    # 对时间序列进行重采样，以每天为单位，使用epoch作为起点，并计算每日均值
    result_1 = ts_1.resample("D", origin="epoch").mean()
    # 同样对时间序列进行重采样，使用24小时为单位，epoch作为起点，并计算每日均值
    result_2 = ts_1.resample("24h", origin="epoch").mean()
    # 使用断言来验证两个重采样结果是否相等
    tm.assert_series_equal(result_1, result_2)
    
    # 检查即使在没有时区信息的情况下，使用epoch作为起点的重采样行为是否一致
    ts_no_tz = ts_1.tz_localize(None)
    result_3 = ts_no_tz.resample("D", origin="epoch").mean()
    result_4 = ts_no_tz.resample("24h", origin="epoch").mean()
    # 断言验证不带时区信息的重采样结果与带时区信息的结果在频率上是否相等
    tm.assert_series_equal(result_1, result_3.tz_localize(rng.tz), check_freq=False)
    tm.assert_series_equal(result_1, result_4.tz_localize(rng.tz), check_freq=False)
    
    # 检查在两个不同时区（+2小时和+5小时）下是否会得到类似的结果
    start, end = "2000-10-01 23:30:00+0200", "2000-12-02 00:30:00+0200"
    # 使用给定的起始时间、结束时间和频率生成日期时间范围，并将其转换为指定单位的时间序列
    rng = date_range(start, end, freq="7min").as_unit(unit)
    # 根据同一组随机数列再次创建时间序列，使用生成的时间索引
    ts_2 = Series(random_values, index=rng)
    # 对新生成的时间序列进行重采样，以每天为单位，使用epoch作为起点，并计算每日均值
    result_5 = ts_2.resample("D", origin="epoch").mean()
    # 同样对时间序列进行重采样，使用24小时为单位，epoch作为起点，并计算每日均值
    result_6 = ts_2.resample("24h", origin="epoch").mean()
    # 断言验证不带时区信息的result_1与带时区信息的result_5和result_6是否相等
    tm.assert_series_equal(result_1.tz_localize(None), result_5.tz_localize(None))
    tm.assert_series_equal(result_1.tz_localize(None), result_6.tz_localize(None))
# 定义一个测试函数，用于验证在夏令时(Daylight Saving Time, DST)情况下，使用不同的起始点(origin)进行重采样的行为
def test_resample_origin_with_day_freq_on_dst(unit):
    # 设定时区为 "America/Chicago"
    tz = "America/Chicago"

    # 定义一个内部函数，用于创建时间序列对象
    def _create_series(values, timestamps, freq="D"):
        # 创建一个时间戳索引，每个时间戳都使用指定的时区，频率为给定的频率，并允许模糊匹配
        return Series(
            values,
            index=DatetimeIndex(
                [Timestamp(t, tz=tz) for t in timestamps], freq=freq, ambiguous=True
            ).as_unit(unit),
        )

    # 测试在夏令时背景下，原始(origin)设置为不同值时的行为
    start = Timestamp("2013-11-02", tz=tz)
    end = Timestamp("2013-11-03 23:59", tz=tz)
    # 生成日期范围，频率为每小时
    rng = date_range(start, end, freq="1h").as_unit(unit)
    # 创建时间序列，每个时间点的值为1
    ts = Series(np.ones(len(rng)), index=rng)

    # 期望的结果时间序列，使用内部函数_create_series创建
    expected = _create_series([24.0, 25.0], ["2013-11-02", "2013-11-03"])
    # 遍历不同的原始点(origin)设置，对时间序列进行按天重采样并求和，然后进行结果断言
    for origin in ["epoch", "start", "start_day", start, None]:
        result = ts.resample("D", origin=origin).sum()
        tm.assert_series_equal(result, expected)

    # 测试在夏令时背景下，使用起始点(origin)和偏移量(offset)进行重采样的复杂行为
    start = Timestamp("2013-11-03", tz=tz)
    end = Timestamp("2013-11-03 23:59", tz=tz)
    rng = date_range(start, end, freq="1h").as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)

    # 期望的结果时间序列的时间戳格式
    expected_ts = ["2013-11-02 22:00-05:00", "2013-11-03 22:00-06:00"]
    expected = _create_series([23.0, 2.0], expected_ts)
    # 使用起始点(origin="start")和偏移量(offset="-2h")对时间序列进行按天重采样并求和，然后进行结果断言
    result = ts.resample("D", origin="start", offset="-2h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 22:00-05:00", "2013-11-03 21:00-06:00"]
    expected = _create_series([22.0, 3.0], expected_ts, freq="24h")
    result = ts.resample("24h", origin="start", offset="-2h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 02:00-05:00", "2013-11-03 02:00-06:00"]
    expected = _create_series([3.0, 22.0], expected_ts)
    result = ts.resample("D", origin="start", offset="2h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 23:00-05:00", "2013-11-03 23:00-06:00"]
    expected = _create_series([24.0, 1.0], expected_ts)
    result = ts.resample("D", origin="start", offset="-1h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 01:00-05:00", "2013-11-03 01:00:00-0500"]
    expected = _create_series([1.0, 24.0], expected_ts)
    result = ts.resample("D", origin="start", offset="1h").sum()
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于验证按日重采样时的行为，包括闭合(closed)和标签(label)设置
def test_resample_daily_anchored(unit):
    # 生成日期范围，频率为每分钟
    rng = date_range("1/1/2000 0:00:00", periods=10000, freq="min").as_unit(unit)
    # 创建时间序列，随机数值
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan  # 设置前两个值为NaN，以确保结果一致

    # 对时间序列的子集进行按日重采样，计算均值
    result = ts[2:].resample("D", closed="left", label="left").mean()
    # 对整个时间序列进行按日重采样，计算均值
    expected = ts.resample("D", closed="left", label="left").mean()
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于验证按月周期(period)重采样时的行为
def test_resample_to_period_monthly_buglet(unit):
    # 生成日期范围，频率为每天
    rng = date_range("1/1/2000", "12/31/2000").as_unit(unit)
    # 创建时间序列，随机数值
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    # 对时间序列 `ts` 进行重采样，使用"ME"表示月末，计算均值并转换为周期性
    result = ts.resample("ME").mean().to_period()
    
    # 创建预期的时间索引，从"Jan-2000"到"Dec-2000"，频率为每月("M")
    exp_index = period_range("Jan-2000", "Dec-2000", freq="M")
    
    # 断言重采样后的结果的索引与预期的索引相等
    tm.assert_index_equal(result.index, exp_index)
def test_period_with_agg():
    # aggregate a period resampler with a lambda
    s2 = Series(
        np.random.default_rng(2).integers(0, 5, 50),  # 生成一个包含随机整数的序列，长度为50，取值范围在0到5之间
        index=period_range("2012-01-01", freq="h", periods=50),  # 创建一个时间段索引，频率为小时，从2012-01-01开始，包含50个时间点
        dtype="float64",  # 设置数据类型为浮点数
    )

    expected = s2.to_timestamp().resample("D").mean().to_period()  # 将时间序列转换为时间戳，按天重采样求均值后转换回时间段
    msg = "Resampling with a PeriodIndex is deprecated"  # 警告消息字符串
    with tm.assert_produces_warning(FutureWarning, match=msg):  # 断言产生未来警告，匹配警告消息
        rs = s2.resample("D")  # 对序列按天进行重采样
    result = rs.agg(lambda x: x.mean())  # 使用 lambda 函数对重采样结果进行聚合求均值
    tm.assert_series_equal(result, expected)  # 断言两个序列是否相等


def test_resample_segfault(unit):
    # GH 8573
    # segfaulting in older versions
    all_wins_and_wagers = [
        (1, datetime(2013, 10, 1, 16, 20), 1, 0),
        (2, datetime(2013, 10, 1, 16, 10), 1, 0),
        (2, datetime(2013, 10, 1, 18, 15), 1, 0),
        (2, datetime(2013, 10, 1, 16, 10, 31), 1, 0),
    ]

    df = DataFrame.from_records(
        all_wins_and_wagers, columns=("ID", "timestamp", "A", "B")
    ).set_index("timestamp")  # 从记录列表创建数据帧，设置时间戳列为索引
    df.index = df.index.as_unit(unit)  # 将索引单位设置为指定单位
    msg = "DataFrameGroupBy.resample operated on the grouping columns"  # 警告消息字符串
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生弃用警告，匹配警告消息
        result = df.groupby("ID").resample("5min").sum()  # 按ID分组后，对每组进行5分钟的重采样求和
    msg = "DataFrameGroupBy.apply operated on the grouping columns"  # 警告消息字符串
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生弃用警告，匹配警告消息
        expected = df.groupby("ID").apply(lambda x: x.resample("5min").sum())  # 对每组应用函数，进行5分钟的重采样求和
    tm.assert_frame_equal(result, expected)  # 断言两个数据帧是否相等


def test_resample_dtype_preservation(unit):
    # GH 12202
    # validation tests for dtype preservation

    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W").as_unit(unit),  # 创建日期范围，按指定单位转换
            "group": [1, 1, 2, 2],  # 分组列
            "val": Series([5, 6, 7, 8], dtype="int32"),  # 值列，设置数据类型为int32
        }
    ).set_index("date")  # 设置日期列为索引

    result = df.resample("1D").ffill()  # 按天重采样并前向填充缺失值
    assert result.val.dtype == np.int32  # 断言结果的值列数据类型为int32

    msg = "DataFrameGroupBy.resample operated on the grouping columns"  # 警告消息字符串
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生弃用警告，匹配警告消息
        result = df.groupby("group").resample("1D").ffill()  # 按组进行重采样并前向填充缺失值
    assert result.val.dtype == np.int32  # 断言结果的值列数据类型为int32


def test_resample_dtype_coercion(unit):
    pytest.importorskip("scipy.interpolate")  # 导入scipy.interpolate，若失败则跳过该测试用例

    # GH 16361
    df = {"a": [1, 3, 1, 4]}
    df = DataFrame(df, index=date_range("2017-01-01", "2017-01-04").as_unit(unit))  # 创建数据帧，设置索引为日期范围转换后的单位

    expected = df.astype("float64").resample("h").mean()["a"].interpolate("cubic")  # 转换为浮点数，按小时重采样求均值并立方插值

    result = df.resample("h")["a"].mean().interpolate("cubic")  # 按小时重采样求均值并立方插值
    tm.assert_series_equal(result, expected)  # 断言两个序列是否相等

    result = df.resample("h").mean()["a"].interpolate("cubic")  # 按小时重采样求均值后，对结果列立方插值
    tm.assert_series_equal(result, expected)  # 断言两个序列是否相等


def test_weekly_resample_buglet(unit):
    # #1327
    rng = date_range("1/1/2000", freq="B", periods=20).as_unit(unit)  # 创建工作日日期范围，按指定单位转换
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)  # 创建随机标准正态分布的时间序列

    resampled = ts.resample("W").mean()  # 按周重采样求均值
    expected = ts.resample("W-SUN").mean()  # 按周重采样求均值，每周以周日结束
    # 使用测试工具库中的函数比较两个时间序列是否相等
    tm.assert_series_equal(resampled, expected)
def test_monthly_resample_error(unit):
    # #1451
    # 创建日期范围，从指定日期开始，生成5000个时间点，按给定频率转换为指定单位
    dates = date_range("4/16/2012 20:00", periods=5000, freq="h").as_unit(unit)
    # 创建一个时间序列，使用正态分布的随机数填充，索引为上述生成的日期范围
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    # 进行重新采样为 "ME"（每月末频率）
    ts.resample("ME")


def test_nanosecond_resample_error():
    # GH 12307 - 当使用 pd.tseries.offsets.Nano 作为周期时，值会落在最后一个区间之后的问题
    start = 1443707890427
    exp_start = 1443707890400
    # 生成一个日期范围，起始日期为给定的时间戳转换成日期格式，生成10个时间点，频率为 "100ns"
    indx = date_range(start=pd.to_datetime(start), periods=10, freq="100ns")
    # 创建一个时间序列，索引为上述生成的日期范围，值为序号
    ts = Series(range(len(indx)), index=indx)
    # 对时间序列进行重新采样，以 pd.tseries.offsets.Nano(100) 为周期
    r = ts.resample(pd.tseries.offsets.Nano(100))
    # 对重新采样结果进行均值聚合
    result = r.agg("mean")

    # 生成期望的日期范围，起始日期为给定的期望起始时间戳，生成10个时间点，频率为 "100ns"
    exp_indx = date_range(start=pd.to_datetime(exp_start), periods=10, freq="100ns")
    # 创建一个期望的时间序列，索引为上述生成的期望日期范围，值为序号，并设置数据类型为浮点型
    exp = Series(range(len(exp_indx)), index=exp_indx, dtype=float)

    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, exp)


def test_resample_anchored_intraday(unit):
    # #1471, #1458

    # 生成一个日期范围，从 "1/1/2012" 到 "4/1/2012"，频率为 "100min"，转换为指定单位
    rng = date_range("1/1/2012", "4/1/2012", freq="100min").as_unit(unit)
    # 创建一个数据框，索引为上述生成的日期范围的月份，数据为月份
    df = DataFrame(rng.month, index=rng)

    # 对数据框进行重新采样为 "ME"（每月末频率）并计算均值
    result = df.resample("ME").mean()
    # 期望的数据框重新采样为 "ME"（每月末频率），转换为周期后，转换为时间戳，并设置截止时间
    expected = df.resample("ME").mean().to_period()
    expected = expected.to_timestamp(how="end")
    # 调整索引，加上纳秒精度的微调，再次转换为指定单位
    expected.index += Timedelta(1, "ns") - Timedelta(1, "D")
    expected.index = expected.index.as_unit(unit)._with_freq("infer")
    # 断言期望的索引频率为 "ME"
    assert expected.index.freq == "ME"
    # 比较结果和期望的数据框
    tm.assert_frame_equal(result, expected)

    # 对数据框进行重新采样为 "ME"（每月末频率），左闭合，计算均值
    result = df.resample("ME", closed="left").mean()
    # 将数据框向前偏移1天，重新采样为 "ME"（每月末频率），转换为时间戳，并设置截止时间
    exp = df.shift(1, freq="D").resample("ME").mean().to_period()
    exp = exp.to_timestamp(how="end")

    # 调整索引，加上纳秒精度的微调，再次转换为指定单位
    exp.index = exp.index + Timedelta(1, "ns") - Timedelta(1, "D")
    exp.index = exp.index.as_unit(unit)._with_freq("infer")
    # 断言期望的索引频率为 "ME"
    assert exp.index.freq == "ME"
    # 比较结果和期望的数据框
    tm.assert_frame_equal(result, exp)


def test_resample_anchored_intraday2(unit):
    # 生成一个日期范围，从 "1/1/2012" 到 "4/1/2012"，频率为 "100min"，转换为指定单位
    rng = date_range("1/1/2012", "4/1/2012", freq="100min").as_unit(unit)
    # 创建一个数据框，索引为上述生成的日期范围的月份，数据为月份
    df = DataFrame(rng.month, index=rng)

    # 对数据框进行重新采样为 "QE"（每季末频率），计算均值
    result = df.resample("QE").mean()
    # 期望的数据框重新采样为 "QE"（每季末频率），转换为时间戳，并设置截止时间
    expected = df.resample("QE").mean().to_period()
    expected = expected.to_timestamp(how="end")
    # 调整索引，加上纳秒精度的微调，设置频率为 "QE"
    expected.index += Timedelta(1, "ns") - Timedelta(1, "D")
    expected.index._data.freq = "QE"
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    # 比较结果和期望的数据框
    tm.assert_frame_equal(result, expected)

    # 对数据框进行重新采样为 "QE"（每季末频率），左闭合，计算均值
    expected = df.shift(1, freq="D").resample("QE").mean()
    expected = expected.to_period()
    expected = expected.to_timestamp(how="end")
    # 调整索引，加上纳秒精度的微调，设置频率为 "QE"
    expected.index += Timedelta(1, "ns") - Timedelta(1, "D")
    expected.index._data.freq = "QE"
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    # 比较结果和期望的数据框
    tm.assert_frame_equal(result, expected)


def test_resample_anchored_intraday3(simple_date_range_series, unit):
    # 使用给定的函数生成一个简单的时间序列，索引从 "2012-04-29 23:00" 到 "2012-04-30 5:00"，频率为 "h"
    ts = simple_date_range_series("2012-04-29 23:00", "2012-04-30 5:00", freq="h")
    # 转换时间序列的索引为指定单位
    ts.index = ts.index.as_unit(unit)
    # 对时间序列进行重新采样为 "ME"（每月末频率），计算均值
    resampled = ts.resample("ME").mean()
    # 断言重新采样后的长度为1
    assert len(resampled) == 1
@pytest.mark.parametrize("freq", ["MS", "BMS", "QS-MAR", "YS-DEC", "YS-JUN"])
# 使用 pytest 的 parametrize 装饰器，定义了一个参数化测试函数，测试不同的频率（月初、工作日月初、季初、年底12月和年中6月）
def test_resample_anchored_monthstart(simple_date_range_series, freq, unit):
    # 生成一个简单的日期范围序列
    ts = simple_date_range_series("1/1/2000", "12/31/2002")
    # 将时间序列索引转换为指定的时间单位
    ts.index = ts.index.as_unit(unit)
    # 对时间序列进行重采样，计算频率为 freq 的平均值
    ts.resample(freq).mean()


@pytest.mark.parametrize("label, sec", [[None, 2.0], ["right", "4.2"]])
# 使用 pytest 的 parametrize 装饰器，定义了另一个参数化测试函数，测试不同的标签和秒数
def test_resample_anchored_multiday(label, sec):
    # 当对跨越多天的范围进行重采样时，确保使用起始日期来确定偏移量。
    # 修复一个问题，即一天的时间段不是频率的倍数。
    #
    # 参见：https://github.com/pandas-dev/pandas/issues/8683

    # 创建两个日期范围
    index1 = date_range("2014-10-14 23:06:23.206", periods=3, freq="400ms")
    index2 = date_range("2014-10-15 23:00:00", periods=2, freq="2200ms")
    # 合并这两个日期范围，生成一个新的索引
    index = index1.union(index2)

    # 创建一个随机数据的 Series，索引为合并后的 index
    s = Series(np.random.default_rng(2).standard_normal(5), index=index)

    # 确保左侧闭合的工作方式
    result = s.resample("2200ms", label=label).mean()
    assert result.index[-1] == Timestamp(f"2014-10-15 23:00:{sec}00")


def test_corner_cases(unit):
    # 杂项测试覆盖

    # 创建一个日期范围，频率为分钟，然后将其转换为指定的时间单位
    rng = date_range("1/1/2000", periods=12, freq="min").as_unit(unit)
    # 创建一个随机数据的 Series，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # 对时间序列进行重采样，计算频率为 "5min"，闭合方式为右侧，标签为左侧的平均值
    result = ts.resample("5min", closed="right", label="left").mean()
    # 生成预期的索引
    ex_index = date_range("1999-12-31 23:55", periods=4, freq="5min").as_unit(unit)
    # 断言重采样后的索引与预期索引相等
    tm.assert_index_equal(result.index, ex_index)


def test_corner_cases_date(simple_date_range_series, unit):
    # 对时间序列进行重采样，计算频率为 "ME" 的平均值，然后转换为周期
    ts = simple_date_range_series("2000-04-28", "2000-04-30 11:00", freq="h")
    ts.index = ts.index.as_unit(unit)
    result = ts.resample("ME").mean().to_period()
    # 断言结果长度为1
    assert len(result) == 1
    # 断言结果的索引为指定的周期
    assert result.index[0] == Period("2000-04", freq="M")


def test_anchored_lowercase_buglet(unit):
    # 创建一个时间范围，频率为秒，然后转换为指定的时间单位
    dates = date_range("4/16/2012 20:00", periods=50000, freq="s").as_unit(unit)
    # 创建一个随机数据的 Series，索引为 dates
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    # 对时间序列进行重采样，计算频率为 "d" 的平均值
    ts.resample("d").mean()


def test_upsample_apply_functions(unit):
    # #1596
    # 创建一个日期范围，频率为小时，然后转换为指定的时间单位
    rng = date_range("2012-06-12", periods=4, freq="h").as_unit(unit)

    # 创建一个随机数据的 Series，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # 对时间序列进行重采样，计算频率为 "20min" 的平均值和总和
    result = ts.resample("20min").aggregate(["mean", "sum"])
    # 断言结果是一个 DataFrame 对象
    assert isinstance(result, DataFrame)


def test_resample_not_monotonic(unit):
    # 创建一个日期范围，频率为小时，然后转换为指定的时间单位
    rng = date_range("2012-06-12", periods=200, freq="h").as_unit(unit)
    # 创建一个随机数据的 Series，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # 打乱时间序列的顺序
    ts = ts.take(np.random.default_rng(2).permutation(len(ts)))

    # 对时间序列进行重采样，计算频率为 "D" 的总和
    result = ts.resample("D").sum()
    # 生成预期结果，对原时间序列排序后再进行重采样
    exp = ts.sort_index().resample("D").sum()
    # 断言重采样后的结果与预期结果相等
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    "dtype",
    [
        # 列表包含多个元素，每个元素是一个字符串或者一个参数化测试（pytest.param）
        "int64",  # 字符串元素，表示数据类型 int64
        "int32",  # 字符串元素，表示数据类型 int32
        "float64",  # 字符串元素，表示数据类型 float64
        pytest.param(
            "float32",  # 参数化测试的参数，表示数据类型 float32
            marks=pytest.mark.xfail(  # 使用 pytest.mark.xfail 标记该测试为预期失败
                reason="Empty groups cause x.mean() to return float64"  # 失败的原因说明
            ),
        ),
    ],
# 定义一个测试函数，用于测试 resample 方法中的 bug 1688
def test_resample_median_bug_1688(dtype, unit):
    # GH#55958
    # 创建一个 DatetimeIndex 对象，包含两个日期时间
    dti = DatetimeIndex(
        [datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 5, 0)]
    ).as_unit(unit)
    # 创建一个 DataFrame 对象，包含两行数据，索引为 dti，数据类型为 dtype
    df = DataFrame(
        [1, 2],
        index=dti,
        dtype=dtype,
    )

    # 对 df 进行分钟级重采样，并使用 lambda 函数计算均值，将结果与期望值进行比较
    result = df.resample("min").apply(lambda x: x.mean())
    exp = df.asfreq("min")
    tm.assert_frame_equal(result, exp)

    # 对 df 进行分钟级重采样，并计算中位数，将结果与期望值进行比较
    result = df.resample("min").median()
    tm.assert_frame_equal(result, exp)


# 定义一个测试函数，用于测试 lambda 函数在时间序列 resample 中的使用
def test_how_lambda_functions(simple_date_range_series, unit):
    # 创建一个简单的时间序列对象 ts
    ts = simple_date_range_series("1/1/2000", "4/1/2000")
    # 将 ts 的索引转换为指定的时间单位
    ts.index = ts.index.as_unit(unit)

    # 对 ts 进行 ME 级别的重采样，并使用 lambda 函数计算均值，将结果与期望值进行比较
    result = ts.resample("ME").apply(lambda x: x.mean())
    exp = ts.resample("ME").mean()
    tm.assert_series_equal(result, exp)

    # 计算 ts 的 ME 级别重采样后的均值，并为结果命名为 "foo"
    foo_exp = ts.resample("ME").mean()
    foo_exp.name = "foo"
    # 计算 ts 的 ME 级别重采样后的标准差，并为结果命名为 "bar"
    bar_exp = ts.resample("ME").std()
    bar_exp.name = "bar"

    # 对 ts 进行 ME 级别的重采样，并同时应用两个 lambda 函数计算均值和标准差，将结果与期望值进行比较
    result = ts.resample("ME").apply([lambda x: x.mean(), lambda x: x.std(ddof=1)])
    result.columns = ["foo", "bar"]
    tm.assert_series_equal(result["foo"], foo_exp)
    tm.assert_series_equal(result["bar"], bar_exp)

    # 对 ts 进行 ME 级别的重采样，并使用字典形式的聚合函数，将结果与期望值进行比较，不检查名称匹配
    result = ts.resample("ME").aggregate(
        {"foo": lambda x: x.mean(), "bar": lambda x: x.std(ddof=1)}
    )
    tm.assert_series_equal(result["foo"], foo_exp, check_names=False)
    tm.assert_series_equal(result["bar"], bar_exp, check_names=False)


# 定义一个测试函数，用于测试在不等时间间隔下的重采样
def test_resample_unequal_times(unit):
    # #1772
    # 定义起始和结束时间
    start = datetime(1999, 3, 1, 5)
    # 结束时间小于起始时间的情况，创建时间范围为每 30 分钟的日期时间索引，并按指定单位转换
    end = datetime(2012, 7, 31, 4)
    bad_ind = date_range(start, end, freq="30min").as_unit(unit)
    # 创建一个 DataFrame 对象，包含一列名称为 "close" 的数据，索引为 bad_ind
    df = DataFrame({"close": 1}, index=bad_ind)

    # 对 df 进行 "YS" 级别的重采样，并求和
    df.resample("YS").sum()


# 定义一个测试函数，用于测试重采样的一致性
def test_resample_consistency(unit):
    # GH 6418
    # 测试使用 bfill / limit / reindex 方法的重采样一致性

    # 创建一个包含四个 30 分钟频率的日期时间索引
    i30 = date_range("2002-02-02", periods=4, freq="30min").as_unit(unit)
    # 创建一个 Series 对象 s，包含四个浮点数，并使用 i30 作为索引
    s = Series(np.arange(4.0), index=i30)
    # 将第三个元素设置为 NaN
    s.iloc[2] = np.nan

    # 使用重采样方法将时间频率提高为每 10 分钟，使用 bfill 方法填充缺失值
    i10 = date_range(i30[0], i30[-1], freq="10min").as_unit(unit)

    s10 = s.reindex(index=i10, method="bfill")
    s10_2 = s.reindex(index=i10, method="bfill", limit=2)
    with tm.assert_produces_warning(FutureWarning):
        rl = s.reindex_like(s10, method="bfill", limit=2)
    r10_2 = s.resample("10Min").bfill(limit=2)
    r10 = s.resample("10Min").bfill()

    # 比较 s10_2, r10, r10_2, rl 四个对象，验证它们是否相等
    tm.assert_series_equal(s10_2, r10)
    tm.assert_series_equal(s10_2, r10_2)
    tm.assert_series_equal(s10_2, rl)


# 定义日期时间列表 dates1，包含多个 datetime 对象
dates1: list[DatetimeNaTType] = [
    datetime(2014, 10, 1),
    datetime(2014, 9, 3),
    datetime(2014, 11, 5),
    datetime(2014, 9, 5),
    datetime(2014, 10, 8),
    datetime(2014, 7, 15),
]

# 定义日期时间列表 dates2，包含 dates1 的组合，并插入 NaT 值
dates2: list[DatetimeNaTType] = (
    dates1[:2] + [pd.NaT] + dates1[2:4] + [pd.NaT] + dates1[4:]
)
# 定义日期时间列表 dates3，将 dates1 列表的首尾添加 NaT 值
dates3 = [pd.NaT] + dates1 + [pd.NaT]
@pytest.mark.parametrize("dates", [dates1, dates2, dates3])
def test_resample_timegrouper(dates, unit):
    # 使用给定的日期列表和时间单位创建 DatetimeIndex 对象
    dates = DatetimeIndex(dates).as_unit(unit)
    # 创建 DataFrame，其中包含列'A'为日期，列'B'为对应长度的序列
    df = DataFrame({"A": dates, "B": np.arange(len(dates))})
    # 对 DataFrame 按日期列'A'进行索引设置，并按'ME'频率重采样后计数
    result = df.set_index("A").resample("ME").count()
    # 创建预期的 DatetimeIndex 对象，并按'ME'频率设置索引
    exp_idx = DatetimeIndex(
        ["2014-07-31", "2014-08-31", "2014-09-30", "2014-10-31", "2014-11-30"],
        freq="ME",
        name="A",
    ).as_unit(unit)
    # 创建预期的 DataFrame，包含列'B'，其值为 [1, 0, 2, 2, 1]，索引为预期的 DatetimeIndex 对象
    expected = DataFrame({"B": [1, 0, 2, 2, 1]}, index=exp_idx)
    # 如果 DataFrame 中列'A'存在缺失值，清除预期 DataFrame 的频率信息
    if df["A"].isna().any():
        expected.index = expected.index._with_freq(None)
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 按日期列'A'进行分组，并计数每个分组
    result = df.groupby(Grouper(freq="ME", key="A")).count()
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dates", [dates1, dates2, dates3])
def test_resample_timegrouper2(dates, unit):
    # 使用给定的日期列表和时间单位创建 DatetimeIndex 对象
    dates = DatetimeIndex(dates).as_unit(unit)

    # 创建 DataFrame，包含列'A'为日期，列'B'为对应长度的序列，列'C'为对应长度的序列
    df = DataFrame({"A": dates, "B": np.arange(len(dates)), "C": np.arange(len(dates))})
    # 对 DataFrame 按日期列'A'进行索引设置，并按'ME'频率重采样后计数
    result = df.set_index("A").resample("ME").count()

    # 创建预期的 DatetimeIndex 对象，并按'ME'频率设置索引
    exp_idx = DatetimeIndex(
        ["2014-07-31", "2014-08-31", "2014-09-30", "2014-10-31", "2014-11-30"],
        freq="ME",
        name="A",
    ).as_unit(unit)
    # 创建预期的 DataFrame，包含列'B'和'C'，其值分别为 [1, 0, 2, 2, 1]，索引为预期的 DatetimeIndex 对象
    expected = DataFrame(
        {"B": [1, 0, 2, 2, 1], "C": [1, 0, 2, 2, 1]},
        index=exp_idx,
        columns=["B", "C"],
    )
    # 如果 DataFrame 中列'A'存在缺失值，清除预期 DataFrame 的频率信息
    if df["A"].isna().any():
        expected.index = expected.index._with_freq(None)
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 按日期列'A'进行分组，并计数每个分组
    result = df.groupby(Grouper(freq="ME", key="A")).count()
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_resample_nunique(unit):
    # 创建包含日期和ID信息的 DataFrame
    df = DataFrame(
        {
            "ID": {
                Timestamp("2015-06-05 00:00:00"): "0010100903",
                Timestamp("2015-06-08 00:00:00"): "0010150847",
            },
            "DATE": {
                Timestamp("2015-06-05 00:00:00"): "2015-06-05",
                Timestamp("2015-06-08 00:00:00"): "2015-06-08",
            },
        }
    )
    # 将 DataFrame 的索引转换为指定的时间单位
    df.index = df.index.as_unit(unit)
    # 按'D'频率对 DataFrame 进行重采样
    r = df.resample("D")
    # 按'D'频率对 DataFrame 进行分组
    g = df.groupby(Grouper(freq="D"))
    # 创建预期的 Series，包含按'D'频率分组后每组'ID'列的唯一值数量
    expected = df.groupby(Grouper(freq="D")).ID.apply(lambda x: x.nunique())
    # 断言预期 Series 的名称为'ID'
    assert expected.name == "ID"

    # 对于重采样对象 r 和分组对象 g，分别计算'ID'列的唯一值数量，并使用测试工具比较结果
    for t in [r, g]:
        result = t.ID.nunique()
        tm.assert_series_equal(result, expected)

    # 对 DataFrame 中'ID'列按'D'频率进行重采样，并计算唯一值数量，使用测试工具比较结果
    result = df.ID.resample("D").nunique()
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 中'ID'列按'D'频率进行分组，并计算唯一值数量，使用测试工具比较结果
    result = df.ID.groupby(Grouper(freq="D")).nunique()
    tm.assert_series_equal(result, expected)


def test_resample_nunique_preserves_column_level_names(unit):
    # 见 GitHub 问题 #23222
    # 创建包含随机数据的 DataFrame，列名为['A', 'B', 'C', 'D']，行索引为日期范围
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=5, freq="D"),
    ).abs()
    # 将 DataFrame 的行索引转换为指定的时间单位
    df.index = df.index.as_unit(unit)
    # 将 DataFrame 的列名设为多级索引，级别为['lev0', 'lev1']
    df.columns = pd.MultiIndex.from_arrays(
        [df.columns.tolist()] * 2, names=["lev0", "lev1"]
    )
    # 对 DataFrame 按'1h'频率进行重采样，并计算唯一值数量
    result = df.resample("1h").nunique()
    # 使用测试工具 `tm` 的断言函数 `assert_index_equal` 检查 DataFrame `df` 的列索引是否与结果 DataFrame `result` 的列索引相等。
    tm.assert_index_equal(df.columns, result.columns)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.nunique(),  # 使用函数 lambda x: x.nunique() 对数据进行去重计数
        lambda x: x.agg(Series.nunique),  # 使用 agg 方法调用 Series 类的 nunique 方法进行聚合操作
        lambda x: x.agg("nunique"),  # 使用 agg 方法调用字符串 "nunique" 进行聚合操作
    ],
    ids=["nunique", "series_nunique", "nunique_str"],  # 对不同函数设定标识符以便区分测试结果
)
def test_resample_nunique_with_date_gap(func, unit):
    # GH 13453
    # 由于所有元素都是唯一的，这些操作应该得到相同的结果
    index = date_range("1-1-2000", "2-15-2000", freq="h").as_unit(unit)  # 创建一个日期时间索引
    index2 = date_range("4-15-2000", "5-15-2000", freq="h").as_unit(unit)  # 创建另一个日期时间索引
    index3 = index.append(index2)  # 将两个索引合并
    s = Series(range(len(index3)), index=index3, dtype="int64")  # 创建一个整数序列，使用合并后的索引
    r = s.resample("ME")  # 对序列进行 "ME" 频率的重采样
    result = r.count()  # 计算重采样后的序列计数
    expected = func(r)  # 对重采样后的序列应用指定的函数
    tm.assert_series_equal(result, expected)  # 断言重采样后的结果与预期结果相等


def test_resample_group_info(unit):
    # GH10914

    # 使用固定种子以保证唯一性
    n = 100
    k = 10
    prng = np.random.default_rng(2)

    dr = date_range(start="2015-08-27", periods=n // 10, freq="min").as_unit(unit)  # 创建一个日期时间索引
    ts = Series(prng.integers(0, n // k, n).astype("int64"), index=prng.choice(dr, n))  # 创建一个随机整数序列

    left = ts.resample("30min").nunique()  # 对序列进行 "30min" 频率的重采样，并计算唯一值数量
    ix = date_range(start=ts.index.min(), end=ts.index.max(), freq="30min").as_unit(
        unit
    )  # 创建一个 "30min" 频率的日期时间索引

    vals = ts.values  # 获取序列的值
    bins = np.searchsorted(ix.values, ts.index, side="right")  # 在 ix 值中搜索序列索引的位置

    sorter = np.lexsort((vals, bins))  # 使用值和 bins 进行排序
    vals, bins = vals[sorter], bins[sorter]

    mask = np.r_[True, vals[1:] != vals[:-1]]  # 创建一个掩码，标记值变化的位置
    mask |= np.r_[True, bins[1:] != bins[:-1]]  # 标记 bins 变化的位置

    arr = np.bincount(bins[mask] - 1, minlength=len(ix)).astype("int64", copy=False)  # 计算每个 bin 中的数量
    right = Series(arr, index=ix)  # 创建一个包含计数的 Series

    tm.assert_series_equal(left, right)  # 断言重采样后的结果与预期结果相等


def test_resample_size(unit):
    n = 10000
    dr = date_range("2015-09-19", periods=n, freq="min").as_unit(unit)  # 创建一个日期时间索引
    ts = Series(
        np.random.default_rng(2).standard_normal(n),
        index=np.random.default_rng(2).choice(dr, n),
    )  # 创建一个正态分布的随机数序列

    left = ts.resample("7min").size()  # 对序列进行 "7min" 频率的重采样，并计算每个区间的大小
    ix = date_range(start=left.index.min(), end=ts.index.max(), freq="7min").as_unit(
        unit
    )  # 创建一个 "7min" 频率的日期时间索引

    bins = np.searchsorted(ix.values, ts.index.values, side="right")  # 在 ix 值中搜索序列索引的位置
    val = np.bincount(bins, minlength=len(ix) + 1)[1:].astype("int64", copy=False)  # 计算每个 bin 中的数量

    right = Series(val, index=ix)  # 创建一个包含计数的 Series
    tm.assert_series_equal(left, right)  # 断言重采样后的结果与预期结果相等


def test_resample_across_dst():
    # 测试对一个包含夏令时变更前后值的 DatetimeIndex 进行重采样
    # Issue: 14682

    # 我们将从这个 DatetimeIndex 开始
    # （注意夏令时在 03:00+02:00 -> 02:00+01:00 发生变更）
    df1 = DataFrame([1477786980, 1477790580], columns=["ts"])
    dti1 = DatetimeIndex(
        pd.to_datetime(df1.ts, unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Madrid")
    )

    # 预期的重采样后的 DatetimeIndex
    # （注意夏令时变更后的时间）
    df2 = DataFrame([1477785600, 1477789200], columns=["ts"])
    # 使用 pd.to_datetime 将 df2.ts 列转换为时间戳，并设定单位为秒
    # 调用 dt.tz_localize 方法将时间戳本地化为 UTC 时区
    # 调用 dt.tz_convert 方法将 UTC 时区时间转换为 Europe/Madrid 时区时间
    dti2 = DatetimeIndex(
        pd.to_datetime(df2.ts, unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Madrid"),
        freq="h",
    )
    
    # 创建 DataFrame 对象，传入包含两个元素的列表作为数据，使用 dti1 作为索引
    df = DataFrame([5, 5], index=dti1)
    
    # 对 DataFrame df 进行重采样，按照每小时 ('h') 的频率进行汇总求和
    result = df.resample(rule="h").sum()
    
    # 创建 DataFrame 对象，传入包含两个元素的列表作为数据，使用 dti2 作为索引
    expected = DataFrame([5, 5], index=dti2)
    
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_groupby_with_dst_time_change(unit):
    # GH 24972
    # 创建一个 DatetimeIndex 对象，包含两个具有纳秒精度的时间戳，时区为 UTC，然后转换时区为 America/Chicago，并按指定的单位重新分组
    index = (
        DatetimeIndex([1478064900001000000, 1480037118776792000], tz="UTC")
        .tz_convert("America/Chicago")
        .as_unit(unit)
    )

    # 创建一个包含两行数据的 DataFrame，索引使用上面创建的时间索引对象
    df = DataFrame([1, 2], index=index)
    
    # 对 DataFrame 进行按天分组，取每组的最后一个值
    result = df.groupby(Grouper(freq="1d")).last()
    
    # 创建一个预期的 DatetimeIndex 对象，包含从 "2016-11-02" 到 "2016-11-24" 的日期范围，时区为 America/Chicago，并按指定的单位重新分组
    expected_index_values = date_range(
        "2016-11-02", "2016-11-24", freq="d", tz="America/Chicago"
    ).as_unit(unit)

    # 创建一个包含预期结果的 DataFrame，索引使用上面创建的预期时间索引对象
    index = DatetimeIndex(expected_index_values)
    expected = DataFrame([1.0] + ([np.nan] * 21) + [2.0], index=index)
    
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_resample_dst_anchor(unit):
    # 5172
    # 创建一个 DatetimeIndex 对象，包含一个特定的日期时间对象，时区为 US/Eastern，并按指定的单位重新分组
    dti = DatetimeIndex([datetime(2012, 11, 4, 23)], tz="US/Eastern").as_unit(unit)
    
    # 创建一个包含一个数据值的 DataFrame，索引使用上面创建的时间索引对象
    df = DataFrame([5], index=dti)

    # 创建一个新的 DatetimeIndex 对象，包含索引的日期部分，频率为天
    dti = DatetimeIndex(df.index.normalize(), freq="D").as_unit(unit)
    expected = DataFrame([5], index=dti)
    
    # 使用 pytest 的 assert_frame_equal 函数比较 resample 后的 DataFrame 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(df.resample(rule="D").sum(), expected)
    
    # 对 DataFrame 进行 resample 操作，频率为每月的开始
    df.resample(rule="MS").sum()
    
    # 使用 pytest 的 assert_frame_equal 函数比较 resample 后的 DataFrame 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(
        df.resample(rule="MS").sum(),
        DataFrame(
            [5],
            index=DatetimeIndex(
                [datetime(2012, 11, 1)], tz="US/Eastern", freq="MS"
            ).as_unit(unit),
        ),
    )


def test_resample_dst_anchor2(unit):
    # 创建一个 DatetimeIndex 对象，包含从 "2013-09-30" 到 "2013-11-02" 的日期范围，频率为每30分钟，时区为 Europe/Paris，并按指定的单位重新分组
    dti = date_range(
        "2013-09-30", "2013-11-02", freq="30Min", tz="Europe/Paris"
    ).as_unit(unit)
    
    # 创建一个包含多列数据的 DataFrame，列名为 "a", "b", "c"，索引使用上面创建的时间索引对象，数据类型为 int64
    values = range(dti.size)
    df = DataFrame({"a": values, "b": values, "c": values}, index=dti, dtype="int64")
    
    # 定义每列的聚合方式
    how = {"a": "min", "b": "max", "c": "count"}

    # 对 DataFrame 进行 resample 操作，频率为每周的星期一
    rs = df.resample("W-MON")
    result = rs.agg(how)[["a", "b", "c"]]
    
    # 创建一个预期结果的 DataFrame，索引使用 date_range 创建的日期范围对象
    expected = DataFrame(
        {
            "a": [0, 48, 384, 720, 1056, 1394],
            "b": [47, 383, 719, 1055, 1393, 1586],
            "c": [48, 336, 336, 336, 338, 193],
        },
        index=date_range(
            "9/30/2013", "11/4/2013", freq="W-MON", tz="Europe/Paris"
        ).as_unit(unit),
    )
    
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(
        result,
        expected,
        "W-MON Frequency",
    )

    # 对 DataFrame 进行 resample 操作，频率为每两周的星期一
    rs2 = df.resample("2W-MON")
    result2 = rs2.agg(how)[["a", "b", "c"]]
    
    # 创建一个预期结果的 DataFrame，索引使用 date_range 创建的日期范围对象
    expected2 = DataFrame(
        {
            "a": [0, 48, 720, 1394],
            "b": [47, 719, 1393, 1586],
            "c": [48, 672, 674, 193],
        },
        index=date_range(
            "9/30/2013", "11/11/2013", freq="2W-MON", tz="Europe/Paris"
        ).as_unit(unit),
    )
    
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(
        result2,
        expected2,
        "2W-MON Frequency",
    )

    # 对 DataFrame 进行 resample 操作，频率为每月的开始
    rs3 = df.resample("MS")
    result3 = rs3.agg(how)[["a", "b", "c"]]
    
    # 创建一个预期结果的 DataFrame，索引使用 date_range 创建的日期范围对象
    expected3 = DataFrame(
        {"a": [0, 48, 1538], "b": [47, 1537, 1586], "c": [48, 1490, 49]},
        index=date_range("9/1/2013", "11/1/2013", freq="MS", tz="Europe/Paris").as_unit(
            unit
        ),
    )
    
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(
        result3,
        expected3,
        "MS Frequency",
    )

    # 对 DataFrame 进行 resample 操作，频率为每两个月的开始
    rs4 = df.resample("2MS")
    result4 = rs4.agg(how)[["a", "b", "c"]]
    # 创建预期的 DataFrame，包含指定的列和索引，使用指定的时间频率和时区
    expected4 = DataFrame(
        {"a": [0, 1538], "b": [1537, 1586], "c": [1538, 49]},
        index=date_range(
            "9/1/2013", "11/1/2013", freq="2MS", tz="Europe/Paris"
        ).as_unit(unit),
    )
    # 使用测试工具验证 result4 和 expected4 是否相等，输出信息为 "2MS Frequency"
    tm.assert_frame_equal(
        result4,
        expected4,
        "2MS Frequency",
    )

    # 从 df 中选取指定日期范围内的数据
    df_daily = df["10/26/2013":"10/29/2013"]
    # 对每日数据进行重采样为每日频率
    rs_d = df_daily.resample("D")
    # 对重采样后的数据进行聚合操作，计算每列的最小值、最大值和计数
    result_d = rs_d.agg({"a": "min", "b": "max", "c": "count"})[["a", "b", "c"]]
    # 创建预期的每日频率 DataFrame，包含指定的列和索引，使用指定的时间频率和时区
    expected_d = DataFrame(
        {
            "a": [1248, 1296, 1346, 1394],
            "b": [1295, 1345, 1393, 1441],
            "c": [48, 50, 48, 48],
        },
        index=date_range(
            "10/26/2013", "10/29/2013", freq="D", tz="Europe/Paris"
        ).as_unit(unit),
    )
    # 使用测试工具验证 result_d 和 expected_d 是否相等，输出信息为 "D Frequency"
    tm.assert_frame_equal(
        result_d,
        expected_d,
        "D Frequency",
    )
def test_downsample_across_dst(unit):
    # GH 8531
    # 设置时区为 "Europe/Berlin"
    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    # 创建一个 datetime 对象，表示日期为 2014 年 10 月 26 日
    dt = datetime(2014, 10, 26)
    # 根据给定日期创建一个时间范围，每 2 小时一个时间点，作为指定单元的日期范围
    dates = date_range(dt.astimezone(tz), periods=4, freq="2h").as_unit(unit)
    # 创建一个 Series 对象，每个时间点的值为 5，然后对其进行小时重采样求均值
    result = Series(5, index=dates).resample("h").mean()
    # 创建一个预期的 Series 对象，包含特定时间范围内的均值和 NaN 值
    expected = Series(
        [5.0, np.nan] * 3 + [5.0],
        index=date_range(dt.astimezone(tz), periods=7, freq="h").as_unit(unit),
    )
    # 使用测试框架进行结果的比较
    tm.assert_series_equal(result, expected)


def test_downsample_across_dst_weekly(unit):
    # GH 9119, GH 21459
    # 创建一个 DataFrame 对象，指定索引为按周重采样后的日期范围
    df = DataFrame(
        index=DatetimeIndex(
            ["2017-03-25", "2017-03-26", "2017-03-27", "2017-03-28", "2017-03-29"],
            tz="Europe/Amsterdam",
        ).as_unit(unit),
        data=[11, 12, 13, 14, 15],
    )
    # 对 DataFrame 进行周重采样并求和
    result = df.resample("1W").sum()
    # 创建预期的 DataFrame 对象，包含按周重采样后的期望值
    expected = DataFrame(
        [23, 42],
        index=DatetimeIndex(
            ["2017-03-26", "2017-04-02"], tz="Europe/Amsterdam", freq="W"
        ).as_unit(unit),
    )
    # 使用测试框架进行结果的比较
    tm.assert_frame_equal(result, expected)


def test_downsample_across_dst_weekly_2(unit):
    # GH 9119, GH 21459
    # 创建一个索引对象，表示在特定时区和频率下的日期范围
    idx = date_range("2013-04-01", "2013-05-01", tz="Europe/London", freq="h").as_unit(
        unit
    )
    # 创建一个 Series 对象，索引为上述日期范围，数据类型为 np.float64
    s = Series(index=idx, dtype=np.float64)
    # 对 Series 进行周重采样并求均值
    result = s.resample("W").mean()
    # 创建预期的 Series 对象，包含按周重采样后的期望值
    expected = Series(
        index=date_range("2013-04-07", freq="W", periods=5, tz="Europe/London").as_unit(
            unit
        ),
        dtype=np.float64,
    )
    # 使用测试框架进行结果的比较
    tm.assert_series_equal(result, expected)


def test_downsample_dst_at_midnight(unit):
    # GH 25758
    # 创建一个起始和结束时间点的 datetime 对象列表，频率为每小时
    start = datetime(2018, 11, 3, 12)
    end = datetime(2018, 11, 5, 12)
    index = date_range(start, end, freq="1h").as_unit(unit)
    # 将索引对象设定为 UTC 时区后转换为 "America/Havana" 时区
    index = index.tz_localize("UTC").tz_convert("America/Havana")
    # 创建一个数据列表，作为 DataFrame 的数据，索引为上述索引对象
    data = list(range(len(index)))
    dataframe = DataFrame(data, index=index)
    # 对 DataFrame 进行按日分组并求均值
    result = dataframe.groupby(Grouper(freq="1D")).mean()

    # 创建预期的 DataFrame 对象，包含特定日期范围内的均值
    dti = date_range("2018-11-03", periods=3).tz_localize(
        "America/Havana", ambiguous=True
    )
    dti = DatetimeIndex(dti, freq="D").as_unit(unit)
    expected = DataFrame([7.5, 28.0, 44.5], index=dti)
    # 使用测试框架进行结果的比较
    tm.assert_frame_equal(result, expected)


def test_resample_with_nat(unit):
    # GH 13020
    # 创建一个 DatetimeIndex 对象，包含 NaT 和指定日期时间字符串
    index = DatetimeIndex(
        [
            pd.NaT,
            "1970-01-01 00:00:00",
            pd.NaT,
            "1970-01-01 00:00:01",
            "1970-01-01 00:00:02",
        ]
    ).as_unit(unit)
    # 创建一个 DataFrame 对象，数据为列表 [2, 3, 5, 7, 11]，索引为上述 DatetimeIndex 对象
    frame = DataFrame([2, 3, 5, 7, 11], index=index)

    # 创建一个 DatetimeIndex 对象，包含指定日期时间字符串，作为预期的索引
    index_1s = DatetimeIndex(
        ["1970-01-01 00:00:00", "1970-01-01 00:00:01", "1970-01-01 00:00:02"]
    ).as_unit(unit)
    # 创建一个预期的 DataFrame 对象，数据为 [3.0, 7.0, 11.0]，索引为上述 DatetimeIndex 对象
    frame_1s = DataFrame([3.0, 7.0, 11.0], index=index_1s)
    # 使用测试框架进行结果的比较
    tm.assert_frame_equal(frame.resample("1s").mean(), frame_1s)

    # 创建一个 DatetimeIndex 对象，包含指定日期时间字符串，作为预期的索引
    index_2s = DatetimeIndex(["1970-01-01 00:00:00", "1970-01-01 00:00:02"]).as_unit(
        unit
    )
    # 创建一个预期的 DataFrame 对象，数据为 [5.0, 11.0]，索引为上述 DatetimeIndex 对象
    frame_2s = DataFrame([5.0, 11.0], index=index_2s)
    # 使用测试框架进行结果的比较
    tm.assert_frame_equal(frame.resample("2s").mean(), frame_2s)
    # 创建一个时间索引对象，包含一个时间戳 "1970-01-01 00:00:00"，并按照指定的时间单位进行转换
    index_3s = DatetimeIndex(["1970-01-01 00:00:00"]).as_unit(unit)
    
    # 创建一个数据帧，其中包含一个浮点数值 7.0，使用上面创建的时间索引作为索引
    frame_3s = DataFrame([7.0], index=index_3s)
    
    # 使用测试框架中的函数验证对原始数据帧进行“3秒”时间间隔重新采样后的平均值与预期的数据帧 frame_3s 是否相等
    tm.assert_frame_equal(frame.resample("3s").mean(), frame_3s)
    
    # 使用测试框架中的函数验证对原始数据帧进行“60秒”时间间隔重新采样后的平均值与预期的数据帧 frame_3s 是否相等
    tm.assert_frame_equal(frame.resample("60s").mean(), frame_3s)
def test_resample_datetime_values(unit):
    # GH 13119
    # 校验在重采样引入 NaT 值时 datetime 类型得以保留

    # 创建日期列表和对应的 DataFrame
    dates = [datetime(2016, 1, 15), datetime(2016, 1, 19)]
    df = DataFrame({"timestamp": dates}, index=dates)
    # 将索引转换为指定单位
    df.index = df.index.as_unit(unit)

    # 期望的结果 Series
    exp = Series(
        [datetime(2016, 1, 15), pd.NaT, datetime(2016, 1, 19)],
        index=date_range("2016-01-15", periods=3, freq="2D").as_unit(unit),
        name="timestamp",
    )

    # 进行重采样并验证结果
    res = df.resample("2D").first()["timestamp"]
    tm.assert_series_equal(res, exp)
    res = df["timestamp"].resample("2D").first()
    tm.assert_series_equal(res, exp)


def test_resample_apply_with_additional_args(unit):
    # GH 14615
    # 创建时间序列和设定索引名
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    series = Series(range(len(index)), index=index)
    series.index.name = "index"

    # 定义应用于数据的函数 f，并将索引转换为指定单位
    def f(data, add_arg):
        return np.mean(data) * add_arg

    series.index = series.index.as_unit(unit)

    # 进行重采样应用函数并验证结果
    multiplier = 10
    result = series.resample("D").apply(f, multiplier)
    expected = series.resample("D").mean().multiply(multiplier)
    tm.assert_series_equal(result, expected)

    # 测试作为关键字参数传递的情况
    result = series.resample("D").apply(f, add_arg=multiplier)
    expected = series.resample("D").mean().multiply(multiplier)
    tm.assert_series_equal(result, expected)


def test_resample_apply_with_additional_args2():
    # 测试 DataFrame
    def f(data, add_arg):
        return np.mean(data) * add_arg

    multiplier = 10

    # 创建 DataFrame，并进行分组和重采样操作
    df = DataFrame({"A": 1, "B": 2}, index=date_range("2017", periods=10))
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby("A").resample("D").agg(f, multiplier).astype(float)
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.groupby("A").resample("D").mean().multiply(multiplier)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize(
    "n1, freq1, n2, freq2",
    [
        (30, "s", 0.5, "Min"),
        (60, "s", 1, "Min"),
        (3600, "s", 1, "h"),
        (60, "Min", 1, "h"),
        (21600, "s", 0.25, "D"),
        (86400, "s", 1, "D"),
        (43200, "s", 0.5, "D"),
        (1440, "Min", 1, "D"),
        (12, "h", 0.5, "D"),
        (24, "h", 1, "D"),
    ],
)
def test_resample_equivalent_offsets(n1, freq1, n2, freq2, k, unit):
    # GH 24127
    # 创建日期时间索引和 Series 对象
    n1_ = n1 * k
    n2_ = n2 * k
    dti = date_range("1991-09-05", "1991-09-12", freq=freq1).as_unit(unit)
    ser = Series(range(len(dti)), index=dti)

    # 执行两种不同频率的重采样并验证结果的一致性
    result1 = ser.resample(str(n1_) + freq1).mean()
    result2 = ser.resample(str(n2_) + freq2).mean()
    tm.assert_series_equal(result1, result2)
    [
        # 第一个元组：起始日期 "19910905", 终止日期 "19920406", 类型 "D", 预期起始日期 "19910905", 预期终止日期 "19920407"
        ("19910905", "19920406", "D", "19910905", "19920407"),
        # 第二个元组：起始日期 "19910905 00:00", 终止日期 "19920406 06:00", 类型 "D", 预期起始日期 "19910905", 预期终止日期 "19920407"
        ("19910905 00:00", "19920406 06:00", "D", "19910905", "19920407"),
        # 第三个元组：起始日期 "19910905 06:00", 终止日期 "19920406 06:00", 类型 "h", 预期起始日期 "19910905 06:00", 预期终止日期 "19920406 07:00"
        ("19910905 06:00", "19920406 06:00", "h", "19910905 06:00", "19920406 07:00"),
        # 第四个元组：起始日期 "19910906", 终止日期 "19920406", 类型 "ME", 预期起始日期 "19910831", 预期终止日期 "19920430"
        ("19910906", "19920406", "ME", "19910831", "19920430"),
        # 第五个元组：起始日期 "19910831", 终止日期 "19920430", 类型 "ME", 预期起始日期 "19910831", 预期终止日期 "19920531"
        ("19910831", "19920430", "ME", "19910831", "19920531"),
        # 第六个元组：起始日期 "1991-08", 终止日期 "1992-04", 类型 "ME", 预期起始日期 "19910831", 预期终止日期 "19920531"
        ("1991-08", "1992-04", "ME", "19910831", "19920531"),
    ],
@pytest.mark.parametrize(
    "duplicates", [True, False]
)
def test_resample_apply_product(duplicates, unit):
    # GH 5586
    # 创建时间索引，频率为"ME"（月末），共12个时期，转换为指定时间单位
    index = date_range(start="2012-01-31", freq="ME", periods=12).as_unit(unit)

    # 创建时间序列 ts 和包含两列的 DataFrame df
    ts = Series(range(12), index=index)
    df = DataFrame({"A": ts, "B": ts + 2})
    if duplicates:
        df.columns = ["A", "A"]  # 如果 duplicates 为 True，则重命名列名为"A"

    # 对 DataFrame 进行季度频率（"QE"）的重采样，应用 np.prod 函数
    result = df.resample("QE").apply(np.prod)

    # 创建预期的 DataFrame，指定索引和列名
    expected = DataFrame(
        np.array([[0, 24], [60, 210], [336, 720], [990, 1716]], dtype=np.int64),
        index=DatetimeIndex(
            ["2012-03-31", "2012-06-30", "2012-09-30", "2012-12-31"], freq="QE-DEC"
        ).as_unit(unit),
        columns=df.columns,
    )

    # 使用测试框架中的函数验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "first,last,freq_in,freq_out,exp_last",
    [
        (
            "2020-03-28",
            "2020-03-31",
            "D",
            "24h",
            "2020-03-30 01:00",
        ),  # 包括夏令时转换
        (
            "2020-03-28",
            "2020-10-27",
            "D",
            "24h",
            "2020-10-27 00:00",
        ),  # 包括夏令时的进入和结束
        (
            "2020-10-25",
            "2020-10-27",
            "D",
            "24h",
            "2020-10-26 23:00",
        ),  # 包括夏令时的结束
        (
            "2020-03-28",
            "2020-03-31",
            "24h",
            "D",
            "2020-03-30 00:00",
        ),  # 同上，但从24小时转换到日频率
        ("2020-03-28", "2020-10-27", "24h", "D", "2020-10-27 00:00"),
        ("2020-10-25", "2020-10-27", "24h", "D", "2020-10-26 00:00"),
    ],
)
def test_resample_calendar_day_with_dst(
    first: str, last: str, freq_in: str, freq_out: str, exp_last: str, unit
):
    # GH 35219
    # 创建时间序列 ts，使用指定的时区 "Europe/Amsterdam"，进行日频率的重采样，前向填充
    ts = Series(
        1.0, date_range(first, last, freq=freq_in, tz="Europe/Amsterdam").as_unit(unit)
    )
    result = ts.resample(freq_out).ffill()

    # 创建预期的时间序列 expected，使用指定的时区 "Europe/Amsterdam"
    expected = Series(
        1.0,
        date_range(first, exp_last, freq=freq_out, tz="Europe/Amsterdam").as_unit(unit),
    )

    # 使用测试框架中的函数验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func", ["min", "max", "first", "last"]
)
def test_resample_aggregate_functions_min_count(func, unit):
    # GH#37768
    # 创建时间索引，频率为"ME"（月末），共3个时期，转换为指定时间单位
    index = date_range(start="2020", freq="ME", periods=3).as_unit(unit)

    # 创建包含 NaN 值的时间序列 ser
    ser = Series([1, np.nan, np.nan], index)

    # 对时间序列进行季度频率（"QE"）的重采样，应用指定的聚合函数（min/max/first/last），并设置最小有效值数为2
    result = getattr(ser.resample("QE"), func)(min_count=2)
    # 创建一个预期的 Pandas Series 对象，其中包含一个 NaN 值
    expected = Series(
        [np.nan],
        # 使用指定的日期时间索引创建 DatetimeIndex 对象，并按照给定频率转换为指定单位
        index=DatetimeIndex(["2020-03-31"], freq="QE-DEC").as_unit(unit),
    )
    # 使用 Pandas 的测试工具模块 tm 来比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
def test_resample_unsigned_int(any_unsigned_int_numpy_dtype, unit):
    # 定义一个测试函数，用于测试无符号整数类型的数据重采样功能
    df = DataFrame(
        # 创建一个 DataFrame 对象，索引为时间范围，频率为指定单位
        index=date_range(start="2000-01-01", end="2000-01-03 23", freq="12h").as_unit(
            unit
        ),
        columns=["x"],
        # 数据为列表 [0, 1, 0] 重复两次，数据类型为给定的任意无符号整数类型
        data=[0, 1, 0] * 2,
        dtype=any_unsigned_int_numpy_dtype,
    )
    # 筛选出索引早于 "2000-01-02" 或晚于 "2000-01-03" 的行
    df = df.loc[(df.index < "2000-01-02") | (df.index > "2000-01-03"), :]

    # 对 DataFrame 进行按日重采样，取每日的最大值
    result = df.resample("D").max()

    # 创建一个预期的 DataFrame 对象，索引为时间范围，频率为指定单位
    expected = DataFrame(
        [1, np.nan, 0],
        columns=["x"],
        index=date_range(start="2000-01-01", end="2000-01-03 23", freq="D").as_unit(
            unit
        ),
    )
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_long_rule_non_nano():
    # 定义一个测试函数，用于测试长周期（100年）的时间频率重采样功能
    idx = date_range("0300-01-01", "2000-01-01", unit="s", freq="100YE")
    # 创建一个时间序列对象，数据为给定的索引和值
    ser = Series([1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5], index=idx)
    # 对时间序列进行 200年周期的重采样，取周期均值
    result = ser.resample("200YE").mean()

    # 创建预期的索引对象和时间序列对象，用于比较结果
    expected_idx = DatetimeIndex(
        np.array(
            [
                "0300-12-31",
                "0500-12-31",
                "0700-12-31",
                "0900-12-31",
                "1100-12-31",
                "1300-12-31",
                "1500-12-31",
                "1700-12-31",
                "1900-12-31",
            ]
        ).astype("datetime64[s]"),
        freq="200YE-DEC",
    )
    expected = Series([1.0, 3.0, 6.5, 4.0, 3.0, 6.5, 4.0, 3.0, 6.5], index=expected_idx)
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_resample_empty_series_with_tz():
    # 定义一个测试函数，用于测试带时区信息的空时间序列重采样功能
    df = DataFrame({"ts": [], "values": []}).astype(
        {"ts": "datetime64[ns, Atlantic/Faroe]"}
    )
    # 对空 DataFrame 进行以 2个月 为周期的重采样，计算左闭右开时间区间的值的总和
    result = df.resample("2MS", on="ts", closed="left", label="left", origin="start")[
        "values"
    ].sum()

    # 创建预期的索引对象和时间序列对象，用于比较结果
    expected_idx = DatetimeIndex(
        [], freq="2MS", name="ts", dtype="datetime64[ns, Atlantic/Faroe]"
    )
    expected = Series([], index=expected_idx, name="values", dtype="float64")
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("freq", ["2M", "2m", "2Q", "2Q-SEP", "2q-sep", "1Y", "2Y-MAR"])
def test_resample_M_Q_Y_raises(freq):
    # 定义一个测试函数，用于测试不支持的时间频率引发异常的情况
    msg = f"Invalid frequency: {freq}"

    s = Series(range(10), index=date_range("20130101", freq="d", periods=10))
    # 使用 pytest 断言异常的方式验证 resample 方法在给定的频率下是否引发了 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        s.resample(freq).mean()


@pytest.mark.parametrize("freq", ["2BM", "1bm", "1BQ", "2BQ-MAR", "2bq=-mar"])
def test_resample_BM_BQ_raises(freq):
    # 定义一个测试函数，用于测试不支持的业务日时间频率引发异常的情况
    msg = f"Invalid frequency: {freq}"

    s = Series(range(10), index=date_range("20130101", freq="d", periods=10))
    # 使用 pytest 断言异常的方式验证 resample 方法在给定的频率下是否引发了 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        s.resample(freq).mean()


def test_resample_ms_closed_right(unit):
    # 定义一个测试函数，用于测试毫秒级时间戳右闭合重采样的功能
    dti = date_range(start="2020-01-31", freq="1min", periods=6000, unit=unit)
    df = DataFrame({"ts": dti}, index=dti)
    # 对 DataFrame 进行以月为周期的重采样，保留右边界的值
    grouped = df.resample("MS", closed="right")
    result = grouped.last()
    # 创建一个 DatetimeIndex 对象，包含两个日期时间，频率为指定的 unit
    exp_dti = DatetimeIndex(
        [datetime(2020, 1, 1), datetime(2020, 2, 1)], freq="MS"
    ).as_unit(unit)
    # 创建一个 DataFrame 对象，包含一个列名为 'ts' 的时间戳列，指定索引为 exp_dti
    expected = DataFrame(
        {"ts": [datetime(2020, 2, 1), datetime(2020, 2, 4, 3, 59)]},
        index=exp_dti,
    ).astype(f"M8[{unit}]")
    # 使用测试工具 tm.assert_frame_equal() 检查 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("freq", ["B", "C"])
def test_resample_c_b_closed_right(freq: str, unit):
    # 创建一个日期时间索引，从 '2020-01-31' 开始，频率为 '1min'，共 6000 个时间点，时间单位由参数 unit 指定
    dti = date_range(start="2020-01-31", freq="1min", periods=6000, unit=unit)
    # 创建一个 DataFrame，包含一个名为 'ts' 的列，索引使用 dti 中的日期时间
    df = DataFrame({"ts": dti}, index=dti)
    # 对 DataFrame 进行按频率 resample 操作，使用 closed="right" 参数
    grouped = df.resample(freq, closed="right")
    # 获取 resample 后每个组的最后一个值
    result = grouped.last()

    # 期望的日期时间索引，使用指定的频率 freq 和 unit 转换成所需的时间单位
    exp_dti = DatetimeIndex(
        [
            datetime(2020, 1, 30),
            datetime(2020, 1, 31),
            datetime(2020, 2, 3),
            datetime(2020, 2, 4),
        ],
        freq=freq,
    ).as_unit(unit)
    # 创建一个期望的 DataFrame，包含一个名为 'ts' 的列，指定索引为 exp_dti
    expected = DataFrame(
        {
            "ts": [
                datetime(2020, 1, 31),
                datetime(2020, 2, 3),
                datetime(2020, 2, 4),
                datetime(2020, 2, 4, 3, 59),
            ]
        },
        index=exp_dti,
    ).astype(f"M8[{unit}]")
    # 使用 pytest 的 assert 函数检查 result 是否与 expected 相等
    tm.assert_frame_equal(result, expected)


def test_resample_b_55282(unit):
    # 创建一个日期时间索引，从 '2023-09-26' 开始，周期为 6，频率为 '12h'，时间单位由参数 unit 指定
    dti = date_range("2023-09-26", periods=6, freq="12h", unit=unit)
    # 创建一个 Series，包含整数值 1 到 6，索引使用 dti 中的日期时间
    ser = Series([1, 2, 3, 4, 5, 6], index=dti)
    # 对 Series 进行按频率 resample 操作，使用 closed="right" 和 label="right" 参数，计算平均值
    result = ser.resample("B", closed="right", label="right").mean()

    # 期望的日期时间索引，使用频率 "B"，时间单位由 unit 转换
    exp_dti = DatetimeIndex(
        [
            datetime(2023, 9, 26),
            datetime(2023, 9, 27),
            datetime(2023, 9, 28),
            datetime(2023, 9, 29),
        ],
        freq="B",
    ).as_unit(unit)
    # 创建一个期望的 Series，包含平均值为 [1.0, 2.5, 4.5, 6.0]，索引为 exp_dti
    expected = Series(
        [1.0, 2.5, 4.5, 6.0],
        index=exp_dti,
    )
    # 使用 pytest 的 assert 函数检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


@td.skip_if_no("pyarrow")
@pytest.mark.parametrize(
    "tz",
    [
        None,
        pytest.param(
            "UTC",
            marks=pytest.mark.xfail(
                condition=is_platform_windows(),
                reason="TODO: Set ARROW_TIMEZONE_DATABASE env var in CI",
            ),
        ),
    ],
)
def test_arrow_timestamp_resample(tz):
    # GH 56371
    # 创建一个 Series，包含从 '2020-01-01' 开始的五个时间点，数据类型为 'timestamp[ns][pyarrow]'
    idx = Series(date_range("2020-01-01", periods=5), dtype="timestamp[ns][pyarrow]")
    # 如果 tz 不为 None，则对索引进行时区本地化处理
    if tz is not None:
        idx = idx.dt.tz_localize(tz)
    # 创建一个期望的 Series，包含浮点数值为 [0.0, 1.0, 2.0, 3.0, 4.0]，索引为 idx
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    # 对期望的 Series 进行按频率 "1D" 的 resample 操作，计算平均值
    result = expected.resample("1D").mean()
    # 使用 pytest 的 assert 函数检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("freq", ["1A", "2A-MAR"])
def test_resample_A_raises(freq):
    # 创建一个错误消息，内容为 "Invalid frequency: " 后接 freq 的第二个字符开始的子字符串
    msg = f"Invalid frequency: {freq[1:]}"

    # 创建一个 Series，包含从 '20130101' 开始，频率为 'd'，共 10 个时间点的整数值 0 到 9
    s = Series(range(10), index=date_range("20130101", freq="d", periods=10))
    # 使用 pytest 的 raises 函数检查调用 s.resample(freq).mean() 是否抛出 ValueError 异常，并匹配错误消息 msg
    with pytest.raises(ValueError, match=msg):
        s.resample(freq).mean()
```