# `D:\src\scipysrc\pandas\pandas\tests\resample\test_base.py`

```
# 导入需要的模块和函数
from datetime import datetime

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas.core.dtypes.common import is_extension_array_dtype  # 导入判断扩展数组类型的函数

import pandas as pd  # 导入Pandas库，用于数据处理和分析
from pandas import (  # 导入Pandas中的各种数据结构和索引类型
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm  # 导入Pandas测试模块，用于编写测试辅助函数

from pandas.core.groupby.groupby import DataError  # 导入分组时可能出现的错误类型
from pandas.core.groupby.grouper import Grouper  # 导入分组器
from pandas.core.indexes.datetimes import date_range  # 导入生成日期范围的函数
from pandas.core.indexes.period import period_range  # 导入生成周期范围的函数
from pandas.core.indexes.timedeltas import timedelta_range  # 导入生成时间差范围的函数
from pandas.core.resample import _asfreq_compat  # 导入重采样相关的函数和类


@pytest.fixture(  # 定义pytest的fixture，用于返回各种插值方法的参数
    params=[
        "linear",
        "time",
        "index",
        "values",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "barycentric",
        "krogh",
        "from_derivatives",
        "piecewise_polynomial",
        "pchip",
        "akima",
    ],
)
def all_1d_no_arg_interpolation_methods(request):
    return request.param


@pytest.mark.parametrize("freq", ["2D", "1h"])  # 对频率参数进行pytest参数化
@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),  # 生成时间差范围的索引
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 生成日期范围的索引
    ],
)
def test_asfreq(frame_or_series, index, freq):
    # 根据输入的索引创建DataFrame或Series对象
    obj = frame_or_series(range(len(index)), index=index)
    # 根据索引类型选择对应的生成范围函数
    idx_range = date_range if isinstance(index, DatetimeIndex) else timedelta_range

    # 进行重采样并设为频率
    result = obj.resample(freq).asfreq()
    # 生成新的索引范围
    new_index = idx_range(obj.index[0], obj.index[-1], freq=freq)
    # 生成期望的重采样结果
    expected = obj.reindex(new_index)
    # 使用Pandas测试模块检查结果是否几乎相等
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),  # 生成时间差范围的索引
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 生成日期范围的索引
    ],
)
def test_asfreq_fill_value(index):
    # 对重采样时填充值的测试，解决问题3715

    # 创建带有索引的Series对象
    ser = Series(range(len(index)), index=index, name="a")
    # 根据索引类型选择对应的生成范围函数
    idx_range = date_range if isinstance(index, DatetimeIndex) else timedelta_range

    # 进行重采样并设为每小时频率
    result = ser.resample("1h").asfreq()
    # 生成新的索引范围
    new_index = idx_range(ser.index[0], ser.index[-1], freq="1h")
    # 生成期望的重采样结果
    expected = ser.reindex(new_index)
    # 使用Pandas测试模块检查Series对象是否相等
    tm.assert_series_equal(result, expected)

    # 将Series对象显式转换为浮点数类型，避免设置None时的隐式转换
    frame = ser.astype("float").to_frame("value")
    # 设置部分数据为None
    frame.iloc[1] = None
    # 进行重采样并设为每小时频率，填充值为4.0
    result = frame.resample("1h").asfreq(fill_value=4.0)
    # 生成新的索引范围
    new_index = idx_range(frame.index[0], frame.index[-1], freq="1h")
    # 生成期望的重采样结果，填充值为4.0
    expected = frame.reindex(new_index, fill_value=4.0)
    # 使用Pandas测试模块检查DataFrame对象是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),  # 生成时间差范围的索引
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 生成日期范围的索引
        period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 生成周期范围的索引
    ],
)
def test_resample_interpolate(index):
    # GH#12925
    # 创建一个 DataFrame 对象，使用给定的索引和长度
    df = DataFrame(range(len(index)), index=index)
    
    # 初始化警告变量为 None
    warn = None
    
    # 检查 DataFrame 的索引类型是否为 PeriodIndex 类型
    if isinstance(df.index, PeriodIndex):
        # 如果索引类型为 PeriodIndex，则设置警告类型为 FutureWarning
        warn = FutureWarning
    
    # 设置警告信息的字符串
    msg = "Resampling with a PeriodIndex is deprecated"
    
    # 使用上下文管理器确保在特定警告条件下会产生警告信息
    with tm.assert_produces_warning(warn, match=msg):
        # 对 DataFrame 进行重新采样到每分钟，并插值处理
        result = df.resample("1min").asfreq().interpolate()
        # 生成预期的重新采样结果，不进行频率调整但进行插值处理
        expected = df.resample("1min").interpolate()
    
    # 断言两个 DataFrame 对象在预期情况下相等
    tm.assert_frame_equal(result, expected)
# 测试函数：对正常间隔采样和非网格采样的插值进行测试
def test_resample_interpolate_regular_sampling_off_grid(
    all_1d_no_arg_interpolation_methods,
):
    # 导入必要的依赖库 pytest，如果缺少 scipy 库则跳过该测试
    pytest.importorskip("scipy")
    
    # 创建时间序列索引，从 "2000-01-01 00:01:00" 开始，每 2 小时一个频率，共5个时间点
    index = date_range("2000-01-01 00:01:00", periods=5, freq="2h")
    
    # 创建 Series 对象，使用 np.arange(5.0) 数组作为数据，指定上面创建的索引
    ser = Series(np.arange(5.0), index)
    
    # 从参数中获取所有 1D 无参数插值方法
    method = all_1d_no_arg_interpolation_methods
    
    # 将序列重采样为 1 小时间隔，并使用给定的插值方法进行插值
    ser_resampled = ser.resample("1h").interpolate(method)
    
    # 检查重采样后的第一个值是否为 NaN，除了第一个值之外的其他值都不应为 NaN
    assert np.isnan(ser_resampled.iloc[0])
    assert not ser_resampled.iloc[1:].isna().any()
    
    if method not in ["nearest", "zero"]:
        # 对于不是 "nearest" 或 "zero" 方法的情况，检查重采样后的值是否接近预期值
        assert np.all(
            np.isclose(ser_resampled.values[1:], np.arange(0.5, 4.5, 0.5), rtol=1.0e-1)
        )


# 测试函数：对非正常间隔采样的插值进行测试
def test_resample_interpolate_irregular_sampling(all_1d_no_arg_interpolation_methods):
    # 导入必要的依赖库 pytest，如果缺少 scipy 库则跳过该测试
    pytest.importorskip("scipy")
    
    # 创建 Series 对象，使用 np.linspace(0.0, 1.0, 5) 数组作为数据，指定非正常间隔的时间索引
    ser = Series(
        np.linspace(0.0, 1.0, 5),
        index=DatetimeIndex(
            [
                "2000-01-01 00:00:03",
                "2000-01-01 00:00:22",
                "2000-01-01 00:00:24",
                "2000-01-01 00:00:31",
                "2000-01-01 00:00:39",
            ]
        ),
    )
    
    # 将序列重采样为 5 秒间隔，并使用给定的插值方法进行插值
    ser_resampled = ser.resample("5s").interpolate(all_1d_no_arg_interpolation_methods)
    
    # 检查重采样后的第一个值是否为 NaN，除了第一个值之外的其他值都不应为 NaN
    assert np.isnan(ser_resampled.iloc[0])
    assert not ser_resampled.iloc[1:].isna().any()


# 测试函数：对非日期时间索引抛出异常的情况进行测试
def test_raises_on_non_datetimelike_index():
    # 创建一个空的 DataFrame 对象 xp
    xp = DataFrame()
    
    # 错误信息，说明只能处理 DatetimeIndex、TimedeltaIndex 或 PeriodIndex 类型的索引
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'RangeIndex'"
    )
    
    # 使用 pytest 的断言检查是否会抛出预期的 TypeError 异常，并包含指定的错误信息
    with pytest.raises(TypeError, match=msg):
        xp.resample("YE")


# 参数化测试：对空序列进行重采样的情况进行测试
@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="D", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_empty_series(freq, index, resample_method):
    # GH12771 & GH12868
    
    # 创建一个指定索引和 float 类型的空 Series 对象 ser
    ser = Series(index=index, dtype=float)
    
    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        # 对于 TimedeltaIndex 类型的索引，需要指定固定的频率 'freq'，例如 '24h' 或 '3D'，不能是 <MonthEnd>
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        # 使用 pytest 的断言检查是否会抛出预期的 ValueError 异常，并包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # 如果索引是 PeriodIndex 类型，将 'freq' 转换为相应的 Period 频率 'M'
        freq = "M"
    
    # 暂时没有使用的变量声明
    warn = None
    # 检查序列的索引是否为 PeriodIndex 类型
    if isinstance(ser.index, PeriodIndex):
        # 如果是 PeriodIndex 类型，设定警告类型为 FutureWarning
        warn = FutureWarning
    
    # 设置警告信息字符串
    msg = "Resampling with a PeriodIndex is deprecated"
    
    # 使用 assert_produces_warning 上下文管理器确保在 resample 操作时产生警告
    with tm.assert_produces_warning(warn, match=msg):
        # 对序列进行频率重采样
        rs = ser.resample(freq)
    
    # 调用 resample 方法得到结果
    result = getattr(rs, resample_method)()

    # 如果重采样方法是 "ohlc"
    if resample_method == "ohlc":
        # 创建一个空的 DataFrame，用于与预期结果比较
        expected = DataFrame(
            [], index=ser.index[:0], columns=["open", "high", "low", "close"]
        )
        # 将预期结果的索引设置为与序列兼容的频率
        expected.index = _asfreq_compat(ser.index, freq)
        # 使用 assert_frame_equal 检查结果与预期是否相等（忽略数据类型）
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        # 复制原始序列作为预期结果
        expected = ser.copy()
        # 将预期结果的索引设置为与序列兼容的频率
        expected.index = _asfreq_compat(ser.index, freq)
        # 使用 assert_series_equal 检查结果与预期是否相等（忽略数据类型）
        tm.assert_series_equal(result, expected, check_dtype=False)

    # 使用 assert_index_equal 检查结果的索引与预期索引是否相等
    tm.assert_index_equal(result.index, expected.index)
    # 使用 assert 断言确保结果的频率与预期的频率相同
    assert result.index.freq == expected.index.freq
@pytest.mark.parametrize(
    "freq",
    [
        # 第一个参数化测试：标记为预期失败，原因是不清楚为什么会失败
        pytest.param("ME", marks=pytest.mark.xfail(reason="Don't know why this fails")),
        "D",
        "h",
    ],
)
def test_resample_nat_index_series(freq, resample_method):
    # GH39227
    # 创建一个 Series 对象，索引为 PeriodIndex，值为 [NaT, NaT, NaT, NaT, NaT]
    ser = Series(range(5), index=PeriodIndex([NaT] * 5, freq=freq))

    # 设置警告消息
    msg = "Resampling with a PeriodIndex is deprecated"
    # 断言会产生 FutureWarning 警告，并匹配给定的消息
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # 对 Series 进行重新采样
        rs = ser.resample(freq)
    # 使用指定的重采样方法进行操作
    result = getattr(rs, resample_method)()

    if resample_method == "ohlc":
        # 如果重采样方法是 "ohlc"，期望得到一个空的 DataFrame，索引与 ser 的前0个元素的索引相同
        expected = DataFrame(
            [], index=ser.index[:0], columns=["open", "high", "low", "close"]
        )
        # 断言结果与期望的 DataFrame 相等，忽略数据类型检查
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        # 否则，期望得到一个空的 Series，与 ser 的前0个元素相同
        expected = ser[:0].copy()
        # 断言结果与期望的 Series 相等，忽略数据类型检查
        tm.assert_series_equal(result, expected, check_dtype=False)
    # 断言结果的索引与期望的索引相等
    tm.assert_index_equal(result.index, expected.index)
    # 断言结果的频率与期望的频率相等
    assert result.index.freq == expected.index.freq


@pytest.mark.parametrize(
    "index",
    [
        # 第二个参数化测试：创建一个空的 PeriodIndex，频率为 'D'，名称为 'a'
        PeriodIndex([], freq="D", name="a"),
        # 创建一个空的 DatetimeIndex，名称为 'a'
        DatetimeIndex([], name="a"),
        # 创建一个空的 TimedeltaIndex，名称为 'a'
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
@pytest.mark.parametrize("resample_method", ["count", "size"])
def test_resample_count_empty_series(freq, index, resample_method):
    # GH28427
    # 创建一个 Series 对象，索引为 index
    ser = Series(index=index)
    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        # 如果 freq 是 "ME" 并且索引类型是 TimedeltaIndex，抛出 ValueError 异常，匹配指定消息
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # 如果 freq 是 "ME" 并且索引类型是 PeriodIndex，将 freq 转换为对应的 Period 频率 "M"
        freq = "M"

    warn = None
    if isinstance(ser.index, PeriodIndex):
        # 如果索引类型是 PeriodIndex，设置警告类型为 FutureWarning
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"
    # 断言会产生警告，匹配指定的消息
    with tm.assert_produces_warning(warn, match=msg):
        # 对 Series 进行重新采样
        rs = ser.resample(freq)

    # 使用指定的重采样方法进行操作
    result = getattr(rs, resample_method)()

    # 调用辅助函数 _asfreq_compat，将索引转换为指定的频率
    index = _asfreq_compat(ser.index, freq)

    # 创建一个预期的空 Series，dtype 为 "int64"，索引为 index，名称与 ser 相同
    expected = Series([], dtype="int64", index=index, name=ser.name)

    # 断言结果与期望的 Series 相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index", [DatetimeIndex([]), TimedeltaIndex([]), PeriodIndex([], freq="D")]
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_empty_dataframe(index, freq, resample_method):
    # GH13212
    # 创建一个空的 DataFrame，索引为 index
    df = DataFrame(index=index)
    # 对于 freq 是 "ME" 并且索引类型是 TimedeltaIndex 的情况
    if freq == "ME" and isinstance(df.index, TimedeltaIndex):
        # 抛出 ValueError 异常，匹配指定消息
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            # 进行重新采样，不包含分组键
            df.resample(freq, group_keys=False)
        return
    # 如果频率为 "ME" 并且索引是 PeriodIndex 类型
    elif freq == "ME" and isinstance(df.index, PeriodIndex):
        # 索引是 PeriodIndex，因此将频率转换为对应的 Period 频率 "M"
        freq = "M"

    # 初始化警告为 None
    warn = None
    # 如果索引是 PeriodIndex 类型
    if isinstance(df.index, PeriodIndex):
        # 设置警告类型为 FutureWarning
        warn = FutureWarning
    # 警告消息
    msg = "Resampling with a PeriodIndex is deprecated"
    # 使用警告和消息断言将来的警告
    with tm.assert_produces_warning(warn, match=msg):
        # 对数据框进行频率重采样，不分组键
        rs = df.resample(freq, group_keys=False)
    # 调用指定的重采样方法
    result = getattr(rs, resample_method)()
    # 如果重采样方法是 "ohlc"
    if resample_method == "ohlc":
        # TODO: 对于 len(df.columns) > 0 的情况尚未进行测试
        # 创建 MultiIndex，包含每列的 ["open", "high", "low", "close"] 值
        mi = MultiIndex.from_product([df.columns, ["open", "high", "low", "close"]])
        # 创建一个空的 DataFrame，使用 df.index[:0] 作为索引，列使用 mi，数据类型为 np.float64
        expected = DataFrame([], index=df.index[:0], columns=mi, dtype=np.float64)
        # 调整索引以与给定频率兼容
        expected.index = _asfreq_compat(df.index, freq)

    # 如果重采样方法不是 "size"
    elif resample_method != "size":
        # 复制原始数据框作为期望的结果
        expected = df.copy()
    else:
        # GH14962 特定情况下的处理
        # 创建一个空的 Series，数据类型为 np.int64
        expected = Series([], dtype=np.int64)

    # 调整索引以与给定频率兼容
    expected.index = _asfreq_compat(df.index, freq)

    # 断言结果索引与期望索引相等
    tm.assert_index_equal(result.index, expected.index)
    # 断言结果的频率与期望的频率相等
    assert result.index.freq == expected.index.freq
    # 断言结果几乎等于期望值
    tm.assert_almost_equal(result, expected)

    # 对于 GH13212 的测试，目前保持为 df 的大小
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_resample_count_empty_dataframe 和 test_resample_size_empty_dataframe 参数化输入数据
@pytest.mark.parametrize(
    "index", [DatetimeIndex([]), TimedeltaIndex([]), PeriodIndex([], freq="D")]
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_count_empty_dataframe(freq, index):
    # GH28427: GitHub issue reference

    # 创建一个空 DataFrame，指定索引为 index，列为 ["a"]，数据类型为 object
    empty_frame_dti = DataFrame(index=index, columns=Index(["a"], dtype=object))

    # 如果 freq 是 "ME" 并且 empty_frame_dti 的索引是 TimedeltaIndex 类型
    if freq == "ME" and isinstance(empty_frame_dti.index, TimedeltaIndex):
        # 设定异常消息
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配消息内容为 msg
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return

    # 如果 freq 是 "ME" 并且 empty_frame_dti 的索引是 PeriodIndex 类型
    elif freq == "ME" and isinstance(empty_frame_dti.index, PeriodIndex):
        # 将 freq 转换为 "M"，因为索引是 PeriodIndex 类型

    # 如果 empty_frame_dti 的索引是 PeriodIndex 类型
    warn = None
    if isinstance(empty_frame_dti.index, PeriodIndex):
        # 设置警告类型为 FutureWarning
        warn = FutureWarning
    # 设定警告消息
    msg = "Resampling with a PeriodIndex is deprecated"
    # 使用 tm.assert_produces_warning 检查是否产生 warn 类型的警告，并匹配消息内容为 msg
    with tm.assert_produces_warning(warn, match=msg):
        # 对 empty_frame_dti 进行频率重采样，得到 rs
        rs = empty_frame_dti.resample(freq)
    # 对 rs 执行 count 操作，得到结果 result
    result = rs.count()

    # 调用 _asfreq_compat 函数，将 empty_frame_dti 的索引根据 freq 转换为对应的索引
    index = _asfreq_compat(empty_frame_dti.index, freq)

    # 创建一个期望的 DataFrame，数据类型为 int64，索引为 index，列为 ["a"]
    expected = DataFrame(dtype="int64", index=index, columns=Index(["a"], dtype=object))

    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_resample_size_empty_dataframe 参数化输入数据
@pytest.mark.parametrize(
    "index", [DatetimeIndex([]), TimedeltaIndex([]), PeriodIndex([], freq="D")]
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_size_empty_dataframe(freq, index):
    # GH28427: GitHub issue reference

    # 创建一个空 DataFrame，指定索引为 index，列为 ["a"]，数据类型为 object
    empty_frame_dti = DataFrame(index=index, columns=Index(["a"], dtype=object))

    # 如果 freq 是 "ME" 并且 empty_frame_dti 的索引是 TimedeltaIndex 类型
    if freq == "ME" and isinstance(empty_frame_dti.index, TimedeltaIndex):
        # 设定异常消息
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配消息内容为 msg
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return

    # 如果 freq 是 "ME" 并且 empty_frame_dti 的索引是 PeriodIndex 类型
    elif freq == "ME" and isinstance(empty_frame_dti.index, PeriodIndex):
        # 将 freq 转换为 "M"，因为索引是 PeriodIndex 类型

    # 设定警告消息
    msg = "Resampling with a PeriodIndex"
    warn = None
    # 如果 empty_frame_dti 的索引是 PeriodIndex 类型
    if isinstance(empty_frame_dti.index, PeriodIndex):
        # 设置警告类型为 FutureWarning
        warn = FutureWarning
    # 使用 tm.assert_produces_warning 检查是否产生 warn 类型的警告，并匹配消息内容为 msg
    with tm.assert_produces_warning(warn, match=msg):
        # 对 empty_frame_dti 进行频率重采样，得到 rs
        rs = empty_frame_dti.resample(freq)
    # 对 rs 执行 size 操作，得到结果 result
    result = rs.size()

    # 调用 _asfreq_compat 函数，将 empty_frame_dti 的索引根据 freq 转换为对应的索引
    index = _asfreq_compat(empty_frame_dti.index, freq)

    # 创建一个期望的 Series，数据类型为 int64，索引为 index，数据为空
    expected = Series([], dtype="int64", index=index)

    # 使用 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_resample_empty_dtypes 参数化输入数据
@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="M", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("dtype", [float, int, object, "datetime64[ns]"])
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_resample_empty_dtypes(index, dtype, resample_method):
    # Empty series were sometimes causing a segfault (for the functions
    # 如果 index 是 PeriodIndex 类型，则进行以下处理
    warn = None
    if isinstance(index, PeriodIndex):
        # 创建一个空的 PeriodIndex 对象，使用 "B" 频率，指定 index 的名称
        index = PeriodIndex([], freq="B", name=index.name)
        # 设置 warn 变量为 FutureWarning，用于后续的警告信息
        warn = FutureWarning
    # 设置警告消息内容
    msg = "Resampling with a PeriodIndex is deprecated"

    # 创建一个空的 Series 对象，使用空列表、指定的 index 和 dtype
    empty_series_dti = Series([], index, dtype)
    # 断言在执行下面的操作时会产生警告，并匹配指定的警告消息
    with tm.assert_produces_warning(warn, match=msg):
        # 对创建的空 Series 对象进行重采样，频率为 "d"，并关闭分组键
        rs = empty_series_dti.resample("d", group_keys=False)
    try:
        # 尝试调用 rs 对象的指定重采样方法（由 resample_method 指定）
        getattr(rs, resample_method)()
    except DataError:
        # 捕获 DataError 异常，通常忽略这些异常因为某些组合是无效的
        # （例如使用 np.object_ 类型的数据执行均值操作）
        pass
@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="D", name="a"),  # 创建一个空的 PeriodIndex 对象，频率为天（'D'），名称为'a'
        DatetimeIndex([], name="a"),  # 创建一个空的 DatetimeIndex 对象，名称为'a'
        TimedeltaIndex([], name="a"),  # 创建一个空的 TimedeltaIndex 对象，名称为'a'
    ],
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_apply_to_empty_series(index, freq):
    # GH 14313
    ser = Series(index=index)  # 使用给定的 index 创建一个空的 Series 对象

    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):  # 断言抛出 ValueError 异常，异常信息需匹配 msg
            ser.resample(freq)
        return  # 如果满足条件，直接返回，不执行后续代码
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # index 是 PeriodIndex 类型，因此转换为相应的 Period 频率 'M'
        freq = "M"

    msg = "Resampling with a PeriodIndex"  # 警告信息字符串
    warn = None
    if isinstance(ser.index, PeriodIndex):
        warn = FutureWarning  # 如果 index 是 PeriodIndex 类型，则警告为 FutureWarning

    with tm.assert_produces_warning(warn, match=msg):
        rs = ser.resample(freq, group_keys=False)  # 对 Series 对象进行重新采样，不分组键

    result = rs.apply(lambda x: 1)  # 对重新采样后的结果应用 lambda 函数
    with tm.assert_produces_warning(warn, match=msg):
        expected = ser.resample(freq).apply("sum")  # 对重新采样后的结果应用 'sum' 函数

    tm.assert_series_equal(result, expected, check_dtype=False)  # 断言两个 Series 对象是否相等


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),  # 创建一个时间增量范围对象，频率为天（'D'）
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 创建一个日期范围对象，频率为天（'D'）
        period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 创建一个周期范围对象，频率为天（'D'）
    ],
)
def test_resampler_is_iterable(index):
    # GH 15314
    series = Series(range(len(index)), index=index)  # 使用给定的 index 创建一个 Series 对象
    freq = "h"  # 设置频率为小时（'h'）
    tg = Grouper(freq=freq, convention="start")  # 创建一个 Grouper 对象，设置频率和约定方式为起始点
    msg = "Resampling with a PeriodIndex"  # 警告信息字符串
    warn = None
    if isinstance(series.index, PeriodIndex):
        warn = FutureWarning  # 如果 index 是 PeriodIndex 类型，则警告为 FutureWarning

    with tm.assert_produces_warning(warn, match=msg):
        grouped = series.groupby(tg)  # 对 Series 对象进行分组操作

    with tm.assert_produces_warning(warn, match=msg):
        resampled = series.resample(freq)  # 对 Series 对象进行重新采样
    for (rk, rv), (gk, gv) in zip(resampled, grouped):
        assert rk == gk  # 断言重新采样后的索引与分组后的索引相等
        tm.assert_series_equal(rv, gv)  # 断言重新采样后的结果与分组后的结果相等


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),  # 创建一个时间增量范围对象，频率为天（'D'）
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 创建一个日期范围对象，频率为天（'D'）
        period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),  # 创建一个周期范围对象，频率为天（'D'）
    ],
)
def test_resample_quantile(index):
    # GH 15023
    ser = Series(range(len(index)), index=index)  # 使用给定的 index 创建一个 Series 对象
    q = 0.75  # 设置分位数
    freq = "h"  # 设置频率为小时（'h'）

    msg = "Resampling with a PeriodIndex"  # 警告信息字符串
    warn = None
    if isinstance(ser.index, PeriodIndex):
        warn = FutureWarning  # 如果 index 是 PeriodIndex 类型，则警告为 FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = ser.resample(freq).quantile(q)  # 对 Series 对象进行重新采样，并计算分位数
        expected = ser.resample(freq).agg(lambda x: x.quantile(q)).rename(ser.name)  # 对重新采样后的结果应用分位数计算，并重命名结果 Series 的名称
    tm.assert_series_equal(result, expected)  # 断言两个 Series 对象是否相等


@pytest.mark.parametrize("how", ["first", "last"])
def test_first_last_skipna(any_real_nullable_dtype, skipna, how):
    # GH#57019
    # 待补充具体的测试代码
    # 检查是否是扩展数组数据类型，并获取空值的默认表示方式
    if is_extension_array_dtype(any_real_nullable_dtype):
        na_value = Series(dtype=any_real_nullable_dtype).dtype.na_value
    else:
        # 如果不是扩展数组数据类型，则使用 NumPy 中的 NaN 表示空值
        na_value = np.nan
    
    # 创建一个 DataFrame 对象 df，包括三列数据 'a', 'b', 'c'，并设置索引为指定日期范围
    df = DataFrame(
        {
            "a": [2, 1, 1, 2],
            "b": [na_value, 3.0, na_value, 4.0],
            "c": [na_value, 3.0, na_value, 4.0],
        },
        index=date_range("2020-01-01", periods=4, freq="D"),
        dtype=any_real_nullable_dtype,
    )
    
    # 对 df 进行按照"ME"（Month End）频率进行重新采样，得到一个 Resampler 对象 rs
    rs = df.resample("ME")
    
    # 根据传入的 how 参数获取 rs 对象的对应方法（例如 sum、mean 等）
    method = getattr(rs, how)
    
    # 调用获取的方法（如 sum、mean 等）对 rs 进行处理，设置是否跳过 NaN 值
    result = method(skipna=skipna)

    # 将字符串日期 "2020-01-31" 转换为 Timestamp 对象，并指定单位为纳秒
    ts = pd.to_datetime("2020-01-31").as_unit("ns")
    
    # 根据 df 的行数乘以 ts 的值进行分组，得到一个 GroupBy 对象 gb
    gb = df.groupby(df.shape[0] * [ts])
    
    # 根据传入的 how 参数对 gb 进行聚合操作（例如 sum、mean 等），设置是否跳过 NaN 值
    expected = getattr(gb, how)(skipna=skipna)
    
    # 设置期望结果的索引频率为"ME"（Month End）
    expected.index.freq = "ME"
    
    # 使用测试工具 tm 中的 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```