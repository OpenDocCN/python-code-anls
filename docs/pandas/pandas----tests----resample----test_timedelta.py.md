# `D:\src\scipysrc\pandas\pandas\tests\resample\test_timedelta.py`

```
from datetime import timedelta  # 导入 timedelta 类用于处理时间间隔

import numpy as np  # 导入 numpy 库并重命名为 np
import pytest  # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas 内部测试装饰器模块

import pandas as pd  # 导入 pandas 库并重命名为 pd
from pandas import (  # 导入 pandas 中的 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 pandas 测试模块

from pandas.core.indexes.timedeltas import timedelta_range  # 导入 pandas 时间间隔索引相关函数


def test_asfreq_bug():
    df = DataFrame(data=[1, 3], index=[timedelta(), timedelta(minutes=3)])  # 创建一个 DataFrame，包含时间间隔索引
    result = df.resample("1min").asfreq()  # 对 DataFrame 进行时间重采样到每分钟，并填充空白值
    expected = DataFrame(  # 创建期望结果的 DataFrame
        data=[1, np.nan, np.nan, 3],  # 数据为 [1, NaN, NaN, 3]
        index=timedelta_range("0 day", periods=4, freq="1min"),  # 时间索引范围为每分钟
    )
    tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与期望是否一致


def test_resample_with_nat():
    # GH 13223
    index = pd.to_timedelta(["0s", pd.NaT, "2s"])  # 创建时间间隔索引，包含 NaT（Not a Time）空值
    result = DataFrame({"value": [2, 3, 5]}, index).resample("1s").mean()  # 对 DataFrame 进行时间重采样到每秒，并计算均值
    expected = DataFrame(  # 创建期望结果的 DataFrame
        {"value": [2.5, np.nan, 5.0]},  # 数据为 [2.5, NaN, 5.0]
        index=timedelta_range("0 day", periods=3, freq="1s"),  # 时间索引范围为每秒
    )
    tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与期望是否一致


def test_resample_as_freq_with_subperiod():
    # GH 13022
    index = timedelta_range("00:00:00", "00:10:00", freq="5min")  # 创建时间间隔索引，每 5 分钟一个时间点
    df = DataFrame(data={"value": [1, 5, 10]}, index=index)  # 创建包含值的 DataFrame
    result = df.resample("2min").asfreq()  # 对 DataFrame 进行时间重采样到每 2 分钟，并填充空白值
    expected_data = {"value": [1, np.nan, np.nan, np.nan, np.nan, 10]}  # 期望的数据
    expected = DataFrame(  # 创建期望结果的 DataFrame
        data=expected_data,  # 使用期望的数据
        index=timedelta_range("00:00:00", "00:10:00", freq="2min"),  # 时间索引范围为每 2 分钟
    )
    tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与期望是否一致


def test_resample_with_timedeltas():
    expected = DataFrame({"A": np.arange(1480)})  # 创建一个期望的 DataFrame，包含列 A 和从 0 到 1479 的数据
    expected = expected.groupby(expected.index // 30).sum()  # 对期望的 DataFrame 进行分组求和
    expected.index = timedelta_range("0 days", freq="30min", periods=50)  # 更新期望结果的时间索引

    df = DataFrame(  # 创建一个 DataFrame，包含列 A 和从 0 到 1479 的数据，时间索引以分钟为单位
        {"A": np.arange(1480)}, index=pd.to_timedelta(np.arange(1480), unit="min")
    )
    result = df.resample("30min").sum()  # 对 DataFrame 进行时间重采样到每 30 分钟，并求和

    tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与期望是否一致

    s = df["A"]  # 从 DataFrame 中选择列 A，创建一个 Series 对象
    result = s.resample("30min").sum()  # 对 Series 进行时间重采样到每 30 分钟，并求和
    tm.assert_series_equal(result, expected["A"])  # 使用测试模块验证结果与期望的列 A 是否一致


def test_resample_single_period_timedelta():
    s = Series(list(range(5)), index=timedelta_range("1 day", freq="s", periods=5))  # 创建一个时间间隔索引的 Series
    result = s.resample("2s").sum()  # 对 Series 进行时间重采样到每 2 秒，并求和
    expected = Series([1, 5, 4], index=timedelta_range("1 day", freq="2s", periods=3))  # 创建期望的 Series
    tm.assert_series_equal(result, expected)  # 使用测试模块验证结果与期望是否一致


def test_resample_timedelta_idempotency():
    # GH 12072
    index = timedelta_range("0", periods=9, freq="10ms")  # 创建时间间隔索引，间隔为 10 毫秒
    series = Series(range(9), index=index)  # 创建一个 Series 包含索引和值
    result = series.resample("10ms").mean()  # 对 Series 进行时间重采样到每 10 毫秒，并求均值
    expected = series.astype(float)  # 将 Series 转换为浮点型作为期望结果
    tm.assert_series_equal(result, expected)  # 使用测试模块验证结果与期望是否一致


def test_resample_offset_with_timedeltaindex():
    # GH 10530 & 31809
    rng = timedelta_range(start="0s", periods=25, freq="s")  # 创建时间间隔索引，每秒一个时间点，共 25 个时间点
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)  # 创建一个随机数填充的 Series

    with_base = ts.resample("2s", offset="5s").mean()  # 对 Series 进行时间重采样到每 2 秒，并设置起始偏移量为 5 秒
    without_base = ts.resample("2s").mean()  # 对 Series 进行时间重采样到每 2 秒

    exp_without_base = timedelta_range(start="0s", end="25s", freq="2s")  # 期望的时间间隔索引，每 2 秒一个时间点
    exp_with_base = timedelta_range(start="5s", end="29s", freq="2s")  # 期望的时间间隔索引，从 5 秒开始，每 2 秒一个时间点
    # 使用 pandas.testing.assert_index_equal() 函数比较 without_base 的索引和 exp_without_base 是否相等
    tm.assert_index_equal(without_base.index, exp_without_base)
    # 使用 pandas.testing.assert_index_equal() 函数比较 with_base 的索引和 exp_with_base 是否相等
    tm.assert_index_equal(with_base.index, exp_with_base)
# 测试函数：测试在时间增量索引下对分类数据进行重新采样
def test_resample_categorical_data_with_timedeltaindex():
    # GH #12169
    # 创建一个 DataFrame，包含一个名为 'Group_obj' 的列，所有值为 'A'，索引为秒单位的时间增量索引
    df = DataFrame({"Group_obj": "A"}, index=pd.to_timedelta(list(range(20)), unit="s"))
    # 将 'Group_obj' 列的数据类型转换为分类类型，存储到新列 'Group' 中
    df["Group"] = df["Group_obj"].astype("category")
    # 对 DataFrame 进行10秒重新采样，使用聚合函数获取每组的众数
    result = df.resample("10s").agg(lambda x: (x.value_counts().index[0]))
    # 生成预期的时间增量索引，频率为10秒
    exp_tdi = pd.TimedeltaIndex(np.array([0, 10], dtype="m8[s]"), freq="10s").as_unit(
        "ns"
    )
    # 创建预期的 DataFrame，包含 'Group_obj' 和 'Group' 列，索引为 exp_tdi
    expected = DataFrame(
        {"Group_obj": ["A", "A"], "Group": ["A", "A"]},
        index=exp_tdi,
    )
    # 在列上重新索引预期的 DataFrame，保留 'Group_obj' 和 'Group' 列
    expected = expected.reindex(["Group_obj", "Group"], axis=1)
    # 将 'Group_obj' 列的数据类型转换为分类类型，存储到新列 'Group' 中
    expected["Group"] = expected["Group_obj"].astype("category")
    # 使用测试工具函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试在时间增量值下对时间增量进行重新采样
def test_resample_timedelta_values():
    # GH 13119
    # 创建一个时间增量范围，频率为4天，存储到名为 'times' 的 Series 中
    times = timedelta_range("1 day", "6 day", freq="4D")
    # 创建一个 DataFrame，包含 'time' 列，索引和列均使用 'times'
    df = DataFrame({"time": times}, index=times)
    # 创建预期的 Series，包含 'times' 的副本，第二个值设置为 NaT
    times2 = timedelta_range("1 day", "6 day", freq="2D")
    exp = Series(times2, index=times2, name="time")
    exp.iloc[1] = pd.NaT
    # 对 'time' 列进行2天重新采样，获取第一个值，存储到 res
    res = df.resample("2D").first()["time"]
    # 使用测试工具函数检查 res 和 exp 是否相等
    tm.assert_series_equal(res, exp)
    # 对 'time' 列进行2天重新采样，存储到 res
    res = df["time"].resample("2D").first()
    # 使用测试工具函数检查 res 和 exp 是否相等
    tm.assert_series_equal(res, exp)


# 使用参数化测试的测试函数：测试在时间增量边缘情况下的重新采样
@pytest.mark.parametrize(
    "start, end, freq, resample_freq",
    [
        ("8h", "21h59min50s", "10s", "3h"),  # GH 30353 example
        ("3h", "22h", "1h", "5h"),
        ("527D", "5006D", "3D", "10D"),
        ("1D", "10D", "1D", "2D"),  # GH 13022 example
        # tests that worked before GH 33498:
        ("8h", "21h59min50s", "10s", "2h"),
        ("0h", "21h59min50s", "10s", "3h"),
        ("10D", "85D", "D", "2D"),
    ],
)
def test_resample_timedelta_edge_case(start, end, freq, resample_freq):
    # GH 33498
    # 创建一个时间增量范围，指定开始、结束和频率，存储到 idx
    idx = timedelta_range(start=start, end=end, freq=freq)
    # 创建一个 Series，索引为 idx，值为递增的整数
    s = Series(np.arange(len(idx)), index=idx)
    # 对 Series 进行 resample 操作，使用最小值聚合
    result = s.resample(resample_freq).min()
    # 生成预期的时间增量索引，频率为 resample_freq
    expected_index = timedelta_range(freq=resample_freq, start=start, end=end)
    # 使用测试工具函数检查 result 的索引和 expected_index 是否相等
    tm.assert_index_equal(result.index, expected_index)
    # 检查 result 的索引频率与 expected_index 是否相等
    assert result.index.freq == expected_index.freq
    # 检查 result 的最后一个元素不是 NaN
    assert not np.isnan(result.iloc[-1])


# 使用参数化测试的测试函数：测试在时间增量值下进行重新采样时，不产生空组
@pytest.mark.parametrize("duplicates", [True, False])
def test_resample_with_timedelta_yields_no_empty_groups(duplicates):
    # GH 10603
    # 创建一个 DataFrame，包含随机正态分布数据，索引为时间增量范围
    df = DataFrame(
        np.random.default_rng(2).normal(size=(10000, 4)),
        index=timedelta_range(start="0s", periods=10000, freq="3906250ns"),
    )
    if duplicates:
        # 如果 duplicates 为 True，则将列名设置为非唯一值
        df.columns = ["A", "B", "A", "C"]
    # 从时间增量索引为 '1s' 开始，对数据进行3秒重新采样，应用 lambda 函数获取每组的长度
    result = df.loc["1s":, :].resample("3s").apply(lambda x: len(x))
    # 创建预期的 DataFrame，包含每列长度为 768 的行，索引为时间增量范围
    expected = DataFrame(
        [[768] * 4] * 12 + [[528] * 4],
        index=timedelta_range(start="1s", periods=13, freq="3s"),
    )
    # 设置预期 DataFrame 的列名与 df 相同
    expected.columns = df.columns
    # 使用测试工具函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试在时间增量值下进行分位数重新采样
def test_resample_quantile_timedelta(unit):
    # GH: 29485
    # 创建一个时间数据类型，精度为给定单位（unit），例如 's' 表示秒
    dtype = np.dtype(f"m8[{unit}]")
    # 创建一个 DataFrame 对象，包含一个名为 'value' 的列，列值为时间增量数据
    # 这些时间增量数据是从 0 开始的 4 个元素，单位为秒，并转换为指定的 dtype 类型
    df = DataFrame(
        {"value": pd.to_timedelta(np.arange(4), unit="s").astype(dtype)},
        index=pd.date_range("20200101", periods=4, tz="UTC"),
    )
    # 对 DataFrame 进行时间重采样，每2天进行重采样，并计算分位数为 0.99 的值
    result = df.resample("2D").quantile(0.99)
    # 创建一个期望的 DataFrame 对象，包含一个名为 'value' 的列，列值是预期的时间增量数据
    # 这些数据是两个元素，分别为 0.99 分位数的时间值
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),
                pd.Timedelta("0 days 00:00:02.990000"),
            ]
        },
        index=pd.date_range("20200101", periods=2, tz="UTC", freq="2D"),
    ).astype(dtype)
    # 使用测试框架中的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试时间序列重新采样的功能（向右闭合）
def test_resample_closed_right():
    # GH#45414: 引用GitHub上的issue编号，用于跟踪相关问题
    # 创建一个时间增量索引，每隔30秒增加一次，共10个时间点
    idx = pd.Index([pd.Timedelta(seconds=120 + i * 30) for i in range(10)])
    # 创建一个时间序列，索引为上述时间增量索引，值为0到9
    ser = Series(range(10), index=idx)
    # 对时间序列进行重新采样，每分钟一个时间段，右侧闭合，标签为右侧
    result = ser.resample("min", closed="right", label="right").sum()
    # 创建一个期望的时间序列，索引为每隔60秒增加一次的时间增量索引，共6个时间点
    expected = Series(
        [0, 3, 7, 11, 15, 9],
        index=pd.TimedeltaIndex(
            [pd.Timedelta(seconds=120 + i * 60) for i in range(6)], freq="min"
        ),
    )
    # 使用测试工具库中的方法，断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)


# 根据条件跳过测试，如果没有安装pyarrow库则跳过这个测试函数
@td.skip_if_no("pyarrow")
def test_arrow_duration_resample():
    # GH 56371: 引用GitHub上的issue编号，用于跟踪相关问题
    # 创建一个时间增量索引，以1天为间隔，共5个时间点，数据类型为pyarrow的duration[ns]
    idx = pd.Index(timedelta_range("1 day", periods=5), dtype="duration[ns][pyarrow]")
    # 创建一个期望的时间序列，索引为上述时间增量索引，值为0到4，数据类型为float64
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    # 对期望的时间序列进行重新采样，每天一个时间段，计算平均值
    result = expected.resample("1D").mean()
    # 使用测试工具库中的方法，断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)
```