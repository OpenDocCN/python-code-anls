# `D:\src\scipysrc\pandas\pandas\tests\resample\test_time_grouper.py`

```
# 导入必要的模块和库
from datetime import datetime  # 导入datetime模块中的datetime类
from operator import methodcaller  # 导入methodcaller函数

import numpy as np  # 导入NumPy库并重命名为np
import pytest  # 导入pytest测试框架

import pandas as pd  # 导入Pandas库并重命名为pd
from pandas import (  # 导入Pandas中的DataFrame、Index、Series、Timestamp类
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入Pandas测试模块中的tm对象
from pandas.core.groupby.grouper import Grouper  # 导入Pandas中的Grouper类
from pandas.core.indexes.datetimes import date_range  # 导入Pandas中的date_range函数


@pytest.fixture
def test_series():
    # 返回一个包含随机标准正态分布数据的Series对象，索引为从"1/1/2000"开始的1000天
    return Series(
        np.random.default_rng(2).standard_normal(1000),
        index=date_range("1/1/2000", periods=1000),
    )


def test_apply(test_series):
    # 创建一个Grouper对象，以年为频率，标签为右边界，闭合方式为右边界
    grouper = Grouper(freq="YE", label="right", closed="right")

    # 根据Grouper对象对test_series进行分组
    grouped = test_series.groupby(grouper)

    # 定义一个函数f，对每个组进行操作，返回排序后的最后3个值
    def f(x):
        return x.sort_values()[-3:]

    # 对分组后的数据应用函数f
    applied = grouped.apply(f)

    # 期望的结果：按年对test_series应用函数f
    expected = test_series.groupby(lambda x: x.year).apply(f)

    # 丢弃结果中的第一级索引，并将结果赋给applied和expected
    applied.index = applied.index.droplevel(0)
    expected.index = expected.index.droplevel(0)

    # 使用测试模块tm中的函数assert_series_equal比较applied和expected是否相等
    tm.assert_series_equal(applied, expected)


def test_count(test_series):
    # 每隔3个数据设置为NaN
    test_series[::3] = np.nan

    # 期望的结果：按年对test_series进行计数
    expected = test_series.groupby(lambda x: x.year).count()

    # 创建一个Grouper对象，以年为频率，标签为右边界，闭合方式为右边界
    grouper = Grouper(freq="YE", label="right", closed="right")

    # 根据Grouper对象对test_series进行分组，并计算每组的计数
    result = test_series.groupby(grouper).count()
    expected.index = result.index

    # 使用测试模块tm中的函数assert_series_equal比较result和expected是否相等
    tm.assert_series_equal(result, expected)

    # 对test_series进行年度重采样，并计算每组的计数
    result = test_series.resample("YE").count()
    expected.index = result.index

    # 使用测试模块tm中的函数assert_series_equal比较result和expected是否相等
    tm.assert_series_equal(result, expected)


def test_numpy_reduction(test_series):
    # 对test_series进行年度重采样，闭合方式为右边界，并计算每组的乘积
    result = test_series.resample("YE", closed="right").prod()

    # 期望的结果：按年对test_series进行乘积运算
    expected = test_series.groupby(lambda x: x.year).agg(np.prod)
    expected.index = result.index

    # 使用测试模块tm中的函数assert_series_equal比较result和expected是否相等
    tm.assert_series_equal(result, expected)


def test_apply_iteration():
    # #2300
    N = 1000
    ind = date_range(start="2000-01-01", freq="D", periods=N)
    df = DataFrame({"open": 1, "close": 2}, index=ind)
    tg = Grouper(freq="ME")

    # 获取TimeGrouper对象tg对DataFrame df的分组器grouper
    grouper, _ = tg._get_grouper(df)

    # 对DataFrame df按照分组器grouper进行分组，不显示分组键
    grouped = df.groupby(grouper, group_keys=False)

    # 定义函数f，对每个组进行操作，计算"close"列与"open"列的比值
    def f(df):
        return df["close"] / df["open"]

    # 对分组后的数据应用函数f
    result = grouped.apply(f)

    # 使用测试模块tm中的函数assert_index_equal比较result的索引与df的索引是否相等
    tm.assert_index_equal(result.index, df.index)


@pytest.mark.parametrize(
    "index",
    [
        Index([1, 2]),
        Index(["a", "b"]),
        Index([1.1, 2.2]),
        pd.MultiIndex.from_arrays([[1, 2], ["a", "b"]]),
    ],
)
def test_fails_on_no_datetime_index(index):
    # 获取索引对象index的类型名称
    name = type(index).__name__

    # 创建一个DataFrame对象df，包含"a"列和索引对象index
    df = DataFrame({"a": range(len(index))}, index=index)

    # 错误消息
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex "
        f"or PeriodIndex, but got an instance of '{name}'"
    )

    # 使用pytest断言，期望引发TypeError并且错误消息包含msg内容
    with pytest.raises(TypeError, match=msg):
        df.groupby(Grouper(freq="D"))


def test_aaa_group_order():
    # GH 12840
    # 检查TimeGrouper对象的稳定排序
    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4))
    df = DataFrame(data, columns=["A", "B", "C", "D"])
    df["key"] = [
        datetime(2013, 1, 1),
        datetime(2013, 1, 2),
        datetime(2013, 1, 3),
        datetime(2013, 1, 4),
        datetime(2013, 1, 5),
        datetime(2013, 1, 6),
        datetime(2013, 1, 7),
        datetime(2013, 1, 8),
        datetime(2013, 1, 9),
        datetime(2013, 1, 10),
        datetime(2013, 1, 11),
        datetime(2013, 1, 12),
        datetime(2013, 1, 13),
        datetime(2013, 1, 14),
        datetime(2013, 1, 15),
        datetime(2013, 1, 16),
        datetime(2013, 1, 17),
        datetime(2013, 1, 18),
        datetime(2013, 1, 19),
        datetime(2013, 1, 20),
    )
    ] * 4
    # 创建一个包含四个空列表的列表，每个列表都是空的，这个操作等同于 [[]] * 4
    grouped = df.groupby(Grouper(key="key", freq="D"))
    # 使用 pandas 中的 groupby 方法，按照日期（每天）分组 DataFrame df

    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 1)), df[::5])
    # 断言：验证 groupby 对象 grouped 中日期为 2013-01-01 的分组结果与 df[::5] 是否相等
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 2)), df[1::5])
    # 断言：验证 groupby 对象 grouped 中日期为 2013-01-02 的分组结果与 df[1::5] 是否相等
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 3)), df[2::5])
    # 断言：验证 groupby 对象 grouped 中日期为 2013-01-03 的分组结果与 df[2::5] 是否相等
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 4)), df[3::5])
    # 断言：验证 groupby 对象 grouped 中日期为 2013-01-04 的分组结果与 df[3::5] 是否相等
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 5)), df[4::5])
    # 断言：验证 groupby 对象 grouped 中日期为 2013-01-05 的分组结果与 df[4::5] 是否相等
def test_aggregate_normal(resample_method):
    """Check TimeGrouper's aggregation is identical as normal groupby."""

    # 创建一个20行4列的随机数数据，使用正态分布
    data = np.random.default_rng(2).standard_normal((20, 4))
    # 创建一个普通的DataFrame，列名为 ["A", "B", "C", "D"]，并加入一个名为 "key" 的列
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    normal_df["key"] = [1, 2, 3, 4, 5] * 4

    # 创建一个新的DataFrame，与normal_df相同的数据和列名，但"key"列中的数据类型为日期时间
    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    dt_df["key"] = Index(
        [
            datetime(2013, 1, 1),
            datetime(2013, 1, 2),
            datetime(2013, 1, 3),
            datetime(2013, 1, 4),
            datetime(2013, 1, 5),
        ]
        * 4,
        dtype="M8[ns]",
    )

    # 对普通DataFrame按"key"列进行分组
    normal_grouped = normal_df.groupby("key")
    # 对带日期时间的DataFrame按时间窗口进行分组
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    # 调用指定的聚合方法（如sum、mean等）来获取期望的聚合结果
    expected = getattr(normal_grouped, resample_method)()
    # 使用相同的方法在日期时间分组上获取结果
    dt_result = getattr(dt_grouped, resample_method)()
    # 将普通分组结果的索引设置为日期时间范围
    expected.index = date_range(start="2013-01-01", freq="D", periods=5, name="key")
    # 使用时间窗口分组结果的索引
    tm.assert_equal(expected, dt_result)


@pytest.mark.xfail(reason="if TimeGrouper is used included, 'nth' doesn't work yet")
def test_aggregate_nth():
    """Check TimeGrouper's aggregation is identical as normal groupby."""

    # 创建一个20行4列的随机数数据，使用正态分布
    data = np.random.default_rng(2).standard_normal((20, 4))
    # 创建一个普通的DataFrame，列名为 ["A", "B", "C", "D"]，并加入一个名为 "key" 的列
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    normal_df["key"] = [1, 2, 3, 4, 5] * 4

    # 创建一个新的DataFrame，与normal_df相同的数据和列名，但"key"列中的数据类型为日期时间
    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    dt_df["key"] = [
        datetime(2013, 1, 1),
        datetime(2013, 1, 2),
        datetime(2013, 1, 3),
        datetime(2013, 1, 4),
        datetime(2013, 1, 5),
    ] * 4

    # 对普通DataFrame按"key"列进行分组
    normal_grouped = normal_df.groupby("key")
    # 对带日期时间的DataFrame按时间窗口进行分组
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    # 在普通分组结果上使用nth方法获取第3个条目
    expected = normal_grouped.nth(3)
    # 将普通分组结果的索引设置为日期时间范围
    expected.index = date_range(start="2013-01-01", freq="D", periods=5, name="key")
    # 在时间窗口分组结果上使用nth方法获取第3个条目
    dt_result = dt_grouped.nth(3)
    # 比较期望的DataFrame和实际的DataFrame是否相等
    tm.assert_frame_equal(expected, dt_result)


@pytest.mark.parametrize(
    "method, method_args, unit",
    [
        ("sum", {}, 0),
        ("sum", {"min_count": 0}, 0),
        ("sum", {"min_count": 1}, np.nan),
        ("prod", {}, 1),
        ("prod", {"min_count": 0}, 1),
        ("prod", {"min_count": 1}, np.nan),
    ],
)
def test_resample_entirely_nat_window(method, method_args, unit):
    """Test resampling handling of entirely NaN or zero series."""

    # 创建一个Series，包含2个0和2个NaN，索引为2017年的日期范围
    ser = Series([0] * 2 + [np.nan] * 2, index=date_range("2017", periods=4))
    # 使用methodcaller调用指定方法（如sum、prod等）来处理NaN或零的情况
    result = methodcaller(method, **method_args)(ser.resample("2d"))

    # 期望的日期时间索引，频率为2天，从2017-01-01开始，包含2个日期
    exp_dti = pd.DatetimeIndex(["2017-01-01", "2017-01-03"], dtype="M8[ns]", freq="2D")
    # 创建期望的Series，根据method和method_args指定的方法计算结果
    expected = Series([0.0, unit], index=exp_dti)
    # 比较期望的Series和实际的结果Series是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, fill_value",
    [("min", np.nan), ("max", np.nan), ("sum", 0), ("prod", 1), ("count", 0)],
)
def test_aggregate_with_nat(func, fill_value):
    """Check TimeGrouper's aggregation is identical as normal groupby
    if NaT is included, 'var', 'std', 'mean', 'first','last'
    and 'nth' doesn't work yet"""

    # 创建一个20行4列的随机数数据，使用正态分布，并将其转换为int64类型
    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4)).astype("int64")
    # 创建一个普通的DataFrame，使用给定的数据和列名
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    # 添加一个名为"key"的新列，其中包含重复的值和NaN
    normal_df["key"] = [1, 2, np.nan, 4, 5] * 4
    
    # 创建一个日期时间的DataFrame，使用给定的数据和列名
    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    # 添加一个名为"key"的新列，其中包含日期时间索引和NaT（Not a Time）
    dt_df["key"] = Index(
        [
            datetime(2013, 1, 1),
            datetime(2013, 1, 2),
            pd.NaT,
            datetime(2013, 1, 4),
            datetime(2013, 1, 5),
        ]
        * 4,
        dtype="M8[ns]",
    )
    
    # 对普通DataFrame按"key"列进行分组
    normal_grouped = normal_df.groupby("key")
    # 对日期时间DataFrame按日期时间索引进行分组，使用日频率
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))
    
    # 对普通DataFrame分组后调用指定的聚合函数（func）
    normal_result = getattr(normal_grouped, func)()
    # 对日期时间DataFrame分组后调用指定的聚合函数（func）
    dt_result = getattr(dt_grouped, func)()
    
    # 创建一个DataFrame，用指定的值填充，索引为[3]，列名为["A", "B", "C", "D"]
    pad = DataFrame([[fill_value] * 4], index=[3], columns=["A", "B", "C", "D"])
    # 将normal_result和pad进行连接
    expected = pd.concat([normal_result, pad])
    # 对连接后的DataFrame按索引排序
    expected = expected.sort_index()
    
    # 创建一个日期时间索引，从指定的起始日期开始，频率为日，周期为5
    dti = date_range(
        start="2013-01-01",
        freq="D",
        periods=5,
        name="key",
        unit=dt_df["key"]._values.unit,
    )
    # 将expected的索引设置为dti，但频率为None（取消频率）
    expected.index = dti._with_freq(None)  # TODO: is this desired?
    
    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(expected, dt_result)
    # 断言dt_result的索引名称为"key"
    assert dt_result.index.name == "key"
def test_aggregate_with_nat_size():
    # GH 9925
    # 设置数据的行数
    n = 20
    # 生成一个随机数据矩阵，并转换为 int64 类型的 DataFrame
    data = np.random.default_rng(2).standard_normal((n, 4)).astype("int64")
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    # 创建包含 NaN 值的 key 列
    normal_df["key"] = [1, 2, np.nan, 4, 5] * 4

    # 复制数据矩阵生成第二个 DataFrame
    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    # 创建包含 NaT (Not a Time) 值的 key 列
    dt_df["key"] = Index(
        [
            datetime(2013, 1, 1),
            datetime(2013, 1, 2),
            pd.NaT,
            datetime(2013, 1, 4),
            datetime(2013, 1, 5),
        ]
        * 4,
        dtype="M8[ns]",
    )

    # 对 normal_df 按 key 列进行分组
    normal_grouped = normal_df.groupby("key")
    # 对 dt_df 按 Grouper 对象 (包含时间频率 'D') 进行分组
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    # 对 normal_grouped 执行 size() 操作，得到结果 Series
    normal_result = normal_grouped.size()
    # 对 dt_grouped 执行 size() 操作，得到结果 Series
    dt_result = dt_grouped.size()

    # 创建包含一个元素的 Series，index 是 [3]
    pad = Series([0], index=[3])
    # 将 normal_result 和 pad 进行合并
    expected = pd.concat([normal_result, pad])
    # 对合并后的 Series 按索引排序
    expected = expected.sort_index()
    # 生成一个时间范围的 DatetimeIndex，用于作为 expected 的索引
    expected.index = date_range(
        start="2013-01-01",
        freq="D",
        periods=5,
        name="key",
        unit=dt_df["key"]._values.unit,
    )._with_freq(None)
    # 使用 assert_series_equal 检查 expected 和 dt_result 是否相等
    tm.assert_series_equal(expected, dt_result)
    # 检查 dt_result 的索引名称是否为 "key"
    assert dt_result.index.name == "key"


def test_repr():
    # GH18203
    # 测试 Grouper 对象的 repr 输出是否符合预期
    result = repr(Grouper(key="A", freq="h"))
    expected = (
        "TimeGrouper(key='A', freq=<Hour>, sort=True, dropna=True, "
        "closed='left', label='left', how='mean', "
        "convention='e', origin='start_day')"
    )
    # 断言 result 和 expected 是否相等
    assert result == expected

    # 测试带有 origin 参数的 Grouper 对象的 repr 输出是否符合预期
    result = repr(Grouper(key="A", freq="h", origin="2000-01-01"))
    expected = (
        "TimeGrouper(key='A', freq=<Hour>, sort=True, dropna=True, "
        "closed='left', label='left', how='mean', "
        "convention='e', origin=Timestamp('2000-01-01 00:00:00'))"
    )
    # 断言 result 和 expected 是否相等
    assert result == expected


@pytest.mark.parametrize(
    "method, method_args, expected_values",
    [
        ("sum", {}, [1, 0, 1]),
        ("sum", {"min_count": 0}, [1, 0, 1]),
        ("sum", {"min_count": 1}, [1, np.nan, 1]),
        ("sum", {"min_count": 2}, [np.nan, np.nan, np.nan]),
        ("prod", {}, [1, 1, 1]),
        ("prod", {"min_count": 0}, [1, 1, 1]),
        ("prod", {"min_count": 1}, [1, np.nan, 1]),
        ("prod", {"min_count": 2}, [np.nan, np.nan, np.nan]),
    ],
)
def test_upsample_sum(method, method_args, expected_values):
    # 创建一个 Series 对象
    ser = Series(1, index=date_range("2017", periods=2, freq="h"))
    # 对 ser 进行 resample 操作，生成一个 resampled 对象
    resampled = ser.resample("30min")
    # 创建一个 DatetimeIndex 对象
    index = pd.DatetimeIndex(
        ["2017-01-01T00:00:00", "2017-01-01T00:30:00", "2017-01-01T01:00:00"],
        dtype="M8[ns]",
        freq="30min",
    )
    # 使用 methodcaller 对 resampled 进行指定方法的调用，返回结果 Series
    result = methodcaller(method, **method_args)(resampled)
    # 创建一个预期结果的 Series 对象
    expected = Series(expected_values, index=index)
    # 使用 assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.fixture
def groupy_test_df():
    # 创建一个测试用的 DataFrame 对象，包含 price 和 volume 列
    return DataFrame(
        {"price": [10, 11, 9], "volume": [50, 60, 50]},
        index=date_range("01/01/2018", periods=3, freq="W"),
    )


def test_groupby_resample_interpolate_raises(groupy_test_df):
    # GH 35325
    # 测试 groupby、resample 和 interpolate 的组合是否会引发异常
    # 这里需要填入具体的测试内容，包括对异常的处理或预期结果的验证
    pass
    # 复制测试数据框，使其索引名为 None
    groupy_test_df_without_index_name = groupy_test_df.copy()
    # 将复制后的数据框的索引名设置为 None
    groupy_test_df_without_index_name.index.name = None

    # 将两个数据框放入列表中，分别是原始数据框和没有索引名的复制数据框
    dfs = [groupy_test_df, groupy_test_df_without_index_name]

    # 遍历数据框列表
    for df in dfs:
        # 设置警告信息内容
        msg = "DataFrameGroupBy.resample operated on the grouping columns"
        # 断言代码块中产生 DeprecationWarning 警告且包含特定的警告信息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 断言抛出 NotImplementedError 异常且包含特定的错误信息
            with pytest.raises(
                NotImplementedError,
                match="Direct interpolation of MultiIndex data frames is "
                "not supported",
            ):
                # 在分组后的数据框上进行 resample 操作，并尝试使用线性插值方法
                df.groupby("volume").resample("1D").interpolate(method="linear")
# 定义一个测试函数，用于测试带有 apply 语法的分组、重采样和插值操作
def test_groupby_resample_interpolate_with_apply_syntax(groupy_test_df):
    # GH 35325

    # 复制测试数据框，并将其索引名称设置为 None
    groupy_test_df_without_index_name = groupy_test_df.copy()
    groupy_test_df_without_index_name.index.name = None

    # 创建包含两个数据框的列表
    dfs = [groupy_test_df, groupy_test_df_without_index_name]

    # 遍历数据框列表
    for df in dfs:
        # 对数据框按 'volume' 分组，并应用重采样和线性插值方法
        result = df.groupby("volume").apply(
            lambda x: x.resample("1d").interpolate(method="linear"),
            include_groups=False,
        )

        # 预期的 'volume' 和 'week_starting' 索引
        volume = [50] * 15 + [60]
        week_starting = list(pd.date_range("2018-01-07", "2018-01-21")) + [
            pd.Timestamp("2018-01-14")
        ]
        expected_ind = pd.MultiIndex.from_arrays(
            [volume, week_starting],
            names=["volume", df.index.name],
        )

        # 预期的数据框结果
        expected = pd.DataFrame(
            data={
                "price": [
                    10.0,
                    9.928571428571429,
                    9.857142857142858,
                    9.785714285714286,
                    9.714285714285714,
                    9.642857142857142,
                    9.571428571428571,
                    9.5,
                    9.428571428571429,
                    9.357142857142858,
                    9.285714285714286,
                    9.214285714285714,
                    9.142857142857142,
                    9.071428571428571,
                    9.0,
                    11.0,
                ]
            },
            index=expected_ind,
        )
        # 断言数据框的相等性
        pd.testing.assert_frame_equal(result, expected)


# 定义另一个测试函数，与上一个函数类似，但是在插值时存在缺失的锚点
def test_groupby_resample_interpolate_with_apply_syntax_off_grid(groupy_test_df):
    """Similar test as test_groupby_resample_interpolate_with_apply_syntax but
    with resampling that results in missing anchor points when interpolating.
    See GH#21351."""
    # GH#21351

    # 对数据框按 'volume' 分组，并应用重采样和线性插值方法（间隔为 265 小时）
    result = groupy_test_df.groupby("volume").apply(
        lambda x: x.resample("265h").interpolate(method="linear"), include_groups=False
    )

    # 预期的 'volume' 和 'week_starting' 索引
    volume = [50, 50, 60]
    week_starting = pd.DatetimeIndex(
        [
            pd.Timestamp("2018-01-07"),
            pd.Timestamp("2018-01-18 01:00:00"),
            pd.Timestamp("2018-01-14"),
        ]
    ).as_unit("ns")
    expected_ind = pd.MultiIndex.from_arrays(
        [volume, week_starting],
        names=["volume", "week_starting"],
    )

    # 预期的数据框结果
    expected = pd.DataFrame(
        data={
            "price": [
                10.0,
                9.21131,
                11.0,
            ]
        },
        index=expected_ind,
    )
    # 断言数据框的相等性，忽略名称的检查
    pd.testing.assert_frame_equal(result, expected, check_names=False)
```