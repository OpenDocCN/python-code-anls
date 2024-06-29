# `D:\src\scipysrc\pandas\pandas\tests\resample\test_resampler_grouper.py`

```
#`
from textwrap import dedent  # 导入 dedent 函数，用于去除多行字符串的首尾空白字符

import numpy as np  # 导入 numpy 模块，简化数值计算
import pytest  # 导入 pytest 模块，用于测试

from pandas.compat import is_platform_windows  # 从 pandas.compat 模块导入 is_platform_windows 函数

import pandas as pd  # 导入 pandas 模块，简化数据操作
from pandas import (
    DataFrame,  # 导入 DataFrame 类
    Index,  # 导入 Index 类
    Series,  # 导入 Series 类
    TimedeltaIndex,  # 导入 TimedeltaIndex 类
    Timestamp,  # 导入 Timestamp 类
)
import pandas._testing as tm  # 导入 pandas 的测试工具模块 tm
from pandas.core.indexes.datetimes import date_range  # 从 pandas.core.indexes.datetimes 导入 date_range 函数

@pytest.fixture  # 定义 pytest 测试用例的 fixture，提供测试数据
def test_frame():
    # 创建一个包含两列数据的 DataFrame，列 A 包含 20 个 1、12 个 2 和 8 个 3，列 B 为 0 到 39 的序列
    return DataFrame(
        {"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)},
        index=date_range("1/1/2000", freq="s", periods=40),  # 设置行索引为从 2000-01-01 开始的秒级时间序列
    )

def test_tab_complete_ipython6_warning(ip):
    from IPython.core.completer import provisionalcompleter  # 从 IPython.core.completer 导入 provisionalcompleter

    code = dedent(
        """\
    import numpy as np  # 导入 numpy 模块
    from pandas import Series, date_range  # 从 pandas 导入 Series 和 date_range 函数
    data = np.arange(10, dtype=np.float64)  # 创建一个浮点型数组，元素从 0 到 9
    index = date_range("2020-01-01", periods=len(data))  # 创建一个从 2020-01-01 开始的日期索引，长度与 data 数组相同
    s = Series(data, index=index)  # 创建一个 Series 对象，数据为 data，索引为 index
    rs = s.resample("D")  # 对 Series 进行日频重采样
    """
    )
    ip.run_cell(code)  # 在 ipython 环境中运行上述代码块

    # GH 31324 新版 jedi 提示 deprecated 警告，2021-02-02 解决
    with tm.assert_produces_warning(None, raise_on_extra_warnings=False):  # 断言不会产生警告，允许额外的警告
        with provisionalcompleter("ignore"):  # 使用 provisionalcompleter，忽略提示中的警告
            list(ip.Completer.completions("rs.", 1))  # 获取 ipython 的补全建议，针对 "rs." 的第一个补全

def test_deferred_with_groupby():
    # GH 12486，测试 groupby 和 resample 的延迟操作支持
    data = [
        ["2010-01-01", "A", 2],
        ["2010-01-02", "A", 3],
        ["2010-01-05", "A", 8],
        ["2010-01-10", "A", 7],
        ["2010-01-13", "A", 3],
        ["2010-01-01", "B", 5],
        ["2010-01-03", "B", 2],
        ["2010-01-04", "B", 1],
        ["2010-01-11", "B", 7],
        ["2010-01-14", "B", 3],
    ]

    df = DataFrame(data, columns=["date", "id", "score"])  # 创建 DataFrame，列名为 date、id、score
    df.date = pd.to_datetime(df.date)  # 将 date 列转换为 datetime 类型

    def f_0(x):
        return x.set_index("date").resample("D").asfreq()  # 定义一个函数，设置日期为索引，进行日频重采样

    msg = "DataFrameGroupBy.apply operated on the grouping columns"  # 定义第一个警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生 DeprecationWarning 警告
        expected = df.groupby("id").apply(f_0)  # 对 df 按 "id" 分组，应用 f_0 函数

    msg = "DataFrameGroupBy.resample operated on the grouping columns"  # 定义第二个警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生 DeprecationWarning 警告
        result = df.set_index("date").groupby("id").resample("D").asfreq()  # 设置日期为索引，按 "id" 分组，进行日频重采样
    tm.assert_frame_equal(result, expected)  # 断言 result 和 expected DataFrame 相等

    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W"),  # 创建一个从 2016-01-01 开始的周频时间序列
            "group": [1, 1, 2, 2],  # 定义分组列 group
            "val": [5, 6, 7, 8],  # 定义值列 val
        }
    ).set_index("date")  # 将 date 列设置为索引

    def f_1(x):
        return x.resample("1D").ffill()  # 定义一个函数，进行日频重采样，前向填充缺失值

    msg = "DataFrameGroupBy.apply operated on the grouping columns"  # 定义第三个警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生 DeprecationWarning 警告
        expected = df.groupby("group").apply(f_1)  # 对 df 按 "group" 分组，应用 f_1 函数
    msg = "DataFrameGroupBy.resample operated on the grouping columns"  # 定义第四个警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 断言产生 DeprecationWarning 警告
        result = df.groupby("group").resample("1D").ffill()  # 按 "group" 分组，进行日频重采样，前向填充缺失值
    tm.assert_frame_equal(result, expected)  # 断言 result 和 expected DataFrame 相等

def test_getitem(test_frame):
    g = test_frame.groupby("A")  # 根据 "A" 列进行分组，返回 GroupBy 对象
    # 使用 GroupBy 对象 g 中的列 B，对每个组进行重采样，每2秒计算一次均值，并返回期望的结果
    expected = g.B.apply(lambda x: x.resample("2s").mean())
    
    # 对 GroupBy 对象 g 进行重采样，每2秒计算一次列 B 的均值，并返回结果
    result = g.resample("2s").B.mean()
    tm.assert_series_equal(result, expected)
    
    # 对 GroupBy 对象 g 中的列 B 进行重采样，每2秒计算一次均值，并返回结果
    result = g.B.resample("2s").mean()
    tm.assert_series_equal(result, expected)
    
    # 设置警告消息文本
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 使用上下文管理器确保在进行以下操作时触发 DeprecationWarning 警告，并检查警告消息是否匹配
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 GroupBy 对象 g 进行重采样，每2秒计算一次均值后，获取列 B 的结果
        result = g.resample("2s").mean().B
    # 检查结果是否与期望的结果相等
    tm.assert_series_equal(result, expected)
def test_getitem_multiple():
    # GH 13174
    # 多次选择后的多个调用会导致别名问题
    data = [{"id": 1, "buyer": "A"}, {"id": 2, "buyer": "B"}]
    # 创建 DataFrame 对象，使用给定数据和日期索引
    df = DataFrame(data, index=date_range("2016-01-01", periods=2))
    # 对 DataFrame 进行按 id 分组，并对日期进行重新采样为每日
    r = df.groupby("id").resample("1D")
    # 计算按 id 分组后，每日的 "buyer" 列的计数
    result = r["buyer"].count()

    # 构建预期的多重索引
    exp_mi = pd.MultiIndex.from_arrays([[1, 2], df.index], names=("id", None))
    # 创建预期的 Series，包括每个 id 下 "buyer" 列的计数
    expected = Series(
        [1, 1],
        index=exp_mi,
        name="buyer",
    )
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 再次计算按 id 分组后，每日的 "buyer" 列的计数
    result = r["buyer"].count()
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_groupby_resample_on_api_with_getitem():
    # GH 17813
    # 创建包含 id、date 和 data 列的 DataFrame
    df = DataFrame(
        {"id": list("aabbb"), "date": date_range("1-1-2016", periods=5), "data": 1}
    )
    # 使用 date 列设置索引，并按 id 分组后，对日期进行重新采样为每2天，计算 "data" 列的和
    exp = df.set_index("date").groupby("id").resample("2D")["data"].sum()
    # 对原始 DataFrame 按 id 分组后，对日期进行重新采样为每2天，计算 "data" 列的和
    result = df.groupby("id").resample("2D", on="date")["data"].sum()
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, exp)


def test_groupby_with_origin():
    # GH 31809

    freq = "1399min"  # 小于24小时的素数频率
    start, end = "1/1/2000 00:00:00", "1/31/2000 00:00"
    middle = "1/15/2000 00:00:00"

    # 使用特定频率创建日期范围
    rng = date_range(start, end, freq="1231min")  # 素数频率
    # 创建具有随机数据的时间序列
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    # 从中间日期到结束日期的时间序列切片
    ts2 = ts[middle:end]

    # 证明没有固定起始点的分组器在处理不寻常频率时无法工作
    simple_grouper = pd.Grouper(freq=freq)
    # 对时间序列按照简单的分组器进行分组，并聚合计数
    count_ts = ts.groupby(simple_grouper).agg("count")
    # 对切片后的时间序列再次使用简单的分组器进行分组，并聚合计数
    count_ts = count_ts[middle:end]
    # 对比两个时间序列索引是否相等，预期会抛出异常
    with pytest.raises(AssertionError, match="Index are different"):
        tm.assert_index_equal(count_ts.index, count_ts2.index)

    # 在1970-01-01 00:00:00上测试起始点
    origin = Timestamp(0)
    # 使用调整后的起始点创建分组器
    adjusted_grouper = pd.Grouper(freq=freq, origin=origin)
    # 对时间序列按调整后的分组器进行分组，并聚合计数
    adjusted_count_ts = ts.groupby(adjusted_grouper).agg("count")
    # 对切片后的时间序列再次使用调整后的分组器进行分组，并聚合计数
    adjusted_count_ts = adjusted_count_ts[middle:end]
    # 断言两个 Series 对象是否相等
    adjusted_count_ts2 = ts2.groupby(adjusted_grouper).agg("count")
    tm.assert_series_equal(adjusted_count_ts, adjusted_count_ts2)

    # 在2049-10-18 20:00:00上测试起始点
    origin_future = Timestamp(0) + pd.Timedelta("1399min") * 30_000
    # 使用未来调整后的起始点创建分组器
    adjusted_grouper2 = pd.Grouper(freq=freq, origin=origin_future)
    # 对时间序列按未来调整后的分组器进行分组，并聚合计数
    adjusted2_count_ts = ts.groupby(adjusted_grouper2).agg("count")
    # 对切片后的时间序列再次使用未来调整后的分组器进行分组，并聚合计数
    adjusted2_count_ts = adjusted2_count_ts[middle:end]
    # 断言两个 Series 对象是否相等
    adjusted2_count_ts2 = ts2.groupby(adjusted_grouper2).agg("count")
    tm.assert_series_equal(adjusted2_count_ts, adjusted2_count_ts2)

    # 两个分组器都使用了调整后的时间戳，是1399分钟的倍数
    # 即使调整后的时间戳在未来，它们也应该相等
    tm.assert_series_equal(adjusted_count_ts, adjusted2_count_ts2)


def test_nearest():
    # GH 17496
    # 最近重新采样
    index = date_range("1/1/2000", periods=3, freq="min")
    # 创建时间序列，并对其进行最近的重新采样为每20秒
    result = Series(range(3), index=index).resample("20s").nearest()
    # 创建预期的 Series 对象，包含指定的值和索引
    expected = Series(
        [0, 0, 1, 1, 1, 2, 2],  # Series 对象的值列表
        index=pd.DatetimeIndex(  # 使用 pd.DatetimeIndex 创建时间索引
            [
                "2000-01-01 00:00:00",
                "2000-01-01 00:00:20",
                "2000-01-01 00:00:40",
                "2000-01-01 00:01:00",
                "2000-01-01 00:01:20",
                "2000-01-01 00:01:40",
                "2000-01-01 00:02:00",
            ],
            dtype="datetime64[ns]",  # 时间索引的数据类型
            freq="20s",  # 时间索引的频率
        ),
    )
    # 使用测试框架中的 assert_series_equal 方法验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "f",
    [
        "first",
        "last",
        "median",
        "sem",
        "sum",
        "mean",
        "min",
        "max",
        "size",
        "count",
        "nearest",
        "bfill",
        "ffill",
        "asfreq",
        "ohlc",
    ],
)
# 定义一个参数化测试函数，用于测试不同的聚合函数和DataFrameGroupBy对象
def test_methods(f, test_frame):
    # 根据列'A'对测试数据进行分组
    g = test_frame.groupby("A")
    # 对分组结果进行时间重采样，采样频率为2秒
    r = g.resample("2s")

    # 设置警告消息，用于检测是否产生特定的警告
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 使用getattr动态调用r对象的方法f，并获取结果
        result = getattr(r, f)()
    
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象g应用lambda函数，动态调用时间重采样后的对象r的方法f，并获取结果
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)())
    
    # 断言结果与期望值相等
    tm.assert_equal(result, expected)


def test_methods_nunique(test_frame):
    # 只针对Series对象进行测试
    # 根据列'A'对测试数据进行分组
    g = test_frame.groupby("A")
    # 对分组结果进行时间重采样，采样频率为2秒
    r = g.resample("2s")
    # 获取时间重采样后的B列数据的唯一值数量
    result = r.B.nunique()
    # 对分组对象g应用lambda函数，获取时间重采样后的B列数据的唯一值数量
    expected = g.B.apply(lambda x: x.resample("2s").nunique())
    # 断言Series对象的结果与期望值相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("f", ["std", "var"])
# 参数化测试函数，用于测试标准差和方差函数
def test_methods_std_var(f, test_frame):
    # 根据列'A'对测试数据进行分组
    g = test_frame.groupby("A")
    # 对分组结果进行时间重采样，采样频率为2秒
    r = g.resample("2s")
    # 设置警告消息，用于检测是否产生特定的警告
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 使用getattr动态调用r对象的方法f，并获取结果
        result = getattr(r, f)(ddof=1)
    
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象g应用lambda函数，动态调用时间重采样后的对象r的方法f，并获取结果
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)(ddof=1))
    
    # 断言DataFrame对象的结果与期望值相等
    tm.assert_frame_equal(result, expected)


def test_apply(test_frame):
    # 根据列'A'对测试数据进行分组
    g = test_frame.groupby("A")
    # 对分组结果进行时间重采样，采样频率为2秒
    r = g.resample("2s")

    # reduction
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 获取时间重采样后的数据求和结果
        expected = g.resample("2s").sum()

    # 定义一个函数f_0，对时间重采样后的数据进行求和操作
    def f_0(x):
        return x.resample("2s").sum()

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象r应用函数f_0，并获取结果
        result = r.apply(f_0)
    
    # 定义一个函数f_1，对时间重采样后的数据应用lambda函数进行求和操作
    def f_1(x):
        return x.resample("2s").apply(lambda y: y.sum())

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言调用特定方法时产生了特定类型的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象g应用函数f_1，并获取结果
        result = g.apply(f_1)
    
    # 将期望值的数据类型转换为int64，因为y.sum()在32位架构上会产生int64类型
    expected = expected.astype("int64")
    # 断言DataFrame对象的结果与期望值相等
    tm.assert_frame_equal(result, expected)


def test_apply_with_mutated_index():
    # GH 15169
    # 创建一个日期索引
    index = date_range("1-1-2015", "12-31-15", freq="D")
    # 创建一个DataFrame对象，包含随机数据和日期索引
    df = DataFrame(
        data={"col1": np.random.default_rng(2).random(len(index))}, index=index
    )

    # 定义一个函数f，返回一个Series对象
    def f(x):
        s = Series([1, 2], index=["a", "b"])
        return s

    # 使用分组器Grouper按月对数据进行分组，并应用函数f
    expected = df.groupby(pd.Grouper(freq="ME")).apply(f)

    # 使用时间重采样按月对数据应用函数f
    result = df.resample("ME").apply(f)
    tm.assert_frame_equal(result, expected)
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等

    # A case for series
    # 下面是针对 Series 的测试案例

    expected = df["col1"].groupby(pd.Grouper(freq="ME"), group_keys=False).apply(f)
    # 计算按照每月结束 (ME) 频率分组的 df["col1"] 的预期结果，应用函数 f

    result = df["col1"].resample("ME").apply(f)
    # 使用每月结束 (ME) 频率重新采样 df["col1"]，应用函数 f

    tm.assert_series_equal(result, expected)
    # 使用测试工具 tm.assert_series_equal 检查 resample 后的 result 和预期的 expected 是否相等
def test_apply_columns_multilevel():
    # GH 16231
    # 创建一个多级索引的列
    cols = pd.MultiIndex.from_tuples([("A", "a", "", "one"), ("B", "b", "i", "two")])
    # 创建一个日期时间索引
    ind = date_range(start="2017-01-01", freq="15Min", periods=8)
    # 创建一个DataFrame，使用cols作为列索引，ind作为行索引，初始化数据为0
    df = DataFrame(np.array([0] * 16).reshape(8, 2), index=ind, columns=cols)
    # 根据列名的最后一个元素是否为"one"来选择聚合函数，形成聚合字典
    agg_dict = {col: (np.sum if col[3] == "one" else np.mean) for col in df.columns}
    # 对DataFrame进行按小时重采样，并应用agg_dict中定义的聚合函数
    result = df.resample("h").apply(lambda x: agg_dict[x.name](x))
    # 创建预期的DataFrame，索引为每小时的时间戳，列为多级列索引
    expected = DataFrame(
        2 * [[0, 0.0]],
        index=date_range(start="2017-01-01", freq="1h", periods=2),
        columns=pd.MultiIndex.from_tuples(
            [("A", "a", "", "one"), ("B", "b", "i", "two")]
        ),
    )
    # 断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_apply_non_naive_index():
    # 定义一个加权分位数函数
    def weighted_quantile(series, weights, q):
        # 对序列进行排序
        series = series.sort_values()
        # 计算加权累计和
        cumsum = weights.reindex(series.index).fillna(0).cumsum()
        # 计算分位点
        cutoff = cumsum.iloc[-1] * q
        # 返回符合条件的第一个元素
        return series[cumsum >= cutoff].iloc[0]

    # 创建时间索引
    times = date_range("2017-6-23 18:00", periods=8, freq="15min", tz="UTC")
    # 创建数据Series
    data = Series([1.0, 1, 1, 1, 1, 2, 2, 0], index=times)
    # 创建权重Series
    weights = Series([160.0, 91, 65, 43, 24, 10, 1, 0], index=times)

    # 对数据Series进行按天重采样，并应用weighted_quantile函数
    result = data.resample("D").apply(weighted_quantile, weights=weights, q=0.5)
    # 创建预期的Series，索引为指定日期的时间戳
    ind = date_range(
        "2017-06-23 00:00:00+00:00", "2017-06-23 00:00:00+00:00", freq="D", tz="UTC"
    )
    expected = Series([1.0], index=ind)
    # 断言结果Series与预期Series相等
    tm.assert_series_equal(result, expected)


def test_resample_groupby_with_label(unit):
    # GH 13235
    # 创建时间索引
    index = date_range("2000-01-01", freq="2D", periods=5, unit=unit)
    # 创建DataFrame，指定index和数据列
    df = DataFrame(index=index, data={"col0": [0, 0, 1, 1, 2], "col1": [1, 1, 1, 1, 1]})
    # 生成警告消息
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 断言产生特定警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame进行分组后按周重采样，使用左闭右开标签
        result = df.groupby("col0").resample("1W", label="left").sum()

    # 创建多级索引的预期DataFrame
    mi = [
        np.array([0, 0, 1, 2], dtype=np.int64),
        np.array(
            ["1999-12-26", "2000-01-02", "2000-01-02", "2000-01-02"],
            dtype=f"M8[{unit}]",
        ),
    ]
    mindex = pd.MultiIndex.from_arrays(mi, names=["col0", None])
    expected = DataFrame(
        data={"col0": [0, 0, 2, 2], "col1": [1, 1, 2, 1]}, index=mindex
    )

    # 断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_consistency_with_window(test_frame):
    # consistent return values with window
    # 获取测试DataFrame
    df = test_frame
    # 预期的索引
    expected = Index([1, 2, 3], name="A")
    # 生成警告消息
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 断言产生特定警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按'A'列分组后进行窗口平均值计算
        result = df.groupby("A").resample("2s").mean()
    # 断言结果的索引级别数量为2
    assert result.index.nlevels == 2
    # 断言结果的第一个索引级别与预期的索引相等
    tm.assert_index_equal(result.index.levels[0], expected)

    # 对DataFrame按'A'列分组后进行窗口平均值计算
    result = df.groupby("A").rolling(20).mean()
    # 断言结果的索引级别数量为2
    assert result.index.nlevels == 2
    # 断言结果的第一个索引级别与预期的索引相等
    tm.assert_index_equal(result.index.levels[0], expected)


def test_median_duplicate_columns():
    # GH 14233
    # 这个测试函数还没有实现内容，待补充
    # 创建一个 DataFrame 对象 df，其中包含20行和3列的随机标准正态分布数据
    df = DataFrame(
        np.random.default_rng(2).standard_normal((20, 3)),
        columns=list("aaa"),
        index=date_range("2012-01-01", periods=20, freq="s"),
    )
    
    # 使用 resample 方法对 df 进行重采样，将时间间隔从每秒钟变为每5秒钟，并计算每个5秒钟内数据的中位数，将结果保存在 result 中
    result = df.resample("5s").median()
    
    # 修改 df 的列名，将原来的列名 "aaa" 修改为 ["a", "b", "c"]
    df.columns = ["a", "b", "c"]
    
    # 重新对修改后的 df 进行重采样，将时间间隔从每秒钟变为每5秒钟，并计算每个5秒钟内数据的中位数，将结果保存在 expected 中
    expected = df.resample("5s").median()
    
    # 将 expected 的列名修改为与 result 相同的列名，确保两个 DataFrame 在比较时列名一致
    expected.columns = result.columns
    
    # 使用 tm.assert_frame_equal 方法比较 result 和 expected 两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(result, expected)
def test_apply_to_one_column_of_df():
    # GH: 36951
    # 创建一个 DataFrame，包含两列数据 "col" 和 "col1"，以及日期索引
    df = DataFrame(
        {"col": range(10), "col1": range(10, 20)},
        index=date_range("2012-01-01", periods=10, freq="20min"),
    )

    # 使用 resample 方法对数据进行重采样为每小时，并应用 lambda 函数计算 "col" 列的和
    result = df.resample("h").apply(lambda group: group.col.sum())
    # 期望的结果是一个 Series，包含指定日期范围内每小时 "col" 列的和
    expected = Series(
        [3, 12, 21, 9], index=date_range("2012-01-01", periods=4, freq="h")
    )
    # 使用 assert_series_equal 检查计算结果和期望结果是否一致
    tm.assert_series_equal(result, expected)

    # 使用 resample 方法再次重采样为每小时，并应用 lambda 函数计算 "col" 列的和，使用字典索引方式访问
    result = df.resample("h").apply(lambda group: group["col"].sum())
    # 再次检查计算结果和期望结果是否一致
    tm.assert_series_equal(result, expected)


def test_resample_groupby_agg():
    # GH: 33548
    # 创建一个包含分类列 "cat"、数值列 "num" 和日期列 "date" 的 DataFrame
    df = DataFrame(
        {
            "cat": [
                "cat_1",
                "cat_1",
                "cat_2",
                "cat_1",
                "cat_2",
                "cat_1",
                "cat_2",
                "cat_1",
            ],
            "num": [5, 20, 22, 3, 4, 30, 10, 50],
            "date": [
                "2019-2-1",
                "2018-02-03",
                "2020-3-11",
                "2019-2-2",
                "2019-2-2",
                "2018-12-4",
                "2020-3-11",
                "2020-12-12",
            ],
        }
    )
    # 将 "date" 列转换为 datetime 类型
    df["date"] = pd.to_datetime(df["date"])

    # 对 "cat" 列进行分组，并在日期上进行年度（YE）重采样
    resampled = df.groupby("cat").resample("YE", on="date")
    # 期望的结果是每个类别的数值列 "num" 的年度总和
    expected = resampled[["num"]].sum()
    # 使用 agg 方法计算数值列 "num" 的总和
    result = resampled.agg({"num": "sum"})

    # 使用 assert_frame_equal 检查计算结果和期望结果是否一致
    tm.assert_frame_equal(result, expected)


def test_resample_groupby_agg_listlike():
    # GH 42905
    # 创建一个包含单个值的 DataFrame，其中类别为 "beta"，数值为 69
    ts = Timestamp("2021-02-28 00:00:00")
    df = DataFrame({"class": ["beta"], "value": [69]}, index=Index([ts], name="date"))
    # 对类别 "class" 进行分组，并在日期上进行每月（ME）重采样，对数值列 "value" 应用 sum 和 size 聚合
    resampled = df.groupby("class").resample("ME")["value"]
    # 期望的结果是一个包含 sum 和 size 列的 DataFrame
    expected = DataFrame(
        [[69, 1]],
        index=pd.MultiIndex.from_tuples([("beta", ts)], names=["class", "date"]),
        columns=["sum", "size"],
    )
    # 使用 assert_frame_equal 检查计算结果和期望结果是否一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_empty(keys):
    # GH 26411
    # 创建一个空的 DataFrame，具有列名 "a" 和 "b"，以及空的时间增量索引
    df = DataFrame([], columns=["a", "b"], index=TimedeltaIndex([]))
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 使用 assert_produces_warning 检查是否产生 DeprecationWarning 警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组键进行分组，并在时间上进行每秒（00:00:01）的重采样，计算平均值
        result = df.groupby(keys).resample(rule=pd.to_timedelta("00:00:01")).mean()
    # 创建一个期望的空 DataFrame
    expected = (
        DataFrame(columns=["a", "b"])
        .set_index(keys, drop=False)
        .set_index(TimedeltaIndex([]), append=True)
    )
    if len(keys) == 1:
        expected.index.name = keys[0]

    # 使用 assert_frame_equal 检查计算结果和期望结果是否一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("consolidate", [True, False])
def test_resample_groupby_agg_object_dtype_all_nan(consolidate):
    # https://github.com/pandas-dev/pandas/issues/39329

    # 创建一个日期范围从 "2020-01-01" 开始，共 15 天，频率为每日的时间索引
    dates = date_range("2020-01-01", periods=15, freq="D")
    # 创建第一个 DataFrame，包含固定的键 'key' 和 'date'，以及列 'col1' 和 'col_object'
    df1 = DataFrame({"key": "A", "date": dates, "col1": range(15), "col_object": "val"})
    # 创建第二个 DataFrame，只包含 'key'、'date' 和 'col1' 列
    df2 = DataFrame({"key": "B", "date": dates, "col1": range(15)})
    # 将 df1 和 df2 按行连接成一个新的 DataFrame，忽略原始索引
    df = pd.concat([df1, df2], ignore_index=True)
    # 如果 consolidate 为 True，则调用 _consolidate() 方法以压缩数据
    if consolidate:
        df = df._consolidate()

    # 设置警告消息内容
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 使用 assert_produces_warning 确保在执行下面代码块时产生 DeprecationWarning 警告，且警告信息与 msg 匹配
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 'key' 列分组，并对 'date' 列进行周频率重采样，使用每组的最小值
        result = df.groupby(["key"]).resample("W", on="date").min()

    # 创建一个 MultiIndex，其中包含两个级别，分别是 'key' 和 'date'，并且 'date' 级别转换为纳秒单位
    idx = pd.MultiIndex.from_arrays(
        [
            ["A"] * 3 + ["B"] * 3,
            pd.to_datetime(["2020-01-05", "2020-01-12", "2020-01-19"] * 2).as_unit(
                "ns"
            ),
        ],
        names=["key", "date"],
    )
    # 创建预期的 DataFrame，包含 'key'、'col1' 和 'col_object' 列，并设置其索引为 idx
    expected = DataFrame(
        {
            "key": ["A"] * 3 + ["B"] * 3,
            "col1": [0, 5, 12] * 2,
            "col_object": ["val"] * 3 + [np.nan] * 3,
        },
        index=idx,
    )
    # 使用 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_groupby_resample_with_list_of_keys():
    # GH 47362
    # 创建一个包含日期、分组和数值的DataFrame对象
    df = DataFrame(
        data={
            "date": date_range(start="2016-01-01", periods=8),
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "val": [1, 7, 5, 2, 3, 10, 5, 1],
        }
    )
    # 对分组后的数据按照日期进行2天为间隔的重采样，并计算每组的均值
    result = df.groupby("group").resample("2D", on="date")[["val"]].mean()

    # 构建预期的多级索引，包括分组和日期
    mi_exp = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], df["date"]._values[::2]], names=["group", "date"]
    )
    # 创建预期结果的DataFrame对象，包括均值数据，使用预期的多级索引
    expected = DataFrame(
        data={
            "val": [4.0, 3.5, 6.5, 3.0],
        },
        index=mi_exp,
    )
    # 检验实际结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_resample_no_index(keys):
    # GH 47705
    # 创建一个空的DataFrame对象，包括列"a", "b"和"date"
    df = DataFrame([], columns=["a", "b", "date"])
    # 将"date"列转换为日期时间类型
    df["date"] = pd.to_datetime(df["date"])
    # 将"date"列设置为索引
    df = df.set_index("date")
    # 设置警告信息的匹配模式
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 检验是否会产生DeprecationWarning警告，并匹配警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组后的数据按照指定时间间隔重采样，并计算均值
        result = df.groupby(keys).resample(rule=pd.to_timedelta("00:00:01")).mean()
    # 创建预期结果的DataFrame对象，包括列"a", "b"和"date"，设置合适的索引
    expected = DataFrame(columns=["a", "b", "date"]).set_index(keys, drop=False)
    # 将"date"列转换为日期时间类型
    expected["date"] = pd.to_datetime(expected["date"])
    # 将"date"列设置为索引，并追加到现有索引中
    expected = expected.set_index("date", append=True, drop=True)
    # 如果keys长度为1，设置预期结果的索引名称
    if len(keys) == 1:
        expected.index.name = keys[0]
    # 检验实际结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


def test_resample_no_columns():
    # GH#52484
    # 创建一个空的DataFrame对象，设置时间索引为一组日期时间
    df = DataFrame(
        index=Index(
            pd.to_datetime(
                ["2018-01-01 00:00:00", "2018-01-01 12:00:00", "2018-01-02 00:00:00"]
            ),
            name="date",
        )
    )
    # 对分组后的数据按照指定时间间隔重采样，并计算均值
    result = df.groupby([0, 0, 1]).resample(rule=pd.to_timedelta("06:00:00")).mean()
    # 创建预期结果的日期时间索引
    index = pd.to_datetime(
        [
            "2018-01-01 00:00:00",
            "2018-01-01 06:00:00",
            "2018-01-01 12:00:00",
            "2018-01-02 00:00:00",
        ]
    )
    # 创建预期结果的DataFrame对象，设置合适的多级索引
    expected = DataFrame(
        index=pd.MultiIndex(
            levels=[np.array([0, 1], dtype=np.intp), index],
            codes=[[0, 0, 0, 1], [0, 1, 2, 3]],
            names=[None, "date"],
        )
    )

    # GH#52710 - Index comes out as 32-bit on 64-bit Windows
    # 检验实际结果和预期结果是否相等，同时检查索引类型是否需要验证（在非Windows平台上）
    tm.assert_frame_equal(result, expected, check_index_type=not is_platform_windows())


def test_groupby_resample_size_all_index_same():
    # GH 46826
    # 创建一个包含"A"和"B"列的DataFrame对象，设置时间索引为一组日期时间
    df = DataFrame(
        {"A": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3, "B": np.arange(12)},
        index=date_range("31/12/2000 18:00", freq="h", periods=12),
    )
    # 设置警告信息的匹配模式
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 检验是否会产生DeprecationWarning警告，并匹配警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组后的数据按照天进行重采样，并计算大小
        result = df.groupby("A").resample("D").size()

    # 创建预期结果的多级索引，包括"A"和日期时间
    mi_exp = pd.MultiIndex.from_arrays(
        [
            [1, 1, 2, 2],
            pd.DatetimeIndex(["2000-12-31", "2001-01-01"] * 2, dtype="M8[ns]"),
        ],
        names=["A", None],
    )
    # 创建预期结果的Series对象，设置合适的索引
    expected = Series(
        3,
        index=mi_exp,
    )
    # 使用 pandas.testing 模块中的 assert_series_equal 函数来比较 result 和 expected 两个序列是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于验证在索引上使用列表键对数据框进行分组和重新采样的功能
def test_groupby_resample_on_index_with_list_of_keys():
    # GH 50840
    # 创建一个包含"group"和"val"列的数据框，同时指定日期索引从"2016-01-01"开始，共8个时间点
    df = DataFrame(
        data={
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "val": [3, 1, 4, 1, 5, 9, 2, 6],
        },
        index=date_range(start="2016-01-01", periods=8, name="date"),
    )
    # 对数据框按"group"列进行分组，然后对每组数据以2天为间隔重新采样，计算"val"列的均值
    result = df.groupby("group").resample("2D")[["val"]].mean()

    # 创建预期的多级索引，包含组别和每隔2天的日期索引
    mi_exp = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], df.index[::2]], names=["group", "date"]
    )
    # 创建预期的数据框，包含"val"列的均值
    expected = DataFrame(
        data={
            "val": [2.0, 2.5, 7.0, 4.0],
        },
        index=mi_exp,
    )
    # 使用测试框架验证结果数据框与预期数据框是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，验证在索引上使用列表键对具有多列的数据框进行分组和重新采样的功能
def test_groupby_resample_on_index_with_list_of_keys_multi_columns():
    # GH 50876
    # 创建一个包含"group", "first_val", "second_val", "third_val"列的数据框，
    # 同时指定日期索引从"2016-01-01"开始，共8个时间点
    df = DataFrame(
        data={
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "first_val": [3, 1, 4, 1, 5, 9, 2, 6],
            "second_val": [2, 7, 1, 8, 2, 8, 1, 8],
            "third_val": [1, 4, 1, 4, 2, 1, 3, 5],
        },
        index=date_range(start="2016-01-01", periods=8, name="date"),
    )
    # 对数据框按"group"列进行分组，然后对每组数据以2天为间隔重新采样，计算"first_val"和"second_val"列的均值
    result = df.groupby("group").resample("2D")[["first_val", "second_val"]].mean()

    # 创建预期的多级索引，包含组别和每隔2天的日期索引
    mi_exp = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], df.index[::2]], names=["group", "date"]
    )
    # 创建预期的数据框，包含"first_val"和"second_val"列的均值
    expected = DataFrame(
        data={
            "first_val": [2.0, 2.5, 7.0, 4.0],
            "second_val": [4.5, 4.5, 5.0, 4.5],
        },
        index=mi_exp,
    )
    # 使用测试框架验证结果数据框与预期数据框是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，验证在索引上使用列表键对数据框进行分组和重新采样时，处理缺少列的情况
def test_groupby_resample_on_index_with_list_of_keys_missing_column():
    # GH 50876
    # 创建一个包含"group"和"val"列的数据框，
    # 同时指定日期索引从"2016-01-01"开始，共8个时间点
    df = DataFrame(
        data={
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "val": [3, 1, 4, 1, 5, 9, 2, 6],
        },
        index=Series(
            date_range(start="2016-01-01", periods=8),
            name="date",
        ),
    )
    # 对数据框按"group"列进行分组
    gb = df.groupby("group")
    # 对每组数据以2天为间隔重新采样，期望抛出"Columns not found"的错误，匹配给定的正则表达式
    rs = gb.resample("2D")
    with pytest.raises(KeyError, match="Columns not found"):
        # 尝试访问不存在的"val_not_in_dataframe"列
        rs[["val_not_in_dataframe"]]
```