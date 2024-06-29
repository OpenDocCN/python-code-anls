# `D:\src\scipysrc\pandas\pandas\tests\resample\test_resample_api.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 导入 re 模块，用于正则表达式操作
import re

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas._libs 包中导入 lib 模块
from pandas._libs import lib

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 从 pandas 包中导入 DataFrame、NamedAgg、Series 类
from pandas import (
    DataFrame,
    NamedAgg,
    Series,
)
# 导入 pandas._testing 包，并使用 tm 别名
import pandas._testing as tm
# 从 pandas.core.indexes.datetimes 包中导入 date_range 函数
from pandas.core.indexes.datetimes import date_range

# 定义 pytest 的 fixture，生成一个 DatetimeIndex
@pytest.fixture
def dti():
    return date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="Min")

# 定义 pytest 的 fixture，生成一个随机 Series 对象
@pytest.fixture
def _test_series(dti):
    return Series(np.random.default_rng(2).random(len(dti)), dti)

# 定义 pytest 的 fixture，生成一个 DataFrame 对象
@pytest.fixture
def test_frame(dti, _test_series):
    return DataFrame({"A": _test_series, "B": _test_series, "C": np.arange(len(dti))})

# 测试函数，验证 Series 对象的字符串表示是否包含特定文本
def test_str(_test_series):
    r = _test_series.resample("h")
    assert (
        "DatetimeIndexResampler [freq=<Hour>, closed=left, "
        "label=left, convention=start, origin=start_day]" in str(r)
    )

    r = _test_series.resample("h", origin="2000-01-01")
    assert (
        "DatetimeIndexResampler [freq=<Hour>, closed=left, "
        "label=left, convention=start, origin=2000-01-01 00:00:00]" in str(r)
    )

# 测试函数，验证 Series 对象的 resample 方法的功能
def test_api(_test_series):
    r = _test_series.resample("h")
    result = r.mean()
    assert isinstance(result, Series)
    assert len(result) == 217

    r = _test_series.to_frame().resample("h")
    result = r.mean()
    assert isinstance(result, DataFrame)
    assert len(result) == 217

# 测试函数，验证 GroupBy 对象与 resample 结合使用时的行为
def test_groupby_resample_api():
    # GH 12448
    # .groupby(...).resample(...) hitting warnings
    # when appropriate
    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W"),
            "group": [1, 1, 2, 2],
            "val": [5, 6, 7, 8],
        }
    ).set_index("date")

    # replication step
    i = (
        date_range("2016-01-03", periods=8).tolist()
        + date_range("2016-01-17", periods=8).tolist()
    )
    index = pd.MultiIndex.from_arrays([[1] * 8 + [2] * 8, i], names=["group", "date"])
    expected = DataFrame({"val": [5] * 7 + [6] + [7] * 7 + [8]}, index=index)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby("group").apply(lambda x: x.resample("1D").ffill())[["val"]]
    tm.assert_frame_equal(result, expected)

# 测试函数，验证 GroupBy 对象与 resample 结合使用时的行为
def test_groupby_resample_on_api():
    # GH 15021
    # .groupby(...).resample(on=...) results in an unexpected
    # keyword warning.
    df = DataFrame(
        {
            "key": ["A", "B"] * 5,
            "dates": date_range("2016-01-01", periods=10),
            "values": np.random.default_rng(2).standard_normal(10),
        }
    )

    expected = df.set_index("dates").groupby("key").resample("D").mean()
    result = df.groupby("key").resample("D", on="dates").mean()
    tm.assert_frame_equal(result, expected)

# 测试函数，验证 DataFrame 对象的 resample 方法在不生成分组键的情况下的行为
def test_resample_group_keys():
    df = DataFrame({"A": 1, "B": 2}, index=date_range("2000", periods=10))
    expected = df.copy()

    # group_keys=False
    g = df.resample("5D", group_keys=False)
    # 对DataFrame进行分组重采样，并应用lambda函数，将结果赋给result变量
    result = g.apply(lambda x: x)
    # 使用测试工具tm.assert_frame_equal检查result与期望值expected是否相等
    tm.assert_frame_equal(result, expected)

    # 当group_keys参数未指定时，默认为False
    g = df.resample("5D")
    # 再次对DataFrame进行分组重采样，并应用lambda函数，将结果赋给result变量
    result = g.apply(lambda x: x)
    # 使用测试工具tm.assert_frame_equal检查result与期望值expected是否相等
    tm.assert_frame_equal(result, expected)

    # 当指定group_keys=True时
    # 将expected的索引设置为一个多级索引，第一级是每5天重复的日期时间，第二级是原始的索引
    expected.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(["2000-01-01", "2000-01-06"]).as_unit("ns").repeat(5),
            expected.index,
        ]
    )
    g = df.resample("5D", group_keys=True)
    # 再次对DataFrame进行分组重采样，并应用lambda函数，将结果赋给result变量
    result = g.apply(lambda x: x)
    # 使用测试工具tm.assert_frame_equal检查result与期望值expected是否相等
    tm.assert_frame_equal(result, expected)
def test_pipe(test_frame, _test_series):
    # GH17905

    # series
    r = _test_series.resample("h")  # 对时间序列进行小时级别的重采样
    expected = r.max() - r.mean()   # 计算重采样后的最大值与平均值的差
    result = r.pipe(lambda x: x.max() - x.mean())  # 使用 pipe 方法计算同样的结果
    tm.assert_series_equal(result, expected)  # 断言两个序列是否相等

    # dataframe
    r = test_frame.resample("h")  # 对数据框进行小时级别的重采样
    expected = r.max() - r.mean()  # 计算重采样后的最大值与平均值的差
    result = r.pipe(lambda x: x.max() - x.mean())  # 使用 pipe 方法计算同样的结果
    tm.assert_frame_equal(result, expected)  # 断言两个数据框是否相等


def test_getitem(test_frame):
    r = test_frame.resample("h")  # 对数据框进行小时级别的重采样
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns)  # 断言重采样后的列索引与原始数据框的列索引是否相等

    r = test_frame.resample("h")["B"]  # 对数据框按小时重采样并选择列 'B'
    assert r._selected_obj.name == test_frame.columns[1]  # 断言选择的列名称是否为原始数据框的第二列名称

    # technically this is allowed
    r = test_frame.resample("h")["A", "B"]  # 对数据框按小时重采样并选择列 'A' 和 'B'
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])  # 断言选择的列索引是否与原始数据框的第一列和第二列索引相等

    r = test_frame.resample("h")["A", "B"]  # 对数据框按小时重采样并选择列 'A' 和 'B'
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])  # 断言选择的列索引是否与原始数据框的第一列和第二列索引相等


@pytest.mark.parametrize("key", [["D"], ["A", "D"]])
def test_select_bad_cols(key, test_frame):
    g = test_frame.resample("h")  # 对数据框进行小时级别的重采样
    # 'A' should not be referenced as a bad column...
    # will have to rethink regex if you change message!
    msg = r"^\"Columns not found: 'D'\"$"
    with pytest.raises(KeyError, match=msg):  # 使用 pytest 断言引发 KeyError，并匹配指定消息正则表达式
        g[key]


def test_attribute_access(test_frame):
    r = test_frame.resample("h")  # 对数据框进行小时级别的重采样
    tm.assert_series_equal(r.A.sum(), r["A"].sum())  # 断言对列 'A' 进行求和的结果是否相等


@pytest.mark.parametrize("attr", ["groups", "ngroups", "indices"])
def test_api_compat_before_use(attr):
    # make sure that we are setting the binner
    # on these attributes
    rng = date_range("1/1/2012", periods=100, freq="s")  # 创建一个时间索引
    ts = Series(np.arange(len(rng)), index=rng)  # 创建一个时间序列
    rs = ts.resample("30s")  # 对时间序列进行30秒级别的重采样

    # before use
    getattr(rs, attr)  # 获取重采样后的对象的指定属性

    # after grouper is initialized is ok
    rs.mean()  # 对重采样后的对象进行均值计算
    getattr(rs, attr)  # 再次获取重采样后的对象的指定属性


def tests_raises_on_nuisance(test_frame):
    df = test_frame
    df["D"] = "foo"  # 在数据框中创建一个新列 'D'，并填充值 'foo'
    r = df.resample("h")  # 对数据框进行小时级别的重采样
    result = r[["A", "B"]].mean()  # 计算重采样后的 'A' 和 'B' 列的均值
    expected = pd.concat([r.A.mean(), r.B.mean()], axis=1)  # 创建预期的均值数据框
    tm.assert_frame_equal(result, expected)  # 断言两个数据框是否相等

    expected = r[["A", "B", "C"]].mean()  # 计算重采样后的 'A'、'B' 和 'C' 列的均值
    msg = re.escape("agg function failed [how->mean,dtype->")  # 创建用于匹配异常消息的正则表达式
    with pytest.raises(TypeError, match=msg):  # 使用 pytest 断言引发 TypeError，并匹配指定消息正则表达式
        r.mean()
    result = r.mean(numeric_only=True)  # 计算重采样后的数值列的均值
    tm.assert_frame_equal(result, expected)  # 断言两个数据框是否相等


def test_downsample_but_actually_upsampling():
    # this is reindex / asfreq
    rng = date_range("1/1/2012", periods=100, freq="s")  # 创建一个时间索引
    ts = Series(np.arange(len(rng), dtype="int64"), index=rng)  # 创建一个时间序列
    result = ts.resample("20s").asfreq()  # 对时间序列进行20秒级别的重采样，并转换为频率（asfreq）
    expected = Series(
        [0, 20, 40, 60, 80],  # 预期结果的值列表
        index=date_range("2012-01-01 00:00:00", freq="20s", periods=5),  # 预期结果的时间索引
    )
    tm.assert_series_equal(result, expected)  # 断言两个时间序列是否相等


def test_combined_up_downsampling_of_irregular():
    # since we are really doing an operation like this
    # ts2.resample('2s').mean().ffill()
    # preserve these semantics

    rng = date_range("1/1/2012", periods=100, freq="s")  # 创建一个时间索引
    # 创建一个时间序列 `ts`，其索引是 `rng` 的长度，值是从 0 到长度减一的整数序列
    ts = Series(np.arange(len(rng)), index=rng)
    # 从 `ts` 中选择特定索引位置的数据，创建一个新的时间序列 `ts2`
    ts2 = ts.iloc[[0, 1, 2, 3, 5, 7, 11, 15, 16, 25, 30]]
    
    # 对时间序列 `ts2` 进行重新采样，每2秒取均值，并用前向填充的方式填充缺失值，得到结果 `result`
    result = ts2.resample("2s").mean().ffill()
    
    # 创建一个预期结果的时间序列 `expected`
    expected = Series(
        [
            0.5,
            2.5,
            5.0,
            7.0,
            7.0,
            11.0,
            11.0,
            15.0,
            16.0,
            16.0,
            16.0,
            16.0,
            25.0,
            25.0,
            25.0,
            30.0,
        ],
        index=pd.DatetimeIndex(
            [
                "2012-01-01 00:00:00",
                "2012-01-01 00:00:02",
                "2012-01-01 00:00:04",
                "2012-01-01 00:00:06",
                "2012-01-01 00:00:08",
                "2012-01-01 00:00:10",
                "2012-01-01 00:00:12",
                "2012-01-01 00:00:14",
                "2012-01-01 00:00:16",
                "2012-01-01 00:00:18",
                "2012-01-01 00:00:20",
                "2012-01-01 00:00:22",
                "2012-01-01 00:00:24",
                "2012-01-01 00:00:26",
                "2012-01-01 00:00:28",
                "2012-01-01 00:00:30",
            ],
            dtype="datetime64[ns]",
            freq="2s",
        ),
    )
    
    # 使用 `tm.assert_series_equal` 函数断言 `result` 应当等于 `expected`
    tm.assert_series_equal(result, expected)
# 定义测试函数，用于对时间序列进行转换验证
def test_transform_series(_test_series):
    # 对时间序列进行20分钟的重采样
    r = _test_series.resample("20min")
    # 计算每个时间窗口内的均值作为预期结果
    expected = _test_series.groupby(pd.Grouper(freq="20min")).transform("mean")
    # 对重采样后的结果再次计算均值
    result = r.transform("mean")
    # 使用测试工具函数验证结果是否符合预期
    tm.assert_series_equal(result, expected)


# 使用参数化测试框架pytest.mark.parametrize装饰器定义测试函数
@pytest.mark.parametrize("on", [None, "date"])
def test_transform_frame(on):
    # GH#47079
    # 创建一个日期范围索引
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    # 创建一个包含随机数据的DataFrame，列名为A和B，行索引为日期
    df = DataFrame(
        np.random.default_rng(2).random((10, 2)), columns=list("AB"), index=index
    )
    # 根据20分钟频率对DataFrame进行分组，并计算每组的均值作为预期结果
    expected = df.groupby(pd.Grouper(freq="20min")).transform("mean")
    # 如果on参数为"date"，则将日期索引移动到列，结果DataFrame将具有RangeIndex
    if on == "date":
        expected = expected.reset_index(drop=True)
        df = df.reset_index()

    # 对DataFrame按照20分钟频率进行重采样，根据on参数选择重采样的基准
    r = df.resample("20min", on=on)
    # 对重采样后的结果计算均值
    result = r.transform("mean")
    # 使用测试工具函数验证结果DataFrame是否与预期相等
    tm.assert_frame_equal(result, expected)


# 使用参数化测试框架pytest.mark.parametrize装饰器定义测试函数
@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.resample("20min", group_keys=False),
        lambda x: x.groupby(pd.Grouper(freq="20min"), group_keys=False),
    ],
    ids=["resample", "groupby"],
)
def test_apply_without_aggregation(func, _test_series):
    # 测试resample和groupby是否能够在不进行聚合的情况下正常工作
    t = func(_test_series)
    # 对结果应用一个恒等函数，不改变结果
    result = t.apply(lambda x: x)
    # 使用测试工具函数验证结果是否与原始序列相等
    tm.assert_series_equal(result, _test_series)


# 定义测试函数，验证在不进行聚合的情况下应用函数的一致性
def test_apply_without_aggregation2(_test_series):
    # 将时间序列转为DataFrame，列名为"foo"，并按20分钟频率重采样，不生成组键
    grouped = _test_series.to_frame(name="foo").resample("20min", group_keys=False)
    # 对"foo"列应用一个恒等函数，不改变结果
    result = grouped["foo"].apply(lambda x: x)
    # 使用测试工具函数验证结果是否与原始序列的重命名结果相等
    tm.assert_series_equal(result, _test_series.rename("foo"))


# 定义测试函数，验证聚合操作的一致性
def test_agg_consistency():
    # 确保在相似的聚合操作中选择列表和非选择列表的一致性
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=date_range("1/1/2012", freq="s", periods=1000),
        columns=["A", "B", "C"],
    )

    # 对DataFrame按3分钟频率进行重采样
    r = df.resample("3min")

    # 预期出现KeyError异常，异常信息应包含特定的消息字符串
    msg = r"Label\(s\) \['r1', 'r2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        # 对重采样结果应用聚合函数，要求求解"r1"列的均值和"r2"列的和
        r.agg({"r1": "mean", "r2": "sum"})


# 定义测试函数，验证整数和字符串列混合时的聚合操作的一致性
def test_agg_consistency_int_str_column_mix():
    # GH#39025
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 2)),
        index=date_range("1/1/2012", freq="s", periods=1000),
        columns=[1, "a"],
    )

    # 对DataFrame按3分钟频率进行重采样
    r = df.resample("3min")

    # 预期出现KeyError异常，异常信息应包含特定的消息字符串
    msg = r"Label\(s\) \[2, 'b'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        # 对重采样结果应用聚合函数，要求求解整数列2的均值和字符串列"b"的和
        r.agg({2: "mean", "b": "sum"})


# 使用pytest.fixture装饰器定义测试用例的前置条件
@pytest.fixture
def index():
    # 创建一个日期范围索引，从2005年1月1日到2005年1月10日，每日频率
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    return index


# 使用pytest.fixture装饰器定义测试用例的前置条件
@pytest.fixture
def df(index):
    # 创建一个DataFrame，包含随机数据，列名为A和B，行索引为日期范围索引
    frame = DataFrame(
        np.random.default_rng(2).random((10, 2)), columns=list("AB"), index=index
    )
    return frame


# 使用pytest.fixture装饰器定义测试用例的前置条件
@pytest.fixture
def df_col(df):
    # 将DataFrame重置索引，返回DataFrame
    return df.reset_index()


# 使用pytest.fixture装饰器定义测试用例的前置条件
@pytest.fixture
def df_mult(df_col, index):
    # 这个fixture没有具体的实现代码，留空
    pass
    # 复制 DataFrame df_col，并赋给 df_mult
    df_mult = df_col.copy()
    # 使用 pd.MultiIndex.from_arrays 方法创建多级索引，其中第一级为从 0 到 9 的整数范围，第二级为给定的 index 列表，索引级别命名为 "index" 和 "date"
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), index], names=["index", "date"]
    )
    # 返回具有多级索引的 DataFrame df_mult
    return df_mult
@pytest.fixture
def a_mean(df):
    # 返回 DataFrame `df` 按照每2天重新取样后列 'A' 的平均值
    return df.resample("2D")["A"].mean()


@pytest.fixture
def a_std(df):
    # 返回 DataFrame `df` 按照每2天重新取样后列 'A' 的标准差
    return df.resample("2D")["A"].std()


@pytest.fixture
def a_sum(df):
    # 返回 DataFrame `df` 按照每2天重新取样后列 'A' 的和
    return df.resample("2D")["A"].sum()


@pytest.fixture
def b_mean(df):
    # 返回 DataFrame `df` 按照每2天重新取样后列 'B' 的平均值
    return df.resample("2D")["B"].mean()


@pytest.fixture
def b_std(df):
    # 返回 DataFrame `df` 按照每2天重新取样后列 'B' 的标准差
    return df.resample("2D")["B"].std()


@pytest.fixture
def b_sum(df):
    # 返回 DataFrame `df` 按照每2天重新取样后列 'B' 的和
    return df.resample("2D")["B"].sum()


@pytest.fixture
def df_resample(df):
    # 返回 DataFrame `df` 按照每2天重新取样后的对象
    return df.resample("2D")


@pytest.fixture
def df_col_resample(df_col):
    # 返回 DataFrame `df_col` 按照每2天重新取样后的对象，基于 'date' 列
    return df_col.resample("2D", on="date")


@pytest.fixture
def df_mult_resample(df_mult):
    # 返回多重索引 DataFrame `df_mult` 按照每2天重新取样后的对象，基于 'date' 级别
    return df_mult.resample("2D", level="date")


@pytest.fixture
def df_grouper_resample(df):
    # 返回 DataFrame `df` 按照每2天分组后的对象
    return df.groupby(pd.Grouper(freq="2D"))


@pytest.fixture(
    params=["df_resample", "df_col_resample", "df_mult_resample", "df_grouper_resample"]
)
def cases(request):
    # 根据请求的参数名返回对应的 fixture 对象
    return request.getfixturevalue(request.param)


def test_agg_mixed_column_aggregation(cases, a_mean, a_std, b_mean, b_std, request):
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_product([["A", "B"], ["mean", "<lambda_0>"]])
    # 如果当前测试用例涉及多重索引的 DataFrame
    if "df_mult" in request.node.callspec.id:
        # 计算 'date' 列的平均值和标准差
        date_mean = cases["date"].mean()
        date_std = cases["date"].std()
        expected = pd.concat([date_mean, date_std, expected], axis=1)
        expected.columns = pd.MultiIndex.from_product(
            [["date", "A", "B"], ["mean", "<lambda_0>"]]
        )
    # 对 cases 中的数据进行聚合操作，计算平均值和标准差
    result = cases.aggregate([np.mean, lambda x: np.std(x, ddof=1)])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg",
    [
        {"func": {"A": np.mean, "B": lambda x: np.std(x, ddof=1)}},
        {"A": ("A", np.mean), "B": ("B", lambda x: np.std(x, ddof=1))},
        {"A": NamedAgg("A", np.mean), "B": NamedAgg("B", lambda x: np.std(x, ddof=1))},
    ],
)
def test_agg_both_mean_std_named_result(cases, a_mean, b_std, agg):
    expected = pd.concat([a_mean, b_std], axis=1)
    # 对 cases 中的数据按照指定的聚合方式进行计算
    result = cases.aggregate(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)


def test_agg_both_mean_std_dict_of_list(cases, a_mean, a_std):
    expected = pd.concat([a_mean, a_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([("A", "mean"), ("A", "std")])
    # 对 cases 中的 'A' 列按照 'mean' 和 'std' 进行聚合计算
    result = cases.aggregate({"A": ["mean", "std"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg", [{"func": ["mean", "sum"]}, {"mean": "mean", "sum": "sum"}]
)
def test_agg_both_mean_sum(cases, a_mean, a_sum, agg):
    expected = pd.concat([a_mean, a_sum], axis=1)
    expected.columns = ["mean", "sum"]
    # 对 cases 中的 'A' 列按照 'mean' 和 'sum' 进行聚合计算
    result = cases["A"].aggregate(**agg)
    tm.assert_frame_equal(result, expected)
    [
        # 列表中的第一个字典，包含单个键值对 "A" 对应的字典
        {"A": {"mean": "mean", "sum": "sum"}},
    
        # 列表中的第二个字典，包含两个键值对 "A" 和 "B" 对应的字典
        {
            "A": {"mean": "mean", "sum": "sum"},
            "B": {"mean2": "mean", "sum2": "sum"},
        },
    ],
# 当前测试函数用于验证对于给定的DataFrame `cases`，使用不同的聚合规范是否会引发指定的错误。

def test_agg_dict_of_dict_specificationerror(cases, agg):
    # 设置错误消息内容
    msg = "nested renamer is not supported"
    # 使用 pytest 的断言来检查是否抛出了 pd.errors.SpecificationError 异常，并验证其错误消息
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        # 在 cases 上调用 aggregate 方法，传入参数 agg 进行聚合操作
        cases.aggregate(agg)


def test_agg_dict_of_lists(cases, a_mean, a_std, b_mean, b_std):
    # 创建预期的 DataFrame，将各个列拼接起来
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    # 设置列名为多级索引
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    )
    # 在 cases 上调用 aggregate 方法，使用字典形式的规范进行聚合操作
    result = cases.aggregate({"A": ["mean", "std"], "B": ["mean", "std"]})
    # 使用 assert_frame_equal 来比较 result 和 expected 是否相等，检查它们是否近似相等
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "agg",
    [
        {"func": {"A": np.sum, "B": lambda x: np.std(x, ddof=1)}},
        {"A": ("A", np.sum), "B": ("B", lambda x: np.std(x, ddof=1))},
        {"A": NamedAgg("A", np.sum), "B": NamedAgg("B", lambda x: np.std(x, ddof=1))},
    ],
)
def test_agg_with_lambda(cases, agg):
    # 对 cases 中的 'B' 列应用 lambda 函数进行标准差计算
    rcustom = cases["B"].apply(lambda x: np.std(x, ddof=1))
    # 创建预期的 DataFrame，将 'A' 列的和与 'B' 列的自定义标准差拼接起来
    expected = pd.concat([cases["A"].sum(), rcustom], axis=1)
    # 在 cases 上调用 agg 方法，使用 **agg 传递字典形式的聚合规范
    result = cases.agg(**agg)
    # 使用 assert_frame_equal 来比较 result 和 expected 是否相等，检查它们是否近似相等
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "agg",
    [
        {"func": {"result1": np.sum, "result2": np.mean}},
        {"A": ("result1", np.sum), "B": ("result2", np.mean)},
        {"A": NamedAgg("result1", np.sum), "B": NamedAgg("result2", np.mean)},
    ],
)
def test_agg_no_column(cases, agg):
    # 设置错误消息内容
    msg = r"Label\(s\) \['result1', 'result2'\] do not exist"
    # 使用 pytest 的断言来检查是否抛出了 KeyError 异常，并验证其错误消息
    with pytest.raises(KeyError, match=msg):
        # 在 cases 中选择列 'A' 和 'B'，并对其应用 agg 方法，传入参数 agg 进行聚合操作
        cases[["A", "B"]].agg(**agg)


@pytest.mark.parametrize(
    "cols, agg",
    [
        [None, {"A": ["sum", "std"], "B": ["mean", "std"]}],
        [
            [
                "A",
                "B",
            ],
            {"A": ["sum", "std"], "B": ["mean", "std"]},
        ],
    ],
)
def test_agg_specificationerror_nested(cases, cols, agg, a_sum, a_std, b_mean, b_std):
    # 创建预期的 DataFrame，将各个列的和与标准差拼接起来
    expected = pd.concat([a_sum, a_std, b_mean, b_std], axis=1)
    # 设置列名为多级索引
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "sum"), ("A", "std"), ("B", "mean"), ("B", "std")]
    )
    # 如果 cols 不为 None，则对 cases 进行列选择，否则使用整个 cases
    if cols is not None:
        obj = cases[cols]
    else:
        obj = cases
    # 在 obj 上调用 agg 方法，传入参数 agg 进行聚合操作
    result = obj.agg(agg)
    # 使用 assert_frame_equal 来比较 result 和 expected 是否相等，检查它们是否近似相等
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "agg", [{"A": ["sum", "std"]}, {"A": ["sum", "std"], "B": ["mean", "std"]}]
)
def test_agg_specificationerror_series(cases, agg):
    # 设置错误消息内容
    msg = "nested renamer is not supported"
    # 使用 pytest 的断言来检查是否抛出了 pd.errors.SpecificationError 异常，并验证其错误消息
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        # 在 cases 的 'A' 列上调用 agg 方法，传入参数 agg 进行聚合操作
        cases["A"].agg(agg)


def test_agg_specificationerror_invalid_names(cases):
    # 设置错误消息内容
    msg = r"Label\(s\) \['B'\] do not exist"
    # 使用 pytest 的断言来检查是否抛出了 KeyError 异常，并验证其错误消息
    with pytest.raises(KeyError, match=msg):
        # 在 cases 的 'A' 列上调用 agg 方法，传入带有无效列名 'B' 的聚合规范
        cases[["A"]].agg({"A": ["sum", "std"], "B": ["mean", "std"]})


def test_agg_nested_dicts():
    # 待实现的测试函数，暂无代码内容
    # 创建一个日期范围索引，从 2005 年 1 月 1 日到 2005 年 1 月 10 日，频率为每天
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    # 设置索引的名称为 "date"
    index.name = "date"
    # 创建一个 DataFrame，包含随机生成的 10 行 2 列的数据，列名为 'A' 和 'B'，使用上述索引
    df = DataFrame(
        np.random.default_rng(2).random((10, 2)), columns=list("AB"), index=index
    )
    # 重置 DataFrame 的索引，生成一个新的 DataFrame df_col
    df_col = df.reset_index()
    # 复制 df_col 生成一个新的 DataFrame df_mult
    df_mult = df_col.copy()
    # 设置 df_mult 的索引为多级索引，第一级是 range(10)，第二级是 df 的原始索引，级别名称分别为 "index" 和 "date"
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), df.index], names=["index", "date"]
    )
    # 对原始 DataFrame df 进行2天的重采样，生成一个 Resampler 对象 r
    r = df.resample("2D")
    # 构建不同情况的列表 cases，每个元素都是一个重采样对象或操作
    cases = [
        r,
        df_col.resample("2D", on="date"),
        df_mult.resample("2D", level="date"),
        df.groupby(pd.Grouper(freq="2D")),
    ]

    # 设置错误消息
    msg = "nested renamer is not supported"
    # 对 cases 中的每个对象执行以下操作
    for t in cases:
        # 使用 pytest 检查是否抛出 SpecificationError 异常，并匹配错误消息 msg
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            # 尝试对重采样对象 t 进行聚合操作，指定了一个无法支持的嵌套结构
            t.aggregate({"r1": {"A": ["mean", "sum"]}, "r2": {"B": ["mean", "sum"]}})

    # 对 cases 中的每个对象执行以下操作
    for t in cases:
        # 使用 pytest 检查是否抛出 SpecificationError 异常，并匹配错误消息 msg
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            # 尝试对 t 中选择的列进行聚合操作，指定了一个无法支持的嵌套结构
            t[["A", "B"]].agg(
                {"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}}
            )

        # 使用 pytest 检查是否抛出 SpecificationError 异常，并匹配错误消息 msg
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            # 尝试对 t 中所有列进行聚合操作，指定了一个无法支持的嵌套结构
            t.agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})
def test_try_aggregate_non_existing_column():
    # GH 16766
    # 创建一个包含日期、x、y数据的列表
    data = [
        {"dt": datetime(2017, 6, 1, 0), "x": 1.0, "y": 2.0},
        {"dt": datetime(2017, 6, 1, 1), "x": 2.0, "y": 2.0},
        {"dt": datetime(2017, 6, 1, 2), "x": 3.0, "y": 1.5},
    ]
    # 将数据转换为DataFrame，并以'dt'列作为索引
    df = DataFrame(data).set_index("dt")

    # 当'z'列不存在时会引发错误
    msg = r"Label\(s\) \['z'\] do not exist"
    # 使用pytest检查是否引发KeyError并匹配错误消息
    with pytest.raises(KeyError, match=msg):
        df.resample("30min").agg({"x": ["mean"], "y": ["median"], "z": ["sum"]})


def test_agg_list_like_func_with_args():
    # 50624
    # 创建一个包含整数x的DataFrame，日期索引从'2020-01-01'开始，频率为每天
    df = DataFrame(
        {"x": [1, 2, 3]}, index=date_range("2020-01-01", periods=3, freq="D")
    )

    # 定义一个带有默认参数的函数foo1
    def foo1(x, a=1, c=0):
        return x + a + c

    # 定义一个带有默认参数的函数foo2
    def foo2(x, b=2, c=0):
        return x + b + c

    # 当使用不期望的关键字参数'b'调用foo1时会引发TypeError
    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.resample("D").agg([foo1, foo2], 3, b=3, c=4)

    # 执行聚合操作并验证结果是否符合预期
    result = df.resample("D").agg([foo1, foo2], 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        index=date_range("2020-01-01", periods=3, freq="D"),
        columns=pd.MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),
    )
    tm.assert_frame_equal(result, expected)


def test_selection_api_validation():
    # GH 13500
    # 创建一个日期范围，从2005年1月1日到2005年1月10日，每日频率
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")

    # 创建一个包含日期和整数列'a'的DataFrame
    rng = np.arange(len(index), dtype=np.int64)
    df = DataFrame(
        {"date": index, "a": rng},
        index=pd.MultiIndex.from_arrays([rng, index], names=["v", "d"]),
    )
    df_exp = DataFrame({"a": rng}, index=index)

    # 当使用非DatetimeIndex时会引发TypeError
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'Index'"
    )
    with pytest.raises(TypeError, match=msg):
        df.resample("2D", level="v")

    # 当同时指定key和level时会引发ValueError
    msg = "The Grouper cannot specify both a key and a level!"
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", on="date", level="d")

    # 当指定不可哈希类型'list'时会引发TypeError
    msg = "unhashable type: 'list'"
    with pytest.raises(TypeError, match=msg):
        df.resample("2D", on=["a", "date"])

    # 当指定的level未找到时会引发KeyError
    msg = r"\"Level \['a', 'date'\] not found\""
    with pytest.raises(KeyError, match=msg):
        df.resample("2D", level=["a", "date"])

    # 不允许向上采样时会引发ValueError
    msg = (
        "Upsampling from level= or on= selection is not supported, use "
        r"\.set_index\(\.\.\.\) to explicitly set index to datetime-like"
    )
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", level="d").asfreq()
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", on="date").asfreq()

    # 执行聚合操作并验证结果是否符合预期
    exp = df_exp.resample("2D").sum()
    exp.index.name = "date"
    result = df.resample("2D", on="date").sum()
    tm.assert_frame_equal(exp, result)

    # 当使用datetime64类型进行sum操作时会引发TypeError
    exp.index.name = "d"
    with pytest.raises(
        TypeError, match="datetime64 type does not support operation 'sum'"
    ):
        df.resample("2D", level="d").sum()
    # 对 DataFrame 进行重新采样，按照每2天（"2D"）的频率对 "d" 索引级别进行求和，仅包括数值列
    result = df.resample("2D", level="d").sum(numeric_only=True)
    # 使用测试框架中的 assert_frame_equal 函数，比较预期结果 exp 和实际结果 result 的内容是否相等
    tm.assert_frame_equal(exp, result)
@pytest.mark.parametrize(
    "start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods",
    [
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end",
            None,
            [0, 18, 27, 63],
            "20001002 00:26:00",
            4,
        ),
        (
            "20200101 8:26:35",
            "20200101 9:31:58",
            "77s",
            [1] * 51,
            "7min",
            "end",
            "right",
            [1, 6, 5, 6, 5, 6, 5, 6, 5, 6],
            "2020-01-01 09:30:45",
            10,
        ),
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end",
            "left",
            [0, 18, 27, 39, 24],
            "20001002 00:43:00",
            5,
        ),
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end_day",
            None,
            [3, 15, 45, 45],
            "2000-10-02 00:29:00",
            4,
        ),
    ],
)
def test_end_and_end_day_origin(
    start,
    end,
    freq,
    data,
    resample_freq,
    origin,
    closed,
    exp_data,
    exp_end,
    exp_periods,
):
    """
    测试不同的起始时间、结束时间、频率、数据、重采样频率、原点、关闭方式对于特定操作的影响。

    参数:
    - start: 起始时间戳
    - end: 结束时间戳
    - freq: 原始数据频率
    - data: 原始数据列表
    - resample_freq: 重采样频率
    - origin: 时间戳对齐原点
    - closed: 时间戳闭合方式
    - exp_data: 期望的重采样数据
    - exp_end: 期望的重采样结束时间
    - exp_periods: 期望的重采样周期数
    """
    # 此处包含了多个测试用例，每个用例定义了不同的输入条件和预期输出结果。
    # 执行重采样聚合操作
    result = df.resample(resample_freq, origin=origin, closed=closed).agg("sum")
    # 根据预期数据创建预期结果的 DataFrame
    expected = DataFrame(exp_data, index=date_range(start=start, end=exp_end, freq=freq))
    # 断言实际结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)
    resample_freq,    # 变量：重新采样频率
    origin,           # 变量：数据的起始时间或原点
    closed,           # 变量：指定区间闭合方式（开闭区间）
    exp_data,         # 变量：实验数据集
    exp_end,          # 变量：实验结束时间点
    exp_periods,      # 变量：实验周期数
@pytest.mark.parametrize(
    # 参数化测试，指定参数method, numeric_only, expected_data
    "method, numeric_only, expected_data",
    [
        # 下面是一系列测试参数组合
        ("sum", True, {"num": [25]}),
        ("sum", False, {"cat": ["cat_1cat_2"], "num": [25]}),
        ("sum", lib.no_default, {"cat": ["cat_1cat_2"], "num": [25]}),
        ("prod", True, {"num": [100]}),
        ("prod", False, "can't multiply sequence"),
        ("prod", lib.no_default, "can't multiply sequence"),
        ("min", True, {"num": [5]}),
        ("min", False, {"cat": ["cat_1"], "num": [5]}),
        ("min", lib.no_default, {"cat": ["cat_1"], "num": [5]}),
        ("max", True, {"num": [20]}),
        ("max", False, {"cat": ["cat_2"], "num": [20]}),
        ("max", lib.no_default, {"cat": ["cat_2"], "num": [20]}),
        ("first", True, {"num": [5]}),
        ("first", False, {"cat": ["cat_1"], "num": [5]}),
        ("first", lib.no_default, {"cat": ["cat_1"], "num": [5]}),
        ("last", True, {"num": [20]}),
        ("last", False, {"cat": ["cat_2"], "num": [20]}),
        ("last", lib.no_default, {"cat": ["cat_2"], "num": [20]}),
        ("mean", True, {"num": [12.5]}),
        ("mean", False, "Could not convert"),
        ("mean", lib.no_default, "Could not convert"),
        ("median", True, {"num": [12.5]}),
        ("median", False, r"Cannot convert \['cat_1' 'cat_2'\] to numeric"),
        ("median", lib.no_default, r"Cannot convert \['cat_1' 'cat_2'\] to numeric"),
        ("std", True, {"num": [10.606601717798213]}),
        ("std", False, "could not convert string to float"),
        ("std", lib.no_default, "could not convert string to float"),
        ("var", True, {"num": [112.5]}),
        ("var", False, "could not convert string to float"),
        ("var", lib.no_default, "could not convert string to float"),
        ("sem", True, {"num": [7.5]}),
        ("sem", False, "could not convert string to float"),
        ("sem", lib.no_default, "could not convert string to float"),
    ],
)
def test_frame_downsample_method(method, numeric_only, expected_data):
    # 测试 DataFrameGroupBy 对象的下采样方法行为

    # 创建一个包含两行数据的时间索引
    index = date_range("2018-01-01", periods=2, freq="D")
    # 创建一个预期的索引，包含一行数据，频率为每年一次
    expected_index = date_range("2018-12-31", periods=1, freq="YE")
    # 创建一个包含类别和数字列的 DataFrame
    df = DataFrame({"cat": ["cat_1", "cat_2"], "num": [5, 20]}, index=index)
    # 对 DataFrame 进行年度下采样
    resampled = df.resample("YE")

    # 根据参数确定是否使用 numeric_only 选项
    if numeric_only is lib.no_default:
        kwargs = {}
    else:
        kwargs = {"numeric_only": numeric_only}

    # 根据方法名获取对应的函数
    func = getattr(resampled, method)
    # 如果期望数据类型为字符串
    if isinstance(expected_data, str):
        # 如果方法为统计函数中的一种 ("var", "mean", "median", "prod")
        if method in ("var", "mean", "median", "prod"):
            # 设置异常类型为 TypeError
            klass = TypeError
            # 生成匹配的异常消息，转义特殊字符
            msg = re.escape(f"agg function failed [how->{method},dtype->")
        else:
            # 否则设置异常类型为 ValueError
            klass = ValueError
            # 使用期望数据作为异常消息
            msg = expected_data
        # 使用 pytest 检查是否抛出指定类型和匹配消息的异常
        with pytest.raises(klass, match=msg):
            # 调用 func 函数，传入 kwargs 参数
            _ = func(**kwargs)
    else:
        # 如果期望数据不是字符串，则执行以下操作
        # 调用 func 函数，传入 kwargs 参数，并接收返回结果
        result = func(**kwargs)
        # 根据期望数据和索引创建 DataFrame 对象
        expected = DataFrame(expected_data, index=expected_index)
        # 使用 pandas 的测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "method, numeric_only, expected_data",
    [
        # 参数化测试数据集，包括方法名、numeric_only标志和预期结果数据
        ("sum", True, ()),  # 使用sum方法，numeric_only为True，预期结果为空元组
        ("sum", False, ["cat_1cat_2"]),  # 使用sum方法，numeric_only为False，预期结果为包含字符串"cat_1cat_2"的列表
        ("sum", lib.no_default, ["cat_1cat_2"]),  # 使用sum方法，numeric_only为lib.no_default，预期结果同上
        ("prod", True, ()),  # 使用prod方法，numeric_only为True，预期结果为空元组
        ("prod", False, ()),  # 使用prod方法，numeric_only为False，预期结果为空元组
        ("prod", lib.no_default, ()),  # 使用prod方法，numeric_only为lib.no_default，预期结果为空元组
        ("min", True, ()),  # 使用min方法，numeric_only为True，预期结果为空元组
        ("min", False, ["cat_1"]),  # 使用min方法，numeric_only为False，预期结果为包含字符串"cat_1"的列表
        ("min", lib.no_default, ["cat_1"]),  # 使用min方法，numeric_only为lib.no_default，预期结果同上
        ("max", True, ()),  # 使用max方法，numeric_only为True，预期结果为空元组
        ("max", False, ["cat_2"]),  # 使用max方法，numeric_only为False，预期结果为包含字符串"cat_2"的列表
        ("max", lib.no_default, ["cat_2"]),  # 使用max方法，numeric_only为lib.no_default，预期结果同上
        ("first", True, ()),  # 使用first方法，numeric_only为True，预期结果为空元组
        ("first", False, ["cat_1"]),  # 使用first方法，numeric_only为False，预期结果为包含字符串"cat_1"的列表
        ("first", lib.no_default, ["cat_1"]),  # 使用first方法，numeric_only为lib.no_default，预期结果同上
        ("last", True, ()),  # 使用last方法，numeric_only为True，预期结果为空元组
        ("last", False, ["cat_2"]),  # 使用last方法，numeric_only为False，预期结果为包含字符串"cat_2"的列表
        ("last", lib.no_default, ["cat_2"]),  # 使用last方法，numeric_only为lib.no_default，预期结果同上
    ],
)
def test_series_downsample_method(method, numeric_only, expected_data):
    # GH#46442 测试SeriesGroupBy在使用numeric_only参数时的行为

    index = date_range("2018-01-01", periods=2, freq="D")  # 创建一个日期索引
    expected_index = date_range("2018-12-31", periods=1, freq="YE")  # 创建预期的日期索引
    df = Series(["cat_1", "cat_2"], index=index)  # 创建一个包含两个值的Series对象
    resampled = df.resample("YE")  # 对Series对象进行年度重采样
    kwargs = {} if numeric_only is lib.no_default else {"numeric_only": numeric_only}  # 根据numeric_only参数设置kwargs字典

    func = getattr(resampled, method)  # 获取resampled对象上的指定方法的函数
    if numeric_only and numeric_only is not lib.no_default:
        # 如果numeric_only为True且不是lib.no_default，则期望引发TypeError异常
        msg = rf"Cannot use numeric_only=True with SeriesGroupBy\.{method}"
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    elif method == "prod":
        # 如果方法是prod，则期望引发TypeError异常，消息中包含特定字符串
        msg = re.escape("agg function failed [how->prod,dtype->")
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    else:
        # 否则，执行函数并比较结果和预期的Series对象
        result = func(**kwargs)
        expected = Series(expected_data, index=expected_index)
        tm.assert_series_equal(result, expected)


def test_resample_empty():
    # GH#52484 检查空DataFrame对象的重采样行为
    df = DataFrame(
        index=pd.to_datetime(
            ["2018-01-01 00:00:00", "2018-01-01 12:00:00", "2018-01-02 00:00:00"]
        )  # 创建带有指定日期时间索引的DataFrame对象
    )
    expected = DataFrame(
        index=pd.to_datetime(
            [
                "2018-01-01 00:00:00",
                "2018-01-01 08:00:00",
                "2018-01-01 16:00:00",
                "2018-01-02 00:00:00",
            ]  # 创建预期的DataFrame对象，带有指定日期时间索引
        )
    )
    result = df.resample("8h").mean()  # 对DataFrame对象进行每8小时的重采样并计算均值
    tm.assert_frame_equal(result, expected)  # 断言结果DataFrame与预期DataFrame相等
```