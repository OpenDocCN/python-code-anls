# `D:\src\scipysrc\pandas\pandas\tests\groupby\aggregate\test_aggregate.py`

```
"""
test .agg behavior / note that .apply is tested generally in test_groupby.py
"""

import datetime  # 导入日期时间模块
import functools  # 导入 functools 模块
from functools import partial  # 导入 functools 模块中的 partial 函数

import numpy as np  # 导入 NumPy 库，并简写为 np
import pytest  # 导入 pytest 测试框架

from pandas.errors import SpecificationError  # 从 pandas.errors 中导入 SpecificationError 异常类

from pandas.core.dtypes.common import is_integer_dtype  # 从 pandas.core.dtypes.common 中导入 is_integer_dtype 函数

import pandas as pd  # 导入 Pandas 库，并简写为 pd
from pandas import (  # 从 Pandas 中导入多个类和函数
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    to_datetime,
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并简写为 tm
from pandas.core.groupby.grouper import Grouping  # 从 pandas.core.groupby.grouper 中导入 Grouping 类


def test_groupby_agg_no_extra_calls():
    # GH#31760
    df = DataFrame({"key": ["a", "b", "c", "c"], "value": [1, 2, 3, 4]})
    gb = df.groupby("key")["value"]  # 对 DataFrame df 按 'key' 列进行分组，并选择 'value' 列

    def dummy_func(x):
        assert len(x) != 0  # 断言：x 的长度不为 0
        return x.sum()  # 返回 x 列的和

    gb.agg(dummy_func)  # 对分组对象 gb 应用 dummy_func 函数进行聚合


def test_agg_regression1(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])  # 对 tsframe 按年份和月份进行分组
    result = grouped.agg("mean")  # 对分组后的数据应用均值聚合
    expected = grouped.mean()  # 计算分组后数据的均值
    tm.assert_frame_equal(result, expected)  # 使用测试工具 tm 检查 result 和 expected 是否相等


def test_agg_must_agg(df):
    grouped = df.groupby("A")["C"]  # 对 DataFrame df 按 'A' 列分组，并选择 'C' 列

    msg = "Must produce aggregated value"
    with pytest.raises(Exception, match=msg):  # 使用 pytest 检查是否抛出特定异常和消息
        grouped.agg(lambda x: x.describe())  # 对分组后的数据应用 describe 函数
    with pytest.raises(Exception, match=msg):  # 再次使用 pytest 检查是否抛出特定异常和消息
        grouped.agg(lambda x: x.index[:2])  # 对分组后的索引应用切片操作


def test_agg_ser_multi_key(df):
    f = lambda x: x.sum()  # 定义一个函数 f，计算 x 列的和
    results = df.C.groupby([df.A, df.B]).aggregate(f)  # 对 df 中的 'C' 列按 'A' 和 'B' 列进行分组，并应用函数 f 进行聚合
    expected = df.groupby(["A", "B"]).sum()["C"]  # 对 df 按 'A' 和 'B' 列进行分组，并计算 'C' 列的和
    tm.assert_series_equal(results, expected)  # 使用测试工具 tm 检查结果 results 和期望 expected 是否相等


def test_agg_with_missing_values():
    # GH#58810
    missing_df = DataFrame(  # 创建包含缺失值的 DataFrame
        {
            "nan": [np.nan, np.nan, np.nan, np.nan],  # 包含 NaN 值的列
            "na": [pd.NA, pd.NA, pd.NA, pd.NA],  # 包含 pd.NA 值的列
            "nat": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],  # 包含 pd.NaT 值的列
            "none": [None, None, None, None],  # 包含 None 值的列
            "values": [1, 2, 3, 4],  # 值列
        }
    )

    result = missing_df.agg(x=("nan", "min"), y=("na", "min"), z=("values", "sum"))  # 对 DataFrame 应用多个聚合函数

    expected = DataFrame(  # 期望的结果 DataFrame
        {
            "nan": [np.nan, np.nan, np.nan],  # 期望的 NaN 列
            "na": [np.nan, np.nan, np.nan],  # 期望的 pd.NA 列
            "values": [np.nan, np.nan, 10.0],  # 期望的值列
        },
        index=["x", "y", "z"],  # 指定索引
    )

    tm.assert_frame_equal(result, expected)  # 使用测试工具 tm 检查结果 result 和期望 expected 是否相等


def test_groupby_aggregation_mixed_dtype():
    # GH 6212
    expected = DataFrame(  # 期望的结果 DataFrame
        {
            "v1": [5, 5, 7, np.nan, 3, 3, 4, 1],  # 'v1' 列的值
            "v2": [55, 55, 77, np.nan, 33, 33, 44, 11],  # 'v2' 列的值
        },
        index=MultiIndex.from_tuples(  # 使用 MultiIndex 创建索引
            [
                (1, 95),  # 索引条目
                (1, 99),  # 索引条目
                (2, 95),  # 索引条目
                (2, 99),  # 索引条目
                ("big", "damp"),  # 索引条目
                ("blue", "dry"),  # 索引条目
                ("red", "red"),  # 索引条目
                ("red", "wet"),  # 索引条目
            ],
            names=["by1", "by2"],  # 设置索引名称
        ),
    )
    # 创建一个 DataFrame 对象，包含四列数据：v1, v2, by1, by2
    df = DataFrame(
        {
            "v1": [1, 3, 5, 7, 8, 3, 5, np.nan, 4, 5, 7, 9],  # 第一列数据 v1
            "v2": [11, 33, 55, 77, 88, 33, 55, np.nan, 44, 55, 77, 99],  # 第二列数据 v2
            "by1": ["red", "blue", 1, 2, np.nan, "big", 1, 2, "red", 1, np.nan, 12],  # 第三列数据 by1
            "by2": [  # 第四列数据 by2
                "wet",
                "dry",
                99,
                95,
                np.nan,
                "damp",
                95,
                99,
                "red",
                99,
                np.nan,
                np.nan,
            ],
        }
    )
    
    # 根据 by1 和 by2 列进行分组，生成一个 GroupBy 对象 g
    g = df.groupby(["by1", "by2"])
    
    # 对分组后的数据取出 v1 和 v2 列，并计算均值，生成一个新的 DataFrame 对象 result
    result = g[["v1", "v2"]].mean()
    
    # 使用测试工具 tm.assert_frame_equal 检查 result 是否与期望结果 expected 相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试在边缘情况下的聚合和应用操作
def test_agg_apply_corner(ts, tsframe):
    # 对时间序列进行分组，但由于都是 NaN，结果将全部是 NaN
    grouped = ts.groupby(ts * np.nan, group_keys=False)
    # 断言时间序列的数据类型为 np.float64
    assert ts.dtype == np.float64

    # 使用 float64 值进行分组将导致索引也是 float64 类型
    exp = Series([], dtype=np.float64, index=Index([], dtype=np.float64))
    # 断言分组后的求和结果与预期的空 Series 相等
    tm.assert_series_equal(grouped.sum(), exp)
    # 断言分组后通过 agg("sum") 求和的结果与预期的空 Series 相等
    tm.assert_series_equal(grouped.agg("sum"), exp)
    # 断言应用 apply("sum") 后的结果与预期的空 Series 相等，忽略索引类型检查
    tm.assert_series_equal(grouped.apply("sum"), exp, check_index_type=False)

    # 对 DataFrame 进行相同的操作
    grouped = tsframe.groupby(tsframe["A"] * np.nan, group_keys=False)
    # 创建一个与 tsframe 列相同但索引为空的空 DataFrame
    exp_df = DataFrame(
        columns=tsframe.columns,
        dtype=float,
        index=Index([], name="A", dtype=np.float64),
    )
    # 断言分组后的求和结果与预期的空 DataFrame 相等
    tm.assert_frame_equal(grouped.sum(), exp_df)
    # 断言分组后通过 agg("sum") 求和的结果与预期的空 DataFrame 相等
    tm.assert_frame_equal(grouped.agg("sum"), exp_df)

    # 应用 np.sum 函数对分组进行操作，期望结果与空 DataFrame 相等
    res = grouped.apply(np.sum, axis=0)
    tm.assert_frame_equal(res, exp_df)


# 测试处理包含 NaN 分组的情况
def test_with_na_groups(any_real_numpy_dtype):
    index = Index(np.arange(10))
    values = Series(np.ones(10), index, dtype=any_real_numpy_dtype)
    # 创建包含 NaN 的标签 Series
    labels = Series(
        [np.nan, "foo", "bar", "bar", np.nan, np.nan, "bar", "bar", np.nan, "foo"],
        index=index,
    )

    # 根据标签对值进行分组，期望得到包含两个分组的 Series，对应值分别为 4 和 2
    grouped = values.groupby(labels)
    agged = grouped.agg(len)
    expected = Series([4, 2], index=["bar", "foo"])
    # 断言聚合后的结果与预期的 Series 相等，忽略数据类型检查
    tm.assert_series_equal(agged, expected, check_dtype=False)

    # 定义一个函数 f，显式返回浮点数长度
    def f(x):
        return float(len(x))

    # 根据标签对值进行分组，并使用函数 f 进行聚合，期望得到浮点数长度的 Series
    agged = grouped.agg(f)
    expected = Series([4.0, 2.0], index=["bar", "foo"])
    # 断言聚合后的结果与预期的 Series 相等
    tm.assert_series_equal(agged, expected)


# 测试处理多级索引 DataFrame 的聚合操作
def test_agg_grouping_is_list_tuple(ts):
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=pd.date_range("2000-01-01", periods=30, freq="B"),
    )

    # 根据年份对 DataFrame 进行分组
    grouped = df.groupby(lambda x: x.year)
    # 获取分组器的分组向量
    grouper = grouped._grouper.groupings[0].grouping_vector
    # 修改分组器的分组方式为使用 ts 的索引，期望结果与均值相等
    grouped._grouper.groupings[0] = Grouping(ts.index, list(grouper))

    # 断言通过 agg("mean") 聚合的结果与均值相等
    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)

    # 再次修改分组器的分组方式为使用元组表示的 grouper，期望结果与均值相等
    grouped._grouper.groupings[0] = Grouping(ts.index, tuple(grouper))

    # 断言通过 agg("mean") 聚合的结果与均值相等
    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


# 测试多级索引 DataFrame 使用字符串函数进行聚合操作
def test_agg_python_multiindex(multiindex_dataframe_random_data):
    # 根据 ["A", "B"] 多级索引进行分组
    grouped = multiindex_dataframe_random_data.groupby(["A", "B"])

    # 断言通过 agg("mean") 聚合的结果与均值相等
    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


# 使用参数化测试，测试根据不同函数进行分组聚合操作
@pytest.mark.parametrize(
    "groupbyfunc", [lambda x: x.weekday(), [lambda x: x.month, lambda x: x.weekday()]]
)
def test_aggregate_str_func(tsframe, groupbyfunc):
    # 根据给定的 groupbyfunc 函数进行分组
    grouped = tsframe.groupby(groupbyfunc)

    # 对单个 Series 进行聚合操作，期望得到标准差的 Series
    result = grouped["A"].agg("std")
    expected = grouped["A"].std()
    # 断言聚合后的结果与预期的 Series 相等
    tm.assert_series_equal(result, expected)

    # 组合 frame 按函数名进行分组
    # 对分组后的数据框进行聚合操作，计算每个分组的变量
    result = grouped.aggregate("var")
    
    # 使用 Pandas 中的 var() 方法，计算每个分组的方差
    expected = grouped.var()
    
    # 使用 Pandas Testing 模块中的 assert_frame_equal 函数比较计算结果和期望结果的数据框是否相等
    tm.assert_frame_equal(result, expected)
    
    # 按照指定的函数对数据框进行分组聚合操作
    result = grouped.agg({"A": "var", "B": "std", "C": "mean", "D": "sem"})
    
    # 创建一个新的数据框，包含按照每列指定函数计算得到的结果
    expected = DataFrame(
        {
            "A": grouped["A"].var(),
            "B": grouped["B"].std(),
            "C": grouped["C"].mean(),
            "D": grouped["D"].sem(),
        }
    )
    
    # 使用 Pandas Testing 模块中的 assert_frame_equal 函数比较计算结果和期望结果的数据框是否相等
    tm.assert_frame_equal(result, expected)
# 测试标准差函数在掩码数据类型上的行为
def test_std_masked_dtype(any_numeric_ea_dtype):
    # 创建一个包含两列的 DataFrame，其中一列包含缺失值（pd.NA）
    df = DataFrame(
        {
            "a": [2, 1, 1, 1, 2, 2, 1],
            "b": Series([pd.NA, 1, 2, 1, 1, 1, 2], dtype="Float64"),
        }
    )
    # 对 DataFrame 按列 'a' 分组，计算每组的标准差
    result = df.groupby("a").std()
    # 创建一个期望的 DataFrame，包含标准差的预期值
    expected = DataFrame(
        {"b": [0.57735, 0]}, index=Index([1, 2], name="a"), dtype="Float64"
    )
    # 使用断言比较计算结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试在轴向为1时聚合字符串操作是否会触发 ValueError
def test_agg_str_with_kwarg_axis_1_raises(df, reduction_func):
    # 对 DataFrame 根据第一层级索引进行分组
    gb = df.groupby(level=0)
    # 准备错误消息，用于匹配抛出的 ValueError
    msg = f"Operation {reduction_func} does not support axis=1"
    # 使用 pytest 的上下文管理器检查是否抛出预期的 ValueError
    with pytest.raises(ValueError, match=msg):
        gb.agg(reduction_func, axis=1)


# 测试逐项聚合函数的行为
def test_aggregate_item_by_item(df):
    # 根据列 'A' 对 DataFrame 进行分组
    grouped = df.groupby("A")

    # 定义一个逐项聚合函数，计算每组的大小（元素个数）
    aggfun_0 = lambda ser: ser.size
    # 对分组后的结果应用逐项聚合函数
    result = grouped.agg(aggfun_0)
    # 计算列 'A' 中值为 'foo' 的元素个数
    foosum = (df.A == "foo").sum()
    # 计算列 'A' 中值为 'bar' 的元素个数
    barsum = (df.A == "bar").sum()
    # 获取结果 DataFrame 的列数
    K = len(result.columns)

    # 使用断言比较预期结果和实际结果是否相等
    # 期望针对值为 'foo' 的行返回相应的元素个数
    exp = Series(np.array([foosum] * K), index=list("BCD"), name="foo")
    tm.assert_series_equal(result.xs("foo"), exp)

    # 期望针对值为 'bar' 的行返回相应的元素个数
    exp = Series(np.array([barsum] * K), index=list("BCD"), name="bar")
    tm.assert_almost_equal(result.xs("bar"), exp)

    # 定义另一个逐项聚合函数，计算每组的大小（元素个数）
    def aggfun_1(ser):
        return ser.size

    # 对 DataFrame 重新进行分组，并应用新定义的逐项聚合函数
    result = DataFrame().groupby(df.A).agg(aggfun_1)
    # 使用断言检查结果是否为 DataFrame 类型
    assert isinstance(result, DataFrame)
    # 使用断言检查结果 DataFrame 是否为空
    assert len(result) == 0


# 测试在聚合时处理异常输出的情况
def test_wrap_agg_out(three_group):
    # 根据列 'A' 和 'B' 对 DataFrame 进行分组
    grouped = three_group.groupby(["A", "B"])

    # 定义一个函数，如果序列的数据类型是对象，则引发 TypeError
    def func(ser):
        if ser.dtype == object:
            raise TypeError("Test error message")
        return ser.sum()

    # 使用 pytest 的上下文管理器检查是否抛出预期的 TypeError
    with pytest.raises(TypeError, match="Test error message"):
        grouped.aggregate(func)

    # 对分组后的结果选择特定的列，并应用聚合函数
    result = grouped[["D", "E", "F"]].aggregate(func)
    # 从原始 DataFrame 中选择列 'A'、'B'、'D'、'E'、'F' 的子集
    exp_grouped = three_group.loc[:, ["A", "B", "D", "E", "F"]]
    # 对子集 DataFrame 进行重新分组，并应用聚合函数
    expected = exp_grouped.groupby(["A", "B"]).aggregate(func)
    # 使用断言比较预期结果和实际结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试多函数聚合时维持顺序的行为
def test_agg_multiple_functions_maintain_order(df):
    # 准备一个包含多个聚合函数的列表
    funcs = [("mean", np.mean), ("max", np.max), ("min", np.min)]
    # 对 DataFrame 根据列 'A' 进行分组，并应用多个聚合函数
    result = df.groupby("A")["C"].agg(funcs)
    # 准备一个期望的索引对象，包含聚合结果 DataFrame 的列名
    exp_cols = Index(["mean", "max", "min"])
    # 使用断言比较预期的列名和实际的列名是否相等
    tm.assert_index_equal(result.columns, exp_cols)


# 测试 Series 对象的索引名称
def test_series_index_name(df):
    # 根据列 'A' 对 DataFrame 中的列 'C' 进行分组，并应用均值函数
    grouped = df.loc[:, ["C"]].groupby(df["A"])
    # 对分组后的结果应用 lambda 函数计算均值
    result = grouped.agg(lambda x: x.mean())
    # 使用断言检查结果的索引名称是否为 'A'
    assert result.index.name == "A"


# 测试在聚合时使用相同名称的多个函数的行为
def test_agg_multiple_functions_same_name():
    # 创建一个具有随机标准正态分布数据的 DataFrame
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=pd.date_range("1/1/2012", freq="s", periods=1000),
        columns=["A", "B", "C"],
    )
    # 对 DataFrame 进行 3 分钟的重采样，并应用两个分位数聚合函数
    result = df.resample("3min").agg(
        {"A": [partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]}
    )
    # 准备期望的索引，包含聚合结果 DataFrame 的列名
    expected_index = pd.date_range("1/1/2012", freq="3min", periods=6)
    # 准备期望的列标签，包含聚合结果 DataFrame 的多级索引
    expected_columns = MultiIndex.from_tuples([("A", "quantile"), ("A", "quantile")])
    # 准备期望的值，使用循环计算不同分位数的聚合值
    expected_values = np.array(
        [df.resample("3min").A.quantile(q=q).values for q in [0.9999, 0.1111]]
    ).T
    # 输出需要进行断言比较的 DataFrame
    # 创建一个 DataFrame 对象，并使用给定的数据、列名和索引初始化它
    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    # 使用测试工具中的函数（tm.assert_frame_equal）比较 result 和 expected 两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
# 测试聚合多个函数，其中ohlc（开、高、低、收）已经存在
def test_agg_multiple_functions_same_name_with_ohlc_present():
    # GitHub issue 30880

    # 创建一个DataFrame，包含1000行3列的随机标准正态分布数据，以秒为频率，从"1/1/2012"开始，名为"dti"
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=pd.date_range("1/1/2012", freq="s", periods=1000, name="dti"),
        columns=Index(["A", "B", "C"], name="alpha"),
    )

    # 对DataFrame进行3分钟的重采样，并对列"A"应用多个聚合函数
    result = df.resample("3min").agg(
        {"A": ["ohlc", partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]}
    )

    # 期望的索引，以3分钟为频率，从"1/1/2012"开始，共6个时间点，名为"dti"
    expected_index = pd.date_range("1/1/2012", freq="3min", periods=6, name="dti")

    # 期望的列，使用MultiIndex包含不同层级的元组，表示不同的聚合方式和统计量
    expected_columns = MultiIndex.from_tuples(
        [
            ("A", "ohlc", "open"),
            ("A", "ohlc", "high"),
            ("A", "ohlc", "low"),
            ("A", "ohlc", "close"),
            ("A", "quantile", "0.9999"),
            ("A", "quantile", "0.1111"),
        ],
        names=["alpha", None, None],
    )

    # 非ohlc的期望数值，使用quantile计算不同分位数的值
    non_ohlc_expected_values = np.array(
        [df.resample("3min").A.quantile(q=q).values for q in [0.9999, 0.1111]]
    ).T

    # 合并期望数值，包括ohlc的开高低收和非ohlc的分位数值
    expected_values = np.hstack(
        [df.resample("3min").A.ohlc(), non_ohlc_expected_values]
    )

    # 期望的DataFrame，具有预期的值、列和索引
    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )

    # 使用测试工具比较结果DataFrame和期望DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试多函数的元组和非元组方式
def test_multiple_functions_tuples_and_non_tuples(df):
    # GitHub issue 1359

    # 删除DataFrame中的列"B"和"C"
    df = df.drop(columns=["B", "C"])

    # 定义元组形式和非元组形式的聚合函数列表
    funcs = [("foo", "mean"), "std"]
    ex_funcs = [("foo", "mean"), ("std", "std")]

    # 对分组后的DataFrame列"D"进行聚合操作，使用funcs列表中的函数
    result = df.groupby("A")["D"].agg(funcs)
    # 期望的结果，使用ex_funcs中的函数进行聚合
    expected = df.groupby("A")["D"].agg(ex_funcs)

    # 使用测试工具比较结果DataFrame和期望DataFrame是否相等
    tm.assert_frame_equal(result, expected)

    # 对分组后的整个DataFrame进行聚合操作，使用funcs列表中的函数
    result = df.groupby("A").agg(funcs)
    # 期望的结果，使用ex_funcs中的函数进行聚合
    expected = df.groupby("A").agg(ex_funcs)

    # 使用测试工具比较结果DataFrame和期望DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试更灵活的DataFrame多函数聚合操作
def test_more_flexible_frame_multi_function(df):
    # 对DataFrame按"A"列分组
    grouped = df.groupby("A")

    # 对分组后的DataFrame进行多函数聚合，计算列"C"和"D"的均值
    exmean = grouped.agg({"C": "mean", "D": "mean"})
    # 对分组后的DataFrame进行多函数聚合，计算列"C"和"D"的标准差
    exstd = grouped.agg({"C": "std", "D": "std"})

    # 构建期望的结果DataFrame，包括均值和标准差，使用MultiIndex标记不同的统计量和列名
    expected = concat([exmean, exstd], keys=["mean", "std"], axis=1)
    expected = expected.swaplevel(0, 1, axis=1).sort_index(level=0, axis=1)

    # 定义聚合操作的字典，指定每列要应用的函数列表
    d = {"C": ["mean", "std"], "D": ["mean", "std"]}
    # 对分组后的DataFrame应用多函数聚合操作
    result = grouped.aggregate(d)

    # 使用测试工具比较结果DataFrame和期望DataFrame是否相等
    tm.assert_frame_equal(result, expected)

    # 对分组后的DataFrame应用特定的聚合操作，其中列"D"使用不同的函数列表
    result = grouped.aggregate({"C": "mean", "D": ["mean", "std"]})
    expected = grouped.aggregate({"C": "mean", "D": ["mean", "std"]})

    # 使用测试工具比较结果DataFrame和期望DataFrame是否相等
    tm.assert_frame_equal(result, expected)

    # 定义自定义的NumPy函数
    def numpymean(x):
        return np.mean(x)

    def numpystd(x):
        return np.std(x, ddof=1)

    # 尝试使用具有嵌套重命名的聚合操作字典，预期会引发异常
    msg = r"nested renamer is not supported"
    d = {"C": "mean", "D": {"foo": "mean", "bar": "std"}}
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)

    # 但是，如果没有重命名，这些函数可以正常工作
    d = {"C": ["mean"], "D": [numpymean, numpystd]}
    grouped.aggregate(d)


# 测试多函数灵活混合使用
def test_multi_function_flexible_mix(df):
    # GitHub issue 1268
    # 根据"A"列对DataFrame进行分组
    grouped = df.groupby("A")

    # 预期结果定义
    d = {"C": {"foo": "mean", "bar": "std"}, "D": {"sum": "sum"}}
    # 使用列选择和重命名操作
    msg = r"nested renamer is not supported"
    # 使用pytest的raises函数验证是否抛出SpecificationError异常，并匹配指定的错误消息
    with pytest.raises(SpecificationError, match=msg):
        # 对分组后的DataFrame应用聚合操作
        grouped.aggregate(d)

    # 测试1
    d = {"C": {"foo": "mean", "bar": "std"}, "D": "sum"}
    # 使用列选择和重命名操作
    with pytest.raises(SpecificationError, match=msg):
        # 对分组后的DataFrame应用聚合操作
        grouped.aggregate(d)

    # 测试2
    d = {"C": {"foo": "mean", "bar": "std"}, "D": "sum"}
    # 使用列选择和重命名操作
    with pytest.raises(SpecificationError, match=msg):
        # 对分组后的DataFrame应用聚合操作
        grouped.aggregate(d)
# 定义一个测试函数，测试 groupby 和 aggregate 方法是否正确处理布尔类型数据
def test_groupby_agg_coercing_bools():
    # 创建包含三列数据的 DataFrame，其中 'a' 列有重复的值
    dat = DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3], "c": [None, None, 1, 1]})
    # 对 DataFrame 按 'a' 列进行分组
    gp = dat.groupby("a")

    # 创建一个 Index 对象，指定名称为 'a'，包含索引值 1 和 2
    index = Index([1, 2], name="a")

    # 对分组后的 'b' 列应用 lambda 函数，判断是否所有值均不为 0
    result = gp["b"].aggregate(lambda x: (x != 0).all())
    # 创建预期结果 Series，索引为 index，值为布尔值列表
    expected = Series([False, True], index=index, name="b")
    # 断言 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 对分组后的 'c' 列应用 lambda 函数，判断是否所有值均为 None
    result = gp["c"].aggregate(lambda x: x.isnull().all())
    # 创建预期结果 Series，索引为 index，值为布尔值列表
    expected = Series([True, False], index=index, name="c")
    # 断言 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，测试 groupby 和 agg 方法是否正确处理字典参数获取的列名
def test_groupby_agg_dict_with_getitem():
    # 创建包含两列数据的 DataFrame，其中 'A' 列有重复的值
    dat = DataFrame({"A": ["A", "A", "B", "B", "B"], "B": [1, 2, 1, 1, 2]})
    # 对 DataFrame 按 'A' 列进行分组，并对 'B' 列应用字典 {"B": "sum"} 形式的聚合
    result = dat.groupby("A")[["B"]].agg({"B": "sum"})

    # 创建预期结果 DataFrame，包含两行，索引为 ['A', 'B']
    expected = DataFrame({"B": [3, 4]}, index=["A", "B"]).rename_axis("A", axis=0)

    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试 groupby 和 agg 方法是否正确处理具有重复列名的 DataFrame
def test_groupby_agg_dict_dup_columns():
    # 创建包含三行四列的 DataFrame，其中最后一列列名为 'c' 重复
    df = DataFrame(
        [[1, 2, 3, 4], [1, 3, 4, 5], [2, 4, 5, 6]],
        columns=["a", "b", "c", "c"],
    )
    # 对 DataFrame 按 'a' 列进行分组，并对 'b' 列应用字典 {"b": "sum"} 形式的聚合
    gb = df.groupby("a")
    result = gb.agg({"b": "sum"})

    # 创建预期结果 DataFrame，包含两行，索引为 [1, 2]
    expected = DataFrame({"b": [5, 4]}, index=Index([1, 2], name="a"))
    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试不同聚合操作在布尔数据上的结果类型是否正确
@pytest.mark.parametrize(
    "op",
    [
        lambda x: x.sum(),
        lambda x: x.cumsum(),
        lambda x: x.transform("sum"),
        lambda x: x.transform("cumsum"),
        lambda x: x.agg("sum"),
        lambda x: x.agg("cumsum"),
    ],
)
def test_bool_agg_dtype(op):
    # 创建包含两行两列的 DataFrame，其中 'b' 列包含布尔值
    df = DataFrame({"a": [1, 1], "b": [False, True]})
    # 创建 Series，使用 'a' 列作为索引
    s = df.set_index("a")["b"]

    # 对 DataFrame 按 'a' 列进行分组，应用传入的聚合操作 op，返回 'b' 列的数据类型
    result = op(df.groupby("a"))["b"].dtype
    # 断言 result 的数据类型是否为整数类型
    assert is_integer_dtype(result)

    # 对 Series s 按 'a' 列进行分组，应用传入的聚合操作 op，返回结果的数据类型
    result = op(s.groupby("a")).dtype
    # 断言 result 的数据类型是否为整数类型
    assert is_integer_dtype(result)


# 定义一个测试函数，测试 DataFrame 的 apply、aggregate 和 transform 方法返回结果的数据类型是否正确
@pytest.mark.parametrize(
    "keys, agg_index",
    [
        (["a"], Index([1], name="a")),  # 设置 keys 为 ["a"]，agg_index 为 Index([1], name="a")
        (["a", "b"], MultiIndex([[1], [2]], [[0], [0]], names=["a", "b"]))  # 设置 keys 为 ["a", "b"]，agg_index 为 MultiIndex 对象
    ],
)
@pytest.mark.parametrize(
    "input_dtype", ["bool", "int32", "int64", "float32", "float64"]  # 参数化输入的数据类型
)
@pytest.mark.parametrize(
    "result_dtype", ["bool", "int32", "int64", "float32", "float64"]  # 参数化预期结果的数据类型
)
@pytest.mark.parametrize("method", ["apply", "aggregate", "transform"])  # 参数化测试方法
def test_callable_result_dtype_frame(
    keys, agg_index, input_dtype, result_dtype, method
):
    # 创建包含三列的 DataFrame，其中 'c' 列的数据类型由 input_dtype 指定
    df = DataFrame({"a": [1], "b": [2], "c": [True]})
    df["c"] = df["c"].astype(input_dtype)

    # 调用 getattr 函数获取对 DataFrame 按 keys 分组后的 method 方法，返回结果
    op = getattr(df.groupby(keys)[["c"]], method)
    # 应用 lambda 函数将 'c' 列转换为 result_dtype，然后选取第一行数据
    result = op(lambda x: x.astype(result_dtype).iloc[0])

    # 根据 method 的值设定预期结果的索引类型
    expected_index = pd.RangeIndex(0, 1) if method == "transform" else agg_index
    # 创建预期结果 DataFrame，只包含 'c' 列，数据类型为 result_dtype
    expected = DataFrame({"c": [df["c"].iloc[0]]}, index=expected_index).astype(
        result_dtype
    )
    # 如果 method 为 'apply'，则设置预期结果的列名为 [0]
    if method == "apply":
        expected.columns.names = [0]
    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
    [
        # 创建一个元组列表，包含两个元组
        (["a"], Index([1], name="a")),  
        # 第一个元组包含单个元素列表 ["a"] 和名为 "a" 的索引对象，索引值为 1
        (["a", "b"], MultiIndex([[1], [2]], [[0], [0]], names=["a", "b"]))
        # 第二个元组包含两个元素列表 ["a", "b"] 和名为 "a", "b" 的多级索引对象，第一级索引为 [1, 2]，第二级索引为 [0, 0]
    ],
@pytest.mark.parametrize("input", [True, 1, 1.0])
# 参数化测试，输入参数为布尔值True、整数1、浮点数1.0
@pytest.mark.parametrize("dtype", [bool, int, float])
# 参数化测试，数据类型参数为bool、int、float
@pytest.mark.parametrize("method", ["apply", "aggregate", "transform"])
# 参数化测试，方法参数为"apply"、"aggregate"、"transform"
def test_callable_result_dtype_series(keys, agg_index, input, dtype, method):
    # GH 21240
    # 创建一个包含特定列 'a', 'b', 'c' 的 DataFrame
    df = DataFrame({"a": [1], "b": [2], "c": [input]})
    # 获取分组后 'c' 列对应方法（apply/aggregate/transform）的操作对象
    op = getattr(df.groupby(keys)["c"], method)
    # 对操作对象应用 lambda 函数，将其类型转换为指定的数据类型，并取第一个值
    result = op(lambda x: x.astype(dtype).iloc[0])
    # 根据方法类型确定预期的索引类型
    expected_index = pd.RangeIndex(0, 1) if method == "transform" else agg_index
    # 创建预期的 Series，其值为 'c' 列的第一个值，并指定索引和名称
    expected = Series([df["c"].iloc[0]], index=expected_index, name="c").astype(dtype)
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)


def test_order_aggregate_multiple_funcs():
    # GH 25692
    # 创建一个包含 'A' 和 'B' 列的 DataFrame
    df = DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 3, 4]})
    # 对 'A' 列进行分组聚合，包括 "sum", "max", "mean", "ohlc", "min" 五种方法
    res = df.groupby("A").agg(["sum", "max", "mean", "ohlc", "min"])
    # 获取聚合结果的第二级列标签
    result = res.columns.levels[1]
    # 创建预期的索引对象，包含 "sum", "max", "mean", "ohlc", "min"
    expected = Index(["sum", "max", "mean", "ohlc", "min"])
    # 断言结果与预期相等
    tm.assert_index_equal(result, expected)


def test_ohlc_ea_dtypes(any_numeric_ea_dtype):
    # GH#37493
    # 创建一个包含 'a' 和 'b' 列的 DataFrame，指定数据类型为参数传入的类型
    df = DataFrame(
        {"a": [1, 1, 2, 3, 4, 4], "b": [22, 11, pd.NA, 10, 20, pd.NA]},
        dtype=any_numeric_ea_dtype,
    )
    # 对 'a' 列进行分组
    gb = df.groupby("a")
    # 对分组后的数据执行 ohlc() 聚合操作
    result = gb.ohlc()
    # 创建预期的 DataFrame，包含 'b' 列的 "open", "high", "low", "close" 四个级别的 MultiIndex
    expected = DataFrame(
        [[22, 22, 11, 11], [pd.NA] * 4, [10] * 4, [20] * 4],
        columns=MultiIndex.from_product([["b"], ["open", "high", "low", "close"]]),
        index=Index([1, 2, 3, 4], dtype=any_numeric_ea_dtype, name="a"),
        dtype=any_numeric_ea_dtype,
    )
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)

    # 对 'a' 列进行分组，并保持结果中的索引列
    gb2 = df.groupby("a", as_index=False)
    # 对分组后的数据执行 ohlc() 聚合操作
    result2 = gb2.ohlc()
    # 重置预期结果的索引
    expected2 = expected.reset_index()
    # 断言结果与预期相等
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
# 参数化测试，数据类型参数为 np.int64 和 np.uint64
@pytest.mark.parametrize("how", ["first", "last", "min", "max", "mean", "median"])
# 参数化测试，方法参数为 "first", "last", "min", "max", "mean", "median"
def test_uint64_type_handling(dtype, how):
    # GH 26310
    # 创建一个包含 'x' 和 'y' 列的 DataFrame
    df = DataFrame({"x": 6903052872240755750, "y": [1, 2]})
    # 对 'y' 列进行分组，并对 'x' 列应用指定方法进行聚合
    expected = df.groupby("y").agg({"x": how})
    # 将 'x' 列的数据类型转换为指定的数据类型
    df.x = df.x.astype(dtype)
    # 再次对 'y' 列进行分组，并对 'x' 列应用指定方法进行聚合
    result = df.groupby("y").agg({"x": how})
    # 如果方法不是 "mean" 或 "median"，则将结果的 'x' 列转换为 np.int64 类型
    if how not in ("mean", "median"):
        result.x = result.x.astype(np.int64)
    # 断言结果与预期相等，精确匹配
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_func_duplicates_raises():
    # GH28426
    # 设置错误消息
    msg = "Function names"
    # 创建一个包含 'A' 和 'B' 列的 DataFrame
    df = DataFrame({"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]})
    # 使用 pytest.raises 断言捕获 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        # 对 'A' 列进行分组，并尝试对其应用包含重复函数名的聚合操作
        df.groupby("A").agg(["min", "min"])


@pytest.mark.parametrize(
    "index",
    [
        pd.CategoricalIndex(list("abc")),
        pd.interval_range(0, 3),
        pd.period_range("2020", periods=3, freq="D"),
        MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ],
)
# 参数化测试，索引参数为四种不同类型的索引对象
def test_agg_index_has_complex_internals(index):
    # GH 31223
    # 创建一个包含 'group' 和 'value' 列的 DataFrame，指定索引为参数传入的索引对象
    df = DataFrame({"group": [1, 1, 2], "value": [0, 1, 0]}, index=index)
    # 对 'group' 列进行分组，并对 'value' 列应用 Series.nunique 方法进行聚合
    result = df.groupby("group").agg({"value": Series.nunique})
    # 创建预期的 DataFrame，包含两列：'group' 列和 'value' 列，分别为 [1, 2] 和 [2, 1]，并将 'group' 列设为索引
    expected = DataFrame({"group": [1, 2], "value": [2, 1]}).set_index("group")
    # 使用测试工具（如 pandas 的 tm.assert_frame_equal）比较 result 和 expected 的内容是否一致
    tm.assert_frame_equal(result, expected)
def test_agg_split_block():
    # 根据该 GitHub issue 进行测试：https://github.com/pandas-dev/pandas/issues/31522
    # 创建一个包含三列的 DataFrame
    df = DataFrame(
        {
            "key1": ["a", "a", "b", "b", "a"],
            "key2": ["one", "two", "one", "two", "one"],
            "key3": ["three", "three", "three", "six", "six"],
        }
    )
    # 对 DataFrame 按 'key1' 分组，然后计算每组的最小值
    result = df.groupby("key1").min()
    # 创建一个预期的 DataFrame，包含指定的键和索引
    expected = DataFrame(
        {"key2": ["one", "one"], "key3": ["six", "six"]},
        index=Index(["a", "b"], name="key1"),
    )
    # 使用测试工具函数确认 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_agg_split_object_part_datetime():
    # 根据该 GitHub pull request 进行测试：https://github.com/pandas-dev/pandas/pull/31616
    # 创建一个包含多列的 DataFrame，其中包含日期、字符串和整数
    df = DataFrame(
        {
            "A": pd.date_range("2000", periods=4),
            "B": ["a", "b", "c", "d"],
            "C": [1, 2, 3, 4],
            "D": ["b", "c", "d", "e"],
            "E": pd.date_range("2000", periods=4),
            "F": [1, 2, 3, 4],
        }
    ).astype(object)
    # 根据多列进行分组，然后计算每组的最小值
    result = df.groupby([0, 0, 0, 0]).min()
    # 创建一个预期的 DataFrame，包含指定的键和索引，并且数据类型为 object
    expected = DataFrame(
        {
            "A": [pd.Timestamp("2000")],
            "B": ["a"],
            "C": [1],
            "D": ["b"],
            "E": [pd.Timestamp("2000")],
            "F": [1],
        },
        index=np.array([0]),
        dtype=object,
    )
    # 使用测试工具函数确认 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


class TestNamedAggregationSeries:
    def test_series_named_agg(self):
        # 创建一个包含四个元素的 Series
        df = Series([1, 2, 3, 4])
        # 根据多列进行分组，然后应用 sum 和 min 聚合函数，给聚合结果命名为 'a' 和 'b'
        gr = df.groupby([0, 0, 1, 1])
        result = gr.agg(a="sum", b="min")
        # 创建一个预期的 DataFrame，包含指定的列和索引
        expected = DataFrame(
            {"a": [3, 7], "b": [1, 3]}, columns=["a", "b"], index=np.array([0, 1])
        )
        # 使用测试工具函数确认 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 再次调用 agg 函数，调整列的顺序为 'b' 和 'a'
        result = gr.agg(b="min", a="sum")
        expected = expected[["b", "a"]]
        # 使用测试工具函数确认 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_no_args_raises(self):
        # 创建一个包含两个元素的 Series，并根据多列进行分组
        gr = Series([1, 2]).groupby([0, 1])
        # 使用 pytest 来测试是否会抛出 TypeError 异常，并包含指定的错误消息
        with pytest.raises(TypeError, match="Must provide"):
            gr.agg()

        # 但是我们允许传入空列表
        result = gr.agg([])
        expected = DataFrame(columns=[])
        # 使用测试工具函数确认 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_series_named_agg_duplicates_no_raises(self):
        # 进行一个特定的测试，关于处理重复列的聚合操作
        gr = Series([1, 2, 3]).groupby([0, 0, 1])
        # 对分组后的数据进行聚合，给聚合结果命名为 'a' 和 'b'
        grouped = gr.agg(a="sum", b="sum")
        # 创建一个预期的 DataFrame，包含指定的列和索引
        expected = DataFrame({"a": [3, 3], "b": [3, 3]}, index=np.array([0, 1]))
        # 使用测试工具函数确认 result 和 expected 是否相等
        tm.assert_frame_equal(expected, grouped)

    def test_mangled(self):
        # 创建一个包含三个元素的 Series，并根据多列进行分组
        gr = Series([1, 2, 3]).groupby([0, 0, 1])
        # 对分组后的数据进行聚合，使用 lambda 函数分别计算 'a' 和 'b' 的值
        result = gr.agg(a=lambda x: 0, b=lambda x: 1)
        # 创建一个预期的 DataFrame，包含指定的列和索引
        expected = DataFrame({"a": [0, 0], "b": [1, 1]}, index=np.array([0, 1]))
        # 使用测试工具函数确认 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "inp",
        [
            pd.NamedAgg(column="anything", aggfunc="min"),
            ("anything", "min"),
            ["anything", "min"],
        ],
    )
    # 定义一个测试函数，用于测试具名聚合操作和命名元组的功能
    def test_named_agg_nametuple(self, inp):
        # 用于标识 GitHub issue 34422 的相关测试内容
        s = Series([1, 1, 2, 2, 3, 3, 4, 5])
        # 准备错误消息字符串，指示期望得到一个函数，但实际接收到的是输入对象的类型名
        msg = f"func is expected but received {type(inp).__name__}"
        # 使用 pytest 来检测是否抛出预期的 TypeError 异常，并验证其错误消息
        with pytest.raises(TypeError, match=msg):
            # 对序列 s 按照其值进行分组，然后进行聚合操作，期望聚合操作中使用的是输入对象 inp
            s.groupby(s.values).agg(a=inp)
class TestNamedAggregationDataFrame:
    # 测试聚合函数的重命名功能
    def test_agg_relabel(self):
        # 创建一个示例数据框
        df = DataFrame(
            {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
        )
        # 使用 groupby 分组后，对每组进行聚合操作，并为聚合结果重新命名列名
        result = df.groupby("group").agg(a_max=("A", "max"), b_max=("B", "max"))
        # 期望的结果数据框，包含重命名后的列名和相应的数据
        expected = DataFrame(
            {"a_max": [1, 3], "b_max": [6, 8]},
            index=Index(["a", "b"], name="group"),
            columns=["a_max", "b_max"],
        )
        # 断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)

        # 测试无关顺序的情况
        p98 = functools.partial(np.percentile, q=98)
        # 对多个列进行聚合操作，并包括自定义百分位数函数
        result = df.groupby("group").agg(
            b_min=("B", "min"),
            a_min=("A", "min"),
            a_mean=("A", "mean"),
            a_max=("A", "max"),
            b_max=("B", "max"),
            a_98=("A", p98),
        )
        # 期望的结果数据框，包含多个聚合列和自定义百分位数的计算结果
        expected = DataFrame(
            {
                "b_min": [5, 7],
                "a_min": [0, 2],
                "a_mean": [0.5, 2.5],
                "a_max": [1, 3],
                "b_max": [6, 8],
                "a_98": [0.98, 2.98],
            },
            index=Index(["a", "b"], name="group"),
            columns=["b_min", "a_min", "a_mean", "a_max", "b_max", "a_98"],
        )
        # 断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)

    # 测试非标识符作为列名的聚合重命名
    def test_agg_relabel_non_identifier(self):
        # 创建一个示例数据框
        df = DataFrame(
            {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
        )
        # 对指定列进行聚合操作，并使用非标识符作为重命名的列名
        result = df.groupby("group").agg(**{"my col": ("A", "max")})
        # 期望的结果数据框，包含重命名后的列名和相应的数据
        expected = DataFrame({"my col": [1, 3]}, index=Index(["a", "b"], name="group"))
        # 断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)

    # 测试不会引发错误的重复聚合列
    def test_duplicate_no_raises(self):
        # GH 28426, 如果在相同列上使用相同的输入函数，不应引发错误
        # 创建一个示例数据框
        df = DataFrame({"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]})

        # 对分组后的数据进行聚合操作，并为每个聚合结果重新命名列名
        grouped = df.groupby("A").agg(a=("B", "min"), b=("B", "min"))
        # 期望的结果数据框，包含重新命名后的列名和相应的数据
        expected = DataFrame({"a": [1, 3], "b": [1, 3]}, index=Index([0, 1], name="A"))
        # 断言两个数据框是否相等
        tm.assert_frame_equal(grouped, expected)

        quant50 = functools.partial(np.percentile, q=50)
        quant70 = functools.partial(np.percentile, q=70)
        quant50.__name__ = "quant50"
        quant70.__name__ = "quant70"

        # 创建一个示例数据框
        test = DataFrame({"col1": ["a", "a", "b", "b", "b"], "col2": [1, 2, 3, 4, 5]})

        # 对分组后的数据进行聚合操作，并包括自定义百分位数函数
        grouped = test.groupby("col1").agg(
            quantile_50=("col2", quant50), quantile_70=("col2", quant70)
        )
        # 期望的结果数据框，包含多个聚合列和自定义百分位数的计算结果
        expected = DataFrame(
            {"quantile_50": [1.5, 4.0], "quantile_70": [1.7, 4.4]},
            index=Index(["a", "b"], name="col1"),
        )
        # 断言两个数据框是否相等
        tm.assert_frame_equal(grouped, expected)
    # 定义测试方法，用于测试按级别进行聚合和重新标记
    def test_agg_relabel_with_level(self):
        # 创建一个包含多层索引的 DataFrame
        df = DataFrame(
            {"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]},
            index=MultiIndex.from_product([["A", "B"], ["a", "b"]]),
        )
        # 按第一层级进行分组，对各列进行聚合并重新命名
        result = df.groupby(level=0).agg(
            aa=("A", "max"), bb=("A", "min"), cc=("B", "mean")
        )
        # 创建预期的结果 DataFrame
        expected = DataFrame(
            {"aa": [0, 1], "bb": [0, 1], "cc": [1.5, 3.5]}, index=["A", "B"]
        )
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试按其他方式进行聚合时抛出异常
    def test_agg_relabel_other_raises(self):
        # 创建一个简单的 DataFrame
        df = DataFrame({"A": [0, 0, 1], "B": [1, 2, 3]})
        # 按'A'列进行分组
        grouped = df.groupby("A")
        # 准备匹配的异常消息
        match = "Must provide"
        # 使用 pytest 断言抛出指定类型和消息的异常
        with pytest.raises(TypeError, match=match):
            grouped.agg(foo=1)

        with pytest.raises(TypeError, match=match):
            grouped.agg()

        with pytest.raises(TypeError, match=match):
            grouped.agg(a=("B", "max"), b=(1, 2, 3))

    # 定义测试方法，用于测试聚合时缺少标签抛出异常
    def test_missing_raises(self):
        # 创建一个简单的 DataFrame
        df = DataFrame({"A": [0, 1], "B": [1, 2]})
        # 准备匹配的异常消息
        msg = r"Label\(s\) \['C'\] do not exist"
        # 使用 pytest 断言抛出指定类型和消息的异常
        with pytest.raises(KeyError, match=msg):
            df.groupby("A").agg(c=("C", "sum"))

    # 定义测试方法，用于测试使用命名元组进行聚合
    def test_agg_namedtuple(self):
        # 创建一个简单的 DataFrame
        df = DataFrame({"A": [0, 1], "B": [1, 2]})
        # 按'A'列进行分组，对'B'列进行命名聚合
        result = df.groupby("A").agg(
            b=pd.NamedAgg("B", "sum"), c=pd.NamedAgg(column="B", aggfunc="count")
        )
        # 创建预期的结果 DataFrame
        expected = df.groupby("A").agg(b=("B", "sum"), c=("B", "count"))
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在聚合时使用函数对列进行处理
    def test_mangled(self):
        # 创建一个包含多列的 DataFrame
        df = DataFrame({"A": [0, 1], "B": [1, 2], "C": [3, 4]})
        # 按'A'列进行分组，对'B'列和'C'列应用自定义函数进行聚合
        result = df.groupby("A").agg(b=("B", lambda x: 0), c=("C", lambda x: 1))
        # 创建预期的结果 DataFrame
        expected = DataFrame({"b": [0, 0], "c": [1, 1]}, index=Index([0, 1], name="A"))
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3",
    [
        (
            (("y", "A"), "max"),  # 第一个参数组合：以 ('y', 'A') 列进行最大值聚合
            (("y", "A"), np.mean),  # 第二个参数组合：以 ('y', 'A') 列进行均值聚合
            (("y", "B"), "mean"),  # 第三个参数组合：以 ('y', 'B') 列进行均值聚合
            [1, 3],  # 预期结果1：最大值聚合结果
            [0.5, 2.5],  # 预期结果2：均值聚合结果
            [5.5, 7.5],  # 预期结果3：均值聚合结果
        ),
        (
            (("y", "A"), lambda x: max(x)),  # 第一个参数组合：以 ('y', 'A') 列，使用最大值函数聚合
            (("y", "A"), lambda x: 1),  # 第二个参数组合：以 ('y', 'A') 列，直接返回1
            (("y", "B"), np.mean),  # 第三个参数组合：以 ('y', 'B') 列进行均值聚合
            [1, 3],  # 预期结果1：最大值函数聚合结果
            [1, 1],  # 预期结果2：直接返回1的结果
            [5.5, 7.5],  # 预期结果3：均值聚合结果
        ),
        (
            pd.NamedAgg(("y", "A"), "max"),  # 第一个参数组合：以 ('y', 'A') 列进行最大值聚合
            pd.NamedAgg(("y", "B"), np.mean),  # 第二个参数组合：以 ('y', 'B') 列进行均值聚合
            pd.NamedAgg(("y", "A"), lambda x: 1),  # 第三个参数组合：以 ('y', 'A') 列，返回1的命名聚合
            [1, 3],  # 预期结果1：最大值聚合结果
            [5.5, 7.5],  # 预期结果2：均值聚合结果
            [1, 1],  # 预期结果3：返回1的结果
        ),
    ],
)
def test_agg_relabel_multiindex_column(
    agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3
):
    # GH 29422, add tests for multiindex column cases
    df = DataFrame(
        {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
    )
    df.columns = MultiIndex.from_tuples([("x", "group"), ("y", "A"), ("y", "B")])
    idx = Index(["a", "b"], name=("x", "group"))

    result = df.groupby(("x", "group")).agg(a_max=(("y", "A"), "max"))
    expected = DataFrame({"a_max": [1, 3]}, index=idx)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(("x", "group")).agg(
        col_1=agg_col1, col_2=agg_col2, col_3=agg_col3
    )
    expected = DataFrame(
        {"col_1": agg_result1, "col_2": agg_result2, "col_3": agg_result3}, index=idx
    )
    tm.assert_frame_equal(result, expected)


def test_agg_relabel_multiindex_raises_not_exist():
    # GH 29422, add test for raises scenario when aggregate column does not exist
    df = DataFrame(
        {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
    )
    df.columns = MultiIndex.from_tuples([("x", "group"), ("y", "A"), ("y", "B")])

    with pytest.raises(KeyError, match="do not exist"):
        df.groupby(("x", "group")).agg(a=(("Y", "a"), "max"))


def test_agg_relabel_multiindex_duplicates():
    # GH29422, add test for raises scenario when getting duplicates
    # GH28426, after this change, duplicates should also work if the relabelling is
    # different
    df = DataFrame(
        {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
    )
    df.columns = MultiIndex.from_tuples([("x", "group"), ("y", "A"), ("y", "B")])

    result = df.groupby(("x", "group")).agg(
        a=(("y", "A"), "min"), b=(("y", "A"), "min")
    )
    idx = Index(["a", "b"], name=("x", "group"))
    expected = DataFrame({"a": [0, 2], "b": [0, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"c": ["min"]}, {"b": [], "c": ["min"]}])
def test_groupby_aggregate_empty_key(kwargs):
    # GH: 32580
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [1, 2, 4]})
    result = df.groupby("a").agg(kwargs)
    # 创建预期的DataFrame对象，包含两行数据 [1, 4]，行索引为整数64位数值类型的Index对象，名称为'a'
    # 列为多级索引的MultiIndex对象，包含一个元组["c", "min"]，表示列名和层级名
    expected = DataFrame(
        [1, 4],
        index=Index([1, 2], dtype="int64", name="a"),
        columns=MultiIndex.from_tuples([["c", "min"]]),
    )
    # 使用测试工具tm.assert_frame_equal()比较结果DataFrame对象(result)和预期的DataFrame对象(expected)
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试在空键的情况下，空返回值的聚合操作是否正常
def test_groupby_aggregate_empty_key_empty_return():
    # 创建一个包含三列的数据帧，每列都包含几个整数
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [1, 2, 4]})
    # 对数据帧按列 'a' 进行分组，然后对 'b' 列应用空列表作为聚合函数
    result = df.groupby("a").agg({"b": []})
    # 创建一个期望的空数据帧，包含一个空的多级索引列 'b'
    expected = DataFrame(columns=MultiIndex(levels=[["b"], []], codes=[[], []]))
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试在多级索引数据帧中应用空列表聚合的情况
def test_groupby_aggregate_empty_with_multiindex_frame():
    # 创建一个空列名为 'a', 'b', 'c' 的数据帧
    df = DataFrame(columns=["a", "b", "c"])
    # 对数据帧按 ['a', 'b'] 列进行分组，禁用分组键作为索引，然后对 'c' 列应用 list 函数作为聚合函数
    result = df.groupby(["a", "b"], group_keys=False).agg(d=("c", list))
    # 创建一个期望的数据帧，包含一个列 'd'，并且有一个空的多级索引 ['a', 'b']
    expected = DataFrame(
        columns=["d"], index=MultiIndex([[], []], [[], []], names=["a", "b"])
    )
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试当聚合函数重新标记列名且指定 as_index=False 时是否丢失结果
def test_grouby_agg_loses_results_with_as_index_false_relabel():
    # 创建一个包含 'key' 和 'val' 列的数据帧
    df = DataFrame(
        {"key": ["x", "y", "z", "x", "y", "z"], "val": [1.0, 0.8, 2.0, 3.0, 3.6, 0.75]}
    )
    # 对数据帧按 'key' 列进行分组，禁用分组键作为索引
    grouped = df.groupby("key", as_index=False)
    # 对分组后的数据帧应用最小值聚合函数，并重新命名为 'min_val'
    result = grouped.agg(min_val=pd.NamedAgg(column="val", aggfunc="min"))
    # 创建一个期望的数据帧，包含 'key' 和 'min_val' 两列
    expected = DataFrame({"key": ["x", "y", "z"], "min_val": [1.0, 0.8, 0.75]})
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试当聚合函数重新标记列名且指定 as_index=False 时是否丢失结果，同时检查多级索引是否以正确的顺序返回
def test_grouby_agg_loses_results_with_as_index_false_relabel_multiindex():
    # 创建一个包含 'key', 'key1', 'val' 列的数据帧
    df = DataFrame(
        {
            "key": ["x", "y", "x", "y", "x", "x"],
            "key1": ["a", "b", "c", "b", "a", "c"],
            "val": [1.0, 0.8, 2.0, 3.0, 3.6, 0.75],
        }
    )
    # 对数据帧按 ['key', 'key1'] 列进行分组，禁用分组键作为索引
    grouped = df.groupby(["key", "key1"], as_index=False)
    # 对分组后的数据帧应用最小值聚合函数，并重新命名为 'min_val'
    result = grouped.agg(min_val=pd.NamedAgg(column="val", aggfunc="min"))
    # 创建一个期望的数据帧，包含 'key', 'key1', 'min_val' 三列，且 'key', 'key1' 是多级索引
    expected = DataFrame(
        {"key": ["x", "x", "y"], "key1": ["a", "c", "b"], "min_val": [1.0, 0.75, 0.8]}
    )
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试在指定 as_index=True 时对分组应用聚合函数的情况
def test_groupby_as_index_agg(df):
    # 对数据帧按 'A' 列进行分组，保留分组键作为索引
    grouped = df.groupby("A", as_index=False)

    # 对分组后的数据帧中的 'C', 'D' 列应用平均值聚合函数
    result = grouped[["C", "D"]].agg("mean")
    # 创建一个期望的数据帧，包含平均值聚合后的 'C', 'D' 列
    expected = grouped.mean(numeric_only=True)
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result, expected)

    # 对分组后的数据帧中的 'C' 列应用平均值聚合函数，'D' 列应用求和聚合函数
    result2 = grouped.agg({"C": "mean", "D": "sum"})
    # 创建一个期望的数据帧，包含 'C' 列的平均值和 'D' 列的求和
    expected2 = grouped.mean(numeric_only=True)
    expected2["D"] = grouped.sum()["D"]
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result2, expected2)

    # 对数据帧按 ['A', 'B'] 列进行分组，保留分组键作为索引
    grouped = df.groupby(["A", "B"], as_index=False)

    # 对分组后的数据帧应用平均值聚合函数
    result = grouped.agg("mean")
    # 创建一个期望的数据帧，包含平均值聚合后的所有列
    expected = grouped.mean()
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result, expected)

    # 对分组后的数据帧中的 'C' 列应用平均值聚合函数，'D' 列应用求和聚合函数
    result2 = grouped.agg({"C": "mean", "D": "sum"})
    # 创建一个期望的数据帧，包含 'C' 列的平均值和 'D' 列的求和
    expected2 = grouped.mean()
    expected2["D"] = grouped.sum()["D"]
    # 使用测试工具比较结果数据帧和期望数据帧
    tm.assert_frame_equal(result2, expected2)
    # 计算分组后列 "C" 的和，返回一个 Series
    expected3 = grouped["C"].sum()
    
    # 将上一步得到的 Series 转换为 DataFrame，并将列名 "C" 改为 "Q"
    expected3 = DataFrame(expected3).rename(columns={"C": "Q"})
    
    # 准备错误消息字符串
    msg = "nested renamer is not supported"
    
    # 使用 pytest 检查是否会抛出 SpecificationError 异常，并匹配特定错误消息
    with pytest.raises(SpecificationError, match=msg):
        # 对分组后的列 "C" 应用聚合函数 {"Q": "sum"}，预期会抛出异常
        grouped["C"].agg({"Q": "sum"})

    # 创建一个 DataFrame df，其中包含随机整数数据，列名为 "jim", "joe", "jolie"
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, (50, 3)),
        columns=["jim", "joe", "jolie"],
    )
    
    # 创建一个 Series ts，其中包含随机整数数据，名称为 "jim"
    ts = Series(np.random.default_rng(2).integers(5, 10, 50), name="jim")

    # 根据 Series ts 对 DataFrame df 进行分组
    gr = df.groupby(ts)
    
    # 对每个分组调用 nth(0) 方法，内部调用 set_selection_from_grouper 方法
    gr.nth(0)

    # 遍历操作列表 ["mean", "max", "count", "idxmax", "cumsum", "all"]
    for attr in ["mean", "max", "count", "idxmax", "cumsum", "all"]:
        # 根据 ts 对 df 进行分组，并不将分组键作为索引
        gr = df.groupby(ts, as_index=False)
        # 调用 getattr(gr, attr)() 方法，返回 DataFrame left
        left = getattr(gr, attr)()

        # 根据 ts.values 对 df 进行分组，将分组键作为索引，并重置索引
        gr = df.groupby(ts.values, as_index=True)
        # 调用 getattr(gr, attr)().reset_index(drop=True) 方法，返回 DataFrame right
        right = getattr(gr, attr)().reset_index(drop=True)

        # 使用 tm.assert_frame_equal 检查 left 和 right 是否相等
        tm.assert_frame_equal(left, right)
@pytest.mark.parametrize(
    "func", [lambda s: s.mean(), lambda s: np.mean(s), lambda s: np.nanmean(s)]
)
def test_multiindex_custom_func(func):
    # 使用 parametrize 标记，为 func 参数传入三个不同的函数：s.mean()、np.mean(s)、np.nanmean(s)
    # 测试多层索引下的自定义聚合函数，见 GH 31777
    data = [[1, 4, 2], [5, 7, 1]]
    # 创建 DataFrame，指定列的多层索引
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays(
            [[1, 1, 2], [3, 4, 3]], names=["Sisko", "Janeway"]
        ),
    )
    # 对 DataFrame 按照第一列和第二列进行分组聚合，应用传入的 func 函数
    result = df.groupby(np.array([0, 1])).agg(func)
    # 期望的结果字典
    expected_dict = {
        (1, 3): {0: 1.0, 1: 5.0},
        (1, 4): {0: 4.0, 1: 7.0},
        (2, 3): {0: 2.0, 1: 1.0},
    }
    # 构建期望的 DataFrame
    expected = DataFrame(expected_dict, index=np.array([0, 1]), columns=df.columns)
    # 使用 assert_frame_equal 检查结果是否与期望一致
    tm.assert_frame_equal(result, expected)


def myfunc(s):
    return np.percentile(s, q=0.90)


@pytest.mark.parametrize("func", [lambda s: np.percentile(s, q=0.90), myfunc])
def test_lambda_named_agg(func):
    # 见 GH 28467
    # 使用 parametrize 标记，为 func 参数传入两个不同的函数：np.percentile(s, q=0.90) 和 myfunc
    animals = DataFrame(
        {
            "kind": ["cat", "dog", "cat", "dog"],
            "height": [9.1, 6.0, 9.5, 34.0],
            "weight": [7.9, 7.5, 9.9, 198.0],
        }
    )
    # 对 animals DataFrame 按照 'kind' 列进行分组聚合，分别计算 'height' 列的均值和 func 函数的结果
    result = animals.groupby("kind").agg(
        mean_height=("height", "mean"), perc90=("height", func)
    )
    # 期望的结果 DataFrame
    expected = DataFrame(
        [[9.3, 9.1036], [20.0, 6.252]],
        columns=["mean_height", "perc90"],
        index=Index(["cat", "dog"], name="kind"),
    )
    # 使用 assert_frame_equal 检查结果是否与期望一致
    tm.assert_frame_equal(result, expected)


def test_aggregate_mixed_types():
    # 见 GH 16916
    # 创建一个包含混合类型的 DataFrame
    df = DataFrame(
        data=np.array([0] * 9).reshape(3, 3), columns=list("XYZ"), index=list("abc")
    )
    df["grouping"] = ["group 1", "group 1", 2]
    # 对 DataFrame 按照 'grouping' 列进行分组聚合，将每列的值转换为列表形式
    result = df.groupby("grouping").aggregate(lambda x: x.tolist())
    # 期望的结果 DataFrame
    expected_data = [[[0], [0], [0]], [[0, 0], [0, 0], [0, 0]]]
    expected = DataFrame(
        expected_data,
        index=Index([2, "group 1"], dtype="object", name="grouping"),
        columns=Index(["X", "Y", "Z"], dtype="object"),
    )
    # 使用 assert_frame_equal 检查结果是否与期望一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="Not implemented;see GH 31256")
def test_aggregate_udf_na_extension_type():
    # https://github.com/pandas-dev/pandas/pull/31359
    # This is currently failing to cast back to Int64Dtype.
    # The presence of the NA causes two problems
    # 1. NA is not an instance of Int64Dtype.type (numpy.int64)
    # 2. The presence of an NA forces object type, so the non-NA values is
    #    a Python int rather than a NumPy int64. Python ints aren't
    #    instances of numpy.int64.
    def aggfunc(x):
        if all(x > 2):
            return 1
        else:
            return pd.NA

    df = DataFrame({"A": pd.array([1, 2, 3])})
    # 对 DataFrame 按照 [1, 1, 2] 列进行分组聚合，应用自定义的 aggfunc 函数
    result = df.groupby([1, 1, 2]).agg(aggfunc)
    # 期望的结果 DataFrame
    expected = DataFrame({"A": pd.array([1, pd.NA], dtype="Int64")}, index=[1, 2])
    # 使用 assert_frame_equal 检查结果是否与期望一致
    tm.assert_frame_equal(result, expected)


class TestLambdaMangling:
    def test_basic(self):
        # 创建一个测试用的 DataFrame，包含两列 "A" 和 "B"
        df = DataFrame({"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]})
        
        # 对 DataFrame 进行分组（按列 "A"），并对列 "B" 应用两个 lambda 函数进行聚合
        result = df.groupby("A").agg({"B": [lambda x: 0, lambda x: 1]})
        
        # 创建期望的 DataFrame，包含多级列索引 ("B", "<lambda_0>") 和 ("B", "<lambda_1>")
        expected = DataFrame(
            {("B", "<lambda_0>"): [0, 0], ("B", "<lambda_1>"): [1, 1]},
            index=Index([0, 1], name="A"),
        )
        
        # 使用 pytest 框架的断言方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_mangle_series_groupby(self):
        # 创建一个 Series 对象，然后按给定的键进行分组
        gr = Series([1, 2, 3, 4]).groupby([0, 0, 1, 1])
        
        # 对分组后的结果应用两个 lambda 函数进行聚合
        result = gr.agg([lambda x: 0, lambda x: 1])
        
        # 创建期望的 DataFrame，包含列 "<lambda_0>" 和 "<lambda_1>"，以及索引 [0, 1]
        exp_data = {"<lambda_0>": [0, 0], "<lambda_1>": [1, 1]}
        expected = DataFrame(exp_data, index=np.array([0, 1]))
        
        # 使用 pytest 框架的断言方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(reason="GH-26611. kwargs for multi-agg.")
    def test_with_kwargs(self):
        # 定义两个 lambda 函数，每个函数有默认参数
        f1 = lambda x, y, b=1: x.sum() + y + b
        f2 = lambda x, y, b=2: x.sum() + y * b
        
        # 对 Series 进行分组（按键 [0, 0]），然后应用 f1 和 f2 两个函数进行聚合
        result = Series([1, 2]).groupby([0, 0]).agg([f1, f2], 0)
        
        # 创建期望的 DataFrame，包含列 "<lambda_0>" 和 "<lambda_1>"，以及对应的结果值
        expected = DataFrame({"<lambda_0>": [4], "<lambda_1>": [6]})
        
        # 使用 pytest 框架的断言方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 对同一组 Series 应用 f1 和 f2 两个函数进行聚合，同时传递额外的参数 b=10
        result = Series([1, 2]).groupby([0, 0]).agg([f1, f2], 0, b=10)
        
        # 创建期望的 DataFrame，包含列 "<lambda_0>" 和 "<lambda_1>"，以及对应的结果值
        expected = DataFrame({"<lambda_0>": [13], "<lambda_1>": [30]})
        
        # 使用 pytest 框架的断言方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_agg_with_one_lambda(self):
        # 创建一个包含 "kind"、"height" 和 "weight" 列的 DataFrame
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0],
                "weight": [7.9, 7.5, 9.9, 198.0],
            }
        )

        # 定义要应用的聚合函数的列名
        columns = ["height_sqr_min", "height_max", "weight_max"]
        
        # 创建期望的 DataFrame，包含指定列名和相应的聚合结果
        expected = DataFrame(
            {
                "height_sqr_min": [82.81, 36.00],
                "height_max": [9.5, 34.0],
                "weight_max": [9.9, 198.0],
            },
            index=Index(["cat", "dog"], name="kind"),
            columns=columns,
        )

        # 使用 pd.NamedAgg 方式，对 DataFrame 进行分组并应用聚合函数
        result1 = df.groupby(by="kind").agg(
            height_sqr_min=pd.NamedAgg(column="height", aggfunc=lambda x: np.min(x**2)),
            height_max=pd.NamedAgg(column="height", aggfunc="max"),
            weight_max=pd.NamedAgg(column="weight", aggfunc="max"),
        )
        
        # 使用 pytest 框架的断言方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result1, expected)

        # 使用 (col, aggfunc) 方式，对 DataFrame 进行分组并应用聚合函数
        result2 = df.groupby(by="kind").agg(
            height_sqr_min=("height", lambda x: np.min(x**2)),
            height_max=("height", "max"),
            weight_max=("weight", "max"),
        )
        
        # 使用 pytest 框架的断言方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result2, expected)
    def test_agg_multiple_lambda(self):
        # GH25719, test for DataFrameGroupby.agg with multiple lambdas
        # with mixed aggfunc
        
        # 创建一个测试用的 DataFrame，包含动物种类、身高和体重数据
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0],
                "weight": [7.9, 7.5, 9.9, 198.0],
            }
        )
        
        # 定义预期的列名和数据内容的 DataFrame
        columns = [
            "height_sqr_min",
            "height_max",
            "weight_max",
            "height_max_2",
            "weight_min",
        ]
        expected = DataFrame(
            {
                "height_sqr_min": [82.81, 36.00],
                "height_max": [9.5, 34.0],
                "weight_max": [9.9, 198.0],
                "height_max_2": [9.5, 34.0],
                "weight_min": [7.9, 7.5],
            },
            index=Index(["cat", "dog"], name="kind"),
            columns=columns,
        )

        # 检查 agg(key=(col, aggfunc)) 的情况
        result1 = df.groupby(by="kind").agg(
            height_sqr_min=("height", lambda x: np.min(x**2)),
            height_max=("height", "max"),
            weight_max=("weight", "max"),
            height_max_2=("height", lambda x: np.max(x)),
            weight_min=("weight", lambda x: np.min(x)),
        )
        
        # 断言 result1 和预期的 DataFrame 相等
        tm.assert_frame_equal(result1, expected)

        # 检查 pd.NamedAgg 的情况
        result2 = df.groupby(by="kind").agg(
            height_sqr_min=pd.NamedAgg(column="height", aggfunc=lambda x: np.min(x**2)),
            height_max=pd.NamedAgg(column="height", aggfunc="max"),
            weight_max=pd.NamedAgg(column="weight", aggfunc="max"),
            height_max_2=pd.NamedAgg(column="height", aggfunc=lambda x: np.max(x)),
            weight_min=pd.NamedAgg(column="weight", aggfunc=lambda x: np.min(x)),
        )
        
        # 断言 result2 和预期的 DataFrame 相等
        tm.assert_frame_equal(result2, expected)
def test_pass_args_kwargs_duplicate_columns(tsframe, as_index):
    # 设置测试框架的列名称，包括重复的列名
    tsframe.columns = ["A", "B", "A", "C"]
    # 根据月份对数据进行分组，根据参数决定是否保留分组键作为索引
    gb = tsframe.groupby(lambda x: x.month, as_index=as_index)

    # 对分组后的数据进行聚合计算，计算每列的80th百分位数
    res = gb.agg(np.percentile, 80, axis=0)

    # 期望的数据，根据每个月的数据计算80th百分位数
    ex_data = {
        1: tsframe[tsframe.index.month == 1].quantile(0.8),
        2: tsframe[tsframe.index.month == 2].quantile(0.8),
    }
    expected = DataFrame(ex_data).T
    # 如果不保留分组键作为索引，则插入一个新的索引列
    if not as_index:
        expected.insert(0, "index", [1, 2])
        expected.index = Index(range(2))

    # 断言两个数据框是否相等
    tm.assert_frame_equal(res, expected)


def test_groupby_get_by_index():
    # GH 33439
    # 创建包含两列的数据框
    df = DataFrame({"A": ["S", "W", "W"], "B": [1.0, 1.0, 2.0]})
    # 根据列'A'进行分组，并使用lambda函数获取每组的最后一个元素作为聚合结果
    res = df.groupby("A").agg({"B": lambda x: x.get(x.index[-1])})
    # 期望的数据框，将'A'列设置为索引
    expected = DataFrame({"A": ["S", "W"], "B": [1.0, 2.0]}).set_index("A")
    # 断言两个数据框是否相等
    tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "grp_col_dict, exp_data",
    [
        ({"nr": "min", "cat_ord": "min"}, {"nr": [1, 5], "cat_ord": ["a", "c"]}),
        ({"cat_ord": "min"}, {"cat_ord": ["a", "c"]}),
        ({"nr": "min"}, {"nr": [1, 5]}),
    ],
)
def test_groupby_single_agg_cat_cols(grp_col_dict, exp_data):
    # test single aggregations on ordered categorical cols GHGH27800

    # 创建输入数据框
    input_df = DataFrame(
        {
            "nr": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_ord": list("aabbccdd"),
            "cat": list("aaaabbbb"),
        }
    )

    # 将'cat'和'cat_ord'列转换为分类类型，并保持'cat_ord'列的顺序
    input_df = input_df.astype({"cat": "category", "cat_ord": "category"})
    input_df["cat_ord"] = input_df["cat_ord"].cat.as_ordered()
    # 对数据框按'cat'列进行分组，并应用聚合操作
    result_df = input_df.groupby("cat", observed=False).agg(grp_col_dict)

    # 创建期望的数据框
    cat_index = pd.CategoricalIndex(
        ["a", "b"], categories=["a", "b"], ordered=False, name="cat", dtype="category"
    )
    expected_df = DataFrame(data=exp_data, index=cat_index)

    # 如果期望数据中包含'cat_ord'列，则保持其为有序分类列
    if "cat_ord" in expected_df:
        dtype = input_df["cat_ord"].dtype
        expected_df["cat_ord"] = expected_df["cat_ord"].astype(dtype)

    # 断言两个数据框是否相等
    tm.assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    "grp_col_dict, exp_data",
    [
        ({"nr": ["min", "max"], "cat_ord": "min"}, [(1, 4, "a"), (5, 8, "c")]),
        ({"nr": "min", "cat_ord": ["min", "max"]}, [(1, "a", "b"), (5, "c", "d")]),
        ({"cat_ord": ["min", "max"]}, [("a", "b"), ("c", "d")]),
    ],
)
def test_groupby_combined_aggs_cat_cols(grp_col_dict, exp_data):
    # test combined aggregations on ordered categorical cols GH27800

    # 创建输入数据框
    input_df = DataFrame(
        {
            "nr": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_ord": list("aabbccdd"),
            "cat": list("aaaabbbb"),
        }
    )

    # 将'cat'和'cat_ord'列转换为分类类型，并保持'cat_ord'列的顺序
    input_df = input_df.astype({"cat": "category", "cat_ord": "category"})
    input_df["cat_ord"] = input_df["cat_ord"].cat.as_ordered()
    # 对输入数据按照 "cat" 列进行分组，应用 grp_col_dict 中的聚合函数进行聚合，不考虑未观察到的分类
    result_df = input_df.groupby("cat", observed=False).agg(grp_col_dict)

    # 创建期望的数据框架 expected_df

    # 创建一个分类索引对象，指定类别为 ["a", "b"]，无序，名称为 "cat"，数据类型为 category
    cat_index = pd.CategoricalIndex(
        ["a", "b"], categories=["a", "b"], ordered=False, name="cat", dtype="category"
    )

    # 解包 grp_col_dict，创建多级索引元组
    # 这些元组将用于创建期望的数据框架索引
    multi_index_list = []
    for k, v in grp_col_dict.items():
        if isinstance(v, list):
            multi_index_list.extend([k, value] for value in v)
        else:
            multi_index_list.append([k, v])
    multi_index = MultiIndex.from_tuples(tuple(multi_index_list))

    # 使用 exp_data 数据和 multi_index 列名创建 DataFrame 对象，以 cat_index 作为索引
    expected_df = DataFrame(data=exp_data, columns=multi_index, index=cat_index)

    # 遍历 expected_df 的每一列
    for col in expected_df.columns:
        # 如果列是元组并且包含 "cat_ord"，则进行类型转换以保留有序分类
        if isinstance(col, tuple) and "cat_ord" in col:
            expected_df[col] = expected_df[col].astype(input_df["cat_ord"].dtype)

    # 使用测试工具比较 result_df 和 expected_df，确保它们相等
    tm.assert_frame_equal(result_df, expected_df)
def test_nonagg_agg():
    # GH 35490 - Single/Multiple agg of non-agg function give same results
    # GH 35490号问题 - 对非聚合函数进行单个/多个agg操作得到相同的结果
    # TODO: agg should raise for functions that don't aggregate
    # TODO: agg应该对不进行聚合的函数抛出异常
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 2, 1]})
    # 创建DataFrame对象df，包含列"a"和"b"
    g = df.groupby("a")
    # 按列"a"对df进行分组，返回GroupBy对象g

    result = g.agg(["cumsum"])
    # 对分组后的g对象应用累计和函数"cumsum"进行聚合，返回结果DataFrame result
    result.columns = result.columns.droplevel(-1)
    # 删除结果DataFrame result的最内层列索引
    expected = g.agg("cumsum")
    # 对分组后的g对象应用累计和函数"cumsum"进行聚合，返回期望的结果DataFrame expected

    tm.assert_frame_equal(result, expected)
    # 使用测试框架tm进行结果的比较，确保result与expected相等


def test_aggregate_datetime_objects():
    # https://github.com/pandas-dev/pandas/issues/36003
    # 确保我们不会因超出范围的日期时间而引发错误，但保持对象dtype不变
    df = DataFrame(
        {
            "A": ["X", "Y"],
            "B": [
                datetime.datetime(2005, 1, 1, 10, 30, 23, 540000),
                datetime.datetime(3005, 1, 1, 10, 30, 23, 540000),
            ],
        }
    )
    # 创建DataFrame对象df，包含列"A"和"B"，B列包含两个日期时间对象
    result = df.groupby("A").B.max()
    # 对df按列"A"分组，并计算每组中B列的最大值，返回Series对象result
    expected = df.set_index("A")["B"]
    # 将df按列"A"设置为索引，返回Series对象expected，包含B列

    tm.assert_series_equal(result, expected)
    # 使用测试框架tm进行结果的比较，确保result与expected相等


def test_groupby_index_object_dtype():
    # GH 40014
    df = DataFrame({"c0": ["x", "x", "x"], "c1": ["x", "x", "y"], "p": [0, 1, 2]})
    # 创建DataFrame对象df，包含列"c0", "c1"和"p"
    df.index = df.index.astype("O")
    # 将df的索引转换为对象类型("O")
    grouped = df.groupby(["c0", "c1"])
    # 对df按列"c0"和"c1"进行分组，返回GroupBy对象grouped
    res = grouped.p.agg(lambda x: all(x > 0))
    # 对grouped中的列"p"应用lambda函数，判断每组中所有元素是否大于0，返回结果Series res
    # 检查在使用对象类型索引时，agg()提供用户定义函数是否产生正确的索引形状

    expected_index = MultiIndex.from_tuples(
        [("x", "x"), ("x", "y")], names=("c0", "c1")
    )
    # 创建MultiIndex对象expected_index，包含索引标签为("c0", "c1")的元组
    expected = Series([False, True], index=expected_index, name="p")
    # 创建Series对象expected，包含值[False, True]，使用expected_index作为索引，名称为"p"

    tm.assert_series_equal(res, expected)
    # 使用测试框架tm进行结果的比较，确保res与expected相等


def test_timeseries_groupby_agg():
    # GH#43290
    def func(ser):
        if ser.isna().all():
            return None
        return np.sum(ser)

    # 定义函数func，接受Series对象ser作为参数，检查是否所有元素均为缺失值，是则返回None，否则返回元素和

    df = DataFrame([1.0], index=[pd.Timestamp("2018-01-16 00:00:00+00:00")])
    # 创建DataFrame对象df，包含值为1.0的列，索引为时间戳
    res = df.groupby(lambda x: 1).agg(func)
    # 对df按lambda函数分组，应用func函数进行聚合，返回结果DataFrame res

    expected = DataFrame([[1.0]], index=[1])
    # 创建期望的DataFrame对象expected，包含值为1.0的列，索引为1

    tm.assert_frame_equal(res, expected)
    # 使用测试框架tm进行结果的比较，确保res与expected相等


def test_groupby_agg_precision(any_real_numeric_dtype):
    if any_real_numeric_dtype in tm.ALL_INT_NUMPY_DTYPES:
        max_value = np.iinfo(any_real_numeric_dtype).max
    if any_real_numeric_dtype in tm.FLOAT_NUMPY_DTYPES:
        max_value = np.finfo(any_real_numeric_dtype).max
    if any_real_numeric_dtype in tm.FLOAT_EA_DTYPES:
        max_value = np.finfo(any_real_numeric_dtype.lower()).max
    if any_real_numeric_dtype in tm.ALL_INT_EA_DTYPES:
        max_value = np.iinfo(any_real_numeric_dtype.lower()).max

    # 根据传入参数any_real_numeric_dtype确定max_value的值

    df = DataFrame(
        {
            "key1": ["a"],
            "key2": ["b"],
            "key3": pd.array([max_value], dtype=any_real_numeric_dtype),
        }
    )
    # 创建DataFrame对象df，包含列"key1", "key2"和"key3"，key3列包含max_value的元素

    arrays = [["a"], ["b"]]
    index = MultiIndex.from_arrays(arrays, names=("key1", "key2"))
    # 创建MultiIndex对象index，包含索引标签为("key1", "key2")的数组

    expected = DataFrame(
        {"key3": pd.array([max_value], dtype=any_real_numeric_dtype)}, index=index
    )
    # 创建期望的DataFrame对象expected，包含key3列，索引为index

    result = df.groupby(["key1", "key2"]).agg(lambda x: x)
    # 对df按列"key1"和"key2"进行分组，应用lambda函数进行聚合，返回结果DataFrame result

    tm.assert_frame_equal(result, expected)
    # 使用测试框架tm进行结果的比较，确保result与expected相等


def test_groupby_aggregate_directory(reduction_func):
    # 该函数的代码不完整，无法提供注释
    pass
    # 如果 reduction_func 是 "corrwith" 或者 "nth"，则返回 None，不执行后续代码
    if reduction_func in ["corrwith", "nth"]:
        return None
    
    # 创建一个包含两行数据的 DataFrame 对象，第二行第二列是 NaN
    obj = DataFrame([[0, 1], [0, np.nan]])
    
    # 对 DataFrame 按照第一列进行分组，并使用 reduction_func 聚合数据，返回一个 Series 对象
    result_reduced_series = obj.groupby(0).agg(reduction_func)
    
    # 对 DataFrame 按照第一列进行分组，并对第二列应用 reduction_func 聚合数据，返回一个新的 DataFrame 对象
    result_reduced_frame = obj.groupby(0).agg({1: reduction_func})
    
    # 如果 reduction_func 是 "size" 或者 "ngroup"，则执行以下断言
    if reduction_func in ["size", "ngroup"]:
        # 断言 result_reduced_series 和 result_reduced_frame[1] 的内容相等，忽略名称检查
        tm.assert_series_equal(
            result_reduced_series, result_reduced_frame[1], check_names=False
        )
    else:
        # 断言 result_reduced_series 和 result_reduced_frame 的内容相等
        tm.assert_frame_equal(result_reduced_series, result_reduced_frame)
        # 断言 result_reduced_series 的数据类型和 result_reduced_frame 的数据类型相等
        tm.assert_series_equal(
            result_reduced_series.dtypes, result_reduced_frame.dtypes
        )
def test_group_mean_timedelta_nat():
    # GH43132
    data = Series(["1 day", "3 days", "NaT"], dtype="timedelta64[ns]")
    expected = Series(["2 days"], dtype="timedelta64[ns]", index=np.array([0]))

    result = data.groupby([0, 0, 0]).mean()
    # 对数据按照分组进行均值计算

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (  # no timezone
            ["2021-01-01T00:00", "NaT", "2021-01-01T02:00"],
            ["2021-01-01T01:00"],
        ),
        (  # timezone
            ["2021-01-01T00:00-0100", "NaT", "2021-01-01T02:00-0100"],
            ["2021-01-01T01:00-0100"],
        ),
    ],
)
def test_group_mean_datetime64_nat(input_data, expected_output):
    # GH43132
    data = to_datetime(Series(input_data))
    expected = to_datetime(Series(expected_output, index=np.array([0])))

    result = data.groupby([0, 0, 0]).mean()
    # 对数据按照分组进行均值计算

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, output", [("mean", [8 + 18j, 10 + 22j]), ("sum", [40 + 90j, 50 + 110j])]
)
def test_groupby_complex(func, output):
    # GH#43701
    data = Series(np.arange(20).reshape(10, 2).dot([1, 2j]))
    result = data.groupby(data.index % 2).agg(func)
    # 对复数数据按照分组运用指定函数（mean或sum）进行聚合计算

    expected = Series(output)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max", "var"])
def test_groupby_complex_raises(func):
    # GH#43701
    data = Series(np.arange(20).reshape(10, 2).dot([1, 2j]))
    msg = "No matching signature found"
    with pytest.raises(TypeError, match=msg):
        data.groupby(data.index % 2).agg(func)
    # 对复数数据按照分组尝试使用不支持的函数（min、max或var），应该引发类型错误异常


@pytest.mark.parametrize(
    "test, constant",
    [
        ([[20, "A"], [20, "B"], [10, "C"]], {0: [10, 20], 1: ["C", ["A", "B"]]}),
        ([[20, "A"], [20, "B"], [30, "C"]], {0: [20, 30], 1: [["A", "B"], "C"]}),
        ([["a", 1], ["a", 1], ["b", 2], ["b", 3]], {0: ["a", "b"], 1: [1, [2, 3]]}),
        pytest.param(
            [["a", 1], ["a", 2], ["b", 3], ["b", 3]],
            {0: ["a", "b"], 1: [[1, 2], 3]},
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_agg_of_mode_list(test, constant):
    # GH#25581
    df1 = DataFrame(test)
    result = df1.groupby(0).agg(Series.mode)
    # 使用聚合函数求解列的众数（mode），通常情况下只返回一个值，但在平局时可能返回一个列表。

    expected = DataFrame(constant)
    expected = expected.set_index(0)

    tm.assert_frame_equal(result, expected)


def test_dataframe_groupy_agg_list_like_func_with_args():
    # GH#50624
    df = DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    gb = df.groupby("y")

    def foo1(x, a=1, c=0):
        return x.sum() + a + c

    def foo2(x, b=2, c=0):
        return x.sum() + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        gb.agg([foo1, foo2], 3, b=3, c=4)
    # 尝试使用带有额外参数的自定义聚合函数列表（foo1和foo2），应该引发类型错误异常

    result = gb.agg([foo1, foo2], 3, c=4)
    # 创建一个预期的 DataFrame 对象，用于与测试结果进行比较
    expected = DataFrame(
        # 定义数据内容，包括三行两列的数值
        [[8, 8], [9, 9], [10, 10]],
        # 指定行索引为 ["a", "b", "c"]，并设置索引名为 "y"
        index=Index(["a", "b", "c"], name="y"),
        # 指定列索引为 [("x", "foo1"), ("x", "foo2")]，创建一个多级索引
        columns=MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),
    )
    
    # 使用测试框架中的断言方法，验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_series_groupy_agg_list_like_func_with_args():
    # 定义一个 Series 对象
    s = Series([1, 2, 3])
    # 根据 Series 对象进行分组
    sgb = s.groupby(s)

    # 定义两个函数 foo1 和 foo2，分别带有默认参数 a 和 b
    def foo1(x, a=1, c=0):
        return x.sum() + a + c

    def foo2(x, b=2, c=0):
        return x.sum() + b + c

    # 使用 pytest 检查是否会抛出 TypeError 异常
    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        sgb.agg([foo1, foo2], 3, b=3, c=4)

    # 对 sgb 进行聚合操作，传入参数 3 和 c=4
    result = sgb.agg([foo1, foo2], 3, c=4)
    # 创建一个 DataFrame 对象作为期望结果
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]], index=Index([1, 2, 3]), columns=["foo1", "foo2"]
    )
    # 使用 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_agg_groupings_selection():
    # 创建一个 DataFrame 对象
    df = DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "c": [5, 6, 7]})
    # 根据列 'a' 和 'b' 进行分组
    gb = df.groupby(["a", "b"])
    # 选择部分分组列 'b' 和 'c'
    selected_gb = gb[["b", "c"]]
    # 对选定的分组进行聚合操作，计算和
    result = selected_gb.agg(lambda x: x.sum())
    # 创建一个期望结果的 DataFrame 对象
    index = MultiIndex(
        levels=[[1, 2], [3, 4]], codes=[[0, 1], [0, 1]], names=["a", "b"]
    )
    expected = DataFrame({"b": [6, 4], "c": [11, 7]}, index=index)
    # 使用 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_with_as_index_false_subset_to_a_single_column():
    # 创建一个 DataFrame 对象
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    # 根据列 'a' 进行分组，不设置索引
    gb = df.groupby("a", as_index=False)["b"]
    # 对分组进行多个聚合操作，计算和和平均值
    result = gb.agg(["sum", "mean"])
    # 创建一个期望结果的 DataFrame 对象
    expected = DataFrame({"a": [1, 2], "sum": [7, 5], "mean": [3.5, 5.0]})
    # 使用 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_agg_with_as_index_false_with_list():
    # 创建一个 DataFrame 对象
    df = DataFrame({"a1": [0, 0, 1], "a2": [2, 3, 3], "b": [4, 5, 6]})
    # 根据列 'a1' 和 'a2' 进行分组，不设置索引
    gb = df.groupby(by=["a1", "a2"], as_index=False)
    # 对分组进行聚合操作，计算和
    result = gb.agg(["sum"])

    # 创建一个期望结果的 DataFrame 对象
    expected = DataFrame(
        data=[[0, 2, 4], [0, 3, 5], [1, 3, 6]],
        columns=MultiIndex.from_tuples([("a1", ""), ("a2", ""), ("b", "sum")]),
    )
    # 使用 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_groupby_agg_extension_timedelta_cumsum_with_named_aggregation():
    # 创建一个期望结果的 DataFrame 对象
    expected = DataFrame(
        {
            "td": {
                0: pd.Timedelta("0 days 01:00:00"),
                1: pd.Timedelta("0 days 01:15:00"),
                2: pd.Timedelta("0 days 01:15:00"),
            }
        }
    )
    # 创建一个 DataFrame 对象
    df = DataFrame(
        {
            "td": Series(
                ["0 days 01:00:00", "0 days 00:15:00", "0 days 01:15:00"],
                dtype="timedelta64[ns]",
            ),
            "grps": ["a", "a", "b"],
        }
    )
    # 根据列 'grps' 进行分组
    gb = df.groupby("grps")
    # 对分组进行聚合操作，计算累积和
    result = gb.agg(td=("td", "cumsum"))
    # 使用 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_empty_group():
    # 定义一个函数 func，用于处理空分组
    def func(x):
        if len(x) == 0:
            raise ValueError("length must not be 0")
        return len(x)

    # 创建一个 DataFrame 对象
    df = DataFrame(
        {"A": pd.Categorical(["a", "a"], categories=["a", "b", "c"]), "B": [1, 1]}
    )
    # 定义一个错误消息
    msg = "length must not be 0"
    # 使用 pytest 模块来测试代码中的异常情况，验证是否抛出 ValueError，并且匹配预期的错误消息（msg）
    with pytest.raises(ValueError, match=msg):
        # 对 DataFrame df 根据列 "A" 进行分组，observed=False 表示忽略不存在的分组键
        # 对分组后的结果应用聚合函数 func
        df.groupby("A", observed=False).agg(func)
# 定义一个测试函数，用于验证在多列中存在重复列名时的分组聚合情况
def test_groupby_aggregation_duplicate_columns_single_dict_value():
    # 标识问题的GitHub issue编号
    # 创建一个包含数据的DataFrame，其中包括多个重复的列名“c”
    df = DataFrame(
        [[1, 2, 3, 4], [1, 3, 4, 5], [2, 4, 5, 6]],
        columns=["a", "b", "c", "c"],
    )
    # 根据列“a”进行分组
    gb = df.groupby("a")
    # 对分组后的结果进行聚合，计算列“c”的和
    result = gb.agg({"c": "sum"})

    # 创建预期的DataFrame，期望结果为两行，包含两列名为“c”的列
    expected = DataFrame(
        [[7, 9], [5, 6]], columns=["c", "c"], index=Index([1, 2], name="a")
    )
    # 使用测试工具验证结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_duplicate_columns_multiple_dict_values():
    # GH#55041
    # 创建一个包含数据的DataFrame，其中包括多个重复的列名“c”
    df = DataFrame(
        [[1, 2, 3, 4], [1, 3, 4, 5], [2, 4, 5, 6]],
        columns=["a", "b", "c", "c"],
    )
    # 根据列“a”进行分组
    gb = df.groupby("a")
    # 对分组后的结果进行多种聚合，包括对列“c”进行求和、最小值、最大值等操作
    result = gb.agg({"c": ["sum", "min", "max", "min"]})

    # 创建预期的DataFrame，期望结果为两行，包含MultiIndex的列结构
    expected = DataFrame(
        [[7, 3, 4, 3, 9, 4, 5, 4], [5, 5, 5, 5, 6, 6, 6, 6]],
        columns=MultiIndex(
            levels=[["c"], ["sum", "min", "max"]],
            codes=[[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 1, 0, 1, 2, 1]],
        ),
        index=Index([1, 2], name="a"),
    )
    # 使用测试工具验证结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_duplicate_columns_some_empty_result():
    # GH#55041
    # 创建一个包含数据的DataFrame，其中包含多个重复的列名“b”和“c”
    df = DataFrame(
        [
            [1, 9843, 43, 54, 7867],
            [2, 940, 9, -34, 44],
            [1, -34, -546, -549358, 0],
            [2, 244, -33, -100, 44],
        ],
        columns=["a", "b", "b", "c", "c"],
    )
    # 根据列“a”进行分组
    gb = df.groupby("a")
    # 对分组后的结果进行聚合，对列“b”使用空列表（无操作），对列“c”计算方差
    result = gb.agg({"b": [], "c": ["var"]})

    # 创建预期的DataFrame，期望结果为两行，包含MultiIndex的列结构
    expected = DataFrame(
        [[1.509268e11, 30944844.5], [2.178000e03, 0.0]],
        columns=MultiIndex(levels=[["c"], ["var"]], codes=[[0, 0], [0, 0]]),
        index=Index([1, 2], name="a"),
    )
    # 使用测试工具验证结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_multi_index_duplicate_columns():
    # GH#55041
    # 创建一个包含数据的DataFrame，其中包含多级索引和重复的列名
    df = DataFrame(
        [
            [1, -9843, 43, 54, 7867],
            [2, 940, 9, -34, 44],
            [1, -34, 546, -549358, 0],
            [2, 244, -33, -100, 44],
        ],
        columns=MultiIndex(
            levels=[["level1.1", "level1.2"], ["level2.1", "level2.2"]],
            codes=[[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]],
        ),
        index=MultiIndex(
            levels=[["level1.1", "level1.2"], ["level2.1", "level2.2"]],
            codes=[[0, 0, 0, 1], [0, 1, 1, 0]],
        ),
    )
    # 根据第一级索引进行分组
    gb = df.groupby(level=0)
    # 对分组后的结果进行聚合，计算特定列的最小值
    result = gb.agg({("level1.1", "level2.2"): "min"})

    # 创建预期的DataFrame，期望结果为两行，包含MultiIndex的列结构
    expected = DataFrame(
        [[-9843, 9], [244, -33]],
        columns=MultiIndex(levels=[["level1.1"], ["level2.2"]], codes=[[0, 0], [0, 0]]),
        index=Index(["level1.1", "level1.2"]),
    )
    # 使用测试工具验证结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_func_list_multi_index_duplicate_columns():
    # GH#55041
    # 创建一个 DataFrame 对象，包含特定的数据和列索引
    df = DataFrame(
        [
            [1, -9843, 43, 54, 7867],
            [2, 940, 9, -34, 44],
            [1, -34, 546, -549358, 0],
            [2, 244, -33, -100, 44],
        ],
        columns=MultiIndex(
            levels=[["level1.1", "level1.2"], ["level2.1", "level2.2"]],
            codes=[[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]],
        ),
        index=MultiIndex(
            levels=[["level1.1", "level1.2"], ["level2.1", "level2.2"]],
            codes=[[0, 0, 0, 1], [0, 1, 1, 0]],
        ),
    )
    # 根据 DataFrame 的第一级索引进行分组
    gb = df.groupby(level=0)
    # 对分组后的数据进行聚合操作，计算指定索引位置的列的最小值和最大值
    result = gb.agg({("level1.1", "level2.2"): ["min", "max"]})

    # 创建一个期望的 DataFrame 对象，包含预期的数据和列索引
    expected = DataFrame(
        [[-9843, 940, 9, 546], [244, 244, -33, -33]],
        columns=MultiIndex(
            levels=[["level1.1"], ["level2.2"], ["min", "max"]],
            codes=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]],
        ),
        index=Index(["level1.1", "level1.2"]),
    )
    # 使用测试框架中的函数验证两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
```