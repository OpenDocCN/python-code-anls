# `D:\src\scipysrc\pandas\pandas\tests\groupby\aggregate\test_other.py`

```
"""
test all other .agg behavior
"""

import datetime as dt  # 导入 datetime 模块，简写为 dt
from functools import partial  # 导入 functools 模块中的 partial 函数

import numpy as np  # 导入 NumPy 库，简写为 np
import pytest  # 导入 pytest 测试框架

from pandas.errors import SpecificationError  # 从 pandas.errors 中导入 SpecificationError 异常类

import pandas as pd  # 导入 pandas 库，简写为 pd
from pandas import (  # 从 pandas 中导入多个对象
    DataFrame,  # 数据框对象
    Index,  # 索引对象
    MultiIndex,  # 多重索引对象
    PeriodIndex,  # 时间段索引对象
    Series,  # 系列对象
    date_range,  # 日期范围生成函数
    period_range,  # 时间段范围生成函数
)
import pandas._testing as tm  # 导入 pandas._testing 模块，简写为 tm

from pandas.io.formats.printing import pprint_thing  # 从 pandas.io.formats.printing 中导入 pprint_thing 函数


def test_agg_partial_failure_raises():
    # GH#43741
    # 测试聚合操作中的部分失败是否会引发异常

    df = DataFrame(  # 创建一个 DataFrame 对象
        {
            "data1": np.random.default_rng(2).standard_normal(5),  # 创建包含随机数列的列 'data1'
            "data2": np.random.default_rng(2).standard_normal(5),  # 创建包含随机数列的列 'data2'
            "key1": ["a", "a", "b", "b", "a"],  # 创建包含字符串的列 'key1'
            "key2": ["one", "two", "one", "two", "one"],  # 创建包含字符串的列 'key2'
        }
    )
    grouped = df.groupby("key1")  # 按 'key1' 列进行分组

    def peak_to_peak(arr):
        # 计算数组的峰值到峰值距离
        return arr.max() - arr.min()

    with pytest.raises(TypeError, match="unsupported operand type"):  # 检查是否会抛出 TypeError 异常
        grouped.agg([peak_to_peak])  # 对分组数据应用聚合函数 peak_to_peak

    with pytest.raises(TypeError, match="unsupported operand type"):  # 再次检查是否会抛出 TypeError 异常
        grouped.agg(peak_to_peak)  # 对分组数据应用聚合函数 peak_to_peak


def test_agg_datetimes_mixed():
    data = [[1, "2012-01-01", 1.0], [2, "2012-01-02", 2.0], [3, None, 3.0]]

    df1 = DataFrame(  # 创建第一个 DataFrame 对象
        {
            "key": [x[0] for x in data],  # 创建 'key' 列，从 data 中获取第一列
            "date": [x[1] for x in data],  # 创建 'date' 列，从 data 中获取第二列
            "value": [x[2] for x in data],  # 创建 'value' 列，从 data 中获取第三列
        }
    )

    data = [  # 对 data 中的每行数据进行处理
        [
            row[0],
            (dt.datetime.strptime(row[1], "%Y-%m-%d").date() if row[1] else None),  # 将日期字符串转换为 datetime.date 对象，如果为 None 则保持为 None
            row[2],
        ]
        for row in data
    ]

    df2 = DataFrame(  # 创建第二个 DataFrame 对象
        {
            "key": [x[0] for x in data],  # 创建 'key' 列，从处理后的数据中获取第一列
            "date": [x[1] for x in data],  # 创建 'date' 列，从处理后的数据中获取第二列
            "value": [x[2] for x in data],  # 创建 'value' 列，从处理后的数据中获取第三列
        }
    )

    df1["weights"] = df1["value"] / df1["value"].sum()  # 计算 'weights' 列，表示每个值在总和中的权重
    gb1 = df1.groupby("date").aggregate("sum")  # 按 'date' 列进行分组并对数值列进行求和聚合

    df2["weights"] = df1["value"] / df1["value"].sum()  # 计算 'weights' 列，与第一个 DataFrame 相同的处理
    gb2 = df2.groupby("date").aggregate("sum")  # 按 'date' 列进行分组并对数值列进行求和聚合

    assert len(gb1) == len(gb2)  # 断言两个分组后的 DataFrame 长度相等


def test_agg_period_index():
    prng = period_range("2012-1-1", freq="M", periods=3)  # 创建一个时间段索引对象
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)), index=prng)  # 创建一个随机数值的 DataFrame，并使用时间段索引
    rs = df.groupby(level=0).sum()  # 按索引的第一层级进行分组并对数值列进行求和
    assert isinstance(rs.index, PeriodIndex)  # 断言结果的索引类型为 PeriodIndex

    # GH 3579
    index = period_range(start="1999-01", periods=5, freq="M")  # 创建另一个时间段索引对象
    s1 = Series(np.random.default_rng(2).random(len(index)), index=index)  # 创建随机数值的 Series 对象，使用指定的时间段索引
    s2 = Series(np.random.default_rng(2).random(len(index)), index=index)  # 创建随机数值的 Series 对象，使用相同的时间段索引
    df = DataFrame.from_dict({"s1": s1, "s2": s2})  # 从字典创建 DataFrame 对象，包含两个列 's1' 和 's2'
    grouped = df.groupby(df.index.month)  # 按月份对索引进行分组
    list(grouped)  # 将分组对象转换为列表


def test_agg_dict_parameter_cast_result_dtypes():
    # GH 12821
    # 测试聚合操作中字典参数是否正确转换结果的数据类型

    df = DataFrame(  # 创建一个 DataFrame 对象
        {
            "class": ["A", "A", "B", "B", "C", "C", "D", "D"],  # 创建 'class' 列，包含多个类别
            "time": date_range("1/1/2011", periods=8, freq="h"),  # 创建 'time' 列，包含时间序列
        }
    )
    df.loc[[0, 1, 2, 5], "time"] = None  # 将 'time' 列中的部分值设置为 None

    # test for `first` function
    exp = df.loc[[0, 3, 4, 6]].set_index("class")  # 创建期望的结果 DataFrame，按 'class' 列设置索引
    grouped = df.groupby("class")  # 按 'class' 列进行分组
    tm.assert_frame_equal(grouped.first(), exp)  # 断言分组后应用 'first' 函数得到的结果与期望的结果相等
    # 检验分组数据的第一个元素是否与期望结果相等
    tm.assert_frame_equal(grouped.agg("first"), exp)
    # 检验按照指定函数（此处为 "first"）聚合后的数据是否与期望结果相等
    tm.assert_frame_equal(grouped.agg({"time": "first"}), exp)
    # 检验分组后的时间序列的第一个元素是否与期望结果相等
    tm.assert_series_equal(grouped.time.first(), exp["time"])
    # 检验按照指定函数（此处为 "first"）聚合后的时间序列的第一个元素是否与期望结果相等
    tm.assert_series_equal(grouped.time.agg("first"), exp["time"])

    # 对 `last` 函数进行测试
    exp = df.loc[[0, 3, 4, 7]].set_index("class")
    grouped = df.groupby("class")
    # 检验分组数据的最后一个元素是否与期望结果相等
    tm.assert_frame_equal(grouped.last(), exp)
    # 检验按照指定函数（此处为 "last"）聚合后的数据是否与期望结果相等
    tm.assert_frame_equal(grouped.agg("last"), exp)
    # 检验按照指定函数（{"time": "last"}）聚合后的数据是否与期望结果相等
    tm.assert_frame_equal(grouped.agg({"time": "last"}), exp)
    # 检验分组后的时间序列的最后一个元素是否与期望结果相等
    tm.assert_series_equal(grouped.time.last(), exp["time"])
    # 检验按照指定函数（此处为 "last"）聚合后的时间序列的最后一个元素是否与期望结果相等
    tm.assert_series_equal(grouped.time.agg("last"), exp["time"])

    # 计数
    exp = Series([2, 2, 2, 2], index=Index(list("ABCD"), name="class"), name="time")
    # 检验分组后时间序列的元素数量是否与期望结果相等
    tm.assert_series_equal(grouped.time.agg(len), exp)
    # 检验分组后时间序列的大小是否与期望结果相等
    tm.assert_series_equal(grouped.time.size(), exp)

    exp = Series([0, 1, 1, 2], index=Index(list("ABCD"), name="class"), name="time")
    # 检验分组后时间序列的非空元素数量（即非NaN值的数量）是否与期望结果相等
    tm.assert_series_equal(grouped.time.count(), exp)
def test_agg_cast_results_dtypes():
    # similar to GH12821
    # xref #11444
    # 创建包含12个datetime对象的列表，每个对象代表2015年的一个月份的第一天
    u = [dt.datetime(2015, x + 1, 1) for x in range(12)]
    # 创建一个包含字符的列表，用于DataFrame的构建
    v = list("aaabbbbbbccd")
    # 根据提供的字典创建DataFrame对象df，包含"X"和"Y"两列
    df = DataFrame({"X": v, "Y": u})

    # 对DataFrame对象df按列"X"进行分组，计算"Y"列每组的长度
    result = df.groupby("X")["Y"].agg(len)
    # 对DataFrame对象df按列"X"进行分组，计算"Y"列每组的计数
    expected = df.groupby("X")["Y"].count()
    # 使用测试框架tm检查结果是否相等
    tm.assert_series_equal(result, expected)


def test_aggregate_float64_no_int64():
    # see gh-11199
    # 根据提供的字典创建DataFrame对象df，包含"a", "b", "c"三列
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 4, 5], "c": [1, 2, 3, 4, 5]})

    # 根据提供的字典创建期望的DataFrame对象expected，包含"a"列和对应的索引
    expected = DataFrame({"a": [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = "b"

    # 对DataFrame对象df按列"b"进行分组，计算"a"列的均值
    result = df.groupby("b")[["a"]].mean()
    # 使用测试框架tm检查结果是否相等
    tm.assert_frame_equal(result, expected)

    # 根据提供的字典创建期望的DataFrame对象expected，包含"a"列、"c"列和对应的索引
    expected = DataFrame({"a": [1, 2.5, 4, 5], "c": [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = "b"

    # 对DataFrame对象df按列"b"进行分组，计算"a"列和"c"列的均值
    result = df.groupby("b")[["a", "c"]].mean()
    # 使用测试框架tm检查结果是否相等
    tm.assert_frame_equal(result, expected)


def test_aggregate_api_consistency():
    # GH 9052
    # 确保通过字典进行的聚合操作一致
    # 根据提供的字典创建DataFrame对象df，包含"A", "B", "C", "D"四列
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    # 对DataFrame对象df按["A", "B"]进行分组
    grouped = df.groupby(["A", "B"])
    # 计算分组后"C"列的均值和总和
    c_mean = grouped["C"].mean()
    c_sum = grouped["C"].sum()
    # 计算分组后"D"列的均值和总和
    d_mean = grouped["D"].mean()
    d_sum = grouped["D"].sum()

    # 对"D"列按"sum"和"mean"进行聚合操作，期望得到的DataFrame对象expected
    result = grouped["D"].agg(["sum", "mean"])
    expected = pd.concat([d_sum, d_mean], axis=1)
    expected.columns = ["sum", "mean"]
    # 使用测试框架tm检查结果是否相等
    tm.assert_frame_equal(result, expected, check_like=True)

    # 对整个DataFrame对象按["A", "B"]进行聚合操作，期望得到的DataFrame对象expected
    result = grouped.agg(["sum", "mean"])
    expected = pd.concat([c_sum, c_mean, d_sum, d_mean], axis=1)
    expected.columns = MultiIndex.from_product([["C", "D"], ["sum", "mean"]])
    # 使用测试框架tm检查结果是否相等
    tm.assert_frame_equal(result, expected, check_like=True)

    # 对["D", "C"]列按["sum", "mean"]进行聚合操作，期望得到的DataFrame对象expected
    result = grouped[["D", "C"]].agg(["sum", "mean"])
    expected = pd.concat([d_sum, d_mean, c_sum, c_mean], axis=1)
    expected.columns = MultiIndex.from_product([["D", "C"], ["sum", "mean"]])
    # 使用测试框架tm检查结果是否相等
    tm.assert_frame_equal(result, expected, check_like=True)

    # 对"C"列和"D"列按指定字典{"C": "mean", "D": "sum"}进行聚合操作，期望得到的DataFrame对象expected
    result = grouped.agg({"C": "mean", "D": "sum"})
    expected = pd.concat([d_sum, c_mean], axis=1)
    # 使用测试框架tm检查结果是否相等
    tm.assert_frame_equal(result, expected, check_like=True)

    # 对"C"列和"D"列按指定字典{"C": ["mean", "sum"], "D": ["mean", "sum"]}进行聚合操作，期望得到的DataFrame对象expected
    result = grouped.agg({"C": ["mean", "sum"], "D": ["mean", "sum"]})
    expected = pd.concat([c_mean, c_sum, d_mean, d_sum], axis=1)
    expected.columns = MultiIndex.from_product([["C", "D"], ["mean", "sum"]])

    # 使用pytest断言检查是否会引发预期的KeyError异常
    msg = r"Label\(s\) \['r', 'r2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        grouped[["D", "C"]].agg({"r": "sum", "r2": "mean"})


def test_agg_dict_renaming_deprecation():
    # 15931
    # 根据提供的字典创建DataFrame对象df，包含"A", "B", "C"三列
    df = DataFrame({"A": [1, 1, 1, 2, 2], "B": range(5), "C": range(5)})

    # 预期的错误消息，指示嵌套的重命名不受支持
    msg = r"nested renamer is not supported"
    # 使用 pytest 模块测试是否抛出 SpecificationError 异常，并验证异常消息是否匹配给定的 msg 变量
    with pytest.raises(SpecificationError, match=msg):
        # 对 DataFrame 按列 "A" 进行分组，然后对 "B" 列应用 ["sum", "max"] 聚合函数，
        # 对 "C" 列应用 ["count", "min"] 聚合函数。此处测试是否抛出异常。
        df.groupby("A").agg(
            {"B": {"foo": ["sum", "max"]}, "C": {"bar": ["count", "min"]}}
        )
    
    # 设置 msg 变量，用于验证是否抛出 KeyError 异常，并检查异常消息是否匹配给定的 msg
    msg = r"Label\(s\) \['ma'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        # 对 DataFrame 按列 "A" 进行分组，然后只选择 "B" 和 "C" 列，并尝试对 "ma" 列应用 "max" 聚合函数。
        # 此处测试是否抛出异常。
        df.groupby("A")[["B", "C"]].agg({"ma": "max"})
    
    # 设置 msg 变量，用于验证是否抛出 SpecificationError 异常，并检查异常消息是否匹配给定的 msg
    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        # 对 DataFrame 按列 "A" 进行分组，然后尝试对 "B" 列应用 {"foo": "count"} 的复合形式聚合。
        # 此处测试是否抛出异常。
        df.groupby("A").B.agg({"foo": "count"})
# 定义一个测试函数，用于验证聚合操作的兼容性
def test_agg_compat():
    # 创建一个 DataFrame 包含四列数据
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,  # 生成随机数据并加上1.0
            "D": np.arange(8),  # 创建一个从0到7的整数序列
        }
    )

    # 对 DataFrame 进行按列 A 和 B 进行分组
    g = df.groupby(["A", "B"])

    # 定义一个错误消息，用于匹配 pytest 的异常输出
    msg = r"nested renamer is not supported"

    # 使用 pytest 检查 g["D"] 的聚合操作是否引发了 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"C": ["sum", "std"]})

    # 再次使用 pytest 检查 g["D"] 的另一种聚合操作是否引发了 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"C": "sum", "D": "std"})


# 定义一个测试函数，用于验证不支持嵌套字典的聚合操作
def test_agg_nested_dicts():
    # 创建一个 DataFrame 包含四列数据，与上一个函数类似
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    # 对 DataFrame 进行按列 A 和 B 进行分组
    g = df.groupby(["A", "B"])

    # 定义一个错误消息，用于匹配 pytest 的异常输出
    msg = r"nested renamer is not supported"

    # 使用 pytest 检查 g 的聚合操作中包含嵌套字典的情况是否引发了 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        g.aggregate({"r1": {"C": ["mean", "sum"]}, "r2": {"D": ["mean", "sum"]}})

    # 再次使用 pytest 检查 g 的聚合操作中包含嵌套字典的另一种情况是否引发了 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        g.agg({"C": {"ra": ["mean", "std"]}, "D": {"rb": ["mean", "std"]}})

    # 对 g["D"] 的聚合操作中，如果结果列名与原始列名相同，验证是否引发了 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"result1": np.sum, "result2": np.mean})

    # 再次对 g["D"] 的聚合操作中，如果结果列名与原始列名相同，验证是否引发了 SpecificationError 异常，并匹配错误消息
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"D": np.sum, "result2": np.mean})


# 定义一个测试函数，用于验证 Series 对象的多键聚合操作
def test_series_agg_multikey():
    # 创建一个时间序列 Series，从 '2020-01-01' 开始，包含10个元素
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    # 对 Series 按照年份和月份进行分组
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])

    # 对分组后的结果进行求和聚合操作
    result = grouped.agg("sum")
    # 计算预期的分组求和结果
    expected = grouped.sum()
    # 使用 pytest 的工具函数验证两个 Series 是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于验证 Series 对象的多键聚合操作，使用纯 Python 实现
def test_series_agg_multi_pure_python():
    # 创建一个 DataFrame 对象 `data`，包含多列数据
    data = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),  # 创建一个标准正态分布的随机数列，长度为11
            "E": np.random.default_rng(2).standard_normal(11),  # 创建一个标准正态分布的随机数列，长度为11
            "F": np.random.default_rng(2).standard_normal(11),  # 创建一个标准正态分布的随机数列，长度为11
        }
    )
    
    # 定义一个名为 `bad` 的函数，对传入的参数 x 执行断言，确保其 values 属性的基础对象不为空
    def bad(x):
        assert len(x.values.base) > 0
        return "foo"
    
    # 使用 `data` 对象按列 "A" 和 "B" 分组，然后应用函数 `bad` 对每组进行聚合操作，得到结果 `result`
    result = data.groupby(["A", "B"]).agg(bad)
    
    # 创建预期的 DataFrame `expected`，使用 `data` 对象按列 "A" 和 "B" 分组，然后应用匿名函数对每组进行聚合，结果始终为 "foo"
    expected = data.groupby(["A", "B"]).agg(lambda x: "foo")
    
    # 使用 `tm.assert_frame_equal` 断言函数比较 `result` 和 `expected`，确保它们相等
    tm.assert_frame_equal(result, expected)
def test_agg_consistency():
    # 定义内部函数 P1，计算输入数据的第1百分位数
    def P1(a):
        return np.percentile(a.dropna(), q=1)

    # 创建 DataFrame df，包含三列数据：col1, col2, date
    df = DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": [10, 25, 26, 31],
            "date": [
                dt.date(2013, 2, 10),
                dt.date(2013, 2, 10),
                dt.date(2013, 2, 11),
                dt.date(2013, 2, 11),
            ],
        }
    )

    # 对 df 按 'date' 列进行分组
    g = df.groupby("date")

    # 计算分组后每组的 P1 函数的聚合结果
    expected = g.agg([P1])
    
    # 将多级索引的列转换为单级索引
    expected.columns = expected.columns.levels[0]

    # 计算分组后每组的 P1 函数的聚合结果（另一种写法）
    result = g.agg(P1)
    
    # 使用测试工具检查两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_agg_callables():
    # 创建 DataFrame df，包含两列数据：'foo' 和 'bar'
    df = DataFrame({"foo": [1, 2], "bar": [3, 4]}).astype(np.int64)

    # 定义一个可调用的类 fn_class
    class fn_class:
        def __call__(self, x):
            return sum(x)

    # 等效的可调用对象列表
    equiv_callables = [
        sum,
        np.sum,
        lambda x: sum(x),
        lambda x: x.sum(),
        partial(sum),
        fn_class(),
    ]

    # 以 'foo' 列分组，对 'bar' 列应用 sum 函数进行聚合
    expected = df.groupby("foo").agg("sum")
    
    # 遍历等效的可调用对象列表，对 'bar' 列应用每个可调用对象进行聚合
    for ecall in equiv_callables:
        result = df.groupby("foo").agg(ecall)
        
        # 使用测试工具检查结果 DataFrame 是否与预期相等
        tm.assert_frame_equal(result, expected)


def test_agg_over_numpy_arrays():
    # 创建包含数组的 DataFrame df，包含 'category' 和 'arraydata' 两列
    df = DataFrame(
        [
            [1, np.array([10, 20, 30])],
            [1, np.array([40, 50, 60])],
            [2, np.array([20, 30, 40])],
        ],
        columns=["category", "arraydata"],
    )
    
    # 对 'category' 列进行分组
    gb = df.groupby("category")

    # 期望的聚合结果，对 'arraydata' 列应用 sum 函数
    expected_data = [[np.array([50, 70, 90])], [np.array([20, 30, 40])]]
    expected_index = Index([1, 2], name="category")
    expected_column = ["arraydata"]
    expected = DataFrame(expected_data, index=expected_index, columns=expected_column)

    # 使用 numeric_only=False 对每组应用 sum 函数进行聚合
    alt = gb.sum(numeric_only=False)
    
    # 使用测试工具检查结果 DataFrame 是否与预期相等
    tm.assert_frame_equal(alt, expected)

    # 对每组应用 sum 函数进行聚合（另一种写法）
    result = gb.agg("sum", numeric_only=False)
    
    # 使用测试工具检查结果 DataFrame 是否与预期相等
    tm.assert_frame_equal(result, expected)

    # FIXME: 原始版本的测试调用了 `gb.agg(sum)`，
    # 如果传入 `numeric_only=False` 则会引发 TypeError


@pytest.mark.parametrize("as_period", [True, False])
def test_agg_tzaware_non_datetime_result(as_period):
    # 讨论在处理 tzaware 值时的问题，涉及不保持数据类型的函数
    dti = date_range("2012-01-01", periods=4, tz="UTC")
    if as_period:
        dti = dti.tz_localize(None).to_period("D")

    # 创建 DataFrame df，包含两列 'a' 和 'b'
    df = DataFrame({"a": [0, 0, 1, 1], "b": dti})
    
    # 对 'a' 列进行分组
    gb = df.groupby("a")

    # 对每组 'b' 列应用 lambda 函数，保持数据类型
    result = gb["b"].agg(lambda x: x.iloc[0])
    expected = Series(dti[::2], name="b")
    expected.index.name = "a"
    
    # 使用测试工具检查结果 Series 是否与预期相等
    tm.assert_series_equal(result, expected)

    # 对每组 'b' 列应用 lambda 函数，不保持数据类型
    result = gb["b"].agg(lambda x: x.iloc[0].year)
    expected = Series([2012, 2012], name="b")
    expected.index.name = "a"
    
    # 使用测试工具检查结果 Series 是否与预期相等
    tm.assert_series_equal(result, expected)

    # 对每组 'b' 列应用 lambda 函数，不保持数据类型
    result = gb["b"].agg(lambda x: x.iloc[-1] - x.iloc[0])
    expected = Series([pd.Timedelta(days=1), pd.Timedelta(days=1)], name="b")
    
    # 使用测试工具检查结果 Series 是否与预期相等
    tm.assert_series_equal(result, expected)
    # 设置索引名称为 "a"
    expected.index.name = "a"
    
    # 如果指定了 as_period 参数为 True，则重新定义 expected 为包含两个偏移量对象的 Series，并设置其名称为 "b"
    if as_period:
        expected = Series([pd.offsets.Day(1), pd.offsets.Day(1)], name="b")
        expected.index.name = "a"
    
    # 使用测试框架中的函数检查 result 和 expected Series 是否相等
    tm.assert_series_equal(result, expected)
def test_agg_timezone_round_trip():
    # GH 15426
    # 创建一个带有时区信息的时间戳对象
    ts = pd.Timestamp("2016-01-01 12:00:00", tz="US/Pacific")
    # 创建一个包含时间序列的数据框
    df = DataFrame({"a": 1, "b": [ts + dt.timedelta(minutes=nn) for nn in range(10)]})

    # 使用groupby函数按列'a'分组，计算每组'b'列的最小值，并取第一个结果
    result1 = df.groupby("a")["b"].agg("min").iloc[0]
    # 使用groupby函数按列'a'分组，对每组'b'列应用自定义函数np.min，并取第一个结果
    result2 = df.groupby("a")["b"].agg(lambda x: np.min(x)).iloc[0]
    # 使用groupby函数按列'a'分组，计算每组'b'列的最小值，并取第一个结果
    result3 = df.groupby("a")["b"].min().iloc[0]

    # 断言结果是否等于初始创建的时间戳对象ts
    assert result1 == ts
    assert result2 == ts
    assert result3 == ts

    # 创建一个日期时间戳列表，带有时区信息'US/Pacific'
    dates = [
        pd.Timestamp(f"2016-01-0{i:d} 12:00:00", tz="US/Pacific") for i in range(1, 5)
    ]
    # 创建一个包含日期时间戳列'A'和'B'的数据框
    df = DataFrame({"A": ["a", "b"] * 2, "B": dates})
    # 对数据框按列'A'进行分组
    grouped = df.groupby("A")

    # 选取数据框中第一行'B'列的时间戳对象ts
    ts = df["B"].iloc[0]
    # 断言时间戳对象ts是否等于分组后第一行'B'列的时间戳对象
    assert ts == grouped.nth(0)["B"].iloc[0]
    # 断言时间戳对象ts是否等于分组后第一行'B'列的时间戳对象
    assert ts == grouped.head(1)["B"].iloc[0]
    # 断言时间戳对象ts是否等于分组后第一行'B'列的时间戳对象
    assert ts == grouped.first()["B"].iloc[0]

    # GH#27110 应用iloc应该返回一个数据框，发出警告信息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 断言时间戳对象ts是否等于应用lambda函数后的结果
        assert ts == grouped.apply(lambda x: x.iloc[0]).iloc[0, 1]

    # 选取数据框中第三行'B'列的时间戳对象ts
    ts = df["B"].iloc[2]
    # 断言时间戳对象ts是否等于分组后最后一行'B'列的时间戳对象
    assert ts == grouped.last()["B"].iloc[0]

    # GH#27110 应用iloc应该返回一个数据框，发出警告信息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 断言时间戳对象ts是否等于应用lambda函数后的结果
        assert ts == grouped.apply(lambda x: x.iloc[-1]).iloc[0, 1]


def test_sum_uint64_overflow():
    # see gh-14758
    # 将数据框转换为uint64类型并避免溢出
    df = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=object)
    df = df + 9223372036854775807

    # 创建一个索引对象，包含uint64类型的数据
    index = Index(
        [9223372036854775808, 9223372036854775810, 9223372036854775812], dtype=np.uint64
    )
    # 创建一个期望结果的数据框，包含对应的数据类型为object
    expected = DataFrame(
        {1: [9223372036854775809, 9223372036854775811, 9223372036854775813]},
        index=index,
        dtype=object,
    )

    # 设置期望数据框的索引名称
    expected.index.name = 0
    # 对数据框进行分组并求和，包括非数值列
    result = df.groupby(0).sum(numeric_only=False)
    # 断言分组求和后的结果与期望结果是否相等
    tm.assert_frame_equal(result, expected)

    # 对数据框进行分组并求和，只保留数值列
    result2 = df.groupby(0).sum(numeric_only=True)
    # 期望结果保留空列
    expected2 = expected[[]]
    # 断言分组求和后的结果与期望结果是否相等
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize(
    "structure, cast_as",
    [
        (tuple, tuple),
        (list, list),
        (lambda x: tuple(x), tuple),
        (lambda x: list(x), list),
    ],
)
def test_agg_structs_dataframe(structure, cast_as):
    # 创建一个包含三列'A', 'B', 'C'的数据框
    df = DataFrame(
        {"A": [1, 1, 1, 3, 3, 3], "B": [1, 1, 1, 4, 4, 4], "C": [1, 1, 1, 3, 4, 4]}
    )

    # 对数据框按列'A', 'B'进行分组，应用传入的structure函数
    result = df.groupby(["A", "B"]).aggregate(structure)
    # 创建一个期望结果的数据框，包含对应的数据类型为cast_as
    expected = DataFrame(
        {"C": {(1, 1): cast_as([1, 1, 1]), (3, 4): cast_as([3, 4, 4])}}
    )
    # 设置期望结果数据框的索引名称
    expected.index.names = ["A", "B"]
    # 断言分组聚合后的结果与期望结果是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "structure, cast_as",
    [
        (tuple, tuple),
        (list, list),
        (lambda x: tuple(x), tuple),
        (lambda x: list(x), list),
    ],
)
# 定义一个测试函数，用于测试在DataFrame上应用聚合操作时的行为
def test_agg_structs_series(structure, cast_as):
    # 创建一个包含三列的DataFrame，每列有多个重复值
    df = DataFrame(
        {"A": [1, 1, 1, 3, 3, 3], "B": [1, 1, 1, 4, 4, 4], "C": [1, 1, 1, 3, 4, 4]}
    )

    # 根据'A'列进行分组，对'C'列应用给定的聚合函数（structure），得到结果Series
    result = df.groupby("A")["C"].aggregate(structure)

    # 创建预期的Series，其值是cast_as函数应用到特定列表的结果，使用索引'1'和'3'
    expected = Series([cast_as([1, 1, 1]), cast_as([3, 4, 4])], index=[1, 3], name="C")
    # 设置预期Series的索引名称为'A'
    expected.index.name = "A"

    # 使用测试框架检查结果Series和预期Series是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试在DataFrame上对分组后的类别数据应用np.nansum聚合函数的行为
def test_agg_category_nansum(observed):
    # 定义类别列表
    categories = ["a", "b", "c"]

    # 创建一个包含'A'列（带类别信息）和'B'列的DataFrame
    df = DataFrame(
        {"A": pd.Categorical(["a", "a", "b"], categories=categories), "B": [1, 2, 3]}
    )

    # 根据'A'列进行分组，对'B'列应用np.nansum聚合函数，得到结果Series
    result = df.groupby("A", observed=observed).B.agg(np.nansum)

    # 创建预期的Series，其值是按类别索引计算的np.nansum结果
    expected = Series(
        [3, 3, 0],
        index=pd.CategoricalIndex(["a", "b", "c"], categories=categories, name="A"),
        name="B",
    )
    
    # 如果observed参数为True，则从预期Series中剔除值为0的条目
    if observed:
        expected = expected[expected != 0]

    # 使用测试框架检查结果Series和预期Series是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，测试在DataFrame上对分组后的列应用lambda函数以生成列表的行为
def test_agg_list_like_func():
    # 创建一个包含两列的DataFrame，'A'列和'B'列各包含三个字符串元素
    df = DataFrame({"A": [str(x) for x in range(3)], "B": [str(x) for x in range(3)]})

    # 根据'A'列进行分组，对'B'列应用lambda函数，生成结果DataFrame
    grouped = df.groupby("A", as_index=False, sort=False)
    result = grouped.agg({"B": lambda x: list(x)})

    # 创建预期的DataFrame，其'B'列包含由字符串列表组成的单元素列表
    expected = DataFrame(
        {"A": [str(x) for x in range(3)], "B": [[str(x)] for x in range(3)]}
    )

    # 使用测试框架检查结果DataFrame和预期DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试在DataFrame上对分组后的列应用带有时区的lambda函数的行为
def test_agg_lambda_with_timezone():
    # 创建一个包含'tag'列和'date'列的DataFrame，其中'date'列包含带有时区信息的时间戳
    df = DataFrame(
        {
            "tag": [1, 1],
            "date": [
                pd.Timestamp("2018-01-01", tz="UTC"),
                pd.Timestamp("2018-01-02", tz="UTC"),
            ],
        }
    )

    # 根据'tag'列进行分组，对'date'列应用lambda函数，生成结果DataFrame
    result = df.groupby("tag").agg({"date": lambda e: e.head(1)})

    # 创建预期的DataFrame，包含带有索引和列名称的时间戳
    expected = DataFrame(
        [pd.Timestamp("2018-01-01", tz="UTC")],
        index=Index([1], name="tag"),
        columns=["date"],
    )

    # 使用测试框架检查结果DataFrame和预期DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 使用pytest的参数化装饰器定义一个测试函数，测试在DataFrame上对分组后的列应用自定义聚合函数的行为
@pytest.mark.parametrize(
    "err_cls",
    [
        NotImplementedError,
        RuntimeError,
        KeyError,
        IndexError,
        OSError,
        ValueError,
        ArithmeticError,
        AttributeError,
    ],
)
def test_groupby_agg_err_catching(err_cls):
    # 确保除了TypeError或AssertionError外，捕获所有其他异常
    # 引入非标准的EA以确保我们不会进入ndarray路径
    from pandas.tests.extension.decimal.array import (
        DecimalArray,
        make_data,
        to_decimal,
    )

    # 创建一个包含'id1'列，'id2'列和'decimals'列（带有自定义数组数据）的DataFrame
    data = make_data()[:5]
    df = DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    # 创建预期的Series，其值是通过自定义函数'weird_func'处理后的结果
    expected = Series(to_decimal([data[0], data[3]]))

    # 定义一个奇怪的函数'weird_func'，如果长度为0则抛出指定的异常，否则返回第一个元素
    def weird_func(x):
        if len(x) == 0:
            raise err_cls
        return x.iloc[0]

    # 根据'id1'列进行分组，对'decimals'列应用自定义函数'weird_func'，生成结果Series
    result = df["decimals"].groupby(df["id1"]).agg(weird_func)

    # 使用测试框架检查结果Series和预期Series是否相等，不检查名称
    tm.assert_series_equal(result, expected, check_names=False)
```