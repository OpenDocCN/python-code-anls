# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_filters.py`

```
# 导入必要的库和模块
from string import ascii_lowercase

# 导入numpy库并用np作为别名
import numpy as np

# 导入pytest库用于单元测试
import pytest

# 导入pandas库并用pd作为别名
import pandas as pd

# 从pandas中导入DataFrame、Series和Timestamp类
from pandas import (
    DataFrame,
    Series,
    Timestamp,
)

# 导入pandas内部测试模块，用tm作为别名
import pandas._testing as tm


# 定义测试函数test_filter_series
def test_filter_series():
    # 创建一个Series对象s
    s = Series([1, 3, 20, 5, 22, 24, 7])
    
    # 创建期望的奇数Series对象和对应的索引
    expected_odd = Series([1, 3, 5, 7], index=[0, 1, 3, 6])
    
    # 创建期望的偶数Series对象和对应的索引
    expected_even = Series([20, 22, 24], index=[2, 4, 5])
    
    # 根据奇偶性分组，返回每个元素对2取模的结果
    grouper = s.apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = s.groupby(grouper)
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与期望的奇数结果相等
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() < 10), expected_odd)
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与期望的偶数结果相等
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() > 10), expected_even)
    
    # 测试dropna=False的情况
    tm.assert_series_equal(
        grouped.filter(lambda x: x.mean() < 10, dropna=False),
        expected_odd.reindex(s.index),
    )
    tm.assert_series_equal(
        grouped.filter(lambda x: x.mean() > 10, dropna=False),
        expected_even.reindex(s.index),
    )


# 定义测试函数test_filter_single_column_df
def test_filter_single_column_df():
    # 创建一个DataFrame对象df
    df = DataFrame([1, 3, 20, 5, 22, 24, 7])
    
    # 创建期望的奇数DataFrame对象和对应的索引
    expected_odd = DataFrame([1, 3, 5, 7], index=[0, 1, 3, 6])
    
    # 创建期望的偶数DataFrame对象和对应的索引
    expected_even = DataFrame([20, 22, 24], index=[2, 4, 5])
    
    # 根据第一列元素的奇偶性进行分组
    grouper = df[0].apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = df.groupby(grouper)
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与期望的奇数结果相等
    tm.assert_frame_equal(grouped.filter(lambda x: x.mean() < 10), expected_odd)
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与期望的偶数结果相等
    tm.assert_frame_equal(grouped.filter(lambda x: x.mean() > 10), expected_even)
    
    # 测试dropna=False的情况
    tm.assert_frame_equal(
        grouped.filter(lambda x: x.mean() < 10, dropna=False),
        expected_odd.reindex(df.index),
    )
    tm.assert_frame_equal(
        grouped.filter(lambda x: x.mean() > 10, dropna=False),
        expected_even.reindex(df.index),
    )


# 定义测试函数test_filter_multi_column_df
def test_filter_multi_column_df():
    # 创建一个包含多列数据的DataFrame对象df
    df = DataFrame({"A": [1, 12, 12, 1], "B": [1, 1, 1, 1]})
    
    # 根据第一列元素的奇偶性进行分组
    grouper = df["A"].apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = df.groupby(grouper)
    
    # 创建期望的DataFrame对象和对应的索引
    expected = DataFrame({"A": [12, 12], "B": [1, 1]}, index=[1, 2])
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与期望的结果相等
    tm.assert_frame_equal(
        grouped.filter(lambda x: x["A"].sum() - x["B"].sum() > 10), expected
    )


# 定义测试函数test_filter_mixed_df
def test_filter_mixed_df():
    # 创建一个包含多列数据的DataFrame对象df
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    
    # 根据第一列元素的奇偶性进行分组
    grouper = df["A"].apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = df.groupby(grouper)
    
    # 创建期望的DataFrame对象和对应的索引
    expected = DataFrame({"A": [12, 12], "B": ["b", "c"]}, index=[1, 2])
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与期望的结果相等
    tm.assert_frame_equal(grouped.filter(lambda x: x["A"].sum() > 10), expected)


# 定义测试函数test_filter_out_all_groups
def test_filter_out_all_groups():
    # 创建一个Series对象s
    s = Series([1, 3, 20, 5, 22, 24, 7])
    
    # 根据元素值的奇偶性进行分组
    grouper = s.apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = s.groupby(grouper)
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与空Series对象相等
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() > 1000), s[[]])
    
    # 创建一个包含多列数据的DataFrame对象df
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    
    # 根据第一列元素的奇偶性进行分组
    grouper = df["A"].apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = df.groupby(grouper)
    
    # 使用内部测试模块tm的方法断言，验证过滤后的结果与空DataFrame对象相等
    tm.assert_frame_equal(grouped.filter(lambda x: x["A"].sum() > 1000), df.loc[[]])


# 定义测试函数test_filter_out_no_groups
def test_filter_out_no_groups():
    # 创建一个Series对象s
    s = Series([1, 3, 20, 5, 22, 24, 7])
    
    # 根据元素值的奇偶性进行分组
    grouper = s.apply(lambda x: x % 2)
    
    # 根据分组结果进行分组操作
    grouped = s.groupby(grouper)
    
    # 执行过滤操作，不做任何断言，测试过程中是否会出现错误
    filtered = grouped.filter(lambda x: x.mean() > 0)
    # 使用测试工具tm.assert_series_equal检查filtered和s两个Series是否相等
    tm.assert_series_equal(filtered, s)
def test_filter_out_no_groups_dataframe():
    # 创建一个测试数据框，包含列"A"和"B"
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split())
    # 根据列"A"的奇偶性分组
    grouper = df["A"].apply(lambda x: x % 2)
    # 按照分组结果对数据框进行分组
    grouped = df.groupby(grouper)
    # 过滤出分组后，列"A"的均值大于0的分组
    filtered = grouped.filter(lambda x: x["A"].mean() > 0)
    # 使用测试框架检查过滤后的数据框是否与原始数据框相等
    tm.assert_frame_equal(filtered, df)


def test_filter_out_all_groups_in_df():
    # GH12768
    # 创建一个测试数据框，包含列"a"和"b"
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 0]})
    # 按列"a"分组
    res = df.groupby("a")
    # 过滤出分组后，列"b"的和大于5的分组，保留NaN值
    res = res.filter(lambda x: x["b"].sum() > 5, dropna=False)
    # 创建一个期望的数据框，包含NaN值
    expected = DataFrame({"a": [np.nan] * 3, "b": [np.nan] * 3})
    # 使用测试框架检查过滤后的数据框是否与期望的数据框相等
    tm.assert_frame_equal(expected, res)


def test_filter_out_all_groups_in_df_dropna_true():
    # GH12768
    # 创建一个测试数据框，包含列"a"和"b"
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 0]})
    # 按列"a"分组
    res = df.groupby("a")
    # 过滤出分组后，列"b"的和大于5的分组，丢弃NaN值
    res = res.filter(lambda x: x["b"].sum() > 5, dropna=True)
    # 创建一个期望的数据框，不包含任何行
    expected = DataFrame({"a": [], "b": []}, dtype="int64")
    # 使用测试框架检查过滤后的数据框是否与期望的数据框相等
    tm.assert_frame_equal(expected, res)


def test_filter_condition_raises():
    # 定义一个函数，如果求和为零则抛出ValueError异常，否则返回求和结果是否大于零
    def raise_if_sum_is_zero(x):
        if x.sum() == 0:
            raise ValueError
        return x.sum() > 0

    # 创建一个测试序列
    s = Series([-1, 0, 1, 2])
    # 根据序列元素的奇偶性分组
    grouper = s.apply(lambda x: x % 2)
    # 按照分组结果对序列进行分组
    grouped = s.groupby(grouper)
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        grouped.filter(raise_if_sum_is_zero)


def test_filter_bad_shapes():
    # 创建一个测试数据框，包含列"A"、"B"和"C"
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    # 获取列"B"作为一个序列
    s = df["B"]
    # 按列"B"对数据框进行分组
    g_df = df.groupby("B")
    # 按序列s对s进行分组
    g_s = s.groupby(s)

    # 定义一个返回输入参数的函数
    f = lambda x: x
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "filter function returned a DataFrame, but expected a scalar bool"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)

    # 定义一个判断元素是否等于1的函数
    f = lambda x: x == 1
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "filter function returned a DataFrame, but expected a scalar bool"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)

    # 定义一个返回数组外积的函数
    f = lambda x: np.outer(x, x)
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "can't multiply sequence by non-int of type 'str'"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    # 使用测试框架检查是否会捕获TypeError异常，异常消息为指定的字符串
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)


def test_filter_nan_is_false():
    # 创建一个测试数据框，包含列"A"、"B"和"C"
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    # 获取列"B"作为一个序列
    s = df["B"]
    # 按列"B"对数据框进行分组
    g_df = df.groupby(df["B"])
    # 按序列s对s进行分组
    g_s = s.groupby(s)

    # 定义一个返回NaN的函数
    f = lambda x: np.nan
    # 使用测试框架检查过滤后的数据框是否与期望的数据框相等
    tm.assert_frame_equal(g_df.filter(f), df.loc[[]])
    # 使用测试框架检查过滤后的序列是否与期望的序列相等
    tm.assert_series_equal(g_s.filter(f), s[[]])


def test_filter_pdna_is_false():
    # 特别是，在过滤器中尝试调用bool(pd.NA)时不会引发异常
    # 创建一个测试数据框，包含列"A"、"B"和"C"
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    # 获取列"B"作为一个序列
    ser = df["B"]
    # 按列"B"对数据框进行分组
    g_df = df.groupby(df["B"])
    # 按序列ser对ser进行分组
    g_s = ser.groupby(ser)

    # 定义一个返回pd.NA的函数
    func = lambda x: pd.NA
    # 对数据框进行过滤，不会引发异常
    res = g_df.filter(func)
    # 使用测试工具比较两个数据帧的内容是否相等
    tm.assert_frame_equal(res, df.loc[[]])
    # 使用筛选函数对数据进行过滤操作，并将结果存储在变量 res 中
    res = g_s.filter(func)
    # 使用测试工具比较两个序列的内容是否相等
    tm.assert_series_equal(res, ser[[]])
def test_filter_against_workaround_ints():
    # Series of ints
    s = Series(np.random.default_rng(2).integers(0, 100, 10))
    # Apply rounding to nearest 10
    grouper = s.apply(lambda x: np.round(x, -1))
    # Group by rounded values
    grouped = s.groupby(grouper)
    # Define filter function
    f = lambda x: x.mean() > 10

    # Old way of filtering based on transform and type conversion
    old_way = s[grouped.transform(f).astype("bool")]
    # New way of filtering using filter function
    new_way = grouped.filter(f)
    # Assert equality after sorting
    tm.assert_series_equal(new_way.sort_values(), old_way.sort_values())


def test_filter_against_workaround_floats():
    # Series of floats
    s = 100 * Series(np.random.default_rng(2).random(10))
    # Apply rounding to nearest 10
    grouper = s.apply(lambda x: np.round(x, -1))
    # Group by rounded values
    grouped = s.groupby(grouper)
    # Define filter function
    f = lambda x: x.mean() > 10

    # Old way of filtering based on transform and type conversion
    old_way = s[grouped.transform(f).astype("bool")]
    # New way of filtering using filter function
    new_way = grouped.filter(f)
    # Assert equality after sorting
    tm.assert_series_equal(new_way.sort_values(), old_way.sort_values())


def test_filter_against_workaround_dataframe():
    # Set up DataFrame of ints, floats, strings.
    letters = np.array(list(ascii_lowercase))
    N = 10
    # Generate random letters based on given seed
    random_letters = letters.take(
        np.random.default_rng(2).integers(0, 26, N, dtype=int)
    )
    # Create DataFrame
    df = DataFrame(
        {
            "ints": Series(np.random.default_rng(2).integers(0, 10, N)),
            "floats": N / 10 * Series(np.random.default_rng(2).random(N)),
            "letters": Series(random_letters),
        }
    )

    # Group by ints; filter on floats.
    grouped = df.groupby("ints")
    old_way = df[grouped.floats.transform(lambda x: x.mean() > N / 2).astype("bool")]
    new_way = grouped.filter(lambda x: x["floats"].mean() > N / 2)
    tm.assert_frame_equal(new_way, old_way)

    # Group by floats (rounded); filter on strings.
    grouper = df.floats.apply(lambda x: np.round(x, -1))
    grouped = df.groupby(grouper)
    old_way = df[grouped.letters.transform(lambda x: len(x) < N / 2).astype("bool")]
    new_way = grouped.filter(lambda x: len(x.letters) < N / 2)
    tm.assert_frame_equal(new_way, old_way)

    # Group by strings; filter on ints.
    grouped = df.groupby("letters")
    old_way = df[grouped.ints.transform(lambda x: x.mean() > N / 2).astype("bool")]
    new_way = grouped.filter(lambda x: x["ints"].mean() > N / 2)
    tm.assert_frame_equal(new_way, old_way)


def test_filter_using_len():
    # GH 4447
    # Create DataFrame with specified columns
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    # Group by column 'B'
    grouped = df.groupby("B")
    # Filter groups where length is greater than 2
    actual = grouped.filter(lambda x: len(x) > 2)
    # Expected result based on filtering condition
    expected = DataFrame(
        {"A": np.arange(2, 6), "B": list("bbbb"), "C": np.arange(2, 6)},
        index=np.arange(2, 6, dtype=np.int64),
    )
    tm.assert_frame_equal(actual, expected)

    # Another filtering condition
    actual = grouped.filter(lambda x: len(x) > 4)
    # Expected empty DataFrame since condition isn't met
    expected = df.loc[[]]
    tm.assert_frame_equal(actual, expected)


def test_filter_using_len_series():
    # GH 4447
    # Create Series with specified name
    s = Series(list("aabbbbcc"), name="B")
    # Group by values in the Series
    grouped = s.groupby(s)
    # Filter groups where length is greater than 2
    actual = grouped.filter(lambda x: len(x) > 2)
    # Expected result based on filtering condition
    expected = Series(4 * ["b"], index=np.arange(2, 6, dtype=np.int64), name="B")
    tm.assert_series_equal(actual, expected)
    # 使用 `grouped` 对象的 `filter` 方法筛选出长度大于4的分组
    actual = grouped.filter(lambda x: len(x) > 4)
    # 创建一个空的 Series，用于存储预期的结果
    expected = s[[]]
    # 使用测试工具（`tm`）比较 `actual` 和 `expected` 的 Series 是否相等
    tm.assert_series_equal(actual, expected)
@pytest.mark.parametrize(
    "index", [range(8), range(7, -1, -1), [0, 2, 1, 3, 4, 6, 5, 7]]
)
# 定义测试函数，使用参数化装饰器，传入不同的索引列表进行参数化测试
def test_filter_maintains_ordering(index):
    # GH 4621: GitHub issue编号
    # 创建 DataFrame 对象，包含两列数据："pid" 和 "tag"，并指定索引为参数化传入的索引列表
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    # 提取 "pid" 列为 Series 对象
    s = df["pid"]
    # 根据 "tag" 列分组 DataFrame
    grouped = df.groupby("tag")
    # 对分组后的结果应用 filter 函数，筛选出满足条件的分组
    actual = grouped.filter(lambda x: len(x) > 1)
    # 期望结果为筛选后的 DataFrame
    expected = df.iloc[[1, 2, 4, 7]]
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(actual, expected)

    # 使用 "pid" 列分组 Series 对象
    grouped = s.groupby(df["tag"])
    # 对分组后的结果应用 filter 函数，筛选出满足条件的分组
    actual = grouped.filter(lambda x: len(x) > 1)
    # 期望结果为筛选后的 Series
    expected = s.iloc[[1, 2, 4, 7]]
    # 断言实际结果与期望结果相等
    tm.assert_series_equal(actual, expected)


# 测试函数：验证多个时间戳的 filter 函数
def test_filter_multiple_timestamp():
    # GH 10114: GitHub issue编号
    # 创建 DataFrame 对象，包含三列数据："A"、"B" 和 "C"
    df = DataFrame(
        {
            "A": np.arange(5, dtype="int64"),
            "B": ["foo", "bar", "foo", "bar", "bar"],
            "C": Timestamp("20130101"),
        }
    )

    # 根据 ["B", "C"] 列分组 DataFrame
    grouped = df.groupby(["B", "C"])

    # 对分组后的 "A" 列应用 filter 函数，始终返回原始 "A" 列
    result = grouped["A"].filter(lambda x: True)
    # 断言结果与原始 "A" 列相等
    tm.assert_series_equal(df["A"], result)

    # 对分组后的 "A" 列应用 transform 函数，返回每个组的长度
    result = grouped["A"].transform(len)
    # 期望结果为每组的长度组成的 Series
    expected = Series([2, 3, 2, 3, 3], name="A")
    # 断言结果与期望结果相等
    tm.assert_series_equal(result, expected)

    # 对整个 DataFrameGroupBy 对象应用 filter 函数，始终返回原始 DataFrame
    result = grouped.filter(lambda x: True)
    # 断言结果与原始 DataFrame 相等
    tm.assert_frame_equal(df, result)

    # 对整个 DataFrameGroupBy 对象应用 transform 函数，返回每组的求和结果
    result = grouped.transform("sum")
    # 期望结果为每组的求和结果组成的 DataFrame
    expected = DataFrame({"A": [2, 8, 2, 8, 8]})
    # 断言结果与期望结果相等
    tm.assert_frame_equal(result, expected)

    # 对整个 DataFrameGroupBy 对象应用 transform 函数，返回每组的长度
    result = grouped.transform(len)
    # 期望结果为每组的长度组成的 DataFrame
    expected = DataFrame({"A": [2, 3, 2, 3, 3]})
    # 断言结果与期望结果相等
    tm.assert_frame_equal(result, expected)


# 测试函数：验证带有非唯一整数索引的 filter 和 transform 函数
def test_filter_and_transform_with_non_unique_int_index():
    # GH4620: GitHub issue编号
    # 创建 DataFrame 对象，包含两列数据："pid" 和 "tag"，并指定索引为非唯一整数索引列表
    index = [1, 1, 1, 2, 1, 1, 0, 1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    # 根据 "tag" 列分组 DataFrame
    grouped_df = df.groupby("tag")
    # 提取 "pid" 列为 Series 对象
    ser = df["pid"]
    # 使用 "tag" 列分组 Series 对象
    grouped_ser = ser.groupby(df["tag"])
    # 期望结果的索引列表
    expected_indexes = [1, 2, 4, 7]

    # 对 DataFrameGroupBy 对象应用 filter 函数，筛选出满足条件的分组
    actual = grouped_df.filter(lambda x: len(x) > 1)
    # 期望结果为筛选后的 DataFrame
    expected = df.iloc[expected_indexes]
    # 断言结果与期望结果相等
    tm.assert_frame_equal(actual, expected)

    # 对 DataFrameGroupBy 对象应用带 dropna=False 参数的 filter 函数
    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # 期望结果为拷贝原 DataFrame，并将数据类型转换为 float64，然后将部分行设置为 NaN
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    # 断言结果与期望结果相等
    tm.assert_frame_equal(actual, expected)

    # 对 SeriesGroupBy 对象应用 filter 函数，筛选出满足条件的分组
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    # 期望结果为筛选后的 Series
    expected = ser.take(expected_indexes)
    # 断言结果与期望结果相等
    tm.assert_series_equal(actual, expected)

    # 对 SeriesGroupBy 对象应用带 dropna=False 参数的 filter 函数
    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    # 期望结果为手动创建的 Series，部分值为 NaN，因为带有 dropna=False 参数时行为会有所不同
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # 断言结果与期望结果相等
    tm.assert_series_equal(actual, expected)

    # 对 SeriesGroupBy 对象应用 transform 函数，返回每组的长度
    actual = grouped_ser.transform(len)
    # 期望结果为每组的长度组成的 Series
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    # 断言结果与期望结果相等
    tm.assert_series_equal(actual, expected)

    # 对 DataFrameGroupBy 对象应用 transform 函数，返回 "pid" 列每组的长度
    actual = grouped_df.pid.transform(len)
    # 使用测试框架中的函数比较两个序列是否相等，并断言它们相等
    tm.assert_series_equal(actual, expected)
# 定义一个测试函数，用于验证处理多个非唯一整数索引的数据集情况
def test_filter_and_transform_with_multiple_non_unique_int_index():
    # GH4620：指定测试案例编号，表示这是针对GitHub Issue 4620的测试
    index = [1, 1, 1, 2, 0, 0, 0, 1]  # 创建一个非唯一整数索引列表
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,  # 使用上面的索引列表作为DataFrame的索引
    )
    grouped_df = df.groupby("tag")  # 按照'tag'列分组DataFrame
    ser = df["pid"]  # 从DataFrame中获取'pid'列作为Series
    grouped_ser = ser.groupby(df["tag"])  # 根据'tag'列分组Series
    expected_indexes = [1, 2, 4, 7]  # 预期结果中的索引列表

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)  # 对DataFrame进行过滤操作
    expected = df.iloc[expected_indexes]  # 根据预期索引获取DataFrame的子集
    tm.assert_frame_equal(actual, expected)  # 使用测试框架比较实际和预期结果的DataFrame是否相等

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)  # 进行不删除NaN值的过滤操作
    expected = df.copy().astype("float64")  # 复制DataFrame并将其类型转换为float64以避免上溢
    expected.iloc[[0, 3, 5, 6]] = np.nan  # 设置预期结果中指定位置的NaN值
    tm.assert_frame_equal(actual, expected)  # 使用测试框架比较实际和预期结果的DataFrame是否相等

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)  # 对Series进行过滤操作
    expected = ser.take(expected_indexes)  # 根据预期索引获取Series的子集
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)  # 进行不删除NaN值的过滤操作
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ 手动创建，因为这可能会让人感到困惑！（指出手动创建是因为数据可能会混乱）
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等

    # Transform Series
    actual = grouped_ser.transform(len)  # 对Series进行转换操作，计算每个分组的长度
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")  # 创建预期的Series结果
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)  # 对DataFrameGroupBy对象的'pid'列进行转换操作，计算每个分组的长度
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等


# 定义一个测试函数，用于验证处理非唯一浮点数索引的数据集情况
def test_filter_and_transform_with_non_unique_float_index():
    # GH4620：指定测试案例编号，表示这是针对GitHub Issue 4620的测试
    index = np.array([1, 1, 1, 2, 1, 1, 0, 1], dtype=float)  # 创建一个非唯一浮点数索引数组
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,  # 使用上面的索引数组作为DataFrame的索引
    )
    grouped_df = df.groupby("tag")  # 按照'tag'列分组DataFrame
    ser = df["pid"]  # 从DataFrame中获取'pid'列作为Series
    grouped_ser = ser.groupby(df["tag"])  # 根据'tag'列分组Series
    expected_indexes = [1, 2, 4, 7]  # 预期结果中的索引列表

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)  # 对DataFrame进行过滤操作
    expected = df.iloc[expected_indexes]  # 根据预期索引获取DataFrame的子集
    tm.assert_frame_equal(actual, expected)  # 使用测试框架比较实际和预期结果的DataFrame是否相等

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)  # 进行不删除NaN值的过滤操作
    expected = df.copy().astype("float64")  # 复制DataFrame并将其类型转换为float64以避免上溢
    expected.iloc[[0, 3, 5, 6]] = np.nan  # 设置预期结果中指定位置的NaN值
    tm.assert_frame_equal(actual, expected)  # 使用测试框架比较实际和预期结果的DataFrame是否相等

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)  # 对Series进行过滤操作
    expected = ser.take(expected_indexes)  # 根据预期索引获取Series的子集
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)  # 进行不删除NaN值的过滤操作
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ 手动创建，因为这可能会让人感到困惑！（指出手动创建是因为数据可能会混乱）
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等

    # Transform Series
    actual = grouped_ser.transform(len)  # 对Series进行转换操作，计算每个分组的长度
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")  # 创建预期的Series结果
    tm.assert_series_equal(actual, expected)  # 使用测试框架比较实际和预期结果的Series是否相等
    # 调用 DataFrame 列（pid）的 transform 方法，对每个分组计算其长度（即每个分组内的元素数量），返回一个 Series
    actual = grouped_df.pid.transform(len)
    # 使用测试工具（tm）的 assert_series_equal 方法，比较计算得到的 Series（actual）和期望的 Series（expected），确认它们是否相等
    tm.assert_series_equal(actual, expected)
# 定义一个测试函数，用于测试在非唯一时间戳索引情况下的数据过滤和转换
def test_filter_and_transform_with_non_unique_timestamp_index():
    # GH4620，指明这段代码是与 GitHub 问题编号4620 相关的测试
    t0 = Timestamp("2013-09-30 00:05:00")  # 创建一个时间戳对象 t0
    t1 = Timestamp("2013-10-30 00:05:00")  # 创建一个时间戳对象 t1
    t2 = Timestamp("2013-11-30 00:05:00")  # 创建一个时间戳对象 t2
    index = [t1, t1, t1, t2, t1, t1, t0, t1]  # 创建一个时间戳索引列表 index
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},  # 创建一个数据框 df，包含两列数据
        index=index,  # 设置数据框的索引为 index
    )
    grouped_df = df.groupby("tag")  # 根据 'tag' 列对 df 进行分组，得到分组后的数据框 grouped_df
    ser = df["pid"]  # 从 df 中取出 'pid' 列，得到一个序列对象 ser
    grouped_ser = ser.groupby(df["tag"])  # 根据 'tag' 列对 ser 进行分组，得到分组后的序列对象 grouped_ser
    expected_indexes = [1, 2, 4, 7]  # 期望的索引列表

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)  # 对 grouped_df 应用过滤器，保留长度大于1的分组
    expected = df.iloc[expected_indexes]  # 期望的结果数据框，通过索引 expected_indexes 进行切片获得
    tm.assert_frame_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)  # 使用 dropna=False 参数对 grouped_df 进行过滤
    expected = df.copy().astype("float64")  # 复制 df，并将其转换为 float64 类型，以避免 NaN 被提升
    expected.iloc[[0, 3, 5, 6]] = np.nan  # 将索引为 [0, 3, 5, 6] 的位置设置为 NaN
    tm.assert_frame_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)  # 对 grouped_ser 应用过滤器，保留长度大于1的分组
    expected = ser.take(expected_indexes)  # 通过 take 方法从 ser 中取出期望的索引值
    tm.assert_series_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)  # 使用 dropna=False 参数对 grouped_ser 进行过滤
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # 手动创建期望的序列，因为此处可能会有混淆
    tm.assert_series_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    # Transform Series
    actual = grouped_ser.transform(len)  # 对 grouped_ser 应用变换函数 len，得到每个分组的长度
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")  # 期望的结果序列
    tm.assert_series_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)  # 对 grouped_df 的 'pid' 列应用变换函数 len
    tm.assert_series_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较


# 定义一个测试函数，用于测试在非唯一字符串索引情况下的数据过滤和转换
def test_filter_and_transform_with_non_unique_string_index():
    # GH4620，指明这段代码是与 GitHub 问题编号4620 相关的测试
    index = list("bbbcbbab")  # 创建一个非唯一字符串索引列表 index
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},  # 创建一个数据框 df，包含两列数据
        index=index,  # 设置数据框的索引为 index
    )
    grouped_df = df.groupby("tag")  # 根据 'tag' 列对 df 进行分组，得到分组后的数据框 grouped_df
    ser = df["pid"]  # 从 df 中取出 'pid' 列，得到一个序列对象 ser
    grouped_ser = ser.groupby(df["tag"])  # 根据 'tag' 列对 ser 进行分组，得到分组后的序列对象 grouped_ser
    expected_indexes = [1, 2, 4, 7]  # 期望的索引列表

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)  # 对 grouped_df 应用过滤器，保留长度大于1的分组
    expected = df.iloc[expected_indexes]  # 期望的结果数据框，通过索引 expected_indexes 进行切片获得
    tm.assert_frame_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)  # 使用 dropna=False 参数对 grouped_df 进行过滤
    expected = df.copy().astype("float64")  # 复制 df，并将其转换为 float64 类型，以避免 NaN 被提升
    expected.iloc[[0, 3, 5, 6]] = np.nan  # 将索引为 [0, 3, 5, 6] 的位置设置为 NaN
    tm.assert_frame_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)  # 对 grouped_ser 应用过滤器，保留长度大于1的分组
    expected = ser.take(expected_indexes)  # 通过 take 方法从 ser 中取出期望的索引值
    tm.assert_series_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)  # 使用 dropna=False 参数对 grouped_ser 进行过滤
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # 手动创建期望的序列，因为此处可能会有混淆
    tm.assert_series_equal(actual, expected)  # 使用测试框架进行实际结果和期望结果的比较

    # Transform Series
    actual = grouped_ser.transform(len)  # 对 grouped_ser 应用变换函数 len，得到每个分组的长度
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")  # 期望的结果序列
    # 断言：验证实际结果 Series 是否与预期结果相等
    tm.assert_series_equal(actual, expected)
    
    # DataFrameGroupBy 的 transform 操作：对分组后的 DataFrame 中的某一列进行长度变换
    actual = grouped_df.pid.transform(len)
    # 断言：验证实际结果 Series 是否与预期结果相等
    tm.assert_series_equal(actual, expected)
# 测试函数：检验 filter 方法是否能够正确访问分组后的列
def test_filter_has_access_to_grouped_cols():
    # 创建一个 DataFrame，包含两列 A 和 B
    df = DataFrame([[1, 2], [1, 3], [5, 6]], columns=["A", "B"])
    # 按照列 "A" 进行分组
    g = df.groupby("A")
    # 使用 filter 方法，筛选出满足条件的分组结果
    filt = g.filter(lambda x: x["A"].sum() == 2)
    # 断言筛选后的结果与预期的 DataFrame 的行索引一致
    tm.assert_frame_equal(filt, df.iloc[[0, 1]])

# 测试函数：检验 filter 方法是否会强制要求返回布尔值
def test_filter_enforces_scalarness():
    # 创建一个 DataFrame，包含三列 "a", "b", "c"
    df = DataFrame(
        [
            ["best", "a", "x"],
            ["worst", "b", "y"],
            ["best", "c", "x"],
            ["best", "d", "y"],
            ["worst", "d", "y"],
            ["worst", "d", "y"],
            ["best", "d", "z"],
        ],
        columns=["a", "b", "c"],
    )
    # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配指定的错误消息
    with pytest.raises(TypeError, match="filter function returned a.*"):
        df.groupby("c").filter(lambda g: g["a"] == "best")

# 测试函数：检验 filter 方法是否会在返回非布尔值时引发错误
def test_filter_non_bool_raises():
    # 创建一个 DataFrame，包含三列 "a", "b", "c"
    df = DataFrame(
        [
            ["best", "a", 1],
            ["worst", "b", 1],
            ["best", "c", 1],
            ["best", "d", 1],
            ["worst", "d", 1],
            ["worst", "d", 1],
            ["best", "d", 1],
        ],
        columns=["a", "b", "c"],
    )
    # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配指定的错误消息
    with pytest.raises(TypeError, match="filter function returned a.*"):
        df.groupby("a").filter(lambda g: g.c.mean())

# 测试函数：检验 filter 方法在空分组中使用 dropna 参数的行为
def test_filter_dropna_with_empty_groups():
    # 创建一个随机数据的 Series，并按照索引值进行分组
    data = Series(np.random.default_rng(2).random(9), index=np.repeat([1, 2, 3], 3))
    grouped = data.groupby(level=0)
    # 使用 filter 方法，并设置 dropna=False，期望结果中保留 NaN 值
    result_false = grouped.filter(lambda x: x.mean() > 1, dropna=False)
    expected_false = Series([np.nan] * 9, index=np.repeat([1, 2, 3], 3))
    tm.assert_series_equal(result_false, expected_false)
    # 使用 filter 方法，并设置 dropna=True，期望结果中没有任何数据
    result_true = grouped.filter(lambda x: x.mean() > 1, dropna=True)
    expected_true = Series(index=pd.Index([], dtype=int), dtype=np.float64)
    tm.assert_series_equal(result_true, expected_true)

# 测试函数：检验在聚合函数后使用 filter 方法的一致性行为
def test_filter_consistent_result_before_after_agg_func():
    # 创建一个 DataFrame，包含 "data" 和 "key" 两列
    df = DataFrame({"data": range(6), "key": list("ABCABC")})
    grouper = df.groupby("key")
    # 使用 filter 方法并断言结果与预期的 DataFrame 相同
    result = grouper.filter(lambda x: True)
    expected = DataFrame({"data": range(6), "key": list("ABCABC")})
    tm.assert_frame_equal(result, expected)
    # 对分组进行 sum 聚合操作
    grouper.sum()
    # 再次使用 filter 方法并断言结果与预期的 DataFrame 相同
    result = grouper.filter(lambda x: True)
    tm.assert_frame_equal(result, expected)
```