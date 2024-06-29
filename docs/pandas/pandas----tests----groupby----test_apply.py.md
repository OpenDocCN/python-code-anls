# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_apply.py`

```
# 从 datetime 模块中导入 date 和 datetime 类
# 导入 numpy 库并使用别名 np
# 导入 pytest 库
# 导入 pandas 库并使用别名 pd
# 从 pandas 库中导入 DataFrame、Index、MultiIndex、Series 和 bdate_range 函数
# 导入 pandas._testing 库并使用别名 tm
# 从 pandas.tests.groupby 中导入 get_groupby_method_args 函数
from datetime import (
    date,
    datetime,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    bdate_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


def test_apply_func_that_appends_group_to_list_without_copy():
    # GH: 17718
    # 创建一个 DataFrame，所有值为 1，索引为重复的整数列表，列为单列 0，并重置索引
    df = DataFrame(1, index=list(range(10)) * 10, columns=[0]).reset_index()
    # 初始化一个空列表 groups，用于存储分组后的数据
    groups = []

    def store(group):
        # 将每个分组追加到 groups 列表中
        groups.append(group)

    # 定义警告信息的字符串
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 pytest 的 assert_produces_warning 上下文管理器捕获 DeprecationWarning 异常，匹配警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 按 'index' 列分组并应用 store 函数
        df.groupby("index").apply(store)
    # 期望的结果 DataFrame
    expected_value = DataFrame(
        {"index": [0] * 10, 0: [1] * 10}, index=pd.RangeIndex(0, 100, 10)
    )

    # 使用 pandas._testing 的 assert_frame_equal 函数比较 groups[0] 和 expected_value 是否相等
    tm.assert_frame_equal(groups[0], expected_value)


def test_apply_index_date(using_infer_string):
    # GH 5788
    # 时间戳列表
    ts = [
        "2011-05-16 00:00",
        "2011-05-16 01:00",
        "2011-05-16 02:00",
        "2011-05-16 03:00",
        "2011-05-17 02:00",
        "2011-05-17 03:00",
        "2011-05-17 04:00",
        "2011-05-17 05:00",
        "2011-05-18 02:00",
        "2011-05-18 03:00",
        "2011-05-18 04:00",
        "2011-05-18 05:00",
    ]
    # 创建 DataFrame，包含值和时间戳作为索引
    df = DataFrame(
        {
            "value": [
                1.40893,
                1.40760,
                1.40750,
                1.40649,
                1.40893,
                1.40760,
                1.40750,
                1.40649,
                1.40893,
                1.40760,
                1.40750,
                1.40649,
            ],
        },
        index=Index(pd.to_datetime(ts), name="date_time"),
    )
    # 期望的结果是按日期分组后的索引的最大值
    expected = df.groupby(df.index.date).idxmax()
    # 使用 apply 函数计算每个分组的索引的最大值
    result = df.groupby(df.index.date).apply(lambda x: x.idxmax())
    # 使用 pandas._testing 的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_apply_index_date_object(using_infer_string):
    # GH 5789
    # 不自动强制转换日期
    ts = [
        "2011-05-16 00:00",
        "2011-05-16 01:00",
        "2011-05-16 02:00",
        "2011-05-16 03:00",
        "2011-05-17 02:00",
        "2011-05-17 03:00",
        "2011-05-17 04:00",
        "2011-05-17 05:00",
        "2011-05-18 02:00",
        "2011-05-18 03:00",
        "2011-05-18 04:00",
        "2011-05-18 05:00",
    ]
    # 创建包含日期和时间的 DataFrame
    df = DataFrame([row.split() for row in ts], columns=["date", "time"])
    # 添加值列
    df["value"] = [
        1.40893,
        1.40760,
        1.40750,
        1.40649,
        1.40893,
        1.40760,
        1.40750,
        1.40649,
        1.40893,
        1.40760,
        1.40750,
        1.40649,
    ]
    # 根据 using_infer_string 变量确定 dtype 类型
    dtype = "string[pyarrow_numpy]" if using_infer_string else object
    # 期望的索引对象，不自动强制转换日期
    exp_idx = Index(
        ["2011-05-16", "2011-05-17", "2011-05-18"], dtype=dtype, name="date"
    )
    # 期望的结果是按日期分组后的最大时间
    expected = Series(["00:00", "02:00", "02:00"], index=exp_idx)
    # 定义警告信息的字符串
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 上下文管理器来确保特定的警告被触发，这里期望是 DeprecationWarning，且警告消息匹配变量 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame df 按照 "date" 列进行分组，不创建分组键，然后应用函数
        result = df.groupby("date", group_keys=False).apply(
            # 对每个分组 x 执行操作，选择具有最大值的 "value" 列的对应 "time" 列的值
            lambda x: x["time"][x["value"].idxmax()]
        )
    # 使用 assert_series_equal 函数比较 result 和期望的结果 expected，确保它们在内容上完全一致
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "df, group_names",
    [
        # 参数化测试数据集，包括DataFrame和对应的group_names列表
        (DataFrame({"a": [1, 1, 1, 2, 3], "b": ["a", "a", "a", "b", "c"]}), [1, 2, 3]),
        (DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}), [0, 1]),
        (DataFrame({"a": [1]}), [1]),
        (DataFrame({"a": [1, 1, 1, 2, 2, 1, 1, 2], "b": range(8)}), [1, 2]),
        (DataFrame({"a": [1, 2, 3, 1, 2, 3], "two": [4, 5, 6, 7, 8, 9]}), [1, 2, 3]),
        (
            DataFrame(
                {
                    "a": list("aaabbbcccc"),
                    "B": [3, 4, 3, 6, 5, 2, 1, 9, 5, 4],
                    "C": [4, 0, 2, 2, 2, 7, 8, 6, 2, 8],
                }
            ),
            ["a", "b", "c"],
        ),
        (DataFrame([[1, 2, 3], [2, 2, 3]], columns=["a", "b", "c"]), [1, 2]),
    ],
    ids=[
        "GH2936",
        "GH7739 & GH10519",
        "GH10519",
        "GH2656",
        "GH12155",
        "GH20084",
        "GH21417",
    ],
)
def test_group_apply_once_per_group(df, group_names):
    # GH2936, GH7739, GH10519, GH2656, GH12155, GH20084, GH21417

    # 此测试确保每个分组只评估一次函数。
    # 之前函数在第一组上被评估了两次，以检查Cython索引滑块是否安全使用。
    # 此测试确保副作用（向列表追加）仅在每个分组中触发一次。

    names = []
    # 无法对函数进行参数化，因为它们需要外部的`names`来检测副作用

    def f_copy(group):
        # 使用快速应用路径进行操作
        names.append(group.name)
        return group.copy()

    def f_nocopy(group):
        # 使用慢速应用路径进行操作
        names.append(group.name)
        return group

    def f_scalar(group):
        # GH7739, GH2656
        names.append(group.name)
        return 0

    def f_none(group):
        # GH10519, GH12155, GH21417
        names.append(group.name)

    def f_constant_df(group):
        # GH2936, GH20084
        names.append(group.name)
        return DataFrame({"a": [1], "b": [1]})

    for func in [f_copy, f_nocopy, f_scalar, f_none, f_constant_df]:
        del names[:]
        
        # 断言触发DeprecationWarning警告，以匹配给定的消息
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            df.groupby("a", group_keys=False).apply(func)
        
        # 断言names列表与group_names相等
        assert names == group_names


def test_group_apply_once_per_group2(capsys):
    # GH: 31111
    # groupby-apply 需要执行len(set(group_by_columns))次

    expected = 2  # `apply`函数在当前测试中应该调用的次数

    # 创建测试数据框
    df = DataFrame(
        {
            "group_by_column": [0, 0, 0, 0, 1, 1, 1, 1],
            "test_column": ["0", "2", "4", "6", "8", "10", "12", "14"],
        },
        index=["0", "2", "4", "6", "8", "10", "12", "14"],
    )

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 pytest 中的 assert_produces_warning 上下文管理器来捕获特定的警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 进行分组操作，并禁用分组键
        df.groupby("group_by_column", group_keys=False).apply(
            # 对每个分组应用匿名函数，打印消息 "function_called"
            lambda df: print("function_called")
        )

    # 从 capsys 中读取标准输出，并计算输出中 "function_called" 出现的次数
    result = capsys.readouterr().out.count("function_called")
    # 如果 `groupby` 行为与预期不符，此测试将失败
    assert result == expected
def test_apply_fast_slow_identical():
    # GH 31613

    # 创建一个测试用的 DataFrame，包含两列 'A' 和 'b'
    df = DataFrame({"A": [0, 0, 1], "b": range(3)})

    # 定义一个返回组本身的函数
    def slow(group):
        return group

    # 定义一个返回组的副本的函数
    def fast(group):
        return group.copy()

    # 设置一个警告信息，用于检查 apply 操作是否作用在分组列上
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    # 断言使用 fast 函数进行 apply 操作后的 DataFrame 与使用 slow 函数相同
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        fast_df = df.groupby("A", group_keys=False).apply(fast)
    # 断言使用 slow 函数进行 apply 操作后的 DataFrame 与原始 DataFrame 相同
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        slow_df = df.groupby("A", group_keys=False).apply(slow)

    # 断言两个 DataFrame 在内容上完全相等
    tm.assert_frame_equal(fast_df, slow_df)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x,
        lambda x: x[:],
        lambda x: x.copy(deep=False),
        lambda x: x.copy(deep=True),
    ],
)
def test_groupby_apply_identity_maybecopy_index_identical(func):
    # GH 14927
    # 函数返回是否为输入数据的副本不应该影响结果的索引结构，因为用户无法透明地了解这一点

    # 创建一个测试用的 DataFrame，包含 'g', 'a', 'b' 三列
    df = DataFrame({"g": [1, 2, 2, 2], "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

    # 设置一个警告信息，用于检查 apply 操作是否作用在分组列上
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    # 断言使用指定函数进行 apply 操作后的结果 DataFrame 与原始 DataFrame 相同
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby("g", group_keys=False).apply(func)
    tm.assert_frame_equal(result, df)


def test_apply_with_mixed_dtype():
    # GH3480, apply with mixed dtype on axis=1 breaks in 0.11

    # 创建一个测试用的 DataFrame，包含两列 'foo1' 和 'foo2'
    df = DataFrame(
        {
            "foo1": np.random.default_rng(2).standard_normal(6),
            "foo2": ["one", "two", "two", "three", "one", "two"],
        }
    )

    # 对 DataFrame 沿着 axis=1 进行 apply 操作，并返回结果的数据类型
    result = df.apply(lambda x: x, axis=1).dtypes
    expected = df.dtypes

    # 断言 apply 操作后的数据类型与预期数据类型完全相等
    tm.assert_series_equal(result, expected)

    # GH 3610 incorrect dtype conversion with as_index=False

    # 创建一个测试用的 DataFrame，包含 'c1' 和 'c2' 两列
    df = DataFrame({"c1": [1, 2, 6, 6, 8]})
    df["c2"] = df.c1 / 2.0

    # 使用 groupby 对 'c2' 列进行分组，并计算均值，然后重置索引
    result1 = df.groupby("c2").mean().reset_index().c2
    # 使用 groupby 对 'c2' 列进行分组，并计算均值，不重置索引
    result2 = df.groupby("c2", as_index=False).mean().c2

    # 断言两种方法计算结果的 'c2' 列完全相等
    tm.assert_series_equal(result1, result2)


def test_groupby_as_index_apply():
    # GH #4648 and #3417

    # 创建一个测试用的 DataFrame，包含 'item_id', 'user_id', 'time' 三列
    df = DataFrame(
        {
            "item_id": ["b", "b", "a", "c", "a", "b"],
            "user_id": [1, 2, 1, 1, 3, 1],
            "time": range(6),
        }
    )

    # 根据 'user_id' 列进行分组，as_index=True
    g_as = df.groupby("user_id", as_index=True)
    # 根据 'user_id' 列进行分组，as_index=False
    g_not_as = df.groupby("user_id", as_index=False)

    # 使用 head(2) 方法获取分组后的前两行的索引，as_index=True
    res_as = g_as.head(2).index
    # 使用 head(2) 方法获取分组后的前两行的索引，as_index=False
    res_not_as = g_not_as.head(2).index

    # 预期的索引结果
    exp = Index([0, 1, 2, 4])

    # 断言两种分组方式得到的索引完全相等
    tm.assert_index_equal(res_as, exp)
    tm.assert_index_equal(res_not_as, exp)

    # 设置一个警告信息，用于检查 apply 操作是否作用在分组列上
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    # 断言使用 apply 方法对分组后的 DataFrame 进行处理后得到的索引与预期索引完全相等
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res_as_apply = g_as.apply(lambda x: x.head(2)).index
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res_not_as_apply = g_not_as.apply(lambda x: x.head(2)).index
    # apply doesn't maintain the original ordering
    # apply 方法不保持原始顺序
    
    # changed in GH5610 as the as_index=False returns a MI here
    # 在 GH5610 中更改，因为 as_index=False 在这里返回一个 MultiIndex
    
    # Define expected indices for testing purposes
    # 为测试目的定义预期的索引
    
    exp_not_as_apply = Index([0, 2, 1, 4])
    
    # Define a list of tuples representing expected MultiIndex with names
    # 定义一个元组列表，表示具有名称的预期 MultiIndex
    tp = [(1, 0), (1, 2), (2, 1), (3, 4)]
    
    # Create expected MultiIndex with specified names
    # 使用指定的名称创建预期的 MultiIndex
    exp_as_apply = MultiIndex.from_tuples(tp, names=["user_id", None])
    
    # Verify that res_as_apply matches exp_as_apply
    # 验证 res_as_apply 是否与 exp_as_apply 匹配
    tm.assert_index_equal(res_as_apply, exp_as_apply)
    
    # Verify that res_not_as_apply matches exp_not_as_apply
    # 验证 res_not_as_apply 是否与 exp_not_as_apply 匹配
    tm.assert_index_equal(res_not_as_apply, exp_not_as_apply)
# 测试用例：对于 DataFrame 的 groupby 操作，应用 apply 函数后保持索引不改变
def test_groupby_as_index_apply_str():
    # 创建索引对象，包含字符 'a', 'b', 'c', 'd', 'e'
    ind = Index(list("abcde"))
    # 创建 DataFrame，包含两列，使用上述索引对象作为索引
    df = DataFrame([[1, 2], [2, 3], [1, 4], [1, 5], [2, 6]], index=ind)
    # 需要显示的警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言操作应该产生 DeprecationWarning 警告，并且警告消息匹配 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 进行 groupby 操作，不作为索引，不显示组键，然后应用 lambda 函数并获取索引
        res = df.groupby(0, as_index=False, group_keys=False).apply(lambda x: x).index
    # 断言结果索引与初始索引相同
    tm.assert_index_equal(res, ind)


# 测试用例：应用 describe 函数后，保持分组的列名
def test_apply_concat_preserve_names(three_group):
    # 对三列数据进行按列 A 和 B 进行分组
    grouped = three_group.groupby(["A", "B"])

    # 描述函数1：生成描述统计结果，并将索引名设置为 "stat"
    def desc(group):
        result = group.describe()
        result.index.name = "stat"
        return result

    # 描述函数2：生成描述统计结果，截取相同长度并设置索引名为 "stat"，返回结果
    def desc2(group):
        result = group.describe()
        result.index.name = "stat"
        result = result[: len(group)]
        # weirdo
        return result

    # 描述函数3：生成描述统计结果，并根据分组长度设置索引名，返回截取后的结果
    def desc3(group):
        result = group.describe()
        # names are different
        result.index.name = f"stat_{len(group):d}"
        result = result[: len(group)]
        # weirdo
        return result

    # 需要显示的警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    # 断言操作应该产生 DeprecationWarning 警告，并且警告消息匹配 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组应用 desc 函数
        result = grouped.apply(desc)
    # 断言结果索引应该包含 ("A", "B", "stat")
    assert result.index.names == ("A", "B", "stat")

    # 断言操作应该产生 DeprecationWarning 警告，并且警告消息匹配 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组应用 desc2 函数
        result2 = grouped.apply(desc2)
    # 断言结果索引应该包含 ("A", "B", "stat")
    assert result2.index.names == ("A", "B", "stat")

    # 断言操作应该产生 DeprecationWarning 警告，并且警告消息匹配 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组应用 desc3 函数
        result3 = grouped.apply(desc3)
    # 断言结果索引应该包含 ("A", "B", None)
    assert result3.index.names == ("A", "B", None)


# 测试用例：将 Series 按月份分组，应用自定义函数 f 后返回 DataFrame，保持索引不变
def test_apply_series_to_frame():
    # 自定义函数 f，对分组中的数据进行数学操作
    def f(piece):
        with np.errstate(invalid="ignore"):
            logged = np.log(piece)
        # 返回一个包含多个列的 DataFrame，包括原始值、去均值化值和对数值
        return DataFrame(
            {"value": piece, "demeaned": piece - piece.mean(), "logged": logged}
        )

    # 生成时间序列
    dr = bdate_range("1/1/2000", periods=10)
    # 生成随机数列
    ts = Series(np.random.default_rng(2).standard_normal(10), index=dr)

    # 按月份对时间序列进行分组，不显示组键
    grouped = ts.groupby(lambda x: x.month, group_keys=False)
    # 应用自定义函数 f
    result = grouped.apply(f)

    # 断言结果应该是一个 DataFrame 类型
    assert isinstance(result, DataFrame)
    # 断言结果不应该具有名字属性
    assert not hasattr(result, "name")  # GH49907
    # 断言结果的索引应该与原始时间序列索引相同
    tm.assert_index_equal(result.index, ts.index)


# 测试用例：对 DataFrame 按列 A 和 B 进行分组，应用 len 函数后返回 Series，保持索引不变
def test_apply_series_yield_constant(df):
    # 对 DataFrame 按列 A 和 B 进行分组，应用 len 函数
    result = df.groupby(["A", "B"])["C"].apply(len)
    # 断言结果索引的前两个名字应该是 ("A", "B")
    assert result.index.names[:2] == ("A", "B")


# 测试用例：对 DataFrame 按列 A 和 B 进行分组，应用 len 函数后返回 Series，名字为 None
def test_apply_frame_yield_constant(df):
    # 需要显示的警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言操作应该产生 DeprecationWarning 警告，并且警告消息匹配 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 A 和 B 进行分组，应用 len 函数
        result = df.groupby(["A", "B"]).apply(len)
    # 断言结果应该是一个 Series 类型
    assert isinstance(result, Series)
    # 断言结果的名字属性应该是 None
    assert result.name is None

    # 对 DataFrame 按列 A 和 B 进行分组，应用 len 函数
    result = df.groupby(["A", "B"])[["C", "D"]].apply(len)
    # 断言结果应该是一个 Series 类型
    assert isinstance(result, Series)
    # 断言结果的名字属性应该是 None
    assert result.name is None


# 测试用例：对 DataFrame 按列 A 和 B 进行分组，无额外操作
def test_apply_frame_to_series(df):
    grouped = df.groupby(["A", "B"])
    # 定义一条警告消息，用于匹配 DeprecationWarning 类型的警告
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    
    # 使用 assert_produces_warning 上下文管理器来确保在应用函数时触发特定类型的警告，并匹配消息内容
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组后的数据应用 len 函数，得到结果
        result = grouped.apply(len)
    
    # 期望的结果是分组后的数据框中按列 'C' 的计数
    expected = grouped.count()["C"]
    
    # 断言结果的索引应与期望结果的索引相等
    tm.assert_index_equal(result.index, expected.index)
    
    # 断言结果的值数组应与期望结果的值数组相等
    tm.assert_numpy_array_equal(result.values, expected.values)
def test_apply_frame_not_as_index_column_name(df):
    # GH 35964 - path within _wrap_applied_output not hit by a test
    # 根据指定列 ["A", "B"] 对 DataFrame 进行分组，不将其作为索引，返回分组对象
    grouped = df.groupby(["A", "B"], as_index=False)
    # 设置警告信息，用于检查 DeprecationWarning 是否被触发
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象应用 len 函数，返回应用后的结果
        result = grouped.apply(len)
    # 使用 grouped.count() 计算每组的计数，重命名列名 {"C": np.nan}，并移除 "D" 列
    expected = grouped.count().rename(columns={"C": np.nan}).drop(columns="D")
    # 检查结果的索引与期望的索引是否相等
    tm.assert_index_equal(result.index, expected.index)
    # 检查结果的值数组与期望的值数组是否相等
    tm.assert_numpy_array_equal(result.values, expected.values)


def test_apply_frame_concat_series():
    def trans(group):
        # 对分组后的每个组，按列 "B" 进行分组，计算 "C" 列的总和，排序后返回前两个值
        return group.groupby("B")["C"].sum().sort_values().iloc[:2]

    def trans2(group):
        # 根据 group 的索引重新对 df 进行索引，然后按 "B" 列进行分组，计算总和并排序后返回前两个值
        grouped = group.groupby(df.reindex(group.index)["B"])
        return grouped.sum().sort_values().iloc[:2]

    # 创建 DataFrame 对象 df，包含三列，使用随机数填充
    df = DataFrame(
        {
            "A": np.random.default_rng(2).integers(0, 5, 1000),
            "B": np.random.default_rng(2).integers(0, 5, 1000),
            "C": np.random.default_rng(2).standard_normal(1000),
        }
    )

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 df 按列 "A" 进行分组，并应用 trans 函数
        result = df.groupby("A").apply(trans)
    # 对 df 按列 "A" 进行分组，并应用 trans2 函数
    exp = df.groupby("A")["C"].apply(trans2)
    # 检查结果 Series 是否相等，不检查名称
    tm.assert_series_equal(result, exp, check_names=False)
    # 断言结果 Series 的名称为 "C"
    assert result.name == "C"


def test_apply_transform(ts):
    # 对时间序列 ts 按月份进行分组，不生成分组键作为索引
    grouped = ts.groupby(lambda x: x.month, group_keys=False)
    # 对每个分组应用 lambda 函数，使每个元素乘以2
    result = grouped.apply(lambda x: x * 2)
    # 对每个分组应用 transform 函数，使每个元素乘以2
    expected = grouped.transform(lambda x: x * 2)
    # 检查结果 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_apply_multikey_corner(tsframe):
    # 对时间序列框架 tsframe 按年份和月份进行分组
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])

    def f(group):
        # 对每个组按列 "A" 进行排序，返回排序后的后5行
        return group.sort_values("A")[-5:]

    # 对分组对象应用函数 f，返回结果
    result = grouped.apply(f)
    # 遍历分组对象，检查每个组是否与 f 函数应用后的结果相等
    for key, group in grouped:
        tm.assert_frame_equal(result.loc[key], f(group))


@pytest.mark.parametrize("group_keys", [True, False])
def test_apply_chunk_view(group_keys):
    # 创建 DataFrame 对象 df，包含两列
    df = DataFrame({"key": [1, 1, 1, 2, 2, 2, 3, 3, 3], "value": range(9)})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 df 按 "key" 列进行分组，并应用 lambda 函数，返回前两行
        result = df.groupby("key", group_keys=group_keys).apply(lambda x: x.iloc[:2])
    # 从 df 中取出指定索引位置的行，作为期望的结果
    expected = df.take([0, 1, 3, 4, 6, 7])
    if group_keys:
        # 如果 group_keys 为 True，创建 MultiIndex，并将其设置为期望的结果的索引
        expected.index = MultiIndex.from_arrays(
            [[1, 1, 2, 2, 3, 3], expected.index], names=["key", None]
        )

    # 检查结果 DataFrame 是否与期望的结果 DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_apply_no_name_column_conflict():
    # 创建 DataFrame 对象 df，包含三列，其中两列名称相同
    df = DataFrame(
        {
            "name": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "name2": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            "value": range(9, -1, -1),
        }
    )

    # 对 df 按 ["name", "name2"] 列进行分组
    grouped = df.groupby(["name", "name2"])
    # 定义一条警告消息，用于匹配抛出的警告类型
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器确保在操作中产生特定类型的警告，并匹配警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组后的数据应用函数，此处为按照"value"列排序，直接修改原数据
        grouped.apply(lambda x: x.sort_values("value", inplace=True))
# 定义一个测试函数，用于测试在应用函数时处理类型转换失败的情况
def test_apply_typecast_fail():
    # 创建一个 DataFrame 对象，包含列 'd', 'c', 'v'，其中 'd' 是浮点数数组，'c' 是重复的字符串数组，'v' 是连续的浮点数数组
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "c": np.tile(["a", "b", "c"], 2),
            "v": np.arange(1.0, 7.0),
        }
    )

    # 定义一个函数 f，接受一个分组，计算 'v' 列的归一化值，并将结果存储在 'v2' 列中，然后返回修改后的分组
    def f(group):
        v = group["v"]
        group["v2"] = (v - v.min()) / (v.max() - v.min())
        return group

    # 设置警告消息，用于检查是否有关于 DataFrameGroupBy.apply 操作在分组列上的弃用警告
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器检查是否产生特定的 DeprecationWarning 警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 'd' 进行分组，并应用函数 f，将结果存储在 result 变量中
        result = df.groupby("d", group_keys=False).apply(f)

    # 创建一个预期的 DataFrame，复制原始 DataFrame，并在其上添加 'v2' 列，其中包含预期的归一化值
    expected = df.copy()
    expected["v2"] = np.tile([0.0, 0.5, 1], 2)

    # 使用断言函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试在应用函数时处理多重索引失败的情况
def test_apply_multiindex_fail():
    # 创建一个具有多重索引的 DataFrame 对象，包含列 'd', 'c', 'v'，其中 'd' 是浮点数数组，'c' 是重复的字符串数组，'v' 是连续的浮点数数组
    index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "c": np.tile(["a", "b", "c"], 2),
            "v": np.arange(1.0, 7.0),
        },
        index=index,
    )

    # 定义一个函数 f，接受一个分组，计算 'v' 列的归一化值，并将结果存储在 'v2' 列中，然后返回修改后的分组
    def f(group):
        v = group["v"]
        group["v2"] = (v - v.min()) / (v.max() - v.min())
        return group

    # 设置警告消息，用于检查是否有关于 DataFrameGroupBy.apply 操作在分组列上的弃用警告
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器检查是否产生特定的 DeprecationWarning 警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 'd' 进行分组，并应用函数 f，将结果存储在 result 变量中
        result = df.groupby("d", group_keys=False).apply(f)

    # 创建一个预期的 DataFrame，复制原始 DataFrame，并在其上添加 'v2' 列，其中包含预期的归一化值
    expected = df.copy()
    expected["v2"] = np.tile([0.0, 0.5, 1], 2)

    # 使用断言函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试应用函数时的边界情况，此处是对时间序列 DataFrame 对象应用函数
def test_apply_corner(tsframe):
    # 对时间序列 DataFrame 按照年份进行分组，并应用 lambda 函数对每个分组的值乘以 2，将结果存储在 result 变量中
    result = tsframe.groupby(lambda x: x.year, group_keys=False).apply(lambda x: x * 2)
    # 创建一个预期的 DataFrame，将原始时间序列 DataFrame 的每个值乘以 2
    expected = tsframe * 2
    # 使用断言函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试应用函数时不返回复制的情况，这里处理了应用函数可能失败的情况
def test_apply_without_copy():
    # 创建一个 DataFrame 对象，包含列 'id_field', 'category', 'value'，其中 'id_field' 是整数数组，'category' 是字符串数组，'value' 是整数数组
    data = DataFrame(
        {
            "id_field": [100, 100, 200, 300],
            "category": ["a", "b", "c", "c"],
            "value": [1, 2, 3, 4],
        }
    )

    # 定义一个函数 filt1，根据条件返回一个副本或筛选后的 DataFrame
    def filt1(x):
        if x.shape[0] == 1:
            return x.copy()
        else:
            return x[x.category == "c"]

    # 定义一个函数 filt2，根据条件返回原始或筛选后的 DataFrame
    def filt2(x):
        if x.shape[0] == 1:
            return x
        else:
            return x[x.category == "c"]

    # 设置警告消息，用于检查是否有关于 DataFrameGroupBy.apply 操作在分组列上的弃用警告
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器检查是否产生特定的 DeprecationWarning 警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 'id_field' 进行分组，并应用函数 filt1，将结果存储在 expected 变量中
        expected = data.groupby("id_field").apply(filt1)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 'id_field' 进行分组，并应用函数 filt2，将结果存储在 result 变量中
        result = data.groupby("id_field").apply(filt2)

    # 使用断言函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试应用函数时处理非排序轴上的重复值的情况
@pytest.mark.parametrize("test_series", [True, False])
def test_apply_with_duplicated_non_sorted_axis(test_series):
    # 创建一个 DataFrame 对象，包含值为 "x", "p", "x", "p", "x", "o" 的字符串数组，列名为 'X', 'Y'，索引为 [1, 2, 2]
    df = DataFrame(
        [["x", "p"], ["x", "p"], ["x", "o"]], columns=["X", "Y"], index=[1, 2, 2]
    )
    if`
    # 如果 test_series 为 True，执行以下操作
    if test_series:
        # 将 DataFrame df 按列 "Y" 设置索引，并选择列 "X" 作为 Series ser
        ser = df.set_index("Y")["X"]
        # 对 Series ser 按索引分组并应用 lambda 函数，该函数返回原数据，生成新的 Series result
        result = ser.groupby(level=0, group_keys=False).apply(lambda x: x)

        # 对结果按照索引排序，确保顺序
        result = result.sort_index()
        # 期望结果按照索引排序
        expected = ser.sort_index()
        # 断言结果与期望结果相等，抛出异常则测试失败
        tm.assert_series_equal(result, expected)
    else:
        # 定义错误消息
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 使用断言上下文，期望抛出 DeprecationWarning 异常，并匹配错误消息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对 DataFrame df 按列 "Y" 分组，应用 lambda 函数，返回原数据，生成新的 DataFrame result
            result = df.groupby("Y", group_keys=False).apply(lambda x: x)

        # 对结果按照 "Y" 列的值排序，确保顺序
        result = result.sort_values("Y")
        # 期望结果按照 "Y" 列排序
        expected = df.sort_values("Y")
        # 断言结果 DataFrame 与期望 DataFrame 相等，抛出异常则测试失败
        tm.assert_frame_equal(result, expected)
def test_apply_reindex_values():
    # GH: 26209
    # 从一个具有重复索引的groupby对象的单列重新索引导致ValueError（无法从重复轴重新索引）在0.24.2中，问题在#30679中解决了
    values = [1, 2, 3, 4]
    indices = [1, 1, 2, 2]
    # 创建DataFrame对象，包括一个重复索引的group列和值列
    df = DataFrame({"group": ["Group1", "Group2"] * 2, "value": values}, index=indices)
    # 创建期望的Series对象，使用相同的索引
    expected = Series(values, index=indices, name="value")

    def reindex_helper(x):
        # 重新索引输入的Series对象，从最小到最大索引
        return x.reindex(np.arange(x.index.min(), x.index.max() + 1))

    # 对groupby对象的值列应用reindex_helper函数
    result = df.groupby("group", group_keys=False).value.apply(reindex_helper)
    # 使用tm.assert_series_equal断言检查结果是否与期望相同
    tm.assert_series_equal(expected, result)


def test_apply_corner_cases():
    # #535, 不能使用滑动迭代器

    N = 10
    # 创建具有随机整数标签的DataFrame对象
    labels = np.random.default_rng(2).integers(0, 100, size=N)
    df = DataFrame(
        {
            "key": labels,
            "value1": np.random.default_rng(2).standard_normal(N),
            "value2": ["foo", "bar", "baz", "qux", "a"] * (N // 5),
        }
    )

    grouped = df.groupby("key", group_keys=False)

    def f(g):
        # 将value1列乘以2，并将结果作为value3列添加到输入的DataFrame对象g中
        g["value3"] = g["value1"] * 2
        return g

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用tm.assert_produces_warning断言检查警告是否被触发
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = grouped.apply(f)
    # 断言检查结果DataFrame对象中是否包含"value3"列
    assert "value3" in result


def test_apply_numeric_coercion_when_datetime():
    # 在过去，当存在datetime列时，groupby/apply操作对将dtypes转换为数字过于热心。这里是一些相关的GH问题的复制。

    # GH 15670
    # 创建DataFrame对象，包含Number列，Date列和Str列
    df = DataFrame(
        {"Number": [1, 2], "Date": ["2017-03-02"] * 2, "Str": ["foo", "inf"]}
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对Number列进行分组，并应用lambda函数选择每个组的第一个行
        expected = df.groupby(["Number"]).apply(lambda x: x.iloc[0])
    # 将Date列转换为datetime类型
    df.Date = pd.to_datetime(df.Date)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 再次对Number列进行分组，并应用lambda函数选择每个组的第一个行
        result = df.groupby(["Number"]).apply(lambda x: x.iloc[0])
    # 使用tm.assert_series_equal断言检查结果中Str列是否与期望中的相同
    tm.assert_series_equal(result["Str"], expected["Str"])


def test_apply_numeric_coercion_when_datetime_getitem():
    # GH 15421
    # 创建DataFrame对象，包含A列，B列和T列
    df = DataFrame(
        {"A": [10, 20, 30], "B": ["foo", "3", "4"], "T": [pd.Timestamp("12:31:22")] * 3}
    )

    def get_B(g):
        # 返回g的第一个行的B列作为Series对象
        return g.iloc[0][["B"]]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对A列进行分组，并应用get_B函数获取每个组的B列，然后获取结果的B列
        result = df.groupby("A").apply(get_B)["B"]
    # 创建期望的Series对象，使用A列作为索引
    expected = df.B
    expected.index = df.A
    # 使用tm.assert_series_equal断言检查结果与期望是否相同
    tm.assert_series_equal(result, expected)


def test_apply_numeric_coercion_when_datetime_with_nat():
    # GH 14423
    def predictions(tool):
        # 创建一个包含 "p1", "p2", "useTime" 索引的 Series 对象，初始值为 None
        out = Series(index=["p1", "p2", "useTime"], dtype=object)
        # 如果 "step1" 在 tool 的 State 列表中
        if "step1" in list(tool.State):
            # 设置 "p1" 为符合条件的第一个 Machine 值的字符串表示
            out["p1"] = str(tool[tool.State == "step1"].Machine.values[0])
        # 如果 "step2" 在 tool 的 State 列表中
        if "step2" in list(tool.State):
            # 设置 "p2" 为符合条件的第一个 Machine 值的字符串表示
            out["p2"] = str(tool[tool.State == "step2"].Machine.values[0])
            # 设置 "useTime" 为符合条件的第一个 oTime 值的字符串表示
            out["useTime"] = str(tool[tool.State == "step2"].oTime.values[0])
        # 返回包含预测结果的 Series 对象
        return out

    # 创建 DataFrame 对象 df1
    df1 = DataFrame(
        {
            "Key": ["B", "B", "A", "A"],
            "State": ["step1", "step2", "step1", "step2"],
            "oTime": ["", "2016-09-19 05:24:33", "", "2016-09-19 23:59:04"],
            "Machine": ["23", "36L", "36R", "36R"],
        }
    )
    # 复制 DataFrame 对象 df1 为 df2
    df2 = df1.copy()
    # 将 df2 的 oTime 列转换为 datetime 类型
    df2.oTime = pd.to_datetime(df2.oTime)
    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言操作期望的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 使用 groupby 对象 df1 按 Key 分组，应用 predictions 函数后获取结果的 p1 列
        expected = df1.groupby("Key").apply(predictions).p1
    # 断言操作期望的警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 使用 groupby 对象 df2 按 Key 分组，应用 predictions 函数后获取结果的 p1 列
        result = df2.groupby("Key").apply(predictions).p1
    # 断言 expected 和 result 对象是否相等
    tm.assert_series_equal(expected, result)
def test_apply_aggregating_timedelta_and_datetime():
    # 回归测试 GH 15562
    # 在 0.20.0 版本之前，以下的 groupby 导致 ValueError 和 IndexError

    # 创建包含三列的 DataFrame，其中 "clientid" 列是字符串，"datetime" 列是相同的日期时间值
    df = DataFrame(
        {
            "clientid": ["A", "B", "C"],
            "datetime": [np.datetime64("2017-02-01 00:00:00")] * 3,
        }
    )

    # 添加一个名为 "time_delta_zero" 的列，其值为每行 "datetime" 列与自身相减的结果
    df["time_delta_zero"] = df.datetime - df.datetime

    # 设置警告信息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    # 断言确保以下代码块引发 DeprecationWarning 警告，并且警告消息与 "msg" 变量匹配
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 "clientid" 列进行分组，并应用 lambda 函数
        result = df.groupby("clientid").apply(
            lambda ddf: Series(
                {"clientid_age": ddf.time_delta_zero.min(), "date": ddf.datetime.min()}
            )
        )

    # 创建预期的 DataFrame，包含与 "clientid" 列相关联的客户 ID，"clientid_age" 列为 timedelta64(0, 'D')，"date" 列为相同的日期时间值
    expected = DataFrame(
        {
            "clientid": ["A", "B", "C"],
            "clientid_age": [np.timedelta64(0, "D")] * 3,
            "date": [np.datetime64("2017-02-01 00:00:00")] * 3,
        }
    ).set_index("clientid")

    # 断言确保 "result" DataFrame 与 "expected" DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_apply_groupby_datetimeindex():
    # GH 26182
    # 在具有 DatetimeIndex 的 DataFrame 上进行 groupby apply 失败

    # 创建包含数据的二维列表
    data = [["A", 10], ["B", 20], ["B", 30], ["C", 40], ["C", 50]]

    # 创建 DataFrame，列名分别为 "Name" 和 "Value"，索引为从 "2020-09-01" 到 "2020-09-05" 的日期范围
    df = DataFrame(
        data, columns=["Name", "Value"], index=pd.date_range("2020-09-01", "2020-09-05")
    )

    # 按 "Name" 列分组并对 "Value" 列求和
    result = df.groupby("Name").sum()

    # 创建预期的 DataFrame，包含 "Name" 列为 ["A", "B", "C"]，"Value" 列为 [10, 50, 90]
    expected = DataFrame({"Name": ["A", "B", "C"], "Value": [10, 50, 90]})
    expected.set_index("Name", inplace=True)

    # 断言确保 "result" DataFrame 与 "expected" DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_time_field_bug():
    # 测试针对 GH 11324 的错误修复
    # 当 group-by DataFrame 中的非关键字段包含未在 apply 函数返回的时间字段时，会引发异常

    # 创建包含两列的 DataFrame，其中 "a" 列为整数 1，"b" 列为包含 10 个当前日期时间的列表
    df = DataFrame({"a": 1, "b": [datetime.now() for nn in range(10)]})

    # 定义一个不返回日期的函数
    def func_with_no_date(batch):
        return Series({"c": 2})

    # 定义一个返回日期的函数
    def func_with_date(batch):
        return Series({"b": datetime(2015, 1, 1), "c": 2})

    # 设置警告信息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    # 断言确保以下代码块引发 DeprecationWarning 警告，并且警告消息与 "msg" 变量匹配
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 "a" 列进行分组，并应用 func_with_no_date 函数
        dfg_no_conversion = df.groupby(by=["a"]).apply(func_with_no_date)

    # 创建预期的 DataFrame，包含 "c" 列为整数 2，索引为 [1]
    dfg_no_conversion_expected = DataFrame({"c": 2}, index=[1])
    dfg_no_conversion_expected.index.name = "a"

    # 断言确保 "dfg_no_conversion" DataFrame 与 "dfg_no_conversion_expected" DataFrame 相等
    tm.assert_frame_equal(dfg_no_conversion, dfg_no_conversion_expected)

    # 断言确保以下代码块引发 DeprecationWarning 警告，并且警告消息与 "msg" 变量匹配
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 "a" 列进行分组，并应用 func_with_date 函数
        dfg_conversion = df.groupby(by=["a"]).apply(func_with_date)

    # 创建预期的 DataFrame，包含 "b" 列为 Timestamp('2015-01-01 00:00:00')，"c" 列为整数 2，索引为 [1]
    dfg_conversion_expected = DataFrame(
        {"b": pd.Timestamp(2015, 1, 1), "c": 2}, index=[1]
    )
    dfg_conversion_expected.index.name = "a"

    # 断言确保 "dfg_conversion" DataFrame 与 "dfg_conversion_expected" DataFrame 相等
    tm.assert_frame_equal(dfg_conversion, dfg_conversion_expected)


def test_gb_apply_list_of_unequal_len_arrays():
    # GH1738
    # 测试对不等长度数组列表进行 groupby apply
    # 创建一个 DataFrame，包含四列数据：group1、group2、weight、value
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b", "b", "b", "a", "a", "a", "b", "b", "b"],
            "group2": ["c", "c", "d", "d", "d", "e", "c", "c", "d", "d", "d", "e"],
            "weight": [1.1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 1, 2],
            "value": [7.1, 8, 9, 10, 11, 12, 8, 7, 6, 5, 4, 3],
        }
    )
    # 将列 'group1' 和 'group2' 设为索引
    df = df.set_index(["group1", "group2"])
    # 根据多级索引 'group1' 和 'group2' 对 DataFrame 进行分组
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)

    # 定义一个函数 noddy，接受 value 和 weight 作为参数，返回经处理的数组
    def noddy(value, weight):
        # 将 value 与 weight 相乘后生成的数组重复三次，存入 out
        out = np.array(value * weight).repeat(3)
        return out

    # 匿名函数传入 DataFrameGroupBy 对象 df_grouped，对每个分组应用 noddy 函数
    # 由于 noddy 函数返回的数组长度可能不相等，因此需要处理不等长数组的情况
    df_grouped.apply(lambda x: noddy(x.value, x.weight))
# Tests to make sure no errors if apply function returns all None
# values. Issue 9684.
def test_groupby_apply_all_none():
    # 创建一个测试用的DataFrame，包含两列：'groups'和'random_vars'
    test_df = DataFrame({"groups": [0, 0, 1, 1], "random_vars": [8, 7, 4, 5]})

    # 定义一个空的测试函数test_func，没有实际操作
    def test_func(x):
        pass

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器确保出现DeprecationWarning警告，匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按'groups'列进行分组，并对每组应用test_func函数
        result = test_df.groupby("groups").apply(test_func)
    
    # 期望的结果是一个空的DataFrame，列与test_df相同
    expected = DataFrame(columns=test_df.columns)
    # 将expected的数据类型与test_df相匹配
    expected = expected.astype(test_df.dtypes)
    # 使用assert_frame_equal函数检查result与expected是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "in_data, out_idx, out_data",
    [
        # 第一个参数组合的测试数据
        [
            {"groups": [1, 1, 1, 2], "vars": [0, 1, 2, 3]},
            [[1, 1], [0, 2]],
            {"groups": [1, 1], "vars": [0, 2]},
        ],
        # 第二个参数组合的测试数据
        [
            {"groups": [1, 2, 2, 2], "vars": [0, 1, 2, 3]},
            [[2, 2], [1, 3]],
            {"groups": [2, 2], "vars": [1, 3]},
        ],
    ],
)
def test_groupby_apply_none_first(in_data, out_idx, out_data):
    # GH 12824. Tests if apply returns None first.
    # 创建一个测试用的DataFrame，使用输入的数据in_data
    test_df1 = DataFrame(in_data)

    # 定义一个测试函数test_func，根据行数返回数据或None
    def test_func(x):
        if x.shape[0] < 2:
            return None
        return x.iloc[[0, -1]]

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器确保出现DeprecationWarning警告，匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按'groups'列进行分组，并对每组应用test_func函数
        result1 = test_df1.groupby("groups").apply(test_func)
    
    # 根据输出数据设置索引
    index1 = MultiIndex.from_arrays(out_idx, names=["groups", None])
    # 创建期望的DataFrame，使用输出数据out_data和设置的索引
    expected1 = DataFrame(out_data, index=index1)
    # 使用assert_frame_equal函数检查result1与expected1是否相等
    tm.assert_frame_equal(result1, expected1)


def test_groupby_apply_return_empty_chunk():
    # GH 22221: apply filter which returns some empty groups
    # 创建一个测试用的DataFrame，包含'value'和'group'两列
    df = DataFrame({"value": [0, 1], "group": ["filled", "empty"]})
    # 按'group'列对DataFrame进行分组
    groups = df.groupby("group")
    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器确保出现DeprecationWarning警告，匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对每个分组应用lambda函数，过滤'value'列值不为1的数据
        result = groups.apply(lambda group: group[group.value != 1]["value"])
    
    # 创建期望的Series对象，包含输出数据和索引
    expected = Series(
        [0],
        name="value",
        index=MultiIndex.from_product(
            [["empty", "filled"], [0]], names=["group", None]
        ).drop("empty"),
    )
    # 使用assert_series_equal函数检查result与expected是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("meth", ["apply", "transform"])
def test_apply_with_mixed_types(meth):
    # gh-20949
    # 创建一个测试用的DataFrame，包含'A', 'B', 'C'三列
    df = DataFrame({"A": "a a b".split(), "B": [1, 2, 3], "C": [4, 6, 5]})
    # 对DataFrame按'A'列进行分组，不添加分组键
    g = df.groupby("A", group_keys=False)

    # 调用g对象的apply或transform方法，对每组数据进行操作
    result = getattr(g, meth)(lambda x: x / x.sum())
    # 创建期望的DataFrame，包含根据分组后的计算结果
    expected = DataFrame({"B": [1 / 3.0, 2 / 3.0, 1], "C": [0.4, 0.6, 1.0]})
    # 使用assert_frame_equal函数检查result与expected是否相等
    tm.assert_frame_equal(result, expected)


def test_func_returns_object():
    # GH 28652
    # 创建一个测试用的DataFrame，包含'a'列，使用Index作为索引
    df = DataFrame({"a": [1, 2]}, index=Index([1, 2]))
    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用上下文管理器确保出现DeprecationWarning警告，匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按'a'列进行分组，应用lambda函数返回索引
        result = df.groupby("a").apply(lambda g: g.index)
    # 创建一个 Series 对象，其中包含两个 Index 对象作为数据，以及一个指定名称的索引
    expected = Series([Index([1]), Index([2])], index=Index([1, 2], name="a"))
    
    # 使用测试框架中的函数比较 result 和 expected 对象是否相等，如果不相等则会引发异常
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "group_column_dtlike",
    [datetime.today(), datetime.today().date(), datetime.today().time()],
)
# 定义测试函数 test_apply_datetime_issue，用于测试处理日期时间对象时的特定问题
def test_apply_datetime_issue(group_column_dtlike, using_infer_string):
    # GH-28247
    # 当 DataFrame 中的列是日期时间对象，并且列标签与标准整数值（范围为 len(num_columns)）不同时，
    # 在 groupby-apply 操作时会抛出错误

    # 创建包含一行数据的 DataFrame，其中包含列 'a' 和给定的日期时间对象
    df = DataFrame({"a": ["foo"], "b": [group_column_dtlike]})
    # 设置警告消息，用于验证是否产生特定的 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 进行按 'a' 列分组，并应用 lambda 函数以返回一个 Series，索引为 [42]
        result = df.groupby("a").apply(lambda x: Series(["spam"], index=[42]))

    # 根据 using_infer_string 的值确定预期的 DataFrame 的数据类型
    dtype = "string" if using_infer_string else "object"
    # 创建预期的 DataFrame 对象，包含单行数据 "spam"，索引为 ['foo']，列标签为 [42]
    expected = DataFrame(["spam"], Index(["foo"], dtype=dtype, name="a"), columns=[42])
    # 验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_apply_series_return_dataframe_groups，用于测试应用 Series 返回 DataFrame 组的情况
def test_apply_series_return_dataframe_groups():
    # GH 10078
    # 创建包含日期时间和用户代理信息的 DataFrame tdf
    tdf = DataFrame(
        {
            "day": {
                0: pd.Timestamp("2015-02-24 00:00:00"),
                1: pd.Timestamp("2015-02-24 00:00:00"),
                2: pd.Timestamp("2015-02-24 00:00:00"),
                3: pd.Timestamp("2015-02-24 00:00:00"),
                4: pd.Timestamp("2015-02-24 00:00:00"),
            },
            "userAgent": {
                0: "some UA string",
                1: "some UA string",
                2: "some UA string",
                3: "another UA string",
                4: "some UA string",
            },
            "userId": {
                0: "17661101",
                1: "17661101",
                2: "17661101",
                3: "17661101",
                4: "17661101",
            },
        }
    )

    # 定义函数 most_common_values，用于返回 DataFrame 中每列的最常见值构成的 Series
    def most_common_values(df):
        return Series({c: s.value_counts().index[0] for c, s in df.items()})

    # 设置警告消息，用于验证是否产生特定的 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 tdf 按 'day' 列分组，并应用 most_common_values 函数，再取其 "userId" 列
        result = tdf.groupby("day").apply(most_common_values)["userId"]
    # 创建预期的 Series 对象，包含值 "17661101"，索引为 DatetimeIndex(['2015-02-24'], name='day')
    expected = Series(
        ["17661101"], index=pd.DatetimeIndex(["2015-02-24"], name="day"), name="userId"
    )
    # 验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("category", [False, True])
# 定义测试函数 test_apply_multi_level_name，用于测试多级名称时的问题
def test_apply_multi_level_name(category):
    # https://github.com/pandas-dev/pandas/issues/31068
    # 创建列表 b，用于构造 DataFrame df
    b = [1, 2] * 5
    if category:
        # 如果 category 为 True，则将 b 转换为分类类型，预期的索引为 CategoricalIndex([1, 2, 3], categories=[1, 2, 3], name="B")
        b = pd.Categorical(b, categories=[1, 2, 3])
        expected_index = pd.CategoricalIndex([1, 2, 3], categories=[1, 2, 3], name="B")
        expected_values = [20, 25, 0]
    else:
        # 如果 category 为 False，则预期的索引为 Index([1, 2], name="B")
        expected_index = Index([1, 2], name="B")
        expected_values = [20, 25]
    # 创建预期的 DataFrame 对象，包含列 "C" 和 "D"，索引为 expected_index，数据为 expected_values 的复制
    expected = DataFrame(
        {"C": expected_values, "D": expected_values}, index=expected_index
    )

    # 创建 DataFrame df，包含列 "A", "B", "C", "D"，并以 "A", "B" 列作为索引
    df = DataFrame(
        {"A": np.arange(10), "B": b, "C": list(range(10)), "D": list(range(10))}
    ).set_index(["A", "B"])
    # 对 df 按 'B' 列分组，应用 lambda 函数以对每个组求和
    result = df.groupby("B", observed=False).apply(lambda x: x.sum())
    # 使用测试框架的函数来比较两个数据框是否相等，并断言它们相等
    tm.assert_frame_equal(result, expected)
    
    # 断言数据框的索引名称是否为 ["A", "B"]，如果不是则抛出 AssertionError
    assert df.index.names == ["A", "B"]
# 测试函数：测试DataFrameGroupBy对象上的apply方法对datetime结果的数据类型
def test_groupby_apply_datetime_result_dtypes(using_infer_string):
    # 创建一个DataFrame，包含多条记录，每条记录有时间戳、颜色、心情、强度和分数等字段
    data = DataFrame.from_records(
        [
            (pd.Timestamp(2016, 1, 1), "red", "dark", 1, "8"),
            (pd.Timestamp(2015, 1, 1), "green", "stormy", 2, "9"),
            (pd.Timestamp(2014, 1, 1), "blue", "bright", 3, "10"),
            (pd.Timestamp(2013, 1, 1), "blue", "calm", 4, "potato"),
        ],
        columns=["observation", "color", "mood", "intensity", "score"],
    )
    # 设置警告消息，用于验证DataFrameGroupBy.apply是否在分组列上操作
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用assert_produces_warning上下文，验证是否会产生DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对数据按颜色分组，并对每组应用lambda函数，获取第一行数据的数据类型
        result = data.groupby("color").apply(lambda g: g.iloc[0]).dtypes
    # 根据使用的infer_string参数，确定预期的数据类型
    dtype = "string" if using_infer_string else object
    # 创建预期的Series，包含每列数据的预期数据类型
    expected = Series(
        [np.dtype("datetime64[us]"), dtype, dtype, np.int64, dtype],
        index=["observation", "color", "mood", "intensity", "score"],
    )
    # 验证实际结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


# 参数化测试：测试不同类型的索引的情况下，对DataFrameGroupBy对象应用lambda函数的结果
@pytest.mark.parametrize(
    "index",
    [
        pd.CategoricalIndex(list("abc")),
        pd.interval_range(0, 3),
        pd.period_range("2020", periods=3, freq="D"),
        MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ],
)
def test_apply_index_has_complex_internals(index):
    # 创建一个DataFrame，包含组和值两列，索引为给定的index
    df = DataFrame({"group": [1, 1, 2], "value": [0, 1, 0]}, index=index)
    # 设置警告消息，用于验证DataFrameGroupBy.apply是否在分组列上操作
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用assert_produces_warning上下文，验证是否会产生DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对数据按组分组，并对每组应用lambda函数，获取每组的所有数据
        result = df.groupby("group", group_keys=False).apply(lambda x: x)
    # 验证实际结果与原始DataFrame是否一致
    tm.assert_frame_equal(result, df)


# 参数化测试：测试apply函数返回不是pandas对象或标量的情况
@pytest.mark.parametrize(
    "function, expected_values",
    [
        (lambda x: x.index.to_list(), [[0, 1], [2, 3]]),
        (lambda x: set(x.index.to_list()), [{0, 1}, {2, 3}]),
        (lambda x: tuple(x.index.to_list()), [(0, 1), (2, 3)]),
        (
            lambda x: dict(enumerate(x.index.to_list())),
            [{0: 0, 1: 1}, {0: 2, 1: 3}],
        ),
        (
            lambda x: [{n: i} for (n, i) in enumerate(x.index.to_list())],
            [[{0: 0}, {1: 1}], [{0: 2}, {1: 3}]],
        ),
    ],
)
def test_apply_function_returns_non_pandas_non_scalar(function, expected_values):
    # 创建一个DataFrame，只有一列分组信息，名为groups
    df = DataFrame(["A", "A", "B", "B"], columns=["groups"])
    # 设置警告消息，用于验证DataFrameGroupBy.apply是否在分组列上操作
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用assert_produces_warning上下文，验证是否会产生DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对数据按groups列分组，并对每组应用给定的函数
        result = df.groupby("groups").apply(function)
    # 创建预期的Series，包含每组数据的预期值
    expected = Series(expected_values, index=Index(["A", "B"], name="groups"))
    # 验证实际结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


# 测试函数：测试apply函数返回numpy数组的情况
def test_apply_function_returns_numpy_array():
    # 定义一个函数，用于处理每个分组，返回B列的展平后的numpy数组
    def fct(group):
        return group["B"].values.flatten()

    # 创建一个DataFrame，包含A列和B列，B列包括一些NaN值
    df = DataFrame({"A": ["a", "a", "b", "none"], "B": [1, 2, 3, np.nan]})

    # 设置警告消息，用于验证DataFrameGroupBy.apply是否在分组列上操作
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用assert_produces_warning上下文，验证是否会产生DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对数据按A列分组，并对每组应用定义的函数fct
        result = df.groupby("A").apply(fct)
    # 创建一个预期的 Series 对象，包含特定的数据和索引
    expected = Series(
        [[1.0, 2.0], [3.0], [np.nan]],  # Series 包含的数据，这里有三个列表作为数据
        index=Index(["a", "b", "none"], name="A")  # Series 的索引，名称为"A"
    )
    # 使用测试工具包中的方法，验证 result 和 expected Series 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("function", [lambda gr: gr.index, lambda gr: gr.index + 1 - 1])
# 使用 pytest 的 parametrize 装饰器，为函数 test_apply_function_index_return 参数化两个 lambda 函数
def test_apply_function_index_return(function):
    # GH: 22541
    # 创建一个 DataFrame 包含整数列 "id"
    df = DataFrame([1, 2, 2, 2, 1, 2, 3, 1, 3, 1], columns=["id"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在执行 groupby.apply 操作时会产生 DeprecationWarning 警告，并匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 根据 "id" 列进行分组，并对每组应用传入的 function 函数
        result = df.groupby("id").apply(function)
    # 创建预期的 Series 对象，包含索引和预期的索引值
    expected = Series(
        [Index([0, 4, 7, 9]), Index([1, 2, 3, 5]), Index([6, 8])],
        index=Index([1, 2, 3], name="id"),
    )
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)


def test_apply_function_with_indexing_return_column():
    # GH#7002, GH#41480, GH#49256
    # 创建包含两列 "foo1" 和 "foo2" 的 DataFrame
    df = DataFrame(
        {
            "foo1": ["one", "two", "two", "three", "one", "two"],
            "foo2": [1, 2, 4, 4, 5, 6],
        }
    )
    # 对 DataFrame 根据 "foo1" 列进行分组，并对每组应用 lambda 函数计算均值
    result = df.groupby("foo1", as_index=False).apply(lambda x: x.mean())
    # 创建预期的 DataFrame 对象，包含 "foo1" 和 "foo2" 列以及预期的值
    expected = DataFrame(
        {
            "foo1": ["one", "three", "two"],
            "foo2": [3.0, 4.0, 4.0],
        }
    )
    # 断言结果 DataFrame 与预期 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "udf",
    [lambda x: x.copy(), lambda x: x.copy().rename(lambda y: y + 1)],
)
@pytest.mark.parametrize("group_keys", [True, False])
# 使用 pytest 的 parametrize 装饰器，为函数 test_apply_result_type 参数化两个 lambda 函数和一个布尔值
def test_apply_result_type(group_keys, udf):
    # https://github.com/pandas-dev/pandas/issues/34809
    # 我们希望控制组键是否最终出现在索引中，不管 UDF 是否恰好是一个 transform 函数。
    # 创建包含两列 "A" 和 "B" 的 DataFrame
    df = DataFrame({"A": ["a", "b"], "B": [1, 2]})
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在执行 groupby.apply 操作时会产生 DeprecationWarning 警告，并匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 根据 "A" 列进行分组，并对每组应用 udf 函数
        df_result = df.groupby("A", group_keys=group_keys).apply(udf)
    # 对 DataFrame 的 "B" 列根据 "A" 列进行分组，并对每组应用 udf 函数
    series_result = df.B.groupby(df.A, group_keys=group_keys).apply(udf)

    # 根据 group_keys 的值断言结果 DataFrame 或 Series 的索引层级数
    if group_keys:
        assert df_result.index.nlevels == 2
        assert series_result.index.nlevels == 2
    else:
        assert df_result.index.nlevels == 1
        assert series_result.index.nlevels == 1


def test_result_order_group_keys_false():
    # GH 34998
    # apply 的结果顺序不应该依赖于索引是相同还是仅仅相等
    # 创建包含两列 "A" 和 "B" 的 DataFrame
    df = DataFrame({"A": [2, 1, 2], "B": [1, 2, 3]})
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在执行 groupby.apply 操作时会产生 DeprecationWarning 警告，并匹配特定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 根据 "A" 列进行分组，并对每组应用 lambda 函数返回原始的 DataFrame
        result = df.groupby("A", group_keys=False).apply(lambda x: x)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 根据 "A" 列进行分组，并对每组应用 lambda 函数返回 DataFrame 的副本
        expected = df.groupby("A", group_keys=False).apply(lambda x: x.copy())
    # 断言结果 DataFrame 与预期 DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_apply_with_timezones_aware():
    # GH: 27212
    # 创建包含日期时间索引的 DataFrame
    dates = ["2001-01-01"] * 2 + ["2001-01-02"] * 2 + ["2001-01-03"] * 2
    index_no_tz = pd.DatetimeIndex(dates)
    index_tz = pd.DatetimeIndex(dates, tz="UTC")
    df1 = DataFrame({"x": list(range(2)) * 3, "y": range(6), "t": index_no_tz})
    创建一个 DataFrame 对象 df2，其中包含三列：x 列包含两个 0 和两个 1，y 列包含 0 到 5，t 列包含 index_tz 的值。
    df2 = DataFrame({"x": list(range(2)) * 3, "y": range(6), "t": index_tz})

    设置用于匹配警告消息的字符串
    msg = "DataFrameGroupBy.apply operated on the grouping columns"

    在不产生特定警告消息的情况下，对 df1 进行分组并应用 lambda 函数，复制每个分组的 "x" 和 "y" 列。
    使用 assert_produces_warning 进行断言，期望产生 DeprecationWarning 警告，并匹配之前设置的消息。
    result1 = df1.groupby("x", group_keys=False).apply(
        lambda df: df[["x", "y"]].copy()
    )

    在不产生特定警告消息的情况下，对 df2 进行分组并应用 lambda 函数，复制每个分组的 "x" 和 "y" 列。
    使用 assert_produces_warning 进行断言，期望产生 DeprecationWarning 警告，并匹配之前设置的消息。
    result2 = df2.groupby("x", group_keys=False).apply(
        lambda df: df[["x", "y"]].copy()
    )

    断言 result1 和 result2 的内容相等。
    tm.assert_frame_equal(result1, result2)
# 测试函数，用于验证在先调用其他方法后再调用 apply() 时 apply() 的行为不变
def test_apply_is_unchanged_when_other_methods_are_called_first(reduction_func):
    # GH #34656
    # GH #34271
    # 创建一个包含列 'a', 'b', 'c' 的 DataFrame
    df = DataFrame(
        {
            "a": [99, 99, 99, 88, 88, 88],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [10, 20, 30, 40, 50, 60],
        }
    )

    # 期望的结果 DataFrame，具有 'b' 和 'c' 列的总和
    expected = DataFrame(
        {"b": [15, 6], "c": [150, 60]},
        index=Index([88, 99], name="a"),
    )

    # 检查在调用 apply() 之前没有调用其他方法时的输出
    grp = df.groupby(by="a")
    result = grp.apply(np.sum, axis=0, include_groups=False)
    tm.assert_frame_equal(result, expected)

    # 检查在调用 apply() 之前调用了其他方法时的输出
    grp = df.groupby(by="a")
    args = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn = FutureWarning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    with tm.assert_produces_warning(warn, match=msg):
        _ = getattr(grp, reduction_func)(*args)
    result = grp.apply(np.sum, axis=0, include_groups=False)
    tm.assert_frame_equal(result, expected)


# 测试函数，验证在多级索引中使用 apply() 不会将日期转换为时间戳
def test_apply_with_date_in_multiindex_does_not_convert_to_timestamp():
    # GH 29617

    # 创建一个包含列 'A', 'B', 'C' 的 DataFrame，索引为自定义的索引对象
    df = DataFrame(
        {
            "A": ["a", "a", "a", "b"],
            "B": [
                date(2020, 1, 10),
                date(2020, 1, 10),
                date(2020, 2, 10),
                date(2020, 2, 10),
            ],
            "C": [1, 2, 3, 4],
        },
        index=Index([100, 101, 102, 103], name="idx"),
    )

    # 根据 'A' 和 'B' 列进行分组
    grp = df.groupby(["A", "B"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对每个分组应用 lambda 函数，选择每组的第一行
        result = grp.apply(lambda x: x.head(1))

    # 期望的结果 DataFrame，重置索引并创建多级索引
    expected = df.iloc[[0, 2, 3]]
    expected = expected.reset_index()
    expected.index = MultiIndex.from_frame(expected[["A", "B", "idx"]])
    expected = expected.drop(columns=["idx"])

    # 断言结果与期望相等，且索引的第二级为日期类型
    tm.assert_frame_equal(result, expected)
    for val in result.index.levels[1]:
        assert type(val) is date


# 测试函数，验证在索引相同的情况下使用 apply(dropna=True/False) 的行为
def test_apply_dropna_with_indexed_same(dropna):
    # GH 38227
    # GH#43205
    # 创建一个包含列 'col', 'group' 的 DataFrame，使用自定义索引
    df = DataFrame(
        {
            "col": [1, 2, 3, 4, 5],
            "group": ["a", np.nan, np.nan, "b", "b"],
        },
        index=list("xxyxz"),
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 根据 'group' 列分组，根据 dropna 参数决定是否丢弃 NaN 值
        result = df.groupby("group", dropna=dropna, group_keys=False).apply(lambda x: x)
    # 期望的结果 DataFrame，根据 dropna 参数决定是否保留 NaN 值
    expected = df.dropna() if dropna else df.iloc[[0, 3, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)
    [
        # 第一个元素
        [
            # 布尔值 False
            False,
            # 创建 DataFrame 对象
            DataFrame(
                # 数据为二维列表
                [[1, 1, 1], [2, 2, 1]],
                # 设置列索引为 Index 对象，包括字符串 "a" 和 "b"，以及一个空名称的列
                columns=Index(["a", "b", None], dtype=object)
            ),
        ],
        # 第二个元素
        [
            # 布尔值 True
            True,
            # 创建 Series 对象
            Series(
                # 数据为列表 [1, 1]
                [1, 1],
                # 设置索引为 MultiIndex 对象，由元组列表构成，每个元组有两个值，命名为 "a" 和 "b"
                index=MultiIndex.from_tuples([(1, 1), (2, 2)], names=["a", "b"])
            ),
        ],
    ],
# GH 13217
# 创建一个测试函数，用于测试 DataFrameGroupBy 对象的 apply 方法在指定的参数条件下的行为
def test_apply_as_index_constant_lambda(as_index, expected):
    # 创建一个 DataFrame 包含三列数据
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 1, 2, 2], "c": [1, 1, 1, 1]})
    # 设定一个警告信息字符串，用于匹配 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用断言确保在 apply 方法操作分组列时触发 DeprecationWarning 警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 进行分组并应用 lambda 函数返回常数值 1
        result = df.groupby(["a", "b"], as_index=as_index).apply(lambda x: 1)
    # 使用断言确认 apply 方法的返回结果符合预期
    tm.assert_equal(result, expected)


# GH 20420
# 创建一个测试函数，测试对分组后的 DataFrame 进行 apply 操作后排序的行为
def test_sort_index_groups():
    # 创建一个 DataFrame 包含三列数据，并使用默认的行索引
    df = DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 0], "C": [1, 1, 1, 2, 2]},
        index=range(5),
    )
    # 设定一个警告信息字符串，用于匹配 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用断言确保在 apply 方法操作分组列时触发 DeprecationWarning 警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 'C' 进行分组，然后对分组后的 'A' 列数据进行排序
        result = df.groupby("C").apply(lambda x: x.A.sort_index())
    # 创建一个预期的 Series，用于与 apply 方法的结果进行比较
    expected = Series(
        range(1, 6),
        index=MultiIndex.from_tuples(
            [(1, 0), (1, 1), (1, 2), (2, 3), (2, 4)], names=["C", None]
        ),
        name="A",
    )
    # 使用断言确认 apply 方法的返回结果符合预期
    tm.assert_series_equal(result, expected)


# GH 21651
# 创建一个测试函数，测试对 DataFrame 分组后进行 apply 操作，获取分组内切片的行为
def test_positional_slice_groups_datetimelike():
    # 创建一个预期的 DataFrame，包含日期列、数值列和字母列
    expected = DataFrame(
        {
            "date": pd.date_range("2010-01-01", freq="12h", periods=5),
            "vals": range(5),
            "let": list("abcde"),
        }
    )
    # 设定一个警告信息字符串，用于匹配 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用断言确保在 apply 方法操作分组列时触发 DeprecationWarning 警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 'let' 列和日期的日期部分进行分组，禁用分组键的附加
        result = expected.groupby(
            [expected.let, expected.date.dt.date], group_keys=False
        ).apply(lambda x: x.iloc[0:])
    # 使用断言确认 apply 方法的返回结果符合预期
    tm.assert_frame_equal(result, expected)


# GH#42702
# 创建一个测试函数，测试在特定条件下对分组后的 DataFrame 进行 apply 操作
def test_groupby_apply_shape_cache_safety():
    # 创建一个包含三列数据的 DataFrame
    df = DataFrame({"A": ["a", "a", "b"], "B": [1, 2, 3], "C": [4, 6, 5]})
    # 对 DataFrame 按 'A' 列进行分组
    gb = df.groupby("A")
    # 对分组后的数据框选择 'B' 和 'C' 列，并应用 lambda 函数计算每列的最大值和最小值之差
    result = gb[["B", "C"]].apply(lambda x: x.astype(float).max() - x.min())

    # 创建一个预期的 DataFrame，包含计算结果和行索引
    expected = DataFrame(
        {"B": [1.0, 0.0], "C": [2.0, 0.0]}, index=Index(["a", "b"], name="A")
    )
    # 使用断言确认 apply 方法的返回结果符合预期
    tm.assert_frame_equal(result, expected)


# GH52444
# 创建一个测试函数，测试对 Series 分组后进行 apply 操作
def test_groupby_apply_to_series_name():
    # 创建一个包含三列数据的 DataFrame
    df = DataFrame.from_dict(
        {
            "a": ["a", "b", "a", "b"],
            "b1": ["aa", "ac", "ac", "ad"],
            "b2": ["aa", "aa", "aa", "ac"],
        }
    )
    # 对 DataFrame 按 'a' 列分组，并选择 'b1' 和 'b2' 列
    grp = df.groupby("a")[["b1", "b2"]]
    # 对分组后的数据框进行 apply 操作，将其展开并计算各类别的计数
    result = grp.apply(lambda x: x.unstack().value_counts())

    # 创建一个预期的 Series，包含计算结果和多级索引
    expected_idx = MultiIndex.from_arrays(
        arrays=[["a", "a", "b", "b", "b"], ["aa", "ac", "ac", "ad", "aa"]],
        names=["a", None],
    )
    expected = Series([3, 1, 2, 1, 1], index=expected_idx, name="count")
    # 使用断言确认 apply 方法的返回结果符合预期
    tm.assert_series_equal(result, expected)


# GH#28984
# 创建一个测试函数，测试在特定条件下对 DataFrame 分组后进行 apply 操作
def test_apply_na(dropna):
    # 创建一个包含三列数据的 DataFrame，其中 'z' 列包含 NaN 值
    df = DataFrame(
        {"grp": [1, 1, 2, 2], "y": [1, 0, 2, 5], "z": [1, 2, np.nan, np.nan]}
    )
    # 对 DataFrame 按 'grp' 列进行分组，并根据 dropna 参数决定是否丢弃 NaN 值
    dfgrp = df.groupby("grp", dropna=dropna)
    # 设定一个警告信息字符串，用于匹配 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用测试工具 `tm.assert_produces_warning` 来检查代码段是否会产生特定的 DeprecationWarning 警告，并且警告消息需要匹配变量 `msg`
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对于数据组 dfgrp，对每个组应用 lambda 函数，选取 "z" 列中最大的值对应的行
        result = dfgrp.apply(lambda grp_df: grp_df.nlargest(1, "z"))
    
    # 使用测试工具 `tm.assert_produces_warning` 来检查代码段是否会产生特定的 DeprecationWarning 警告，并且警告消息需要匹配变量 `msg`
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对于数据组 dfgrp，对每个组应用 lambda 函数，按 "z" 列降序排列并取前一行作为结果
        expected = dfgrp.apply(lambda x: x.sort_values("z", ascending=False).head(1))
    
    # 使用测试工具 `tm.assert_frame_equal` 来断言变量 result 和 expected 的值是否相等
    tm.assert_frame_equal(result, expected)
# GH#24903
def test_apply_empty_string_nan_coerce_bug():
    # 定义警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言产生特定警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 创建DataFrame对象，并进行分组操作，应用lambda函数选择每组的最后一行数据
        result = (
            DataFrame(
                {
                    "a": [1, 1, 2, 2],
                    "b": ["", "", "", ""],
                    "c": pd.to_datetime([1, 2, 3, 4], unit="s"),
                }
            )
            .groupby(["a", "b"])
            .apply(lambda df: df.iloc[-1])
        )
    # 期望的DataFrame结果，设置多级索引和列名
    expected = DataFrame(
        [[1, "", pd.to_datetime(2, unit="s")], [2, "", pd.to_datetime(4, unit="s")]],
        columns=["a", "b", "c"],
        index=MultiIndex.from_tuples([(1, ""), (2, "")], names=["a", "b"]),
    )
    # 断言result与expected是否相等
    tm.assert_frame_equal(result, expected)


# GH 44310
@pytest.mark.parametrize("index_values", [[1, 2, 3], [1.0, 2.0, 3.0]])
def test_apply_index_key_error_bug(index_values):
    # 创建DataFrame对象，设置列和索引
    result = DataFrame(
        {
            "a": ["aa", "a2", "a3"],
            "b": [1, 2, 3],
        },
        index=Index(index_values),
    )
    # 期望的DataFrame结果，设置列名和索引名
    expected = DataFrame(
        {
            "b_mean": [2.0, 3.0, 1.0],
        },
        index=Index(["a2", "a3", "aa"], name="a"),
    )
    # 定义警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言产生特定警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组结果应用lambda函数计算'b'列的均值，并构造成Series
        result = result.groupby("a").apply(
            lambda df: Series([df["b"].mean()], index=["b_mean"])
        )
    # 断言result与expected是否相等
    tm.assert_frame_equal(result, expected)


# GH 34455
@pytest.mark.parametrize(
    "arg,idx",
    [
        [
            [
                1,
                2,
                3,
            ],
            [
                0.1,
                0.3,
                0.2,
            ],
        ],
        [
            [
                1,
                2,
                3,
            ],
            [
                0.1,
                0.2,
                0.3,
            ],
        ],
        [
            [
                1,
                4,
                3,
            ],
            [
                0.1,
                0.4,
                0.2,
            ],
        ],
    ],
)
def test_apply_nonmonotonic_float_index(arg, idx):
    # 期望的DataFrame结果，设置一列数据和浮点数索引
    expected = DataFrame({"col": arg}, index=idx)
    # 定义警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言产生特定警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame应用lambda函数，并保留分组键
        result = expected.groupby("col", group_keys=False).apply(lambda x: x)
    # 断言result与expected是否相等
    tm.assert_frame_equal(result, expected)


# GH#46479
@pytest.mark.parametrize("args, kwargs", [([True], {}), ([], {"numeric_only": True})])
def test_apply_str_with_args(df, args, kwargs):
    # 根据"A"列进行分组
    gb = df.groupby("A")
    # 对分组结果应用"sum"函数，传入参数args和kwargs
    result = gb.apply("sum", *args, **kwargs)
    # 期望的DataFrame结果，调用gb对象的sum方法
    expected = gb.sum(numeric_only=True)
    # 断言result与expected是否相等
    tm.assert_frame_equal(result, expected)


# GH 46369
@pytest.mark.parametrize("name", ["some_name", None])
def test_result_name_when_one_group(name):
    # 待添加...
    # 创建一个名为 `ser` 的 Pandas Series 对象，包含整数 1 和 2，并命名为 `name`
    ser = Series([1, 2], name=name)
    
    # 对 `ser` 进行分组操作，按照列标签 ["a", "a"] 进行分组，但不生成分组键，然后对每个分组应用一个函数，
    # 这里的 lambda 函数是返回每个分组本身，最终得到一个新的 Series 对象 `result`
    result = ser.groupby(["a", "a"], group_keys=False).apply(lambda x: x)
    
    # 创建一个期望的 Pandas Series 对象 `expected`，其中包含整数 1 和 2，并命名为 `name`
    expected = Series([1, 2], name=name)
    
    # 使用 Pandas 的测试工具 `assert_series_equal` 检查 `result` 和 `expected` 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "method, op",
    [
        ("apply", lambda gb: gb.values[-1]),  # 定义方法和操作，操作为获取每组最后一个值
        ("apply", lambda gb: gb["b"].iloc[0]),  # 定义方法和操作，操作为获取每组'b'列的第一个值
        ("agg", "skew"),  # 使用agg方法进行偏斜(skew)聚合
        ("agg", "prod"),  # 使用agg方法进行乘积(prod)聚合
        ("agg", "sum"),   # 使用agg方法进行求和(sum)聚合
    ],
)
def test_empty_df(method, op):
    # GH 47985
    # 创建一个空的DataFrame，包含'a'和'b'两列
    empty_df = DataFrame({"a": [], "b": []})
    # 对空DataFrame按照'a'列分组，保留分组键
    gb = empty_df.groupby("a", group_keys=True)
    # 获取分组后'b'列的引用
    group = getattr(gb, "b")

    # 对分组对象应用指定的方法和操作
    result = getattr(group, method)(op)
    # 创建预期结果的Series，其中数据为空，数据类型为float64，索引为空的float64类型
    expected = Series(
        [], name="b", dtype="float64", index=Index([], dtype="float64", name="a")
    )

    # 断言结果与预期一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("include_groups", [True, False])
def test_include_groups(include_groups):
    # GH#7155
    # 创建一个DataFrame，包含'a'和'b'两列
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    # 对DataFrame按照'a'列分组
    gb = df.groupby("a")
    # 如果include_groups为True，设置DeprecationWarning警告，否则为None
    warn = DeprecationWarning if include_groups else None
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在执行时产生指定的警告信息
    with tm.assert_produces_warning(warn, match=msg):
        # 对分组对象应用lambda函数，对每组进行求和操作，根据include_groups参数决定是否包含分组列
        result = gb.apply(lambda x: x.sum(), include_groups=include_groups)
    # 创建预期结果的DataFrame
    expected = DataFrame({"a": [2, 2], "b": [7, 5]}, index=Index([1, 2], name="a"))
    if not include_groups:
        expected = expected[["b"]]
    # 断言结果与预期一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func, value", [(max, 2), (min, 1), (sum, 3)])
def test_builtins_apply(func, value):
    # GH#8155, GH#53974
    # 内置函数作为例如sum(group)的操作，对分组执行求和操作，不包含分组列
    df = DataFrame({0: [1, 1, 2], 1: [3, 4, 5], 2: [3, 4, 5]})
    # 对DataFrame按照第0列分组
    gb = df.groupby(0)
    # 对分组对象应用指定的内置函数，exclude_groups参数设置为False
    result = gb.apply(func, include_groups=False)

    # 创建预期结果的Series
    expected = Series([value, value], index=Index([1, 2], name=0))
    # 断言结果与预期一致
    tm.assert_series_equal(result, expected)


def test_inconsistent_return_type():
    # GH5592
    # 不一致的返回类型
    # 创建一个DataFrame，包含'A', 'B', 'C'三列数据
    df = DataFrame(
        {
            "A": ["Tiger", "Tiger", "Tiger", "Lamb", "Lamb", "Pony", "Pony"],
            "B": Series(np.arange(7), dtype="int64"),
            "C": pd.date_range("20130101", periods=7),
        }
    )

    # 定义一个函数f_0，返回每组的第一个值
    def f_0(grp):
        return grp.iloc[0]

    # 创建预期结果的DataFrame
    expected = df.groupby("A").first()[["B"]]
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在执行时产生指定的警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象应用f_0函数，获取每组的第一个值，然后选择'B'列
        result = df.groupby("A").apply(f_0)[["B"]]
    # 断言结果与预期一致
    tm.assert_frame_equal(result, expected)

    # 定义一个函数f_1，如果组名为'Tiger'，返回None，否则返回每组的第一个值
    def f_1(grp):
        if grp.name == "Tiger":
            return None
        return grp.iloc[0]

    # 断言在执行时产生指定的警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象应用f_1函数，如果组名为'Tiger'，返回NaN，然后选择'B'列
        result = df.groupby("A").apply(f_1)[["B"]]
    # 创建预期结果的DataFrame，将'Tiger'组的'B'列值设置为NaN
    e = expected.copy()
    e.loc["Tiger"] = np.nan
    # 断言结果与预期一致
    tm.assert_frame_equal(result, e)

    # 定义一个函数f_2，如果组名为'Pony'，返回None，否则返回每组的第一个值
    def f_2(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0]

    # 断言在执行时产生指定的警告信息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象应用f_2函数，如果组名为'Pony'，返回NaN，然后选择'B'列
        result = df.groupby("A").apply(f_2)[["B"]]
    # 复制期望的 DataFrame，并将"Pony"行的值设为 NaN
    e = expected.copy()
    e.loc["Pony"] = np.nan
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 e 的内容是否相等
    tm.assert_frame_equal(result, e)

    # 重新审视 5592，并使用日期时间数据
    def f_3(grp):
        # 如果组的名称为"Pony"，返回 None
        if grp.name == "Pony":
            return None
        # 否则返回该组的第一个元素
        return grp.iloc[0]

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 pytest 的 assert_produces_warning 函数检查是否产生 DeprecationWarning，并匹配特定消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按"A"列分组，并应用函数 f_3，选择结果中的["C"]列
        result = df.groupby("A").apply(f_3)[["C"]]
    # 期望的结果为按"A"列分组后的第一个元素的["C"]列
    e = df.groupby("A").first()[["C"]]
    # 将"Pony"行的值设为 pd.NaT (NaT 表示不确定的时间)
    e.loc["Pony"] = pd.NaT
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 e 的内容是否相等
    tm.assert_frame_equal(result, e)

    # 标量输出
    def f_4(grp):
        # 如果组的名称为"Pony"，返回 None
        if grp.name == "Pony":
            return None
        # 否则返回该组的第一个元素的["C"]列的值
        return grp.iloc[0].loc["C"]

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 pytest 的 assert_produces_warning 函数检查是否产生 DeprecationWarning，并匹配特定消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按"A"列分组，并应用函数 f_4
        result = df.groupby("A").apply(f_4)
    # 期望的结果为按"A"列分组后的第一个元素的"C"列的值
    e = df.groupby("A").first()["C"].copy()
    # 将"Pony"行的值设为 NaN
    e.loc["Pony"] = np.nan
    # 将结果的名称设为 None
    e.name = None
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 e 的内容是否相等
    tm.assert_series_equal(result, e)
```