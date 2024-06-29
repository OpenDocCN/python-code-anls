# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_counting.py`

```
from itertools import product
from string import ascii_lowercase

import numpy as np  # 导入 NumPy 库，并用 np 别名表示
import pytest  # 导入 Pytest 测试框架

from pandas import (  # 从 Pandas 库中导入以下对象
    DataFrame,  # 数据框对象
    Index,  # 索引对象
    MultiIndex,  # 多重索引对象
    Period,  # 时期对象
    Series,  # 系列对象
    Timedelta,  # 时间增量对象
    Timestamp,  # 时间戳对象
    date_range,  # 日期范围生成函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


class TestCounting:
    def test_cumcount(self):
        df = DataFrame([["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"])  # 创建包含单列 'A' 的数据框
        g = df.groupby("A")  # 按列 'A' 分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series([0, 1, 2, 0, 3])  # 预期的累计计数结果

        tm.assert_series_equal(expected, g.cumcount())  # 使用 Pandas 测试模块比较实际和预期的累计计数结果
        tm.assert_series_equal(expected, sg.cumcount())  # 同上，但针对分组后的列 'A'

    def test_cumcount_empty(self):
        ge = DataFrame().groupby(level=0)  # 创建空数据框并按第一级别分组
        se = Series(dtype=object).groupby(level=0)  # 创建空系列并按第一级别分组

        # edge case, as this is usually considered float
        e = Series(dtype="int64")  # 创建空整型系列

        tm.assert_series_equal(e, ge.cumcount())  # 使用 Pandas 测试模块比较实际和预期的累计计数结果
        tm.assert_series_equal(e, se.cumcount())  # 同上，但针对系列对象

    def test_cumcount_dupe_index(self):
        df = DataFrame(
            [["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=[0] * 5
        )  # 创建包含单列 'A' 的数据框，并指定重复索引
        g = df.groupby("A")  # 按列 'A' 分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series([0, 1, 2, 0, 3], index=[0] * 5)  # 预期的累计计数结果，使用重复索引

        tm.assert_series_equal(expected, g.cumcount())  # 使用 Pandas 测试模块比较实际和预期的累计计数结果
        tm.assert_series_equal(expected, sg.cumcount())  # 同上，但针对分组后的列 'A'

    def test_cumcount_mi(self):
        mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])  # 创建多重索引
        df = DataFrame([["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=mi)  # 使用多重索引创建数据框
        g = df.groupby("A")  # 按列 'A' 分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series([0, 1, 2, 0, 3], index=mi)  # 预期的累计计数结果，使用多重索引

        tm.assert_series_equal(expected, g.cumcount())  # 使用 Pandas 测试模块比较实际和预期的累计计数结果
        tm.assert_series_equal(expected, sg.cumcount())  # 同上，但针对分组后的列 'A'

    def test_cumcount_groupby_not_col(self):
        df = DataFrame(
            [["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=[0] * 5
        )  # 创建包含单列 'A' 的数据框，并指定重复索引
        g = df.groupby([0, 0, 0, 1, 0])  # 按指定的非列索引进行分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series([0, 1, 2, 0, 3], index=[0] * 5)  # 预期的累计计数结果，使用重复索引

        tm.assert_series_equal(expected, g.cumcount())  # 使用 Pandas 测试模块比较实际和预期的累计计数结果
        tm.assert_series_equal(expected, sg.cumcount())  # 同上，但针对分组后的列 'A'

    def test_ngroup(self):
        df = DataFrame({"A": list("aaaba")})  # 创建包含单列 'A' 的数据框
        g = df.groupby("A")  # 按列 'A' 分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series([0, 0, 0, 1, 0])  # 预期的分组序号结果

        tm.assert_series_equal(expected, g.ngroup())  # 使用 Pandas 测试模块比较实际和预期的分组序号结果
        tm.assert_series_equal(expected, sg.ngroup())  # 同上，但针对分组后的列 'A'

    def test_ngroup_distinct(self):
        df = DataFrame({"A": list("abcde")})  # 创建包含单列 'A' 的数据框
        g = df.groupby("A")  # 按列 'A' 分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series(range(5), dtype="int64")  # 预期的分组序号结果，使用整型数据类型

        tm.assert_series_equal(expected, g.ngroup())  # 使用 Pandas 测试模块比较实际和预期的分组序号结果
        tm.assert_series_equal(expected, sg.ngroup())  # 同上，但针对分组后的列 'A'

    def test_ngroup_one_group(self):
        df = DataFrame({"A": [0] * 5})  # 创建包含单列 'A' 的数据框，所有值为0
        g = df.groupby("A")  # 按列 'A' 分组
        sg = g.A  # 获取分组后的列 'A'

        expected = Series([0] * 5)  # 预期的分组序号结果，所有值为0

        tm.assert_series_equal(expected, g.ngroup())  # 使用 Pandas 测试模块比较实际和预期的分组序号结果
        tm.assert_series_equal(expected, sg.ngroup())  # 同上，但针对分组后的列 'A'
    def test_ngroup_empty(self):
        # 创建一个空的 DataFrame，并按照第一级索引进行分组
        ge = DataFrame().groupby(level=0)
        # 创建一个空的 Series，数据类型为 object，并按照第一级索引进行分组
        se = Series(dtype=object).groupby(level=0)

        # 创建一个空的 Series，数据类型通常为 float，但这里指定为 int64 类型
        # 边界情况，因为通常情况下会被认为是 float 类型
        e = Series(dtype="int64")

        # 断言两个 Series 相等：e 和 ge.ngroup()
        tm.assert_series_equal(e, ge.ngroup())
        # 断言两个 Series 相等：e 和 se.ngroup()
        tm.assert_series_equal(e, se.ngroup())

    def test_ngroup_series_matches_frame(self):
        # 创建一个包含列 A 的 DataFrame，值为 ['a', 'a', 'a', 'b', 'a']
        df = DataFrame({"A": list("aaaba")})
        # 创建一个 Series，值为 ['a', 'a', 'a', 'b', 'a']
        s = Series(list("aaaba"))

        # 断言两个 Series 相等：df.groupby(s).ngroup() 和 s.groupby(s).ngroup()
        tm.assert_series_equal(df.groupby(s).ngroup(), s.groupby(s).ngroup())

    def test_ngroup_dupe_index(self):
        # 创建一个包含列 A 的 DataFrame，值为 ['a', 'a', 'a', 'b', 'a']，索引为 [0, 0, 0, 1, 0]
        df = DataFrame({"A": list("aaaba")}, index=[0] * 5)
        # 按列 A 进行分组
        g = df.groupby("A")
        # 从分组结果中获取列 A
        sg = g.A

        # 期望的结果 Series，值为 [0, 0, 0, 1, 0]，索引为 [0, 0, 0, 1, 0]
        expected = Series([0, 0, 0, 1, 0], index=[0] * 5)

        # 断言两个 Series 相等：g.ngroup() 和期望的结果 expected
        tm.assert_series_equal(expected, g.ngroup())
        # 断言两个 Series 相等：sg.ngroup() 和期望的结果 expected
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_mi(self):
        # 创建一个 MultiIndex，值为 [[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]]，并包含列 A 的 DataFrame
        mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])
        df = DataFrame({"A": list("aaaba")}, index=mi)
        # 按列 A 进行分组
        g = df.groupby("A")
        # 从分组结果中获取列 A
        sg = g.A
        # 期望的结果 Series，值为 [0, 0, 0, 1, 0]，索引为 mi
        expected = Series([0, 0, 0, 1, 0], index=mi)

        # 断言两个 Series 相等：g.ngroup() 和期望的结果 expected
        tm.assert_series_equal(expected, g.ngroup())
        # 断言两个 Series 相等：sg.ngroup() 和期望的结果 expected
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_groupby_not_col(self):
        # 创建一个包含列 A 的 DataFrame，值为 ['a', 'a', 'a', 'b', 'a']，索引为 [0, 0, 0, 1, 0]
        df = DataFrame({"A": list("aaaba")}, index=[0] * 5)
        # 按 [0, 0, 0, 1, 0] 这个列表进行分组
        g = df.groupby([0, 0, 0, 1, 0])
        # 从分组结果中获取列 A
        sg = g.A

        # 期望的结果 Series，值为 [0, 0, 0, 1, 0]，索引为 [0, 0, 0, 1, 0]
        expected = Series([0, 0, 0, 1, 0], index=[0] * 5)

        # 断言两个 Series 相等：g.ngroup() 和期望的结果 expected
        tm.assert_series_equal(expected, g.ngroup())
        # 断言两个 Series 相等：sg.ngroup() 和期望的结果 expected
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_descending(self):
        # 创建一个 DataFrame，包含一列 A，值为 ["a", "a", "b", "a", "b"]
        df = DataFrame(["a", "a", "b", "a", "b"], columns=["A"])
        # 按列 A 进行分组
        g = df.groupby(["A"])

        # 升序排列的结果 Series，值为 [0, 0, 1, 0, 1]
        ascending = Series([0, 0, 1, 0, 1])
        # 降序排列的结果 Series，值为 [1, 1, 0, 1, 0]
        descending = Series([1, 1, 0, 1, 0])

        # 断言两个 Series 相等：(g.ngroups - 1) - ascending 和期望的结果 descending
        tm.assert_series_equal(descending, (g.ngroups - 1) - ascending)
        # 断言两个 Series 相等：g.ngroup(ascending=True) 和期望的结果 ascending
        tm.assert_series_equal(ascending, g.ngroup(ascending=True))
        # 断言两个 Series 相等：g.ngroup(ascending=False) 和期望的结果 descending
        tm.assert_series_equal(descending, g.ngroup(ascending=False))

    def test_ngroup_matches_cumcount(self):
        # 创建一个包含两列 A 和 X 的 DataFrame，值为 [["a", "x"], ["a", "y"], ["b", "x"], ["a", "x"], ["b", "y"]]
        df = DataFrame(
            [["a", "x"], ["a", "y"], ["b", "x"], ["a", "x"], ["b", "y"]],
            columns=["A", "X"],
        )
        # 按列 A 和 X 进行分组
        g = df.groupby(["A", "X"])
        # 分组后的 ngroup 结果
        g_ngroup = g.ngroup()
        # 分组后的 cumcount 结果
        g_cumcount = g.cumcount()
        # 期望的 ngroup 结果 Series，值为 [0, 1, 2, 0, 3]
        expected_ngroup = Series([0, 1, 2, 0, 3])
        # 期望的 cumcount 结果 Series，值为 [0, 0, 0, 1, 0]
        expected_cumcount = Series([0, 0, 0, 1, 0])

        # 断言两个 Series 相等：g_ngroup 和期望的结果 expected_ngroup
        tm.assert_series_equal(g_ngroup, expected_ngroup)
        # 断言两个 Series 相等：g_cumcount 和期望的结果 expected_cumcount
        tm.assert_series_equal(g_cumcount, expected_cumcount)
    def test_ngroup_cumcount_pair(self):
        # 对所有小系列进行暴力比较
        for p in product(range(3), repeat=4):
            # 创建包含单列 "a" 的 DataFrame
            df = DataFrame({"a": p})
            # 按列 "a" 对 DataFrame 进行分组
            g = df.groupby(["a"])

            # 对列 "a" 的值进行排序并去重
            order = sorted(set(p))
            # 根据排序后的顺序，创建一个列表表示每个值在排序后列表中的索引
            ngroupd = [order.index(val) for val in p]
            # 创建一个列表，表示每个值在当前位置之前出现的次数
            cumcounted = [p[:i].count(val) for i, val in enumerate(p)]

            # 断言分组后的组号与预期的一致
            tm.assert_series_equal(g.ngroup(), Series(ngroupd))
            # 断言分组后的累计计数与预期的一致
            tm.assert_series_equal(g.cumcount(), Series(cumcounted))

    def test_ngroup_respects_groupby_order(self, sort):
        # 创建包含随机字母列 "a" 的 DataFrame
        df = DataFrame({"a": np.random.default_rng(2).choice(list("abcdef"), 100)})
        # 按列 "a" 对 DataFrame 进行分组，可以选择是否排序
        g = df.groupby("a", sort=sort)
        # 初始化两个新列，并设置初始值为 -1
        df["group_id"] = -1
        df["group_index"] = -1

        # 遍历分组后的每个组及其索引
        for i, (_, group) in enumerate(g):
            # 将每个组的索引设置为对应的组号
            df.loc[group.index, "group_id"] = i
            # 将每个组内的每个元素的索引设置为其在组内的顺序
            for j, ind in enumerate(group.index):
                df.loc[ind, "group_index"] = j

        # 断言计算得到的组号与预期的一致
        tm.assert_series_equal(Series(df["group_id"].values), g.ngroup())
        # 断言计算得到的组内索引与预期的一致
        tm.assert_series_equal(Series(df["group_index"].values), g.cumcount())

    @pytest.mark.parametrize(
        "datetimelike",
        [
            # 创建不同类型的日期时间数据列表，用于测试
            [Timestamp(f"2016-05-{i:02d} 20:09:25+00:00") for i in range(1, 4)],
            [Timestamp(f"2016-05-{i:02d} 20:09:25") for i in range(1, 4)],
            [Timestamp(f"2016-05-{i:02d} 20:09:25", tz="UTC") for i in range(1, 4)],
            [Timedelta(x, unit="h") for x in range(1, 4)],
            [Period(freq="2W", year=2017, month=x) for x in range(1, 4)],
        ],
    )
    def test_count_with_datetimelike(self, datetimelike):
        # 对于 issue #13393 的测试，当计算日期时间类型列时 DataframeGroupBy.count() 失败的情况

        # 创建包含列 "x" 和日期时间列 "y" 的 DataFrame
        df = DataFrame({"x": ["a", "a", "b"], "y": datetimelike})
        # 对列 "x" 进行分组并计算每组的计数
        res = df.groupby("x").count()
        # 创建预期的 DataFrame，包含每个组的计数结果
        expected = DataFrame({"y": [2, 1]}, index=["a", "b"])
        expected.index.name = "x"
        # 断言实际结果与预期结果一致
        tm.assert_frame_equal(expected, res)

    def test_count_with_only_nans_in_first_group(self):
        # 对于 GH21956 的测试，当第一组中只有 NaN 时的情况

        # 创建包含列 "A", "B", "C" 的 DataFrame，其中第一列 "A" 中全为 NaN
        df = DataFrame({"A": [np.nan, np.nan], "B": ["a", "b"], "C": [1, 2]})
        # 按列 "A", "B" 进行分组并计算每组的计数
        result = df.groupby(["A", "B"]).C.count()
        # 创建预期的 Series，表示每个组的计数结果
        mi = MultiIndex(levels=[[], ["a", "b"]], codes=[[], []], names=["A", "B"])
        expected = Series([], index=mi, dtype=np.int64, name="C")
        # 断言实际结果与预期结果一致
        tm.assert_series_equal(result, expected, check_index_type=False)

    def test_count_groupby_column_with_nan_in_groupby_column(self):
        # 对于 https://github.com/pandas-dev/pandas/issues/32841 的测试，当分组列中含有 NaN 时的情况

        # 创建包含列 "A", "B" 的 DataFrame
        df = DataFrame({"A": [1, 1, 1, 1, 1], "B": [5, 4, np.nan, 3, 0]})
        # 按列 "B" 进行分组并计算每组的计数
        res = df.groupby(["B"]).count()
        # 创建预期的 DataFrame，包含每个组的计数结果
        expected = DataFrame(
            index=Index([0.0, 3.0, 4.0, 5.0], name="B"), data={"A": [1, 1, 1, 1]}
        )
        # 断言实际结果与预期结果一致
        tm.assert_frame_equal(expected, res)
    def test_groupby_count_dateparseerror(self):
        # 创建一个日期范围，从 "1/1/2012" 开始，频率为每5分钟，共10个时间点
        dr = date_range(start="1/1/2012", freq="5min", periods=10)

        # 创建一个序列，包含了从0到9的数字，索引为日期范围和0到9的数组成的元组
        ser = Series(np.arange(10), index=[dr, np.arange(10)])
        
        # 按照条件将序列分组，条件为第二个元素是否为偶数，然后统计每组的数量
        grouped = ser.groupby(lambda x: x[1] % 2 == 0)
        
        # 计算分组后每组的数量
        result = grouped.count()

        # 创建另一个序列，包含了从0到9的数字，索引为0到9的数组和日期范围的元组
        ser = Series(np.arange(10), index=[np.arange(10), dr])
        
        # 按照条件将序列分组，条件为第一个元素是否为偶数，然后统计每组的数量
        grouped = ser.groupby(lambda x: x[0] % 2 == 0)
        
        # 计算分组后每组的数量，作为预期结果
        expected = grouped.count()

        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)
# 测试函数：测试基于时间间隔的分组计数功能（使用 Cython 实现）
def test_groupby_timedelta_cython_count():
    # 创建一个包含两列的数据框
    df = DataFrame(
        {"g": list("ab" * 2), "delta": np.arange(4).astype("timedelta64[ns]")}
    )
    # 预期的结果，包含两行，索引为 ['a', 'b']，列名为 'delta'
    expected = Series([2, 2], index=Index(["a", "b"], name="g"), name="delta")
    # 对数据框按 'g' 列进行分组，计算每组 'delta' 列的非空值数量
    result = df.groupby("g").delta.count()
    # 断言预期结果与计算结果相等
    tm.assert_series_equal(expected, result)


# 测试函数：测试计数功能
def test_count():
    n = 1 << 15  # 位操作，计算 2 的 15 次方
    dr = date_range("2015-08-30", periods=n // 10, freq="min")

    # 创建包含多列的数据框，包括随机生成的数据
    df = DataFrame(
        {
            "1st": np.random.default_rng(2).choice(list(ascii_lowercase), n),
            "2nd": np.random.default_rng(2).integers(0, 5, n),
            "3rd": np.random.default_rng(2).standard_normal(n).round(3),
            "4th": np.random.default_rng(2).integers(-10, 10, n),
            "5th": np.random.default_rng(2).choice(dr, n),
            "6th": np.random.default_rng(2).standard_normal(n).round(3),
            "7th": np.random.default_rng(2).standard_normal(n).round(3),
            "8th": np.random.default_rng(2).choice(dr, n) - np.random.default_rng(2).choice(dr, 1),
            "9th": np.random.default_rng(2).choice(list(ascii_lowercase), n),
        }
    )

    # 针对除 '1st', '2nd', '4th' 列之外的每一列，随机将部分值设为 NaN
    for col in df.columns.drop(["1st", "2nd", "4th"]):
        df.loc[np.random.default_rng(2).choice(n, n // 10), col] = np.nan

    df["9th"] = df["9th"].astype("category")  # 将 '9th' 列转换为分类数据类型

    # 对 '1st', '2nd' 和 ['1st', '2nd'] 三种情况进行分组，并分别计算各组的计数
    for key in ["1st", "2nd", ["1st", "2nd"]]:
        # 左侧为按 key 列分组后的计数结果
        left = df.groupby(key).count()
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 右侧为应用 DataFrame.count 函数后去除 key 列的计数结果
            right = df.groupby(key).apply(DataFrame.count).drop(key, axis=1)
        # 断言左右两侧的数据框相等
        tm.assert_frame_equal(left, right)


# 测试函数：测试计数非空值的功能
def test_count_non_nulls():
    # 创建包含特定数据的数据框
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, np.nan]],
        columns=["A", "B", "C"],
    )

    # 按 'A' 列分组，并计算每组的计数
    count_as = df.groupby("A").count()
    # 以非索引形式按 'A' 列分组，并计算每组的计数
    count_not_as = df.groupby("A", as_index=False).count()

    # 预期的结果数据框
    expected = DataFrame([[1, 2], [0, 0]], columns=["B", "C"], index=[1, 3])
    expected.index.name = "A"
    # 断言非索引分组和普通分组的计数结果与预期结果相等
    tm.assert_frame_equal(count_not_as, expected.reset_index())
    tm.assert_frame_equal(count_as, expected)

    # 对 'A' 列的计数，期望结果为一个 Series
    count_B = df.groupby("A")["B"].count()
    # 断言计算结果与预期结果相等
    tm.assert_series_equal(count_B, expected["B"])


# 测试函数：测试计数对象类型数据的功能
def test_count_object():
    # 创建包含特定数据的数据框
    df = DataFrame({"a": ["a"] * 3 + ["b"] * 3, "c": [2] * 3 + [3] * 3})
    # 对 'c' 列分组，并计算每组 'a' 列的计数
    result = df.groupby("c").a.count()
    # 预期结果的 Series
    expected = Series([3, 3], index=Index([2, 3], name="c"), name="a")
    # 断言计算结果与预期结果相等
    tm.assert_series_equal(result, expected)


# 测试函数：测试计数对象类型数据中 NaN 值的功能
def test_count_object_nan():
    # 创建包含特定数据的数据框
    df = DataFrame({"a": ["a", np.nan, np.nan] + ["b"] * 3, "c": [2] * 3 + [3] * 3})
    # 对 'c' 列分组，并计算每组 'a' 列的计数
    result = df.groupby("c").a.count()
    # 预期结果的 Series
    expected = Series([1, 3], index=Index([2, 3], name="c"), name="a")
    # 断言计算结果与预期结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("typ", ["object", "float32"])
def test_count_cross_type(typ):
    # GH8169
    # 设置 float64 类型以避免设置 NaN 时的类型转换
    # 使用 NumPy 随机数生成器创建两个 (10, 2) 的整数数组，并将它们水平堆叠在一起，然后转换为 float64 类型
    vals = np.hstack(
        (
            np.random.default_rng(2).integers(0, 5, (10, 2)),
            np.random.default_rng(2).integers(0, 2, (10, 2)),
        )
    ).astype("float64")

    # 用生成的 vals 数组创建一个 DataFrame，指定列名为 ["a", "b", "c", "d"]
    df = DataFrame(vals, columns=["a", "b", "c", "d"])
    
    # 将 DataFrame 中值为 2 的元素替换为 NaN（缺失值）
    df[df == 2] = np.nan
    
    # 根据列 "c" 和 "d" 对 DataFrame 进行分组，并统计每组的计数，保存为期望的结果
    expected = df.groupby(["c", "d"]).count()

    # 将 DataFrame 中的列 "a" 和 "b" 转换为指定类型 typ
    df["a"] = df["a"].astype(typ)
    df["b"] = df["b"].astype(typ)
    
    # 再次根据列 "c" 和 "d" 对 DataFrame 进行分组，并统计每组的计数，保存为结果
    result = df.groupby(["c", "d"]).count()
    
    # 使用测试工具来比较结果 DataFrame 和期望 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试DataFrame对象的分组计数功能
def test_lower_int_prec_count():
    # 创建一个DataFrame对象，包含四列数据：a是int8类型数组，b是uint32类型数组，c是int16类型数组，grp是重复的ab字符串列表
    df = DataFrame(
        {
            "a": np.array([0, 1, 2, 100], np.int8),
            "b": np.array([1, 2, 3, 6], np.uint32),
            "c": np.array([4, 5, 6, 8], np.int16),
            "grp": list("ab" * 2),
        }
    )
    # 对DataFrame按照'grp'列进行分组，然后计算每个分组中的行数
    result = df.groupby("grp").count()
    # 创建一个预期的DataFrame对象，包含两列数据：a、b、c每列分别包含两个值，索引为grp列的值
    expected = DataFrame(
        {"a": [2, 2], "b": [2, 2], "c": [2, 2]}, index=Index(list("ab"), name="grp")
    )
    # 使用断言方法assert_frame_equal来比较测试结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试处理在特定对象上抛出异常时的计数功能
def test_count_uses_size_on_exception():
    # 定义一个自定义的异常类RaisingObjectException
    class RaisingObjectException(Exception):
        pass

    # 定义一个会抛出异常的类RaisingObject
    class RaisingObject:
        def __init__(self, msg="I will raise inside Cython") -> None:
            super().__init__()
            self.msg = msg

        def __eq__(self, other):
            # 当在Cython中检查到异常时调用，抛出自定义异常RaisingObjectException
            raise RaisingObjectException(self.msg)

    # 创建一个DataFrame对象，包含两列数据：a是包含抛出异常对象的列表，grp是重复的ab字符串列表
    df = DataFrame({"a": [RaisingObject() for _ in range(4)], "grp": list("ab" * 2)})
    # 对DataFrame按照'grp'列进行分组，然后计算每个分组中的行数
    result = df.groupby("grp").count()
    # 创建一个预期的DataFrame对象，包含一列数据a，其中每个值为2，索引为grp列的值
    expected = DataFrame({"a": [2, 2]}, index=Index(list("ab"), name="grp"))
    # 使用断言方法assert_frame_equal来比较测试结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试处理包含Arrow字符串数组的计数功能
def test_count_arrow_string_array(any_string_dtype):
    # 导入pyarrow库，如果导入失败则跳过此测试（importorskip的作用）
    pytest.importorskip("pyarrow")
    # 创建一个DataFrame对象，包含两列数据：a是整数数组，b是包含字符串的Series对象，dtype由参数any_string_dtype指定
    df = DataFrame(
        {"a": [1, 2, 3], "b": Series(["a", "b", "a"], dtype=any_string_dtype)}
    )
    # 对DataFrame按照'a'列进行分组，然后计算每个分组中的行数
    result = df.groupby("a").count()
    # 创建一个预期的DataFrame对象，包含一列数据b，其中每个值为1，索引为a列的值
    expected = DataFrame({"b": [1]}, index=Index([1, 2, 3], name="a"))
    # 使用断言方法assert_frame_equal来比较测试结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)
```