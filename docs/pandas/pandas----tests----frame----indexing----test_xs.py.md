# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_xs.py`

```
# 导入正则表达式模块
import re

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 pytest 测试框架
import pytest

# 从 pandas 库中导入特定模块
from pandas import (
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    Series,
    concat,
)
# 导入 pandas 内部测试模块，并重命名为 tm
import pandas._testing as tm

# 从 pandas 时间序列偏移模块中导入工作日偏移量
from pandas.tseries.offsets import BDay

# 定义 pytest 的 fixture，生成一个包含多层索引的 DataFrame 对象
@pytest.fixture
def four_level_index_dataframe():
    # 创建一个包含浮点数的 NumPy 数组
    arr = np.array(
        [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.364],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
            [-0.6662, -0.5243, -0.358, 0.89145, 2.5838],
        ]
    )
    # 创建一个多层次索引对象
    index = MultiIndex(
        levels=[["a", "x"], ["b", "q"], [10.0032, 20.0, 30.0], [3, 4, 5]],
        codes=[[0, 0, 1], [0, 1, 1], [0, 1, 2], [2, 1, 0]],
        names=["one", "two", "three", "four"],
    )
    # 返回一个 DataFrame 对象，用上述数组和索引创建
    return DataFrame(arr, index=index, columns=list("ABCDE"))


# 定义测试类 TestXS
class TestXS:
    # 定义测试方法 test_xs，接受 float_frame 的参数
    def test_xs(self, float_frame):
        # 获取 float_frame 的第 5 个索引
        idx = float_frame.index[5]
        # 使用 xs 方法获取特定索引行的数据
        xs = float_frame.xs(idx)
        # 遍历 xs 的每个项及其值
        for item, value in xs.items():
            # 如果值是 NaN，则断言 float_frame 中对应项也是 NaN
            if np.isnan(value):
                assert np.isnan(float_frame[item][idx])
            else:
                # 否则断言值与 float_frame 中对应项相等
                assert value == float_frame[item][idx]

    # 定义测试方法 test_xs_mixed
    def test_xs_mixed(self):
        # 创建一个包含混合类型数据的测试数据字典
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        # 根据测试数据创建一个 DataFrame 对象
        frame = DataFrame(test_data)
        # 使用 xs 方法获取特定索引行的数据
        xs = frame.xs("1")
        # 断言 xs 的数据类型是 np.object_
        assert xs.dtype == np.object_
        # 断言 xs 的 "A" 列的值为 1
        assert xs["A"] == 1
        # 断言 xs 的 "B" 列的值为 "1"
        assert xs["B"] == "1"

    # 定义测试方法 test_xs_dt_error，接受 datetime_frame 的参数
    def test_xs_dt_error(self, datetime_frame):
        # 使用 pytest 的 raises 方法断言 xs 方法在给定条件下会抛出 KeyError 异常
        with pytest.raises(
            KeyError, match=re.escape("Timestamp('1999-12-31 00:00:00')")
        ):
            datetime_frame.xs(datetime_frame.index[0] - BDay())

    # 定义测试方法 test_xs_other，接受 float_frame 的参数
    def test_xs_other(self, float_frame):
        # 复制 float_frame 对象
        float_frame_orig = float_frame.copy()
        # 使用 xs 方法获取特定列的数据
        series = float_frame.xs("A", axis=1)
        # 获取 float_frame 中的 "A" 列
        expected = float_frame["A"]
        # 使用 pandas._testing 模块的方法断言两个 Series 对象相等
        tm.assert_series_equal(series, expected)

        # 再次使用 xs 方法获取特定列的数据
        series = float_frame.xs("A", axis=1)
        # 修改 series 的值为 5
        series[:] = 5
        # 断言修改后的 series 不会影响 float_frame 的 "A" 列
        tm.assert_series_equal(float_frame["A"], float_frame_orig["A"])
        # 断言 float_frame 的 "A" 列中并非所有值都是 5
        assert not (expected == 5).all()

    # 定义测试方法 test_xs_corner
    def test_xs_corner(self):
        # 创建一个空的 DataFrame 对象，仅包含索引
        df = DataFrame(index=[0])
        # 向 df 中添加五列数据
        df["A"] = 1.0
        df["B"] = "foo"
        df["C"] = 2.0
        df["D"] = "bar"
        df["E"] = 3.0

        # 使用 xs 方法获取特定索引行的数据
        xs = df.xs(0)
        # 创建一个期望的 Series 对象，包含相同的数据和索引
        exp = Series([1.0, "foo", 2.0, "bar", 3.0], index=list("ABCDE"), name=0)
        # 使用 pandas._testing 模块的方法断言两个 Series 对象相等
        tm.assert_series_equal(xs, exp)

        # 创建一个仅包含索引的空 DataFrame 对象
        df = DataFrame(index=["a", "b", "c"])
        # 使用 xs 方法获取特定索引行的数据
        result = df.xs("a")
        # 创建一个期望的空 Series 对象，其名字为 "a"，数据类型为 np.float64
        expected = Series([], name="a", dtype=np.float64)
        # 使用 pandas._testing 模块的方法断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法 test_xs_duplicates
    def test_xs_duplicates(self):
        # 创建一个包含随机标准正态分布数据的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=["b", "b", "c", "b", "a"],
        )

        # 使用 xs 方法获取特定索引行的数据
        cross = df.xs("c")
        # 获取 df 中的第三行数据
        exp = df.iloc[2]
        # 使用 pandas._testing 模块的方法断言两个 Series 对象相等
        tm.assert_series_equal(cross, exp)
    def test_xs_keep_level(self):
        # 创建一个 DataFrame 对象，包含了销售数据，以年、口味和星期为多级索引
        df = DataFrame(
            {
                "day": {0: "sat", 1: "sun"},
                "flavour": {0: "strawberry", 1: "strawberry"},
                "sales": {0: 10, 1: 12},
                "year": {0: 2008, 1: 2008},
            }
        ).set_index(["year", "flavour", "day"])
        # 使用 xs 方法获取指定 level="day" 的子数据框，并保留 day 这个级别
        result = df.xs("sat", level="day", drop_level=False)
        # 期望的结果是取出第一行
        expected = df[:1]
        # 断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 xs 方法获取指定 level=["year", "day"] 的子数据框，并保留 year 和 day 这两个级别
        result = df.xs((2008, "sat"), level=["year", "day"], drop_level=False)
        # 期望的结果是取出第一行
        tm.assert_frame_equal(result, expected)

    def test_xs_view(self):
        # 在版本 0.14 中，如果可能的话会返回视图，否则返回副本，但这取决于 numpy 的实现
        # 创建一个 DataFrame 对象，包含了从 0 到 19 的数据，形状为 4 行 5 列
        dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
        # 创建一个 dm 的副本
        df_orig = dm.copy()

        # 使用 xs 方法获取索引为 2 的行，并将其全部设置为 20，这里期望会触发 chained assignment 错误
        with tm.raises_chained_assignment_error():
            dm.xs(2)[:] = 20
        # 断言两个数据框是否相等
        tm.assert_frame_equal(dm, df_orig)
class TestXSWithMultiIndex:
    def test_xs_doc_example(self):
        # TODO: 更具描述性的名称
        # 根据 advanced.rst 中的示例
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        # 将数组转换为元组列表
        tuples = list(zip(*arrays))

        # 使用元组列表创建多级索引
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        # 创建一个 DataFrame，填充随机数值，使用多级索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )

        # 使用 xs 方法获取特定级别的切片
        result = df.xs(("one", "bar"), level=("second", "first"), axis=1)

        # 预期的 DataFrame 是原始 DataFrame 的特定切片
        expected = df.iloc[:, [0]]
        tm.assert_frame_equal(result, expected)

    def test_xs_integer_key(self):
        # 参见 GitHub 问题编号 #2107
        # 创建日期范围和标识符列表
        dates = range(20111201, 20111205)
        ids = list("abcde")
        # 使用笛卡尔积创建多级索引
        index = MultiIndex.from_product([dates, ids], names=["date", "secid"])
        # 创建一个 DataFrame，填充随机数值，使用多级索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 3)),
            index,
            ["X", "Y", "Z"],
        )

        # 使用 xs 方法获取特定级别的切片
        result = df.xs(20111201, level="date")
        # 预期的 DataFrame 是原始 DataFrame 的特定切片
        expected = df.loc[20111201, :]
        tm.assert_frame_equal(result, expected)

    def test_xs_level(self, multiindex_dataframe_random_data):
        # 使用随机数据的多级索引 DataFrame
        df = multiindex_dataframe_random_data
        # 使用 xs 方法获取特定级别的切片
        result = df.xs("two", level="second")
        # 预期的 DataFrame 是原始 DataFrame 指定级别的切片
        expected = df[df.index.get_level_values(1) == "two"]
        expected.index = Index(["foo", "bar", "baz", "qux"], name="first")
        tm.assert_frame_equal(result, expected)

    def test_xs_level_eq_2(self):
        # 创建随机数组并使用它们创建多级索引
        arr = np.random.default_rng(2).standard_normal((3, 5))
        index = MultiIndex(
            levels=[["a", "p", "x"], ["b", "q", "y"], ["c", "r", "z"]],
            codes=[[2, 0, 1], [2, 0, 1], [2, 0, 1]],
        )
        df = DataFrame(arr, index=index)
        # 预期的 DataFrame 是原始 DataFrame 指定级别的切片
        expected = DataFrame(arr[1:2], index=[["a"], ["b"]])
        result = df.xs("c", level=2)
        tm.assert_frame_equal(result, expected)

    def test_xs_setting_with_copy_error(self, multiindex_dataframe_random_data):
        # 这是在 0.14 版本中的复制操作
        # 使用随机数据的多级索引 DataFrame
        df = multiindex_dataframe_random_data
        # 复制原始 DataFrame
        df_orig = df.copy()
        # 使用 xs 方法获取特定级别的切片
        result = df.xs("two", level="second")

        # 修改切片的值
        result[:] = 10
        # 断言修改后的 DataFrame 与原始的 DataFrame 相等
        tm.assert_frame_equal(df, df_orig)

    def test_xs_setting_with_copy_error_multiple(self, four_level_index_dataframe):
        # 这是在 0.14 版本中的复制操作
        # 使用四级索引的 DataFrame
        df = four_level_index_dataframe
        # 复制原始 DataFrame
        df_orig = df.copy()
        # 使用 xs 方法获取特定级别的切片
        result = df.xs(("a", 4), level=["one", "four"])

        # 修改切片的值
        result[:] = 10
        # 断言修改后的 DataFrame 与原始的 DataFrame 相等
        tm.assert_frame_equal(df, df_orig)

    @pytest.mark.parametrize("key, level", [("one", "second"), (["one"], ["second"])])
    def test_xs_with_duplicates(self, key, level, multiindex_dataframe_random_data):
        # 测试函数：测试处理有重复索引的情况下的 xs 方法
        # 见 GitHub issue #13719
        # 获取随机生成的多级索引数据帧
        frame = multiindex_dataframe_random_data
        # 将数据帧复制两次进行连接
        df = concat([frame] * 2)
        # 断言索引不是唯一的
        assert df.index.is_unique is False
        # 期望结果是取出特定条件下的数据帧并复制两次进行连接
        expected = concat([frame.xs("one", level="second")] * 2)

        # 如果 key 是列表，则使用元组形式的 key 调用 xs 方法，指定 level
        if isinstance(key, list):
            result = df.xs(tuple(key), level=level)
        else:
            # 否则直接使用 key 调用 xs 方法，指定 level
            result = df.xs(key, level=level)
        # 使用测试工具函数断言两个数据帧是否相等
        tm.assert_frame_equal(result, expected)

    def test_xs_missing_values_in_index(self):
        # 测试函数：测试索引中缺失值在 xs 方法中的保留情况
        # 见 GitHub issue #6574
        # 定义包含缺失值的索引的数据
        acc = [
            ("a", "abcde", 1),
            ("b", "bbcde", 2),
            ("y", "yzcde", 25),
            ("z", "xbcde", 24),
            ("z", None, 26),
            ("z", "zbcde", 25),
            ("z", "ybcde", 26),
        ]
        # 创建数据帧并设置多级索引
        df = DataFrame(acc, columns=["a1", "a2", "cnt"]).set_index(["a1", "a2"])
        # 期望结果是选择特定条件下的数据帧
        expected = DataFrame(
            {"cnt": [24, 26, 25, 26]},
            index=Index(["xbcde", np.nan, "zbcde", "ybcde"], name="a2"),
        )

        # 使用 xs 方法选择特定条件下的数据帧
        result = df.xs("z", level="a1")
        # 使用测试工具函数断言两个数据帧是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "key, level, exp_arr, exp_index",
        [
            # 参数化测试函数，测试指定轴为 1 时的 xs 方法行为
            ("a", "lvl0", lambda x: x[:, 0:2], Index(["bar", "foo"], name="lvl1")),
            ("foo", "lvl1", lambda x: x[:, 1:2], Index(["a"], name="lvl0")),
        ],
    )
    def test_xs_named_levels_axis_eq_1(self, key, level, exp_arr, exp_index):
        # 测试函数：测试指定轴为 1 时命名级别的 xs 方法行为
        # 见 GitHub issue #2903
        # 使用随机数生成正态分布的数组
        arr = np.random.default_rng(2).standard_normal((4, 4))
        # 创建具有多级索引的数据帧
        index = MultiIndex(
            levels=[["a", "b"], ["bar", "foo", "hello", "world"]],
            codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            names=["lvl0", "lvl1"],
        )
        df = DataFrame(arr, columns=index)
        # 使用 xs 方法选择特定条件下的数据帧
        result = df.xs(key, level=level, axis=1)
        # 期望结果是选择特定条件下的数据帧
        expected = DataFrame(exp_arr(arr), columns=exp_index)
        # 使用测试工具函数断言两个数据帧是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer",
        [
            # 参数化测试函数，测试多级级别选择的 xs 方法行为
            lambda df: df.xs(("a", 4), level=["one", "four"]),
            lambda df: df.xs("a").xs(4, level="four"),
        ],
    )
    def test_xs_level_multiple(self, indexer, four_level_index_dataframe):
        # 测试函数：测试多级级别选择的 xs 方法行为
        # 获取四级索引的数据帧
        df = four_level_index_dataframe
        # 期望结果是选择特定条件下的数据帧
        expected_values = [[0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(
            levels=[["q"], [20.0]], codes=[[0], [0]], names=["two", "three"]
        )
        expected = DataFrame(
            expected_values, index=expected_index, columns=list("ABCDE")
        )
        # 使用 indexer 函数选择特定条件下的数据帧
        result = indexer(df)
        # 使用测试工具函数断言两个数据帧是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer", [lambda df: df.xs("a", level=0), lambda df: df.xs("a")]
    )
    # 测试函数，用于测试索引器函数对于四级索引数据框的操作
    def test_xs_level0(self, indexer, four_level_index_dataframe):
        # 获取四级索引数据框
        df = four_level_index_dataframe
        # 期望的数值数据
        expected_values = [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.364],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
        ]
        # 期望的多级索引
        expected_index = MultiIndex(
            levels=[["b", "q"], [10.0032, 20.0], [4, 5]],
            codes=[[0, 1], [0, 1], [1, 0]],
            names=["two", "three", "four"],
        )
        # 期望的数据框
        expected = DataFrame(
            expected_values, index=expected_index, columns=list("ABCDE")
        )

        # 调用索引器函数进行索引操作
        result = indexer(df)
        # 使用测试框架检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，测试多级索引数据框的 xs 方法返回的值是否正确
    def test_xs_values(self, multiindex_dataframe_random_data):
        # 获取多级索引数据框
        df = multiindex_dataframe_random_data
        # 调用 xs 方法获取特定索引组合的数据值
        result = df.xs(("bar", "two")).values
        # 获取预期的数据值
        expected = df.values[4]
        # 使用测试框架检查结果与期望是否几乎相等
        tm.assert_almost_equal(result, expected)

    # 测试函数，测试 xs 方法与 loc 方法返回的结果是否相等
    def test_xs_loc_equality(self, multiindex_dataframe_random_data):
        # 获取多级索引数据框
        df = multiindex_dataframe_random_data
        # 调用 xs 方法获取特定索引组合的数据系列
        result = df.xs(("bar", "two"))
        # 使用 loc 方法获取特定索引组合的数据系列
        expected = df.loc[("bar", "two")]
        # 使用测试框架检查结果与期望是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数，测试 xs 方法中使用 IndexSlice 参数的情况
    def test_xs_IndexSlice_argument_not_implemented(self, frame_or_series):
        # GH#35301

        # 创建多级索引对象
        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        # 创建数据框对象，使用随机正态分布数据
        obj = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)
        # 根据参数选择数据框或者数据系列
        if frame_or_series is Series:
            obj = obj[0]

        # 预期的数据结果，获取部分数据并去除第一级索引
        expected = obj.iloc[-2:].droplevel(0)

        # 使用 xs 方法获取特定索引组合的数据并进行测试断言
        result = obj.xs(IndexSlice[("foo", "qux", 0), :])
        tm.assert_equal(result, expected)

        # 使用 loc 方法获取特定索引组合的数据并进行测试断言
        result = obj.loc[IndexSlice[("foo", "qux", 0), :]]
        tm.assert_equal(result, expected)

    # 测试函数，测试在非 MultiIndex 索引情况下使用 xs 方法是否会引发异常
    def test_xs_levels_raises(self, frame_or_series):
        # 创建数据框或者数据系列对象
        obj = DataFrame({"A": [1, 2, 3]})
        # 根据参数选择数据框或者数据系列
        if frame_or_series is Series:
            obj = obj["A"]

        # 预期的错误消息
        msg = "Index must be a MultiIndex"
        # 使用 pytest 框架验证是否会抛出预期的异常消息
        with pytest.raises(TypeError, match=msg):
            obj.xs(0, level="as")

    # 测试函数，测试 xs 方法中 drop_level 参数为 False 的情况
    def test_xs_multiindex_droplevel_false(self):
        # GH#19056
        # 创建多级索引对象
        mi = MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "x")], names=["level1", "level2"]
        )
        # 创建数据框对象，包含多级索引
        df = DataFrame([[1, 2, 3]], columns=mi)
        # 使用 xs 方法获取特定索引的数据，并保留多级索引
        result = df.xs("a", axis=1, drop_level=False)
        # 预期的数据框结果，保留多级索引的部分
        expected = DataFrame(
            [[1, 2]],
            columns=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y")], names=["level1", "level2"]
            ),
        )
        # 使用测试框架检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，测试 xs 方法中 drop_level 参数为 False 的情况（索引为单级）
    def test_xs_droplevel_false(self):
        # GH#19056
        # 创建数据框对象，包含单级索引
        df = DataFrame([[1, 2, 3]], columns=Index(["a", "b", "c"]))
        # 使用 xs 方法获取特定索引的数据，并保留单级索引
        result = df.xs("a", axis=1, drop_level=False)
        # 预期的数据框结果，只包含指定索引列
        expected = DataFrame({"a": [1]})
        # 使用测试框架检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)
    def test_xs_droplevel_false_view(self):
        # 测试用例名称：test_xs_droplevel_false_view
        # GitHub 问题编号：GH#37832

        # 创建一个包含一行数据的 DataFrame，列索引为 ["a", "b", "c"]
        df = DataFrame([[1, 2, 3]], columns=Index(["a", "b", "c"]))

        # 使用 xs 方法获取列索引为 "a" 的数据，保留层级(drop_level=False)
        result = df.xs("a", axis=1, drop_level=False)

        # 断言：检查 result 是否与 df 共享内存，即视图相同的数据
        assert np.shares_memory(result.iloc[:, 0]._values, df.iloc[:, 0]._values)

        # 修改原始 DataFrame 的数据，验证结果 result 未受影响
        df.iloc[0, 0] = 2

        # 预期结果 DataFrame，只包含列名为 "a" 的数据 [1]
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

        # 创建一个包含一行数据的 DataFrame，列索引为 ["a", "b", "c"]
        df = DataFrame([[1, 2.5, "a"]], columns=Index(["a", "b", "c"]))

        # 使用 xs 方法获取列索引为 "a" 的数据，保留层级(drop_level=False)
        result = df.xs("a", axis=1, drop_level=False)

        # 修改原始 DataFrame 的数据，验证结果 result 未受影响
        df.iloc[0, 0] = 2

        # 预期结果 DataFrame，只包含列名为 "a" 的数据 [1]
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_list_indexer_droplevel_false(self):
        # 测试用例名称：test_xs_list_indexer_droplevel_false
        # GitHub 问题编号：GH#41760

        # 创建一个 MultiIndex，包含三个元组作为列索引
        mi = MultiIndex.from_tuples([("x", "m", "a"), ("x", "n", "b"), ("y", "o", "c")])

        # 创建一个包含两行三列数据的 DataFrame，列索引为 mi
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)

        # 使用 xs 方法获取列索引为 ("x", "y") 的数据，保留层级(drop_level=False)，期望抛出 KeyError
        with pytest.raises(KeyError, match="y"):
            df.xs(("x", "y"), drop_level=False, axis=1)
```