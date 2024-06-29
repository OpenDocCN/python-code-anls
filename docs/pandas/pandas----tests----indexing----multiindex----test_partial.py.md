# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_partial.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

from pandas import (  # 从 Pandas 库中导入以下子模块和函数
    DataFrame,  # 数据帧结构
    DatetimeIndex,  # 时间索引类型
    MultiIndex,  # 多级索引类型
    date_range,  # 日期范围生成器
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块

class TestMultiIndexPartial:
    def test_getitem_partial_int(self):
        # GH 12416
        # with single item
        l1 = [10, 20]  # 定义列表 l1，包含整数 10 和 20
        l2 = ["a", "b"]  # 定义列表 l2，包含字符串 "a" 和 "b"
        df = DataFrame(index=range(2), columns=MultiIndex.from_product([l1, l2]))
        # 创建一个 2x2 的数据帧 df，其索引为 0 到 1，列索引为 l1 和 l2 的笛卡尔积
        expected = DataFrame(index=range(2), columns=l2)
        # 创建期望的数据帧，索引为 0 到 1，列索引为 l2
        result = df[20]  # 从 df 中选择列索引为 20 的列
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

        # with list
        expected = DataFrame(
            index=range(2), columns=MultiIndex.from_product([l1[1:], l2])
        )
        # 创建期望的数据帧，索引为 0 到 1，列索引为 l1 的第二个元素到末尾与 l2 的笛卡尔积
        result = df[[20]]  # 从 df 中选择列索引为 20 的列（作为列表）
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

        # missing item:
        with pytest.raises(KeyError, match="1"):
            df[1]  # 尝试选择索引为 1 的列，预期引发 KeyError
        with pytest.raises(KeyError, match=r"'\[1\] not in index'"):
            df[[1]]  # 尝试选择索引为 1 的列（作为列表），预期引发包含特定消息的 KeyError

    def test_series_slice_partial(self):
        pass  # 此方法暂未实现，占位符

    def test_xs_partial(
        self,
        multiindex_dataframe_random_data,
        multiindex_year_month_day_dataframe_random_data,
    ):
        frame = multiindex_dataframe_random_data  # 从参数获取随机多级索引数据帧
        ymd = multiindex_year_month_day_dataframe_random_data  # 从参数获取随机年月日多级索引数据帧
        result = frame.xs("foo")  # 选择 frame 数据帧中的 "foo" 层级
        result2 = frame.loc["foo"]  # 使用 loc 方法选择 frame 数据帧中的 "foo" 行
        expected = frame.T["foo"].T  # 从 frame 数据帧转置后选择 "foo" 行
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等
        tm.assert_frame_equal(result, result2)  # 断言结果与 result2 相等

        result = ymd.xs((2000, 4))  # 选择 ymd 数据帧中 (2000, 4) 元组索引
        expected = ymd.loc[2000, 4]  # 使用 loc 方法选择 ymd 数据帧中 (2000, 4) 行
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

        # ex from #1796
        index = MultiIndex(
            levels=[["foo", "bar"], ["one", "two"], [-1, 1]],
            codes=[
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
        )
        # 创建多级索引 index，包含三个层级及其对应的代码
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=index,
            columns=list("abcd"),
        )
        # 使用随机数据创建数据帧 df，索引为 index，列为 'a', 'b', 'c', 'd'

        result = df.xs(("foo", "one"))  # 选择 df 数据帧中 ("foo", "one") 多级索引
        expected = df.loc["foo", "one"]  # 使用 loc 方法选择 df 数据帧中 ("foo", "one") 行
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

    def test_getitem_partial(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data  # 从参数获取随机年月日多级索引数据帧
        ymd = ymd.T  # 对数据帧进行转置
        result = ymd[2000, 2]  # 选择转置后数据帧中 (2000, 2) 元组索引

        expected = ymd.reindex(columns=ymd.columns[ymd.columns.codes[1] == 1])
        # 使用 reindex 方法重新索引 ymd 数据帧的列，选择列码为 1 的列
        expected.columns = expected.columns.droplevel(0).droplevel(0)
        # 移除重新索引后的列的两个顶层级别
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

    def test_fancy_slice_partial(
        self,
        multiindex_dataframe_random_data,
        multiindex_year_month_day_dataframe_random_data,
    ):
        pass  # 此方法暂未实现，占位符
    ):
        # 使用 multiindex_dataframe_random_data 进行测试
        frame = multiindex_dataframe_random_data
        # 从 frame 中选择标签索引从 "bar" 到 "baz" 的行
        result = frame.loc["bar":"baz"]
        # 期望的结果是 frame 中行索引从 3 到 7 的数据
        expected = frame[3:7]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 使用 multiindex_year_month_day_dataframe_random_data 进行测试
        ymd = multiindex_year_month_day_dataframe_random_data
        # 从 ymd 中选择标签索引从 (2000, 2) 到 (2000, 4) 的行
        result = ymd.loc[(2000, 2) : (2000, 4)]
        # 获取 ymd 索引的第二层级编码
        lev = ymd.index.codes[1]
        # 期望的结果是 ymd 中第二层级索引值在 1 到 3 范围内的数据
        expected = ymd[(lev >= 1) & (lev <= 3)]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

    def test_getitem_partial_column_select(self):
        # 创建一个多级索引对象 idx
        idx = MultiIndex(
            codes=[[0, 0, 0], [0, 1, 1], [1, 0, 1]],
            levels=[["a", "b"], ["x", "y"], ["p", "q"]],
        )
        # 使用随机数据创建一个 DataFrame df，并使用 idx 作为索引
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=idx)

        # 选择 df 中索引为 ("a", "y") 的行和所有列
        result = df.loc[("a", "y"), :]
        # 期望的结果是 df 中索引为 ("a", "y") 的行和所有列的数据
        expected = df.loc[("a", "y")]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 选择 df 中索引为 ("a", "y") 的行和指定列 [1, 0]
        result = df.loc[("a", "y"), [1, 0]]
        # 期望的结果是 df 中索引为 ("a", "y") 的行和指定列 [1, 0] 的数据
        expected = df.loc[("a", "y")][[1, 0]]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 使用 pytest 检查是否会引发 KeyError，匹配消息为 "('a', 'foo')"
        with pytest.raises(KeyError, match=r"\('a', 'foo'\)"):
            df.loc[("a", "foo"), :]

    def test_partial_set(
        self,
        multiindex_year_month_day_dataframe_random_data,
    ):
        # GH #397
        # 使用 multiindex_year_month_day_dataframe_random_data 进行测试，并复制为 df 和 exp
        ymd = multiindex_year_month_day_dataframe_random_data
        df = ymd.copy()
        exp = ymd.copy()

        # 修改 df 中索引为 2000 年 4 月的数据为 0
        df.loc[2000, 4] = 0
        # 修改 exp 中行索引从 65 到 85 的数据为 0
        exp.iloc[65:85] = 0
        # 断言 df 与 exp 相等
        tm.assert_frame_equal(df, exp)

        # 检查是否会引发 chained assignment 错误
        with tm.raises_chained_assignment_error():
            df["A"].loc[2000, 4] = 1
        # 修改 df 中索引为 (2000, 4) 的 "A" 列数据为 1
        df.loc[(2000, 4), "A"] = 1
        # 修改 exp 中行索引从 65 到 85、列索引为 0 的数据为 1
        exp.iloc[65:85, 0] = 1
        # 断言 df 与 exp 相等
        tm.assert_frame_equal(df, exp)

        # 修改 df 中索引为 2000 年的所有数据为 5
        df.loc[2000] = 5
        # 修改 exp 中前 100 行的数据为 5
        exp.iloc[:100] = 5
        # 断言 df 与 exp 相等
        tm.assert_frame_equal(df, exp)

        # 检查是否会引发 chained assignment 错误
        # 暂时可以正常工作
        with tm.raises_chained_assignment_error():
            df["A"].iloc[14] = 5
        # 检查 df 和 exp 中 "A" 列第 14 行的值是否相等
        assert df["A"].iloc[14] == exp["A"].iloc[14]

    @pytest.mark.parametrize("dtype", [int, float])
    def test_getitem_intkey_leading_level(
        self, multiindex_year_month_day_dataframe_random_data, dtype
    ):
        # GH#33355 不要在第一个级别是 int 类型时回退到位置索引
        ymd = multiindex_year_month_day_dataframe_random_data
        # 获取 ymd 索引的所有层级
        levels = ymd.index.levels
        # 将 ymd 索引的第一个层级转换为指定的 dtype 类型，并设置回 ymd 索引
        ymd.index = ymd.index.set_levels([levels[0].astype(dtype)] + levels[1:])
        # 获取 "A" 列的 Series 对象 ser
        ser = ymd["A"]
        # 获取 ser 的索引 mi
        mi = ser.index
        # 断言 mi 的类型是 MultiIndex
        assert isinstance(mi, MultiIndex)
        # 检查 mi 的第一个层级的数据类型是否符合预期
        if dtype is int:
            assert mi.levels[0].dtype == np.dtype(int)
        else:
            assert mi.levels[0].dtype == np.float64

        # 检查 mi 的第一个层级是否不包含值 14
        assert 14 not in mi.levels[0]
        # 检查 mi 和其层级是否都不应回退到位置索引
        assert not mi.levels[0]._should_fallback_to_positional
        assert not mi._should_fallback_to_positional

        # 检查是否会引发 KeyError，消息包含 "14"
        with pytest.raises(KeyError, match="14"):
            ser[14]

    # ---------------------------------------------------------------------
    # 测试在多索引数据帧中使用 .loc[] 设置多个索引的部分数据
    def test_setitem_multiple_partial(self, multiindex_dataframe_random_data):
        # 复制数据帧以备期望和结果使用
        frame = multiindex_dataframe_random_data
        expected = frame.copy()
        result = frame.copy()
        # 使用 .loc[] 设置索引为 "foo" 和 "bar" 的行为 0
        result.loc[["foo", "bar"]] = 0
        # 分别设置期望中 "foo" 和 "bar" 行为 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 重新设置期望和结果数据帧
        expected = frame.copy()
        result = frame.copy()
        # 使用 .loc[] 设置索引从 "foo" 到 "bar" 的行为 0
        result.loc["foo":"bar"] = 0
        # 分别设置期望中 "foo" 和 "bar" 行为 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 从数据帧中的列"A"复制数据以备期望和结果使用
        expected = frame["A"].copy()
        result = frame["A"].copy()
        # 使用 .loc[] 设置索引为 "foo" 和 "bar" 的行为 0
        result.loc[["foo", "bar"]] = 0
        # 分别设置期望中 "foo" 和 "bar" 行为 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # 重新设置期望和结果数据列"A"
        expected = frame["A"].copy()
        result = frame["A"].copy()
        # 使用 .loc[] 设置索引从 "foo" 到 "bar" 的行为 0
        result.loc["foo":"bar"] = 0
        # 分别设置期望中 "foo" 和 "bar" 行为 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    # 使用参数化测试标记对日期时间索引的部分获取进行测试
    @pytest.mark.parametrize(
        "indexer, exp_idx, exp_values",
        [
            (
                slice("2019-2", None),
                DatetimeIndex(["2019-02-01"], dtype="M8[ns]"),
                [2, 3],
            ),
            (
                slice(None, "2019-2"),
                date_range("2019", periods=2, freq="MS"),
                [0, 1, 2, 3],
            ),
        ],
    )
    def test_partial_getitem_loc_datetime(self, indexer, exp_idx, exp_values):
        # GH: 25165
        # 创建一个日期范围为2019年，频率为月初的日期索引
        date_idx = date_range("2019", periods=2, freq="MS")
        # 创建一个多级索引数据帧，索引为日期和列为0,1
        df = DataFrame(
            list(range(4)),
            index=MultiIndex.from_product([date_idx, [0, 1]], names=["x", "y"]),
        )
        # 创建期望的数据帧，索引与给定的期望索引和值相对应
        expected = DataFrame(
            exp_values,
            index=MultiIndex.from_product([exp_idx, [0, 1]], names=["x", "y"]),
        )
        # 获取索引对应的结果数据帧，并断言与期望数据帧相等
        result = df[indexer]
        tm.assert_frame_equal(result, expected)
        # 使用 .loc[] 获取索引对应的结果数据帧，并断言与期望数据帧相等
        result = df.loc[indexer]
        tm.assert_frame_equal(result, expected)

        # 使用 .loc[] 按行(axis=0)获取索引对应的结果数据帧，并断言与期望数据帧相等
        result = df.loc(axis=0)[indexer]
        tm.assert_frame_equal(result, expected)

        # 使用 .loc[] 获取索引对应的结果数据帧，并断言与期望数据帧相等
        result = df.loc[indexer, :]
        tm.assert_frame_equal(result, expected)

        # 对数据帧进行级别交换和排序，以备期望和结果使用
        df2 = df.swaplevel(0, 1).sort_index()
        expected = expected.swaplevel(0, 1).sort_index()

        # 使用 .loc[] 按列获取索引对应的结果数据帧，并断言与期望数据帧相等
        result = df2.loc[:, indexer, :]
        tm.assert_frame_equal(result, expected)
def test_loc_getitem_partial_both_axis():
    # 定义测试函数名称，测试部分使用 loc 获取元素，涵盖两个轴向
    # 用于标记 GitHub 问题编号为 gh-12660
    iterables = [["a", "b"], [2, 1]]
    # 创建多级索引列和行
    columns = MultiIndex.from_product(iterables, names=["col1", "col2"])
    rows = MultiIndex.from_product(iterables, names=["row1", "row2"])
    # 创建一个 DataFrame，填充随机标准正态分布数据
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)), index=rows, columns=columns
    )
    # 预期结果是对 df 进行 iloc 操作选取部分数据，再用 droplevel 移除层级索引
    expected = df.iloc[:2, 2:].droplevel("row1").droplevel("col1", axis=1)
    # 使用 loc 从 df 中选择特定元素，结果存储在 result 中
    result = df.loc["a", "b"]
    # 使用测试框架 tm 进行比较，确认 result 和 expected 相等
    tm.assert_frame_equal(result, expected)
```