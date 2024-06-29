# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_series.py`

```
import numpy as np  # 导入 NumPy 库，用于支持数组和矩阵运算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入多个子模块和类
    DataFrame,  # 用于处理表格数据的主要数据结构
    DatetimeIndex,  # 日期时间索引类，支持时间序列数据的索引
    Index,  # pandas 的索引对象，用于标签索引
    MultiIndex,  # 多级索引对象，支持多层次索引
    Series,  # pandas 的一维标记数组数据结构
    concat,  # 用于沿指定轴连接 pandas 对象（如 Series 和 DataFrame）
    date_range,  # 生成日期范围的函数
)
import pandas._testing as tm  # 导入 pandas 测试工具模块

class TestSeriesConcat:
    def test_concat_series(self):
        ts = Series(  # 创建一个 Series 对象
            np.arange(20, dtype=np.float64),  # 用 NumPy 数组生成一组浮点数
            index=date_range("2020-01-01", periods=20),  # 指定日期时间索引
            name="foo",  # 指定 Series 的名称
        )
        ts.name = "foo"  # 修改 Series 的名称

        pieces = [ts[:5], ts[5:15], ts[15:]]  # 将 Series 拆分为几部分

        result = concat(pieces)  # 将拆分的部分连接成一个新的 Series
        tm.assert_series_equal(result, ts)  # 断言连接结果与原始 Series 相等
        assert result.name == ts.name  # 断言连接结果的名称与原始 Series 相同

        result = concat(pieces, keys=[0, 1, 2])  # 在连接时指定多级索引的键
        expected = ts.copy()  # 复制原始 Series 对象
        exp_codes = [np.repeat([0, 1, 2], [len(x) for x in pieces]), np.arange(len(ts))]  # 创建多级索引的编码
        exp_index = MultiIndex(  # 创建多级索引对象
            levels=[[0, 1, 2], DatetimeIndex(ts.index.to_numpy(dtype="M8[ns]"))],  # 设置索引的层级和标签
            codes=exp_codes,  # 设置索引的编码
        )
        expected.index = exp_index  # 将预期的多级索引应用到预期结果上
        tm.assert_series_equal(result, expected)  # 断言连接结果与预期结果相等

    def test_concat_empty_and_non_empty_series_regression(self):
        # GH 18187 regression test，检验空和非空 Series 的连接
        s1 = Series([1])  # 创建一个包含单个元素的 Series
        s2 = Series([], dtype=object)  # 创建一个空的对象类型 Series

        expected = s1.astype(object)  # 将 s1 转换为对象类型
        result = concat([s1, s2])  # 连接两个 Series
        tm.assert_series_equal(result, expected)  # 断言连接结果与预期结果相等

    def test_concat_series_axis1(self):
        ts = Series(  # 创建一个 Series 对象
            np.arange(10, dtype=np.float64),  # 用 NumPy 数组生成一组浮点数
            index=date_range("2020-01-01", periods=10),  # 指定日期时间索引
        )

        pieces = [ts[:-2], ts[2:], ts[2:-2]]  # 将 Series 按索引切片成几部分

        result = concat(pieces, axis=1)  # 沿着 axis=1 方向连接切片后的 Series
        expected = DataFrame(pieces).T  # 创建预期的 DataFrame，并进行转置操作
        tm.assert_frame_equal(result, expected)  # 断言连接结果与预期结果相等

        result = concat(pieces, keys=["A", "B", "C"], axis=1)  # 在连接时指定多级索引的键和 axis=1
        expected = DataFrame(pieces, index=["A", "B", "C"]).T  # 创建预期的 DataFrame，并进行转置操作
        tm.assert_frame_equal(result, expected)  # 断言连接结果与预期结果相等

    def test_concat_series_axis1_preserves_series_names(self):
        # preserve series names, #2489，保留 Series 的名称
        s = Series(np.random.default_rng(2).standard_normal(5), name="A")  # 创建一个带有名称的 Series 对象
        s2 = Series(np.random.default_rng(2).standard_normal(5), name="B")  # 创建另一个带有名称的 Series 对象

        result = concat([s, s2], axis=1)  # 沿 axis=1 方向连接两个 Series
        expected = DataFrame({"A": s, "B": s2})  # 创建预期的 DataFrame
        tm.assert_frame_equal(result, expected)  # 断言连接结果与预期结果相等

        s2.name = None  # 将 s2 的名称设为 None
        result = concat([s, s2], axis=1)  # 再次沿 axis=1 方向连接两个 Series
        tm.assert_index_equal(result.columns, Index(["A", 0], dtype="object"))  # 断言连接结果的列索引与预期相等

    def test_concat_series_axis1_with_reindex(self, sort):
        # must reindex, #2603，必须重新索引
        s = Series(  # 创建一个 Series 对象
            np.random.default_rng(2).standard_normal(3),  # 用 NumPy 数组生成一组标准正态分布的随机数
            index=["c", "a", "b"],  # 指定索引标签
            name="A",  # 指定 Series 的名称
        )
        s2 = Series(  # 创建另一个 Series 对象
            np.random.default_rng(2).standard_normal(4),  # 用 NumPy 数组生成一组标准正态分布的随机数
            index=["d", "a", "b", "c"],  # 指定索引标签
            name="B",  # 指定 Series 的名称
        )
        result = concat([s, s2], axis=1, sort=sort)  # 沿 axis=1 方向连接两个 Series，并指定排序选项
        expected = DataFrame({"A": s, "B": s2}, index=["c", "a", "b", "d"])  # 创建预期的 DataFrame
        if sort:  # 如果 sort 为 True
            expected = expected.sort_index()  # 对预期结果按索引排序
        tm.assert_frame_equal(result, expected)  # 断言连接结果与预期结果相等
    def test_concat_series_axis1_names_applied(self):
        # 确保在 axis=1 上不忽略 names 参数，参考 issue #23490
        s = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        # 进行 axis=1 的连接操作，指定 keys 为 ["a", "b"]，并设置 names=["A"]
        result = concat([s, s2], axis=1, keys=["a", "b"], names=["A"])
        # 期望结果是一个 DataFrame，列为 ["a", "b"]，并命名为 "A"
        expected = DataFrame(
            [[1, 4], [2, 5], [3, 6]], columns=Index(["a", "b"], name="A")
        )
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 进行 axis=1 的连接操作，指定 keys 为 [("a", 1), ("b", 2)]，并设置 names=["A", "B"]
        result = concat([s, s2], axis=1, keys=[("a", 1), ("b", 2)], names=["A", "B"])
        # 期望结果是一个 DataFrame，列为 [("a", 1), ("b", 2)]，并设置 names=["A", "B"]
        expected = DataFrame(
            [[1, 4], [2, 5], [3, 6]],
            columns=MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["A", "B"]),
        )
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_same_names_ignore_index(self):
        dates = date_range("01-Jan-2013", "01-Jan-2014", freq="MS")[0:-1]
        s1 = Series(
            np.random.default_rng(2).standard_normal(len(dates)),
            index=dates,
            name="value",
        )
        s2 = Series(
            np.random.default_rng(2).standard_normal(len(dates)),
            index=dates,
            name="value",
        )

        # 进行 axis=1 的连接操作，忽略 index
        result = concat([s1, s2], axis=1, ignore_index=True)
        # 期望结果的列名为一个 RangeIndex
        expected = Index(range(2))
        # 断言结果的列与期望相等
        tm.assert_index_equal(result.columns, expected, exact=True)

    @pytest.mark.parametrize("s1name", [np.int64(190), 190])
    def test_concat_series_name_npscalar_tuple(self, s1name):
        # GH21015
        s2name = (43, 0)
        # 创建带有不同命名方式的 Series 对象 s1 和 s2
        s1 = Series({"a": 1, "b": 2}, name=s1name)
        s2 = Series({"c": 5, "d": 6}, name=s2name)
        # 进行 Series 的连接操作
        result = concat([s1, s2])
        # 期望结果是一个包含所有数据的 Series 对象
        expected = Series({"a": 1, "b": 2, "c": 5, "d": 6})
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_concat_series_partial_columns_names(self):
        # GH10698
        named_series = Series([1, 2], name="foo")
        unnamed_series1 = Series([1, 2])
        unnamed_series2 = Series([4, 5])

        # 进行 axis=1 的连接操作，结果中列名由 ["foo", 0, 1] 组成
        result = concat([named_series, unnamed_series1, unnamed_series2], axis=1)
        # 期望结果是一个 DataFrame，包含列名 ["foo", 0, 1]
        expected = DataFrame(
            {"foo": [1, 2], 0: [1, 2], 1: [4, 5]}, columns=["foo", 0, 1]
        )
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 进行 axis=1 的连接操作，设置 keys=["red", "blue", "yellow"]
        result = concat(
            [named_series, unnamed_series1, unnamed_series2],
            axis=1,
            keys=["red", "blue", "yellow"],
        )
        # 期望结果是一个 DataFrame，列名为 ["red", "blue", "yellow"]
        expected = DataFrame(
            {"red": [1, 2], "blue": [1, 2], "yellow": [4, 5]},
            columns=["red", "blue", "yellow"],
        )
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 进行 axis=1 的连接操作，忽略 index
        result = concat(
            [named_series, unnamed_series1, unnamed_series2], axis=1, ignore_index=True
        )
        # 期望结果是一个 DataFrame，列名为 RangeIndex
        expected = DataFrame({0: [1, 2], 1: [1, 2], 2: [4, 5]})
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

    def test_concat_series_length_one_reversed(self, frame_or_series):
        # GH39401
        obj = frame_or_series([100])
        # 对长度为一的 Series 进行反向连接操作
        result = concat([obj.iloc[::-1]])
        # 断言结果与原始对象相等
        tm.assert_equal(result, obj)
```