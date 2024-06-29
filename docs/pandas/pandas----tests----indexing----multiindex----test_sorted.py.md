# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_sorted.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入以下模块：
    NA,  # 缺失值的常量
    DataFrame,  # 数据帧结构，用于处理二维表格数据
    MultiIndex,  # 多级索引对象，用于层次化索引
    Series,  # 序列对象，用于一维标签数据结构
    array,  # 用于创建 Pandas 对象的数组
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块


class TestMultiIndexSorted:
    def test_getitem_multilevel_index_tuple_not_sorted(self):
        index_columns = list("abc")  # 创建包含字符 'a', 'b', 'c' 的列表
        df = DataFrame(  # 创建数据帧对象 df
            [[0, 1, 0, "x"], [0, 0, 1, "y"]], columns=index_columns + ["data"]
        )
        df = df.set_index(index_columns)  # 将 index_columns 设置为 df 的索引
        query_index = df.index[:1]  # 获取 df 索引的前一项作为查询索引
        rs = df.loc[query_index, "data"]  # 根据查询索引和列名提取数据

        xp_idx = MultiIndex.from_tuples([(0, 1, 0)], names=["a", "b", "c"])  # 创建多级索引对象 xp_idx
        xp = Series(["x"], index=xp_idx, name="data")  # 创建 Series 对象 xp
        tm.assert_series_equal(rs, xp)  # 使用测试工具比较 rs 和 xp 的一致性

    def test_getitem_slice_not_sorted(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data  # 多级索引数据帧赋值给 frame
        df = frame.sort_index(level=1).T  # 对索引的第二级进行排序，并转置数据帧 df

        # buglet with int typechecking
        result = df.iloc[:, : np.int32(3)]  # 使用整数类型检查处理 df 的切片操作
        expected = df.reindex(columns=df.columns[:3])  # 重新索引列到前三列
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较 result 和 expected 的一致性

    @pytest.mark.parametrize("key", [None, lambda x: x])
    def test_frame_getitem_not_sorted2(self, key):
        # 13431
        df = DataFrame(  # 创建数据帧对象 df
            {
                "col1": ["b", "d", "b", "a"],
                "col2": [3, 1, 1, 2],
                "data": ["one", "two", "three", "four"],
            }
        )

        df2 = df.set_index(["col1", "col2"])  # 使用列 'col1' 和 'col2' 设置 df2 的索引
        df2_original = df2.copy()  # 创建 df2 的副本 df2_original

        df2.index = df2.index.set_levels(["b", "d", "a"], level="col1")  # 设置索引 'col1' 的级别为 ['b', 'd', 'a']
        df2.index = df2.index.set_codes([0, 1, 0, 2], level="col1")  # 设置索引 'col1' 的代码为 [0, 1, 0, 2]
        assert not df2.index.is_monotonic_increasing  # 断言索引不是单调递增的

        assert df2_original.index.equals(df2.index)  # 断言 df2_original 的索引与 df2 的索引相等
        expected = df2.sort_index(key=key)  # 根据给定的键对 df2 进行排序，生成期望的排序结果
        assert expected.index.is_monotonic_increasing  # 断言期望的索引是单调递增的

        result = df2.sort_index(level=0, key=key)  # 根据第一级索引和给定的键对 df2 进行排序，生成结果
        assert result.index.is_monotonic_increasing  # 断言结果的索引是单调递增的
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较 result 和 expected 的一致性

    def test_sort_values_key(self):
        arrays = [  # 创建包含多个列表的数组
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays)  # 使用 zip 函数将多个数组打包成元组
        index = MultiIndex.from_tuples(tuples)  # 从元组创建多级索引对象
        index = index.sort_values(  # 根据第三个字母排序索引
            key=lambda x: x.map(lambda entry: entry[2])
        )
        result = DataFrame(range(8), index=index)  # 创建数据帧对象 result，指定索引和数据范围

        arrays = [  # 创建另一个包含多个列表的数组
            ["foo", "foo", "bar", "bar", "qux", "qux", "baz", "baz"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays)  # 使用 zip 函数将多个数组打包成元组
        expected = DataFrame(range(8), index=MultiIndex.from_tuples(tuples))  # 创建期望的数据帧对象

        tm.assert_frame_equal(result, expected)  # 使用测试工具比较 result 和 expected 的一致性
    # 定义一个测试方法，用于测试带有缺失值的 argsort 函数
    def test_argsort_with_na(self):
        # GH48495：GitHub 上的 issue 编号
        # 创建包含两个 Int64 类型数组的列表，其中一个数组包含 NA（缺失值）
        arrays = [
            array([2, NA, 1], dtype="Int64"),
            array([1, 2, 3], dtype="Int64"),
        ]
        # 使用数组创建一个多级索引对象
        index = MultiIndex.from_arrays(arrays)
        # 对索引进行排序并返回排序后的结果
        result = index.argsort()
        # 预期的排序结果，使用 np.intp 类型的数组存储
        expected = np.array([2, 0, 1], dtype=np.intp)
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，用于测试带有缺失值的 sort_values 函数
    def test_sort_values_with_na(self):
        # GH48495：GitHub 上的 issue 编号
        # 创建包含两个 Int64 类型数组的列表，其中一个数组包含 NA（缺失值）
        arrays = [
            array([2, NA, 1], dtype="Int64"),
            array([1, 2, 3], dtype="Int64"),
        ]
        # 使用数组创建一个多级索引对象
        index = MultiIndex.from_arrays(arrays)
        # 对索引进行排序并更新原来的索引对象
        index = index.sort_values()
        # 创建一个 DataFrame 对象，其中包含从 0 到 2 的整数，使用排序后的索引作为行索引
        result = DataFrame(range(3), index=index)

        # 创建另一个包含两个 Int64 类型数组的列表，其中一个数组包含 NA（缺失值）
        arrays = [
            array([1, 2, NA], dtype="Int64"),
            array([3, 1, 2], dtype="Int64"),
        ]
        # 使用数组创建一个多级索引对象
        index = MultiIndex.from_arrays(arrays)
        # 创建预期的 DataFrame 对象，包含从 0 到 2 的整数，使用未排序的索引作为行索引
        expected = DataFrame(range(3), index=index)

        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试未排序状态下的 DataFrame 的获取操作
    def test_frame_getitem_not_sorted(self, multiindex_dataframe_random_data):
        # 获取一个多级索引 DataFrame 对象
        frame = multiindex_dataframe_random_data
        # 转置 DataFrame 对象
        df = frame.T
        # 在 DataFrame 中插入一列名为 "foo"，值为 "foo"
        df["foo", "four"] = "foo"

        # 创建一个包含所有列名的列表，每个列表包含一个 numpy 数组
        arrays = [np.array(x) for x in zip(*df.columns.values)]

        # 使用列名 "foo" 获取 DataFrame 中的一列数据
        result = df["foo"]
        # 使用 loc 方法获取 DataFrame 中所有行的 "foo" 列数据
        result2 = df.loc[:, "foo"]
        # 根据条件重新索引 DataFrame，保留包含 "foo" 列数据的列
        expected = df.reindex(columns=df.columns[arrays[0] == "foo"])
        # 删除列的第一级索引
        expected.columns = expected.columns.droplevel(0)
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # 转置 DataFrame 对象
        df = df.T
        # 使用 xs 方法获取包含 "foo" 标签的行数据
        result = df.xs("foo")
        # 使用 loc 方法获取标签为 "foo" 的所有行数据
        result2 = df.loc["foo"]
        # 根据条件重新索引 DataFrame，保留包含 "foo" 标签的行
        expected = df.reindex(df.index[arrays[0] == "foo"])
        # 删除行的第一级索引
        expected.index = expected.index.droplevel(0)
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    # 定义一个测试方法，用于测试未排序状态下的 Series 的获取操作
    def test_series_getitem_not_sorted(self):
        # 创建包含两个字符串数组的列表
        arrays = [
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        # 将数组转换为元组
        tuples = zip(*arrays)
        # 使用元组创建一个多级索引对象
        index = MultiIndex.from_tuples(tuples)
        # 创建一个 Series 对象，包含标准正态分布的随机数，使用索引对象作为索引
        s = Series(np.random.default_rng(2).standard_normal(8), index=index)

        # 创建一个包含所有索引值的列表，每个列表包含一个 numpy 数组
        arrays = [np.array(x) for x in zip(*index.values)]

        # 使用索引标签 "qux" 获取 Series 中的数据
        result = s["qux"]
        # 使用 loc 方法获取标签为 "qux" 的所有数据
        result2 = s.loc["qux"]
        # 根据条件获取包含 "qux" 标签的 Series 数据
        expected = s[arrays[0] == "qux"]
        # 删除索引的第一级索引
        expected.index = expected.index.droplevel(0)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
```