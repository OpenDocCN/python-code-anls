# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_insert.py`

```
"""
test_insert is specifically for the DataFrame.insert method; not to be
confused with tests with "insert" in their names that are really testing
__setitem__.
"""

# 引入所需的库和模块
import numpy as np  # 引入 NumPy 库
import pytest  # 引入 pytest 库

# 引入 pandas 库中的特定错误和类
from pandas.errors import PerformanceWarning
from pandas import (
    DataFrame,  # 引入 DataFrame 类
    Index,  # 引入 Index 类
)
import pandas._testing as tm  # 引入 pandas 测试模块中的 tm 别名

# 定义 TestDataFrameInsert 类，用于测试 DataFrame.insert 方法
class TestDataFrameInsert:
    
    # 测试 DataFrame.insert 方法
    def test_insert(self):
        # 创建一个 DataFrame 对象，包含随机数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=np.arange(5),
            columns=["c", "b", "a"],
        )

        # 在列索引 0 处插入新列 'foo'，内容为已有列 'a' 的数据
        df.insert(0, "foo", df["a"])
        # 断言新的列索引顺序为 ['foo', 'c', 'b', 'a']
        tm.assert_index_equal(df.columns, Index(["foo", "c", "b", "a"]))
        # 断言新插入的列 'foo' 数据与原有列 'a' 数据相同
        tm.assert_series_equal(df["a"], df["foo"], check_names=False)

        # 在列索引 2 处插入新列 'bar'，内容为已有列 'c' 的数据
        df.insert(2, "bar", df["c"])
        # 断言新的列索引顺序为 ['foo', 'c', 'bar', 'b', 'a']
        tm.assert_index_equal(df.columns, Index(["foo", "c", "bar", "b", "a"]))
        # 断言新插入的列 'bar' 数据与原有列 'c' 数据相同
        tm.assert_almost_equal(df["c"], df["bar"], check_names=False)

        # 测试插入重复列名 'a'，应抛出 ValueError 异常
        with pytest.raises(ValueError, match="already exists"):
            df.insert(1, "a", df["b"])

        # 测试插入重复列名 'c'，应抛出 ValueError 异常
        msg = "cannot insert c, already exists"
        with pytest.raises(ValueError, match=msg):
            df.insert(1, "c", df["b"])

        # 设置列名字段 'some_name'
        df.columns.name = "some_name"
        # 在列索引 0 处插入新列 'baz'，内容为已有列 'c' 的数据
        df.insert(0, "baz", df["c"])
        # 断言列名字段 'some_name' 未被修改
        assert df.columns.name == "some_name"

    # 测试 DataFrame.insert 方法中的 bug 修复，用例编号 4032
    def test_insert_column_bug_4032(self):
        # GH#4032，插入一列并进行重命名导致错误
        df = DataFrame({"b": [1.1, 2.2]})

        df = df.rename(columns={})
        df.insert(0, "a", [1, 2])
        result = df.rename(columns={})

        expected = DataFrame([[1, 1.1], [2, 2.2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        df.insert(0, "c", [1.3, 2.3])
        result = df.rename(columns={})

        expected = DataFrame([[1.3, 1, 1.1], [2.3, 2, 2.2]], columns=["c", "a", "b"])
        tm.assert_frame_equal(result, expected)

    # 测试允许插入重复列名的情况，用例编号 14291
    def test_insert_with_columns_dups(self):
        # GH#14291
        df = DataFrame()
        df.insert(0, "A", ["g", "h", "i"], allow_duplicates=True)
        df.insert(0, "A", ["d", "e", "f"], allow_duplicates=True)
        df.insert(0, "A", ["a", "b", "c"], allow_duplicates=True)
        exp = DataFrame(
            [["a", "d", "g"], ["b", "e", "h"], ["c", "f", "i"]], columns=["A", "A", "A"]
        )
        tm.assert_frame_equal(df, exp)

    # 测试 DataFrame.insert 方法中的性能警告
    def test_insert_item_cache(self, performance_warning):
        # 创建一个具有随机数据的 DataFrame 对象
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        ser = df[0]
        expected_warning = PerformanceWarning if performance_warning else None

        # 断言在插入大量列时会产生性能警告
        with tm.assert_produces_warning(expected_warning):
            for n in range(100):
                df[n + 3] = df[1] * n

        # 修改原 DataFrame 的一部分数据
        ser.iloc[0] = 99
        # 断言修改后的值与预期不同
        assert df.iloc[0, 0] == df[0][0]
        assert df.iloc[0, 0] != 99
    # 测试插入操作，验证使用 EAs 时不会触发关于分段帧的性能警告
    def test_insert_EA_no_warning(self):
        # 创建一个包含随机整数的 DataFrame，大小为 3x100，数据类型为 Int64
        df = DataFrame(
            np.random.default_rng(2).integers(0, 100, size=(3, 100)), dtype="Int64"
        )
        # 使用上下文管理器验证不会产生任何警告
        with tm.assert_produces_warning(None):
            # 在 DataFrame 中插入名为 "a" 的新列，内容为数组 [1, 2, 3]
            df["a"] = np.array([1, 2, 3])

    # 测试 DataFrame 的插入操作
    def test_insert_frame(self):
        # 创建一个包含两列的 DataFrame，列名为 "col1" 和 "col2"
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # 错误消息，指示期望的对象是一维的，但实际传入的是有两列的 DataFrame
        msg = (
            "Expected a one-dimensional object, got a DataFrame with 2 columns instead."
        )
        # 使用 pytest 验证插入操作会引发 ValueError，并且错误消息与预期匹配
        with pytest.raises(ValueError, match=msg):
            # 在第 1 列（索引为 1）插入名为 "newcol" 的新列，内容为整个 DataFrame df
            df.insert(1, "newcol", df)

    # 测试在指定位置插入整数列的操作
    def test_insert_int64_loc(self):
        # 创建一个包含一列 "a" 的 DataFrame
        df = DataFrame({"a": [1, 2]})
        # 在整数位置 0（类型为 np.int64）插入名为 "b" 的新列，内容为整数 0
        df.insert(np.int64(0), "b", 0)
        # 使用 tm.assert_frame_equal 验证插入后 DataFrame df 符合预期
        tm.assert_frame_equal(df, DataFrame({"b": [0, 0], "a": [1, 2]}))
```