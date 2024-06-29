# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_pop.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 导入 Pandas 库中的特定模块
    DataFrame,       # DataFrame 数据结构，用于表格数据
    MultiIndex,      # 多级索引，用于复杂索引结构
    Series,          # Series 数据结构，用于一维标签化数据
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


class TestDataFramePop:
    def test_pop(self, float_frame):
        float_frame.columns.name = "baz"  # 设置 DataFrame 列名为 "baz"

        float_frame.pop("A")  # 弹出列名为 "A" 的列
        assert "A" not in float_frame  # 断言确保列 "A" 不在 DataFrame 中

        float_frame["foo"] = "bar"  # 在 DataFrame 中添加新列 "foo" 并赋值为 "bar"
        float_frame.pop("foo")  # 弹出列名为 "foo" 的列
        assert "foo" not in float_frame  # 断言确保列 "foo" 不在 DataFrame 中
        assert float_frame.columns.name == "baz"  # 断言确保列名仍为 "baz"

        # gh-10912: inplace ops cause caching issue
        a = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"], index=["X", "Y"])
        b = a.pop("B")  # 弹出列名为 "B" 的列，并将其赋值给变量 b
        b += 1  # 对变量 b 中的数据进行修改

        # 原始 DataFrame
        expected = DataFrame([[1, 3], [4, 6]], columns=["A", "C"], index=["X", "Y"])
        tm.assert_frame_equal(a, expected)  # 使用测试模块验证 DataFrame a 与期望结果的一致性

        # 结果
        expected = Series([2, 5], index=["X", "Y"], name="B") + 1  # 创建预期的 Series 对象
        tm.assert_series_equal(b, expected)  # 使用测试模块验证 Series b 与期望结果的一致性

    def test_pop_non_unique_cols(self):
        df = DataFrame({0: [0, 1], 1: [0, 1], 2: [4, 5]})
        df.columns = ["a", "b", "a"]  # 设置 DataFrame 的列名，其中有重复的列名 "a"

        res = df.pop("a")  # 弹出列名为 "a" 的列，并将结果赋给变量 res
        assert type(res) == DataFrame  # 断言确保 res 是 DataFrame 类型
        assert len(res) == 2  # 断言确保 res 的长度为 2
        assert len(df.columns) == 1  # 断言确保 df 的列数为 1
        assert "b" in df.columns  # 断言确保 "b" 列在 df 的列中
        assert "a" not in df.columns  # 断言确保 "a" 列不在 df 的列中
        assert len(df.index) == 2  # 断言确保 df 的行数为 2

    def test_mixed_depth_pop(self):
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)  # 创建多级索引对象
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        df1 = df.copy()  # 复制 DataFrame 到 df1
        df2 = df.copy()  # 复制 DataFrame 到 df2
        result = df1.pop("a")  # 弹出名为 "a" 的列，并将结果赋给 result
        expected = df2.pop(("a", "", ""))  # 弹出指定多级索引 ("a", "", "") 的列，并将结果赋给 expected
        tm.assert_series_equal(expected, result, check_names=False)  # 使用测试模块验证 result 与 expected 的一致性，忽略名称检查
        tm.assert_frame_equal(df1, df2)  # 使用测试模块验证 df1 与 df2 的一致性
        assert result.name == "a"  # 断言确保 result 的名称为 "a"

        expected = df1["top"]  # 获取 df1 中 "top" 列的数据
        df1 = df1.drop(["top"], axis=1)  # 在 df1 中删除 "top" 列
        result = df2.pop("top")  # 在 df2 中弹出 "top" 列
        tm.assert_frame_equal(expected, result)  # 使用测试模块验证 expected 与 result 的一致性
        tm.assert_frame_equal(df1, df2)  # 使用测试模块验证 df1 与 df2 的一致性
```