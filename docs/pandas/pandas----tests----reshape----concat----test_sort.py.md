# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_sort.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库用于单元测试

import pandas as pd  # 导入 Pandas 库
from pandas import DataFrame  # 从 Pandas 中导入 DataFrame 类
import pandas._testing as tm  # 导入 Pandas 测试模块

class TestConcatSort:
    def test_concat_sorts_columns(self, sort):
        # GH-4588
        # 创建 DataFrame df1 包含列 'b' 和 'a'，并指定列的顺序
        df1 = DataFrame({"a": [1, 2], "b": [1, 2]}, columns=["b", "a"])
        # 创建 DataFrame df2 包含列 'a' 和 'c'
        df2 = DataFrame({"a": [3, 4], "c": [5, 6]})

        # 对于 sort=True/None
        # 创建期望的 DataFrame，包含列 'a', 'b', 'c'，并设定列的顺序
        expected = DataFrame(
            {"a": [1, 2, 3, 4], "b": [1, 2, None, None], "c": [None, None, 5, 6]},
            columns=["a", "b", "c"],
        )

        if sort is False:
            # 如果 sort 是 False，则调整期望的列顺序为 'b', 'a', 'c'
            expected = expected[["b", "a", "c"]]

        # 默认情况
        with tm.assert_produces_warning(None):
            # 执行 concat 操作，忽略索引，根据参数 sort 进行排序
            result = pd.concat([df1, df2], ignore_index=True, sort=sort)
        # 断言 result 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_concat_sorts_index(self, sort):
        # 创建 DataFrame df1 包含列 'a' 和索引 'c', 'a', 'b'
        df1 = DataFrame({"a": [1, 2, 3]}, index=["c", "a", "b"])
        # 创建 DataFrame df2 包含列 'b' 和索引 'a', 'b'
        df2 = DataFrame({"b": [1, 2]}, index=["a", "b"])

        # 对于 sort=True/None
        # 创建期望的 DataFrame，包含列 'a', 'b'，索引 'a', 'b', 'c'
        expected = DataFrame(
            {"a": [2, 3, 1], "b": [1, 2, None]},
            index=["a", "b", "c"],
            columns=["a", "b"],
        )
        if sort is False:
            # 如果 sort 是 False，则根据索引顺序 'c', 'a', 'b' 调整期望的 DataFrame
            expected = expected.loc[["c", "a", "b"]]

        # 警告并默认排序
        with tm.assert_produces_warning(None):
            # 执行 concat 操作，按轴 1 进行连接，根据参数 sort 进行排序
            result = pd.concat([df1, df2], axis=1, sort=sort)
        # 断言 result 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_concat_inner_sort(self, sort):
        # https://github.com/pandas-dev/pandas/pull/20613
        # 创建 DataFrame df1 包含列 'b', 'a', 'c'
        df1 = DataFrame(
            {"a": [1, 2], "b": [1, 2], "c": [1, 2]}, columns=["b", "a", "c"]
        )
        # 创建 DataFrame df2 包含列 'a', 'b'，并指定索引 3, 4
        df2 = DataFrame({"a": [1, 2], "b": [3, 4]}, index=[3, 4])

        with tm.assert_produces_warning(None):
            # 对于 inner 连接，未设置 sort 不应产生警告，因为它不会进行排序
            result = pd.concat([df1, df2], sort=sort, join="inner", ignore_index=True)

        # 创建期望的 DataFrame 包含列 'a', 'b'，并根据 sort 参数调整列的顺序
        expected = DataFrame({"b": [1, 2, 3, 4], "a": [1, 2, 1, 2]}, columns=["b", "a"])
        if sort is True:
            expected = expected[["a", "b"]]
        # 断言 result 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_concat_aligned_sort(self):
        # GH-4588
        # 创建 DataFrame df 包含列 'c', 'b', 'a'
        df = DataFrame({"c": [1, 2], "b": [3, 4], "a": [5, 6]}, columns=["c", "b", "a"])
        # 执行 concat 操作，忽略索引，根据参数 sort=True 进行排序
        result = pd.concat([df, df], sort=True, ignore_index=True)
        # 创建期望的 DataFrame 包含列 'a', 'b', 'c'，并设定列的顺序
        expected = DataFrame(
            {"a": [5, 6, 5, 6], "b": [3, 4, 3, 4], "c": [1, 2, 1, 2]},
            columns=["a", "b", "c"],
        )
        # 断言 result 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 执行 concat 操作，内连接，根据参数 sort=True 进行排序，忽略索引
        result = pd.concat(
            [df, df[["c", "b"]]], join="inner", sort=True, ignore_index=True
        )
        # 创建期望的 DataFrame 包含列 'b', 'c'，并根据 sort 参数调整列的顺序
        expected = expected[["b", "c"]]
        # 断言 result 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    def test_concat_aligned_sort_does_not_raise(self):
        # GH-4588
        # 在此测试中，我们验证排序时不会因为内部排序而引发 TypeError 异常。
        # 创建一个包含整数和字符串列的 DataFrame
        df = DataFrame({1: [1, 2], "a": [3, 4]}, columns=[1, "a"])
        # 期望的结果 DataFrame，包含两次复制原 DataFrame 的内容
        expected = DataFrame({1: [1, 2, 1, 2], "a": [3, 4, 3, 4]}, columns=[1, "a"])
        # 进行 concat 操作，忽略索引，进行排序
        result = pd.concat([df, df], ignore_index=True, sort=True)
        # 使用测试工具检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    def test_concat_frame_with_sort_false(self):
        # GH 43375
        # 创建一个包含多个 DataFrame 的列表，并进行 concat 操作，sort 参数设为 False
        result = pd.concat(
            [DataFrame({i: i}, index=[i]) for i in range(2, 0, -1)], sort=False
        )
        # 期望的结果 DataFrame，包含两个 DataFrame 的组合
        expected = DataFrame([[2, np.nan], [np.nan, 1]], index=[2, 1], columns=[2, 1])

        # 使用测试工具检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # GH 37937
        # 创建两个 DataFrame，并进行按列拼接操作，sort 参数设为 False
        df1 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[1, 2, 3])
        df2 = DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]}, index=[3, 1, 6])
        result = pd.concat([df2, df1], axis=1, sort=False)
        # 期望的结果 DataFrame，按指定的索引和列名排列
        expected = DataFrame(
            [
                [7.0, 10.0, 3.0, 6.0],
                [8.0, 11.0, 1.0, 4.0],
                [9.0, 12.0, np.nan, np.nan],
                [np.nan, np.nan, 2.0, 5.0],
            ],
            index=[3, 1, 6, 2],
            columns=["c", "d", "a", "b"],
        )
        # 使用测试工具检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    def test_concat_sort_none_raises(self):
        # GH#41518
        # 创建一个简单的 DataFrame
        df = DataFrame({1: [1, 2], "a": [3, 4]})
        # 准备用于引发 ValueError 异常的错误消息
        msg = "The 'sort' keyword only accepts boolean values; None was passed."
        # 使用 pytest 的上下文管理器检查是否正确引发了 ValueError 异常，并且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            pd.concat([df, df], sort=None)
```