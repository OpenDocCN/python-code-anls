# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_is_monotonic.py`

```
# 导入必要的库和模块
import numpy as np
import pytest

# 从 pandas 中导入 DataFrame, Index, Series 类
from pandas import (
    DataFrame,
    Index,
    Series,
)

# 导入 pandas 内部测试模块
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器为 test_is_monotonic_increasing 函数添加参数化测试
@pytest.mark.parametrize(
    "in_vals, out_vals",
    [
        # Basics: strictly increasing (T), strictly decreasing (F),
        # abs val increasing (F), non-strictly increasing (T)
        ([1, 2, 5, 3, 2, 0, 4, 5, -6, 1, 1], [True, False, False, True]),
        # Test with inf vals
        (
            [1, 2.1, np.inf, 3, 2, np.inf, -np.inf, 5, 11, 1, -np.inf],
            [True, False, True, False],
        ),
        # Test with nan vals; should always be False
        (
            [1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan],
            [False, False, False, False],
        ),
    ],
)
# 定义测试函数 test_is_monotonic_increasing，用于检验数据列是否单调递增
def test_is_monotonic_increasing(in_vals, out_vals):
    # GH 17015
    # 创建包含测试数据的字典 source_dict
    source_dict = {
        "A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "B": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
        "C": in_vals,
    }
    # 使用 source_dict 创建 DataFrame 对象 df
    df = DataFrame(source_dict)
    # 对 df 根据列 'B' 进行分组，计算每组中列 'C' 是否单调递增，并返回结果
    result = df.groupby("B").C.is_monotonic_increasing
    # 创建索引对象 index，包含字符列表 ['a', 'b', 'c', 'd']，并命名为 'B'
    index = Index(list("abcd"), name="B")
    # 创建预期的 Series 对象 expected，包含索引 index 和数据 out_vals，命名为 'C'
    expected = Series(index=index, data=out_vals, name="C")
    # 断言 result 与 expected 的 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 通过应用 lambda 函数检查 result 是否与手动计算的 is_monotonic_increasing 结果相等
    expected = df.groupby(["B"]).C.apply(lambda x: x.is_monotonic_increasing)
    # 断言 result 与 expected 的 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器为 test_is_monotonic_decreasing 函数添加参数化测试
@pytest.mark.parametrize(
    "in_vals, out_vals",
    [
        # Basics: strictly decreasing (T), strictly increasing (F),
        # abs val decreasing (F), non-strictly increasing (T)
        ([10, 9, 7, 3, 4, 5, -3, 2, 0, 1, 1], [True, False, False, True]),
        # Test with inf vals
        (
            [np.inf, 1, -np.inf, np.inf, 2, -3, -np.inf, 5, -3, -np.inf, -np.inf],
            [True, True, False, True],
        ),
        # Test with nan vals; should always be False
        (
            [1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan],
            [False, False, False, False],
        ),
    ],
)
# 定义测试函数 test_is_monotonic_decreasing，用于检验数据列是否单调递减
def test_is_monotonic_decreasing(in_vals, out_vals):
    # GH 17015
    # 创建包含测试数据的字典 source_dict
    source_dict = {
        "A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "B": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
        "C": in_vals,
    }
    # 使用 source_dict 创建 DataFrame 对象 df
    df = DataFrame(source_dict)
    # 对 df 根据列 'B' 进行分组，计算每组中列 'C' 是否单调递减，并返回结果
    result = df.groupby("B").C.is_monotonic_decreasing
    # 创建索引对象 index，包含字符列表 ['a', 'b', 'c', 'd']，并命名为 'B'
    index = Index(list("abcd"), name="B")
    # 创建预期的 Series 对象 expected，包含索引 index 和数据 out_vals，命名为 'C'
    expected = Series(index=index, data=out_vals, name="C")
    # 断言 result 与 expected 的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```