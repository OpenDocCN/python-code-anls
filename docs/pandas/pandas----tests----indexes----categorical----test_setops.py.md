# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_setops.py`

```
import numpy as np
import pytest

from pandas import (
    CategoricalIndex,  # 导入 pandas 库中的 CategoricalIndex 类
    Index,              # 导入 pandas 库中的 Index 类
)
import pandas._testing as tm  # 导入 pandas 库中的测试工具模块


@pytest.mark.parametrize("na_value", [None, np.nan])
def test_difference_with_na(na_value):
    # 测试函数：测试 CategoricalIndex 的 difference 方法处理缺失值的行为
    # GH 57318  <-- GitHub 上的问题编号

    # 创建一个 CategoricalIndex 对象，包含字符串 'a', 'b', 'c' 和一个 None 值
    ci = CategoricalIndex(["a", "b", "c", None])

    # 创建一个 Index 对象，包含字符串 'c' 和一个可能的缺失值 na_value
    other = Index(["c", na_value])

    # 调用 CategoricalIndex 对象的 difference 方法，计算差集
    result = ci.difference(other)

    # 创建预期的 CategoricalIndex 对象，包含字符串 'a', 'b'，并指定所有可能的类别
    expected = CategoricalIndex(["a", "b"], categories=["a", "b", "c"])

    # 使用测试工具模块中的 assert_index_equal 方法，断言两个 Index 对象相等
    tm.assert_index_equal(result, expected)
```