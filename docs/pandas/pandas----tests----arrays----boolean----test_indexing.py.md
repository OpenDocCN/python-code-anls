# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_indexing.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块

# 使用 Pytest 的参数化装饰器，定义参数化测试函数，测试设置缺失值的情况
@pytest.mark.parametrize("na", [None, np.nan, pd.NA])
def test_setitem_missing_values(na):
    arr = pd.array([True, False, None], dtype="boolean")  # 创建一个 Pandas 数组，包含布尔值和缺失值
    expected = pd.array([True, None, None], dtype="boolean")  # 期望的结果数组，设置了相应的缺失值
    arr[1] = na  # 将数组中索引为 1 的位置设置为 na （None, np.nan, pd.NA 中的一个）
    tm.assert_extension_array_equal(arr, expected)  # 使用 Pandas 测试工具断言，检查设置后的数组是否符合期望
```