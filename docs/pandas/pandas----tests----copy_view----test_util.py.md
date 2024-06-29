# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_util.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类，用于处理数据表格
from pandas.tests.copy_view.util import get_array  # 从 pandas 的测试工具包中导入 get_array 函数，用于获取数据数组


def test_get_array_numpy():
    df = DataFrame({"a": [1, 2, 3]})
    assert np.shares_memory(get_array(df, "a"), get_array(df, "a"))
    # 断言两个数组是否共享内存，检查 get_array 函数的行为是否符合预期


def test_get_array_masked():
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    assert np.shares_memory(get_array(df, "a"), get_array(df, "a"))
    # 断言两个数组是否共享内存，检查 get_array 函数对于含有缺失值的 DataFrame 的行为是否符合预期
```