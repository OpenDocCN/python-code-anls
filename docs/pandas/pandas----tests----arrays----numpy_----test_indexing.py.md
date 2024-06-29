# `D:\src\scipysrc\pandas\pandas\tests\arrays\numpy_\test_indexing.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas.core.dtypes.common import is_scalar  # 从 Pandas 库中导入 is_scalar 函数，用于检查是否为标量值

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 测试工具模块，用于测试辅助功能


class TestSearchsorted:
    def test_searchsorted_string(self, string_dtype):
        arr = pd.array(["a", "b", "c"], dtype=string_dtype)  # 创建包含字符串的 Pandas 数组

        result = arr.searchsorted("a", side="left")  # 在数组中搜索值"a"，返回插入点索引（左侧）
        assert is_scalar(result)  # 断言结果是标量值
        assert result == 0  # 断言插入点索引为0，即"a"在数组中的位置

        result = arr.searchsorted("a", side="right")  # 在数组中搜索值"a"，返回插入点索引（右侧）
        assert is_scalar(result)  # 断言结果是标量值
        assert result == 1  # 断言插入点索引为1，即"a"插入后的位置

    def test_searchsorted_numeric_dtypes_scalar(self, any_real_numpy_dtype):
        arr = pd.array([1, 3, 90], dtype=any_real_numpy_dtype)  # 创建包含数值的 Pandas 数组
        result = arr.searchsorted(30)  # 在数组中搜索值30，返回插入点索引
        assert is_scalar(result)  # 断言结果是标量值
        assert result == 2  # 断言插入点索引为2，即30应该插入的位置

        result = arr.searchsorted([30])  # 在数组中搜索值列表[30]，返回插入点索引数组
        expected = np.array([2], dtype=np.intp)  # 预期的插入点索引数组
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具函数检查结果与预期是否一致

    def test_searchsorted_numeric_dtypes_vector(self, any_real_numpy_dtype):
        arr = pd.array([1, 3, 90], dtype=any_real_numpy_dtype)  # 创建包含数值的 Pandas 数组
        result = arr.searchsorted([2, 30])  # 在数组中搜索值列表[2, 30]，返回插入点索引数组
        expected = np.array([1, 2], dtype=np.intp)  # 预期的插入点索引数组
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具函数检查结果与预期是否一致

    def test_searchsorted_sorter(self, any_real_numpy_dtype):
        arr = pd.array([3, 1, 2], dtype=any_real_numpy_dtype)  # 创建包含数值的 Pandas 数组
        result = arr.searchsorted([0, 3], sorter=np.argsort(arr))  # 在数组中搜索值列表[0, 3]，使用指定的排序器进行排序
        expected = np.array([0, 2], dtype=np.intp)  # 预期的插入点索引数组
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具函数检查结果与预期是否一致
```