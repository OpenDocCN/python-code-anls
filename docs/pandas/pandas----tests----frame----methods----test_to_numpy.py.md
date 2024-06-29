# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_to_numpy.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据

from pandas import (  # 从 Pandas 库中导入以下模块
    DataFrame,       # DataFrame：用于处理二维数据
    Timestamp,       # Timestamp：用于处理时间戳
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestToNumpy:
    def test_to_numpy(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})  # 创建一个包含两列的 DataFrame 对象
        expected = np.array([[1, 3], [2, 4.5]])        # 创建预期的 NumPy 数组
        result = df.to_numpy()                        # 调用 DataFrame 的方法将其转换为 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等

    def test_to_numpy_dtype(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})     # 创建一个包含两列的 DataFrame 对象
        expected = np.array([[1, 3], [2, 4]], dtype="int64")  # 创建预期的 NumPy 数组，指定数据类型为 int64
        result = df.to_numpy(dtype="int64")             # 调用 DataFrame 的方法将其转换为指定数据类型的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)   # 断言结果与预期相等

    def test_to_numpy_copy(self):
        arr = np.random.default_rng(2).standard_normal((4, 3))  # 生成一个随机的 NumPy 数组
        df = DataFrame(arr)                          # 使用 NumPy 数组创建 DataFrame 对象
        assert df.values.base is not arr             # 检查 DataFrame 的数据不与原始数组相同
        assert df.to_numpy(copy=False).base is df.values.base  # 检查返回的 NumPy 数组与 DataFrame 数据共享相同的基础数据
        assert df.to_numpy(copy=True).base is not arr  # 检查返回的 NumPy 数组与原始数组不共享相同的基础数据

        # 当 na_value=np.nan 被传递时，我们仍然不希望复制数据，
        # 这是可以被遵守的，因为我们已经是 NumPy 浮点数
        assert df.to_numpy(copy=False).base is df.values.base  # 再次检查返回的 NumPy 数组与 DataFrame 数据共享相同的基础数据

    def test_to_numpy_mixed_dtype_to_str(self):
        # https://github.com/pandas-dev/pandas/issues/35455
        df = DataFrame([[Timestamp("2020-01-01 00:00:00"), 100.0]])  # 创建一个包含 Timestamp 和浮点数的 DataFrame 对象
        result = df.to_numpy(dtype=str)               # 调用 DataFrame 的方法将其转换为字符串类型的 NumPy 数组
        expected = np.array([["2020-01-01 00:00:00", "100.0"]], dtype=str)  # 创建预期的字符串类型的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等
```