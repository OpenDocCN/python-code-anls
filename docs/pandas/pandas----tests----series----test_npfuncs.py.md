# `D:\src\scipysrc\pandas\pandas\tests\series\test_npfuncs.py`

```
"""
Tests for np.foo applied to Series, not necessarily ufuncs.
"""

# 导入所需的库和模块
import numpy as np
import pytest

# 导入测试装饰器
import pandas.util._test_decorators as td

# 导入 Series 和测试工具
from pandas import Series
import pandas._testing as tm


class TestPtp:
    def test_ptp(self):
        # GH#21614
        # 设置测试数据量 N
        N = 1000
        # 生成标准正态分布的随机数数组 arr
        arr = np.random.default_rng(2).standard_normal(N)
        # 将数组 arr 转换为 Series 对象
        ser = Series(arr)
        # 断言 Series 对象和原始数组的峰-峰值相同
        assert np.ptp(ser) == np.ptp(arr)


def test_numpy_unique(datetime_series):
    # it works!
    # 对 datetime_series 应用 np.unique 函数
    np.unique(datetime_series)


@pytest.mark.parametrize("index", [["a", "b", "c", "d", "e"], None])
def test_numpy_argwhere(index):
    # GH#35331

    # 创建带有指定索引和数据类型的 Series 对象 s
    s = Series(range(5), index=index, dtype=np.int64)

    # 找出 Series 对象中大于 2 的元素的索引位置
    result = np.argwhere(s > 2).astype(np.int64)
    # 期望的结果数组
    expected = np.array([[3], [4]], dtype=np.int64)

    # 断言结果数组与期望数组相等
    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_log_arrow_backed_missing_value():
    # GH#56285

    # 创建包含缺失值的 Series 对象 ser，并指定数据类型为 "float64[pyarrow]"
    ser = Series([1, 2, None], dtype="float64[pyarrow]")
    # 计算 Series 对象中每个元素的自然对数
    result = np.log(ser)
    # 创建一个期望的 Series 对象，使用标准 float64 类型计算其自然对数
    expected = np.log(Series([1, 2, None], dtype="float64"))
    # 断言结果 Series 对象与期望 Series 对象相等
    tm.assert_series_equal(result, expected)
```