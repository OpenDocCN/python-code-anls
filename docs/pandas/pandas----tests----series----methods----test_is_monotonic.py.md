# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_is_monotonic.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据

from pandas import (  # 从 Pandas 库中导入 Series 和 date_range 函数
    Series,
    date_range,
)


class TestIsMonotonic:
    def test_is_monotonic_numeric(self):
        ser = Series(np.random.default_rng(2).integers(0, 10, size=1000))  # 创建一个包含随机整数的 Series 对象
        assert not ser.is_monotonic_increasing  # 断言该 Series 不是单调递增的
        ser = Series(np.arange(1000))  # 创建一个包含从 0 到 999 的整数的 Series 对象
        assert ser.is_monotonic_increasing is True  # 断言该 Series 是单调递增的
        assert ser.is_monotonic_increasing is True  # 再次断言该 Series 是单调递增的
        ser = Series(np.arange(1000, 0, -1))  # 创建一个包含从 1000 到 1 的整数的 Series 对象
        assert ser.is_monotonic_decreasing is True  # 断言该 Series 是单调递减的

    def test_is_monotonic_dt64(self):
        ser = Series(date_range("20130101", periods=10))  # 创建一个包含日期范围的 Series 对象
        assert ser.is_monotonic_increasing is True  # 断言该 Series 是单调递增的
        assert ser.is_monotonic_increasing is True  # 再次断言该 Series 是单调递增的

        ser = Series(list(reversed(ser)))  # 创建一个将原始 Series 反转后的 Series 对象
        assert ser.is_monotonic_increasing is False  # 断言该 Series 不是单调递增的
        assert ser.is_monotonic_decreasing is True  # 断言该 Series 是单调递减的
```